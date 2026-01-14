"""Local runner for executing canary training jobs."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from canary.collect.env_fingerprint import get_env_fingerprint
from canary.collect.metrics import (
    CanaryCallback,
    CanaryMetrics,
    PerformanceMetrics,
    RunConfig,
)
from canary.collect.profiler import ProfilerTrainerCallback
from canary.runner.base import BaseRunner, RunResult

logger = logging.getLogger(__name__)


class LocalRunner(BaseRunner):
    """Runner that executes training jobs locally."""

    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate configuration for local execution."""
        errors = []

        # Check training type
        if self.config.training_type not in ("sft", "dpo", "ppo"):
            errors.append(f"Invalid training_type: {self.config.training_type}")

        # Check batch size
        if self.config.batch_size < 1:
            errors.append("batch_size must be >= 1")

        # Check max_steps
        if self.config.max_steps < 1:
            errors.append("max_steps must be >= 1")

        return len(errors) == 0, errors

    def run(self) -> RunResult:
        """Execute the canary training run locally."""
        is_valid, errors = self.validate_config()
        if not is_valid:
            return RunResult(success=False, error="; ".join(errors))

        run_id = self.generate_run_id()
        output_dir = Path(self.config.output_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting canary run: {run_id}")
        logger.info(f"Output directory: {output_dir}")

        try:
            if self.config.training_type == "dpo":
                metrics = self._run_dpo(run_id, output_dir)
            elif self.config.training_type == "sft":
                metrics = self._run_sft(run_id, output_dir)
            elif self.config.training_type == "ppo":
                metrics = self._run_ppo(run_id, output_dir)
            else:
                return RunResult(
                    success=False,
                    error=f"Training type {self.config.training_type} not yet implemented",
                )

            # Save metrics
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                f.write(metrics.model_dump_json(indent=2))

            logger.info(f"Canary run completed: {run_id}")
            return RunResult(
                success=True,
                metrics=metrics,
                metrics_path=str(metrics_path),
            )

        except Exception as e:
            logger.exception(f"Canary run failed: {run_id}")
            return RunResult(success=False, error=str(e))

    def _run_dpo(self, run_id: str, output_dir: Path) -> CanaryMetrics:
        """Run DPO training canary."""
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        # Determine compute dtype for consistent usage across model and training
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": compute_dtype,
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        model.config.use_cache = False

        logger.info("Loading dataset...")
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        )
        dataset = dataset.shuffle(seed=self.config.seed).select(
            range(min(self.config.dataset_size, len(dataset)))
        )

        # Format for DPO (Anthropic HH-RLHF already has chosen/rejected)
        def format_dpo(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "prompt": "",  # HH-RLHF has full conversations
                "chosen": example.get("chosen", ""),
                "rejected": example.get("rejected", ""),
            }

        dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

        # PEFT configuration
        peft_config = None
        if self.config.use_peft:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # Training arguments (DPOConfig extends TrainingArguments with DPO-specific params)
        # NOTE: Disable mixed precision (bf16/fp16) to avoid TRL 0.26 dtype bug where
        # input_ids get cast to float during concatenated_forward
        training_args = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=1,
            max_steps=self.config.max_steps,
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints in canary
            bf16=False,  # Disabled due to TRL 0.26 dtype bug
            fp16=False,  # Disabled due to TRL 0.26 dtype bug
            report_to=[],
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            # DPO-specific parameters (moved from DPOTrainer constructor in TRL 0.12+)
            beta=self.config.beta,
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_length,
            precompute_ref_log_probs=False,  # Explicitly disable to avoid dtype issues
        )

        # Create callbacks
        canary_callback = CanaryCallback(warmup_steps=self.config.metrics_warmup_steps)
        callbacks = [canary_callback]

        # Add profiler callback if enabled
        profiler_callback = None
        if self.config.profiler.enabled:
            profiler_callback = ProfilerTrainerCallback(self.config.profiler, run_id)
            callbacks.append(profiler_callback)
            logger.info(
                f"Profiler enabled: will capture steps {self.config.profiler.start_step} "
                f"to {self.config.profiler.start_step + self.config.profiler.num_steps}"
            )

        logger.info("Starting DPO training...")

        # Reset CUDA memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Note: DPOTrainer creates a reference model internally, which roughly
        # doubles memory usage. This is expected behavior for DPO training.
        logger.info(
            "DPO training will create a reference model copy. "
            "Expect ~2x memory usage compared to SFT."
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer creates ref model internally
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,  # renamed from tokenizer in TRL 0.26+
            peft_config=peft_config,
            callbacks=callbacks,
        )

        trainer.train()

        # Collect metrics
        env = get_env_fingerprint()
        step_stats = canary_callback.get_step_time_stats()
        stability = canary_callback.get_stability_metrics()
        tokens_per_sec = canary_callback.estimate_tokens_per_sec(
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.config.max_length,
            is_dpo=True,
        )

        # Get profiler summary if profiling was enabled
        profiler_summary = None
        if profiler_callback is not None:
            profiler_summary = profiler_callback.get_summary()
            if profiler_summary:
                profiler_summary = profiler_summary.model_dump()

        return CanaryMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=canary_callback.get_duration_seconds(),
            env=env.model_dump(),
            config=RunConfig(
                model_name=self.config.model_name,
                max_steps=self.config.max_steps,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_length=self.config.max_length,
                learning_rate=self.config.learning_rate,
                beta=self.config.beta,
                dataset_name=self.config.dataset_name,
                dataset_size=len(dataset),
                seed=self.config.seed,
            ),
            perf=PerformanceMetrics(
                step_time=step_stats,
                approx_tokens_per_sec=tokens_per_sec,
                max_mem_mb=canary_callback.max_mem_mb,
                gpu_utilization_avg=canary_callback.get_gpu_utilization_avg(),
                dataloader_wait_pct=canary_callback.get_dataloader_wait_pct(),
            ),
            stability=stability,
            profiler=profiler_summary,
            status="completed",
        )

    def _run_sft(self, run_id: str, output_dir: Path) -> CanaryMetrics:
        """Run SFT training canary."""
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
        )
        from trl import SFTConfig, SFTTrainer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        # Determine compute dtype for consistent usage across model and training
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": compute_dtype,
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        model.config.use_cache = False

        logger.info("Loading dataset...")
        # For SFT, use a simple text dataset
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        )
        dataset = dataset.shuffle(seed=self.config.seed).select(
            range(min(self.config.dataset_size, len(dataset)))
        )

        # For Anthropic HH-RLHF, use the 'chosen' column for SFT
        def format_sft(example: dict[str, Any]) -> dict[str, Any]:
            text = example.get("chosen", example.get("text", ""))
            return {"text": text}

        dataset = dataset.map(format_sft, remove_columns=dataset.column_names)

        # PEFT configuration
        peft_config = None
        if self.config.use_peft:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        # Training arguments (use SFTConfig for TRL 0.26+ compatibility)
        training_args = SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=1,
            max_steps=self.config.max_steps,
            max_length=self.config.max_length,  # moved from SFTTrainer param in TRL 0.26+
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints in canary
            bf16=use_bf16,
            fp16=not use_bf16 and torch.cuda.is_available(),
            report_to=[],
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        # Create callbacks
        canary_callback = CanaryCallback(warmup_steps=self.config.metrics_warmup_steps)
        callbacks = [canary_callback]

        # Add profiler callback if enabled
        profiler_callback = None
        if self.config.profiler.enabled:
            profiler_callback = ProfilerTrainerCallback(self.config.profiler, run_id)
            callbacks.append(profiler_callback)
            logger.info(
                f"Profiler enabled: will capture steps {self.config.profiler.start_step} "
                f"to {self.config.profiler.start_step + self.config.profiler.num_steps}"
            )

        logger.info("Starting SFT training...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,  # renamed from tokenizer in TRL 0.26+
            peft_config=peft_config,
            callbacks=callbacks,
        )

        trainer.train()

        # Collect metrics
        env = get_env_fingerprint()
        step_stats = canary_callback.get_step_time_stats()
        stability = canary_callback.get_stability_metrics()
        tokens_per_sec = canary_callback.estimate_tokens_per_sec(
            self.config.batch_size,
            self.config.gradient_accumulation_steps,
            self.config.max_length,
            is_dpo=False,
        )

        # Get profiler summary if profiling was enabled
        profiler_summary = None
        if profiler_callback is not None:
            profiler_summary = profiler_callback.get_summary()
            if profiler_summary:
                profiler_summary = profiler_summary.model_dump()

        return CanaryMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=canary_callback.get_duration_seconds(),
            env=env.model_dump(),
            config=RunConfig(
                model_name=self.config.model_name,
                max_steps=self.config.max_steps,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_length=self.config.max_length,
                learning_rate=self.config.learning_rate,
                beta=None,
                dataset_name=self.config.dataset_name,
                dataset_size=len(dataset),
                seed=self.config.seed,
            ),
            perf=PerformanceMetrics(
                step_time=step_stats,
                approx_tokens_per_sec=tokens_per_sec,
                max_mem_mb=canary_callback.max_mem_mb,
                gpu_utilization_avg=canary_callback.get_gpu_utilization_avg(),
                dataloader_wait_pct=canary_callback.get_dataloader_wait_pct(),
            ),
            stability=stability,
            profiler=profiler_summary,
            status="completed",
        )

    def _run_ppo(self, run_id: str, output_dir: Path) -> CanaryMetrics:
        """Run PPO training canary.

        PPO uses a generate-then-train loop with a synthetic reward function
        for canary testing. This measures training stability and throughput
        without requiring a trained reward model.
        """
        import math
        import time

        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Required for generation

        logger.info("Loading model with value head...")
        # Determine compute dtype
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        model_kwargs: dict[str, Any] = {
            "torch_dtype": compute_dtype,
            "device_map": "auto",
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # PPO requires model with value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Apply PEFT if configured
        if self.config.use_peft:
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)

        logger.info("Loading dataset...")
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        )
        dataset = dataset.shuffle(seed=self.config.seed).select(
            range(min(self.config.dataset_size, len(dataset)))
        )

        # Format for PPO: extract prompts only
        def format_ppo(example: dict[str, Any]) -> dict[str, Any]:
            # For HH-RLHF, extract the human turn as the prompt
            text = example.get("chosen", "")
            # Extract up to first assistant response as prompt
            if "\n\nAssistant:" in text:
                prompt = text.split("\n\nAssistant:")[0] + "\n\nAssistant:"
            else:
                logger.warning(
                    "Dataset entry missing expected format, using truncated fallback"
                )
                prompt = text[:200]  # Fallback: use first 200 chars
            return {"query": prompt}

        dataset = dataset.map(format_ppo, remove_columns=dataset.column_names)

        # Tokenize prompts
        def tokenize(example: dict[str, Any]) -> dict[str, Any]:
            tokens = tokenizer(
                example["query"],
                truncation=True,
                max_length=self.config.max_prompt_length,
                padding=False,
            )
            return {"input_ids": tokens["input_ids"]}

        dataset = dataset.map(tokenize, remove_columns=["query"])
        dataset.set_format(type="torch")

        # PPO configuration
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.batch_size,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target_kl=self.config.target_kl,
            cliprange=self.config.cliprange,
            vf_coef=self.config.vf_coef,
            seed=self.config.seed,
            log_with=None,  # Disable wandb/tensorboard
        )

        logger.info("Starting PPO training...")
        logger.info(
            "PPO training uses model + ref_model + value_head. "
            "Expect ~3x memory usage compared to SFT."
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None,  # PPOTrainer creates ref model internally
            tokenizer=tokenizer,
            dataset=dataset,
        )

        # Synthetic reward function for canary testing
        def compute_synthetic_reward(responses: list[str]) -> list[torch.Tensor]:
            """Length-based reward for consistent canary testing."""
            rewards = []
            for r in responses:
                # Reward based on response length (normalized)
                reward = min(len(r) / 100.0, 1.0)
                rewards.append(torch.tensor(reward, dtype=compute_dtype))
            return rewards

        # Manual training loop with metrics collection
        run_start_time = time.time()
        step_times: list[float] = []
        all_stats: list[dict[str, Any]] = []
        nan_steps = 0
        inf_steps = 0
        max_mem_mb = 0.0
        loss_values: list[float] = []
        kl_values: list[float] = []

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
        }

        logger.info(f"Running PPO for {self.config.max_steps} steps...")

        step = 0
        for batch in trainer.dataloader:
            if step >= self.config.max_steps:
                break

            step_start = time.time()

            # Get query tensors
            query_tensors = list(batch["input_ids"])

            # Generate responses
            response_tensors = trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs,
            )

            # Decode responses for reward computation
            responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute synthetic rewards
            rewards = compute_synthetic_reward(responses)

            # PPO step
            stats = trainer.step(query_tensors, response_tensors, rewards)
            all_stats.append(stats)

            step_time = time.time() - step_start
            step_times.append(step_time)

            # Track stability metrics
            found_nan = False
            found_inf = False
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    if math.isnan(value):
                        found_nan = True
                    elif math.isinf(value):
                        found_inf = True
                    if "loss" in key.lower() and not math.isnan(value) and not math.isinf(value):
                        loss_values.append(value)
                    if "kl" in key.lower() and not math.isnan(value) and not math.isinf(value):
                        kl_values.append(value)

            if found_nan:
                nan_steps += 1
            if found_inf:
                inf_steps += 1

            # Track memory
            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                max_mem_mb = max(max_mem_mb, mem_mb)

            step += 1

            if step % 10 == 0:
                logger.info(f"Step {step}/{self.config.max_steps} - step_time: {step_time:.2f}s")

        duration_seconds = time.time() - run_start_time

        # Compute step time statistics (exclude warmup)
        from canary.collect.metrics import (
            PerformanceMetrics,
            RunConfig,
            StabilityMetrics,
            summarize_step_times,
        )

        step_stats = summarize_step_times(step_times, self.config.metrics_warmup_steps)

        # Estimate tokens per second
        # PPO: query + response tokens per step
        tokens_per_sec = None
        if step_stats.mean and step_stats.mean > 0:
            avg_response_tokens = self.config.max_new_tokens  # Upper bound
            tokens_per_step = self.config.batch_size * (
                self.config.max_prompt_length + avg_response_tokens
            )
            tokens_per_sec = tokens_per_step / step_stats.mean

        # Detect loss divergence
        loss_diverged = False
        if len(loss_values) > 20:
            early_avg = sum(loss_values[:10]) / 10
            late_avg = sum(loss_values[-10:]) / 10
            # Handle negative losses: check if loss magnitude increased significantly
            if early_avg >= 0:
                # Positive losses: late > early * 1.5 means divergence
                loss_diverged = late_avg > early_avg * 1.5
            else:
                # Negative losses: more negative (larger magnitude) means divergence
                loss_diverged = late_avg < early_avg * 1.5

        env = get_env_fingerprint()

        return CanaryMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            env=env.model_dump(),
            config=RunConfig(
                model_name=self.config.model_name,
                max_steps=self.config.max_steps,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                max_length=self.config.max_length,
                learning_rate=self.config.learning_rate,
                beta=None,  # Not used in PPO
                dataset_name=self.config.dataset_name,
                dataset_size=len(dataset),
                seed=self.config.seed,
            ),
            perf=PerformanceMetrics(
                step_time=step_stats,
                approx_tokens_per_sec=tokens_per_sec,
                max_mem_mb=max_mem_mb,
                gpu_utilization_avg=None,  # Not tracked in manual loop
                dataloader_wait_pct=None,  # Not tracked in manual loop
            ),
            stability=StabilityMetrics(
                nan_steps=nan_steps,
                inf_steps=inf_steps,
                loss_values=loss_values[-100:],
                grad_norm_values=[],  # Not easily accessible in PPO
                final_loss=loss_values[-1] if loss_values else None,
                loss_diverged=loss_diverged,
            ),
            profiler=None,  # Profiler not integrated with PPO manual loop
            status="completed",
        )
