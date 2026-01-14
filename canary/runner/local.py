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
        model_kwargs: dict[str, Any] = {"device_map": "auto"}

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
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
        training_args = DPOConfig(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=1,
            max_steps=self.config.max_steps,
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints in canary
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            report_to=[],
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            # DPO-specific parameters (moved from DPOTrainer constructor in TRL 0.12+)
            beta=self.config.beta,
            max_prompt_length=self.config.max_prompt_length,
            max_length=self.config.max_length,
        )

        # Create callback
        canary_callback = CanaryCallback(warmup_steps=self.config.metrics_warmup_steps)

        logger.info("Starting DPO training...")

        # Reset CUDA memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # DPOTrainer creates ref model internally
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[canary_callback],
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
            ),
            stability=stability,
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
        from trl import SFTTrainer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")
        model_kwargs: dict[str, Any] = {"device_map": "auto"}

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
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

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=1,
            max_steps=self.config.max_steps,
            logging_steps=10,
            save_strategy="no",  # Don't save checkpoints in canary
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            report_to=[],
            seed=self.config.seed,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        # Create callback
        canary_callback = CanaryCallback(warmup_steps=self.config.metrics_warmup_steps)

        logger.info("Starting SFT training...")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            max_seq_length=self.config.max_length,
            callbacks=[canary_callback],
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
            ),
            stability=stability,
            status="completed",
        )
