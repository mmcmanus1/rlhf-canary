"""Tests for PyTorch profiler integration."""

import pytest

from canary.collect.profiler import (
    ProfilerCallback,
    ProfilerConfig,
    ProfilerSummary,
    ProfilerTrainerCallback,
)


class TestProfilerConfig:
    """Tests for ProfilerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProfilerConfig()
        assert config.enabled is False
        assert config.start_step == 50
        assert config.num_steps == 20
        assert config.output_dir == "./profiler_traces"
        assert config.record_shapes is True
        assert config.profile_memory is True
        assert config.with_stack is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProfilerConfig(
            enabled=True,
            start_step=10,
            num_steps=5,
            output_dir="/custom/path",
            with_stack=True,
        )
        assert config.enabled is True
        assert config.start_step == 10
        assert config.num_steps == 5
        assert config.output_dir == "/custom/path"
        assert config.with_stack is True


class TestProfilerSummary:
    """Tests for ProfilerSummary."""

    def test_default_values(self):
        """Test default summary values."""
        summary = ProfilerSummary()
        assert summary.trace_path is None
        assert summary.cuda_time_total_ms == 0.0
        assert summary.cpu_time_total_ms == 0.0
        assert summary.self_cuda_time_total_ms == 0.0
        assert summary.top_cuda_ops == []
        assert summary.top_cpu_ops == []

    def test_with_data(self):
        """Test summary with actual data."""
        summary = ProfilerSummary(
            trace_path="/path/to/trace.json",
            cuda_time_total_ms=100.5,
            cpu_time_total_ms=50.2,
            self_cuda_time_total_ms=80.0,
            top_cuda_ops=[{"name": "aten::mm", "self_cuda_time_ms": 30.0, "count": 100}],
            top_cpu_ops=[{"name": "aten::add", "self_cpu_time_ms": 10.0, "count": 200}],
        )
        assert summary.trace_path == "/path/to/trace.json"
        assert summary.cuda_time_total_ms == 100.5
        assert len(summary.top_cuda_ops) == 1
        assert summary.top_cuda_ops[0]["name"] == "aten::mm"


class TestProfilerCallback:
    """Tests for ProfilerCallback."""

    def test_should_profile_when_disabled(self):
        """Test should_profile returns False when disabled."""
        config = ProfilerConfig(enabled=False)
        callback = ProfilerCallback(config, "test_run")

        assert callback.should_profile(50) is False
        assert callback.should_profile(60) is False

    def test_should_profile_within_window(self):
        """Test should_profile returns True within window."""
        config = ProfilerConfig(enabled=True, start_step=50, num_steps=20)
        callback = ProfilerCallback(config, "test_run")

        # Before window
        assert callback.should_profile(49) is False
        # Start of window
        assert callback.should_profile(50) is True
        # Middle of window
        assert callback.should_profile(60) is True
        # End of window (exclusive)
        assert callback.should_profile(69) is True
        # After window
        assert callback.should_profile(70) is False

    def test_stop_profiling_when_not_started(self):
        """Test stop_profiling returns empty summary when not started."""
        config = ProfilerConfig(enabled=True)
        callback = ProfilerCallback(config, "test_run")

        # Don't start profiling, just stop
        summary = callback.stop_profiling()
        assert summary.trace_path is None
        assert summary.cuda_time_total_ms == 0.0


class TestProfilerTrainerCallback:
    """Tests for ProfilerTrainerCallback integration with HuggingFace Trainer."""

    def test_initialization(self):
        """Test callback initialization."""
        config = ProfilerConfig(enabled=True, start_step=10, num_steps=5)
        callback = ProfilerTrainerCallback(config, "test_run")

        assert callback.config == config
        assert callback.run_id == "test_run"
        assert callback._profiling_started is False
        assert callback._profiling_stopped is False
        assert callback.summary is None

    def test_get_summary_returns_none_when_not_profiled(self):
        """Test get_summary returns None when profiling never started."""
        config = ProfilerConfig(enabled=True)
        callback = ProfilerTrainerCallback(config, "test_run")

        assert callback.get_summary() is None

    def test_on_step_begin_starts_profiling(self):
        """Test that profiling starts at the configured step."""
        config = ProfilerConfig(enabled=True, start_step=5, num_steps=3)
        callback = ProfilerTrainerCallback(config, "test_run")

        class MockState:
            global_step = 5

        class Mock:
            pass

        # Before start_step - should not start
        callback._current_step = 4
        MockState.global_step = 4
        callback.on_step_begin(Mock(), MockState(), Mock())
        assert callback._profiling_started is False

        # At start_step - should start
        MockState.global_step = 5
        callback.on_step_begin(Mock(), MockState(), Mock())
        assert callback._profiling_started is True

    def test_profiling_stops_after_num_steps(self):
        """Test that profiling stops after configured number of steps.

        Note: We mock the internal profiler callback to avoid starting a real
        PyTorch profiler which can conflict with other tests on the same thread.
        """
        config = ProfilerConfig(enabled=True, start_step=5, num_steps=3)
        callback = ProfilerTrainerCallback(config, "test_run")

        class MockState:
            global_step = 5

        class Mock:
            pass

        # Mock the internal profiler callback to avoid real profiler conflicts
        class MockProfilerCallback:
            def start_profiling(self):
                pass

            def step(self):
                pass

            def stop_profiling(self):
                return ProfilerSummary(trace_path="/test/trace.json")

        callback._profiler_callback = MockProfilerCallback()

        # Start profiling
        callback.on_step_begin(Mock(), MockState(), Mock())
        callback._profiling_started = True  # Manually set since we mocked
        assert callback._profiling_started is True
        assert callback._profiling_stopped is False

        # Step through to end (5, 6, 7)
        for step in [5, 6, 7]:
            MockState.global_step = step
            callback.on_step_end(Mock(), MockState(), Mock())

        # At step 8 (5 + 3), profiling should have stopped
        MockState.global_step = 8
        callback.on_step_end(Mock(), MockState(), Mock())
        assert callback._profiling_stopped is True

    def test_on_train_end_stops_profiling(self):
        """Test that training end stops profiling if still active.

        Note: We mock the internal profiler callback to avoid starting a real
        PyTorch profiler which can conflict with other tests on the same thread.
        """
        config = ProfilerConfig(enabled=True, start_step=5, num_steps=100)
        callback = ProfilerTrainerCallback(config, "test_run")

        class MockState:
            global_step = 10

        class Mock:
            pass

        # Mock the internal profiler callback to avoid real profiler conflicts
        class MockProfilerCallback:
            def start_profiling(self):
                pass

            def step(self):
                pass

            def stop_profiling(self):
                return ProfilerSummary(trace_path="/test/trace.json")

        callback._profiler_callback = MockProfilerCallback()

        # Simulate that profiling was started
        callback._profiling_started = True
        assert callback._profiling_started is True
        assert callback._profiling_stopped is False

        # End training before profiling window ends
        callback.on_train_end(Mock(), MockState(), Mock())
        assert callback._profiling_stopped is True

    def test_disabled_profiler_does_nothing(self):
        """Test that disabled profiler doesn't start profiling."""
        config = ProfilerConfig(enabled=False)
        callback = ProfilerTrainerCallback(config, "test_run")

        class MockState:
            global_step = 50

        class Mock:
            pass

        callback.on_step_begin(Mock(), MockState(), Mock())
        assert callback._profiling_started is False

        callback.on_step_end(Mock(), MockState(), Mock())
        assert callback._profiling_stopped is False

        callback.on_train_end(Mock(), MockState(), Mock())
        assert callback.summary is None


class TestProfilerIntegration:
    """Integration tests for profiler with CanaryMetrics."""

    def test_profiler_summary_serializes_to_dict(self):
        """Test that ProfilerSummary can be serialized via model_dump."""
        summary = ProfilerSummary(
            trace_path="/path/to/trace.json",
            cuda_time_total_ms=100.5,
            cpu_time_total_ms=50.2,
            top_cuda_ops=[{"name": "aten::mm", "self_cuda_time_ms": 30.0, "count": 100}],
        )

        # Should be serializable to dict for CanaryMetrics
        dumped = summary.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["trace_path"] == "/path/to/trace.json"
        assert dumped["cuda_time_total_ms"] == 100.5
        assert dumped["top_cuda_ops"][0]["name"] == "aten::mm"

    def test_canary_metrics_accepts_profiler_summary(self):
        """Test that CanaryMetrics accepts profiler summary."""
        from canary.collect.metrics import (
            CanaryMetrics,
            PerformanceMetrics,
            RunConfig,
            StabilityMetrics,
            StepTimeStats,
        )

        profiler_data = ProfilerSummary(
            trace_path="/path/to/trace.json",
            cuda_time_total_ms=100.0,
        ).model_dump()

        metrics = CanaryMetrics(
            run_id="test_run",
            timestamp="2024-01-01T00:00:00",
            duration_seconds=60.0,
            env={"test": "env"},
            config=RunConfig(
                model_name="test-model",
                max_steps=100,
                batch_size=2,
                gradient_accumulation_steps=4,
                max_length=256,
                learning_rate=5e-5,
                dataset_name="test-dataset",
                dataset_size=1000,
            ),
            perf=PerformanceMetrics(
                step_time=StepTimeStats(count=100, mean=0.5),
            ),
            stability=StabilityMetrics(),
            profiler=profiler_data,
            status="completed",
        )

        assert metrics.profiler is not None
        assert metrics.profiler["trace_path"] == "/path/to/trace.json"
        assert metrics.profiler["cuda_time_total_ms"] == 100.0
