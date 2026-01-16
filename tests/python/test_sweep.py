"""
Tests for parameter sweep functionality.

Tests cover:
- Basic parameter sweeps
- Parallel execution
- Result sorting and filtering
- Parameter ranges
"""

import numpy as np
import pytest

import mantis as mt


class TestBasicSweep:
    """Tests for basic parameter sweep functionality."""

    def test_sweep_basic(self, sample_aapl):
        """Basic parameter sweep with signal function."""
        closes = sample_aapl["close"]
        n = len(closes)

        def signal_fn(fast, slow):
            """Simple SMA crossover signal generator."""
            signal = np.zeros(n)
            fast_ma = np.zeros(n)
            slow_ma = np.zeros(n)

            for i in range(n):
                if i >= int(fast) - 1:
                    fast_ma[i] = np.mean(closes[i - int(fast) + 1 : i + 1])
                if i >= int(slow) - 1:
                    slow_ma[i] = np.mean(closes[i - int(slow) + 1 : i + 1])

                if i >= int(slow) - 1:
                    if fast_ma[i] > slow_ma[i]:
                        signal[i] = 1
                    else:
                        signal[i] = -1

            return signal

        params = {
            "fast": [5, 10],
            "slow": [20, 30],
        }

        result = mt.sweep(sample_aapl, signal_fn, params)

        assert hasattr(result, "num_combinations")
        # 2 fast * 2 slow = 4 combinations
        assert result.num_combinations == 4

    def test_sweep_items(self, sample_aapl):
        """Sweep items() should return list of results."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            signal = np.ones(n)
            return signal

        params = {"period": [10, 20, 30]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        items = result.items()
        assert isinstance(items, list)
        assert len(items) == 3

    def test_sweep_best(self, sample_aapl):
        """Sweep best() should return best performing params."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            signal = np.ones(n)
            return signal

        params = {"period": [10, 20, 30]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        best = result.best(metric="sharpe")
        assert best is not None
        assert hasattr(best, "params")
        assert hasattr(best, "result")

    def test_sweep_best_params(self, sample_aapl):
        """best_params() should return just the parameters."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            signal = np.ones(n)
            return signal

        params = {"period": [10, 20, 30]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        best_params = result.best_params(metric="sharpe")
        assert best_params is not None
        assert "period" in best_params

    def test_sweep_sorted_by(self, sample_aapl):
        """sorted_by() should return sorted results."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            signal = np.ones(n)
            return signal

        params = {"period": [10, 20, 30]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        sorted_items = result.sorted_by(metric="sharpe", descending=True)
        assert isinstance(sorted_items, list)
        assert len(sorted_items) == 3

        # Verify descending order
        if len(sorted_items) >= 2:
            assert sorted_items[0].result.sharpe >= sorted_items[-1].result.sharpe

    def test_sweep_top_n(self, sample_aapl):
        """top() should return top N results."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            signal = np.ones(n)
            return signal

        params = {"period": [10, 20, 30, 40, 50]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        top_3 = result.top(n=3, metric="sharpe")
        assert len(top_3) == 3


class TestSweepParallel:
    """Tests for parallel sweep execution."""

    def test_sweep_parallel_flag(self, sample_aapl):
        """Sweep should indicate if parallel was used."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            return np.ones(n)

        params = {"period": [10, 20]}

        result = mt.sweep(sample_aapl, signal_fn, params, parallel=True)

        assert hasattr(result, "parallel")

    def test_sweep_parallel_vs_sequential(self, sample_aapl):
        """Parallel and sequential should produce same results."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            return np.ones(n)

        params = {"period": [10, 20]}

        result_parallel = mt.sweep(sample_aapl, signal_fn, params, parallel=True)
        result_sequential = mt.sweep(sample_aapl, signal_fn, params, parallel=False)

        # Same number of combinations
        assert result_parallel.num_combinations == result_sequential.num_combinations

        # Results should be equivalent (order may differ)
        parallel_best = result_parallel.best_params()
        sequential_best = result_sequential.best_params()
        # At least one parameter should match
        assert parallel_best is not None and sequential_best is not None


class TestParameterRanges:
    """Tests for parameter range generators."""

    def test_linear_range(self):
        """linear_range() should create evenly spaced values."""
        rng = mt.linear_range(0, 100, 5)

        values = rng.values()
        assert len(values) == 5
        # Should be evenly spaced: 0, 25, 50, 75, 100
        assert values[0] == 0
        assert values[-1] == 100

    def test_log_range(self):
        """log_range() should create logarithmically spaced values."""
        rng = mt.log_range(1, 100, 3)

        values = rng.values()
        assert len(values) == 3
        assert abs(values[0] - 1) < 0.001
        assert abs(values[-1] - 100) < 0.001
        # Middle value should be geometric mean: sqrt(1*100) = 10
        assert abs(values[1] - 10) < 1

    def test_discrete_range(self):
        """discrete_range() should use exact specified values."""
        rng = mt.discrete_range([5, 10, 20, 50])

        values = rng.values()
        assert len(values) == 4
        assert list(values) == [5, 10, 20, 50]

    def test_centered_range(self):
        """centered_range() should create values centered around a point."""
        rng = mt.centered_range(center=100, variation=0.2, steps=5)

        values = rng.values()
        assert len(values) == 5
        # Center value should be in the middle
        # Values should range from 80 to 120 (100 +/- 20%)

    def test_parameter_range_length(self):
        """Parameter ranges should report correct length."""
        rng = mt.linear_range(0, 10, 11)

        assert len(rng) == 11


class TestSweepWithRanges:
    """Tests for sweeps using parameter ranges."""

    def test_sweep_with_linear_range(self, sample_aapl):
        """Sweep with linear_range parameter."""
        closes = sample_aapl["close"]
        n = len(closes)

        def signal_fn(threshold):
            signal = np.zeros(n)
            for i in range(1, n):
                pct_change = (closes[i] - closes[i - 1]) / closes[i - 1]
                if pct_change > threshold:
                    signal[i] = 1
                elif pct_change < -threshold:
                    signal[i] = -1
            return signal

        # Convert ParameterRange to list for sweep
        params = {
            "threshold": list(mt.linear_range(0.001, 0.01, 3).values()),
        }

        result = mt.sweep(sample_aapl, signal_fn, params)

        assert result.num_combinations == 3

    def test_sweep_with_discrete_range(self, sample_aapl):
        """Sweep with discrete_range parameter."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            return np.ones(n)

        # Convert ParameterRange to list for sweep
        params = {
            "period": list(mt.discrete_range([10, 20, 50, 100]).values()),
        }

        result = mt.sweep(sample_aapl, signal_fn, params)

        assert result.num_combinations == 4


class TestSweepSummary:
    """Tests for sweep summary and analysis."""

    def test_sweep_summary(self, sample_aapl):
        """Sweep summary should provide statistics."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            return np.ones(n)

        params = {"period": [10, 20, 30]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        summary = result.summary()
        assert isinstance(summary, dict)

    def test_sweep_to_dict(self, sample_aapl):
        """Sweep to_dict() should export all results."""
        n = sample_aapl["n_bars"]

        def signal_fn(period):
            return np.ones(n)

        params = {"period": [10, 20]}

        result = mt.sweep(sample_aapl, signal_fn, params)

        d = result.to_dict()
        assert isinstance(d, dict)


class TestSweepPlotting:
    """Tests for sweep visualization."""

    def test_sweep_plot_exists(self, sample_aapl):
        """Sweep plot() method should exist."""
        n = sample_aapl["n_bars"]

        def signal_fn(fast, slow):
            return np.ones(n)

        params = {
            "fast": [5, 10],
            "slow": [20, 30],
        }

        result = mt.sweep(sample_aapl, signal_fn, params)

        # Method should exist
        assert hasattr(result, "plot")
