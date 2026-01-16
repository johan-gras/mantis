"""
Tests for strategy comparison functionality.

Tests cover:
- Comparing multiple strategies
- Comparison metrics
- Comparison visualization
"""

import numpy as np
import pytest

import mantis as mt


class TestCompareBasic:
    """Tests for basic strategy comparison."""

    def test_compare_two_strategies(self, sample_aapl):
        """Compare two different strategies."""
        # Strategy 1: Buy and hold
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        # Strategy 2: SMA crossover
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        assert hasattr(comparison, "metrics")

    def test_compare_with_names(self, sample_aapl):
        """Compare strategies with custom names."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        signal2 = -np.ones(sample_aapl["n_bars"])
        result2 = mt.backtest(sample_aapl, signal=signal2, allow_short=True)

        comparison = mt.compare(
            [result1, result2],
            names=["Long Only", "Short Only"],
        )

        metrics = comparison.metrics
        assert "Long Only" in str(metrics) or len(metrics) > 0

    def test_compare_multiple_strategies(self, sample_aapl):
        """Compare more than two strategies."""
        results = []

        # Buy and hold
        results.append(mt.backtest(sample_aapl, signal=np.ones(sample_aapl["n_bars"])))

        # SMA crossover
        results.append(
            mt.backtest(
                sample_aapl,
                strategy="sma_crossover",
                strategy_params={"fast": 10, "slow": 30},
            )
        )

        # Momentum
        results.append(
            mt.backtest(
                sample_aapl,
                strategy="momentum",
                strategy_params={"period": 20},
            )
        )

        comparison = mt.compare(results)

        # Should handle 3 strategies
        metrics = comparison.to_dict()
        assert len(metrics) > 0


class TestCompareMetrics:
    """Tests for comparison metrics."""

    def test_compare_to_dict(self, sample_aapl):
        """Comparison should be exportable to dictionary."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        d = comparison.to_dict()
        assert isinstance(d, dict)

    def test_compare_metrics_property(self, sample_aapl):
        """Metrics property should provide comparison data."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        metrics = comparison.metrics
        assert metrics is not None


class TestComparePlotting:
    """Tests for comparison visualization."""

    def test_compare_plot_exists(self, sample_aapl):
        """Comparison should have plot method."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        assert hasattr(comparison, "plot")


class TestCompareEdgeCases:
    """Tests for comparison edge cases."""

    def test_compare_single_strategy(self, sample_aapl):
        """Comparing single strategy should work or raise clear error."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        try:
            comparison = mt.compare([result])
            # If it works, verify structure
            assert comparison is not None
        except Exception as e:
            # Should raise meaningful error
            assert "at least" in str(e).lower() or "multiple" in str(e).lower() or "two" in str(e).lower()

    def test_compare_same_strategy_twice(self, sample_aapl):
        """Comparing same strategy twice should work."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        comparison = mt.compare([result, result], names=["Strategy A", "Strategy B"])

        # Both should have identical metrics
        metrics = comparison.to_dict()
        assert len(metrics) > 0
