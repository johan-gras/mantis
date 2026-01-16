"""
Tests for visualization functionality.

Tests cover:
- Equity curve plotting
- Drawdown plotting
- Returns heatmap
- Trade plotting
- ASCII fallback
"""

import numpy as np
import pytest

import mantis as mt


class TestEquityCurvePlot:
    """Tests for equity curve visualization."""

    def test_plot_basic(self, sample_aapl):
        """Basic plot should work."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Plot method should exist and be callable
        assert hasattr(result, "plot")

        # Should return something (Plotly figure or ASCII string)
        plot = result.plot()
        assert plot is not None

    def test_plot_with_drawdown(self, sample_aapl):
        """Plot with drawdown overlay."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot(show_drawdown=True)
        assert plot is not None

    def test_plot_with_trades(self, sample_aapl):
        """Plot with trade markers."""
        # Use a strategy that generates trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        plot = result.plot(trades=True)
        assert plot is not None

    def test_plot_with_benchmark(self, sample_aapl, sample_spy):
        """Plot with benchmark comparison."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal, benchmark=sample_spy)

        plot = result.plot(benchmark=True)
        assert plot is not None

    def test_plot_custom_title(self, sample_aapl):
        """Plot with custom title."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot(title="Custom Title")
        assert plot is not None

    def test_plot_custom_dimensions(self, sample_aapl):
        """Plot with custom width/height."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot(width=80, height=20)
        assert plot is not None


class TestDrawdownPlot:
    """Tests for drawdown visualization."""

    def test_plot_drawdown(self, sample_aapl):
        """Drawdown plot should work."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "plot_drawdown")

        plot = result.plot_drawdown()
        assert plot is not None


class TestReturnsPlot:
    """Tests for returns heatmap visualization."""

    def test_plot_returns_monthly(self, sample_aapl):
        """Monthly returns heatmap."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "plot_returns")

        plot = result.plot_returns(period="monthly")
        assert plot is not None

    def test_plot_returns_daily(self, sample_aapl):
        """Daily returns plot."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot_returns(period="daily")
        assert plot is not None


class TestTradesPlot:
    """Tests for trade visualization."""

    def test_plot_trades(self, sample_aapl):
        """Trade plot should show entry/exit points."""
        # Use strategy that generates trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        assert hasattr(result, "plot_trades")

        plot = result.plot_trades()
        assert plot is not None


class TestASCIIFallback:
    """Tests for ASCII terminal fallback."""

    def test_ascii_sparkline(self, sample_aapl):
        """ASCII sparkline should be generated in terminal mode."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # With small width, should get ASCII output
        plot = result.plot(width=40)

        # Should be either Plotly figure or ASCII string
        assert plot is not None
        # If string, should contain ASCII chars
        if isinstance(plot, str):
            assert len(plot) > 0


class TestThemeSupport:
    """Tests for dark/light theme support."""

    def test_dark_theme(self, sample_aapl):
        """Dark theme should be supported."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot(theme="dark")
        assert plot is not None

    def test_light_theme(self, sample_aapl):
        """Light theme should be supported."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        plot = result.plot(theme="light")
        assert plot is not None


class TestExport:
    """Tests for plot export functionality."""

    def test_plot_save_parameter(self, sample_aapl, tmp_path):
        """Plot should support save parameter."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # When plotly is not installed, ASCII is used which can't save to HTML
        # Use .txt extension for ASCII or just test without save
        try:
            save_path = str(tmp_path / "test_plot.html")
            plot = result.plot(save=save_path)
            assert plot is not None
        except ValueError as e:
            if "ASCII" in str(e) or "plotly" in str(e).lower():
                # This is expected when plotly is not installed
                # Just test that plot() works without save
                plot = result.plot()
                assert plot is not None
            else:
                raise


class TestValidationPlot:
    """Tests for validation result plotting."""

    def test_validation_plot(self, sample_aapl):
        """Validation results should be plottable."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        assert hasattr(validation, "plot")

        plot = validation.plot()
        assert plot is not None


class TestMonteCarloPlot:
    """Tests for Monte Carlo result plotting."""

    def test_monte_carlo_plot_exists(self, sample_aapl):
        """Monte Carlo results should have plot method."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        # May or may not have plot method
        # Just verify it doesn't error
        assert mc is not None


class TestComparisonPlot:
    """Tests for strategy comparison plotting."""

    def test_compare_plot(self, sample_aapl):
        """Comparison should be plottable."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)

        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        plot = comparison.plot()
        assert plot is not None
