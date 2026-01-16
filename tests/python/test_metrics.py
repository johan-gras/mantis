"""
Tests for performance metrics calculation.

Tests cover:
- Core metrics (returns, Sharpe, drawdown)
- Advanced metrics (DSR, PSR)
- Benchmark metrics (alpha, beta)
- Rolling metrics
- Metrics edge cases
"""

import numpy as np
import pytest

import mantis as mt


class TestCoreMetrics:
    """Tests for core performance metrics."""

    def test_total_return_calculation(self, sample_aapl):
        """Total return should match equity change."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal, cash=100_000)

        # Total return should match: (final - initial) / initial
        expected_return = (result.final_equity - 100_000) / 100_000
        assert abs(result.total_return - expected_return) < 0.0001

    def test_sharpe_ratio_exists(self, sample_aapl):
        """Sharpe ratio should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "sharpe")
        # Sharpe should be finite or NaN (if no trades)
        assert np.isfinite(result.sharpe) or np.isnan(result.sharpe)

    def test_sortino_ratio_exists(self, sample_aapl):
        """Sortino ratio should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "sortino")
        assert np.isfinite(result.sortino) or np.isnan(result.sortino)

    def test_max_drawdown_negative_or_zero(self, sample_aapl):
        """Max drawdown should be <= 0 (expressed as negative percentage)."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert result.max_drawdown <= 0.0

    def test_win_rate_bounds(self, sample_aapl):
        """Win rate should be between 0 and 1."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_calmar_ratio_exists(self, sample_aapl):
        """Calmar ratio should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "calmar")

    def test_volatility_non_negative(self, sample_aapl):
        """Volatility should be non-negative."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "volatility")
        assert result.volatility >= 0 or np.isnan(result.volatility)

    def test_cagr_calculation(self, sample_aapl):
        """CAGR should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "cagr")
        assert np.isfinite(result.cagr) or np.isnan(result.cagr)


class TestTradeMetrics:
    """Tests for trade-related metrics."""

    def test_total_trades_count(self, sample_aapl):
        """Total trades should be counted correctly."""
        # Alternating signal to generate multiple trades
        n_bars = sample_aapl["n_bars"]
        signal = np.zeros(n_bars)
        for i in range(n_bars):
            signal[i] = 1 if (i // 50) % 2 == 0 else -1

        result = mt.backtest(sample_aapl, signal=signal, allow_short=True)

        # Should have multiple trades from signal changes
        assert result.total_trades > 0

    def test_winning_losing_trades(self, sample_aapl):
        """Winning + losing trades should equal total trades (approximately)."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        if result.total_trades > 0:
            # Note: Some trades might be breakeven
            assert result.winning_trades + result.losing_trades <= result.total_trades

    def test_avg_win_loss(self, sample_aapl):
        """Average win should be positive, average loss should be negative."""
        n_bars = sample_aapl["n_bars"]
        signal = np.zeros(n_bars)
        for i in range(n_bars):
            signal[i] = 1 if (i // 50) % 2 == 0 else -1

        result = mt.backtest(sample_aapl, signal=signal, allow_short=True)

        if result.winning_trades > 0:
            assert result.avg_win >= 0
        if result.losing_trades > 0:
            assert result.avg_loss <= 0

    def test_profit_factor(self, sample_aapl):
        """Profit factor should be non-negative."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        if hasattr(result, "profit_factor") and result.total_trades > 0:
            assert result.profit_factor >= 0 or np.isnan(result.profit_factor)


class TestAdvancedMetrics:
    """Tests for advanced statistical metrics."""

    def test_deflated_sharpe_ratio(self, sample_aapl):
        """DSR should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "deflated_sharpe")
        # DSR can be any real number (positive, negative, or zero)

    def test_psr_exists(self, sample_aapl):
        """Probabilistic Sharpe Ratio should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert hasattr(result, "psr")
        # PSR should be a probability between 0 and 1
        if np.isfinite(result.psr):
            assert 0.0 <= result.psr <= 1.0


class TestBenchmarkMetrics:
    """Tests for benchmark comparison metrics."""

    def test_benchmark_comparison(self, sample_aapl, sample_spy):
        """Test benchmark comparison metrics."""
        signal = np.ones(sample_aapl["n_bars"])

        result = mt.backtest(sample_aapl, signal=signal, benchmark=sample_spy)

        # Should have benchmark-related metrics
        assert hasattr(result, "has_benchmark")
        if result.has_benchmark:
            assert hasattr(result, "alpha")
            assert hasattr(result, "beta")

    def test_alpha_beta_calculation(self, sample_aapl, sample_spy):
        """Alpha and beta should be calculated against benchmark."""
        signal = np.ones(sample_aapl["n_bars"])

        result = mt.backtest(sample_aapl, signal=signal, benchmark=sample_spy)

        if result.has_benchmark:
            # Alpha and beta should be finite
            assert np.isfinite(result.alpha) or np.isnan(result.alpha)
            assert np.isfinite(result.beta) or np.isnan(result.beta)


class TestRollingMetrics:
    """Tests for rolling window metrics."""

    def test_rolling_sharpe(self, sample_aapl):
        """Rolling Sharpe calculation."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        rolling = result.rolling_sharpe(window=252)

        # Should return array
        assert isinstance(rolling, np.ndarray)
        # Length should match equity curve (or be close)
        assert len(rolling) > 0

    def test_rolling_drawdown(self, sample_aapl):
        """Rolling drawdown calculation."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        rolling = result.rolling_drawdown()

        assert isinstance(rolling, np.ndarray)
        # All drawdowns should be <= 0
        assert np.all(rolling[~np.isnan(rolling)] <= 0)

    def test_rolling_volatility(self, sample_aapl):
        """Rolling volatility calculation."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        rolling = result.rolling_volatility(window=21)

        assert isinstance(rolling, np.ndarray)
        # All volatilities should be >= 0
        assert np.all(rolling[~np.isnan(rolling)] >= 0)


class TestMetricsEdgeCases:
    """Tests for edge cases in metrics calculation."""

    def test_zero_trades_metrics(self, sample_aapl):
        """Metrics with zero trades."""
        signal = np.zeros(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert result.total_trades == 0
        # Metrics should be defined (NaN or specific value)
        assert np.isnan(result.win_rate) or result.win_rate == 0

    def test_single_winning_trade(self, synthetic_data):
        """Metrics with single winning trade."""
        n_bars = synthetic_data["n_bars"]
        signal = np.zeros(n_bars)
        signal[0:50] = 1  # Long for first 50 bars only

        result = mt.backtest(synthetic_data, signal=signal)

        # Should have at least one trade
        assert result.total_trades >= 1

    def test_all_losing_trades(self, downtrend_data):
        """Test metrics when strategy loses on every trade."""
        signal = np.ones(downtrend_data["n_bars"])  # Long in downtrend

        result = mt.backtest(downtrend_data, signal=signal)

        # Total return should be negative
        assert result.total_return < 0

    def test_metrics_dictionary(self, sample_aapl):
        """Test metrics() method returns dictionary."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        metrics = result.metrics()

        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics


class TestMetricsSummary:
    """Tests for metrics summary and display."""

    def test_summary_method(self, sample_aapl):
        """Test summary() returns formatted string."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key metric names
        assert "return" in summary.lower() or "sharpe" in summary.lower()

    def test_warnings_method(self, sample_aapl):
        """Test warnings() returns list of warnings."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        warnings = result.warnings()

        assert isinstance(warnings, list)
        # All items should be strings
        for warning in warnings:
            assert isinstance(warning, str)


class TestEquityCurve:
    """Tests for equity curve properties."""

    def test_equity_curve_starts_at_initial_capital(self, sample_aapl):
        """Equity curve should start at initial capital."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal, cash=100_000)

        # First value should be close to initial capital
        assert abs(result.equity_curve[0] - 100_000) < 100

    def test_equity_curve_length(self, sample_aapl):
        """Equity curve length should match data length."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Equity curve length should be close to n_bars
        assert len(result.equity_curve) == sample_aapl["n_bars"]

    def test_equity_curve_positive(self, sample_aapl):
        """Equity should always be positive (no bankruptcy)."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert np.all(result.equity_curve > 0)

    def test_equity_curve_final_matches_final_equity(self, sample_aapl):
        """Final equity curve value should match final_equity property."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        assert abs(result.equity_curve[-1] - result.final_equity) < 0.01
