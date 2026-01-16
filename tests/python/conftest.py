"""
Pytest configuration and fixtures for Mantis Python tests.

This module provides shared fixtures used across all test files.
"""

import numpy as np
import pytest

# Import mantis - will fail if not installed
try:
    import mantis as mt
except ImportError:
    pytest.skip("mantis package not installed", allow_module_level=True)


# -----------------------------------------------------------------------------
# Data Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_aapl():
    """Load AAPL sample data (10 years daily)."""
    return mt.load_sample("AAPL")


@pytest.fixture
def sample_spy():
    """Load SPY sample data (10 years daily)."""
    return mt.load_sample("SPY")


@pytest.fixture
def sample_btc():
    """Load BTC sample data (includes weekends)."""
    return mt.load_sample("BTC")


@pytest.fixture
def synthetic_data():
    """
    Create synthetic price data with predictable patterns.

    Returns data with:
    - 252 bars (1 trading year)
    - Upward trending prices
    - Deterministic for reproducibility
    """
    np.random.seed(42)
    n_bars = 252

    # Generate timestamps (daily, starting 2023-01-03)
    start_ts = 1672704000  # 2023-01-03 00:00:00 UTC
    timestamps = np.array([start_ts + i * 86400 for i in range(n_bars)], dtype=np.int64)

    # Generate prices with upward trend + noise
    base_price = 100.0
    daily_return = 0.0005  # ~13% annual return
    prices = np.zeros(n_bars)
    prices[0] = base_price
    for i in range(1, n_bars):
        noise = np.random.normal(0, 0.01)
        prices[i] = prices[i - 1] * (1 + daily_return + noise)

    # Generate OHLCV from close prices
    opens = prices * (1 + np.random.uniform(-0.005, 0.005, n_bars))
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, 0.01, n_bars))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, 0.01, n_bars))
    closes = prices
    volumes = np.random.uniform(1e6, 5e6, n_bars)

    return {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "n_bars": n_bars,
    }


@pytest.fixture
def downtrend_data():
    """
    Create synthetic data with downward trend.

    Useful for testing short selling and drawdown calculations.
    """
    np.random.seed(123)
    n_bars = 252

    start_ts = 1672704000
    timestamps = np.array([start_ts + i * 86400 for i in range(n_bars)], dtype=np.int64)

    # Downward trend
    base_price = 100.0
    daily_return = -0.0005
    prices = np.zeros(n_bars)
    prices[0] = base_price
    for i in range(1, n_bars):
        noise = np.random.normal(0, 0.01)
        prices[i] = prices[i - 1] * (1 + daily_return + noise)

    opens = prices * (1 + np.random.uniform(-0.005, 0.005, n_bars))
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, 0.01, n_bars))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, 0.01, n_bars))
    closes = prices
    volumes = np.random.uniform(1e6, 5e6, n_bars)

    return {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "n_bars": n_bars,
    }


@pytest.fixture
def sideways_data():
    """
    Create synthetic data with no trend (mean-reverting).

    Useful for testing strategies in range-bound markets.
    """
    np.random.seed(456)
    n_bars = 252

    start_ts = 1672704000
    timestamps = np.array([start_ts + i * 86400 for i in range(n_bars)], dtype=np.int64)

    # Sideways with sine wave pattern
    base_price = 100.0
    prices = base_price + 5 * np.sin(np.linspace(0, 4 * np.pi, n_bars))
    prices += np.random.normal(0, 1, n_bars)  # Add noise

    opens = prices * (1 + np.random.uniform(-0.005, 0.005, n_bars))
    highs = np.maximum(opens, prices) * (1 + np.random.uniform(0, 0.01, n_bars))
    lows = np.minimum(opens, prices) * (1 - np.random.uniform(0, 0.01, n_bars))
    closes = prices
    volumes = np.random.uniform(1e6, 5e6, n_bars)

    return {
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "n_bars": n_bars,
    }


# -----------------------------------------------------------------------------
# Signal Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def long_only_signal(synthetic_data):
    """Create a signal that is always long (buy and hold)."""
    n_bars = synthetic_data["n_bars"]
    return np.ones(n_bars)


@pytest.fixture
def short_only_signal(synthetic_data):
    """Create a signal that is always short."""
    n_bars = synthetic_data["n_bars"]
    return -np.ones(n_bars)


@pytest.fixture
def alternating_signal(synthetic_data):
    """Create a signal that alternates between long and short."""
    n_bars = synthetic_data["n_bars"]
    signal = np.zeros(n_bars)
    for i in range(n_bars):
        signal[i] = 1 if (i // 20) % 2 == 0 else -1
    return signal


@pytest.fixture
def random_signal(synthetic_data):
    """Create a random signal for testing (seeded for reproducibility)."""
    np.random.seed(789)
    n_bars = synthetic_data["n_bars"]
    return np.random.choice([-1, 0, 1], size=n_bars).astype(np.float64)


# -----------------------------------------------------------------------------
# Configuration Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Default backtest configuration matching conservative defaults."""
    return {
        "commission": 0.001,
        "slippage": 0.001,
        "cash": 100_000.0,
        "size": 0.10,
        "allow_short": True,
    }


@pytest.fixture
def zero_cost_config():
    """Zero-cost configuration for sanity checks."""
    return {
        "commission": 0.0,
        "slippage": 0.0,
        "cash": 100_000.0,
        "size": 0.10,
        "allow_short": True,
    }


@pytest.fixture
def high_cost_config():
    """High-cost configuration for cost sensitivity testing."""
    return {
        "commission": 0.01,  # 1%
        "slippage": 0.01,    # 1%
        "cash": 100_000.0,
        "size": 0.10,
        "allow_short": True,
    }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def assert_valid_backtest_result(result):
    """
    Assert that a BacktestResult has all required properties and valid values.

    This is a comprehensive validation helper used across tests.
    """
    # Basic properties exist
    assert hasattr(result, "total_return")
    assert hasattr(result, "sharpe")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "total_trades")
    assert hasattr(result, "equity_curve")

    # Equity curve is non-empty
    assert len(result.equity_curve) > 0

    # Max drawdown is non-positive (or zero)
    assert result.max_drawdown <= 0.0 or result.total_trades == 0

    # Win rate in valid range (0-1) or NaN if no trades
    if result.total_trades > 0:
        assert 0.0 <= result.win_rate <= 1.0 or np.isnan(result.win_rate)

    # Final equity should be positive
    assert result.final_equity > 0


def assert_metrics_finite(result):
    """Assert that key metrics are finite (not NaN or inf)."""
    assert np.isfinite(result.total_return)
    assert np.isfinite(result.final_equity)
    assert np.isfinite(result.max_drawdown)

    # Sharpe can be NaN if no trades or zero volatility
    if result.total_trades > 0:
        # May still be inf if zero volatility
        assert np.isfinite(result.sharpe) or np.isinf(result.sharpe)


# Make helper functions available to tests
@pytest.fixture
def validate_result():
    """Provide the validation helper function to tests."""
    return assert_valid_backtest_result


@pytest.fixture
def validate_metrics():
    """Provide the metrics validation helper to tests."""
    return assert_metrics_finite
