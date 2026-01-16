"""
Tests for data loading functionality.

Tests cover:
- Sample data loading (AAPL, SPY, BTC)
- Data format validation
- OHLC constraint verification
- Timestamp ordering
"""

import numpy as np
import pytest

import mantis as mt


class TestLoadSample:
    """Tests for mt.load_sample() function."""

    def test_load_aapl(self):
        """Load AAPL sample data and verify structure."""
        data = mt.load_sample("AAPL")

        assert "timestamp" in data
        assert "open" in data
        assert "high" in data
        assert "low" in data
        assert "close" in data
        assert "volume" in data
        assert "n_bars" in data

        # AAPL should have ~10 years of daily data
        assert data["n_bars"] > 2500

    def test_load_spy(self):
        """Load SPY sample data and verify structure."""
        data = mt.load_sample("SPY")

        assert data["n_bars"] > 2500
        assert len(data["close"]) == data["n_bars"]

    def test_load_btc(self):
        """Load BTC sample data (includes weekends)."""
        data = mt.load_sample("BTC")

        # BTC has more bars due to 24/7 trading
        assert data["n_bars"] > 3000

    def test_load_sample_case_insensitive(self):
        """Sample names should be case-insensitive."""
        data1 = mt.load_sample("AAPL")
        data2 = mt.load_sample("aapl")
        data3 = mt.load_sample("Aapl")

        assert data1["n_bars"] == data2["n_bars"] == data3["n_bars"]

    def test_load_invalid_sample(self):
        """Loading invalid sample should raise error."""
        with pytest.raises(Exception):
            mt.load_sample("INVALID_SYMBOL")

    def test_list_samples(self):
        """List available samples."""
        samples = mt.list_samples()

        assert isinstance(samples, list)
        assert "AAPL" in samples or "aapl" in [s.lower() for s in samples]
        assert "SPY" in samples or "spy" in [s.lower() for s in samples]
        assert "BTC" in samples or "btc" in [s.lower() for s in samples]


class TestDataValidation:
    """Tests for data validation and OHLC constraints."""

    def test_ohlc_constraints(self, sample_aapl):
        """Verify OHLC constraints: High >= Low, High >= Open/Close."""
        opens = sample_aapl["open"]
        highs = sample_aapl["high"]
        lows = sample_aapl["low"]
        closes = sample_aapl["close"]

        # High should be >= Low for all bars
        assert np.all(highs >= lows)

        # High should be >= Open and Close
        assert np.all(highs >= opens)
        assert np.all(highs >= closes)

        # Low should be <= Open and Close
        assert np.all(lows <= opens)
        assert np.all(lows <= closes)

    def test_positive_prices(self, sample_aapl):
        """All prices should be positive."""
        assert np.all(sample_aapl["open"] > 0)
        assert np.all(sample_aapl["high"] > 0)
        assert np.all(sample_aapl["low"] > 0)
        assert np.all(sample_aapl["close"] > 0)

    def test_positive_volume(self, sample_aapl):
        """Volume should be non-negative."""
        assert np.all(sample_aapl["volume"] >= 0)

    def test_timestamps_sorted(self, sample_aapl):
        """Timestamps should be sorted in ascending order."""
        timestamps = sample_aapl["timestamp"]
        assert np.all(timestamps[1:] > timestamps[:-1])

    def test_no_duplicate_timestamps(self, sample_aapl):
        """No duplicate timestamps."""
        timestamps = sample_aapl["timestamp"]
        assert len(timestamps) == len(np.unique(timestamps))


class TestDataArrays:
    """Tests for numpy array properties."""

    def test_array_dtypes(self, sample_aapl):
        """Verify correct data types."""
        assert sample_aapl["timestamp"].dtype == np.int64
        assert sample_aapl["open"].dtype == np.float64
        assert sample_aapl["high"].dtype == np.float64
        assert sample_aapl["low"].dtype == np.float64
        assert sample_aapl["close"].dtype == np.float64
        assert sample_aapl["volume"].dtype == np.float64

    def test_array_lengths_match(self, sample_aapl):
        """All arrays should have the same length."""
        n_bars = sample_aapl["n_bars"]
        assert len(sample_aapl["timestamp"]) == n_bars
        assert len(sample_aapl["open"]) == n_bars
        assert len(sample_aapl["high"]) == n_bars
        assert len(sample_aapl["low"]) == n_bars
        assert len(sample_aapl["close"]) == n_bars
        assert len(sample_aapl["volume"]) == n_bars

    def test_no_nan_values(self, sample_aapl):
        """Sample data should have no NaN values."""
        assert not np.any(np.isnan(sample_aapl["open"]))
        assert not np.any(np.isnan(sample_aapl["high"]))
        assert not np.any(np.isnan(sample_aapl["low"]))
        assert not np.any(np.isnan(sample_aapl["close"]))
        assert not np.any(np.isnan(sample_aapl["volume"]))

    def test_no_inf_values(self, sample_aapl):
        """Sample data should have no infinite values."""
        assert not np.any(np.isinf(sample_aapl["open"]))
        assert not np.any(np.isinf(sample_aapl["high"]))
        assert not np.any(np.isinf(sample_aapl["low"]))
        assert not np.any(np.isinf(sample_aapl["close"]))
        assert not np.any(np.isinf(sample_aapl["volume"]))


class TestMultiSymbol:
    """Tests for multi-symbol data loading."""

    def test_load_multiple_samples(self):
        """Load multiple symbols and verify independence."""
        aapl = mt.load_sample("AAPL")
        spy = mt.load_sample("SPY")

        # Should be independent data
        assert not np.array_equal(aapl["close"], spy["close"])

    def test_sample_data_realistic_prices(self, sample_aapl, sample_spy, sample_btc):
        """Verify prices are in realistic ranges."""
        # AAPL: Typically $50-200 range in 2014-2024
        assert np.min(sample_aapl["close"]) > 10
        assert np.max(sample_aapl["close"]) < 500

        # SPY: Typically $150-500 range
        assert np.min(sample_spy["close"]) > 50
        assert np.max(sample_spy["close"]) < 1000

        # BTC: Wide range $100 - $100,000+
        assert np.min(sample_btc["close"]) > 0
