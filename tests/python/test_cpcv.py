"""Tests for CPCV (Combinatorial Purged Cross-Validation) functionality."""

import numpy as np
import pytest

import mantis as mt


class TestCPCVConfig:
    """Tests for CPCVConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = mt.CPCVConfig()
        assert config.n_splits == 5
        assert config.n_test_splits == 1
        assert config.embargo_days == 5
        assert config.purge_overlapping is True
        assert config.min_bars_per_fold == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = mt.CPCVConfig(
            n_splits=10,
            n_test_splits=2,
            embargo_days=10,
            purge_overlapping=False,
            min_bars_per_fold=100
        )
        assert config.n_splits == 10
        assert config.n_test_splits == 2
        assert config.embargo_days == 10
        assert config.purge_overlapping is False
        assert config.min_bars_per_fold == 100

    def test_config_validation_n_splits(self):
        """Test that n_splits must be >= 2."""
        with pytest.raises(ValueError):
            mt.CPCVConfig(n_splits=1)

    def test_config_validation_test_splits(self):
        """Test that n_test_splits must be < n_splits."""
        with pytest.raises(ValueError):
            mt.CPCVConfig(n_splits=5, n_test_splits=5)

    def test_config_validation_embargo(self):
        """Test that embargo_days must be >= 1."""
        with pytest.raises(ValueError):
            mt.CPCVConfig(embargo_days=0)

    def test_config_repr(self):
        """Test configuration string representation."""
        config = mt.CPCVConfig(n_splits=5, embargo_days=10)
        repr_str = repr(config)
        assert "CPCVConfig" in repr_str
        assert "5" in repr_str
        assert "10" in repr_str


class TestCPCVWithBuiltInStrategy:
    """Tests for CPCV with built-in strategies."""

    @pytest.fixture
    def sample_data(self):
        """Load sample data for tests."""
        return mt.load_sample("AAPL")

    def test_cpcv_basic(self, sample_data):
        """Test basic CPCV execution."""
        result = mt.cpcv(
            sample_data,
            strategy="sma-crossover",
            strategy_params={"fast_period": 10, "slow_period": 30}
        )
        assert result is not None
        assert result.n_folds > 0
        assert result.n_combinations > 0

    def test_cpcv_with_config(self, sample_data):
        """Test CPCV with custom configuration."""
        config = mt.CPCVConfig(n_splits=4, embargo_days=3)
        result = mt.cpcv(
            sample_data,
            strategy="momentum",
            config=config
        )
        assert result.n_folds <= 4  # May be fewer due to purging

    def test_cpcv_different_metrics(self, sample_data):
        """Test CPCV with different metrics."""
        for metric in ["sharpe", "sortino", "return", "calmar", "profit_factor"]:
            result = mt.cpcv(
                sample_data,
                strategy="rsi",
                metric=metric
            )
            assert result.metric == metric

    def test_cpcv_different_strategies(self, sample_data):
        """Test CPCV with different built-in strategies."""
        strategies = [
            ("sma-crossover", {"fast_period": 10, "slow_period": 30}),
            ("momentum", {"period": 20}),
            ("mean-reversion", {"period": 20}),
            ("rsi", {"period": 14}),
            ("macd", {}),
            ("breakout", {"entry_period": 20}),
        ]
        for strategy_name, params in strategies:
            result = mt.cpcv(
                sample_data,
                strategy=strategy_name,
                strategy_params=params
            )
            assert result.n_folds > 0


class TestCPCVWithSignal:
    """Tests for CPCV with custom signal arrays."""

    @pytest.fixture
    def sample_data(self):
        """Load sample data for tests."""
        return mt.load_sample("AAPL")

    def test_cpcv_with_numpy_signal(self, sample_data):
        """Test CPCV with numpy signal array."""
        close = sample_data["close"]
        # Simple momentum signal
        signal = np.where(close[20:] > close[:-20], 1, -1)
        # Pad to match data length
        signal = np.concatenate([np.zeros(20), signal])

        result = mt.cpcv(sample_data, signal=signal)
        assert result.n_folds > 0

    def test_cpcv_with_list_signal(self, sample_data):
        """Test CPCV with list signal."""
        n = len(sample_data["close"])
        signal = [1 if i % 2 == 0 else -1 for i in range(n)]

        result = mt.cpcv(sample_data, signal=signal)
        assert result.n_folds > 0


class TestCPCVResult:
    """Tests for CPCVResult class."""

    @pytest.fixture
    def cpcv_result(self):
        """Get a CPCV result for tests."""
        data = mt.load_sample("AAPL")
        return mt.cpcv(
            data,
            strategy="sma-crossover",
            strategy_params={"fast_period": 10, "slow_period": 30}
        )

    def test_result_properties(self, cpcv_result):
        """Test result properties."""
        assert isinstance(cpcv_result.n_folds, int)
        assert isinstance(cpcv_result.n_combinations, int)
        assert isinstance(cpcv_result.mean_test_score, float)
        assert isinstance(cpcv_result.std_test_score, float)
        assert isinstance(cpcv_result.min_test_score, float)
        assert isinstance(cpcv_result.max_test_score, float)
        assert isinstance(cpcv_result.metric, str)

    def test_coefficient_of_variation(self, cpcv_result):
        """Test coefficient of variation calculation."""
        cv = cpcv_result.coefficient_of_variation()
        assert isinstance(cv, float)
        if cpcv_result.mean_test_score != 0:
            expected_cv = abs(cpcv_result.std_test_score / cpcv_result.mean_test_score)
            assert abs(cv - expected_cv) < 0.001

    def test_is_robust(self, cpcv_result):
        """Test robustness check."""
        # Test with different thresholds
        assert isinstance(cpcv_result.is_robust(), bool)
        assert isinstance(cpcv_result.is_robust(0.5), bool)
        assert isinstance(cpcv_result.is_robust(1.0), bool)

    def test_fold_details(self, cpcv_result):
        """Test fold details retrieval."""
        folds = cpcv_result.fold_details()
        assert isinstance(folds, list)
        assert len(folds) == cpcv_result.n_folds

        for fold in folds:
            assert hasattr(fold, 'fold')
            assert hasattr(fold, 'test_result')

    def test_scores(self, cpcv_result):
        """Test scores retrieval."""
        scores = cpcv_result.scores()
        assert isinstance(scores, list)
        assert len(scores) == cpcv_result.n_folds
        assert all(isinstance(s, float) for s in scores)

    def test_summary(self, cpcv_result):
        """Test summary generation."""
        summary = cpcv_result.summary()
        assert isinstance(summary, str)
        assert "CPCV Analysis" in summary
        assert "Mean score" in summary

    def test_repr_str(self, cpcv_result):
        """Test string representations."""
        repr_str = repr(cpcv_result)
        str_str = str(cpcv_result)

        assert isinstance(repr_str, str)
        assert isinstance(str_str, str)


class TestCPCVFoldResult:
    """Tests for CPCVFoldResult class."""

    @pytest.fixture
    def fold_result(self):
        """Get a fold result for tests."""
        data = mt.load_sample("AAPL")
        result = mt.cpcv(
            data,
            strategy="sma-crossover",
            strategy_params={"fast_period": 10, "slow_period": 30}
        )
        folds = result.fold_details()
        return folds[0] if folds else None

    def test_fold_result_structure(self, fold_result):
        """Test fold result structure."""
        if fold_result is None:
            pytest.skip("No fold results available")

        # Check fold info
        fold = fold_result.fold
        assert hasattr(fold, 'index')
        assert hasattr(fold, 'train_bars')
        assert hasattr(fold, 'test_bars')
        assert hasattr(fold, 'purged_bars')

        # Check test result
        test_result = fold_result.test_result
        assert hasattr(test_result, 'sharpe')
        assert hasattr(test_result, 'total_return')


class TestCPCVErrors:
    """Tests for CPCV error handling."""

    def test_missing_strategy_and_signal(self):
        """Test error when neither strategy nor signal provided."""
        data = mt.load_sample("AAPL")
        with pytest.raises(ValueError):
            mt.cpcv(data)

    def test_unknown_strategy(self):
        """Test error for unknown strategy name."""
        data = mt.load_sample("AAPL")
        with pytest.raises(ValueError):
            mt.cpcv(data, strategy="unknown-strategy")

    def test_unknown_metric(self):
        """Test error for unknown metric name."""
        data = mt.load_sample("AAPL")
        with pytest.raises(ValueError):
            mt.cpcv(data, strategy="sma-crossover", metric="unknown")
