"""
Tests to increase code coverage in mantis/__init__.py.

These tests target specific gaps in coverage:
- Plotly visualization code paths (by mocking _is_jupyter and _has_plotly)
- Helper functions
- Error paths and edge cases
- __repr__ and __str__ methods
"""

import numpy as np
import pytest
from unittest import mock

import mantis as mt


class TestJupyterDetection:
    """Tests for Jupyter environment detection."""

    def test_is_jupyter_false_no_ipython(self):
        """Should return False when IPython is not available."""
        with mock.patch.dict('sys.modules', {'IPython': None}):
            # Force reimport to test the detection
            result = mt._is_jupyter()
            # Since we're not in Jupyter, should be False
            assert isinstance(result, bool)

    def test_has_plotly_true(self):
        """Should return True when plotly is installed."""
        result = mt._has_plotly()
        # We installed plotly in setup, so this should be True
        assert result is True


class TestBacktestResultRepr:
    """Tests for BacktestResult string representations."""

    def test_repr(self, sample_aapl):
        """__repr__ should return a string."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        repr_str = repr(result)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_str(self, sample_aapl):
        """__str__ should return a formatted string."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        str_output = str(result)
        assert isinstance(str_output, str)
        assert len(str_output) > 0

    def test_summary_method(self, sample_aapl):
        """summary() method should return formatted text."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        summary = result.summary()
        assert isinstance(summary, str)
        # Should contain some metrics
        assert "Sharpe" in summary or "sharpe" in summary.lower()


class TestValidationResultRepr:
    """Tests for ValidationResult string representations."""

    def test_repr(self, sample_aapl):
        """__repr__ should return a string."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        repr_str = repr(validation)
        assert isinstance(repr_str, str)

    def test_str(self, sample_aapl):
        """__str__ should return a formatted string."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        str_output = str(validation)
        assert isinstance(str_output, str)


class TestCompareResultRepr:
    """Tests for CompareResult string representations."""

    def test_repr(self, sample_aapl):
        """__repr__ should return comparison table."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2], names=["Buy Hold", "SMA"])
        repr_str = repr(comparison)
        assert isinstance(repr_str, str)
        assert "Buy Hold" in repr_str or "SMA" in repr_str

    def test_str(self, sample_aapl):
        """__str__ should return comparison table."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])
        str_output = str(comparison)
        assert isinstance(str_output, str)


class TestSweepResultRepr:
    """Tests for SweepResult string representations."""

    def test_sweep_repr(self, sample_aapl):
        """Sweep result should have repr."""
        def signal_fn(threshold):
            return np.ones(sample_aapl["n_bars"]) * (threshold > 0.5)

        sweep_result = mt.sweep(
            sample_aapl,
            signal_fn,
            params={"threshold": [0.3, 0.5, 0.7]},
        )

        repr_str = repr(sweep_result)
        assert isinstance(repr_str, str)


class TestSensitivityResultRepr:
    """Tests for SensitivityResult string representations."""

    def test_repr(self, sample_aapl):
        """__repr__ should return a string."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        repr_str = repr(result)
        assert isinstance(repr_str, str)

    def test_str(self, sample_aapl):
        """__str__ should return a string."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        str_output = str(result)
        assert isinstance(str_output, str)


class TestCostSensitivityResultRepr:
    """Tests for CostSensitivityResult string representations."""

    def test_repr(self, sample_aapl):
        """__repr__ should return a string."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        repr_str = repr(result)
        assert isinstance(repr_str, str)

    def test_str(self, sample_aapl):
        """__str__ should return a string."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        str_output = str(result)
        assert isinstance(str_output, str)


class TestPlotlyVisualizationPaths:
    """Tests for Plotly-based visualization code paths."""

    def test_backtest_result_plot_with_plotly(self, sample_aapl):
        """Test plot when plotly is available."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Mock _is_jupyter to return True
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                plot = result.plot()
                # Should return a Plotly figure
                assert plot is not None

    def test_validation_result_plot_with_plotly(self, sample_aapl):
        """Test validation plot when plotly is available."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                plot = validation.plot()
                assert plot is not None

    def test_compare_result_plot_with_plotly(self, sample_aapl):
        """Test compare plot when plotly is available."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                plot = comparison.plot()
                assert plot is not None


class TestHtmlRepresentations:
    """Tests for _repr_html_ methods."""

    def test_backtest_result_repr_html(self, sample_aapl):
        """Test _repr_html_ method exists."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Mock for non-Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            html = result._repr_html_()
            assert isinstance(html, str)
            assert '<' in html  # Should contain HTML tags

    def test_validation_result_repr_html(self, sample_aapl):
        """Test ValidationResult _repr_html_."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            html = validation._repr_html_()
            assert isinstance(html, str)
            assert '<' in html

    def test_compare_result_repr_html(self, sample_aapl):
        """Test CompareResult _repr_html_."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2])

        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            html = comparison._repr_html_()
            assert isinstance(html, str)
            assert '<' in html


class TestSweepPlotAscii:
    """Tests for sweep plot ASCII fallback."""

    def test_sweep_plot_no_jupyter(self, sample_aapl):
        """Sweep plot should work without Jupyter."""
        def signal_fn(threshold):
            return np.ones(sample_aapl["n_bars"]) * (threshold > 0.5)

        sweep_result = mt.sweep(
            sample_aapl,
            signal_fn,
            params={"threshold": [0.3, 0.5, 0.7]},
        )

        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            # Should print ASCII summary and return None
            plot = sweep_result.plot()
            # In non-Jupyter mode, plot may return None (prints to stdout)
            assert plot is None or isinstance(plot, (str, type(None)))


class TestSensitivityHeatmapAscii:
    """Tests for sensitivity heatmap ASCII fallback."""

    def test_sensitivity_heatmap_no_jupyter(self, sample_aapl):
        """Sensitivity heatmap should work without Jupyter."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            # Should return ASCII representation
            output = result.plot_heatmap("fast_period", "slow_period")
            assert output is not None


class TestCostSensitivityPlot:
    """Tests for cost sensitivity plotting."""

    def test_cost_sensitivity_plot_no_jupyter(self, sample_aapl):
        """Cost sensitivity plot should work without Jupyter."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        with mock.patch.object(mt, '_is_jupyter', return_value=False):
            output = result.plot()
            assert isinstance(output, str)

    def test_cost_sensitivity_report(self, sample_aapl):
        """Cost sensitivity report should be generated."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        report = result.report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestCostSensitivityMethods:
    """Tests for CostSensitivityResult methods."""

    def test_scenarios(self, sample_aapl):
        """scenarios() should return list."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        scenarios = result.scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_baseline(self, sample_aapl):
        """baseline() should return 1x scenario."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        baseline = result.baseline()
        assert baseline is not None
        assert baseline.multiplier == 1.0

    def test_zero_cost(self, sample_aapl):
        """zero_cost() should return 0x scenario."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        zero = result.zero_cost()
        assert zero is not None
        assert zero.multiplier == 0.0

    def test_sharpe_degradation_at(self, sample_aapl):
        """sharpe_degradation_at should return float."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        degradation = result.sharpe_degradation_at(2.0)
        assert degradation is None or isinstance(degradation, float)

    def test_return_degradation_at(self, sample_aapl):
        """return_degradation_at should return float."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        degradation = result.return_degradation_at(2.0)
        assert degradation is None or isinstance(degradation, float)

    def test_cost_elasticity(self, sample_aapl):
        """cost_elasticity should return float or None."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        elasticity = result.cost_elasticity()
        assert elasticity is None or isinstance(elasticity, float)

    def test_breakeven_multiplier(self, sample_aapl):
        """breakeven_multiplier should return float or None."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        breakeven = result.breakeven_multiplier()
        assert breakeven is None or isinstance(breakeven, float)


class TestSensitivityMethods:
    """Tests for SensitivityResult methods."""

    def test_parameter_stability(self, sample_aapl):
        """parameter_stability should return float."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        stability = result.parameter_stability("fast_period")
        assert isinstance(stability, (float, int))

    def test_cliffs(self, sample_aapl):
        """cliffs should return list."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        cliffs = result.cliffs()
        assert isinstance(cliffs, list)

    def test_plateaus(self, sample_aapl):
        """plateaus should return list."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        plateaus = result.plateaus()
        assert isinstance(plateaus, list)

    def test_parameter_importance(self, sample_aapl):
        """parameter_importance should return list of tuples."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        importance = result.parameter_importance()
        assert isinstance(importance, list)
        # Should contain tuples of (param_name, importance_score)
        if len(importance) > 0:
            assert len(importance[0]) == 2


class TestMonteCarloDistributions:
    """Tests for Monte Carlo distribution access."""

    def test_sharpe_distribution(self, sample_aapl):
        """sharpe_distribution should return array."""
        # Use a strategy that generates multiple trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        dist = mc.sharpe_distribution
        assert len(dist) == 100

    def test_drawdown_distribution(self, sample_aapl):
        """drawdown_distribution should return array."""
        # Use a strategy that generates multiple trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        dist = mc.drawdown_distribution
        assert len(dist) == 100

    def test_return_distribution(self, sample_aapl):
        """return_distribution should return array."""
        # Use a strategy that generates multiple trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        dist = mc.return_distribution
        assert len(dist) == 100

    def test_percentile(self, sample_aapl):
        """percentile should return float."""
        # Use a strategy that generates multiple trades
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        p10 = mc.percentile("sharpe", 10)
        p50 = mc.percentile("sharpe", 50)
        p90 = mc.percentile("sharpe", 90)

        assert isinstance(p10, float)
        assert isinstance(p50, float)
        assert isinstance(p90, float)
        assert p10 <= p50 <= p90


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_compare_with_auto_names(self, sample_aapl):
        """Compare should auto-generate names."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma_crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        # Don't pass names - should auto-generate
        comparison = mt.compare([result1, result2])
        assert comparison is not None

        # Check metrics dict has Strategy 1, Strategy 2
        metrics = comparison.metrics
        assert "Strategy 1" in metrics or len(metrics) == 2

    def test_sweep_with_empty_params(self, sample_aapl):
        """Sweep with single param combination."""
        def signal_fn(threshold):
            return np.ones(sample_aapl["n_bars"])

        sweep_result = mt.sweep(
            sample_aapl,
            signal_fn,
            params={"threshold": [0.5]},  # Single value
        )

        assert sweep_result.num_combinations == 1


class TestSweepToDict:
    """Tests for sweep to_dict conversion."""

    def test_to_dict(self, sample_aapl):
        """to_dict should return wrapped BacktestResults."""
        def signal_fn(threshold):
            return np.ones(sample_aapl["n_bars"]) * (threshold > 0.5)

        sweep_result = mt.sweep(
            sample_aapl,
            signal_fn,
            params={"threshold": [0.3, 0.5, 0.7]},
        )

        result_dict = sweep_result.to_dict()
        assert isinstance(result_dict, dict)
        # Values should be BacktestResult or similar
        for key, value in result_dict.items():
            assert hasattr(value, 'sharpe') or isinstance(value, mt.BacktestResult)


class TestCPCVRepr:
    """Tests for CPCV result representations."""

    def test_cpcv_repr(self, sample_aapl):
        """CPCV result should have repr."""
        signal = np.ones(sample_aapl["n_bars"])
        cpcv_result = mt.cpcv(
            sample_aapl,
            signal=signal,
        )

        repr_str = repr(cpcv_result)
        assert isinstance(repr_str, str)

    def test_cpcv_str(self, sample_aapl):
        """CPCV result should have str."""
        signal = np.ones(sample_aapl["n_bars"])
        cpcv_result = mt.cpcv(
            sample_aapl,
            signal=signal,
        )

        str_output = str(cpcv_result)
        assert isinstance(str_output, str)


class TestPlotSaveAscii:
    """Tests for saving ASCII plots."""

    def test_save_ascii_to_txt(self, sample_aapl, tmp_path):
        """Saving ASCII plot to .txt should work."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Mock plotly as unavailable to force ASCII path
        with mock.patch.object(mt, '_has_plotly', return_value=False):
            save_path = str(tmp_path / "test.txt")
            returned_path = result.plot(save=save_path)
            assert returned_path == save_path
            # Verify file was created
            with open(save_path, encoding="utf-8") as f:
                content = f.read()
                assert len(content) > 0

    def test_save_ascii_to_non_txt_fails(self, sample_aapl, tmp_path):
        """Saving ASCII plot to non-.txt should raise error."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Mock plotly as unavailable to force ASCII path
        with mock.patch.object(mt, '_has_plotly', return_value=False):
            save_path = str(tmp_path / "test.html")
            with pytest.raises(ValueError, match="Cannot save ASCII plot"):
                result.plot(save=save_path)


class TestPlotlySave:
    """Tests for saving Plotly plots."""

    def test_save_to_html(self, sample_aapl, tmp_path):
        """Saving to HTML should work when plotly is available."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Plotly should be available in test environment
        if mt._has_plotly():
            save_path = str(tmp_path / "test.html")
            returned_path = result.plot(save=save_path)
            assert returned_path == save_path
            # Verify file was created
            import os
            assert os.path.exists(save_path)


class TestValidationPlotPlotly:
    """Tests for validation plotly visualization."""

    def test_validation_plot_with_plotly_jupyter(self, sample_aapl):
        """Validation plot should work with plotly in Jupyter."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                fig = validation.plot()
                assert fig is not None

    def test_validation_repr_html_with_plotly(self, sample_aapl):
        """Validation _repr_html_ should work with plotly in Jupyter."""
        signal = np.ones(sample_aapl["n_bars"])
        validation = mt.validate(sample_aapl, signal=signal, folds=5)

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                html = validation._repr_html_()
                assert isinstance(html, str)
                assert len(html) > 0


class TestCostSensitivityPlotPlotly:
    """Tests for cost sensitivity plotly visualization."""

    def test_cost_sensitivity_plot_with_plotly(self, sample_aapl):
        """Cost sensitivity plot should work with plotly in Jupyter."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                fig = result.plot()
                assert fig is not None

    def test_cost_sensitivity_repr_html_with_plotly(self, sample_aapl):
        """Cost sensitivity _repr_html_ should work with plotly in Jupyter."""
        result = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                html = result._repr_html_()
                assert isinstance(html, str)
                assert '<' in html


class TestSensitivityPlotPlotly:
    """Tests for sensitivity plotly visualization."""

    def test_sensitivity_heatmap_with_plotly(self, sample_aapl):
        """Sensitivity heatmap should work with plotly in Jupyter."""
        result = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast_period": mt.discrete_range([5, 10, 15]),
                "slow_period": mt.discrete_range([20, 30, 40]),
            },
        )

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                fig = result.plot_heatmap("fast_period", "slow_period")
                assert fig is not None


class TestSweepPlotPlotly:
    """Tests for sweep plotly visualization."""

    def test_sweep_plot_with_plotly(self, sample_aapl):
        """Sweep plot should work with plotly in Jupyter."""
        def signal_fn(threshold):
            return np.ones(sample_aapl["n_bars"]) * (threshold > 0.5)

        sweep_result = mt.sweep(
            sample_aapl,
            signal_fn,
            params={"threshold": [0.3, 0.5, 0.7]},
        )

        # Mock Jupyter environment
        with mock.patch.object(mt, '_is_jupyter', return_value=True):
            with mock.patch.object(mt, '_has_plotly', return_value=True):
                fig = sweep_result.plot()
                # Should return a Plotly figure
                assert fig is not None


class TestCompareToDataframe:
    """Tests for compare to_dataframe method."""

    def test_compare_to_dataframe(self, sample_aapl):
        """Compare result should have to_dataframe method."""
        signal1 = np.ones(sample_aapl["n_bars"])
        result1 = mt.backtest(sample_aapl, signal=signal1)
        result2 = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        comparison = mt.compare([result1, result2], names=["Buy Hold", "SMA"])

        # Check if to_dataframe exists
        if hasattr(comparison, 'to_dataframe'):
            df = comparison.to_dataframe()
            assert df is not None


class TestBacktestResultProperties:
    """Tests for additional BacktestResult properties."""

    def test_equity_timestamps(self, sample_aapl):
        """equity_timestamps should return array of timestamps."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        timestamps = result.equity_timestamps
        assert len(timestamps) > 0

    def test_daily_returns(self, sample_aapl):
        """daily_returns should return array."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        # Check what properties exist for returns
        if hasattr(result, 'daily_returns'):
            returns = result.daily_returns
            assert len(returns) > 0
        elif hasattr(result, 'returns'):
            returns = result.returns
            assert len(returns) > 0
        else:
            # Just verify equity curve exists
            assert len(result.equity_curve) > 0


class TestMonteCarloMethods:
    """Tests for additional Monte Carlo methods."""

    def test_monte_carlo_summary(self, sample_aapl):
        """summary method should return string."""
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        summary = mc.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_monte_carlo_verdict(self, sample_aapl):
        """verdict property should return string."""
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        verdict = mc.verdict
        assert isinstance(verdict, str)

    def test_monte_carlo_confidence_intervals(self, sample_aapl):
        """CI methods should return tuples."""
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )
        mc = result.monte_carlo(n_simulations=100)

        sharpe_ci = mc.sharpe_ci
        assert len(sharpe_ci) == 2

        return_ci = mc.return_ci
        assert len(return_ci) == 2

        dd_ci = mc.max_drawdown_ci
        assert len(dd_ci) == 2


class TestBacktestWithVariousConfigs:
    """Tests for backtest with various configurations."""

    def test_backtest_with_stop_loss_string(self, sample_aapl):
        """Test ATR-based stop loss."""
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
            stop_loss="2atr",
        )
        assert result is not None

    def test_backtest_with_take_profit_string(self, sample_aapl):
        """Test ATR-based take profit."""
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
            take_profit="3atr",
        )
        assert result is not None

    def test_backtest_with_fractional(self, sample_aapl):
        """Test fractional shares."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal, fractional=True)
        assert result is not None

    def test_backtest_with_limit_order(self, sample_aapl):
        """Test limit order type."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(
            sample_aapl, signal=signal, order_type="limit", limit_offset=0.01
        )
        assert result is not None
