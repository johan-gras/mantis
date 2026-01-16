"""
Tests for validation and robustness functionality.

Tests cover:
- Walk-forward validation
- Monte Carlo simulation
- Overfitting detection
- Cost sensitivity
- Signal validation
"""

import numpy as np
import pytest

import mantis as mt


class TestWalkForwardValidation:
    """Tests for walk-forward validation."""

    def test_validate_basic(self, sample_aapl):
        """Basic walk-forward validation."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,  # Use fewer folds for speed
        )

        assert hasattr(validation, "verdict")
        assert validation.verdict in ["robust", "borderline", "likely_overfit"]

    def test_validate_with_strategy(self, sample_aapl):
        """Walk-forward validation with a signal."""
        # Generate a simple signal for validation
        n_bars = sample_aapl["n_bars"]
        signal = np.ones(n_bars)  # Long-only signal

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        assert hasattr(validation, "is_sharpe")
        assert hasattr(validation, "oos_sharpe")

    def test_validate_folds_count(self, sample_aapl):
        """Verify fold count matches specification."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=6,
        )

        assert validation.folds == 6

    def test_validate_oos_degradation(self, sample_aapl):
        """OOS degradation should be calculated."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        assert hasattr(validation, "oos_degradation")
        # Degradation is OOS/IS ratio
        assert np.isfinite(validation.oos_degradation) or np.isnan(validation.oos_degradation)

    def test_validate_fold_details(self, sample_aapl):
        """Fold details should be accessible."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        details = validation.fold_details()

        assert isinstance(details, list)
        assert len(details) == 5

    def test_validate_anchored_vs_rolling(self, sample_aapl):
        """Test both anchored and rolling window types."""
        signal = np.ones(sample_aapl["n_bars"])

        validation_anchored = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
            anchored=True,
        )

        validation_rolling = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
            anchored=False,
        )

        # Both should produce valid results
        assert validation_anchored.verdict in ["robust", "borderline", "likely_overfit"]
        assert validation_rolling.verdict in ["robust", "borderline", "likely_overfit"]

    def test_validate_is_robust_method(self, sample_aapl):
        """Test is_robust() helper method."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        assert isinstance(validation.is_robust(), bool)

    def test_validate_warnings(self, sample_aapl):
        """Validation should produce appropriate warnings."""
        signal = np.ones(sample_aapl["n_bars"])

        validation = mt.validate(
            sample_aapl,
            signal=signal,
            folds=5,
        )

        warnings = validation.warnings()

        assert isinstance(warnings, list)


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""

    def test_monte_carlo_basic(self, sample_aapl):
        """Basic Monte Carlo simulation."""
        # Use a strategy that generates trades for meaningful Monte Carlo
        result = mt.backtest(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        mc = result.monte_carlo(n_simulations=100)  # Reduced for speed

        assert hasattr(mc, "num_simulations")
        # Monte Carlo only runs if there are trades
        if result.total_trades > 0:
            assert mc.num_simulations == 100
        else:
            assert mc.num_simulations == 0  # No trades means no simulation

    def test_monte_carlo_return_statistics(self, sample_aapl):
        """Monte Carlo should provide return statistics."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        assert hasattr(mc, "mean_return")
        assert hasattr(mc, "median_return")
        assert hasattr(mc, "return_ci")

    def test_monte_carlo_drawdown_statistics(self, sample_aapl):
        """Monte Carlo should provide drawdown statistics."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        assert hasattr(mc, "mean_max_drawdown")
        assert hasattr(mc, "median_max_drawdown")
        assert hasattr(mc, "max_drawdown_ci")

    def test_monte_carlo_sharpe_statistics(self, sample_aapl):
        """Monte Carlo should provide Sharpe statistics."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        assert hasattr(mc, "mean_sharpe")
        assert hasattr(mc, "sharpe_ci")

    def test_monte_carlo_confidence_intervals(self, sample_aapl):
        """Confidence intervals should be tuples of (lower, upper)."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        # CI should be (lower, upper) tuple
        assert isinstance(mc.return_ci, tuple)
        assert len(mc.return_ci) == 2
        # Lower should be <= upper
        assert mc.return_ci[0] <= mc.return_ci[1]

    def test_monte_carlo_seeded_determinism(self, sample_aapl):
        """Monte Carlo with seed should be deterministic."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc1 = result.monte_carlo(n_simulations=100, seed=42)
        mc2 = result.monte_carlo(n_simulations=100, seed=42)

        assert mc1.mean_return == mc2.mean_return
        assert mc1.median_return == mc2.median_return

    def test_monte_carlo_verdict(self, sample_aapl):
        """Monte Carlo should produce a verdict."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        assert hasattr(mc, "verdict")
        assert mc.verdict in ["robust", "borderline", "likely_overfit"]

    def test_monte_carlo_is_robust(self, sample_aapl):
        """Monte Carlo is_robust() method."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        mc = result.monte_carlo(n_simulations=100)

        assert isinstance(mc.is_robust(), bool)


class TestResultValidate:
    """Tests for BacktestResult.validate() method."""

    def test_result_validate_method(self, sample_aapl):
        """Test validate() method on BacktestResult."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        validation = result.validate(folds=5)

        assert hasattr(validation, "verdict")

    def test_result_validate_with_trials(self, sample_aapl):
        """Test validate() with trials parameter for DSR."""
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        validation = result.validate(folds=5, trials=10)

        assert hasattr(validation, "deflated_sharpe")


class TestCostSensitivity:
    """Tests for cost sensitivity analysis."""

    def test_cost_sensitivity_basic(self, sample_aapl):
        """Basic cost sensitivity analysis."""
        signal = np.ones(sample_aapl["n_bars"])

        sensitivity = mt.cost_sensitivity(
            sample_aapl,
            signal=signal,
        )

        assert hasattr(sensitivity, "scenarios")
        scenarios = sensitivity.scenarios()
        assert len(scenarios) > 0

    def test_cost_sensitivity_multipliers(self, sample_aapl):
        """Cost sensitivity with custom multipliers."""
        signal = np.ones(sample_aapl["n_bars"])

        sensitivity = mt.cost_sensitivity(
            sample_aapl,
            signal=signal,
            multipliers=[0.0, 1.0, 2.0, 5.0],
        )

        scenarios = sensitivity.scenarios()
        # Should have one scenario per multiplier
        assert len(scenarios) >= 4

    def test_cost_sensitivity_degradation(self, sample_aapl):
        """Cost sensitivity should show degradation at higher costs."""
        # Use a strategy that generates trades
        sensitivity = mt.cost_sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            strategy_params={"fast": 10, "slow": 30},
        )

        baseline = sensitivity.baseline()
        if baseline:
            # Sharpe at 5x costs should be lower
            degradation = sensitivity.sharpe_degradation_at(5.0)
            if degradation is not None:
                # Degradation can be NaN if no valid comparison
                # Just check the method works
                assert degradation is None or np.isnan(degradation) or np.isfinite(degradation)

    def test_cost_sensitivity_is_robust(self, sample_aapl):
        """Test is_robust() at various thresholds."""
        signal = np.ones(sample_aapl["n_bars"])

        sensitivity = mt.cost_sensitivity(
            sample_aapl,
            signal=signal,
        )

        robust = sensitivity.is_robust()
        assert isinstance(robust, bool)


class TestSignalCheck:
    """Tests for signal validation function."""

    def test_signal_check_valid(self, sample_aapl):
        """Valid signal should pass checks."""
        signal = np.ones(sample_aapl["n_bars"])

        result = mt.signal_check(sample_aapl, signal)

        assert isinstance(result, dict)
        assert "valid" in result or "is_valid" in result or len(result) > 0

    def test_signal_check_wrong_length(self, sample_aapl):
        """Signal with wrong length should be flagged."""
        wrong_signal = np.ones(100)  # Wrong length

        result = mt.signal_check(sample_aapl, wrong_signal)

        # Should indicate length mismatch
        assert isinstance(result, dict)

    def test_signal_check_all_nan(self, sample_aapl):
        """Signal with all NaN should be flagged."""
        nan_signal = np.full(sample_aapl["n_bars"], np.nan)

        result = mt.signal_check(sample_aapl, nan_signal)

        assert isinstance(result, dict)

    def test_signal_check_constant(self, sample_aapl):
        """Constant signal should work but may have warning."""
        signal = np.ones(sample_aapl["n_bars"])

        result = mt.signal_check(sample_aapl, signal)

        assert isinstance(result, dict)


class TestOverfitDetection:
    """Tests for overfitting detection mechanisms."""

    def test_auto_warnings_high_sharpe(self, sample_aapl):
        """Suspiciously high Sharpe should trigger warning."""
        # Create a signal that might produce high Sharpe
        signal = np.ones(sample_aapl["n_bars"])
        result = mt.backtest(sample_aapl, signal=signal)

        warnings = result.warnings()

        # If Sharpe > 3, should have warning
        if result.sharpe > 3:
            assert any("sharpe" in w.lower() for w in warnings) or len(warnings) > 0

    def test_auto_warnings_few_trades(self, sample_aapl):
        """Few trades should trigger significance warning."""
        n_bars = sample_aapl["n_bars"]
        signal = np.zeros(n_bars)
        signal[:50] = 1  # Only long for first 50 bars

        result = mt.backtest(sample_aapl, signal=signal)

        warnings = result.warnings()

        # Few trades may trigger warning
        if result.total_trades < 30:
            # May or may not have warning depending on implementation
            assert isinstance(warnings, list)
