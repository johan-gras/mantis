"""
Tests for sensitivity analysis functionality.

Tests cover:
- Parameter sensitivity analysis
- Heatmap generation
- Stability scoring
- Cliff and plateau detection
"""

import numpy as np
import pytest

import mantis as mt


class TestParameterSensitivity:
    """Tests for parameter sensitivity analysis."""

    def test_sensitivity_basic(self, sample_aapl):
        """Basic sensitivity analysis."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 20, 4),
                "slow": mt.linear_range(30, 60, 4),
            },
        )

        assert hasattr(sensitivity, "best_params")
        best = sensitivity.best_params()
        assert best is not None

    def test_sensitivity_stability_score(self, sample_aapl):
        """Stability score should be between 0 and 1."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        score = sensitivity.stability_score()
        assert 0 <= score <= 1

    def test_sensitivity_parameter_stability(self, sample_aapl):
        """Per-parameter stability should be calculable."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        fast_stability = sensitivity.parameter_stability("fast")
        if fast_stability is not None:
            assert 0 <= fast_stability <= 1

    def test_sensitivity_is_fragile(self, sample_aapl):
        """is_fragile() should return boolean."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        fragile = sensitivity.is_fragile()
        assert isinstance(fragile, bool)


class TestSensitivityHeatmap:
    """Tests for sensitivity heatmap generation."""

    def test_heatmap_generation(self, sample_aapl):
        """Heatmap should be generated for 2D parameter space."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        heatmap = sensitivity.heatmap("fast", "slow")

        if heatmap is not None:
            assert hasattr(heatmap, "x_param")
            assert hasattr(heatmap, "y_param")
            assert heatmap.x_param == "fast"
            assert heatmap.y_param == "slow"

    def test_heatmap_values(self, sample_aapl):
        """Heatmap should have correct dimensions."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        heatmap = sensitivity.heatmap("fast", "slow")

        if heatmap is not None:
            x_values = heatmap.x_values()
            y_values = heatmap.y_values()
            values = heatmap.values()

            assert len(x_values) == 3
            assert len(y_values) == 3
            # Values should be a 2D array
            assert values.shape == (3, 3)


class TestSensitivityCliffs:
    """Tests for cliff detection (sharp performance drops)."""

    def test_cliffs_detection(self, sample_aapl):
        """Cliffs should be detectable."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 50, 10),
                "slow": mt.linear_range(30, 100, 10),
            },
        )

        cliffs = sensitivity.cliffs()
        assert isinstance(cliffs, list)

        # If cliffs exist, verify structure
        for cliff in cliffs:
            assert hasattr(cliff, "parameter")
            assert hasattr(cliff, "drop_pct")


class TestSensitivityPlateaus:
    """Tests for plateau detection (stable regions)."""

    def test_plateaus_detection(self, sample_aapl):
        """Plateaus should be detectable."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 20, 4),
                "slow": mt.linear_range(30, 60, 4),
            },
        )

        plateaus = sensitivity.plateaus()
        assert isinstance(plateaus, list)

        # If plateaus exist, verify structure
        for plateau in plateaus:
            assert hasattr(plateau, "parameter")
            assert hasattr(plateau, "start_value")
            assert hasattr(plateau, "end_value")


class TestSensitivityImportance:
    """Tests for parameter importance ranking."""

    def test_parameter_importance(self, sample_aapl):
        """Parameter importance should rank parameters."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        importance = sensitivity.parameter_importance()
        assert isinstance(importance, list)

        # Should have entries for each parameter
        assert len(importance) >= 1


class TestSensitivitySummary:
    """Tests for sensitivity summary."""

    def test_sensitivity_summary(self, sample_aapl):
        """Summary should provide overview statistics."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        summary = sensitivity.summary()

        assert hasattr(summary, "num_combinations")
        assert hasattr(summary, "stability_score")
        assert hasattr(summary, "is_fragile")

    def test_sensitivity_to_csv(self, sample_aapl):
        """to_csv() should export results."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        csv = sensitivity.to_csv()
        assert isinstance(csv, str)
        # Should have header row
        assert "fast" in csv or "slow" in csv


class TestSensitivityPlotting:
    """Tests for sensitivity visualization."""

    def test_plot_heatmap_exists(self, sample_aapl):
        """plot_heatmap() method should exist."""
        sensitivity = mt.sensitivity(
            sample_aapl,
            strategy="sma-crossover",
            params={
                "fast": mt.linear_range(5, 15, 3),
                "slow": mt.linear_range(30, 50, 3),
            },
        )

        assert hasattr(sensitivity, "plot_heatmap")
