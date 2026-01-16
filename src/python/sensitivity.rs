//! Python bindings for sensitivity analysis.
//!
//! This module exposes the parameter sensitivity and cost sensitivity analysis
//! functionality to Python users.

use crate::cost_sensitivity::{
    run_cost_sensitivity_analysis, CostScenario, CostSensitivityAnalysis, CostSensitivityConfig,
};
use crate::engine::{BacktestConfig, Engine};
use crate::sensitivity::{
    Cliff, HeatmapData, ParameterRange, ParameterResult, Plateau, SensitivityAnalysis,
    SensitivityConfig, SensitivityMetric, SensitivitySummary,
};
use crate::strategies::{
    Breakout, MacdStrategy, MeanReversion, Momentum, RsiStrategy, SmaCrossover,
};
use crate::strategy::Strategy;
use crate::types::Bar;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use super::backtest::extract_bars;

// =============================================================================
// Parameter Range Python Bindings
// =============================================================================

/// Create a linear parameter range.
///
/// Generates evenly spaced values from start to end.
///
/// Args:
///     start: Starting value
///     end: Ending value
///     steps: Number of values to generate
///
/// Returns:
///     ParameterRange for use in sensitivity analysis.
///
/// Example:
///     >>> fast_range = mt.linear_range(5.0, 20.0, 4)
///     >>> # Generates: [5.0, 10.0, 15.0, 20.0]
#[pyfunction]
#[pyo3(signature = (start, end, steps))]
pub fn linear_range(start: f64, end: f64, steps: usize) -> PyParameterRange {
    PyParameterRange {
        inner: ParameterRange::linear(start, end, steps),
    }
}

/// Create a logarithmic parameter range.
///
/// Generates logarithmically spaced values from start to end.
/// Useful for parameters spanning multiple orders of magnitude.
///
/// Args:
///     start: Starting value (must be > 0)
///     end: Ending value (must be > 0)
///     steps: Number of values to generate
///
/// Returns:
///     ParameterRange for use in sensitivity analysis.
///
/// Example:
///     >>> rate_range = mt.log_range(0.001, 0.1, 3)
///     >>> # Generates: [0.001, 0.01, 0.1]
#[pyfunction]
#[pyo3(signature = (start, end, steps))]
pub fn log_range(start: f64, end: f64, steps: usize) -> PyParameterRange {
    PyParameterRange {
        inner: ParameterRange::logarithmic(start, end, steps),
    }
}

/// Create a discrete parameter range from explicit values.
///
/// Args:
///     values: List of parameter values to test
///
/// Returns:
///     ParameterRange for use in sensitivity analysis.
///
/// Example:
///     >>> periods = mt.discrete_range([5, 10, 20, 50])
#[pyfunction]
#[pyo3(signature = (values))]
pub fn discrete_range(values: Vec<f64>) -> PyParameterRange {
    PyParameterRange {
        inner: ParameterRange::discrete(values),
    }
}

/// Create a centered parameter range around a base value.
///
/// Args:
///     center: Center value
///     variation: Plus/minus variation amount
///     steps: Number of values to generate
///
/// Returns:
///     ParameterRange for use in sensitivity analysis.
///
/// Example:
///     >>> threshold_range = mt.centered_range(0.5, 0.1, 5)
///     >>> # Generates: [0.4, 0.45, 0.5, 0.55, 0.6]
#[pyfunction]
#[pyo3(signature = (center, variation, steps))]
pub fn centered_range(center: f64, variation: f64, steps: usize) -> PyParameterRange {
    PyParameterRange {
        inner: ParameterRange::centered(center, variation, steps),
    }
}

/// Python-exposed parameter range.
#[pyclass(name = "ParameterRange")]
#[derive(Debug, Clone)]
pub struct PyParameterRange {
    pub inner: ParameterRange,
}

#[pymethods]
impl PyParameterRange {
    /// Get all values in this range.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.values())
    }

    /// Get the number of values in this range.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let values = self.inner.values();
        if values.len() <= 5 {
            format!("ParameterRange({:?})", values)
        } else {
            format!(
                "ParameterRange([{:.3}, {:.3}, ..., {:.3}] len={})",
                values[0],
                values[1],
                values[values.len() - 1],
                values.len()
            )
        }
    }
}

// =============================================================================
// Sensitivity Metric Python Bindings
// =============================================================================

/// Convert string metric name to SensitivityMetric enum.
fn parse_metric(name: &str) -> Result<SensitivityMetric, String> {
    match name.to_lowercase().as_str() {
        "sharpe" | "sharpe_ratio" => Ok(SensitivityMetric::Sharpe),
        "sortino" | "sortino_ratio" => Ok(SensitivityMetric::Sortino),
        "return" | "total_return" => Ok(SensitivityMetric::Return),
        "calmar" | "calmar_ratio" => Ok(SensitivityMetric::Calmar),
        "profit_factor" | "pf" => Ok(SensitivityMetric::ProfitFactor),
        "win_rate" | "winrate" => Ok(SensitivityMetric::WinRate),
        "max_drawdown" | "drawdown" | "dd" => Ok(SensitivityMetric::MaxDrawdown),
        _ => Err(format!(
            "Unknown metric '{}'. Valid options: sharpe, sortino, return, calmar, profit_factor, win_rate, max_drawdown",
            name
        )),
    }
}

// =============================================================================
// Sensitivity Analysis Python Bindings
// =============================================================================

/// Python-exposed cliff detection result.
#[pyclass(name = "Cliff")]
#[derive(Debug, Clone)]
pub struct PyCliff {
    #[pyo3(get)]
    pub parameter: String,
    #[pyo3(get)]
    pub value_before: f64,
    #[pyo3(get)]
    pub value_after: f64,
    #[pyo3(get)]
    pub metric_before: f64,
    #[pyo3(get)]
    pub metric_after: f64,
    #[pyo3(get)]
    pub drop_pct: f64,
}

impl From<&Cliff> for PyCliff {
    fn from(c: &Cliff) -> Self {
        Self {
            parameter: c.parameter.clone(),
            value_before: c.value_before,
            value_after: c.value_after,
            metric_before: c.metric_before,
            metric_after: c.metric_after,
            drop_pct: c.drop_pct,
        }
    }
}

#[pymethods]
impl PyCliff {
    fn __repr__(&self) -> String {
        format!(
            "Cliff(param='{}', drop={:.1}%, {:.3} -> {:.3})",
            self.parameter, self.drop_pct, self.metric_before, self.metric_after
        )
    }
}

/// Python-exposed plateau detection result.
#[pyclass(name = "Plateau")]
#[derive(Debug, Clone)]
pub struct PyPlateau {
    #[pyo3(get)]
    pub parameter: String,
    #[pyo3(get)]
    pub start_value: f64,
    #[pyo3(get)]
    pub end_value: f64,
    #[pyo3(get)]
    pub avg_metric: f64,
    #[pyo3(get)]
    pub std_metric: f64,
}

impl From<&Plateau> for PyPlateau {
    fn from(p: &Plateau) -> Self {
        Self {
            parameter: p.parameter.clone(),
            start_value: p.start_value,
            end_value: p.end_value,
            avg_metric: p.avg_metric,
            std_metric: p.std_metric,
        }
    }
}

#[pymethods]
impl PyPlateau {
    fn __repr__(&self) -> String {
        format!(
            "Plateau(param='{}', range=[{:.3}, {:.3}], avg={:.3})",
            self.parameter, self.start_value, self.end_value, self.avg_metric
        )
    }
}

/// Python-exposed heatmap data for 2D parameter visualization.
#[pyclass(name = "HeatmapData")]
#[derive(Debug, Clone)]
pub struct PyHeatmapData {
    #[pyo3(get)]
    pub x_param: String,
    #[pyo3(get)]
    pub y_param: String,
    inner: HeatmapData,
}

#[pymethods]
impl PyHeatmapData {
    /// Get X-axis values.
    fn x_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.x_values.clone())
    }

    /// Get Y-axis values.
    fn y_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.inner.y_values.clone())
    }

    /// Get the 2D grid of metric values as a numpy array.
    ///
    /// Returns NaN for missing values.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self.inner.values.len();
        let cols = if rows > 0 {
            self.inner.values[0].len()
        } else {
            0
        };

        let flat: Vec<f64> = self
            .inner
            .values
            .iter()
            .flat_map(|row| row.iter().map(|v| v.unwrap_or(f64::NAN)))
            .collect();

        PyArray2::from_vec2_bound(
            py,
            &self
                .inner
                .values
                .iter()
                .map(|row| row.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
                .collect::<Vec<Vec<f64>>>(),
        )
        .unwrap()
    }

    /// Export to CSV format for external visualization tools.
    fn to_csv(&self) -> String {
        self.inner.to_csv()
    }

    /// Find the best value in the heatmap.
    ///
    /// Returns:
    ///     Tuple of (x_value, y_value, metric_value) for the best point,
    ///     or None if the heatmap is empty.
    fn best(&self) -> Option<(f64, f64, f64)> {
        self.inner.best()
    }

    fn __repr__(&self) -> String {
        format!(
            "HeatmapData(x='{}' [{}], y='{}' [{}])",
            self.x_param,
            self.inner.x_values.len(),
            self.y_param,
            self.inner.y_values.len()
        )
    }
}

impl From<HeatmapData> for PyHeatmapData {
    fn from(h: HeatmapData) -> Self {
        Self {
            x_param: h.x_param.clone(),
            y_param: h.y_param.clone(),
            inner: h,
        }
    }
}

/// Python-exposed parameter sensitivity summary.
#[pyclass(name = "SensitivitySummary")]
#[derive(Debug, Clone)]
pub struct PySensitivitySummary {
    #[pyo3(get)]
    pub num_combinations: usize,
    #[pyo3(get)]
    pub metric: String,
    #[pyo3(get)]
    pub mean_metric: f64,
    #[pyo3(get)]
    pub std_metric: f64,
    #[pyo3(get)]
    pub min_metric: f64,
    #[pyo3(get)]
    pub max_metric: f64,
    #[pyo3(get)]
    pub stability_score: f64,
    #[pyo3(get)]
    pub num_cliffs: usize,
    #[pyo3(get)]
    pub num_plateaus: usize,
    #[pyo3(get)]
    pub is_fragile: bool,
}

impl From<&SensitivitySummary> for PySensitivitySummary {
    fn from(s: &SensitivitySummary) -> Self {
        Self {
            num_combinations: s.num_combinations,
            metric: s.metric.display_name().to_string(),
            mean_metric: s.mean_metric,
            std_metric: s.std_metric,
            min_metric: s.min_metric,
            max_metric: s.max_metric,
            stability_score: s.stability_score,
            num_cliffs: s.num_cliffs,
            num_plateaus: s.num_plateaus,
            is_fragile: s.is_fragile,
        }
    }
}

#[pymethods]
impl PySensitivitySummary {
    fn __repr__(&self) -> String {
        format!(
            "SensitivitySummary(combos={}, stability={:.2}, fragile={})",
            self.num_combinations, self.stability_score, self.is_fragile
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Parameter Sensitivity Summary\n\
             =============================\n\
             Combinations tested: {}\n\
             Metric: {}\n\n\
             Metric Statistics:\n\
               Mean:    {:.4}\n\
               Std Dev: {:.4}\n\
               Min:     {:.4}\n\
               Max:     {:.4}\n\n\
             Stability Score: {:.2} (higher is better)\n\
             Detected Cliffs: {}\n\
             Detected Plateaus: {}\n\
             {}",
            self.num_combinations,
            self.metric,
            self.mean_metric,
            self.std_metric,
            self.min_metric,
            self.max_metric,
            self.stability_score,
            self.num_cliffs,
            self.num_plateaus,
            if self.is_fragile {
                "\nWARNING: Strategy appears FRAGILE - sensitive to parameter changes"
            } else {
                "\nStrategy appears ROBUST across parameter space"
            }
        )
    }
}

/// Python-exposed sensitivity analysis results.
#[pyclass(name = "SensitivityResult")]
#[derive(Debug, Clone)]
pub struct PySensitivityResult {
    #[pyo3(get)]
    pub strategy_name: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub num_combinations: usize,
    inner: SensitivityAnalysis,
}

#[pymethods]
impl PySensitivityResult {
    /// Get the best performing parameter set.
    ///
    /// Returns:
    ///     Dictionary of parameter names to values that produced the best metric.
    fn best_params<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.inner.best_params() {
            Some(params) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in params {
                    dict.set_item(k, v)?;
                }
                Ok(Some(dict))
            }
            None => Ok(None),
        }
    }

    /// Get the overall stability score (0-1, higher is more stable).
    ///
    /// A low coefficient of variation means stable performance across parameters.
    fn stability_score(&self) -> f64 {
        self.inner.stability_score()
    }

    /// Get stability score for a specific parameter.
    ///
    /// Args:
    ///     param_name: Name of the parameter to analyze.
    ///
    /// Returns:
    ///     Stability score (0-1) or None if parameter not found.
    fn parameter_stability(&self, param_name: &str) -> Option<f64> {
        self.inner.parameter_stability(param_name)
    }

    /// Check if the strategy is fragile (high sensitivity to parameters).
    ///
    /// Args:
    ///     threshold: Stability threshold below which strategy is fragile (default 0.5).
    ///
    /// Returns:
    ///     True if strategy is fragile.
    #[pyo3(signature = (threshold=0.5))]
    fn is_fragile(&self, threshold: f64) -> bool {
        self.inner.is_fragile(threshold)
    }

    /// Get 2D heatmap data for two parameters.
    ///
    /// Args:
    ///     x_param: Parameter name for X axis.
    ///     y_param: Parameter name for Y axis.
    ///
    /// Returns:
    ///     HeatmapData object or None if parameters not found.
    fn heatmap(&self, x_param: &str, y_param: &str) -> Option<PyHeatmapData> {
        self.inner
            .heatmap(x_param, y_param)
            .map(PyHeatmapData::from)
    }

    /// Get parameter importance ranking.
    ///
    /// Returns:
    ///     List of (parameter_name, importance_score) tuples,
    ///     sorted by importance (most important first).
    fn parameter_importance<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let importance = self.inner.parameter_importance();
        let items: Vec<(String, f64)> = importance;
        PyList::new_bound(py, items)
    }

    /// Get detected cliffs (sharp performance drops).
    fn cliffs(&self) -> Vec<PyCliff> {
        self.inner.cliffs.iter().map(PyCliff::from).collect()
    }

    /// Get detected plateaus (stable performance regions).
    fn plateaus(&self) -> Vec<PyPlateau> {
        self.inner.plateaus.iter().map(PyPlateau::from).collect()
    }

    /// Get summary statistics.
    fn summary(&self) -> PySensitivitySummary {
        PySensitivitySummary::from(&self.inner.summary())
    }

    /// Export results to CSV format.
    fn to_csv(&self) -> String {
        self.inner.to_csv()
    }

    fn __repr__(&self) -> String {
        format!(
            "SensitivityResult(strategy='{}', symbol='{}', combinations={}, stability={:.2})",
            self.strategy_name,
            self.symbol,
            self.num_combinations,
            self.inner.stability_score()
        )
    }

    fn __str__(&self) -> String {
        self.inner.summary().to_string()
    }
}

// =============================================================================
// Cost Sensitivity Analysis Python Bindings
// =============================================================================

/// Python-exposed cost scenario result.
#[pyclass(name = "CostScenario")]
#[derive(Debug, Clone)]
pub struct PyCostScenario {
    #[pyo3(get)]
    pub multiplier: f64,
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub sharpe: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub total_trades: usize,
    #[pyo3(get)]
    pub total_costs: f64,
    #[pyo3(get)]
    pub avg_cost_per_trade: f64,
}

impl From<&CostScenario> for PyCostScenario {
    fn from(s: &CostScenario) -> Self {
        Self {
            multiplier: s.multiplier,
            total_return: s.result.total_return_pct,
            sharpe: s.result.sharpe_ratio,
            max_drawdown: s.result.max_drawdown_pct,
            total_trades: s.result.total_trades,
            total_costs: s.total_costs,
            avg_cost_per_trade: s.avg_cost_per_trade,
        }
    }
}

#[pymethods]
impl PyCostScenario {
    /// Whether this is the zero-cost baseline.
    fn is_zero_cost(&self) -> bool {
        self.multiplier < 0.001
    }

    /// Whether this is the baseline (1x) scenario.
    fn is_baseline(&self) -> bool {
        (self.multiplier - 1.0).abs() < 0.001
    }

    fn __repr__(&self) -> String {
        format!(
            "CostScenario({}x, return={:+.1}%, sharpe={:.2})",
            self.multiplier, self.total_return, self.sharpe
        )
    }
}

/// Python-exposed cost sensitivity analysis results.
#[pyclass(name = "CostSensitivityResult")]
#[derive(Debug, Clone)]
pub struct PyCostSensitivityResult {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub strategy_name: String,
    inner: CostSensitivityAnalysis,
}

#[pymethods]
impl PyCostSensitivityResult {
    /// Get all scenarios.
    fn scenarios(&self) -> Vec<PyCostScenario> {
        self.inner
            .scenarios
            .iter()
            .map(PyCostScenario::from)
            .collect()
    }

    /// Get scenario at specific multiplier.
    fn scenario_at(&self, multiplier: f64) -> Option<PyCostScenario> {
        self.inner.scenario_at(multiplier).map(PyCostScenario::from)
    }

    /// Get baseline (1x) scenario.
    fn baseline(&self) -> Option<PyCostScenario> {
        self.inner.baseline().map(PyCostScenario::from)
    }

    /// Get zero-cost scenario (theoretical upper bound).
    fn zero_cost(&self) -> Option<PyCostScenario> {
        self.inner.zero_cost().map(PyCostScenario::from)
    }

    /// Calculate Sharpe ratio degradation percentage at given multiplier.
    ///
    /// Args:
    ///     multiplier: Cost multiplier to check (e.g., 5.0 for 5x costs).
    ///
    /// Returns:
    ///     Percentage degradation relative to baseline (1x costs).
    fn sharpe_degradation_at(&self, multiplier: f64) -> Option<f64> {
        self.inner.sharpe_degradation_at(multiplier)
    }

    /// Calculate return degradation percentage at given multiplier.
    ///
    /// Args:
    ///     multiplier: Cost multiplier to check.
    ///
    /// Returns:
    ///     Percentage degradation relative to baseline.
    fn return_degradation_at(&self, multiplier: f64) -> Option<f64> {
        self.inner.return_degradation_at(multiplier)
    }

    /// Check if strategy passes robustness threshold at 5x costs.
    ///
    /// Args:
    ///     threshold_sharpe: Minimum acceptable Sharpe at 5x costs (default 0.5).
    ///
    /// Returns:
    ///     True if strategy is robust to cost increases.
    #[pyo3(signature = (threshold_sharpe=0.5))]
    fn is_robust(&self, threshold_sharpe: f64) -> bool {
        self.inner.is_robust(threshold_sharpe)
    }

    /// Calculate cost elasticity (% change in return per % change in costs).
    fn cost_elasticity(&self) -> Option<f64> {
        self.inner.cost_elasticity()
    }

    /// Calculate breakeven cost multiplier (where returns become zero/negative).
    fn breakeven_multiplier(&self) -> Option<f64> {
        self.inner.breakeven_multiplier()
    }

    /// Generate formatted summary report.
    fn report(&self) -> String {
        self.inner.summary_report()
    }

    fn __repr__(&self) -> String {
        let robust = self.inner.is_robust(0.5);
        format!(
            "CostSensitivityResult(strategy='{}', symbol='{}', scenarios={}, robust={})",
            self.strategy_name,
            self.symbol,
            self.inner.scenarios.len(),
            robust
        )
    }

    fn __str__(&self) -> String {
        self.inner.summary_report()
    }
}

// =============================================================================
// Main Analysis Functions
// =============================================================================

/// Run parameter sensitivity analysis on a built-in strategy.
///
/// Tests how strategy performance varies across different parameter values.
/// This helps identify:
/// - Fragile strategies that only work with specific parameters
/// - Robust strategies that perform well across parameter ranges
/// - Cliffs where performance drops sharply
/// - Plateaus where performance is stable
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     strategy: Name of built-in strategy ("sma-crossover", "momentum", "mean-reversion",
///               "rsi", "macd", "breakout")
///     params: Dictionary mapping parameter names to ParameterRange objects
///     metric: Metric to analyze ("sharpe", "sortino", "return", "calmar",
///             "profit_factor", "win_rate", "max_drawdown")
///     commission: Commission rate (default 0.001 = 0.1%)
///     slippage: Slippage rate (default 0.001 = 0.1%)
///     cash: Initial capital (default 100,000)
///     parallel: Run parameter combinations in parallel (default True)
///
/// Returns:
///     SensitivityResult with analysis results.
///
/// Example:
///     >>> data = mt.load_sample("AAPL")
///     >>> result = mt.sensitivity(
///     ...     data,
///     ...     strategy="sma-crossover",
///     ...     params={
///     ...         "fast_period": mt.linear_range(5, 20, 4),
///     ...         "slow_period": mt.linear_range(20, 60, 5),
///     ...     },
///     ...     metric="sharpe"
///     ... )
///     >>> print(result.stability_score())
///     0.72
///     >>> print(result.best_params())
///     {'fast_period': 10.0, 'slow_period': 40.0}
#[pyfunction]
#[pyo3(signature = (
    data,
    strategy,
    params,
    metric="sharpe",
    commission=0.001,
    slippage=0.001,
    cash=100_000.0,
    parallel=true
))]
pub fn sensitivity(
    py: Python<'_>,
    data: PyObject,
    strategy: &str,
    params: &Bound<'_, PyDict>,
    metric: &str,
    commission: f64,
    slippage: f64,
    cash: f64,
    parallel: bool,
) -> PyResult<PySensitivityResult> {
    // Extract bars from data
    let bars = extract_bars(py, &data)?;
    let symbol = "SYMBOL";

    // Parse metric
    let sens_metric =
        parse_metric(metric).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    // Build parameter ranges
    let mut param_ranges: HashMap<String, ParameterRange> = HashMap::new();
    for (key, value) in params.iter() {
        let name: String = key.extract()?;
        let range: PyRef<PyParameterRange> = value.extract()?;
        param_ranges.insert(name, range.inner.clone());
    }

    // Validate strategy and get parameter names
    let valid_strategies = [
        "sma-crossover",
        "momentum",
        "mean-reversion",
        "rsi",
        "macd",
        "breakout",
    ];
    if !valid_strategies.contains(&strategy) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown strategy '{}'. Valid options: {:?}",
            strategy, valid_strategies
        )));
    }

    // Build sensitivity config
    let mut sens_config = SensitivityConfig::new()
        .metric(sens_metric)
        .parallel(parallel);
    for (name, range) in &param_ranges {
        sens_config = sens_config.add_parameter(name, range.clone());
    }

    // Build backtest config
    let mut bt_config = BacktestConfig::default();
    bt_config.initial_capital = cash;
    bt_config.cost_model.commission_pct = commission;
    bt_config.cost_model.slippage_pct = slippage;
    bt_config.show_progress = false;

    // Run analysis based on strategy type
    let analysis = match strategy {
        "sma-crossover" => SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
            let fast = p.get("fast_period").copied().unwrap_or(10.0) as usize;
            let slow = p.get("slow_period").copied().unwrap_or(30.0) as usize;
            SmaCrossover::new(fast, slow)
        }),
        "momentum" => SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
            let period = p.get("period").copied().unwrap_or(20.0) as usize;
            let threshold = p.get("threshold").copied().unwrap_or(0.0);
            Momentum::new(period, threshold)
        }),
        "mean-reversion" => {
            SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
                let period = p.get("period").copied().unwrap_or(20.0) as usize;
                let num_std = p.get("num_std").copied().unwrap_or(2.0);
                MeanReversion::new(period, num_std)
            })
        }
        "rsi" => SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
            let period = p.get("period").copied().unwrap_or(14.0) as usize;
            let oversold = p.get("oversold").copied().unwrap_or(30.0);
            let overbought = p.get("overbought").copied().unwrap_or(70.0);
            RsiStrategy::new(period, oversold, overbought)
        }),
        "macd" => SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
            let fast = p.get("fast_period").copied().unwrap_or(12.0) as usize;
            let slow = p.get("slow_period").copied().unwrap_or(26.0) as usize;
            let signal = p.get("signal_period").copied().unwrap_or(9.0) as usize;
            MacdStrategy::new(fast, slow, signal)
        }),
        "breakout" => SensitivityAnalysis::run(&bt_config, &sens_config, &bars, symbol, |p| {
            let period = p.get("period").copied().unwrap_or(20.0) as usize;
            Breakout::new(period)
        }),
        _ => unreachable!(),
    };

    let analysis = analysis.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Sensitivity analysis failed: {}", e))
    })?;

    Ok(PySensitivityResult {
        strategy_name: analysis.strategy_name.clone(),
        symbol: analysis.symbol.clone(),
        num_combinations: analysis.results.len(),
        inner: analysis,
    })
}

/// Run cost sensitivity analysis to test strategy robustness to transaction costs.
///
/// Tests the same strategy at multiple cost levels (e.g., 1x, 2x, 5x, 10x) to see
/// how performance degrades. A robust strategy should maintain acceptable returns
/// even with significantly higher costs.
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     signal: Signal array (1=long, -1=short, 0=flat), or None for built-in strategy
///     strategy: Name of built-in strategy if signal is None
///     strategy_params: Parameters for built-in strategy
///     multipliers: List of cost multipliers to test (default [0.0, 1.0, 2.0, 5.0, 10.0])
///     commission: Base commission rate (default 0.001 = 0.1%)
///     slippage: Base slippage rate (default 0.001 = 0.1%)
///     cash: Initial capital (default 100,000)
///     include_zero_cost: Include zero-cost scenario (default True)
///
/// Returns:
///     CostSensitivityResult with analysis results.
///
/// Example:
///     >>> data = mt.load_sample("AAPL")
///     >>> signal = (data['close'] > data['close'].mean()).astype(int)
///     >>> result = mt.cost_sensitivity(data, signal)
///     >>> print(result.sharpe_degradation_at(5.0))
///     45.2
///     >>> print(result.is_robust())
///     True
#[pyfunction]
#[pyo3(signature = (
    data,
    signal=None,
    strategy=None,
    strategy_params=None,
    multipliers=None,
    commission=0.001,
    slippage=0.001,
    cash=100_000.0,
    include_zero_cost=true
))]
pub fn cost_sensitivity(
    py: Python<'_>,
    data: PyObject,
    signal: Option<PyReadonlyArray1<f64>>,
    strategy: Option<&str>,
    strategy_params: Option<&Bound<'_, PyDict>>,
    multipliers: Option<Vec<f64>>,
    commission: f64,
    slippage: f64,
    cash: f64,
    include_zero_cost: bool,
) -> PyResult<PyCostSensitivityResult> {
    // Extract bars from data
    let bars = extract_bars(py, &data)?;
    let symbol = "SYMBOL";

    // Build backtest config
    let mut bt_config = BacktestConfig::default();
    bt_config.initial_capital = cash;
    bt_config.cost_model.commission_pct = commission;
    bt_config.cost_model.slippage_pct = slippage;
    bt_config.show_progress = false;

    // Build cost sensitivity config
    let mults = multipliers.unwrap_or_else(|| {
        if include_zero_cost {
            vec![0.0, 1.0, 2.0, 5.0, 10.0]
        } else {
            vec![1.0, 2.0, 5.0, 10.0]
        }
    });
    let cs_config = CostSensitivityConfig {
        multipliers: mults,
        robustness_threshold_5x: Some(0.5),
        include_zero_cost,
    };

    // Determine which strategy to use
    let strategy_name: String;
    let mut boxed_strategy: Box<dyn Strategy> = if let Some(signal_arr) = signal {
        strategy_name = "signal".to_string();
        let signal_vec: Vec<f64> = signal_arr.as_array().to_vec();
        Box::new(SignalStrategy::new(signal_vec))
    } else if let Some(strat_name) = strategy {
        strategy_name = strat_name.to_string();
        create_strategy(strat_name, strategy_params)?
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'signal' array or 'strategy' name.",
        ));
    };

    // Run cost sensitivity analysis
    let analysis =
        run_cost_sensitivity_analysis(&bt_config, &cs_config, &bars, &mut *boxed_strategy, symbol)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Cost sensitivity analysis failed: {}",
                    e
                ))
            })?;

    Ok(PyCostSensitivityResult {
        symbol: analysis.symbol.clone(),
        strategy_name: analysis.strategy_name.clone(),
        inner: analysis,
    })
}

// =============================================================================
// Helper Types
// =============================================================================

/// Simple signal-based strategy for cost sensitivity analysis.
struct SignalStrategy {
    signals: Vec<f64>,
    index: usize,
}

impl SignalStrategy {
    fn new(signals: Vec<f64>) -> Self {
        Self { signals, index: 0 }
    }
}

impl Strategy for SignalStrategy {
    fn name(&self) -> &str {
        "signal"
    }

    fn on_bar(&mut self, _bars: &[Bar]) -> crate::types::Signal {
        if self.index < self.signals.len() {
            let sig = self.signals[self.index];
            self.index += 1;
            crate::types::Signal::from(sig)
        } else {
            crate::types::Signal::Hold
        }
    }
}

/// Create a built-in strategy from name and optional parameters.
fn create_strategy(name: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<Box<dyn Strategy>> {
    let get_param = |d: &Bound<'_, PyDict>, key: &str, default: f64| -> f64 {
        d.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(default)
    };

    match name.to_lowercase().as_str() {
        "sma-crossover" | "sma_crossover" => {
            let (fast, slow) = if let Some(p) = params {
                (
                    get_param(p, "fast_period", 10.0) as usize,
                    get_param(p, "slow_period", 30.0) as usize,
                )
            } else {
                (10, 30)
            };
            Ok(Box::new(SmaCrossover::new(fast, slow)))
        }
        "momentum" => {
            let (period, threshold) = if let Some(p) = params {
                (
                    get_param(p, "period", 20.0) as usize,
                    get_param(p, "threshold", 0.0),
                )
            } else {
                (20, 0.0)
            };
            Ok(Box::new(Momentum::new(period, threshold)))
        }
        "mean-reversion" | "mean_reversion" => {
            let (period, num_std) = if let Some(p) = params {
                (
                    get_param(p, "period", 20.0) as usize,
                    get_param(p, "num_std", 2.0),
                )
            } else {
                (20, 2.0)
            };
            Ok(Box::new(MeanReversion::new(period, num_std)))
        }
        "rsi" => {
            let (period, oversold, overbought) = if let Some(p) = params {
                (
                    get_param(p, "period", 14.0) as usize,
                    get_param(p, "oversold", 30.0),
                    get_param(p, "overbought", 70.0),
                )
            } else {
                (14, 30.0, 70.0)
            };
            Ok(Box::new(RsiStrategy::new(period, oversold, overbought)))
        }
        "macd" => {
            let (fast, slow, signal) = if let Some(p) = params {
                (
                    get_param(p, "fast_period", 12.0) as usize,
                    get_param(p, "slow_period", 26.0) as usize,
                    get_param(p, "signal_period", 9.0) as usize,
                )
            } else {
                (12, 26, 9)
            };
            Ok(Box::new(MacdStrategy::new(fast, slow, signal)))
        }
        "breakout" => {
            let period = if let Some(p) = params {
                get_param(p, "period", 20.0) as usize
            } else {
                20
            };
            Ok(Box::new(Breakout::new(period)))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown strategy '{}'. Valid options: sma-crossover, momentum, mean-reversion, rsi, macd, breakout",
            name
        ))),
    }
}
