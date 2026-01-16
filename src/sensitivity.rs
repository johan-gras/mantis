//! Parameter Sensitivity Analysis
//!
//! This module provides tools to analyze how strategy performance varies with parameter changes.
//! It helps identify fragile strategies where small parameter changes cause large performance swings,
//! versus robust strategies that perform consistently across a range of parameter values.
//!
//! # Features
//!
//! - **Parameter Grid Generation**: Create systematic grids of parameter combinations
//! - **Sensitivity Metrics**: Quantify parameter stability and detect fragile strategies
//! - **Heatmap Generation**: 2D visualization-ready data for parameter surface analysis
//! - **Cliff Detection**: Identify parameter values where performance drops sharply
//! - **Plateau Detection**: Find stable regions in parameter space
//!
//! # Example
//!
//! ```ignore
//! use mantis::sensitivity::{SensitivityConfig, SensitivityAnalysis, ParameterRange};
//! use mantis::engine::BacktestConfig;
//! use mantis::strategies::SmaCrossover;
//!
//! let sensitivity_config = SensitivityConfig::new()
//!     .add_parameter("fast_period", ParameterRange::linear(5.0, 20.0, 4))
//!     .add_parameter("slow_period", ParameterRange::linear(20.0, 60.0, 5));
//!
//! // Run analysis
//! let base_config = BacktestConfig::default();
//! let data: Vec<mantis::types::Bar> = vec![/* bars */];
//! let analysis = SensitivityAnalysis::run(
//!     &base_config,
//!     &sensitivity_config,
//!     &data,
//!     "AAPL",
//!     |params| SmaCrossover::new(
//!         *params.get("fast_period").unwrap() as usize,
//!         *params.get("slow_period").unwrap() as usize,
//!     ),
//! ).unwrap();
//!
//! // Check stability
//! println!("Parameter stability: {:.2}", analysis.stability_score());
//! println!("Best parameters: {:?}", analysis.best_params());
//! ```

use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::strategy::Strategy;
use crate::types::Bar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

/// Defines how to generate parameter values for sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    /// Linear range: start, end, num_steps
    Linear { start: f64, end: f64, steps: usize },
    /// Logarithmic range: start, end, num_steps (for parameters spanning orders of magnitude)
    Logarithmic { start: f64, end: f64, steps: usize },
    /// Explicit list of values
    Discrete { values: Vec<f64> },
    /// Range around a center value: center, variation (±), num_steps
    Centered {
        center: f64,
        variation: f64,
        steps: usize,
    },
}

impl ParameterRange {
    /// Create a linear range from start to end with n steps
    pub fn linear(start: f64, end: f64, steps: usize) -> Self {
        Self::Linear { start, end, steps }
    }

    /// Create a linear range from integer values
    pub fn linear_int(start: i64, end: i64, steps: usize) -> Self {
        Self::Linear {
            start: start as f64,
            end: end as f64,
            steps,
        }
    }

    /// Create a logarithmic range from start to end with n steps
    pub fn logarithmic(start: f64, end: f64, steps: usize) -> Self {
        Self::Logarithmic { start, end, steps }
    }

    /// Create a range from explicit values
    pub fn discrete(values: Vec<f64>) -> Self {
        Self::Discrete { values }
    }

    /// Create a range from integer values
    pub fn discrete_int(values: Vec<i64>) -> Self {
        Self::Discrete {
            values: values.into_iter().map(|v| v as f64).collect(),
        }
    }

    /// Create a centered range: center ± variation with n steps
    pub fn centered(center: f64, variation: f64, steps: usize) -> Self {
        Self::Centered {
            center,
            variation,
            steps,
        }
    }

    /// Generate all values in this range
    pub fn values(&self) -> Vec<f64> {
        match self {
            Self::Linear { start, end, steps } => {
                if *steps <= 1 {
                    return vec![*start];
                }
                let step_size = (end - start) / (*steps as f64 - 1.0);
                (0..*steps).map(|i| start + step_size * i as f64).collect()
            }
            Self::Logarithmic { start, end, steps } => {
                if *steps <= 1 {
                    return vec![*start];
                }
                let log_start = start.ln();
                let log_end = end.ln();
                let step_size = (log_end - log_start) / (*steps as f64 - 1.0);
                (0..*steps)
                    .map(|i| (log_start + step_size * i as f64).exp())
                    .collect()
            }
            Self::Discrete { values } => values.clone(),
            Self::Centered {
                center,
                variation,
                steps,
            } => {
                if *steps <= 1 {
                    return vec![*center];
                }
                let start = center - variation;
                let end = center + variation;
                let step_size = (end - start) / (*steps as f64 - 1.0);
                (0..*steps).map(|i| start + step_size * i as f64).collect()
            }
        }
    }

    /// Get the number of values in this range
    pub fn len(&self) -> usize {
        match self {
            Self::Linear { steps, .. } => *steps,
            Self::Logarithmic { steps, .. } => *steps,
            Self::Discrete { values } => values.len(),
            Self::Centered { steps, .. } => *steps,
        }
    }

    /// Check if the range is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Type alias for constraint functions
type ConstraintFn = std::sync::Arc<dyn Fn(&HashMap<String, f64>) -> bool + Send + Sync>;

/// Configuration for parameter sensitivity analysis
#[derive(Serialize, Deserialize)]
pub struct SensitivityConfig {
    /// Parameter names and their ranges
    pub parameters: HashMap<String, ParameterRange>,
    /// Primary metric to analyze (e.g., "sharpe", "return", "calmar")
    pub metric: SensitivityMetric,
    /// Whether to run backtests in parallel
    pub parallel: bool,
    /// Constraint function to filter invalid parameter combinations
    /// (e.g., fast_period must be less than slow_period)
    /// Not serialized - must be set after deserialization if needed
    #[serde(skip)]
    constraint: Option<ConstraintFn>,
}

impl std::fmt::Debug for SensitivityConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SensitivityConfig")
            .field("parameters", &self.parameters)
            .field("metric", &self.metric)
            .field("parallel", &self.parallel)
            .field(
                "constraint",
                &self.constraint.as_ref().map(|_| "<constraint fn>"),
            )
            .finish()
    }
}

impl Clone for SensitivityConfig {
    fn clone(&self) -> Self {
        Self {
            parameters: self.parameters.clone(),
            metric: self.metric,
            parallel: self.parallel,
            constraint: self.constraint.clone(),
        }
    }
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            parameters: HashMap::new(),
            metric: SensitivityMetric::Sharpe,
            parallel: true,
            constraint: None,
        }
    }
}

impl SensitivityConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter range
    pub fn add_parameter(mut self, name: &str, range: ParameterRange) -> Self {
        self.parameters.insert(name.to_string(), range);
        self
    }

    /// Set the metric to analyze
    pub fn metric(mut self, metric: SensitivityMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Enable/disable parallel execution
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Add a constraint function to filter invalid parameter combinations
    pub fn with_constraint<F>(mut self, constraint: F) -> Self
    where
        F: Fn(&HashMap<String, f64>) -> bool + Send + Sync + 'static,
    {
        self.constraint = Some(std::sync::Arc::new(constraint));
        self
    }

    /// Generate all parameter combinations
    pub fn generate_combinations(&self) -> Vec<HashMap<String, f64>> {
        let param_names: Vec<&String> = self.parameters.keys().collect();
        let param_values: Vec<Vec<f64>> = param_names
            .iter()
            .map(|n| self.parameters[*n].values())
            .collect();

        let mut combinations = Vec::new();
        self.generate_combinations_recursive(
            &param_names,
            &param_values,
            0,
            HashMap::new(),
            &mut combinations,
        );

        // Apply constraint if present
        if let Some(ref constraint) = self.constraint {
            combinations.retain(|params| constraint(params));
        }

        combinations
    }

    fn generate_combinations_recursive(
        &self,
        names: &[&String],
        values: &[Vec<f64>],
        depth: usize,
        current: HashMap<String, f64>,
        result: &mut Vec<HashMap<String, f64>>,
    ) {
        if depth == names.len() {
            result.push(current);
            return;
        }

        for &value in &values[depth] {
            let mut next = current.clone();
            next.insert(names[depth].clone(), value);
            self.generate_combinations_recursive(names, values, depth + 1, next, result);
        }
    }

    /// Get total number of combinations to test
    pub fn num_combinations(&self) -> usize {
        self.parameters.values().map(|r| r.len()).product()
    }
}

/// The metric to use for sensitivity analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityMetric {
    /// Sharpe ratio (risk-adjusted returns)
    Sharpe,
    /// Sortino ratio (downside deviation)
    Sortino,
    /// Total return percentage
    Return,
    /// Calmar ratio (return / max drawdown)
    Calmar,
    /// Profit factor (gross profit / gross loss)
    ProfitFactor,
    /// Win rate percentage
    WinRate,
    /// Maximum drawdown (as positive value for comparison)
    MaxDrawdown,
}

impl SensitivityMetric {
    /// Extract metric value from backtest result
    pub fn extract(&self, result: &BacktestResult) -> f64 {
        match self {
            Self::Sharpe => result.sharpe_ratio,
            Self::Sortino => result.sortino_ratio,
            Self::Return => result.total_return_pct,
            Self::Calmar => result.calmar_ratio,
            Self::ProfitFactor => result.profit_factor,
            Self::WinRate => result.win_rate,
            Self::MaxDrawdown => -result.max_drawdown_pct, // Flip sign for comparison
        }
    }

    /// Display name for the metric
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Sharpe => "Sharpe Ratio",
            Self::Sortino => "Sortino Ratio",
            Self::Return => "Total Return %",
            Self::Calmar => "Calmar Ratio",
            Self::ProfitFactor => "Profit Factor",
            Self::WinRate => "Win Rate %",
            Self::MaxDrawdown => "Max Drawdown %",
        }
    }
}

/// Result of a single parameter combination test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterResult {
    /// Parameter values used
    pub params: HashMap<String, f64>,
    /// The metric value
    pub metric_value: f64,
    /// Full backtest result
    pub result: BacktestResult,
}

/// 2D heatmap data for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    /// Name of the X-axis parameter
    pub x_param: String,
    /// X-axis values
    pub x_values: Vec<f64>,
    /// Name of the Y-axis parameter
    pub y_param: String,
    /// Y-axis values
    pub y_values: Vec<f64>,
    /// Metric values as 2D grid [y][x]
    pub values: Vec<Vec<Option<f64>>>,
    /// The metric being displayed
    pub metric: SensitivityMetric,
}

impl HeatmapData {
    /// Export to CSV format for external visualization
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header row with x values
        csv.push_str(&format!("{}/{}", self.y_param, self.x_param));
        for x in &self.x_values {
            csv.push(',');
            csv.push_str(&format!("{:.4}", x));
        }
        csv.push('\n');

        // Data rows
        for (i, y) in self.y_values.iter().enumerate() {
            csv.push_str(&format!("{:.4}", y));
            for val in &self.values[i] {
                csv.push(',');
                match val {
                    Some(v) => csv.push_str(&format!("{:.6}", v)),
                    None => csv.push_str("NaN"),
                }
            }
            csv.push('\n');
        }

        csv
    }

    /// Get value at specific coordinates
    pub fn get(&self, x_idx: usize, y_idx: usize) -> Option<f64> {
        *self.values.get(y_idx)?.get(x_idx)?
    }

    /// Find the best value in the heatmap
    pub fn best(&self) -> Option<(f64, f64, f64)> {
        let mut best: Option<(f64, f64, f64)> = None;
        for (yi, y) in self.y_values.iter().enumerate() {
            for (xi, x) in self.x_values.iter().enumerate() {
                if let Some(val) = self.get(xi, yi) {
                    if best.is_none() || val > best.unwrap().2 {
                        best = Some((*x, *y, val));
                    }
                }
            }
        }
        best
    }
}

/// Cliff detection result - identifies parameters where performance drops sharply
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cliff {
    /// The parameter that has a cliff
    pub parameter: String,
    /// Value before the cliff
    pub value_before: f64,
    /// Value after the cliff
    pub value_after: f64,
    /// Metric value before
    pub metric_before: f64,
    /// Metric value after
    pub metric_after: f64,
    /// Percentage drop
    pub drop_pct: f64,
}

/// Plateau region - area of parameter space with stable performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plateau {
    /// The parameter with a plateau
    pub parameter: String,
    /// Start of the plateau region
    pub start_value: f64,
    /// End of the plateau region
    pub end_value: f64,
    /// Average metric value in the plateau
    pub avg_metric: f64,
    /// Standard deviation of metric in the plateau
    pub std_metric: f64,
}

/// Complete sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// All parameter combination results
    pub results: Vec<ParameterResult>,
    /// Strategy name
    pub strategy_name: String,
    /// Symbol analyzed
    pub symbol: String,
    /// Configuration used
    pub config: SensitivityConfig,
    /// Detected cliffs (sharp performance drops)
    pub cliffs: Vec<Cliff>,
    /// Detected plateaus (stable regions)
    pub plateaus: Vec<Plateau>,
}

impl SensitivityAnalysis {
    /// Run sensitivity analysis with a strategy factory function
    pub fn run<S, F>(
        base_config: &BacktestConfig,
        sensitivity_config: &SensitivityConfig,
        bars: &[Bar],
        symbol: &str,
        strategy_factory: F,
    ) -> Result<Self, Box<dyn Error + Send + Sync>>
    where
        S: Strategy,
        F: Fn(&HashMap<String, f64>) -> S + Send + Sync,
    {
        let combinations = sensitivity_config.generate_combinations();

        let results: Vec<ParameterResult> = if sensitivity_config.parallel {
            combinations
                .par_iter()
                .filter_map(|params| {
                    let mut engine = Engine::new(base_config.clone());
                    engine.add_data(symbol, bars.to_vec());

                    let mut strategy = strategy_factory(params);
                    match engine.run(&mut strategy, symbol) {
                        Ok(result) => {
                            let metric_value = sensitivity_config.metric.extract(&result);
                            Some(ParameterResult {
                                params: params.clone(),
                                metric_value,
                                result,
                            })
                        }
                        Err(_) => None,
                    }
                })
                .collect()
        } else {
            let mut results = Vec::new();
            for params in &combinations {
                let mut engine = Engine::new(base_config.clone());
                engine.add_data(symbol, bars.to_vec());

                let mut strategy = strategy_factory(params);
                if let Ok(result) = engine.run(&mut strategy, symbol) {
                    let metric_value = sensitivity_config.metric.extract(&result);
                    results.push(ParameterResult {
                        params: params.clone(),
                        metric_value,
                        result,
                    });
                }
            }
            results
        };

        let strategy_name = if let Some(first) = results.first() {
            first.result.strategy_name.clone()
        } else {
            "Unknown".to_string()
        };

        let mut analysis = Self {
            results,
            strategy_name,
            symbol: symbol.to_string(),
            config: sensitivity_config.clone(),
            cliffs: Vec::new(),
            plateaus: Vec::new(),
        };

        // Detect cliffs and plateaus
        analysis.detect_cliffs(0.30); // 30% drop threshold
        analysis.detect_plateaus(0.10); // 10% variation threshold

        Ok(analysis)
    }

    /// Get the best performing parameter set
    pub fn best_params(&self) -> Option<&HashMap<String, f64>> {
        self.results
            .iter()
            .max_by(|a, b| a.metric_value.partial_cmp(&b.metric_value).unwrap())
            .map(|r| &r.params)
    }

    /// Get the best result
    pub fn best_result(&self) -> Option<&ParameterResult> {
        self.results
            .iter()
            .max_by(|a, b| a.metric_value.partial_cmp(&b.metric_value).unwrap())
    }

    /// Get the worst result
    pub fn worst_result(&self) -> Option<&ParameterResult> {
        self.results
            .iter()
            .min_by(|a, b| a.metric_value.partial_cmp(&b.metric_value).unwrap())
    }

    /// Calculate overall stability score (0-1, higher is more stable)
    /// A low coefficient of variation means stable performance
    pub fn stability_score(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }

        let values: Vec<f64> = self.results.iter().map(|r| r.metric_value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        if mean.abs() < 1e-10 {
            return 0.0;
        }

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean.abs(); // Coefficient of variation

        // Convert to 0-1 scale: lower CV = higher stability
        // CV of 0 -> score of 1, CV of 1+ -> score approaching 0
        (1.0 - cv.min(1.0)).max(0.0)
    }

    /// Calculate stability score for a specific parameter
    /// Groups results by other parameters and measures variance within each group
    pub fn parameter_stability(&self, param_name: &str) -> Option<f64> {
        if !self.config.parameters.contains_key(param_name) {
            return None;
        }

        let param_values = self.config.parameters[param_name].values();
        if param_values.len() < 2 {
            return Some(1.0); // Perfect stability with only one value
        }

        // For each pair of adjacent parameter values, calculate average metric change
        let mut total_changes = Vec::new();

        for window in param_values.windows(2) {
            let v1 = window[0];
            let v2 = window[1];

            // Get results where this parameter equals v1 or v2
            let results_v1: Vec<_> = self
                .results
                .iter()
                .filter(|r| (r.params.get(param_name).unwrap() - v1).abs() < 1e-10)
                .collect();

            let results_v2: Vec<_> = self
                .results
                .iter()
                .filter(|r| (r.params.get(param_name).unwrap() - v2).abs() < 1e-10)
                .collect();

            // Calculate average metrics
            if !results_v1.is_empty() && !results_v2.is_empty() {
                let avg1 = results_v1.iter().map(|r| r.metric_value).sum::<f64>()
                    / results_v1.len() as f64;
                let avg2 = results_v2.iter().map(|r| r.metric_value).sum::<f64>()
                    / results_v2.len() as f64;

                if avg1.abs() > 1e-10 {
                    let change = ((avg2 - avg1) / avg1).abs();
                    total_changes.push(change);
                }
            }
        }

        if total_changes.is_empty() {
            return Some(1.0);
        }

        let avg_change = total_changes.iter().sum::<f64>() / total_changes.len() as f64;

        // Convert to 0-1 scale
        Some((1.0 - avg_change.min(1.0)).max(0.0))
    }

    /// Detect cliffs - sharp drops in performance between adjacent parameter values
    fn detect_cliffs(&mut self, threshold: f64) {
        for (param_name, range) in &self.config.parameters {
            let values = range.values();
            if values.len() < 2 {
                continue;
            }

            for window in values.windows(2) {
                let v1 = window[0];
                let v2 = window[1];

                // Calculate average metric at each value
                let avg1 = self.average_metric_at(param_name, v1);
                let avg2 = self.average_metric_at(param_name, v2);

                if let (Some(a1), Some(a2)) = (avg1, avg2) {
                    if a1.abs() > 1e-10 {
                        let drop = (a1 - a2) / a1.abs();
                        if drop > threshold {
                            self.cliffs.push(Cliff {
                                parameter: param_name.clone(),
                                value_before: v1,
                                value_after: v2,
                                metric_before: a1,
                                metric_after: a2,
                                drop_pct: drop * 100.0,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Detect plateaus - regions where metric remains relatively stable
    fn detect_plateaus(&mut self, threshold: f64) {
        for (param_name, range) in &self.config.parameters {
            let values = range.values();
            if values.len() < 3 {
                continue;
            }

            let mut plateau_start: Option<usize> = None;

            for i in 0..values.len() {
                let current_avg = self.average_metric_at(param_name, values[i]);

                if i == 0 {
                    plateau_start = Some(0);
                    continue;
                }

                let prev_avg = self.average_metric_at(param_name, values[i - 1]);

                if let (Some(curr), Some(prev)) = (current_avg, prev_avg) {
                    let change = if prev.abs() > 1e-10 {
                        ((curr - prev) / prev).abs()
                    } else {
                        1.0
                    };

                    if change > threshold {
                        // End of plateau
                        if let Some(start_idx) = plateau_start {
                            if i > start_idx + 1 {
                                // At least 3 points in plateau
                                let plateau_values: Vec<f64> = (start_idx..i)
                                    .filter_map(|j| self.average_metric_at(param_name, values[j]))
                                    .collect();

                                if !plateau_values.is_empty() {
                                    let avg = plateau_values.iter().sum::<f64>()
                                        / plateau_values.len() as f64;
                                    let variance = plateau_values
                                        .iter()
                                        .map(|v| (v - avg).powi(2))
                                        .sum::<f64>()
                                        / plateau_values.len() as f64;

                                    self.plateaus.push(Plateau {
                                        parameter: param_name.clone(),
                                        start_value: values[start_idx],
                                        end_value: values[i - 1],
                                        avg_metric: avg,
                                        std_metric: variance.sqrt(),
                                    });
                                }
                            }
                        }
                        plateau_start = Some(i);
                    }
                }
            }

            // Check for plateau extending to the end
            if let Some(start_idx) = plateau_start {
                if values.len() > start_idx + 1 {
                    let plateau_values: Vec<f64> = (start_idx..values.len())
                        .filter_map(|j| self.average_metric_at(param_name, values[j]))
                        .collect();

                    if plateau_values.len() >= 3 {
                        let avg = plateau_values.iter().sum::<f64>() / plateau_values.len() as f64;
                        let variance = plateau_values
                            .iter()
                            .map(|v| (v - avg).powi(2))
                            .sum::<f64>()
                            / plateau_values.len() as f64;

                        self.plateaus.push(Plateau {
                            parameter: param_name.clone(),
                            start_value: values[start_idx],
                            end_value: *values.last().unwrap(),
                            avg_metric: avg,
                            std_metric: variance.sqrt(),
                        });
                    }
                }
            }
        }
    }

    /// Get average metric value at a specific parameter value
    fn average_metric_at(&self, param_name: &str, value: f64) -> Option<f64> {
        let matching: Vec<f64> = self
            .results
            .iter()
            .filter(|r| {
                r.params
                    .get(param_name)
                    .map(|v| (v - value).abs() < 1e-10)
                    .unwrap_or(false)
            })
            .map(|r| r.metric_value)
            .collect();

        if matching.is_empty() {
            None
        } else {
            Some(matching.iter().sum::<f64>() / matching.len() as f64)
        }
    }

    /// Generate 2D heatmap data for two parameters
    pub fn heatmap(&self, x_param: &str, y_param: &str) -> Option<HeatmapData> {
        if !self.config.parameters.contains_key(x_param)
            || !self.config.parameters.contains_key(y_param)
        {
            return None;
        }

        let x_values = self.config.parameters[x_param].values();
        let y_values = self.config.parameters[y_param].values();

        let mut grid: Vec<Vec<Option<f64>>> = vec![vec![None; x_values.len()]; y_values.len()];

        for (yi, y) in y_values.iter().enumerate() {
            for (xi, x) in x_values.iter().enumerate() {
                // Find matching result
                let matching = self.results.iter().find(|r| {
                    let xmatch = r
                        .params
                        .get(x_param)
                        .map(|v| (v - x).abs() < 1e-10)
                        .unwrap_or(false);
                    let ymatch = r
                        .params
                        .get(y_param)
                        .map(|v| (v - y).abs() < 1e-10)
                        .unwrap_or(false);
                    xmatch && ymatch
                });

                if let Some(result) = matching {
                    grid[yi][xi] = Some(result.metric_value);
                }
            }
        }

        Some(HeatmapData {
            x_param: x_param.to_string(),
            x_values,
            y_param: y_param.to_string(),
            y_values,
            values: grid,
            metric: self.config.metric,
        })
    }

    /// Get parameter importance ranking based on stability scores
    pub fn parameter_importance(&self) -> Vec<(String, f64)> {
        let mut importance: Vec<(String, f64)> = self
            .config
            .parameters
            .keys()
            .filter_map(|name| {
                self.parameter_stability(name)
                    .map(|stability| (name.clone(), 1.0 - stability)) // Invert: low stability = high importance
            })
            .collect();

        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }

    /// Check if the strategy is fragile (high sensitivity to parameters)
    pub fn is_fragile(&self, stability_threshold: f64) -> bool {
        self.stability_score() < stability_threshold
    }

    /// Get summary statistics
    pub fn summary(&self) -> SensitivitySummary {
        let metric_values: Vec<f64> = self.results.iter().map(|r| r.metric_value).collect();

        let mean = if metric_values.is_empty() {
            0.0
        } else {
            metric_values.iter().sum::<f64>() / metric_values.len() as f64
        };

        let variance = if metric_values.is_empty() {
            0.0
        } else {
            metric_values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / metric_values.len() as f64
        };

        let min = metric_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = metric_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        SensitivitySummary {
            num_combinations: self.results.len(),
            metric: self.config.metric,
            mean_metric: mean,
            std_metric: variance.sqrt(),
            min_metric: if min.is_finite() { min } else { 0.0 },
            max_metric: if max.is_finite() { max } else { 0.0 },
            stability_score: self.stability_score(),
            num_cliffs: self.cliffs.len(),
            num_plateaus: self.plateaus.len(),
            is_fragile: self.is_fragile(0.5),
        }
    }

    /// Export results to CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        let param_names: Vec<&String> = self.config.parameters.keys().collect();
        for name in &param_names {
            csv.push_str(name);
            csv.push(',');
        }
        csv.push_str("metric_value,sharpe,return_pct,max_dd_pct,trades\n");

        // Data rows
        for result in &self.results {
            for name in &param_names {
                csv.push_str(&format!("{:.6},", result.params.get(*name).unwrap_or(&0.0)));
            }
            csv.push_str(&format!(
                "{:.6},{:.4},{:.4},{:.4},{}\n",
                result.metric_value,
                result.result.sharpe_ratio,
                result.result.total_return_pct,
                result.result.max_drawdown_pct,
                result.result.total_trades,
            ));
        }

        csv
    }
}

/// Summary of sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivitySummary {
    /// Number of parameter combinations tested
    pub num_combinations: usize,
    /// Metric analyzed
    pub metric: SensitivityMetric,
    /// Mean metric value across all combinations
    pub mean_metric: f64,
    /// Standard deviation of metric
    pub std_metric: f64,
    /// Minimum metric value
    pub min_metric: f64,
    /// Maximum metric value
    pub max_metric: f64,
    /// Overall stability score (0-1)
    pub stability_score: f64,
    /// Number of detected cliffs
    pub num_cliffs: usize,
    /// Number of detected plateaus
    pub num_plateaus: usize,
    /// Whether the strategy appears fragile
    pub is_fragile: bool,
}

impl std::fmt::Display for SensitivitySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parameter Sensitivity Summary")?;
        writeln!(f, "=============================")?;
        writeln!(f, "Combinations tested: {}", self.num_combinations)?;
        writeln!(f, "Metric: {}", self.metric.display_name())?;
        writeln!(f)?;
        writeln!(f, "Metric Statistics:")?;
        writeln!(f, "  Mean:    {:.4}", self.mean_metric)?;
        writeln!(f, "  Std Dev: {:.4}", self.std_metric)?;
        writeln!(f, "  Min:     {:.4}", self.min_metric)?;
        writeln!(f, "  Max:     {:.4}", self.max_metric)?;
        writeln!(f)?;
        writeln!(
            f,
            "Stability Score: {:.2} (higher is better)",
            self.stability_score
        )?;
        writeln!(f, "Detected Cliffs: {}", self.num_cliffs)?;
        writeln!(f, "Detected Plateaus: {}", self.num_plateaus)?;
        if self.is_fragile {
            writeln!(
                f,
                "\n⚠️  Strategy appears FRAGILE - sensitive to parameter changes"
            )?;
        } else {
            writeln!(f, "\n✓ Strategy appears ROBUST across parameter space")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BacktestConfig;
    use chrono::Utc;
    use uuid::Uuid;

    /// Create a minimal BacktestResult for testing
    fn make_test_result(sharpe: f64, ret: f64, max_dd: f64) -> BacktestResult {
        BacktestResult {
            strategy_name: "Test".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100_000.0,
            final_equity: 100_000.0 * (1.0 + ret / 100.0),
            total_return_pct: ret,
            annual_return_pct: ret,
            trading_days: 252,
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 60.0,
            avg_win: 500.0,
            avg_loss: -300.0,
            profit_factor: 1.5,
            max_drawdown_pct: max_dd,
            sharpe_ratio: sharpe,
            sortino_ratio: sharpe * 1.2,
            calmar_ratio: ret / max_dd.abs().max(1.0),
            trades: vec![],
            equity_curve: vec![],
            start_time: Utc::now(),
            end_time: Utc::now(),
            experiment_id: Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        }
    }

    #[test]
    fn test_linear_range() {
        let range = ParameterRange::linear(0.0, 10.0, 5);
        let values = range.values();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[4] - 10.0).abs() < 1e-10);
        assert!((values[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_range() {
        let range = ParameterRange::discrete(vec![1.0, 5.0, 10.0, 20.0]);
        let values = range.values();
        assert_eq!(values, vec![1.0, 5.0, 10.0, 20.0]);
    }

    #[test]
    fn test_centered_range() {
        let range = ParameterRange::centered(10.0, 2.0, 5);
        let values = range.values();
        assert_eq!(values.len(), 5);
        assert!((values[0] - 8.0).abs() < 1e-10);
        assert!((values[2] - 10.0).abs() < 1e-10);
        assert!((values[4] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_range() {
        let range = ParameterRange::logarithmic(1.0, 100.0, 3);
        let values = range.values();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 10.0).abs() < 0.01);
        assert!((values[2] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_config_combinations() {
        let config = SensitivityConfig::new()
            .add_parameter("a", ParameterRange::discrete(vec![1.0, 2.0]))
            .add_parameter("b", ParameterRange::discrete(vec![10.0, 20.0, 30.0]));

        let combos = config.generate_combinations();
        assert_eq!(combos.len(), 6); // 2 * 3 = 6
    }

    #[test]
    fn test_config_with_constraint() {
        let config = SensitivityConfig::new()
            .add_parameter("fast", ParameterRange::discrete(vec![5.0, 10.0, 15.0]))
            .add_parameter("slow", ParameterRange::discrete(vec![10.0, 20.0, 30.0]))
            .with_constraint(|params| params.get("fast").unwrap() < params.get("slow").unwrap());

        let combos = config.generate_combinations();
        // Valid: (5,10), (5,20), (5,30), (10,20), (10,30), (15,20), (15,30) = 7
        assert_eq!(combos.len(), 7);
    }

    #[test]
    fn test_heatmap_csv_export() {
        let heatmap = HeatmapData {
            x_param: "fast".to_string(),
            x_values: vec![5.0, 10.0],
            y_param: "slow".to_string(),
            y_values: vec![20.0, 30.0],
            values: vec![vec![Some(1.0), Some(1.2)], vec![Some(0.8), Some(1.1)]],
            metric: SensitivityMetric::Sharpe,
        };

        let csv = heatmap.to_csv();
        assert!(csv.contains("slow/fast"));
        assert!(csv.contains("5.0000"));
        assert!(csv.contains("1.0000"));
    }

    #[test]
    fn test_sensitivity_metric_extract() {
        let result = make_test_result(1.5, 15.0, -10.0);

        assert!((SensitivityMetric::Sharpe.extract(&result) - 1.5).abs() < 1e-10);
        assert!((SensitivityMetric::Return.extract(&result) - 15.0).abs() < 1e-10);
        assert!((SensitivityMetric::MaxDrawdown.extract(&result) - 10.0).abs() < 1e-10);
        // Flipped sign
    }

    #[test]
    fn test_num_combinations() {
        let config = SensitivityConfig::new()
            .add_parameter("a", ParameterRange::linear(0.0, 10.0, 5))
            .add_parameter("b", ParameterRange::linear(0.0, 100.0, 10));

        assert_eq!(config.num_combinations(), 50);
    }

    #[test]
    fn test_empty_analysis_stability() {
        let analysis = SensitivityAnalysis {
            results: vec![],
            strategy_name: "Test".to_string(),
            symbol: "TEST".to_string(),
            config: SensitivityConfig::default(),
            cliffs: vec![],
            plateaus: vec![],
        };

        assert!((analysis.stability_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_stability() {
        // All results have same metric value = perfect stability
        let results = vec![
            ParameterResult {
                params: [("a".to_string(), 1.0)].into_iter().collect(),
                metric_value: 1.0,
                result: make_test_result(1.0, 10.0, -5.0),
            },
            ParameterResult {
                params: [("a".to_string(), 2.0)].into_iter().collect(),
                metric_value: 1.0,
                result: make_test_result(1.0, 10.0, -5.0),
            },
        ];

        let analysis = SensitivityAnalysis {
            results,
            strategy_name: "Test".to_string(),
            symbol: "TEST".to_string(),
            config: SensitivityConfig::new()
                .add_parameter("a", ParameterRange::discrete(vec![1.0, 2.0])),
            cliffs: vec![],
            plateaus: vec![],
        };

        assert!((analysis.stability_score() - 1.0).abs() < 1e-10);
    }
}
