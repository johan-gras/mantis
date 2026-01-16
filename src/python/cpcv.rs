//! Python bindings for Combinatorial Purged Cross-Validation (CPCV).
//!
//! This module exposes CPCV functionality to Python users for robust
//! cross-validation of trading strategies that preserves time-series
//! properties and prevents information leakage.

use crate::cpcv::{CPCVAnalyzer, CPCVConfig, CPCVFold, CPCVMetric, CPCVResult, FoldResult};
use crate::engine::BacktestConfig;
use crate::strategies::{
    BreakoutStrategy, MacdStrategy, MeanReversion, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use crate::strategy::{Strategy, StrategyContext};
use crate::types::Signal;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use super::backtest::extract_bars;
use super::results::PyBacktestResult;

// =============================================================================
// CPCV Configuration Python Bindings
// =============================================================================

/// Configuration for Combinatorial Purged Cross-Validation.
///
/// CPCV is a rigorous cross-validation method for time-series data that:
/// - Splits data into combinatorial folds to maximize data usage
/// - Purges overlapping observations between train and test sets
/// - Enforces embargo periods to prevent information leakage
///
/// Args:
///     n_splits: Number of folds to split data into (default 5)
///     n_test_splits: Number of test folds per combination (default 1)
///     embargo_days: Days of embargo between train and test (default 5)
///     purge_overlapping: Whether to purge overlapping observations (default True)
///     min_bars_per_fold: Minimum bars required per fold (default 50)
///
/// Example:
///     >>> config = mt.CPCVConfig(n_splits=5, embargo_days=10)
///     >>> result = mt.cpcv(data, strategy="sma-crossover", config=config)
#[pyclass(name = "CPCVConfig")]
#[derive(Debug, Clone)]
pub struct PyCPCVConfig {
    #[pyo3(get, set)]
    pub n_splits: usize,
    #[pyo3(get, set)]
    pub n_test_splits: usize,
    #[pyo3(get, set)]
    pub embargo_days: usize,
    #[pyo3(get, set)]
    pub purge_overlapping: bool,
    #[pyo3(get, set)]
    pub min_bars_per_fold: usize,
}

#[pymethods]
impl PyCPCVConfig {
    #[new]
    #[pyo3(signature = (n_splits=5, n_test_splits=1, embargo_days=5, purge_overlapping=true, min_bars_per_fold=50))]
    fn new(
        n_splits: usize,
        n_test_splits: usize,
        embargo_days: usize,
        purge_overlapping: bool,
        min_bars_per_fold: usize,
    ) -> PyResult<Self> {
        if n_splits < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_splits must be at least 2",
            ));
        }
        if n_test_splits >= n_splits {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_test_splits must be less than n_splits",
            ));
        }
        if embargo_days == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "embargo_days must be at least 1",
            ));
        }

        Ok(Self {
            n_splits,
            n_test_splits,
            embargo_days,
            purge_overlapping,
            min_bars_per_fold,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "CPCVConfig(n_splits={}, n_test_splits={}, embargo_days={}, purge={})",
            self.n_splits, self.n_test_splits, self.embargo_days, self.purge_overlapping
        )
    }
}

impl From<&PyCPCVConfig> for CPCVConfig {
    fn from(config: &PyCPCVConfig) -> Self {
        CPCVConfig {
            n_splits: config.n_splits,
            n_test_splits: config.n_test_splits,
            embargo_days: config.embargo_days,
            purge_overlapping: config.purge_overlapping,
            min_bars_per_fold: config.min_bars_per_fold,
        }
    }
}

// =============================================================================
// CPCV Fold Python Bindings
// =============================================================================

/// Information about a single CPCV fold.
///
/// Attributes:
///     index: Fold index number
///     train_start: Training period start timestamp (Unix seconds)
///     train_end: Training period end timestamp
///     test_start: Test period start timestamp
///     test_end: Test period end timestamp
///     embargo_end: End of embargo period after test
///     train_bars: Number of training bars (after purging)
///     test_bars: Number of test bars
///     purged_bars: Number of bars removed due to overlap
#[pyclass(name = "CPCVFold")]
#[derive(Debug, Clone)]
pub struct PyCPCVFold {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub train_start: i64,
    #[pyo3(get)]
    pub train_end: i64,
    #[pyo3(get)]
    pub test_start: i64,
    #[pyo3(get)]
    pub test_end: i64,
    #[pyo3(get)]
    pub embargo_end: i64,
    #[pyo3(get)]
    pub train_bars: usize,
    #[pyo3(get)]
    pub test_bars: usize,
    #[pyo3(get)]
    pub purged_bars: usize,
}

impl From<&CPCVFold> for PyCPCVFold {
    fn from(fold: &CPCVFold) -> Self {
        Self {
            index: fold.index,
            train_start: fold.train_start.timestamp(),
            train_end: fold.train_end.timestamp(),
            test_start: fold.test_start.timestamp(),
            test_end: fold.test_end.timestamp(),
            embargo_end: fold.embargo_end.timestamp(),
            train_bars: fold.train_bars,
            test_bars: fold.test_bars,
            purged_bars: fold.purged_bars,
        }
    }
}

#[pymethods]
impl PyCPCVFold {
    fn __repr__(&self) -> String {
        format!(
            "CPCVFold(index={}, train_bars={}, test_bars={}, purged={})",
            self.index, self.train_bars, self.test_bars, self.purged_bars
        )
    }
}

// =============================================================================
// CPCV Fold Result Python Bindings
// =============================================================================

/// Result from a single CPCV fold evaluation.
///
/// Attributes:
///     fold: The CPCVFold configuration for this result
///     test_result: BacktestResult from running strategy on test set
#[pyclass(name = "CPCVFoldResult")]
#[derive(Debug, Clone)]
pub struct PyCPCVFoldResult {
    #[pyo3(get)]
    pub fold: PyCPCVFold,
    #[pyo3(get)]
    pub test_result: PyBacktestResult,
}

impl PyCPCVFoldResult {
    pub fn from_fold_result(result: &FoldResult) -> Self {
        Self {
            fold: PyCPCVFold::from(&result.fold),
            test_result: PyBacktestResult::from_result(&result.test_result),
        }
    }
}

#[pymethods]
impl PyCPCVFoldResult {
    fn __repr__(&self) -> String {
        format!(
            "CPCVFoldResult(fold={}, sharpe={:.2})",
            self.fold.index, self.test_result.sharpe
        )
    }
}

// =============================================================================
// CPCV Result Python Bindings
// =============================================================================

/// Complete CPCV analysis results.
///
/// Provides summary statistics across all folds and methods to assess
/// strategy robustness.
///
/// Attributes:
///     n_folds: Number of folds evaluated
///     n_combinations: Total number of train/test combinations
///     mean_test_score: Mean metric score across all folds
///     std_test_score: Standard deviation of scores (stability indicator)
///     min_test_score: Worst fold score
///     max_test_score: Best fold score
///     metric: Name of the metric used for evaluation
///
/// Example:
///     >>> result = mt.cpcv(data, strategy="sma-crossover")
///     >>> print(f"Mean Sharpe: {result.mean_test_score:.2f}")
///     >>> print(f"Stability (CV): {result.coefficient_of_variation():.2%}")
///     >>> if result.is_robust(0.3):
///     ...     print("Strategy is robust!")
#[pyclass(name = "CPCVResult")]
#[derive(Debug, Clone)]
pub struct PyCPCVResult {
    #[pyo3(get)]
    pub n_folds: usize,
    #[pyo3(get)]
    pub n_combinations: usize,
    #[pyo3(get)]
    pub mean_test_score: f64,
    #[pyo3(get)]
    pub std_test_score: f64,
    #[pyo3(get)]
    pub min_test_score: f64,
    #[pyo3(get)]
    pub max_test_score: f64,
    #[pyo3(get)]
    pub metric: String,

    // Internal storage
    fold_results: Vec<PyCPCVFoldResult>,
    rust_config: CPCVConfig,
}

impl PyCPCVResult {
    pub fn from_result(result: &CPCVResult, metric_name: &str) -> Self {
        let fold_results: Vec<PyCPCVFoldResult> = result
            .folds
            .iter()
            .map(PyCPCVFoldResult::from_fold_result)
            .collect();

        Self {
            n_folds: result.folds.len(),
            n_combinations: result.n_combinations,
            mean_test_score: result.mean_test_score,
            std_test_score: result.std_test_score,
            min_test_score: result.min_test_score,
            max_test_score: result.max_test_score,
            metric: metric_name.to_string(),
            fold_results,
            rust_config: result.config.clone(),
        }
    }
}

#[pymethods]
impl PyCPCVResult {
    /// Get detailed results for each fold.
    ///
    /// Returns:
    ///     List of CPCVFoldResult objects with full backtest details.
    fn fold_details(&self) -> Vec<PyCPCVFoldResult> {
        self.fold_results.clone()
    }

    /// Calculate coefficient of variation (CV = std/mean).
    ///
    /// A lower CV indicates more stable performance across folds.
    /// Typical interpretation:
    /// - CV < 0.15: Very stable
    /// - CV 0.15-0.30: Moderately stable
    /// - CV > 0.30: High variability (potentially overfit)
    ///
    /// Returns:
    ///     Coefficient of variation, or infinity if mean is zero.
    fn coefficient_of_variation(&self) -> f64 {
        if self.mean_test_score.abs() > 1e-9 {
            self.std_test_score / self.mean_test_score.abs()
        } else {
            f64::INFINITY
        }
    }

    /// Check if strategy is robust based on coefficient of variation.
    ///
    /// A robust strategy has:
    /// 1. Positive mean score
    /// 2. CV <= max_cv (default 0.3)
    ///
    /// Args:
    ///     max_cv: Maximum acceptable coefficient of variation (default 0.3)
    ///
    /// Returns:
    ///     True if strategy passes robustness check.
    #[pyo3(signature = (max_cv=0.3))]
    fn is_robust(&self, max_cv: f64) -> bool {
        let cv = self.coefficient_of_variation();
        cv <= max_cv && self.mean_test_score > 0.0
    }

    /// Get formatted summary of CPCV analysis.
    ///
    /// Returns:
    ///     Multi-line string summarizing CPCV results.
    fn summary(&self) -> String {
        let cv = self.coefficient_of_variation();
        let robust_status = if self.is_robust(0.3) {
            "ROBUST"
        } else {
            "POTENTIALLY OVERFIT"
        };

        format!(
            "CPCV Analysis Summary\n\
             =====================\n\
             Configuration:\n\
               Splits: {} (test splits: {})\n\
               Embargo: {} days\n\
               Purging: {}\n\n\
             Results ({}):\n\
               Folds evaluated: {}\n\
               Mean score: {:.4}\n\
               Std score:  {:.4}\n\
               Min score:  {:.4}\n\
               Max score:  {:.4}\n\n\
             Stability:\n\
               CV: {:.2}%\n\
               Status: {}\n\n\
             Interpretation:\n\
               Lower CV = more stable performance across folds\n\
               Target: CV < 30% for robust strategies",
            self.rust_config.n_splits,
            self.rust_config.n_test_splits,
            self.rust_config.embargo_days,
            if self.rust_config.purge_overlapping {
                "enabled"
            } else {
                "disabled"
            },
            self.metric,
            self.n_folds,
            self.mean_test_score,
            self.std_test_score,
            self.min_test_score,
            self.max_test_score,
            cv * 100.0,
            robust_status
        )
    }

    /// Export fold scores to a list for custom analysis.
    ///
    /// Returns:
    ///     List of metric scores, one per fold.
    fn scores(&self) -> Vec<f64> {
        self.fold_results
            .iter()
            .map(|fr| match self.metric.as_str() {
                "sharpe" => fr.test_result.sharpe,
                "sortino" => fr.test_result.sortino,
                "return" => fr.test_result.total_return,
                "calmar" => fr.test_result.calmar,
                "profit_factor" => fr.test_result.profit_factor,
                _ => fr.test_result.sharpe,
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CPCVResult(folds={}, mean={:.3}, std={:.3}, robust={})",
            self.n_folds,
            self.mean_test_score,
            self.std_test_score,
            self.is_robust(0.3)
        )
    }

    fn __str__(&self) -> String {
        self.summary()
    }
}

// =============================================================================
// CPCV Metric Parsing
// =============================================================================

/// Convert string metric name to CPCVMetric enum.
fn parse_cpcv_metric(name: &str) -> Result<CPCVMetric, String> {
    match name.to_lowercase().as_str() {
        "sharpe" | "sharpe_ratio" => Ok(CPCVMetric::Sharpe),
        "sortino" | "sortino_ratio" => Ok(CPCVMetric::Sortino),
        "return" | "total_return" => Ok(CPCVMetric::Return),
        "calmar" | "calmar_ratio" => Ok(CPCVMetric::Calmar),
        "profit_factor" | "pf" => Ok(CPCVMetric::ProfitFactor),
        _ => Err(format!(
            "Unknown metric '{}'. Valid options: sharpe, sortino, return, calmar, profit_factor",
            name
        )),
    }
}

// =============================================================================
// Signal-Based Strategy for CPCV
// =============================================================================

/// Simple signal-based strategy wrapper for CPCV analysis.
struct SignalStrategy {
    signals: Arc<Vec<f64>>,
    index: usize,
}

impl SignalStrategy {
    fn new(signals: Arc<Vec<f64>>) -> Self {
        Self { signals, index: 0 }
    }
}

impl Strategy for SignalStrategy {
    fn name(&self) -> &str {
        "signal"
    }

    fn on_bar(&mut self, _ctx: &StrategyContext) -> Signal {
        if self.index < self.signals.len() {
            let sig = self.signals[self.index];
            self.index += 1;
            if sig > 0.0 {
                Signal::Long
            } else if sig < 0.0 {
                Signal::Short
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }
}

// =============================================================================
// Main CPCV Function
// =============================================================================

/// Run Combinatorial Purged Cross-Validation on a strategy.
///
/// CPCV is a rigorous cross-validation method designed for time-series data that:
/// - Creates combinatorial folds to maximize data utilization
/// - Purges overlapping observations between training and test sets
/// - Enforces embargo periods to prevent information leakage from future to past
///
/// This is essential for validating ML-based or parameterized trading strategies
/// where traditional k-fold CV would introduce lookahead bias.
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     signal: Signal array (1=long, -1=short, 0=flat), or None for built-in strategy
///     strategy: Name of built-in strategy if signal is None
///         ("sma-crossover", "momentum", "mean-reversion", "rsi", "macd", "breakout")
///     strategy_params: Parameters for built-in strategy (e.g., {"fast_period": 10})
///     config: CPCVConfig object (or None for defaults)
///     metric: Metric to evaluate ("sharpe", "sortino", "return", "calmar", "profit_factor")
///     commission: Commission rate (default 0.001 = 0.1%)
///     slippage: Slippage rate (default 0.001 = 0.1%)
///     cash: Initial capital (default 100,000)
///
/// Returns:
///     CPCVResult with cross-validation statistics and fold details.
///
/// Example:
///     >>> # Using built-in strategy
///     >>> data = mt.load_sample("AAPL")
///     >>> result = mt.cpcv(
///     ...     data,
///     ...     strategy="sma-crossover",
///     ...     strategy_params={"fast_period": 10, "slow_period": 30}
///     ... )
///     >>> print(result.summary())
///
///     >>> # Using custom signal array
///     >>> signal = np.where(data['close'] > data['close'].rolling(20).mean(), 1, -1)
///     >>> result = mt.cpcv(data, signal=signal)
///     >>> if result.is_robust():
///     ...     print("Strategy passes CPCV!")
///
/// References:
///     Lopez de Prado, "Advances in Financial Machine Learning", Chapter 7
#[pyfunction]
#[pyo3(signature = (
    data,
    signal=None,
    strategy=None,
    strategy_params=None,
    config=None,
    metric="sharpe",
    commission=0.001,
    slippage=0.001,
    cash=100_000.0
))]
#[allow(clippy::too_many_arguments)]
pub fn cpcv(
    py: Python<'_>,
    data: PyObject,
    signal: Option<PyReadonlyArray1<f64>>,
    strategy: Option<&str>,
    strategy_params: Option<&Bound<'_, PyDict>>,
    config: Option<&PyCPCVConfig>,
    metric: &str,
    commission: f64,
    slippage: f64,
    cash: f64,
) -> PyResult<PyCPCVResult> {
    // Extract bars from data
    let bars = extract_bars(py, &data)?;
    let symbol = "SYMBOL";

    // Parse metric
    let cpcv_metric =
        parse_cpcv_metric(metric).map_err(pyo3::exceptions::PyValueError::new_err)?;

    // Build CPCV config
    let cpcv_config: CPCVConfig = if let Some(cfg) = config {
        CPCVConfig::from(cfg)
    } else {
        CPCVConfig::default()
    };

    // Build backtest config
    let mut bt_config = BacktestConfig {
        initial_capital: cash,
        show_progress: false,
        ..Default::default()
    };
    bt_config.cost_model.commission_pct = commission;
    bt_config.cost_model.slippage_pct = slippage;

    // Create CPCV analyzer
    let analyzer = CPCVAnalyzer::new(cpcv_config, bt_config);

    // Run CPCV based on strategy type
    let result = if let Some(signal_arr) = signal {
        // Signal-based strategy
        let signal_vec: Vec<f64> = signal_arr.as_array().to_vec();
        let signals = Arc::new(signal_vec);

        analyzer
            .run(&bars, symbol, || Box::new(SignalStrategy::new(signals.clone())), cpcv_metric)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("CPCV analysis failed: {}", e))
            })?
    } else if let Some(strat_name) = strategy {
        // Built-in strategy
        run_cpcv_with_builtin_strategy(&analyzer, &bars, symbol, strat_name, strategy_params, cpcv_metric)?
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Must provide either 'signal' array or 'strategy' name.\n\
             Quick fix: Use strategy='sma-crossover' or provide a signal array.",
        ));
    };

    Ok(PyCPCVResult::from_result(&result, metric))
}

/// Run CPCV with a built-in strategy.
fn run_cpcv_with_builtin_strategy(
    analyzer: &CPCVAnalyzer,
    bars: &[crate::types::Bar],
    symbol: &str,
    strategy_name: &str,
    params: Option<&Bound<'_, PyDict>>,
    metric: CPCVMetric,
) -> PyResult<CPCVResult> {
    let get_param = |d: &Bound<'_, PyDict>, key: &str, default: f64| -> f64 {
        d.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(default)
    };

    let result = match strategy_name.to_lowercase().as_str() {
        "sma-crossover" | "sma_crossover" => {
            let (fast, slow) = if let Some(p) = params {
                (
                    get_param(p, "fast_period", 10.0) as usize,
                    get_param(p, "slow_period", 30.0) as usize,
                )
            } else {
                (10, 30)
            };
            analyzer.run(bars, symbol, move || Box::new(SmaCrossover::new(fast, slow)), metric)
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
            analyzer.run(
                bars,
                symbol,
                move || Box::new(MomentumStrategy::new(period, threshold)),
                metric,
            )
        }
        "mean-reversion" | "mean_reversion" => {
            let (period, num_std, entry_std, exit_std) = if let Some(p) = params {
                (
                    get_param(p, "period", 20.0) as usize,
                    get_param(p, "num_std", 2.0),
                    get_param(p, "entry_std", 2.0),
                    get_param(p, "exit_std", 0.5),
                )
            } else {
                (20, 2.0, 2.0, 0.5)
            };
            analyzer.run(
                bars,
                symbol,
                move || Box::new(MeanReversion::new(period, num_std, entry_std, exit_std)),
                metric,
            )
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
            analyzer.run(
                bars,
                symbol,
                move || Box::new(RsiStrategy::new(period, oversold, overbought)),
                metric,
            )
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
            analyzer.run(
                bars,
                symbol,
                move || Box::new(MacdStrategy::new(fast, slow, signal)),
                metric,
            )
        }
        "breakout" => {
            let (entry_period, exit_period) = if let Some(p) = params {
                (
                    get_param(p, "entry_period", 20.0) as usize,
                    get_param(p, "exit_period", 10.0) as usize,
                )
            } else {
                (20, 10)
            };
            analyzer.run(
                bars,
                symbol,
                move || Box::new(BreakoutStrategy::new(entry_period, exit_period)),
                metric,
            )
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown strategy '{}'. Valid options: sma-crossover, momentum, \
                 mean-reversion, rsi, macd, breakout",
                strategy_name
            )))
        }
    };

    result.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("CPCV analysis failed: {}", e))
    })
}
