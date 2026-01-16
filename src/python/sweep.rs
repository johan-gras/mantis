//! Python bindings for parallel parameter sweep.
//!
//! Provides high-performance parallel parameter sweep using rayon.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;

use crate::engine::{BacktestConfig, Engine};
use crate::portfolio::CostModel;
use crate::strategy::Strategy;
use crate::types::{Bar, ExecutionPrice, Signal};

use super::backtest::extract_bars;
use super::results::PyBacktestResult;

/// Result of a single parameter combination in a sweep.
#[pyclass(name = "SweepResultItem")]
#[derive(Clone)]
pub struct PySweepResultItem {
    /// The parameter combination as a dictionary
    #[pyo3(get)]
    pub params: HashMap<String, f64>,
    /// The backtest result for this combination
    #[pyo3(get)]
    pub result: PyBacktestResult,
}

#[pymethods]
impl PySweepResultItem {
    fn __repr__(&self) -> String {
        format!(
            "SweepResultItem(params={:?}, sharpe={:.4})",
            self.params, self.result.sharpe
        )
    }
}

/// Result of a parallel parameter sweep.
#[pyclass(name = "SweepResult")]
#[derive(Clone)]
pub struct PySweepResult {
    /// All results indexed by parameter combination string
    results: Vec<PySweepResultItem>,
    /// Total number of combinations tested
    #[pyo3(get)]
    pub num_combinations: usize,
    /// Whether parallel execution was used
    #[pyo3(get)]
    pub parallel: bool,
}

#[pymethods]
impl PySweepResult {
    /// Get all results as a list of SweepResultItem objects.
    fn items(&self) -> Vec<PySweepResultItem> {
        self.results.clone()
    }

    /// Get all results as a dictionary mapping param strings to BacktestResult.
    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        for item in &self.results {
            let key = format!("{:?}", item.params);
            dict.set_item(key, item.result.clone().into_py(py))?;
        }
        Ok(dict.into())
    }

    /// Get the best result by a given metric.
    ///
    /// Args:
    ///     metric: Metric to optimize ("sharpe", "sortino", "return", "calmar", "profit_factor")
    ///     maximize: Whether to maximize (default True) or minimize the metric
    ///
    /// Returns:
    ///     The SweepResultItem with the best metric value.
    #[pyo3(signature = (metric="sharpe", maximize=true))]
    fn best(&self, metric: &str, maximize: bool) -> PyResult<Option<PySweepResultItem>> {
        if self.results.is_empty() {
            return Ok(None);
        }

        let best = self
            .results
            .iter()
            .max_by(|a, b| {
                let val_a = get_metric_value_from_result(&a.result, metric);
                let val_b = get_metric_value_from_result(&b.result, metric);
                let cmp = val_a
                    .partial_cmp(&val_b)
                    .unwrap_or(std::cmp::Ordering::Equal);
                if maximize {
                    cmp
                } else {
                    cmp.reverse()
                }
            })
            .cloned();

        Ok(best)
    }

    /// Get the best parameters by a given metric.
    #[pyo3(signature = (metric="sharpe", maximize=true))]
    fn best_params(&self, metric: &str, maximize: bool) -> PyResult<Option<HashMap<String, f64>>> {
        Ok(self.best(metric, maximize)?.map(|item| item.params))
    }

    /// Get results sorted by a metric.
    #[pyo3(signature = (metric="sharpe", descending=true))]
    fn sorted_by(&self, metric: &str, descending: bool) -> Vec<PySweepResultItem> {
        let mut results = self.results.clone();
        results.sort_by(|a, b| {
            let val_a = get_metric_value_from_result(&a.result, metric);
            let val_b = get_metric_value_from_result(&b.result, metric);
            let cmp = val_a
                .partial_cmp(&val_b)
                .unwrap_or(std::cmp::Ordering::Equal);
            if descending {
                cmp.reverse()
            } else {
                cmp
            }
        });
        results
    }

    /// Get top N results by a metric.
    #[pyo3(signature = (n=10, metric="sharpe"))]
    fn top(&self, n: usize, metric: &str) -> Vec<PySweepResultItem> {
        self.sorted_by(metric, true).into_iter().take(n).collect()
    }

    /// Get summary statistics across all parameter combinations.
    fn summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);

        if self.results.is_empty() {
            dict.set_item("num_combinations", 0)?;
            return Ok(dict.into());
        }

        let sharpes: Vec<f64> = self.results.iter().map(|r| r.result.sharpe).collect();
        let returns: Vec<f64> = self.results.iter().map(|r| r.result.total_return).collect();

        dict.set_item("num_combinations", self.num_combinations)?;
        dict.set_item("parallel", self.parallel)?;

        // Sharpe statistics
        dict.set_item(
            "sharpe_mean",
            sharpes.iter().sum::<f64>() / sharpes.len() as f64,
        )?;
        dict.set_item(
            "sharpe_max",
            sharpes.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )?;
        dict.set_item(
            "sharpe_min",
            sharpes.iter().cloned().fold(f64::INFINITY, f64::min),
        )?;

        // Return statistics
        dict.set_item(
            "return_mean",
            returns.iter().sum::<f64>() / returns.len() as f64,
        )?;
        dict.set_item(
            "return_max",
            returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        )?;
        dict.set_item(
            "return_min",
            returns.iter().cloned().fold(f64::INFINITY, f64::min),
        )?;

        // Win rate (positive Sharpe)
        let positive_sharpe = sharpes.iter().filter(|&&s| s > 0.0).count();
        dict.set_item(
            "positive_sharpe_pct",
            positive_sharpe as f64 / sharpes.len() as f64,
        )?;

        Ok(dict.into())
    }

    fn __len__(&self) -> usize {
        self.num_combinations
    }

    fn __repr__(&self) -> String {
        if let Some(best) = self.results.iter().max_by(|a, b| {
            a.result
                .sharpe
                .partial_cmp(&b.result.sharpe)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            format!(
                "SweepResult({} combinations, best_sharpe={:.4}, parallel={})",
                self.num_combinations, best.result.sharpe, self.parallel
            )
        } else {
            format!(
                "SweepResult({} combinations, parallel={})",
                self.num_combinations, self.parallel
            )
        }
    }
}

/// Helper to get a metric value from a PyBacktestResult.
fn get_metric_value_from_result(result: &PyBacktestResult, metric: &str) -> f64 {
    match metric {
        "sharpe" => result.sharpe,
        "sortino" => result.sortino,
        "return" | "total_return" => result.total_return,
        "calmar" => result.calmar,
        "profit_factor" => result.profit_factor,
        "win_rate" => result.win_rate,
        "max_drawdown" => result.max_drawdown,
        "num_trades" | "trades" => result.total_trades as f64,
        "cagr" => result.cagr,
        _ => result.sharpe, // Default to Sharpe
    }
}

/// Build a BacktestConfig from parameters.
#[allow(clippy::too_many_arguments)]
fn build_backtest_config(
    cash: f64,
    commission: f64,
    slippage: f64,
    size: f64,
    allow_short: bool,
    borrow_cost: f64,
    max_position: f64,
    fill_price: &str,
) -> BacktestConfig {
    let mut config = BacktestConfig {
        initial_capital: cash,
        position_size: size,
        allow_short,
        fractional_shares: false, // Default: whole shares (per spec)
        show_progress: false, // Disable progress bar in parallel mode
        ..Default::default()
    };

    // Set cost model
    config.cost_model = CostModel {
        commission_pct: commission,
        slippage_pct: slippage,
        borrow_cost_rate: borrow_cost,
        ..Default::default()
    };

    // Set risk config
    config.risk_config.max_position_size = max_position;

    // Set execution price model
    config.execution_price = match fill_price.to_lowercase().as_str() {
        "open" | "next_open" => ExecutionPrice::Open,
        "close" => ExecutionPrice::Close,
        "vwap" => ExecutionPrice::Vwap,
        "twap" => ExecutionPrice::Twap,
        "midpoint" => ExecutionPrice::Midpoint,
        "random" | "random_in_range" => ExecutionPrice::RandomInRange,
        _ => ExecutionPrice::Open, // Default to open (prevents lookahead)
    };

    config
}

/// Wrapper strategy that uses pre-computed signals from a callable.
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
        "sweep_signal"
    }

    fn on_bar(&mut self, _ctx: &crate::strategy::StrategyContext) -> Signal {
        let sig = if self.index < self.signals.len() {
            self.signals[self.index]
        } else {
            0.0
        };
        self.index += 1;

        // Handle NaN as hold
        if sig.is_nan() {
            return Signal::Hold;
        }

        if sig > 0.0 {
            Signal::Long
        } else if sig < 0.0 {
            Signal::Short
        } else {
            Signal::Exit
        }
    }
}

/// Run a parallel parameter sweep over signal-generating function.
///
/// This runs backtests in parallel across all parameter combinations using rayon.
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     signals_list: List of (params_dict, signal_array) tuples - pre-computed signals for each param combo
///     parallel: Whether to use parallel execution (default True)
///     commission: Commission rate (default 0.001 = 0.1%)
///     slippage: Slippage rate (default 0.001 = 0.1%)
///     size: Position size as fraction of equity (default 0.10 = 10%)
///     cash: Initial capital (default 100,000)
///     stop_loss: Optional stop loss
///     take_profit: Optional take profit
///     allow_short: Whether to allow short positions (default True)
///     borrow_cost: Annual borrow cost rate for shorts (default 0.03 = 3%)
///     max_position: Maximum position size as fraction of equity (default 1.0)
///     fill_price: Execution price model (default "next_open")
///
/// Returns:
///     SweepResult object with all results and helper methods.
///
/// Note: The signals_list is a list of tuples where each tuple contains:
///       - A dictionary of parameter values (e.g., {"threshold": 0.1, "lookback": 20})
///       - A numpy array of signals computed with those parameters
///
/// Example:
///     >>> # In Python wrapper, signal_fn is called for each param combo:
///     >>> signals_list = [(params, signal_fn(**params)) for params in param_combos]
///     >>> result = sweep_raw(data, signals_list, parallel=True)
#[pyfunction]
#[pyo3(signature = (
    data,
    signals_list,
    parallel=true,
    commission=0.001,
    slippage=0.001,
    size=0.10,
    cash=100_000.0,
    stop_loss=None,
    take_profit=None,
    allow_short=true,
    borrow_cost=0.03,
    max_position=1.0,
    fill_price="next_open"
))]
#[allow(clippy::too_many_arguments)]
pub fn sweep(
    py: Python<'_>,
    data: PyObject,
    signals_list: &Bound<'_, PyList>,
    parallel: bool,
    commission: f64,
    slippage: f64,
    size: f64,
    cash: f64,
    #[allow(unused_variables)] stop_loss: Option<PyObject>, // TODO: Add stop/take profit support if needed
    #[allow(unused_variables)] take_profit: Option<PyObject>, // For now, these are typically in the signal generation
    allow_short: bool,
    borrow_cost: f64,
    max_position: f64,
    fill_price: &str,
) -> PyResult<PySweepResult> {
    // Extract bars from data
    let bars = extract_bars(py, &data)?;

    if bars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Data is empty. Cannot run sweep without bars.",
        ));
    }

    // Build base config directly (avoiding private PyBacktestConfig::new)
    let base_config = build_backtest_config(
        cash,
        commission,
        slippage,
        size,
        allow_short,
        borrow_cost,
        max_position,
        fill_price,
    );

    // Extract all param/signal pairs from Python
    let mut sweep_items: Vec<(HashMap<String, f64>, Vec<f64>)> = Vec::new();

    for item in signals_list.iter() {
        // Each item should be a tuple (params_dict, signal_array)
        let tuple = item.downcast::<pyo3::types::PyTuple>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "signals_list items must be tuples of (params_dict, signal_array)",
            )
        })?;

        if tuple.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Each item in signals_list must be a tuple of (params_dict, signal_array)",
            ));
        }

        // Extract params dictionary
        let params_obj = tuple.get_item(0)?;
        let params_dict = params_obj.downcast::<PyDict>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "First element of tuple must be a dictionary of parameters",
            )
        })?;

        let mut params: HashMap<String, f64> = HashMap::new();
        for (key, value) in params_dict.iter() {
            let k: String = key.extract()?;
            let v: f64 = value.extract().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "Parameter '{}' must be a numeric value",
                    k
                ))
            })?;
            params.insert(k, v);
        }

        // Extract signal array
        let signal_obj = tuple.get_item(1)?;
        let signal: Vec<f64> = signal_obj.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "Second element of tuple must be a signal array (list or numpy array)",
            )
        })?;

        // Validate signal length
        if signal.len() != bars.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Signal for params {:?} has {} rows, data has {}",
                params,
                signal.len(),
                bars.len()
            )));
        }

        sweep_items.push((params, signal));
    }

    let num_combinations = sweep_items.len();

    if num_combinations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "signals_list is empty. Need at least one parameter combination.",
        ));
    }

    // Run backtests - parallel or sequential
    let results: Vec<PySweepResultItem> = if parallel && num_combinations > 1 {
        // Release the GIL for parallel execution
        py.allow_threads(|| {
            sweep_items
                .par_iter()
                .filter_map(|(params, signal)| {
                    run_single_backtest(&bars, signal, &base_config, params.clone()).ok()
                })
                .collect()
        })
    } else {
        // Sequential execution
        sweep_items
            .iter()
            .filter_map(|(params, signal)| {
                run_single_backtest(&bars, signal, &base_config, params.clone()).ok()
            })
            .collect()
    };

    Ok(PySweepResult {
        results,
        num_combinations,
        parallel: parallel && num_combinations > 1,
    })
}

/// Run a single backtest with the given signal and config.
fn run_single_backtest(
    bars: &[Bar],
    signal: &[f64],
    config: &BacktestConfig,
    params: HashMap<String, f64>,
) -> Result<PySweepResultItem, String> {
    let mut engine = Engine::new(config.clone());
    engine.add_data("SYMBOL", bars.to_vec());

    let mut strategy = SignalStrategy::new(signal.to_vec());
    let result = engine
        .run(&mut strategy, "SYMBOL")
        .map_err(|e| e.to_string())?;

    Ok(PySweepResultItem {
        params,
        result: PyBacktestResult::from_result(&result),
    })
}
