//! Python-exposed backtest results.
//!
//! Provides the BacktestResult class with methods for accessing metrics,
//! equity curves, trades, and validation.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs::File;
use std::io::Write;

use crate::engine::BacktestResult;
use crate::export::{Exporter, PerformanceSummary};
use crate::viz::{sparkline, walkforward_fold_chart};
use crate::walkforward::WalkForwardResult;

use super::types::PyTrade;

/// Python-exposed backtest results.
#[pyclass(name = "BacktestResult")]
#[derive(Debug, Clone)]
pub struct PyBacktestResult {
    // Core metrics
    #[pyo3(get)]
    pub strategy_name: String,
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub initial_capital: f64,
    #[pyo3(get)]
    pub final_equity: f64,
    #[pyo3(get)]
    pub total_return: f64,
    #[pyo3(get)]
    pub cagr: f64,
    #[pyo3(get)]
    pub sharpe: f64,
    #[pyo3(get)]
    pub sortino: f64,
    #[pyo3(get)]
    pub calmar: f64,
    #[pyo3(get)]
    pub max_drawdown: f64,
    #[pyo3(get)]
    pub win_rate: f64,
    #[pyo3(get)]
    pub profit_factor: f64,
    #[pyo3(get)]
    pub total_trades: usize,
    #[pyo3(get)]
    pub winning_trades: usize,
    #[pyo3(get)]
    pub losing_trades: usize,
    #[pyo3(get)]
    pub avg_win: f64,
    #[pyo3(get)]
    pub avg_loss: f64,
    #[pyo3(get)]
    pub trading_days: usize,

    // Internal data for methods
    equity_values: Vec<f64>,
    equity_timestamps: Vec<i64>,
    trades_data: Vec<PyTrade>,

    // Original result for export methods
    rust_result: BacktestResult,
}

#[pymethods]
impl PyBacktestResult {
    /// Get the equity curve as a numpy array.
    #[getter]
    fn equity_curve<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec_bound(py, self.equity_values.clone())
    }

    /// Get equity timestamps as a numpy array.
    #[getter]
    fn equity_timestamps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec_bound(py, self.equity_timestamps.clone())
    }

    /// Get all trades as a list.
    #[getter]
    fn trades(&self) -> Vec<PyTrade> {
        self.trades_data.clone()
    }

    /// Get all metrics as a dictionary.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("total_return", self.total_return)?;
        dict.set_item("cagr", self.cagr)?;
        dict.set_item("sharpe", self.sharpe)?;
        dict.set_item("sortino", self.sortino)?;
        dict.set_item("calmar", self.calmar)?;
        dict.set_item("max_drawdown", self.max_drawdown)?;
        dict.set_item("win_rate", self.win_rate)?;
        dict.set_item("profit_factor", self.profit_factor)?;
        dict.set_item("total_trades", self.total_trades)?;
        dict.set_item("winning_trades", self.winning_trades)?;
        dict.set_item("losing_trades", self.losing_trades)?;
        dict.set_item("avg_win", self.avg_win)?;
        dict.set_item("avg_loss", self.avg_loss)?;
        dict.set_item("initial_capital", self.initial_capital)?;
        dict.set_item("final_equity", self.final_equity)?;
        dict.set_item("trading_days", self.trading_days)?;
        Ok(dict)
    }

    /// Get a formatted summary string.
    fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Backtest Results: {}\n", self.strategy_name));
        s.push_str(&"-".repeat(50));
        s.push_str("\n");
        s.push_str(&format!(
            "Period          {} trading days\n",
            self.trading_days
        ));
        s.push_str(&format!(
            "Total Return    {:+.1}%\n",
            self.total_return * 100.0
        ));
        s.push_str(&format!("Sharpe Ratio    {:.2}\n", self.sharpe));
        s.push_str(&format!(
            "Max Drawdown    {:.1}%\n",
            self.max_drawdown * 100.0
        ));
        s.push_str(&format!(
            "Win Rate        {:.1}%    ({} wins / {} losses)\n",
            self.win_rate * 100.0,
            self.winning_trades,
            self.losing_trades
        ));
        s
    }

    /// Check for suspicious metrics and return warnings.
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut warnings = Vec::new();

        // Check for suspicious Sharpe
        if self.sharpe > 3.0 {
            warnings.push(format!(
                "Sharpe ratio of {:.2} is suspiciously high (verify data and execution)",
                self.sharpe
            ));
        }

        // Check for suspicious win rate
        if self.win_rate > 0.80 {
            warnings.push(format!(
                "Win rate of {:.0}% is unusually high (check for lookahead bias)",
                self.win_rate * 100.0
            ));
        }

        // Check for suspicious drawdown
        if self.max_drawdown > -0.05 && self.total_trades > 10 {
            warnings.push(format!(
                "Max drawdown of {:.1}% seems too low (verify execution logic)",
                self.max_drawdown * 100.0
            ));
        }

        // Check for statistical significance
        if self.total_trades < 30 {
            warnings.push(format!(
                "Only {} trades - limited statistical significance",
                self.total_trades
            ));
        }

        // Check for profit factor
        if self.profit_factor > 5.0 {
            warnings.push(format!(
                "Profit factor of {:.2} is unusually high",
                self.profit_factor
            ));
        }

        Ok(PyList::new_bound(py, warnings))
    }

    /// Display an ASCII sparkline visualization of the equity curve.
    ///
    /// Args:
    ///     width: Width of the sparkline in characters (default: 40)
    ///
    /// Returns:
    ///     A string containing the formatted equity curve visualization.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> print(results.plot())
    #[pyo3(signature = (width=40))]
    fn plot(&self, width: usize) -> String {
        let spark = sparkline(&self.equity_values, width);
        format!(
            "Backtest Results: {}\n[{}] equity curve\nTotal Return: {:+.1}%  |  Sharpe: {:.2}  |  Max DD: {:.1}%",
            self.symbols.join(", "),
            spark,
            self.total_return * 100.0,
            self.sharpe,
            self.max_drawdown * 100.0
        )
    }

    /// Save the backtest results to a JSON file.
    ///
    /// The file contains all metrics, equity curve, and trades.
    ///
    /// Args:
    ///     path: Path to the output JSON file.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> results.save("experiment_042.json")
    fn save(&self, path: &str) -> PyResult<()> {
        let summary = PerformanceSummary::from_result(&self.rust_result);
        let file = File::create(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to create file: {}", e))
        })?;
        serde_json::to_writer_pretty(file, &summary).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to write JSON: {}", e))
        })?;
        Ok(())
    }

    /// Generate a self-contained HTML report.
    ///
    /// The report includes:
    /// - Summary metrics table
    /// - Equity curve chart (SVG)
    /// - Drawdown chart (SVG)
    /// - Trade list with P&L
    ///
    /// Args:
    ///     path: Path to the output HTML file.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> results.report("experiment_042.html")
    fn report(&self, path: &str) -> PyResult<()> {
        let exporter = Exporter::new(self.rust_result.clone());
        exporter.export_report_html(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to generate HTML report: {}", e))
        })?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestResult(return={:+.1}%, sharpe={:.2}, max_dd={:.1}%, trades={})",
            self.total_return * 100.0,
            self.sharpe,
            self.max_drawdown * 100.0,
            self.total_trades
        )
    }

    fn __str__(&self) -> String {
        self.summary()
    }
}

impl PyBacktestResult {
    /// Create from a Rust BacktestResult.
    pub fn from_result(result: &BacktestResult) -> Self {
        let equity_values: Vec<f64> = result.equity_curve.iter().map(|e| e.equity).collect();
        let equity_timestamps: Vec<i64> = result
            .equity_curve
            .iter()
            .map(|e| e.timestamp.timestamp())
            .collect();
        let trades_data: Vec<PyTrade> = result.trades.iter().map(PyTrade::from).collect();

        Self {
            strategy_name: result.strategy_name.clone(),
            symbols: result.symbols.clone(),
            initial_capital: result.initial_capital,
            final_equity: result.final_equity,
            total_return: result.total_return_pct / 100.0,
            cagr: result.annual_return_pct / 100.0,
            sharpe: result.sharpe_ratio,
            sortino: result.sortino_ratio,
            calmar: result.calmar_ratio,
            max_drawdown: result.max_drawdown_pct / 100.0,
            win_rate: result.win_rate / 100.0,
            profit_factor: result.profit_factor,
            total_trades: result.total_trades,
            winning_trades: result.winning_trades,
            losing_trades: result.losing_trades,
            avg_win: result.avg_win,
            avg_loss: result.avg_loss,
            trading_days: result.trading_days,
            equity_values,
            equity_timestamps,
            trades_data,
            rust_result: result.clone(),
        }
    }
}

/// Details for a single fold in walk-forward analysis.
#[pyclass(name = "FoldDetail")]
#[derive(Debug, Clone)]
pub struct PyFoldDetail {
    #[pyo3(get)]
    pub fold: usize,
    #[pyo3(get)]
    pub is_sharpe: f64,
    #[pyo3(get)]
    pub oos_sharpe: f64,
    #[pyo3(get)]
    pub is_return: f64,
    #[pyo3(get)]
    pub oos_return: f64,
    #[pyo3(get)]
    pub efficiency: f64,
    #[pyo3(get)]
    pub is_bars: usize,
    #[pyo3(get)]
    pub oos_bars: usize,
}

#[pymethods]
impl PyFoldDetail {
    fn __repr__(&self) -> String {
        format!(
            "FoldDetail(fold={}, is_sharpe={:.2}, oos_sharpe={:.2}, efficiency={:.0}%)",
            self.fold,
            self.is_sharpe,
            self.oos_sharpe,
            self.efficiency * 100.0
        )
    }
}

/// Python-exposed validation result.
#[pyclass(name = "ValidationResult")]
#[derive(Debug, Clone)]
pub struct PyValidationResult {
    #[pyo3(get)]
    pub folds: usize,
    #[pyo3(get)]
    pub is_sharpe: f64,
    #[pyo3(get)]
    pub oos_sharpe: f64,
    #[pyo3(get)]
    pub oos_degradation: f64,
    #[pyo3(get)]
    pub verdict: String,
    #[pyo3(get)]
    pub avg_is_return: f64,
    #[pyo3(get)]
    pub avg_oos_return: f64,
    #[pyo3(get)]
    pub efficiency_ratio: f64,
    #[pyo3(get)]
    pub parameter_stability: f64,

    // Internal data for fold_details method
    fold_data: Vec<PyFoldDetail>,

    // Original result for visualization
    rust_result: WalkForwardResult,
}

#[pymethods]
impl PyValidationResult {
    /// Get a formatted summary.
    fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Walk-Forward Analysis ({} folds)\n", self.folds));
        s.push_str(&"-".repeat(40));
        s.push_str("\n");
        s.push_str(&format!("In-sample Sharpe:    {:.2}\n", self.is_sharpe));
        s.push_str(&format!(
            "Out-of-sample:       {:.2}  ({:.0}% of IS)\n",
            self.oos_sharpe,
            self.oos_degradation * 100.0
        ));
        s.push_str(&format!("\nVerdict: {}\n", self.verdict));
        s
    }

    /// Get detailed information for each fold.
    ///
    /// Returns a list of FoldDetail objects with per-fold metrics.
    fn fold_details(&self) -> Vec<PyFoldDetail> {
        self.fold_data.clone()
    }

    /// Check if the validation result indicates a robust strategy.
    fn is_robust(&self) -> bool {
        self.verdict == "robust" || self.verdict == "borderline"
    }

    /// Display an ASCII visualization of fold-by-fold performance.
    ///
    /// Shows in-sample vs out-of-sample returns for each fold with
    /// bar chart representation and efficiency metrics.
    ///
    /// Args:
    ///     width: Width of the bar charts (default: 20)
    ///
    /// Returns:
    ///     A string containing the formatted walk-forward visualization.
    ///
    /// Example:
    ///     >>> validation = mt.validate(data, signal)
    ///     >>> print(validation.plot())
    #[pyo3(signature = (width=20))]
    fn plot(&self, width: usize) -> String {
        walkforward_fold_chart(&self.rust_result, width)
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationResult(folds={}, oos_degradation={:.0}%, verdict='{}')",
            self.folds,
            self.oos_degradation * 100.0,
            self.verdict
        )
    }

    fn __str__(&self) -> String {
        self.summary()
    }
}

impl PyValidationResult {
    /// Create from a WalkForwardResult.
    pub fn from_wf_result(result: &WalkForwardResult) -> Self {
        let is_sharpe = result.avg_is_sharpe;
        let oos_sharpe = result.avg_oos_sharpe;

        let oos_degradation = if is_sharpe != 0.0 {
            oos_sharpe / is_sharpe
        } else {
            0.0
        };

        let verdict = result.verdict().to_string();

        // Extract per-fold details
        let fold_data: Vec<PyFoldDetail> = result
            .windows
            .iter()
            .enumerate()
            .map(|(i, w)| PyFoldDetail {
                fold: i + 1,
                is_sharpe: w.in_sample_result.sharpe_ratio,
                oos_sharpe: w.out_of_sample_result.sharpe_ratio,
                is_return: w.in_sample_result.total_return_pct,
                oos_return: w.out_of_sample_result.total_return_pct,
                efficiency: w.efficiency_ratio,
                is_bars: w.window.is_bars,
                oos_bars: w.window.oos_bars,
            })
            .collect();

        Self {
            folds: result.windows.len(),
            is_sharpe,
            oos_sharpe,
            oos_degradation,
            verdict,
            avg_is_return: result.avg_is_return,
            avg_oos_return: result.avg_oos_return,
            efficiency_ratio: result.avg_efficiency_ratio,
            parameter_stability: result.parameter_stability,
            fold_data,
            rust_result: result.clone(),
        }
    }
}
