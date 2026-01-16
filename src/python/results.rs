//! Python-exposed backtest results.
//!
//! Provides the BacktestResult class with methods for accessing metrics,
//! equity curves, trades, and validation.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs::File;
use uuid::Uuid;

use crate::analytics::{
    rolling_drawdown, rolling_drawdown_windowed, rolling_max_drawdown, rolling_sharpe,
    rolling_volatility, BenchmarkMetrics, PerformanceMetrics,
};
use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::export::{export_walkforward_html, Exporter, PerformanceSummary};
use crate::strategy::{Strategy, StrategyContext};
use crate::types::{Bar, Signal};
use crate::viz::{sparkline, walkforward_fold_chart};
use crate::walkforward::{WalkForwardConfig, WalkForwardResult, WalkForwardWindow, WindowResult};

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

    // Stored data for validate() method
    // These are Option to support legacy results without validation data
    bars: Option<Vec<Bar>>,
    signal: Option<Vec<f64>>,
    backtest_config: Option<BacktestConfig>,

    // Benchmark comparison metrics (optional, only present when benchmark is provided)
    benchmark_metrics: Option<BenchmarkMetrics>,
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

    /// Get the Deflated Sharpe Ratio.
    ///
    /// The Deflated Sharpe Ratio (DSR) adjusts the Sharpe ratio for the
    /// number of trials conducted during strategy development. With more
    /// parameter combinations tested, the probability of finding a high
    /// Sharpe ratio by chance increases. DSR accounts for this multiple
    /// testing bias.
    ///
    /// Returns a value between 0 and the raw Sharpe ratio. Lower values
    /// indicate higher probability that the observed Sharpe is due to
    /// overfitting rather than genuine alpha.
    #[getter]
    fn deflated_sharpe(&self) -> f64 {
        let metrics = PerformanceMetrics::from_result(&self.rust_result);
        metrics.deflated_sharpe_ratio
    }

    /// Get the Probabilistic Sharpe Ratio.
    ///
    /// The Probabilistic Sharpe Ratio (PSR) represents the probability
    /// that the true Sharpe ratio is greater than a benchmark (default 0).
    /// It accounts for the length of the track record, skewness, and
    /// kurtosis of returns.
    ///
    /// Returns a value between 0 and 1:
    /// - > 0.95: High confidence the strategy is genuinely profitable
    /// - 0.80-0.95: Moderate confidence
    /// - < 0.80: Low confidence, results may be due to chance
    #[getter]
    fn psr(&self) -> f64 {
        let metrics = PerformanceMetrics::from_result(&self.rust_result);
        metrics.probabilistic_sharpe_ratio
    }

    /// Get Jensen's alpha (risk-adjusted excess return).
    ///
    /// Alpha represents the strategy's return that is not explained by
    /// exposure to the benchmark. A positive alpha indicates the strategy
    /// outperforms the benchmark on a risk-adjusted basis.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn alpha(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.alpha)
    }

    /// Get portfolio beta (sensitivity to benchmark movements).
    ///
    /// Beta measures how much the strategy's returns move relative to
    /// the benchmark. A beta of 1.0 means the strategy moves 1:1 with
    /// the benchmark. Beta > 1 indicates higher volatility than benchmark.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn beta(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.beta)
    }

    /// Get the benchmark's total return for the backtest period.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn benchmark_return(&self) -> Option<f64> {
        self.benchmark_metrics
            .as_ref()
            .map(|m| m.benchmark_return_pct / 100.0)
    }

    /// Get the excess return (strategy return minus benchmark return).
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn excess_return(&self) -> Option<f64> {
        self.benchmark_metrics
            .as_ref()
            .map(|m| m.excess_return_pct / 100.0)
    }

    /// Get the tracking error (annualized standard deviation of excess returns).
    ///
    /// Tracking error measures how consistently the strategy tracks or
    /// deviates from the benchmark. Lower values indicate the strategy
    /// closely follows the benchmark.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn tracking_error(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.tracking_error)
    }

    /// Get the information ratio (alpha per unit of active risk).
    ///
    /// Information ratio = Alpha / Tracking Error
    /// Higher values indicate better risk-adjusted performance relative
    /// to the benchmark. > 0.5 is considered good, > 1.0 is excellent.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn information_ratio(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.information_ratio)
    }

    /// Get the correlation with the benchmark.
    ///
    /// Returns a value between -1 and 1:
    /// - 1: Perfect positive correlation
    /// - 0: No correlation
    /// - -1: Perfect negative correlation
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn benchmark_correlation(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.correlation)
    }

    /// Get the up-capture ratio.
    ///
    /// Up-capture measures the percentage of benchmark gains captured
    /// when the benchmark is positive. > 100% means outperforming in
    /// up markets, < 100% means lagging in up markets.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn up_capture(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.up_capture)
    }

    /// Get the down-capture ratio.
    ///
    /// Down-capture measures the percentage of benchmark losses captured
    /// when the benchmark is negative. < 100% is good (losing less in
    /// down markets), > 100% means losing more than the benchmark.
    ///
    /// Returns None if no benchmark was provided to backtest().
    #[getter]
    fn down_capture(&self) -> Option<f64> {
        self.benchmark_metrics.as_ref().map(|m| m.down_capture)
    }

    /// Check if benchmark comparison metrics are available.
    #[getter]
    fn has_benchmark(&self) -> bool {
        self.benchmark_metrics.is_some()
    }

    /// Get the annualized volatility of the strategy.
    ///
    /// Volatility is the annualized standard deviation of returns,
    /// measuring the dispersion of returns around the mean.
    /// Higher volatility indicates higher risk.
    ///
    /// Returns a decimal (e.g., 0.156 = 15.6% annualized volatility).
    #[getter]
    fn volatility(&self) -> f64 {
        let metrics = PerformanceMetrics::from_result(&self.rust_result);
        metrics.volatility_annual / 100.0 // Convert from percentage to decimal
    }

    /// Get the maximum drawdown duration in days.
    ///
    /// This is the longest period spent in a drawdown (from peak to
    /// recovery to a new peak). Long drawdown durations can be
    /// psychologically challenging for traders.
    ///
    /// Returns the duration in calendar days.
    #[getter]
    fn max_drawdown_duration(&self) -> i64 {
        let metrics = PerformanceMetrics::from_result(&self.rust_result);
        metrics.max_drawdown_duration_days
    }

    /// Get the average trade duration in days.
    ///
    /// This is the average holding period for closed trades,
    /// indicating the typical time a position is held.
    ///
    /// Returns the duration in calendar days.
    #[getter]
    fn avg_trade_duration(&self) -> f64 {
        let metrics = PerformanceMetrics::from_result(&self.rust_result);
        metrics.avg_holding_period_days
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
        // Add advanced metrics
        let perf_metrics = PerformanceMetrics::from_result(&self.rust_result);
        dict.set_item("deflated_sharpe", perf_metrics.deflated_sharpe_ratio)?;
        dict.set_item("psr", perf_metrics.probabilistic_sharpe_ratio)?;
        // Add new metrics per spec
        dict.set_item("volatility", perf_metrics.volatility_annual / 100.0)?;
        dict.set_item(
            "max_drawdown_duration",
            perf_metrics.max_drawdown_duration_days,
        )?;
        dict.set_item("avg_trade_duration", perf_metrics.avg_holding_period_days)?;
        // Add benchmark metrics if available
        if let Some(ref bm) = self.benchmark_metrics {
            dict.set_item("alpha", bm.alpha)?;
            dict.set_item("beta", bm.beta)?;
            dict.set_item("benchmark_return", bm.benchmark_return_pct / 100.0)?;
            dict.set_item("excess_return", bm.excess_return_pct / 100.0)?;
            dict.set_item("tracking_error", bm.tracking_error)?;
            dict.set_item("information_ratio", bm.information_ratio)?;
            dict.set_item("benchmark_correlation", bm.correlation)?;
            dict.set_item("up_capture", bm.up_capture)?;
            dict.set_item("down_capture", bm.down_capture)?;
        }
        Ok(dict)
    }

    /// Get a formatted summary string.
    fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Backtest Results: {}\n", self.strategy_name));
        s.push_str(&"-".repeat(50));
        s.push('\n');
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

    /// Calculate rolling Sharpe ratio over a sliding window.
    ///
    /// Returns annualized Sharpe ratio for each rolling window. Values before
    /// the window size is reached are NaN.
    ///
    /// Args:
    ///     window: Number of periods for rolling calculation (default: 252 for daily data)
    ///     annualization_factor: Factor to annualize returns (default: 252.0 for daily)
    ///
    /// Returns:
    ///     Numpy array of rolling Sharpe ratios with same length as equity curve.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> rolling = results.rolling_sharpe(window=252)
    #[pyo3(signature = (window=252, annualization_factor=252.0))]
    fn rolling_sharpe<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        annualization_factor: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        // Calculate returns from equity curve
        let returns = self.calculate_returns();
        let rolling = rolling_sharpe(&returns, window, annualization_factor);
        PyArray1::from_vec_bound(py, rolling)
    }

    /// Calculate rolling drawdown from peak equity.
    ///
    /// Returns drawdown as a fraction (negative values) at each point in time.
    /// A value of -0.10 means the equity is 10% below its peak.
    ///
    /// Args:
    ///     window: Optional maximum lookback window for peak (None = all history)
    ///
    /// Returns:
    ///     Numpy array of drawdown values (0 at peaks, negative otherwise).
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> dd = results.rolling_drawdown()
    ///     >>> dd_52week = results.rolling_drawdown(window=252)
    #[pyo3(signature = (window=None))]
    fn rolling_drawdown<'py>(
        &self,
        py: Python<'py>,
        window: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let dd = match window {
            Some(w) => rolling_drawdown_windowed(&self.equity_values, w),
            None => rolling_drawdown(&self.equity_values),
        };
        PyArray1::from_vec_bound(py, dd)
    }

    /// Calculate the worst drawdown within each rolling window.
    ///
    /// Returns the maximum (worst) drawdown observed within each window.
    /// Useful for tracking strategy risk over time.
    ///
    /// Args:
    ///     window: Rolling window size in periods (default: 252 for 1 year of daily data)
    ///
    /// Returns:
    ///     Numpy array of worst drawdown values for each window.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> rolling_max_dd = results.rolling_max_drawdown(window=252)
    #[pyo3(signature = (window=252))]
    fn rolling_max_drawdown<'py>(
        &self,
        py: Python<'py>,
        window: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        let rolling = rolling_max_drawdown(&self.equity_values, window);
        PyArray1::from_vec_bound(py, rolling)
    }

    /// Calculate rolling volatility from equity returns.
    ///
    /// Returns annualized volatility for each rolling window.
    /// Values before the window size is reached are NaN.
    ///
    /// Args:
    ///     window: Number of periods for rolling calculation (default: 21 for monthly)
    ///     annualization_factor: Factor to annualize volatility (default: 252.0)
    ///
    /// Returns:
    ///     Numpy array of annualized volatility values.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> vol = results.rolling_volatility(window=21)
    #[pyo3(signature = (window=21, annualization_factor=252.0))]
    fn rolling_volatility<'py>(
        &self,
        py: Python<'py>,
        window: usize,
        annualization_factor: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let returns = self.calculate_returns();
        let rolling = rolling_volatility(&returns, window, annualization_factor);
        PyArray1::from_vec_bound(py, rolling)
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

    /// Run walk-forward validation on the backtest.
    ///
    /// This is the key method for detecting overfitting. It splits the data
    /// into multiple folds, trains on in-sample data, and tests on out-of-sample
    /// data to measure performance degradation.
    ///
    /// Args:
    ///     folds: Number of walk-forward folds (default: 12)
    ///     train_ratio: Fraction of each window used for in-sample (default: 0.75)
    ///     anchored: Whether to use anchored (expanding) windows (default: True)
    ///
    /// Returns:
    ///     ValidationResult with IS/OOS metrics and verdict.
    ///
    /// Example:
    ///     >>> results = mt.backtest(data, signal)
    ///     >>> validation = results.validate()
    ///     >>> print(validation.verdict)
    ///     'borderline'
    ///     >>> print(validation.oos_degradation)
    ///     0.71
    ///     >>> # With trials parameter for deflated Sharpe adjustment
    ///     >>> validation = results.validate(trials=100)  # 100 parameter combinations tested
    ///     >>> print(validation.deflated_sharpe)
    #[pyo3(signature = (folds=12, train_ratio=0.75, anchored=true, trials=1))]
    fn validate(
        &self,
        folds: usize,
        train_ratio: f64,
        anchored: bool,
        trials: usize,
    ) -> PyResult<PyValidationResult> {
        // Check if we have the required data
        let bars = self.bars.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot validate: backtest was run without storing validation data.\n\
                 This can happen if the result was loaded from a file.\n\n\
                 Quick fix: Use mt.validate(data, signal) instead.",
            )
        })?;

        let signal_vec = self.signal.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Cannot validate: signal data not available.")
        })?;

        let bt_config = self.backtest_config.clone().unwrap_or_default();

        // Validate parameters
        if folds < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need at least 2 folds for walk-forward validation.",
            ));
        }

        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "train_ratio must be between 0 and 1 (exclusive).",
            ));
        }

        // Configure walk-forward
        let wf_config = WalkForwardConfig {
            num_windows: folds,
            in_sample_ratio: train_ratio,
            anchored,
            min_bars_per_window: 50,
        };

        let min_required_bars = wf_config.min_bars_per_window * folds;
        if bars.len() < min_required_bars {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Insufficient data: need at least {} bars for {} folds, got {}.\n\
                 Quick fix: Reduce the number of folds or use more data.",
                min_required_bars,
                folds,
                bars.len()
            )));
        }

        // Run walk-forward validation
        let wf_result = run_walkforward_validation(bars, signal_vec, &bt_config, &wf_config)?;

        Ok(PyValidationResult::from_wf_result_with_trials(
            &wf_result, trials,
        ))
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
    /// Calculate returns from equity curve for rolling calculations.
    fn calculate_returns(&self) -> Vec<f64> {
        if self.equity_values.len() < 2 {
            return vec![0.0; self.equity_values.len()];
        }
        let mut returns = Vec::with_capacity(self.equity_values.len());
        returns.push(0.0); // First return is 0
        for i in 1..self.equity_values.len() {
            let prev = self.equity_values[i - 1];
            let curr = self.equity_values[i];
            let ret = if prev > 0.0 {
                (curr - prev) / prev
            } else {
                0.0
            };
            returns.push(ret);
        }
        returns
    }

    /// Create from a Rust BacktestResult without validation data.
    pub fn from_result(result: &BacktestResult) -> Self {
        Self::from_result_with_data(result, None, None, None)
    }

    /// Create from a loaded PerformanceSummary (from JSON file).
    ///
    /// Note: Results created this way cannot use validate() since
    /// the original data/signal are not available.
    pub fn from_summary(summary: &crate::export::PerformanceSummary) -> Self {
        // Create a minimal result with loaded metrics
        // Note: No equity curve, trades, or validation data available
        Self {
            strategy_name: summary.strategy_name.clone(),
            symbols: summary.symbols.clone(),
            initial_capital: summary.initial_capital,
            final_equity: summary.final_equity,
            total_return: summary.total_return_pct / 100.0,
            cagr: summary.annual_return_pct / 100.0,
            sharpe: summary.sharpe_ratio,
            sortino: summary.sortino_ratio,
            calmar: summary.calmar_ratio,
            max_drawdown: summary.max_drawdown_pct / 100.0,
            win_rate: summary.win_rate / 100.0,
            profit_factor: summary.profit_factor,
            total_trades: summary.total_trades,
            winning_trades: summary.winning_trades,
            losing_trades: summary.losing_trades,
            avg_win: summary.avg_win,
            avg_loss: summary.avg_loss,
            trading_days: summary.trading_days as usize,
            // Empty vectors since we don't have detailed data
            equity_values: Vec::new(),
            equity_timestamps: Vec::new(),
            trades_data: Vec::new(),
            // Create a minimal BacktestResult for internal use
            rust_result: BacktestResult {
                strategy_name: summary.strategy_name.clone(),
                symbols: summary.symbols.clone(),
                config: BacktestConfig::default(),
                start_time: summary.start_time,
                end_time: summary.end_time,
                trading_days: summary.trading_days as usize,
                initial_capital: summary.initial_capital,
                final_equity: summary.final_equity,
                total_return_pct: summary.total_return_pct,
                annual_return_pct: summary.annual_return_pct,
                max_drawdown_pct: summary.max_drawdown_pct,
                sharpe_ratio: summary.sharpe_ratio,
                sortino_ratio: summary.sortino_ratio,
                calmar_ratio: summary.calmar_ratio,
                total_trades: summary.total_trades,
                winning_trades: summary.winning_trades,
                losing_trades: summary.losing_trades,
                win_rate: summary.win_rate,
                avg_win: summary.avg_win,
                avg_loss: summary.avg_loss,
                profit_factor: summary.profit_factor,
                equity_curve: Vec::new(),
                trades: Vec::new(),
                experiment_id: Uuid::new_v4(),
                git_info: None,
                config_hash: String::new(),
                data_checksums: HashMap::new(),
                seed: None,
            },
            // No validation data available for loaded results
            bars: None,
            signal: None,
            backtest_config: None,
            benchmark_metrics: None,
        }
    }

    /// Create from a Rust BacktestResult with validation data.
    pub fn from_result_with_data(
        result: &BacktestResult,
        bars: Option<Vec<Bar>>,
        signal: Option<Vec<f64>>,
        config: Option<BacktestConfig>,
    ) -> Self {
        Self::from_result_with_data_and_benchmark(result, bars, signal, config, None)
    }

    /// Create from a Rust BacktestResult with validation data and optional benchmark metrics.
    pub fn from_result_with_data_and_benchmark(
        result: &BacktestResult,
        bars: Option<Vec<Bar>>,
        signal: Option<Vec<f64>>,
        config: Option<BacktestConfig>,
        benchmark_metrics: Option<BenchmarkMetrics>,
    ) -> Self {
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
            max_drawdown: -result.max_drawdown_pct / 100.0, // Negative per spec
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
            bars,
            signal,
            backtest_config: config,
            benchmark_metrics,
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
    /// Deflated Sharpe ratio adjusted for multiple testing bias.
    /// Uses the `trials` parameter from validate() to account for
    /// the number of parameter combinations tested during strategy development.
    #[pyo3(get)]
    pub deflated_sharpe: f64,
    /// Number of parameter combinations tested (from trials parameter).
    #[pyo3(get)]
    pub trials: usize,

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
        s.push('\n');
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

    /// Check for suspicious validation metrics and return warnings.
    ///
    /// Returns a list of warning messages for:
    /// - OOS/IS degradation < 60% (likely overfit)
    /// - OOS/IS degradation 60-80% (borderline)
    /// - Negative OOS returns
    /// - Low parameter stability
    /// - Insufficient data for robust validation
    ///
    /// Example:
    ///     >>> validation = mt.validate(data, signal)
    ///     >>> for warning in validation.warnings():
    ///     ...     print(f"⚠️ {warning}")
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut warnings = Vec::new();

        // Check OOS/IS degradation per spec (validation-robustness.md line 160, 244)
        if self.oos_degradation < 0.40 {
            warnings.push(format!(
                "OOS/IS ratio of {:.0}% is a red flag - strategy probably won't work live",
                self.oos_degradation * 100.0
            ));
        } else if self.oos_degradation < 0.60 {
            warnings.push(format!(
                "OOS/IS ratio of {:.0}% suggests likely overfitting (threshold: 60%)",
                self.oos_degradation * 100.0
            ));
        } else if self.oos_degradation < 0.80 {
            warnings.push(format!(
                "OOS/IS ratio of {:.0}% is borderline - proceed with caution",
                self.oos_degradation * 100.0
            ));
        }

        // Check for negative OOS returns
        if self.avg_oos_return < 0.0 {
            warnings.push(format!(
                "Average OOS return is negative ({:.1}%) - strategy loses money out-of-sample",
                self.avg_oos_return * 100.0
            ));
        }

        // Check for negative OOS Sharpe
        if self.oos_sharpe < 0.0 {
            warnings.push(format!(
                "OOS Sharpe ratio is negative ({:.2}) - risk-adjusted performance is poor",
                self.oos_sharpe
            ));
        }

        // Check parameter stability (low stability suggests fragile strategy)
        if self.parameter_stability < 0.5 && self.parameter_stability > 0.0 {
            warnings.push(format!(
                "Parameter stability of {:.0}% is low - strategy may be fragile",
                self.parameter_stability * 100.0
            ));
        }

        // Check deflated Sharpe when multiple trials were used
        if self.trials > 1 && self.deflated_sharpe < 0.0 {
            warnings.push(format!(
                "Deflated Sharpe of {:.2} (after {} trials) suggests performance may be spurious",
                self.deflated_sharpe, self.trials
            ));
        }

        // Check for high variance across folds
        let fold_returns: Vec<f64> = self.fold_data.iter().map(|f| f.oos_return).collect();
        if fold_returns.len() > 1 {
            let mean: f64 = fold_returns.iter().sum::<f64>() / fold_returns.len() as f64;
            let variance: f64 = fold_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / fold_returns.len() as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.10 {
                // More than 10% standard deviation
                warnings.push(format!(
                    "High variance across folds (std={:.1}%) - inconsistent performance",
                    std_dev * 100.0
                ));
            }
        }

        Ok(PyList::new_bound(py, warnings))
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

    /// Generate a self-contained HTML report for the validation results.
    ///
    /// The report includes:
    /// - Summary metrics (folds, window type, IS ratio)
    /// - Verdict classification with color coding
    /// - Performance metrics (IS/OOS Sharpe, returns, efficiency)
    /// - Fold-by-fold results table
    /// - Bar chart comparing IS vs OOS performance
    ///
    /// Args:
    ///     path: Path to the output HTML file.
    ///
    /// Example:
    ///     >>> validation = mt.validate(data, signal)
    ///     >>> validation.report("validation_report.html")
    fn report(&self, path: &str) -> PyResult<()> {
        export_walkforward_html(&self.rust_result, path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to generate HTML report: {}", e))
        })?;
        Ok(())
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
    /// Create from a WalkForwardResult with trials parameter for deflated Sharpe.
    pub fn from_wf_result_with_trials(result: &WalkForwardResult, trials: usize) -> Self {
        let is_sharpe = result.avg_is_sharpe;
        let oos_sharpe = result.avg_oos_sharpe;

        let oos_degradation = if is_sharpe != 0.0 {
            oos_sharpe / is_sharpe
        } else {
            0.0
        };

        let verdict = result.verdict().to_string();

        // Calculate deflated Sharpe ratio based on OOS performance
        // Use the combined OOS bars as the number of observations
        let n_observations: usize = result.windows.iter().map(|w| w.window.oos_bars).sum();
        let deflated_sharpe = if trials <= 1 || oos_sharpe.is_nan() || n_observations == 0 {
            oos_sharpe
        } else {
            // Calculate the expected maximum Sharpe ratio under null hypothesis
            // Using Bailey-López de Prado formula
            let n = n_observations as f64;
            let t = trials as f64;

            // Expected maximum of t standard normal random variables
            let gamma = 0.5772156649; // Euler-Mascheroni constant
            let e_max = ((2.0 * t.ln()).sqrt())
                - ((gamma + (4.0 * t.ln()).ln()) / (2.0 * (2.0 * t.ln()).sqrt()));

            // Variance of maximum
            let std_max = (std::f64::consts::PI / 6.0).sqrt() / (2.0 * t.ln()).sqrt();

            // Deflated Sharpe = (SR - E[SR_max]) / Var[SR_max]^0.5
            // But we want a bounded version that still represents the Sharpe ratio
            let deflated = (oos_sharpe - e_max * (1.0 / n.sqrt())) / (1.0 + std_max);
            deflated.max(0.0) // Floor at 0
        };

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
            deflated_sharpe,
            trials,
            fold_data,
            rust_result: result.clone(),
        }
    }

    /// Create from a WalkForwardResult (backwards compatible, trials=1).
    pub fn from_wf_result(result: &WalkForwardResult) -> Self {
        Self::from_wf_result_with_trials(result, 1)
    }
}

// =============================================================================
// Walk-forward validation helpers
// =============================================================================

/// A simple signal-based strategy for walk-forward validation.
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

/// Run a single signal-based backtest.
fn run_signal_backtest(
    bars: &[Bar],
    signal: &[f64],
    config: &BacktestConfig,
) -> Result<BacktestResult, String> {
    let mut engine = Engine::new(config.clone());
    engine.add_data("SYMBOL", bars.to_vec());

    let mut strategy = SignalStrategy::new(signal.to_vec());
    engine
        .run(&mut strategy, "SYMBOL")
        .map_err(|e| e.to_string())
}

/// Internal window structure for walk-forward.
struct WfWindow {
    index: usize,
    is_start_idx: usize,
    is_end_idx: usize,
    oos_start_idx: usize,
    oos_end_idx: usize,
}

/// Calculate window boundaries for walk-forward validation.
fn calculate_windows(bars: &[Bar], config: &WalkForwardConfig) -> Result<Vec<WfWindow>, String> {
    let total_bars = bars.len();
    let window_size = total_bars / config.num_windows;
    let is_size = (window_size as f64 * config.in_sample_ratio) as usize;
    let oos_size = window_size - is_size;

    if oos_size < 10 {
        return Err("Out-of-sample period too small. Reduce folds or increase data.".to_string());
    }

    let mut windows = Vec::with_capacity(config.num_windows);

    for i in 0..config.num_windows {
        let window_start = if config.anchored { 0 } else { i * window_size };

        let window_end = ((i + 1) * window_size).min(total_bars - 1);
        let is_end_idx = if config.anchored {
            (window_end - oos_size).max(window_start)
        } else {
            (window_start + is_size).min(window_end)
        };

        let oos_start_idx = (is_end_idx + 1).min(window_end);
        let oos_end_idx = window_end;

        // Skip if invalid
        if is_end_idx <= window_start || oos_end_idx <= oos_start_idx {
            continue;
        }

        windows.push(WfWindow {
            index: i,
            is_start_idx: window_start,
            is_end_idx,
            oos_start_idx,
            oos_end_idx,
        });
    }

    Ok(windows)
}

/// Run walk-forward validation on bars and signal.
fn run_walkforward_validation(
    bars: &[Bar],
    signal: &[f64],
    bt_config: &BacktestConfig,
    wf_config: &WalkForwardConfig,
) -> PyResult<WalkForwardResult> {
    let windows =
        calculate_windows(bars, wf_config).map_err(pyo3::exceptions::PyValueError::new_err)?;

    if windows.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Could not create any valid walk-forward windows with the given parameters.",
        ));
    }

    let mut window_results = Vec::with_capacity(windows.len());

    for window in &windows {
        // Extract IS data and signal
        let is_bars: Vec<Bar> = bars[window.is_start_idx..=window.is_end_idx].to_vec();
        let is_signal: Vec<f64> = signal[window.is_start_idx..=window.is_end_idx].to_vec();

        // Extract OOS data and signal
        let oos_bars: Vec<Bar> = bars[window.oos_start_idx..=window.oos_end_idx].to_vec();
        let oos_signal: Vec<f64> = signal[window.oos_start_idx..=window.oos_end_idx].to_vec();

        // Run IS backtest
        let is_result = run_signal_backtest(&is_bars, &is_signal, bt_config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "In-sample backtest failed for fold {}: {}",
                window.index + 1,
                e
            ))
        })?;

        // Run OOS backtest
        let oos_result = run_signal_backtest(&oos_bars, &oos_signal, bt_config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Out-of-sample backtest failed for fold {}: {}",
                window.index + 1,
                e
            ))
        })?;

        // Calculate efficiency
        let efficiency = if is_result.total_return_pct.abs() > 0.001 {
            oos_result.total_return_pct / is_result.total_return_pct
        } else {
            0.0
        };

        // Create WalkForwardWindow
        let wf_window = WalkForwardWindow {
            index: window.index,
            is_start: bars[window.is_start_idx].timestamp,
            is_end: bars[window.is_end_idx].timestamp,
            oos_start: bars[window.oos_start_idx].timestamp,
            oos_end: bars[window.oos_end_idx].timestamp,
            is_bars: window.is_end_idx - window.is_start_idx + 1,
            oos_bars: window.oos_end_idx - window.oos_start_idx + 1,
        };

        window_results.push(WindowResult {
            window: wf_window,
            in_sample_result: is_result,
            out_of_sample_result: oos_result,
            efficiency_ratio: efficiency,
            parameter_hash: 0,
        });
    }

    // Build WalkForwardResult
    Ok(build_walkforward_result(wf_config.clone(), window_results))
}

/// Build a WalkForwardResult from window results.
fn build_walkforward_result(
    config: WalkForwardConfig,
    windows: Vec<WindowResult>,
) -> WalkForwardResult {
    let num_windows = windows.len();

    if num_windows == 0 {
        return WalkForwardResult {
            config,
            windows: Vec::new(),
            combined_oos_return: 0.0,
            avg_is_return: 0.0,
            avg_oos_return: 0.0,
            avg_efficiency_ratio: 0.0,
            walk_forward_efficiency: 0.0,
            avg_is_sharpe: 0.0,
            avg_oos_sharpe: 0.0,
            oos_sharpe_threshold_met: false,
            parameter_stability: 0.0,
        };
    }

    let avg_is_sharpe: f64 = windows
        .iter()
        .map(|w| w.in_sample_result.sharpe_ratio)
        .sum::<f64>()
        / num_windows as f64;
    let avg_oos_sharpe: f64 = windows
        .iter()
        .map(|w| w.out_of_sample_result.sharpe_ratio)
        .sum::<f64>()
        / num_windows as f64;
    let avg_is_return: f64 = windows
        .iter()
        .map(|w| w.in_sample_result.total_return_pct)
        .sum::<f64>()
        / num_windows as f64;
    let avg_oos_return: f64 = windows
        .iter()
        .map(|w| w.out_of_sample_result.total_return_pct)
        .sum::<f64>()
        / num_windows as f64;
    let avg_efficiency: f64 =
        windows.iter().map(|w| w.efficiency_ratio).sum::<f64>() / num_windows as f64;

    // Calculate combined OOS return (sum of all OOS returns)
    let combined_oos_return: f64 = windows
        .iter()
        .map(|w| w.out_of_sample_result.total_return_pct)
        .sum();

    // Calculate combined IS return for walk-forward efficiency
    let combined_is_return: f64 = windows
        .iter()
        .map(|w| w.in_sample_result.total_return_pct)
        .sum();

    // Walk-forward efficiency = combined OOS / combined IS (capped at 0-1 range)
    let walk_forward_efficiency = if combined_is_return.abs() > f64::EPSILON {
        (combined_oos_return / combined_is_return).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // OOS Sharpe threshold met if OOS Sharpe >= 0.5 * IS Sharpe (typical threshold)
    let oos_sharpe_threshold_met = avg_oos_sharpe >= 0.5 * avg_is_sharpe;

    // Calculate parameter stability (std dev of OOS Sharpe ratios)
    let mean_oos = avg_oos_sharpe;
    let variance: f64 = windows
        .iter()
        .map(|w| {
            let diff = w.out_of_sample_result.sharpe_ratio - mean_oos;
            diff * diff
        })
        .sum::<f64>()
        / num_windows as f64;
    let parameter_stability = 1.0 / (1.0 + variance.sqrt());

    WalkForwardResult {
        config,
        windows,
        combined_oos_return,
        avg_is_return,
        avg_oos_return,
        avg_efficiency_ratio: avg_efficiency,
        walk_forward_efficiency,
        avg_is_sharpe,
        avg_oos_sharpe,
        oos_sharpe_threshold_met,
        parameter_stability,
    }
}
