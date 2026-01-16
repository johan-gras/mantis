//! Python bindings for backtesting.
//!
//! Provides the main `backtest()` function and configuration classes.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::analytics::BenchmarkMetrics;
use crate::data::{atr, load_csv, load_parquet, DataConfig};
use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::portfolio::CostModel;
use crate::risk::{StopLoss, TakeProfit};
use crate::strategies::{
    BreakoutStrategy, MacdStrategy, MeanReversion, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use crate::strategy::Strategy;
use crate::types::{Bar, DataFrequency, ExecutionPrice, Signal};
use crate::walkforward::{WalkForwardConfig, WalkForwardResult, WalkForwardWindow, WindowResult};

use super::results::PyBacktestResult;

/// Backtest configuration exposed to Python.
#[pyclass(name = "BacktestConfig")]
#[derive(Debug, Clone)]
pub struct PyBacktestConfig {
    #[pyo3(get, set)]
    pub initial_capital: f64,
    #[pyo3(get, set)]
    pub commission: f64,
    #[pyo3(get, set)]
    pub slippage: f64,
    #[pyo3(get, set)]
    pub position_size: f64,
    #[pyo3(get, set)]
    pub allow_short: bool,
    #[pyo3(get, set)]
    pub fractional_shares: bool,
    /// Stop loss specification: float (percentage, e.g., 0.05 for 5%)
    /// or string (ATR-based, e.g., "2atr" for 2x ATR)
    pub stop_loss: Option<StopSpec>,
    /// Take profit specification: float (percentage, e.g., 0.10 for 10%)
    /// or string (ATR-based, e.g., "3atr" for 3x ATR)
    pub take_profit: Option<TakeProfitSpec>,
    #[pyo3(get, set)]
    pub borrow_cost: f64,
    #[pyo3(get, set)]
    pub max_position: f64,
    #[pyo3(get, set)]
    pub fill_price: String,
    /// Data frequency override (e.g., "1min", "5min", "1h", "1D").
    /// If None, frequency is auto-detected from bar timestamps.
    pub freq: Option<DataFrequency>,
    /// Whether to use 24/7 trading hours (for crypto markets).
    /// If None, this is auto-detected based on weekend bars.
    pub trading_hours_24: Option<bool>,
    /// Maximum volume participation rate as a fraction (e.g., 0.10 = 10% of bar volume).
    /// Prevents unrealistic fills in illiquid markets. None means no limit (default).
    #[pyo3(get, set)]
    pub max_volume_participation: Option<f64>,
    /// Order type for signal-generated orders: "market" (default) or "limit".
    /// When "limit", uses limit_offset to calculate limit price from close.
    #[pyo3(get, set)]
    pub order_type: String,
    /// Limit order offset as a fraction of close price (e.g., 0.01 for 1%).
    /// For buys: limit_price = close * (1 - limit_offset) (below close)
    /// For sells: limit_price = close * (1 + limit_offset) (above close)
    /// Only used when order_type="limit".
    #[pyo3(get, set)]
    pub limit_offset: f64,
}

/// Specification for stop loss - can be percentage or ATR-based.
#[derive(Debug, Clone)]
pub enum StopSpec {
    /// Percentage stop loss (e.g., 0.05 for 5%)
    Percentage(f64),
    /// ATR-based stop loss (e.g., 2.0 for 2x ATR)
    Atr(f64),
    /// Trailing stop loss percentage
    Trailing(f64),
}

/// Specification for take profit - can be percentage or ATR-based.
#[derive(Debug, Clone)]
pub enum TakeProfitSpec {
    /// Percentage take profit (e.g., 0.10 for 10%)
    Percentage(f64),
    /// ATR-based take profit (e.g., 3.0 for 3x ATR)
    Atr(f64),
    /// Risk-reward ratio (e.g., 2.0 for 2:1 R:R)
    RiskReward(f64),
}

/// Parse stop loss specification from Python object (float or string).
fn parse_stop_loss(py: Python<'_>, obj: &PyObject) -> PyResult<Option<StopSpec>> {
    // Try as None first
    if obj.is_none(py) {
        return Ok(None);
    }

    // Try as float
    if let Ok(pct) = obj.extract::<f64>(py) {
        return Ok(Some(StopSpec::Percentage(pct)));
    }

    // Try as string
    if let Ok(s) = obj.extract::<String>(py) {
        let s_lower = s.to_lowercase().trim().to_string();

        // Check for ATR format: "2atr", "2.5atr", "2ATR", etc.
        if let Some(num_str) = s_lower.strip_suffix("atr") {
            let multiplier: f64 = num_str.trim().parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid ATR stop loss format: '{}'. Use format like '2atr' or '1.5atr'",
                    s
                ))
            })?;
            return Ok(Some(StopSpec::Atr(multiplier)));
        }

        // Check for trailing format: "5%trailing", "trailing5%", "trail5"
        if s_lower.contains("trail") {
            // Extract numeric part
            let num_str: String = s_lower
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !num_str.is_empty() {
                let pct: f64 = num_str.parse().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid trailing stop format: '{}'. Use format like '5trail' or 'trail5'",
                        s
                    ))
                })?;
                // Convert percentage like "5" to decimal 0.05
                let decimal = if pct > 1.0 { pct / 100.0 } else { pct };
                return Ok(Some(StopSpec::Trailing(decimal)));
            }
        }

        // Try as percentage string: "5%", "0.05"
        let num_str = s_lower.trim_end_matches('%');
        if let Ok(pct) = num_str.parse::<f64>() {
            // If > 1.0, assume it's a percentage like "5" meaning 5%
            let decimal = if pct > 1.0 { pct / 100.0 } else { pct };
            return Ok(Some(StopSpec::Percentage(decimal)));
        }

        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid stop loss: '{}'. Use float (0.05 for 5%) or string ('2atr' for 2x ATR)",
            s
        )));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "stop_loss must be a float, string, or None",
    ))
}

/// Parse frequency string to DataFrequency.
fn parse_freq(s: &str) -> PyResult<DataFrequency> {
    let s_lower = s.to_lowercase().trim().to_string();
    match s_lower.as_str() {
        "1s" | "1sec" | "second1" => Ok(DataFrequency::Second1),
        "5s" | "5sec" | "second5" => Ok(DataFrequency::Second5),
        "10s" | "10sec" | "second10" => Ok(DataFrequency::Second10),
        "15s" | "15sec" | "second15" => Ok(DataFrequency::Second15),
        "30s" | "30sec" | "second30" => Ok(DataFrequency::Second30),
        "1min" | "1m" | "minute1" => Ok(DataFrequency::Minute1),
        "5min" | "5m" | "minute5" => Ok(DataFrequency::Minute5),
        "15min" | "15m" | "minute15" => Ok(DataFrequency::Minute15),
        "30min" | "30m" | "minute30" => Ok(DataFrequency::Minute30),
        "1h" | "1hr" | "hour1" | "hourly" => Ok(DataFrequency::Hour1),
        "4h" | "4hr" | "hour4" => Ok(DataFrequency::Hour4),
        "1d" | "day" | "daily" | "d" => Ok(DataFrequency::Day),
        "1w" | "week" | "weekly" | "w" => Ok(DataFrequency::Week),
        "1mo" | "month" | "monthly" | "m" => Ok(DataFrequency::Month),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid frequency: '{}'. Valid options: 1s, 5s, 10s, 15s, 30s, 1min, 5min, 15min, 30min, 1h, 4h, 1d, 1w, 1mo",
            s
        ))),
    }
}

/// Parse take profit specification from Python object (float or string).
fn parse_take_profit(py: Python<'_>, obj: &PyObject) -> PyResult<Option<TakeProfitSpec>> {
    // Try as None first
    if obj.is_none(py) {
        return Ok(None);
    }

    // Try as float
    if let Ok(pct) = obj.extract::<f64>(py) {
        return Ok(Some(TakeProfitSpec::Percentage(pct)));
    }

    // Try as string
    if let Ok(s) = obj.extract::<String>(py) {
        let s_lower = s.to_lowercase().trim().to_string();

        // Check for ATR format: "3atr", "2.5atr", etc.
        if let Some(num_str) = s_lower.strip_suffix("atr") {
            let multiplier: f64 = num_str.trim().parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid ATR take profit format: '{}'. Use format like '3atr' or '2.5atr'",
                    s
                ))
            })?;
            return Ok(Some(TakeProfitSpec::Atr(multiplier)));
        }

        // Check for risk-reward format: "2rr", "2:1", "2R"
        if s_lower.ends_with("rr") || s_lower.ends_with('r') || s_lower.contains(':') {
            // Extract numeric part
            let num_str: String = s_lower
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !num_str.is_empty() {
                let ratio: f64 = num_str.parse().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid risk-reward format: '{}'. Use format like '2rr' or '2:1'",
                        s
                    ))
                })?;
                return Ok(Some(TakeProfitSpec::RiskReward(ratio)));
            }
        }

        // Try as percentage string: "10%", "0.10"
        let num_str = s_lower.trim_end_matches('%');
        if let Ok(pct) = num_str.parse::<f64>() {
            // If > 1.0, assume it's a percentage like "10" meaning 10%
            let decimal = if pct > 1.0 { pct / 100.0 } else { pct };
            return Ok(Some(TakeProfitSpec::Percentage(decimal)));
        }

        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid take profit: '{}'. Use float (0.10 for 10%) or string ('3atr' for 3x ATR)",
            s
        )));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "take_profit must be a float, string, or None",
    ))
}

#[pymethods]
impl PyBacktestConfig {
    #[new]
    #[pyo3(signature = (
        initial_capital=100_000.0,
        commission=0.001,
        slippage=0.001,
        position_size=0.10,
        allow_short=true,
        fractional_shares=false,
        stop_loss=None,
        take_profit=None,
        borrow_cost=0.03,
        max_position=1.0,
        fill_price="next_open",
        freq=None,
        trading_hours_24=None,
        max_volume_participation=None,
        order_type="market",
        limit_offset=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        initial_capital: f64,
        commission: f64,
        slippage: f64,
        position_size: f64,
        allow_short: bool,
        fractional_shares: bool,
        stop_loss: Option<PyObject>,
        take_profit: Option<PyObject>,
        borrow_cost: f64,
        max_position: f64,
        fill_price: &str,
        freq: Option<&str>,
        trading_hours_24: Option<bool>,
        max_volume_participation: Option<f64>,
        order_type: &str,
        limit_offset: f64,
    ) -> PyResult<Self> {
        let sl_spec = if let Some(ref obj) = stop_loss {
            parse_stop_loss(py, obj)?
        } else {
            None
        };
        let tp_spec = if let Some(ref obj) = take_profit {
            parse_take_profit(py, obj)?
        } else {
            None
        };

        let freq_parsed = if let Some(f) = freq {
            Some(parse_freq(f)?)
        } else {
            None
        };

        // Validate order_type
        let order_type_lower = order_type.to_lowercase();
        if order_type_lower != "market" && order_type_lower != "limit" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid order_type: '{}'. Must be 'market' or 'limit'.",
                order_type
            )));
        }

        Ok(Self {
            initial_capital,
            commission,
            slippage,
            position_size,
            allow_short,
            fractional_shares,
            stop_loss: sl_spec,
            take_profit: tp_spec,
            borrow_cost,
            max_position,
            fill_price: fill_price.to_string(),
            freq: freq_parsed,
            trading_hours_24,
            max_volume_participation,
            order_type: order_type_lower,
            limit_offset,
        })
    }

    #[getter]
    fn get_stop_loss(&self, py: Python<'_>) -> PyObject {
        match &self.stop_loss {
            Some(StopSpec::Percentage(p)) => p.into_py(py),
            Some(StopSpec::Atr(m)) => format!("{}atr", m).into_py(py),
            Some(StopSpec::Trailing(p)) => format!("{}trail", p * 100.0).into_py(py),
            None => py.None(),
        }
    }

    #[getter]
    fn get_take_profit(&self, py: Python<'_>) -> PyObject {
        match &self.take_profit {
            Some(TakeProfitSpec::Percentage(p)) => p.into_py(py),
            Some(TakeProfitSpec::Atr(m)) => format!("{}atr", m).into_py(py),
            Some(TakeProfitSpec::RiskReward(r)) => format!("{}rr", r).into_py(py),
            None => py.None(),
        }
    }

    #[getter]
    fn get_freq(&self, py: Python<'_>) -> PyObject {
        match &self.freq {
            Some(f) => f.description().into_py(py),
            None => py.None(),
        }
    }

    #[getter]
    fn get_trading_hours_24(&self, py: Python<'_>) -> PyObject {
        match self.trading_hours_24 {
            Some(v) => v.into_py(py),
            None => py.None(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BacktestConfig(capital={:.0}, commission={:.4}, slippage={:.4}, size={:.2}, max_position={:.2})",
            self.initial_capital, self.commission, self.slippage, self.position_size, self.max_position
        )
    }
}

impl PyBacktestConfig {
    /// Convert to BacktestConfig, using bars to calculate ATR if needed.
    pub fn to_backtest_config(&self, bars: Option<&[Bar]>) -> BacktestConfig {
        let mut config = BacktestConfig {
            initial_capital: self.initial_capital,
            position_size: self.position_size,
            allow_short: self.allow_short,
            fractional_shares: self.fractional_shares,
            show_progress: false, // Disable progress bar in Python mode
            ..Default::default()
        };

        // Set cost model
        config.cost_model = CostModel {
            commission_pct: self.commission,
            slippage_pct: self.slippage,
            borrow_cost_rate: self.borrow_cost,
            max_volume_participation: self.max_volume_participation,
            ..Default::default()
        };

        // Calculate ATR if bars are provided (use 14-period ATR as default)
        let atr_value = bars.and_then(|b| atr(b, 14)).unwrap_or(0.0);

        // Set risk config
        config.risk_config.max_position_size = self.max_position;

        // Handle stop loss
        if let Some(ref sl_spec) = self.stop_loss {
            config.risk_config.stop_loss = match sl_spec {
                StopSpec::Percentage(pct) => StopLoss::Percentage(*pct),
                StopSpec::Atr(multiplier) => StopLoss::Atr {
                    multiplier: *multiplier,
                    atr_value,
                },
                StopSpec::Trailing(pct) => StopLoss::Trailing(*pct),
            };
        }

        // Handle take profit
        if let Some(ref tp_spec) = self.take_profit {
            config.risk_config.take_profit = match tp_spec {
                TakeProfitSpec::Percentage(pct) => TakeProfit::Percentage(*pct),
                TakeProfitSpec::Atr(multiplier) => TakeProfit::Atr {
                    multiplier: *multiplier,
                    atr_value,
                },
                TakeProfitSpec::RiskReward(ratio) => {
                    // Calculate stop distance for risk-reward
                    let stop_distance = match &self.stop_loss {
                        Some(StopSpec::Percentage(pct)) => *pct,
                        Some(StopSpec::Atr(mult)) => mult * atr_value,
                        Some(StopSpec::Trailing(pct)) => *pct,
                        None => 0.0,
                    };
                    TakeProfit::RiskReward {
                        ratio: *ratio,
                        stop_distance,
                    }
                }
            };
        }

        // Set execution price model
        config.execution_price = match self.fill_price.to_lowercase().as_str() {
            "open" | "next_open" => ExecutionPrice::Open,
            "close" => ExecutionPrice::Close,
            "vwap" => ExecutionPrice::Vwap,
            "twap" => ExecutionPrice::Twap,
            "midpoint" => ExecutionPrice::Midpoint,
            "random" | "random_in_range" => ExecutionPrice::RandomInRange,
            _ => ExecutionPrice::Open, // Default to open (prevents lookahead)
        };

        // Set data frequency if specified
        config.data_frequency = self.freq;

        // Set 24/7 trading hours if specified
        config.trading_hours_24 = self.trading_hours_24;

        // Set order type (limit vs market)
        config.use_limit_orders = self.order_type == "limit";
        config.limit_offset = self.limit_offset;

        config
    }
}

// Keep the From trait for backwards compatibility (without ATR support)
impl From<&PyBacktestConfig> for BacktestConfig {
    fn from(py_config: &PyBacktestConfig) -> Self {
        py_config.to_backtest_config(None)
    }
}

/// Run a backtest on historical data with a signal array.
///
/// This is the main entry point for running backtests from Python.
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     signal: numpy array of signals (1=long, -1=short, 0=flat) or None for built-in strategy
///     strategy: Name of built-in strategy if signal is None ("sma-crossover", "momentum", etc.)
///     strategy_params: Dictionary of strategy parameters
///     config: Optional BacktestConfig object
///     commission: Commission rate (default 0.001 = 0.1%)
///     slippage: Slippage rate (default 0.001 = 0.1%)
///     size: Position size as fraction of equity (default 0.10 = 10%)
///     cash: Initial capital (default 100,000)
///     stop_loss: Optional stop loss percentage (e.g., 0.05 for 5%)
///     take_profit: Optional take profit percentage
///     allow_short: Whether to allow short positions (default True)
///     max_position: Maximum position size as fraction of equity (default 1.0 = 100%)
///     fill_price: Execution price model ("next_open", "close", "vwap", "twap", "midpoint")
///     benchmark: Optional benchmark data for performance comparison (data dict from load())
///     freq: Optional data frequency override ("1min", "5min", "1h", "1d", etc.). Auto-detected if None.
///     trading_hours_24: Whether to use 24/7 trading hours for annualization (crypto). Auto-detected if None.
///     max_volume_participation: Maximum volume participation rate (e.g., 0.10 = 10% of bar volume).
///         Prevents unrealistic fills in illiquid markets. None means no limit (default).
///
/// Returns:
///     BacktestResult object with metrics, equity curve, and trades.
///     If benchmark is provided, also includes alpha, beta, benchmark_return, excess_return.
///
/// Example:
///     >>> data = load("AAPL.csv")
///     >>> signal = np.where(data['close'] > data['close'].mean(), 1, -1)
///     >>> results = backtest(data, signal)
///     >>> print(results.sharpe)
///     1.24
///     >>> # With ATR-based stop loss (2x ATR)
///     >>> results = backtest(data, signal, stop_loss="2atr")
///     >>> # With benchmark comparison
///     >>> spy = load("SPY.csv")
///     >>> results = backtest(data, signal, benchmark=spy)
///     >>> print(results.alpha, results.beta)
///     0.05 1.2
///     >>> # Using ONNX model for inference (requires --features onnx)
///     >>> results = backtest(data, model="model.onnx", features=feature_df)
#[pyfunction]
#[pyo3(signature = (
    data,
    signal=None,
    strategy=None,
    strategy_params=None,
    config=None,
    commission=0.001,
    slippage=0.001,
    size=0.10,
    cash=100_000.0,
    stop_loss=None,
    take_profit=None,
    allow_short=true,
    fractional=false,
    borrow_cost=0.03,
    max_position=1.0,
    fill_price="next_open",
    benchmark=None,
    freq=None,
    trading_hours_24=None,
    max_volume_participation=None,
    order_type="market",
    limit_offset=0.0,
    model=None,
    features=None,
    model_input_size=None,
    signal_threshold=None
))]
#[allow(clippy::too_many_arguments)]
pub fn backtest(
    py: Python<'_>,
    data: PyObject,
    signal: Option<PyReadonlyArray1<f64>>,
    strategy: Option<&str>,
    strategy_params: Option<&Bound<'_, PyDict>>,
    config: Option<&PyBacktestConfig>,
    commission: f64,
    slippage: f64,
    size: f64,
    cash: f64,
    stop_loss: Option<PyObject>,
    take_profit: Option<PyObject>,
    allow_short: bool,
    fractional: bool,
    borrow_cost: f64,
    max_position: f64,
    fill_price: &str,
    benchmark: Option<PyObject>,
    freq: Option<&str>,
    trading_hours_24: Option<bool>,
    max_volume_participation: Option<f64>,
    order_type: &str,
    limit_offset: f64,
    model: Option<&str>,
    features: Option<PyObject>,
    model_input_size: Option<usize>,
    signal_threshold: Option<f64>,
) -> PyResult<PyBacktestResult> {
    // Extract bars from data first (needed for ATR calculation)
    let bars = extract_bars(py, &data)?;

    if bars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Data is empty. Cannot run backtest without bars.",
        ));
    }

    // Build or use provided config (passing bars for ATR calculation)
    let bt_config = if let Some(cfg) = config {
        cfg.to_backtest_config(Some(&bars))
    } else {
        let py_config = PyBacktestConfig::new(
            py,
            cash,
            commission,
            slippage,
            size,
            allow_short,
            fractional,
            stop_loss,
            take_profit,
            borrow_cost,
            max_position,
            fill_price,
            freq,
            trading_hours_24,
            max_volume_participation,
            order_type,
            limit_offset,
        )?;
        py_config.to_backtest_config(Some(&bars))
    };

    // Create engine (clone config so we can store it for validation)
    let config_for_result = bt_config.clone();
    let mut engine = Engine::new(bt_config);
    engine.add_data("SYMBOL", bars.clone());

    // Run backtest based on signal, strategy, or ONNX model
    #[allow(unused_variables)]
    let (result, stored_signal) = if let Some(model_path) = model {
        // ONNX model-based backtest
        #[cfg(feature = "onnx")]
        {
            let features_obj = features.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "ONNX model requires 'features' parameter.\n\n\
                     Usage: mt.backtest(data, model='model.onnx', features=feature_df)\n\n\
                     The features should be a 2D array or DataFrame with one row per bar.",
                )
            })?;

            // Determine input size
            let input_size = if let Some(size) = model_input_size {
                size
            } else {
                // Try to infer from features
                infer_feature_size(py, &features_obj)?
            };

            // Load ONNX model
            let config = crate::onnx::ModelConfig::new(
                std::path::Path::new(model_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model"),
                input_size,
            );
            let mut onnx_model =
                crate::onnx::OnnxModel::from_file(model_path, config).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to load ONNX model '{}': {}\n\n\
                         Common causes:\n\
                           - File not found or invalid path\n\
                           - Model file is corrupted\n\
                           - Model uses unsupported operators\n\n\
                         Quick fix:\n\
                           Check that the model file exists and is a valid ONNX model.",
                        model_path, e
                    ))
                })?;

            // Extract features and run inference
            let signal_vec =
                run_onnx_inference(py, &mut onnx_model, &features_obj, signal_threshold)?;

            // Validate signal length
            if signal_vec.len() != bars.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "SignalShapeMismatch: ONNX model produced {} signals, data has {} bars.\n\n\
                     The features array must have one row per bar in the data.\n\n\
                     Quick fix:\n\
                       Ensure features.shape[0] == len(data['bars'])",
                    signal_vec.len(),
                    bars.len()
                )));
            }

            // Create a signal-based strategy wrapper
            let mut strategy = SignalStrategy::new(signal_vec.clone());
            let result = engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })?;
            (result, Some(signal_vec))
        }

        #[cfg(not(feature = "onnx"))]
        {
            let _ = features; // Suppress unused warning
            let _ = model_input_size;
            let _ = signal_threshold;
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "ONNX support is not enabled. Rebuild with `--features onnx` to enable ONNX model inference.\n\n\
                 Quick fix:\n\
                   pip install mantis-bt[onnx]  # If available\n\
                   # OR pre-compute signals in Python:\n\
                   signals = your_model.predict(features)\n\
                   results = mt.backtest(data, signals)",
            ));
        }
    } else if let Some(signal_arr) = signal {
        // Signal-based backtest
        let signal_vec: Vec<f64> = signal_arr.as_slice()?.to_vec();

        // Validate signal length
        if signal_vec.len() != bars.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "SignalShapeMismatch: Signal has {} rows, data has {}.\n\n\
                 Your signal must have the same length as the price data.\n\n\
                 Common causes:\n\
                   - Used .pct_change() or .diff() which drops the first row\n\
                   - Date index doesn't align\n\n\
                 Quick fix:\n\
                   signal = np.pad(signal, (missing_rows, 0), constant_values=0)",
                signal_vec.len(),
                bars.len()
            )));
        }

        // Create a signal-based strategy wrapper
        let mut strategy = SignalStrategy::new(signal_vec.clone());
        let result = engine.run(&mut strategy, "SYMBOL").map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
        })?;
        (result, Some(signal_vec))
    } else if let Some(strat_name) = strategy {
        // Built-in strategy - cannot store signal for validation
        let result = run_builtin_strategy(&mut engine, strat_name, strategy_params)?;
        (result, None)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Either 'signal' array, 'strategy' name, or 'model' path must be provided.",
        ));
    };

    // Calculate benchmark metrics if benchmark data is provided
    let benchmark_metrics = if let Some(ref bench_data) = benchmark {
        let bench_bars = extract_bars(py, bench_data)?;

        if bench_bars.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Benchmark data is empty.",
            ));
        }

        // Calculate returns from the backtest equity curve
        let portfolio_returns: Vec<f64> = result
            .equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect();

        // Calculate benchmark returns from close prices
        let benchmark_returns: Vec<f64> = bench_bars
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        // Align the return series (use minimum length)
        let min_len = portfolio_returns.len().min(benchmark_returns.len());
        if min_len < 2 {
            None
        } else {
            let portfolio_slice = &portfolio_returns[..min_len];
            let benchmark_slice = &benchmark_returns[..min_len];

            // Calculate benchmark metrics using the existing Rust implementation
            // Signature: calculate(benchmark_name, portfolio_returns, benchmark_returns, risk_free_rate)
            BenchmarkMetrics::calculate(
                "benchmark",
                portfolio_slice,
                benchmark_slice,
                0.0, // risk-free rate (can be made configurable later)
            )
        }
    } else {
        None
    };

    // Return result with stored data for validation support
    Ok(PyBacktestResult::from_result_with_data_and_benchmark(
        &result,
        Some(bars),
        stored_signal,
        Some(config_for_result),
        benchmark_metrics,
    ))
}

/// Check a signal for common issues before running a backtest.
///
/// This validates the signal and provides helpful feedback about potential issues.
///
/// Args:
///     data: Data dictionary from load()
///     signal: numpy array of signals
///
/// Returns:
///     Dictionary with validation results and suggestions.
///
/// Example:
///     >>> data = load("AAPL.csv")
///     >>> signal = np.random.choice([-1, 0, 1], len(data))
///     >>> check = signal_check(data, signal)
///     >>> print(check['status'])
///     'passed'
#[pyfunction]
pub fn signal_check(
    py: Python<'_>,
    data: PyObject,
    signal: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
    let bars = extract_bars(py, &data)?;
    let signal_vec: Vec<f64> = signal.as_slice()?.to_vec();

    let result = PyDict::new_bound(py);

    // Check shape
    let shape_ok = signal_vec.len() == bars.len();
    result.set_item("shape_match", shape_ok)?;
    result.set_item("signal_length", signal_vec.len())?;
    result.set_item("data_length", bars.len())?;

    if !shape_ok {
        result.set_item("status", "failed")?;
        result.set_item(
            "error",
            format!(
                "Signal has {} rows, data has {}",
                signal_vec.len(),
                bars.len()
            ),
        )?;
        return Ok(result.into());
    }

    // Check for NaN values
    let nan_count = signal_vec.iter().filter(|x| x.is_nan()).count();
    result.set_item("nan_count", nan_count)?;

    // Check for infinity
    let inf_count = signal_vec.iter().filter(|x| x.is_infinite()).count();
    result.set_item("inf_count", inf_count)?;

    // Analyze distribution
    let long_count = signal_vec.iter().filter(|&&x| x > 0.0).count();
    let short_count = signal_vec.iter().filter(|&&x| x < 0.0).count();
    let flat_count = signal_vec.iter().filter(|&&x| x == 0.0).count();

    result.set_item("long_count", long_count)?;
    result.set_item("short_count", short_count)?;
    result.set_item("flat_count", flat_count)?;

    let total = signal_vec.len() as f64;
    result.set_item("long_pct", long_count as f64 / total)?;
    result.set_item("short_pct", short_count as f64 / total)?;
    result.set_item("flat_pct", flat_count as f64 / total)?;

    // Check unique values
    let mut unique: Vec<f64> = signal_vec.clone();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique.dedup();
    result.set_item("unique_values", unique.len())?;
    result.set_item("is_discrete", unique.len() <= 10)?;

    // Warnings
    let mut warnings = Vec::new();

    if nan_count > 0 {
        warnings.push(format!(
            "Signal contains {} NaN values (will be treated as 0)",
            nan_count
        ));
    }

    if inf_count > 0 {
        warnings.push(format!(
            "Signal contains {} infinite values (may cause issues)",
            inf_count
        ));
    }

    if long_count == 0 && short_count == 0 {
        warnings.push("Signal is all zeros (no trades will be executed)".to_string());
    }

    if long_count as f64 / total > 0.95 || short_count as f64 / total > 0.95 {
        warnings.push("Signal is heavily skewed to one direction (>95%)".to_string());
    }

    result.set_item("warnings", warnings.clone())?;

    // Overall status
    let status =
        if nan_count > 0 || inf_count > 0 || !shape_ok || (long_count == 0 && short_count == 0) {
            "warning"
        } else {
            "passed"
        };
    result.set_item("status", status)?;

    // Tip based on signal characteristics
    if unique.len() <= 3 {
        result.set_item(
            "tip",
            "Signal looks like classification output. Consider using continuous confidence scores for position sizing.",
        )?;
    } else if unique.len() > 100 {
        result.set_item(
            "tip",
            "Signal has many unique values. This works well with signal-scaled position sizing (size='signal').",
        )?;
    }

    Ok(result.into())
}

/// Helper: extract bars from data object (dict, path string, pandas DataFrame, or polars DataFrame).
pub fn extract_bars(py: Python<'_>, data: &PyObject) -> PyResult<Vec<Bar>> {
    // Try as dictionary first
    if let Ok(dict) = data.downcast_bound::<PyDict>(py) {
        // Check if it has 'bars' key
        if let Some(bars_obj) = dict.get_item("bars")? {
            let bars: Vec<super::types::PyBar> = bars_obj.extract()?;
            return Ok(bars.into_iter().map(Bar::from).collect());
        }

        // Otherwise try to extract arrays
        let timestamps: Vec<i64> = dict
            .get_item("timestamp")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'timestamp' key"))?
            .extract()?;
        let opens: Vec<f64> = dict
            .get_item("open")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'open' key"))?
            .extract()?;
        let highs: Vec<f64> = dict
            .get_item("high")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'high' key"))?
            .extract()?;
        let lows: Vec<f64> = dict
            .get_item("low")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'low' key"))?
            .extract()?;
        let closes: Vec<f64> = dict
            .get_item("close")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'close' key"))?
            .extract()?;
        let volumes: Vec<f64> = dict
            .get_item("volume")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'volume' key"))?
            .extract()?;

        let bars: Vec<Bar> = timestamps
            .into_iter()
            .zip(opens)
            .zip(highs)
            .zip(lows)
            .zip(closes)
            .zip(volumes)
            .map(|(((((ts, o), h), l), c), v)| Bar {
                timestamp: chrono::TimeZone::timestamp_opt(&chrono::Utc, ts, 0).unwrap(),
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v,
            })
            .collect();

        return Ok(bars);
    }

    // Try as string (file path)
    if let Ok(path) = data.extract::<String>(py) {
        let config = DataConfig::default();
        let bars = if path.ends_with(".parquet") || path.ends_with(".pq") {
            load_parquet(&path, &config)
        } else {
            load_csv(&path, &config)
        }
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load data: {}", e)))?;

        return Ok(bars);
    }

    // Try as pandas DataFrame
    if let Ok(bars) = extract_bars_from_pandas(py, data) {
        return Ok(bars);
    }

    // Try as polars DataFrame
    if let Ok(bars) = extract_bars_from_polars(py, data) {
        return Ok(bars);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Data must be a dictionary from load(), a file path string, a pandas DataFrame, or a polars DataFrame.\n\n\
         For DataFrames, ensure columns are named: open/Open, high/High, low/Low, close/Close, volume/Volume.\n\
         The index should be a DatetimeIndex or there should be a 'date'/'timestamp' column.",
    ))
}

/// Extract bars from a pandas DataFrame.
fn extract_bars_from_pandas(py: Python<'_>, data: &PyObject) -> PyResult<Vec<Bar>> {
    // Check if it's a pandas DataFrame by looking at module
    let class = data.getattr(py, "__class__")?;
    let module: String = class.getattr(py, "__module__")?.extract(py)?;
    let class_name: String = class.getattr(py, "__name__")?.extract(py)?;

    if !module.starts_with("pandas") || class_name != "DataFrame" {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Not a pandas DataFrame",
        ));
    }

    // Get column names to find OHLCV columns (case-insensitive)
    let columns: Vec<String> = data
        .getattr(py, "columns")?
        .call_method0(py, "tolist")?
        .extract(py)?;

    let find_column = |names: &[&str]| -> Option<String> {
        for col in &columns {
            let col_lower = col.to_lowercase();
            for name in names {
                if col_lower == *name {
                    return Some(col.clone());
                }
            }
        }
        None
    };

    let open_col = find_column(&["open", "o"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'open'/'Open' column")
    })?;
    let high_col = find_column(&["high", "h"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'high'/'High' column")
    })?;
    let low_col = find_column(&["low", "l"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'low'/'Low' column")
    })?;
    let close_col = find_column(&["close", "c"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'close'/'Close' column")
    })?;
    let volume_col = find_column(&["volume", "vol", "v"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'volume'/'Volume' column")
    })?;

    // Extract OHLCV values using .values.tolist()
    let opens: Vec<f64> = data
        .getattr(py, &open_col as &str)?
        .getattr(py, "values")?
        .call_method0(py, "tolist")?
        .extract(py)?;
    let highs: Vec<f64> = data
        .getattr(py, &high_col as &str)?
        .getattr(py, "values")?
        .call_method0(py, "tolist")?
        .extract(py)?;
    let lows: Vec<f64> = data
        .getattr(py, &low_col as &str)?
        .getattr(py, "values")?
        .call_method0(py, "tolist")?
        .extract(py)?;
    let closes: Vec<f64> = data
        .getattr(py, &close_col as &str)?
        .getattr(py, "values")?
        .call_method0(py, "tolist")?
        .extract(py)?;
    let volumes: Vec<f64> = data
        .getattr(py, &volume_col as &str)?
        .getattr(py, "values")?
        .call_method0(py, "tolist")?
        .extract(py)?;

    // Try to get timestamps from index first
    let timestamps = extract_pandas_timestamps(py, data, &columns)?;

    if timestamps.len() != opens.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Timestamp count ({}) doesn't match row count ({})",
            timestamps.len(),
            opens.len()
        )));
    }

    let bars: Vec<Bar> = timestamps
        .into_iter()
        .zip(opens)
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .map(|(((((ts, o), h), l), c), v)| Bar {
            timestamp: chrono::TimeZone::timestamp_opt(&chrono::Utc, ts, 0).unwrap(),
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
        })
        .collect();

    Ok(bars)
}

/// Extract timestamps from pandas DataFrame index or date column.
fn extract_pandas_timestamps(
    py: Python<'_>,
    data: &PyObject,
    columns: &[String],
) -> PyResult<Vec<i64>> {
    // First, try to get timestamps from the index
    let index = data.getattr(py, "index")?;
    let index_class = index.getattr(py, "__class__")?;
    let index_name: String = index_class.getattr(py, "__name__")?.extract(py)?;

    if index_name == "DatetimeIndex" {
        // Convert DatetimeIndex to Unix timestamps
        let timestamps: Vec<i64> = index
            .call_method1(py, "astype", ("int64",))?
            .call_method1(py, "__floordiv__", (1_000_000_000i64,))? // nanoseconds to seconds
            .call_method0(py, "tolist")?
            .extract(py)?;
        return Ok(timestamps);
    }

    // Try to find a date/timestamp column
    let find_date_column = |names: &[&str]| -> Option<String> {
        for col in columns {
            let col_lower = col.to_lowercase();
            for name in names {
                if col_lower == *name {
                    return Some(col.clone());
                }
            }
        }
        None
    };

    if let Some(date_col) = find_date_column(&["date", "datetime", "timestamp", "time", "dt"]) {
        // Try to convert the date column to timestamps
        let date_series = data.getattr(py, &date_col as &str)?;

        // Try converting via pandas.to_datetime
        let pd = py.import_bound("pandas")?;
        let dt_series = pd.call_method1("to_datetime", (date_series,))?;
        let timestamps: Vec<i64> = dt_series
            .call_method1("astype", ("int64",))?
            .call_method1("__floordiv__", (1_000_000_000i64,))?
            .call_method0("tolist")?
            .extract()?;
        return Ok(timestamps);
    }

    // Fall back to generating sequential timestamps (one day apart starting from 2020-01-01)
    let n_rows: usize = data.call_method0(py, "__len__")?.extract(py)?;
    let base_ts = 1577836800i64; // 2020-01-01 00:00:00 UTC
    let timestamps: Vec<i64> = (0..n_rows).map(|i| base_ts + (i as i64) * 86400).collect();
    Ok(timestamps)
}

/// Extract bars from a polars DataFrame.
fn extract_bars_from_polars(py: Python<'_>, data: &PyObject) -> PyResult<Vec<Bar>> {
    // Check if it's a polars DataFrame by looking at module
    let class = data.getattr(py, "__class__")?;
    let module: String = class.getattr(py, "__module__")?.extract(py)?;
    let class_name: String = class.getattr(py, "__name__")?.extract(py)?;

    if !module.starts_with("polars") || class_name != "DataFrame" {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Not a polars DataFrame",
        ));
    }

    // Get column names
    let columns: Vec<String> = data.getattr(py, "columns")?.extract(py)?;

    let find_column = |names: &[&str]| -> Option<String> {
        for col in &columns {
            let col_lower = col.to_lowercase();
            for name in names {
                if col_lower == *name {
                    return Some(col.clone());
                }
            }
        }
        None
    };

    let open_col = find_column(&["open", "o"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'open'/'Open' column")
    })?;
    let high_col = find_column(&["high", "h"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'high'/'High' column")
    })?;
    let low_col = find_column(&["low", "l"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'low'/'Low' column")
    })?;
    let close_col = find_column(&["close", "c"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'close'/'Close' column")
    })?;
    let volume_col = find_column(&["volume", "vol", "v"]).ok_or_else(|| {
        pyo3::exceptions::PyKeyError::new_err("DataFrame missing 'volume'/'Volume' column")
    })?;

    // Extract column data using .to_list()
    let opens: Vec<f64> = data
        .call_method1(py, "get_column", (&open_col,))?
        .call_method0(py, "to_list")?
        .extract(py)?;
    let highs: Vec<f64> = data
        .call_method1(py, "get_column", (&high_col,))?
        .call_method0(py, "to_list")?
        .extract(py)?;
    let lows: Vec<f64> = data
        .call_method1(py, "get_column", (&low_col,))?
        .call_method0(py, "to_list")?
        .extract(py)?;
    let closes: Vec<f64> = data
        .call_method1(py, "get_column", (&close_col,))?
        .call_method0(py, "to_list")?
        .extract(py)?;
    let volumes: Vec<f64> = data
        .call_method1(py, "get_column", (&volume_col,))?
        .call_method0(py, "to_list")?
        .extract(py)?;

    // Get timestamps
    let timestamps = extract_polars_timestamps(py, data, &columns)?;

    if timestamps.len() != opens.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Timestamp count ({}) doesn't match row count ({})",
            timestamps.len(),
            opens.len()
        )));
    }

    let bars: Vec<Bar> = timestamps
        .into_iter()
        .zip(opens)
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .map(|(((((ts, o), h), l), c), v)| Bar {
            timestamp: chrono::TimeZone::timestamp_opt(&chrono::Utc, ts, 0).unwrap(),
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
        })
        .collect();

    Ok(bars)
}

/// Extract timestamps from polars DataFrame.
fn extract_polars_timestamps(
    py: Python<'_>,
    data: &PyObject,
    columns: &[String],
) -> PyResult<Vec<i64>> {
    let find_date_column = |names: &[&str]| -> Option<String> {
        for col in columns {
            let col_lower = col.to_lowercase();
            for name in names {
                if col_lower == *name {
                    return Some(col.clone());
                }
            }
        }
        None
    };

    if let Some(date_col) = find_date_column(&["date", "datetime", "timestamp", "time", "dt"]) {
        // Get the date column and convert to Unix timestamps
        let date_series = data.call_method1(py, "get_column", (&date_col,))?;

        // Try to cast to datetime and get timestamps
        // In polars, we can use .dt.timestamp("s") for seconds
        let dtype = date_series.call_method0(py, "dtype")?;
        let dtype_str: String = dtype.call_method0(py, "__str__")?.extract(py)?;

        if dtype_str.starts_with("Datetime") || dtype_str.starts_with("Date") {
            // Get timestamp in seconds
            let timestamps: Vec<i64> = date_series
                .getattr(py, "dt")?
                .call_method1(py, "timestamp", ("s",))?
                .call_method0(py, "to_list")?
                .extract(py)?;
            return Ok(timestamps);
        } else {
            // Try to cast to datetime first
            let pl = py.import_bound("polars")?;
            let casted = date_series.call_method1(py, "cast", (pl.getattr("Datetime")?,))?;
            let timestamps: Vec<i64> = casted
                .getattr(py, "dt")?
                .call_method1(py, "timestamp", ("s",))?
                .call_method0(py, "to_list")?
                .extract(py)?;
            return Ok(timestamps);
        }
    }

    // Fall back to sequential timestamps
    let n_rows: usize = data.call_method0(py, "height")?.extract(py)?;
    let base_ts = 1577836800i64; // 2020-01-01 00:00:00 UTC
    let timestamps: Vec<i64> = (0..n_rows).map(|i| base_ts + (i as i64) * 86400).collect();
    Ok(timestamps)
}

/// Wrapper strategy that uses pre-computed signals.
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
            Signal::Exit // 0 means exit/flat
        }
    }
}

/// Run a built-in strategy by name.
fn run_builtin_strategy(
    engine: &mut Engine,
    name: &str,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<crate::engine::BacktestResult> {
    match name {
        "sma-crossover" | "sma_crossover" => {
            let (fast, slow) = if let Some(p) = params {
                let fast: usize = p
                    .get_item("fast")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(10);
                let slow: usize = p
                    .get_item("slow")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(30);
                (fast, slow)
            } else {
                (10, 30)
            };
            let mut strategy = SmaCrossover::new(fast, slow);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        "momentum" => {
            let (lookback, threshold) = if let Some(p) = params {
                let lb: usize = p
                    .get_item("lookback")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(20);
                let th: f64 = p
                    .get_item("threshold")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.0);
                (lb, th)
            } else {
                (20, 0.0)
            };
            let mut strategy = MomentumStrategy::new(lookback, threshold);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        "mean-reversion" | "mean_reversion" => {
            let (period, num_std, entry_std, exit_std) = if let Some(p) = params {
                let per: usize = p
                    .get_item("period")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(20);
                let ns: f64 = p
                    .get_item("num_std")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(2.0);
                let es: f64 = p
                    .get_item("entry_std")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(2.0);
                let xs: f64 = p
                    .get_item("exit_std")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(0.5);
                (per, ns, es, xs)
            } else {
                (20, 2.0, 2.0, 0.5)
            };
            let mut strategy = MeanReversion::new(period, num_std, entry_std, exit_std);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        "rsi" => {
            let (period, oversold, overbought) = if let Some(p) = params {
                let pd: usize = p
                    .get_item("period")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(14);
                let os: f64 = p
                    .get_item("oversold")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(30.0);
                let ob: f64 = p
                    .get_item("overbought")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(70.0);
                (pd, os, ob)
            } else {
                (14, 30.0, 70.0)
            };
            let mut strategy = RsiStrategy::new(period, oversold, overbought);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        "macd" => {
            let (fast, slow, signal_period) = if let Some(p) = params {
                let f: usize = p
                    .get_item("fast")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(12);
                let s: usize = p
                    .get_item("slow")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(26);
                let sp: usize = p
                    .get_item("signal")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(9);
                (f, s, sp)
            } else {
                (12, 26, 9)
            };
            let mut strategy = MacdStrategy::new(fast, slow, signal_period);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        "breakout" => {
            let (entry_period, exit_period) = if let Some(p) = params {
                let ep: usize = p
                    .get_item("entry_period")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(20);
                let xp: usize = p
                    .get_item("exit_period")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or(10);
                (ep, xp)
            } else {
                (20, 10)
            };
            let mut strategy = BreakoutStrategy::new(entry_period, exit_period);
            engine.run(&mut strategy, "SYMBOL").map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Backtest failed: {}", e))
            })
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown strategy: '{}'. Available: sma-crossover, momentum, mean-reversion, rsi, macd, breakout",
            name
        ))),
    }
}

/// Run walk-forward validation on a signal-based strategy.
///
/// This performs walk-forward analysis by splitting the data into multiple
/// folds, running the backtest on each in-sample period, and then testing
/// on the out-of-sample period.
///
/// Args:
///     data: Data dictionary from load() or path to CSV/Parquet file
///     signal: numpy array of signals (1=long, -1=short, 0=flat)
///     folds: Number of walk-forward folds (default 12)
///     in_sample_ratio: Fraction of each fold for in-sample (default 0.75)
///     anchored: Use anchored (growing) windows instead of rolling (default True)
///     config: Optional BacktestConfig object
///     commission: Commission rate (default 0.001 = 0.1%)
///     slippage: Slippage rate (default 0.001 = 0.1%)
///     size: Position size as fraction of equity (default 0.10 = 10%)
///     cash: Initial capital (default 100,000)
///
/// Returns:
///     ValidationResult object with fold details and verdict.
///
/// Example:
///     >>> data = load("AAPL.csv")
///     >>> signal = model.predict(features)
///     >>> validation = validate(data, signal)
///     >>> print(validation.verdict)
///     'robust'
#[pyfunction]
#[pyo3(signature = (
    data,
    signal,
    folds=12,
    in_sample_ratio=0.75,
    anchored=true,
    config=None,
    commission=0.001,
    slippage=0.001,
    size=0.10,
    cash=100_000.0
))]
#[allow(clippy::too_many_arguments)]
pub fn validate(
    py: Python<'_>,
    data: PyObject,
    signal: PyReadonlyArray1<f64>,
    folds: usize,
    in_sample_ratio: f64,
    anchored: bool,
    config: Option<&PyBacktestConfig>,
    commission: f64,
    slippage: f64,
    size: f64,
    cash: f64,
) -> PyResult<super::results::PyValidationResult> {
    // Extract bars and signal first
    let bars = extract_bars(py, &data)?;

    // Build config (pass bars for ATR calculation if needed)
    let bt_config = if let Some(cfg) = config {
        cfg.to_backtest_config(Some(&bars))
    } else {
        let py_config = PyBacktestConfig {
            initial_capital: cash,
            commission,
            slippage,
            position_size: size,
            allow_short: true,
            fractional_shares: false, // Default: whole shares (per spec)
            stop_loss: None,
            take_profit: None,
            borrow_cost: 0.03,
            max_position: 1.0,
            fill_price: "next_open".to_string(),
            freq: None,
            trading_hours_24: None,
            max_volume_participation: None,
            order_type: "market".to_string(),
            limit_offset: 0.0,
        };
        py_config.to_backtest_config(Some(&bars))
    };
    let signal_vec: Vec<f64> = signal.as_slice()?.to_vec();

    if bars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Data is empty. Cannot run validation without bars.",
        ));
    }

    if signal_vec.len() != bars.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SignalShapeMismatch: Signal has {} rows, data has {}.",
            signal_vec.len(),
            bars.len()
        )));
    }

    // Validate walk-forward config
    if folds < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 folds for walk-forward validation.",
        ));
    }

    if in_sample_ratio <= 0.0 || in_sample_ratio >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "in_sample_ratio must be between 0 and 1 (exclusive).",
        ));
    }

    // Calculate window boundaries
    let wf_config = WalkForwardConfig {
        num_windows: folds,
        in_sample_ratio,
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

    // Calculate windows
    let windows = calculate_signal_windows(&bars, &wf_config)?;

    // Run backtest on each window
    let mut window_results = Vec::with_capacity(windows.len());

    for window in &windows {
        // Extract IS data and signal
        let is_bars: Vec<Bar> = bars[window.is_start_idx..=window.is_end_idx].to_vec();
        let is_signal: Vec<f64> = signal_vec[window.is_start_idx..=window.is_end_idx].to_vec();

        // Extract OOS data and signal
        let oos_bars: Vec<Bar> = bars[window.oos_start_idx..=window.oos_end_idx].to_vec();
        let oos_signal: Vec<f64> = signal_vec[window.oos_start_idx..=window.oos_end_idx].to_vec();

        // Run IS backtest
        let is_result = run_signal_backtest(&is_bars, &is_signal, &bt_config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "In-sample backtest failed for fold {}: {}",
                window.index + 1,
                e
            ))
        })?;

        // Run OOS backtest
        let oos_result = run_signal_backtest(&oos_bars, &oos_signal, &bt_config).map_err(|e| {
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

        // Create WalkForwardWindow for the result
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
            parameter_hash: 0, // Signal-based strategies don't have parameters
        });
    }

    // Build WalkForwardResult
    let wf_result = build_wf_result(wf_config, window_results);

    Ok(super::results::PyValidationResult::from_wf_result(
        &wf_result,
    ))
}

/// Internal window structure with indices.
struct SignalWindow {
    index: usize,
    is_start_idx: usize,
    is_end_idx: usize,
    oos_start_idx: usize,
    oos_end_idx: usize,
}

/// Calculate window boundaries for signal-based walk-forward.
fn calculate_signal_windows(
    bars: &[Bar],
    config: &WalkForwardConfig,
) -> PyResult<Vec<SignalWindow>> {
    let total_bars = bars.len();
    let window_size = total_bars / config.num_windows;
    let is_size = (window_size as f64 * config.in_sample_ratio) as usize;
    let oos_size = window_size - is_size;

    if oos_size < 10 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Out-of-sample period too small. Reduce folds or increase data.",
        ));
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

        windows.push(SignalWindow {
            index: i,
            is_start_idx: window_start,
            is_end_idx,
            oos_start_idx,
            oos_end_idx,
        });
    }

    if windows.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Could not create valid walk-forward windows. Try using more data or fewer folds.",
        ));
    }

    Ok(windows)
}

/// Run a backtest with signal array.
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

/// Build WalkForwardResult from window results.
fn build_wf_result(config: WalkForwardConfig, windows: Vec<WindowResult>) -> WalkForwardResult {
    let n = windows.len() as f64;

    let avg_is_return = windows
        .iter()
        .map(|w| w.in_sample_result.total_return_pct)
        .sum::<f64>()
        / n;
    let avg_oos_return = windows
        .iter()
        .map(|w| w.out_of_sample_result.total_return_pct)
        .sum::<f64>()
        / n;
    let avg_efficiency_ratio = windows
        .iter()
        .map(|w| w.efficiency_ratio)
        .filter(|e| e.is_finite())
        .sum::<f64>()
        / n;
    let avg_is_sharpe = windows
        .iter()
        .map(|w| w.in_sample_result.sharpe_ratio)
        .sum::<f64>()
        / n;
    let avg_oos_sharpe = windows
        .iter()
        .map(|w| w.out_of_sample_result.sharpe_ratio)
        .sum::<f64>()
        / n;

    let oos_sharpe_threshold_met = if avg_is_sharpe.abs() > 0.001 {
        avg_oos_sharpe >= avg_is_sharpe * 0.6
    } else {
        false
    };

    // Signal strategies always have stability 1.0 (no parameters)
    let parameter_stability = 1.0;

    let combined_oos_return = windows.iter().fold(1.0, |acc, w| {
        acc * (1.0 + w.out_of_sample_result.total_return_pct / 100.0)
    }) - 1.0;

    let total_is_return = windows.iter().fold(1.0, |acc, w| {
        acc * (1.0 + w.in_sample_result.total_return_pct / 100.0)
    }) - 1.0;

    let walk_forward_efficiency = if total_is_return.abs() > 0.001 {
        combined_oos_return / total_is_return
    } else {
        0.0
    };

    WalkForwardResult {
        config,
        windows,
        combined_oos_return: combined_oos_return * 100.0,
        avg_is_return,
        avg_oos_return,
        avg_efficiency_ratio,
        walk_forward_efficiency,
        avg_is_sharpe,
        avg_oos_sharpe,
        oos_sharpe_threshold_met,
        parameter_stability,
    }
}

/// Infer the feature size from a features object (numpy array, list, or DataFrame).
#[cfg(feature = "onnx")]
fn infer_feature_size(py: Python<'_>, features: &PyObject) -> PyResult<usize> {
    // Try as numpy array with shape attribute
    if let Ok(shape) = features.getattr(py, "shape") {
        let shape_tuple: Vec<usize> = shape.extract(py)?;
        if shape_tuple.len() >= 2 {
            return Ok(shape_tuple[1]); // Number of columns
        } else if shape_tuple.len() == 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Features must be 2D (rows x columns). Got 1D array.\n\n\
                 Quick fix:\n\
                   features = features.reshape(-1, 1)  # For single feature",
            ));
        }
    }

    // Try as DataFrame with columns
    if let Ok(columns) = features.getattr(py, "columns") {
        let col_list: Vec<String> = columns.call_method0(py, "tolist")?.extract(py)?;
        return Ok(col_list.len());
    }

    // Try as list of lists
    if let Ok(rows) = features.extract::<Vec<Vec<f64>>>(py) {
        if !rows.is_empty() {
            return Ok(rows[0].len());
        }
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "Cannot infer feature size. Please provide model_input_size parameter.\n\n\
         Usage: mt.backtest(data, model='model.onnx', features=features, model_input_size=10)",
    ))
}

/// Run ONNX inference on features and return signals.
#[cfg(feature = "onnx")]
fn run_onnx_inference(
    py: Python<'_>,
    model: &mut crate::onnx::OnnxModel,
    features: &PyObject,
    threshold: Option<f64>,
) -> PyResult<Vec<f64>> {
    // Extract features as Vec<Vec<f64>>
    let feature_vecs: Vec<Vec<f64>> = extract_feature_rows(py, features)?;

    if feature_vecs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Features array is empty. Cannot run ONNX inference without features.",
        ));
    }

    // Run batch inference
    let predictions = model.predict_batch(&feature_vecs).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "ONNX inference failed: {}\n\n\
             Common causes:\n\
               - Feature dimensions don't match model input size\n\
               - NaN or infinite values in features\n\n\
             Quick fix:\n\
               - Check model.input_size matches your feature count\n\
               - Use np.nan_to_num(features) to handle NaN values",
            e
        ))
    })?;

    // Convert to signals
    let signals: Vec<f64> = if let Some(thresh) = threshold {
        predictions
            .into_iter()
            .map(|p| {
                let p64 = p as f64;
                if p64 > thresh {
                    1.0
                } else if p64 < -thresh {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        // Use raw predictions as signals
        predictions.into_iter().map(|p| p as f64).collect()
    };

    Ok(signals)
}

/// Extract feature rows from various Python types.
#[cfg(feature = "onnx")]
fn extract_feature_rows(py: Python<'_>, features: &PyObject) -> PyResult<Vec<Vec<f64>>> {
    // Try as list of lists
    if let Ok(rows) = features.extract::<Vec<Vec<f64>>>(py) {
        return Ok(rows);
    }

    // Try as numpy array
    if let Ok(values) = features.call_method0(py, "tolist") {
        if let Ok(rows) = values.extract::<Vec<Vec<f64>>>(py) {
            return Ok(rows);
        }
    }

    // Try as pandas DataFrame (use .values.tolist())
    if let Ok(values) = features.getattr(py, "values") {
        if let Ok(list) = values.call_method0(py, "tolist") {
            if let Ok(rows) = list.extract::<Vec<Vec<f64>>>(py) {
                return Ok(rows);
            }
        }
    }

    // Try as polars DataFrame
    if let Ok(rows_method) = features.call_method0(py, "to_numpy") {
        if let Ok(list) = rows_method.call_method0(py, "tolist") {
            if let Ok(rows) = list.extract::<Vec<Vec<f64>>>(py) {
                return Ok(rows);
            }
        }
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Features must be a 2D numpy array, list of lists, pandas DataFrame, or polars DataFrame.",
    ))
}
