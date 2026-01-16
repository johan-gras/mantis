//! Python-exposed core types.

use chrono::{TimeZone, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::types::{Bar, Trade};

/// Python-exposed OHLCV bar.
#[pyclass(name = "Bar")]
#[derive(Debug, Clone)]
pub struct PyBar {
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub open: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub volume: f64,
}

#[pymethods]
impl PyBar {
    #[new]
    fn new(timestamp: i64, open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Bar(timestamp={}, open={:.2}, high={:.2}, low={:.2}, close={:.2}, volume={:.0})",
            self.timestamp, self.open, self.high, self.low, self.close, self.volume
        )
    }

    /// Convert to dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("timestamp", self.timestamp)?;
        dict.set_item("open", self.open)?;
        dict.set_item("high", self.high)?;
        dict.set_item("low", self.low)?;
        dict.set_item("close", self.close)?;
        dict.set_item("volume", self.volume)?;
        Ok(dict)
    }
}

impl From<&Bar> for PyBar {
    fn from(bar: &Bar) -> Self {
        Self {
            timestamp: bar.timestamp.timestamp(),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }
    }
}

impl From<PyBar> for Bar {
    fn from(bar: PyBar) -> Self {
        Bar {
            timestamp: Utc.timestamp_opt(bar.timestamp, 0).unwrap(),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
        }
    }
}

/// Python-exposed trade information.
#[pyclass(name = "Trade")]
#[derive(Debug, Clone)]
pub struct PyTrade {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub entry_time: i64,
    #[pyo3(get)]
    pub exit_time: Option<i64>,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub exit_price: Option<f64>,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub pnl: f64,
    #[pyo3(get)]
    pub pnl_pct: f64,
    #[pyo3(get)]
    pub direction: String,
    #[pyo3(get)]
    pub commission: f64,
    #[pyo3(get)]
    pub slippage: f64,
}

#[pymethods]
impl PyTrade {
    fn __repr__(&self) -> String {
        format!(
            "Trade(symbol='{}', direction='{}', entry={:.2}, exit={:?}, pnl={:.2})",
            self.symbol, self.direction, self.entry_price, self.exit_price, self.pnl
        )
    }

    /// Convert to dictionary.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("symbol", &self.symbol)?;
        dict.set_item("entry_time", self.entry_time)?;
        dict.set_item("exit_time", self.exit_time)?;
        dict.set_item("entry_price", self.entry_price)?;
        dict.set_item("exit_price", self.exit_price)?;
        dict.set_item("quantity", self.quantity)?;
        dict.set_item("pnl", self.pnl)?;
        dict.set_item("pnl_pct", self.pnl_pct)?;
        dict.set_item("direction", &self.direction)?;
        dict.set_item("commission", self.commission)?;
        dict.set_item("slippage", self.slippage)?;
        Ok(dict)
    }
}

impl From<&Trade> for PyTrade {
    fn from(trade: &Trade) -> Self {
        let direction = match trade.side {
            crate::types::Side::Buy => "long",
            crate::types::Side::Sell => "short",
        };

        // Calculate P&L if we have exit price
        let (pnl, pnl_pct) = if let Some(exit_price) = trade.exit_price {
            let notional = trade.quantity * trade.entry_price;
            let pnl = match trade.side {
                crate::types::Side::Buy => {
                    (exit_price - trade.entry_price) * trade.quantity - trade.commission - trade.slippage
                }
                crate::types::Side::Sell => {
                    (trade.entry_price - exit_price) * trade.quantity - trade.commission - trade.slippage
                }
            };
            let pnl_pct = if notional > 0.0 { pnl / notional } else { 0.0 };
            (pnl, pnl_pct)
        } else {
            (0.0, 0.0)
        };

        Self {
            symbol: trade.symbol.clone(),
            entry_time: trade.entry_time.timestamp(),
            exit_time: trade.exit_time.map(|t| t.timestamp()),
            entry_price: trade.entry_price,
            exit_price: trade.exit_price,
            quantity: trade.quantity,
            pnl,
            pnl_pct,
            direction: direction.to_string(),
            commission: trade.commission,
            slippage: trade.slippage,
        }
    }
}
