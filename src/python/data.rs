//! Python bindings for data loading.
//!
//! Provides functions to load data from CSV/Parquet files and convert
//! between Rust data structures and Python pandas/polars DataFrames.

use std::collections::HashMap;
use std::path::Path;

use chrono::TimeZone;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::data::{
    adjust_for_dividends, adjust_for_splits, list_samples as rust_list_samples, load_csv,
    load_parquet, load_sample as rust_load_sample, DataConfig,
};
use crate::types::{Bar, CorporateAction, CorporateActionType, DividendAdjustMethod, DividendType};

use super::types::PyBar;

/// Load OHLCV data from a CSV or Parquet file.
///
/// Auto-detects file format from extension and column names.
/// Returns data as a list of Bar objects or as a numpy array.
///
/// Args:
///     path: Path to the data file (.csv or .parquet)
///     date_format: Optional date format string (e.g., "%Y-%m-%d")
///
/// Returns:
///     Dictionary with 'bars' (list of Bar objects) and numpy arrays for each column.
///
/// Example:
///     >>> data = load("AAPL.csv")
///     >>> print(len(data['bars']))
///     2520
#[pyfunction]
#[pyo3(signature = (path, date_format=None))]
pub fn load(py: Python<'_>, path: &str, date_format: Option<&str>) -> PyResult<PyObject> {
    let path_obj = Path::new(path);

    // Build config
    let config = DataConfig {
        date_format: date_format.map(String::from),
        ..Default::default()
    };

    // Load based on extension
    let bars: Vec<Bar> = match path_obj.extension().and_then(|e| e.to_str()) {
        Some("parquet") | Some("pq") => load_parquet(path, &config).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load parquet file: {}", e))
        })?,
        Some("csv") => load_csv(path, &config).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load CSV file: {}", e))
        })?,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported file extension: {:?}. Supported: .csv, .parquet, .pq",
                other
            )))
        }
    };

    if bars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Data file is empty or has no valid rows",
        ));
    }

    // Convert to numpy arrays for efficient access
    let n = bars.len();
    let timestamps: Vec<i64> = bars.iter().map(|b| b.timestamp.timestamp()).collect();
    let opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();

    // Create result dictionary
    let dict = PyDict::new_bound(py);

    // Add numpy arrays
    dict.set_item("timestamp", PyArray1::from_vec_bound(py, timestamps))?;
    dict.set_item("open", PyArray1::from_vec_bound(py, opens))?;
    dict.set_item("high", PyArray1::from_vec_bound(py, highs))?;
    dict.set_item("low", PyArray1::from_vec_bound(py, lows))?;
    dict.set_item("close", PyArray1::from_vec_bound(py, closes))?;
    dict.set_item("volume", PyArray1::from_vec_bound(py, volumes))?;

    // Add metadata
    dict.set_item("n_bars", n)?;
    dict.set_item("path", path)?;

    // Convenience: add bars as list of PyBar objects
    let py_bars: Vec<Py<PyBar>> = bars
        .iter()
        .map(|b| Py::new(py, PyBar::from(b)).unwrap())
        .collect();
    dict.set_item("bars", py_bars)?;

    Ok(dict.into())
}

/// Load multiple symbols from a dictionary of paths.
///
/// Args:
///     paths: Dictionary mapping symbol names to file paths
///     date_format: Optional date format string
///
/// Returns:
///     Dictionary mapping symbol names to data dictionaries.
///
/// Example:
///     >>> data = load_multi({"AAPL": "AAPL.csv", "GOOGL": "GOOGL.csv"})
///     >>> print(data["AAPL"]["n_bars"])
///     2520
#[pyfunction]
#[pyo3(signature = (paths, date_format=None))]
pub fn load_multi(
    py: Python<'_>,
    paths: HashMap<String, String>,
    date_format: Option<&str>,
) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);

    for (symbol, path) in paths {
        let data = load(py, &path, date_format)?;
        result.set_item(symbol, data)?;
    }

    Ok(result.into())
}

/// Load all files matching a glob pattern from a directory.
///
/// Symbol names are derived from filenames (without extension).
///
/// Args:
///     pattern: Glob pattern (e.g., "data/*.csv" or "data/**/*.parquet")
///     date_format: Optional date format string
///
/// Returns:
///     Dictionary mapping symbol names to data dictionaries.
///
/// Example:
///     >>> data = load_dir("data/*.csv")
///     >>> print(list(data.keys()))
///     ['AAPL', 'GOOGL', 'MSFT']
#[pyfunction]
#[pyo3(signature = (pattern, date_format=None))]
pub fn load_dir(py: Python<'_>, pattern: &str, date_format: Option<&str>) -> PyResult<PyObject> {
    let result = PyDict::new_bound(py);

    // Use glob to find matching files
    let entries = glob::glob(pattern).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid glob pattern: {}", e))
    })?;

    for entry in entries.flatten() {
        // Get symbol name from filename
        let symbol = entry
            .file_stem()
            .and_then(|s| s.to_str())
            .map(String::from)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Could not extract symbol from filename")
            })?;

        let path_str = entry.to_string_lossy().to_string();
        let data = load(py, &path_str, date_format)?;
        result.set_item(symbol, data)?;
    }

    Ok(result.into())
}

/// Convert bars to a 2D numpy array with columns [timestamp, open, high, low, close, volume].
#[allow(dead_code)]
pub fn bars_to_array<'py>(py: Python<'py>, bars: &[Bar]) -> Bound<'py, PyArray2<f64>> {
    let data: Vec<Vec<f64>> = bars
        .iter()
        .map(|bar| {
            vec![
                bar.timestamp.timestamp() as f64,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
            ]
        })
        .collect();

    PyArray2::from_vec2_bound(py, &data).expect("conversion should succeed")
}

/// List available bundled sample datasets.
///
/// Returns a list of sample names that can be used with `load_sample()`.
/// Sample data is bundled with the package and requires no external files.
///
/// Returns:
///     List of sample data identifiers (e.g., ["AAPL", "SPY", "BTC"])
///
/// Example:
///     >>> samples = list_samples()
///     >>> print(samples)
///     ['AAPL', 'SPY', 'BTC']
#[pyfunction]
pub fn list_samples() -> Vec<&'static str> {
    rust_list_samples()
}

/// Load bundled sample data by name.
///
/// Sample data is embedded in the binary and requires no external files or internet.
/// This is useful for quick demos, testing, and getting started without downloading data.
///
/// Available samples:
///     - "AAPL": Apple Inc. stock (10 years daily, 2014-2024, ~2600 bars)
///     - "SPY": S&P 500 ETF (10 years daily, 2014-2024, ~2600 bars)
///     - "BTC": Bitcoin (10 years daily including weekends, 2014-2024, ~3650 bars)
///
/// Args:
///     name: Sample data identifier (case-insensitive, e.g., "AAPL", "aapl", "Aapl")
///
/// Returns:
///     Dictionary with 'bars' (list of Bar objects) and numpy arrays for each column.
///
/// Raises:
///     ValueError: If the sample name is not recognized
///
/// Example:
///     >>> data = load_sample("AAPL")
///     >>> print(data['n_bars'])
///     2609
///     >>> print(data['close'][-1])  # Last closing price
///     212.45
#[pyfunction]
pub fn load_sample(py: Python<'_>, name: &str) -> PyResult<PyObject> {
    let bars = rust_load_sample(name).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Failed to load sample '{}': {}. Available samples: {:?}",
            name,
            e,
            rust_list_samples()
        ))
    })?;

    if bars.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Sample data is empty",
        ));
    }

    // Convert to numpy arrays for efficient access
    let n = bars.len();
    let timestamps: Vec<i64> = bars.iter().map(|b| b.timestamp.timestamp()).collect();
    let opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();

    // Create result dictionary
    let dict = PyDict::new_bound(py);

    // Add numpy arrays
    dict.set_item("timestamp", PyArray1::from_vec_bound(py, timestamps))?;
    dict.set_item("open", PyArray1::from_vec_bound(py, opens))?;
    dict.set_item("high", PyArray1::from_vec_bound(py, highs))?;
    dict.set_item("low", PyArray1::from_vec_bound(py, lows))?;
    dict.set_item("close", PyArray1::from_vec_bound(py, closes))?;
    dict.set_item("volume", PyArray1::from_vec_bound(py, volumes))?;

    // Add metadata
    dict.set_item("n_bars", n)?;
    dict.set_item("sample_name", name.to_uppercase())?;

    // Convenience: add bars as list of PyBar objects
    let py_bars: Vec<Py<PyBar>> = bars
        .iter()
        .map(|b| Py::new(py, PyBar::from(b)).unwrap())
        .collect();
    dict.set_item("bars", py_bars)?;

    Ok(dict.into())
}

/// Load previously saved backtest results from a JSON file.
///
/// This allows you to reload results that were saved with `results.save()`.
/// Note: Loaded results cannot use `validate()` since the original data/signal
/// are not stored in the JSON file. Use `mt.validate(data, signal)` instead.
///
/// Args:
///     path: Path to the JSON file created by `results.save()`
///
/// Returns:
///     BacktestResult object with the loaded metrics.
///
/// Raises:
///     FileNotFoundError: If the file does not exist
///     ValueError: If the file is not valid JSON or missing required fields
///
/// Example:
///     >>> results = mt.backtest(data, signal)
///     >>> results.save("experiment.json")
///     >>> # Later...
///     >>> loaded = mt.load_results("experiment.json")
///     >>> print(loaded.sharpe)
///     1.24
#[pyfunction]
pub fn load_results(path: &str) -> PyResult<super::results::PyBacktestResult> {
    use crate::export::PerformanceSummary;
    use std::fs;

    // Read and parse the JSON file
    let content = fs::read_to_string(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "Results file not found: {}",
                path
            ))
        } else {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Failed to read results file '{}': {}",
                path, e
            ))
        }
    })?;

    let summary: PerformanceSummary = serde_json::from_str(&content).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid results file '{}': {}.\n\n\
             Make sure this file was created with results.save().",
            path, e
        ))
    })?;

    // Convert to PyBacktestResult
    Ok(super::results::PyBacktestResult::from_summary(&summary))
}

/// Adjust price data for stock splits and dividends.
///
/// This function modifies the input data dictionary in place to apply
/// split and dividend adjustments. This is essential for accurate backtesting
/// when using historical price data that includes corporate actions.
///
/// Args:
///     data: Data dictionary from load() containing numpy arrays
///     splits: Optional list of split dictionaries with keys:
///         - 'date': Unix timestamp or ISO date string of ex-split date
///         - 'ratio': Split ratio (e.g., 2.0 for 2-for-1 split)
///         - 'reverse': Optional boolean, True for reverse split (default: False)
///     dividends: Optional list of dividend dictionaries with keys:
///         - 'date': Unix timestamp or ISO date string of ex-dividend date
///         - 'amount': Dividend amount per share
///         - 'type': Optional dividend type ('cash', 'stock', 'special')
///     method: Dividend adjustment method:
///         - 'proportional' (default): Standard method (price * (1 - div/close))
///         - 'absolute': Subtract dividend amount directly
///         - 'none': No dividend adjustment (only apply splits)
///
/// Returns:
///     Dictionary with adjusted numpy arrays for open, high, low, close, volume.
///
/// Example:
///     >>> data = mt.load("AAPL.csv")
///     >>> # Apply a 4-for-1 split that occurred on 2020-08-31
///     >>> splits = [{'date': '2020-08-31', 'ratio': 4.0}]
///     >>> adjusted = mt.adjust(data, splits=splits)
///     >>> # Use adjusted data for backtesting
///     >>> results = mt.backtest(adjusted, signal)
///
///     >>> # Apply both splits and dividends
///     >>> dividends = [
///     ...     {'date': '2024-05-10', 'amount': 0.25},
///     ...     {'date': '2024-02-09', 'amount': 0.24},
///     ... ]
///     >>> adjusted = mt.adjust(data, splits=splits, dividends=dividends)
#[pyfunction]
#[pyo3(signature = (data, splits=None, dividends=None, method="proportional"))]
pub fn adjust(
    py: Python<'_>,
    data: &Bound<'_, PyDict>,
    splits: Option<&Bound<'_, PyList>>,
    dividends: Option<&Bound<'_, PyList>>,
    method: &str,
) -> PyResult<PyObject> {
    // Extract bars from data dictionary
    let timestamps: Vec<i64> = data
        .get_item("timestamp")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'timestamp' key"))?
        .extract()?;
    let opens: Vec<f64> = data
        .get_item("open")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'open' key"))?
        .extract()?;
    let highs: Vec<f64> = data
        .get_item("high")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'high' key"))?
        .extract()?;
    let lows: Vec<f64> = data
        .get_item("low")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'low' key"))?
        .extract()?;
    let closes: Vec<f64> = data
        .get_item("close")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'close' key"))?
        .extract()?;
    let volumes: Vec<f64> = data
        .get_item("volume")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'volume' key"))?
        .extract()?;

    // Create mutable bars
    let bars_result: Result<Vec<Bar>, _> = timestamps
        .into_iter()
        .zip(opens)
        .zip(highs)
        .zip(lows)
        .zip(closes)
        .zip(volumes)
        .enumerate()
        .map(|(i, (((((ts, o), h), l), c), v))| {
            let timestamp = chrono::Utc.timestamp_opt(ts, 0).single().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid timestamp at row {}: {} (Unix timestamp out of valid range)",
                    i, ts
                ))
            })?;
            Ok(Bar {
                timestamp,
                open: o,
                high: h,
                low: l,
                close: c,
                volume: v,
            })
        })
        .collect();
    let mut bars = bars_result?;

    // Parse and apply splits
    if let Some(splits_list) = splits {
        let split_actions = parse_splits(splits_list)?;
        if !split_actions.is_empty() {
            adjust_for_splits(&mut bars, &split_actions);
        }
    }

    // Parse and apply dividends
    if let Some(dividends_list) = dividends {
        let dividend_actions = parse_dividends(dividends_list)?;
        if !dividend_actions.is_empty() {
            let adjust_method = match method.to_lowercase().as_str() {
                "proportional" => DividendAdjustMethod::Proportional,
                "absolute" => DividendAdjustMethod::Absolute,
                "none" => DividendAdjustMethod::None,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid dividend adjustment method: '{}'. Use 'proportional', 'absolute', or 'none'.",
                        method
                    )))
                }
            };
            adjust_for_dividends(&mut bars, &dividend_actions, adjust_method);
        }
    }

    // Create result dictionary with adjusted data
    let result = PyDict::new_bound(py);

    let adj_timestamps: Vec<i64> = bars.iter().map(|b| b.timestamp.timestamp()).collect();
    let adj_opens: Vec<f64> = bars.iter().map(|b| b.open).collect();
    let adj_highs: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let adj_lows: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let adj_closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let adj_volumes: Vec<f64> = bars.iter().map(|b| b.volume).collect();

    result.set_item("timestamp", PyArray1::from_vec_bound(py, adj_timestamps))?;
    result.set_item("open", PyArray1::from_vec_bound(py, adj_opens))?;
    result.set_item("high", PyArray1::from_vec_bound(py, adj_highs))?;
    result.set_item("low", PyArray1::from_vec_bound(py, adj_lows))?;
    result.set_item("close", PyArray1::from_vec_bound(py, adj_closes))?;
    result.set_item("volume", PyArray1::from_vec_bound(py, adj_volumes))?;
    result.set_item("n_bars", bars.len())?;

    // Copy through any additional metadata from original data
    for key in ["path", "sample_name", "symbol"] {
        if let Some(value) = data.get_item(key)? {
            result.set_item(key, value)?;
        }
    }

    // Add bars as list of PyBar objects
    let py_bars: Vec<Py<PyBar>> = bars
        .iter()
        .map(|b| Py::new(py, PyBar::from(b)).unwrap())
        .collect();
    result.set_item("bars", py_bars)?;

    Ok(result.into())
}

/// Parse splits from Python list of dictionaries.
fn parse_splits(splits_list: &Bound<'_, PyList>) -> PyResult<Vec<CorporateAction>> {
    let mut actions = Vec::new();

    for item in splits_list.iter() {
        let dict: &Bound<'_, PyDict> = item.downcast()?;

        // Parse date
        let ex_date = parse_date_from_dict(dict, "date")?;

        // Parse ratio
        let ratio: f64 = dict
            .get_item("ratio")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Split missing 'ratio' key"))?
            .extract()?;

        if ratio <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid split ratio: {}. Must be > 0.",
                ratio
            )));
        }

        // Check if it's a reverse split
        let is_reverse: bool = dict
            .get_item("reverse")?
            .map(|v| v.extract().unwrap_or(false))
            .unwrap_or(false);

        let action_type = if is_reverse {
            CorporateActionType::ReverseSplit { ratio }
        } else {
            CorporateActionType::Split { ratio }
        };

        actions.push(CorporateAction {
            symbol: String::new(), // Not needed for adjustment
            action_type,
            ex_date,
            record_date: None,
            pay_date: None,
        });
    }

    Ok(actions)
}

/// Parse dividends from Python list of dictionaries.
fn parse_dividends(dividends_list: &Bound<'_, PyList>) -> PyResult<Vec<CorporateAction>> {
    let mut actions = Vec::new();

    for item in dividends_list.iter() {
        let dict: &Bound<'_, PyDict> = item.downcast()?;

        // Parse date
        let ex_date = parse_date_from_dict(dict, "date")?;

        // Parse amount
        let amount: f64 = dict
            .get_item("amount")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Dividend missing 'amount' key"))?
            .extract()?;

        if amount <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid dividend amount: {}. Must be > 0.",
                amount
            )));
        }

        // Parse dividend type (optional, default to Cash)
        let div_type = if let Some(type_value) = dict.get_item("type")? {
            let type_str: String = type_value.extract()?;
            match type_str.to_lowercase().as_str() {
                "cash" => DividendType::Cash,
                "stock" => DividendType::Stock,
                "special" => DividendType::Special,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid dividend type: '{}'. Use 'cash', 'stock', or 'special'.",
                        type_str
                    )))
                }
            }
        } else {
            DividendType::Cash
        };

        actions.push(CorporateAction {
            symbol: String::new(), // Not needed for adjustment
            action_type: CorporateActionType::Dividend { amount, div_type },
            ex_date,
            record_date: None,
            pay_date: None,
        });
    }

    Ok(actions)
}

/// Parse a date from a dictionary value (supports Unix timestamp or ISO date string).
fn parse_date_from_dict(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<chrono::DateTime<chrono::Utc>> {
    let value = dict
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("Missing '{}' key", key)))?;

    // Try as integer (Unix timestamp)
    if let Ok(ts) = value.extract::<i64>() {
        return chrono::Utc.timestamp_opt(ts, 0).single().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid Unix timestamp: {}", ts))
        });
    }

    // Try as string (ISO date or date string)
    if let Ok(date_str) = value.extract::<String>() {
        // Try ISO 8601 format first (YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD)
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&date_str) {
            return Ok(dt.with_timezone(&chrono::Utc));
        }

        // Try simple date format (YYYY-MM-DD)
        if let Ok(naive_date) = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d") {
            return Ok(chrono::Utc.from_utc_datetime(&naive_date.and_hms_opt(0, 0, 0).unwrap()));
        }

        // Try US date format (MM/DD/YYYY)
        if let Ok(naive_date) = chrono::NaiveDate::parse_from_str(&date_str, "%m/%d/%Y") {
            return Ok(chrono::Utc.from_utc_datetime(&naive_date.and_hms_opt(0, 0, 0).unwrap()));
        }

        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Could not parse date string: '{}'. Use YYYY-MM-DD, ISO 8601, or Unix timestamp.",
            date_str
        )));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "'{}' must be a Unix timestamp (int) or date string",
        key
    )))
}
