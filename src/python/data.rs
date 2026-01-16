//! Python bindings for data loading.
//!
//! Provides functions to load data from CSV/Parquet files and convert
//! between Rust data structures and Python pandas/polars DataFrames.

use std::collections::HashMap;
use std::path::Path;

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::data::{
    list_samples as rust_list_samples, load_csv, load_parquet, load_sample as rust_load_sample,
    DataConfig,
};
use crate::types::Bar;

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
