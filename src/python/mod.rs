//! Python bindings for the Mantis backtesting engine.
//!
//! This module provides PyO3-based Python bindings that expose the core
//! functionality of Mantis to Python users, enabling seamless integration
//! with pandas, polars, and numpy workflows.

mod backtest;
mod data;
mod results;
mod sensitivity;
mod types;

use pyo3::prelude::*;

/// Mantis - A high-performance backtesting engine for quantitative trading.
///
/// This module provides the Rust core of the Mantis backtesting engine,
/// exposed to Python via PyO3.
#[pymodule]
fn _mantis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register the version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register data loading functions
    m.add_function(wrap_pyfunction!(data::load, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_multi, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_dir, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_sample, m)?)?;
    m.add_function(wrap_pyfunction!(data::list_samples, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_results, m)?)?;

    // Register main backtest function
    m.add_function(wrap_pyfunction!(backtest::backtest, m)?)?;
    m.add_function(wrap_pyfunction!(backtest::signal_check, m)?)?;
    m.add_function(wrap_pyfunction!(backtest::validate, m)?)?;

    // Register sensitivity analysis functions
    m.add_function(wrap_pyfunction!(sensitivity::sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity::cost_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity::linear_range, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity::log_range, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity::discrete_range, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity::centered_range, m)?)?;

    // Register classes
    m.add_class::<results::PyBacktestResult>()?;
    m.add_class::<results::PyValidationResult>()?;
    m.add_class::<results::PyFoldDetail>()?;
    m.add_class::<types::PyBar>()?;
    m.add_class::<backtest::PyBacktestConfig>()?;

    // Register sensitivity analysis classes
    m.add_class::<sensitivity::PyParameterRange>()?;
    m.add_class::<sensitivity::PySensitivityResult>()?;
    m.add_class::<sensitivity::PySensitivitySummary>()?;
    m.add_class::<sensitivity::PyHeatmapData>()?;
    m.add_class::<sensitivity::PyCliff>()?;
    m.add_class::<sensitivity::PyPlateau>()?;
    m.add_class::<sensitivity::PyCostSensitivityResult>()?;
    m.add_class::<sensitivity::PyCostScenario>()?;

    Ok(())
}
