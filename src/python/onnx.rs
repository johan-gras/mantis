//! Python bindings for ONNX model inference.
//!
//! Provides PyO3 wrappers around the Rust ONNX module for loading and running
//! ONNX models from Python.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Configuration for ONNX model inference.
#[pyclass(name = "ModelConfig")]
#[derive(Debug, Clone)]
pub struct PyModelConfig {
    /// Model name/identifier.
    #[pyo3(get, set)]
    pub name: String,

    /// Model version (semantic versioning recommended).
    #[pyo3(get, set)]
    pub version: String,

    /// Number of input features expected by the model.
    #[pyo3(get, set)]
    pub input_size: usize,

    /// Number of outputs produced by the model.
    #[pyo3(get, set)]
    pub output_size: usize,

    /// Whether to normalize inputs (z-score normalization).
    #[pyo3(get, set)]
    pub normalize_inputs: bool,

    /// Feature means for normalization (if normalize_inputs = true).
    pub feature_means: Option<Vec<f32>>,

    /// Feature standard deviations for normalization.
    pub feature_stds: Option<Vec<f32>>,

    /// Fallback value if inference fails.
    #[pyo3(get, set)]
    pub fallback_value: f32,

    /// Enable verbose logging of inference times.
    #[pyo3(get, set)]
    pub log_latency: bool,

    /// Batch size for batch inference (0 = dynamic).
    #[pyo3(get, set)]
    pub batch_size: usize,
}

#[pymethods]
impl PyModelConfig {
    #[new]
    #[pyo3(signature = (
        name = "unnamed_model",
        input_size = 0,
        version = "1.0.0",
        output_size = 1,
        normalize_inputs = false,
        feature_means = None,
        feature_stds = None,
        fallback_value = 0.0,
        log_latency = false,
        batch_size = 1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        input_size: usize,
        version: &str,
        output_size: usize,
        normalize_inputs: bool,
        feature_means: Option<Vec<f32>>,
        feature_stds: Option<Vec<f32>>,
        fallback_value: f32,
        log_latency: bool,
        batch_size: usize,
    ) -> PyResult<Self> {
        // Validate normalization parameters
        if normalize_inputs {
            if let (Some(ref means), Some(ref stds)) = (&feature_means, &feature_stds) {
                if means.len() != input_size {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature_means length ({}) must match input_size ({})",
                        means.len(),
                        input_size
                    )));
                }
                if stds.len() != input_size {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "feature_stds length ({}) must match input_size ({})",
                        stds.len(),
                        input_size
                    )));
                }
            } else if feature_means.is_some() || feature_stds.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Both feature_means and feature_stds must be provided when normalize_inputs=True",
                ));
            }
        }

        Ok(Self {
            name: name.to_string(),
            version: version.to_string(),
            input_size,
            output_size,
            normalize_inputs,
            feature_means,
            feature_stds,
            fallback_value,
            log_latency,
            batch_size,
        })
    }

    #[getter]
    fn get_feature_means(&self) -> Option<Vec<f32>> {
        self.feature_means.clone()
    }

    #[setter]
    fn set_feature_means(&mut self, means: Option<Vec<f32>>) {
        self.feature_means = means;
    }

    #[getter]
    fn get_feature_stds(&self) -> Option<Vec<f32>> {
        self.feature_stds.clone()
    }

    #[setter]
    fn set_feature_stds(&mut self, stds: Option<Vec<f32>>) {
        self.feature_stds = stds;
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelConfig(name='{}', input_size={}, version='{}')",
            self.name, self.input_size, self.version
        )
    }
}

#[cfg(feature = "onnx")]
impl From<&PyModelConfig> for crate::onnx::ModelConfig {
    fn from(py_config: &PyModelConfig) -> Self {
        let mut config = crate::onnx::ModelConfig::new(&py_config.name, py_config.input_size)
            .with_version(&py_config.version)
            .with_fallback(py_config.fallback_value)
            .with_batch_size(py_config.batch_size);

        if py_config.log_latency {
            config = config.with_latency_logging();
        }

        if py_config.normalize_inputs {
            if let (Some(means), Some(stds)) = (&py_config.feature_means, &py_config.feature_stds) {
                config = config.with_normalization(means.clone(), stds.clone());
            }
        }

        config.output_size = py_config.output_size;
        config
    }
}

/// Statistics tracked during model inference.
#[pyclass(name = "InferenceStats")]
#[derive(Debug, Clone)]
pub struct PyInferenceStats {
    /// Total number of inferences performed.
    #[pyo3(get)]
    pub total_inferences: u64,

    /// Number of successful inferences.
    #[pyo3(get)]
    pub successful_inferences: u64,

    /// Number of failed inferences (fallback used).
    #[pyo3(get)]
    pub failed_inferences: u64,

    /// Total inference time (microseconds).
    #[pyo3(get)]
    pub total_inference_time_us: u64,

    /// Minimum inference time (microseconds).
    #[pyo3(get)]
    pub min_inference_time_us: u64,

    /// Maximum inference time (microseconds).
    #[pyo3(get)]
    pub max_inference_time_us: u64,
}

#[pymethods]
impl PyInferenceStats {
    /// Get average inference time in microseconds.
    fn avg_inference_time_us(&self) -> f64 {
        if self.successful_inferences == 0 {
            0.0
        } else {
            self.total_inference_time_us as f64 / self.successful_inferences as f64
        }
    }

    /// Get average inference time in milliseconds.
    fn avg_inference_time_ms(&self) -> f64 {
        self.avg_inference_time_us() / 1000.0
    }

    /// Get success rate as a percentage.
    fn success_rate(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            (self.successful_inferences as f64 / self.total_inferences as f64) * 100.0
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "InferenceStats(total={}, success_rate={:.1}%, avg_latency={:.2}μs)",
            self.total_inferences,
            self.success_rate(),
            self.avg_inference_time_us()
        )
    }
}

#[cfg(feature = "onnx")]
impl From<&crate::onnx::InferenceStats> for PyInferenceStats {
    fn from(stats: &crate::onnx::InferenceStats) -> Self {
        Self {
            total_inferences: stats.total_inferences,
            successful_inferences: stats.successful_inferences,
            failed_inferences: stats.failed_inferences,
            total_inference_time_us: stats.total_inference_time_us,
            min_inference_time_us: stats.min_inference_time_us,
            max_inference_time_us: stats.max_inference_time_us,
        }
    }
}

/// ONNX model wrapper for inference.
///
/// This class provides Python bindings for the Rust ONNX module, enabling
/// high-performance model inference directly from Python.
///
/// Example:
///     >>> model = mt.OnnxModel("model.onnx", input_size=10)
///     >>> prediction = model.predict([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
///     >>> print(prediction)
///     0.75
#[cfg(feature = "onnx")]
#[pyclass(name = "OnnxModel")]
pub struct PyOnnxModel {
    model: crate::onnx::OnnxModel,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyOnnxModel {
    /// Load an ONNX model from a file.
    ///
    /// Args:
    ///     path: Path to the .onnx model file
    ///     input_size: Number of input features (required)
    ///     config: Optional ModelConfig for advanced settings
    ///     name: Model name for logging (default: filename)
    ///     version: Model version (default: "1.0.0")
    ///     normalize: Whether to normalize inputs (requires means/stds)
    ///     fallback_value: Value to return if inference fails (default: 0.0)
    ///
    /// Example:
    ///     >>> model = mt.OnnxModel("model.onnx", input_size=10)
    ///     >>> model = mt.OnnxModel("model.onnx", input_size=10, fallback_value=0.5)
    #[new]
    #[pyo3(signature = (
        path,
        input_size,
        config = None,
        name = None,
        version = "1.0.0",
        normalize = false,
        feature_means = None,
        feature_stds = None,
        fallback_value = 0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: &str,
        input_size: usize,
        config: Option<&PyModelConfig>,
        name: Option<&str>,
        version: &str,
        normalize: bool,
        feature_means: Option<Vec<f32>>,
        feature_stds: Option<Vec<f32>>,
        fallback_value: f32,
    ) -> PyResult<Self> {
        let rust_config = if let Some(cfg) = config {
            crate::onnx::ModelConfig::from(cfg)
        } else {
            let model_name = name.unwrap_or_else(|| {
                std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("model")
            });

            let mut cfg =
                crate::onnx::ModelConfig::new(model_name, input_size).with_version(version);

            if normalize {
                if let (Some(means), Some(stds)) = (feature_means, feature_stds) {
                    cfg = cfg.with_normalization(means, stds);
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "normalize=True requires feature_means and feature_stds",
                    ));
                }
            }

            cfg = cfg.with_fallback(fallback_value);
            cfg
        };

        let model = crate::onnx::OnnxModel::from_file(path, rust_config).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load ONNX model: {}", e))
        })?;

        Ok(Self { model })
    }

    /// Perform inference on a single feature vector.
    ///
    /// Args:
    ///     features: List or numpy array of input features
    ///
    /// Returns:
    ///     Predicted value (float)
    ///
    /// Example:
    ///     >>> features = [0.1, 0.2, 0.3, 0.4, 0.5]
    ///     >>> prediction = model.predict(features)
    fn predict(&mut self, features: Vec<f64>) -> PyResult<f32> {
        self.model.predict(&features).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Inference failed: {}", e))
        })
    }

    /// Perform batch inference on multiple feature vectors.
    ///
    /// Args:
    ///     batch_features: List of feature vectors
    ///
    /// Returns:
    ///     List of predictions
    ///
    /// Example:
    ///     >>> batch = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    ///     >>> predictions = model.predict_batch(batch)
    fn predict_batch(&mut self, batch_features: Vec<Vec<f64>>) -> PyResult<Vec<f32>> {
        self.model.predict_batch(&batch_features).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Batch inference failed: {}", e))
        })
    }

    /// Get inference statistics.
    fn stats(&self) -> PyInferenceStats {
        PyInferenceStats::from(self.model.stats())
    }

    /// Reset inference statistics.
    fn reset_stats(&mut self) {
        self.model.reset_stats();
    }

    /// Print inference statistics.
    fn print_stats(&self) {
        self.model.print_stats();
    }

    /// Get the model's input size.
    #[getter]
    fn input_size(&self) -> usize {
        self.model.config().input_size
    }

    /// Get the model's output size.
    #[getter]
    fn output_size(&self) -> usize {
        self.model.config().output_size
    }

    /// Get the model's name.
    #[getter]
    fn name(&self) -> String {
        self.model.config().name.clone()
    }

    /// Get the model's version.
    #[getter]
    fn version(&self) -> String {
        self.model.config().version.clone()
    }

    fn __repr__(&self) -> String {
        let config = self.model.config();
        format!(
            "OnnxModel(name='{}', version='{}', input_size={}, output_size={})",
            config.name, config.version, config.input_size, config.output_size
        )
    }
}

/// Load an ONNX model from a file.
///
/// This is a convenience function that wraps OnnxModel creation.
///
/// Args:
///     path: Path to the .onnx model file
///     input_size: Number of input features
///     **kwargs: Additional arguments passed to OnnxModel
///
/// Returns:
///     Loaded OnnxModel
///
/// Example:
///     >>> model = mt.load_model("model.onnx", input_size=10)
#[cfg(feature = "onnx")]
#[pyfunction]
#[pyo3(signature = (path, input_size, name = None, version = "1.0.0", fallback_value = 0.0))]
pub fn load_model(
    path: &str,
    input_size: usize,
    name: Option<&str>,
    version: &str,
    fallback_value: f32,
) -> PyResult<PyOnnxModel> {
    PyOnnxModel::new(
        path,
        input_size,
        None,
        name,
        version,
        false,
        None,
        None,
        fallback_value,
    )
}

/// Generate signals from an ONNX model and feature DataFrame.
///
/// This function runs the ONNX model on each row of the feature DataFrame
/// and returns a numpy array of predictions that can be used as signals.
///
/// Args:
///     model: OnnxModel or path to .onnx file
///     features: Feature array or DataFrame with features for each bar
///     threshold: Optional threshold for converting predictions to signals
///                If provided, predictions > threshold become 1, < -threshold become -1
///
/// Returns:
///     numpy array of predictions/signals
///
/// Example:
///     >>> model = mt.load_model("model.onnx", input_size=10)
///     >>> signals = mt.generate_signals(model, feature_df, threshold=0.5)
///     >>> results = mt.backtest(data, signals)
#[cfg(feature = "onnx")]
#[pyfunction]
#[pyo3(signature = (model, features, threshold = None))]
pub fn generate_signals(
    py: Python<'_>,
    model: &mut PyOnnxModel,
    features: PyObject,
    threshold: Option<f64>,
) -> PyResult<PyObject> {
    // Try to extract features as a list of lists (2D)
    let feature_vecs: Vec<Vec<f64>> = if let Ok(rows) = features.extract::<Vec<Vec<f64>>>(py) {
        rows
    } else if let Ok(dict) = features.downcast_bound::<PyDict>(py) {
        // Try to extract from a dict with numpy arrays
        // Assume it's column-oriented, need to transpose
        let columns: Vec<String> = dict.keys().extract()?;
        if columns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Feature dictionary is empty",
            ));
        }

        // Get the length from the first column
        let first_col: Vec<f64> = dict.get_item(&columns[0])?.unwrap().extract()?;
        let n_rows = first_col.len();
        let n_cols = columns.len();

        // Build row-wise features
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row = Vec::with_capacity(n_cols);
            for col in &columns {
                let col_data: Vec<f64> = dict.get_item(col)?.unwrap().extract()?;
                row.push(col_data[i]);
            }
            rows.push(row);
        }
        rows
    } else {
        // Try as pandas DataFrame
        let values = features.call_method0(py, "values")?;
        let rows: Vec<Vec<f64>> = values.call_method0(py, "tolist")?.extract(py)?;
        rows
    };

    // Run batch inference
    let predictions = model.predict_batch(feature_vecs)?;

    // Apply threshold if provided
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
        predictions.into_iter().map(|p| p as f64).collect()
    };

    // Convert to numpy array
    let np = py.import_bound("numpy")?;
    let arr = np.call_method1("array", (signals,))?;
    Ok(arr.into())
}
