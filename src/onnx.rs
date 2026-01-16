//! ONNX Model Inference
//!
//! This module provides high-performance ONNX model inference for machine learning strategies.
//!
//! # Features
//!
//! - Sub-millisecond inference latency (<1ms per bar target)
//! - Batch inference for parallel backtests
//! - Model versioning and metadata tracking
//! - Automatic fallback on inference errors
//! - Feature normalization and preprocessing
//! - Support for PyTorch, TensorFlow, and JAX exported models
//!
//! # Example
//!
//! ```no_run
//! use mantis::onnx::{OnnxModel, ModelConfig};
//!
//! // Load ONNX model
//! let config = ModelConfig::default();
//! let mut model = OnnxModel::from_file("model.onnx", config).unwrap();
//!
//! // Perform inference
//! let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
//! let prediction = model.predict(&features).unwrap();
//! println!("Prediction: {}", prediction);
//! ```

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Once;
use std::time::Instant;
use tracing::{debug, warn};

// Global initialization for ort runtime (done once per process)
static ORT_INIT: Once = Once::new();

fn ensure_ort_initialized() {
    ORT_INIT.call_once(|| {
        // ort 2.0: init() returns a builder, commit() returns ()
        let _ = ort::init().with_name("mantis").commit();
    });
}

/// Configuration for ONNX model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name/identifier.
    pub name: String,

    /// Model version (semantic versioning recommended).
    pub version: String,

    /// Number of input features expected by the model.
    pub input_size: usize,

    /// Number of outputs produced by the model.
    pub output_size: usize,

    /// Whether to normalize inputs (z-score normalization).
    pub normalize_inputs: bool,

    /// Feature means for normalization (if normalize_inputs = true).
    pub feature_means: Option<Vec<f32>>,

    /// Feature standard deviations for normalization.
    pub feature_stds: Option<Vec<f32>>,

    /// Fallback value if inference fails.
    pub fallback_value: f32,

    /// Enable verbose logging of inference times.
    pub log_latency: bool,

    /// Batch size for batch inference (0 = dynamic).
    pub batch_size: usize,

    /// Use CUDA if available.
    pub use_cuda: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "unnamed_model".to_string(),
            version: "1.0.0".to_string(),
            input_size: 0, // Must be set based on model
            output_size: 1,
            normalize_inputs: false,
            feature_means: None,
            feature_stds: None,
            fallback_value: 0.0,
            log_latency: false,
            batch_size: 1,
            use_cuda: false,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration.
    pub fn new(name: impl Into<String>, input_size: usize) -> Self {
        Self {
            name: name.into(),
            input_size,
            ..Default::default()
        }
    }

    /// Set model version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Enable input normalization with given statistics.
    pub fn with_normalization(mut self, means: Vec<f32>, stds: Vec<f32>) -> Self {
        assert_eq!(
            means.len(),
            self.input_size,
            "Means length must match input_size"
        );
        assert_eq!(
            stds.len(),
            self.input_size,
            "Stds length must match input_size"
        );
        self.normalize_inputs = true;
        self.feature_means = Some(means);
        self.feature_stds = Some(stds);
        self
    }

    /// Set fallback value for failed inferences.
    pub fn with_fallback(mut self, value: f32) -> Self {
        self.fallback_value = value;
        self
    }

    /// Enable latency logging.
    pub fn with_latency_logging(mut self) -> Self {
        self.log_latency = true;
        self
    }

    /// Set batch size for batch inference.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable CUDA execution if available.
    pub fn with_cuda(mut self) -> Self {
        self.use_cuda = true;
        self
    }
}

/// Statistics tracked during model inference.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceStats {
    /// Total number of inferences performed.
    pub total_inferences: u64,

    /// Number of successful inferences.
    pub successful_inferences: u64,

    /// Number of failed inferences (fallback used).
    pub failed_inferences: u64,

    /// Total inference time (microseconds).
    pub total_inference_time_us: u64,

    /// Minimum inference time (microseconds).
    pub min_inference_time_us: u64,

    /// Maximum inference time (microseconds).
    pub max_inference_time_us: u64,
}

impl InferenceStats {
    /// Get average inference time in microseconds.
    pub fn avg_inference_time_us(&self) -> f64 {
        if self.successful_inferences == 0 {
            0.0
        } else {
            self.total_inference_time_us as f64 / self.successful_inferences as f64
        }
    }

    /// Get success rate as a percentage.
    pub fn success_rate(&self) -> f64 {
        if self.total_inferences == 0 {
            0.0
        } else {
            (self.successful_inferences as f64 / self.total_inferences as f64) * 100.0
        }
    }

    /// Record a successful inference.
    fn record_success(&mut self, duration_us: u64) {
        self.total_inferences += 1;
        self.successful_inferences += 1;
        self.total_inference_time_us += duration_us;

        if self.min_inference_time_us == 0 || duration_us < self.min_inference_time_us {
            self.min_inference_time_us = duration_us;
        }

        if duration_us > self.max_inference_time_us {
            self.max_inference_time_us = duration_us;
        }
    }

    /// Record a failed inference.
    fn record_failure(&mut self) {
        self.total_inferences += 1;
        self.failed_inferences += 1;
    }
}

/// ONNX model wrapper for inference.
pub struct OnnxModel {
    /// ONNX Runtime session.
    session: Session,

    /// Model configuration.
    config: ModelConfig,

    /// Inference statistics.
    stats: InferenceStats,
}

impl OnnxModel {
    /// Load an ONNX model from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .onnx model file
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// Result containing the loaded model or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mantis::onnx::{OnnxModel, ModelConfig};
    ///
    /// let config = ModelConfig::new("my_model", 10);
    /// let model = OnnxModel::from_file("model.onnx", config).unwrap();
    /// ```
    pub fn from_file(path: impl AsRef<Path>, config: ModelConfig) -> Result<Self> {
        let path = path.as_ref();

        debug!(
            "Loading ONNX model: {} (version: {})",
            config.name, config.version
        );

        // Ensure ONNX Runtime is initialized (global, done once)
        ensure_ort_initialized();

        // Build session with ort 2.0 API
        let session_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?;

        // CUDA support: Currently disabled as it requires additional setup
        // To enable CUDA, add the ort 'cuda' feature and configure execution providers
        if config.use_cuda {
            warn!("CUDA requested but CUDA execution provider not enabled in this build. Using CPU.");
        }

        // Load model from file (ort 2.0: commit_from_file instead of with_model_from_file)
        let session = session_builder
            .commit_from_file(path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", path))?;

        debug!("ONNX model loaded successfully");

        Ok(Self {
            session,
            config,
            stats: InferenceStats::default(),
        })
    }

    /// Perform inference on a single feature vector.
    ///
    /// # Arguments
    ///
    /// * `features` - Input feature vector
    ///
    /// # Returns
    ///
    /// Predicted value (or fallback value if inference fails)
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mantis::onnx::{OnnxModel, ModelConfig};
    /// # let mut model = OnnxModel::from_file("model.onnx", ModelConfig::default()).unwrap();
    /// let features = vec![0.1, 0.2, 0.3];
    /// let prediction = model.predict(&features).unwrap();
    /// ```
    pub fn predict(&mut self, features: &[f64]) -> Result<f32> {
        let start = Instant::now();

        // Validate input size
        if features.len() != self.config.input_size {
            return Err(anyhow::anyhow!(
                "Input size mismatch: expected {}, got {}",
                self.config.input_size,
                features.len()
            ));
        }

        // Convert to f32 and normalize if needed
        let mut input: Vec<f32> = features.iter().map(|&x| x as f32).collect();

        if self.config.normalize_inputs {
            if let (Some(means), Some(stds)) =
                (&self.config.feature_means, &self.config.feature_stds)
            {
                for i in 0..input.len() {
                    input[i] = (input[i] - means[i]) / stds[i];
                }
            }
        }

        // Run inference
        let result = self.run_inference_internal(&input);

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        match result {
            Ok(output) => {
                self.stats.record_success(duration_us);

                if self.config.log_latency {
                    debug!(
                        "Inference completed in {}μs ({}ms)",
                        duration_us,
                        duration.as_secs_f64() * 1000.0
                    );
                }

                Ok(output)
            }
            Err(e) => {
                self.stats.record_failure();
                warn!(
                    "Inference failed, using fallback value ({}): {}",
                    self.config.fallback_value, e
                );
                Ok(self.config.fallback_value)
            }
        }
    }

    /// Perform batch inference on multiple feature vectors.
    ///
    /// # Arguments
    ///
    /// * `batch_features` - Slice of feature vectors
    ///
    /// # Returns
    ///
    /// Vector of predictions
    pub fn predict_batch(&mut self, batch_features: &[Vec<f64>]) -> Result<Vec<f32>> {
        if batch_features.is_empty() {
            return Ok(Vec::new());
        }

        let mut predictions = Vec::with_capacity(batch_features.len());

        for features in batch_features {
            let pred = self.predict(features)?;
            predictions.push(pred);
        }

        Ok(predictions)
    }

    /// Internal inference method.
    fn run_inference_internal(&mut self, input: &[f32]) -> Result<f32> {
        // Create input tensor (shape: [1, input_size] for batch size 1)
        // ort 2.0 API uses (shape, data) tuple for OwnedTensorArrayData
        let shape = vec![1usize, input.len()];
        let data = input.to_vec();

        // Create tensor from shape and data using ort 2.0 API
        let input_tensor = ort::value::Tensor::<f32>::from_array((shape, data))?;

        // Run inference with ort 2.0 API
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // Extract output with ort 2.0 API
        // try_extract_tensor returns (shape, data_slice) tuple
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        // Get first output value
        let prediction = data
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;

        Ok(prediction)
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get inference statistics.
    pub fn stats(&self) -> &InferenceStats {
        &self.stats
    }

    /// Reset inference statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InferenceStats::default();
    }

    /// Print inference statistics.
    pub fn print_stats(&self) {
        println!("=== ONNX Model Inference Statistics ===");
        println!("Model: {} (v{})", self.config.name, self.config.version);
        println!("Total inferences: {}", self.stats.total_inferences);
        println!("Success rate: {:.2}%", self.stats.success_rate());
        println!(
            "Average latency: {:.2}μs ({:.3}ms)",
            self.stats.avg_inference_time_us(),
            self.stats.avg_inference_time_us() / 1000.0
        );
        println!(
            "Latency range: {}μs - {}μs",
            self.stats.min_inference_time_us, self.stats.max_inference_time_us
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_defaults() {
        let config = ModelConfig::default();
        assert_eq!(config.name, "unnamed_model");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.output_size, 1);
        assert!(!config.normalize_inputs);
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("test_model", 10)
            .with_version("2.0.0")
            .with_fallback(0.5)
            .with_latency_logging();

        assert_eq!(config.name, "test_model");
        assert_eq!(config.version, "2.0.0");
        assert_eq!(config.input_size, 10);
        assert_eq!(config.fallback_value, 0.5);
        assert!(config.log_latency);
    }

    #[test]
    fn test_model_config_normalization() {
        let means = vec![1.0, 2.0, 3.0];
        let stds = vec![0.5, 0.5, 0.5];

        let config = ModelConfig::new("test", 3).with_normalization(means.clone(), stds.clone());

        assert!(config.normalize_inputs);
        assert_eq!(config.feature_means.unwrap(), means);
        assert_eq!(config.feature_stds.unwrap(), stds);
    }

    #[test]
    #[should_panic(expected = "Means length must match input_size")]
    fn test_normalization_length_mismatch() {
        let means = vec![1.0, 2.0];
        let stds = vec![0.5, 0.5, 0.5];

        ModelConfig::new("test", 3).with_normalization(means, stds);
    }

    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::default();

        stats.record_success(500);
        stats.record_success(1000);
        stats.record_success(750);
        stats.record_failure();

        assert_eq!(stats.total_inferences, 4);
        assert_eq!(stats.successful_inferences, 3);
        assert_eq!(stats.failed_inferences, 1);
        assert_eq!(stats.min_inference_time_us, 500);
        assert_eq!(stats.max_inference_time_us, 1000);
        assert_eq!(stats.avg_inference_time_us(), 750.0);
        assert_eq!(stats.success_rate(), 75.0);
    }

    // Note: Full model loading tests require an actual ONNX model file
    // These are better suited for integration tests with sample models
}
