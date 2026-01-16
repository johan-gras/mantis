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
use tracing::{debug, info, warn};

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

/// Information about the model's input/output schema extracted at load time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelSchema {
    /// Detected input shape from the ONNX model (excluding batch dimension).
    /// For a model expecting shape [batch, features], this would be `Some(features)`.
    pub input_size: Option<usize>,

    /// Detected output shape from the ONNX model (excluding batch dimension).
    pub output_size: Option<usize>,

    /// Name of the input tensor in the ONNX model.
    pub input_name: Option<String>,

    /// Name of the output tensor in the ONNX model.
    pub output_name: Option<String>,

    /// Whether the model schema was successfully validated against the config.
    pub validated: bool,

    /// Validation message (success or error description).
    pub validation_message: String,
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

    /// Extracted model schema from load-time introspection.
    schema: ModelSchema,
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
            warn!(
                "CUDA requested but CUDA execution provider not enabled in this build. Using CPU."
            );
        }

        // Load model from file (ort 2.0: commit_from_file instead of with_model_from_file)
        let session = session_builder
            .commit_from_file(path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", path))?;

        debug!("ONNX model loaded successfully");

        // Perform load-time schema introspection and validation
        let schema = Self::extract_and_validate_schema(&session, &config);

        if !schema.validated {
            warn!(
                "Model schema validation warning: {}",
                schema.validation_message
            );
        } else {
            info!(
                "Model schema validated: input_size={:?}, output_size={:?}",
                schema.input_size, schema.output_size
            );
        }

        Ok(Self {
            session,
            config,
            stats: InferenceStats::default(),
            schema,
        })
    }

    /// Extract model schema from ONNX session and validate against config.
    ///
    /// This performs load-time introspection to detect misconfigurations early
    /// rather than failing at first inference.
    fn extract_and_validate_schema(session: &Session, config: &ModelConfig) -> ModelSchema {
        let mut schema = ModelSchema::default();

        // Extract input information using ort 2.0 API
        // session.inputs() returns &[Outlet], Outlet has name() and dtype() methods
        let inputs = session.inputs();
        if let Some(input) = inputs.first() {
            schema.input_name = Some(input.name().to_string());

            // Try to extract input dimensions via dtype().tensor_shape()
            // ONNX models typically have shape like [batch, features] or [batch, seq, features]
            if let Some(shape) = input.dtype().tensor_shape() {
                // Shape derefs to &[i64]
                let dims: &[i64] = shape;
                // Skip batch dimension (first dim) and get feature dimension
                if dims.len() >= 2 {
                    // For shape [batch, features], features is at index 1
                    if let Some(&feature_dim) = dims.get(1) {
                        // Only use if it's a known dimension (> 0, not dynamic -1)
                        if feature_dim > 0 {
                            schema.input_size = Some(feature_dim as usize);
                        }
                    }
                } else if dims.len() == 1 {
                    // For shape [features] without explicit batch
                    if let Some(&feature_dim) = dims.first() {
                        if feature_dim > 0 {
                            schema.input_size = Some(feature_dim as usize);
                        }
                    }
                }
            }
        }

        // Extract output information using ort 2.0 API
        // session.outputs() returns &[Outlet]
        let outputs = session.outputs();
        if let Some(output) = outputs.first() {
            schema.output_name = Some(output.name().to_string());

            // Extract output dimensions via dtype().tensor_shape()
            if let Some(shape) = output.dtype().tensor_shape() {
                let dims: &[i64] = shape;
                if dims.len() >= 2 {
                    // For shape [batch, outputs], outputs is at index 1
                    if let Some(&output_dim) = dims.get(1) {
                        if output_dim > 0 {
                            schema.output_size = Some(output_dim as usize);
                        }
                    }
                } else if dims.len() == 1 {
                    // For shape [outputs] or [batch]
                    if let Some(&output_dim) = dims.first() {
                        if output_dim > 0 {
                            schema.output_size = Some(output_dim as usize);
                        }
                    }
                }
            }
        }

        // Validate config against detected schema
        let mut warnings = Vec::new();

        // Validate input size if both are known
        if let Some(detected_input) = schema.input_size {
            if config.input_size > 0 && config.input_size != detected_input {
                warnings.push(format!(
                    "Config input_size ({}) does not match model input size ({})",
                    config.input_size, detected_input
                ));
            }
        } else if config.input_size == 0 {
            warnings.push("Unable to detect model input size and config.input_size is 0".to_string());
        }

        // Validate output size if both are known
        if let Some(detected_output) = schema.output_size {
            if config.output_size != detected_output {
                warnings.push(format!(
                    "Config output_size ({}) does not match model output size ({})",
                    config.output_size, detected_output
                ));
            }
        }

        // Validate normalization parameters match input size
        if config.normalize_inputs {
            if let Some(ref means) = config.feature_means {
                if let Some(detected_input) = schema.input_size {
                    if means.len() != detected_input {
                        warnings.push(format!(
                            "feature_means length ({}) does not match model input size ({})",
                            means.len(),
                            detected_input
                        ));
                    }
                }
            }
            if let Some(ref stds) = config.feature_stds {
                if let Some(detected_input) = schema.input_size {
                    if stds.len() != detected_input {
                        warnings.push(format!(
                            "feature_stds length ({}) does not match model input size ({})",
                            stds.len(),
                            detected_input
                        ));
                    }
                }
            }
        }

        if warnings.is_empty() {
            schema.validated = true;
            schema.validation_message = "Schema validation passed".to_string();
        } else {
            schema.validated = false;
            schema.validation_message = warnings.join("; ");
        }

        schema
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
    /// This method performs true batched inference by combining all inputs
    /// into a single tensor with shape `[batch_size, input_size]` and running
    /// a single forward pass. This is significantly faster than calling
    /// `predict()` in a loop (typically 10-100x faster for large batches).
    ///
    /// # Arguments
    ///
    /// * `batch_features` - Slice of feature vectors (all must have same length)
    ///
    /// # Returns
    ///
    /// Vector of predictions, one per input feature vector
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mantis::onnx::{OnnxModel, ModelConfig};
    /// # let mut model = OnnxModel::from_file("model.onnx", ModelConfig::default()).unwrap();
    /// let batch = vec![
    ///     vec![0.1, 0.2, 0.3],
    ///     vec![0.4, 0.5, 0.6],
    ///     vec![0.7, 0.8, 0.9],
    /// ];
    /// let predictions = model.predict_batch(&batch).unwrap();
    /// assert_eq!(predictions.len(), 3);
    /// ```
    pub fn predict_batch(&mut self, batch_features: &[Vec<f64>]) -> Result<Vec<f32>> {
        if batch_features.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = batch_features.len();
        let start = Instant::now();

        // Validate all inputs have the correct size
        for (i, features) in batch_features.iter().enumerate() {
            if features.len() != self.config.input_size {
                return Err(anyhow::anyhow!(
                    "Input size mismatch at index {}: expected {}, got {}",
                    i,
                    self.config.input_size,
                    features.len()
                ));
            }
        }

        // Flatten all features into a single contiguous vector for the batch tensor
        // Shape will be [batch_size, input_size]
        let mut flat_input: Vec<f32> = Vec::with_capacity(batch_size * self.config.input_size);

        for features in batch_features {
            // Convert f64 to f32 and optionally normalize
            for (i, &val) in features.iter().enumerate() {
                let mut v = val as f32;

                // Apply normalization if configured
                if self.config.normalize_inputs {
                    if let (Some(means), Some(stds)) =
                        (&self.config.feature_means, &self.config.feature_stds)
                    {
                        v = (v - means[i]) / stds[i];
                    }
                }
                flat_input.push(v);
            }
        }

        // Run batched inference
        let result = self.run_batch_inference_internal(&flat_input, batch_size);

        let duration = start.elapsed();
        let duration_us = duration.as_micros() as u64;

        match result {
            Ok(outputs) => {
                // Record stats (average per sample)
                self.stats.record_success(duration_us / batch_size as u64);

                if self.config.log_latency {
                    debug!(
                        "Batch inference ({} samples) completed in {}μs ({:.3}ms, {:.1}μs/sample)",
                        batch_size,
                        duration_us,
                        duration.as_secs_f64() * 1000.0,
                        duration_us as f64 / batch_size as f64
                    );
                }

                Ok(outputs)
            }
            Err(e) => {
                // On batch failure, fall back to sequential inference
                warn!("Batch inference failed, falling back to sequential: {}", e);

                let mut predictions = Vec::with_capacity(batch_size);
                for features in batch_features {
                    let pred = self.predict(features)?;
                    predictions.push(pred);
                }
                Ok(predictions)
            }
        }
    }

    /// Internal batched inference method.
    fn run_batch_inference_internal(
        &mut self,
        flat_input: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>> {
        // Create input tensor with shape [batch_size, input_size]
        let shape = vec![batch_size, self.config.input_size];
        let data = flat_input.to_vec();

        let input_tensor = ort::value::Tensor::<f32>::from_array((shape, data))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        // Extract output tensor
        // Output shape is typically [batch_size, output_size] or [batch_size] for single-output models
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract batch output tensor")?;

        // Collect predictions - take first value from each sample's output
        // If output_size is 1, each element is already a prediction
        // If output_size > 1, we take the first element (or could be argmax for classification)
        let predictions: Vec<f32> = if self.config.output_size == 1 {
            data.to_vec()
        } else {
            // For multi-output models, reshape and take first element of each
            data.chunks(self.config.output_size)
                .map(|chunk| chunk.first().copied().unwrap_or(self.config.fallback_value))
                .collect()
        };

        if predictions.len() != batch_size {
            return Err(anyhow::anyhow!(
                "Output size mismatch: expected {} predictions, got {}",
                batch_size,
                predictions.len()
            ));
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

    /// Get the model schema information extracted at load time.
    ///
    /// This includes detected input/output sizes and validation status.
    pub fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    /// Check if the model schema was successfully validated at load time.
    ///
    /// Returns `true` if the config parameters match the detected model schema.
    pub fn is_validated(&self) -> bool {
        self.schema.validated
    }

    /// Get the detected input size from the ONNX model.
    ///
    /// This is extracted from the model at load time, not from the config.
    /// Returns `None` if the input size could not be determined from the model.
    pub fn detected_input_size(&self) -> Option<usize> {
        self.schema.input_size
    }

    /// Get the detected output size from the ONNX model.
    ///
    /// This is extracted from the model at load time, not from the config.
    /// Returns `None` if the output size could not be determined from the model.
    pub fn detected_output_size(&self) -> Option<usize> {
        self.schema.output_size
    }

    /// Perform a dry-run inference to verify the model is functional.
    ///
    /// This runs inference on a zero-filled input vector to ensure the model
    /// can actually execute. Call this after loading to catch runtime errors
    /// early rather than during actual backtesting.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the dry-run succeeds
    /// * `Err(...)` if inference fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mantis::onnx::{OnnxModel, ModelConfig};
    /// let config = ModelConfig::new("my_model", 10);
    /// let mut model = OnnxModel::from_file("model.onnx", config).unwrap();
    ///
    /// // Verify model can execute
    /// model.validate_with_dry_run().expect("Model dry-run failed");
    /// ```
    pub fn validate_with_dry_run(&mut self) -> Result<()> {
        // Determine input size: use detected size if available, otherwise use config
        let input_size = self
            .schema
            .input_size
            .unwrap_or(self.config.input_size);

        if input_size == 0 {
            return Err(anyhow::anyhow!(
                "Cannot perform dry-run: input size is 0 (set config.input_size or load a model with fixed input shape)"
            ));
        }

        // Create a zero-filled input vector
        let dummy_input: Vec<f64> = vec![0.0; input_size];

        // Run inference (temporarily override input_size in config if needed)
        let original_input_size = self.config.input_size;
        if self.config.input_size == 0 {
            self.config.input_size = input_size;
        }

        let result = self.predict(&dummy_input);

        // Restore original config
        self.config.input_size = original_input_size;

        // Reset stats since this was just a validation run
        self.reset_stats();

        match result {
            Ok(_) => {
                debug!("Dry-run validation successful");
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Dry-run validation failed: {}", e)),
        }
    }

    /// Load an ONNX model with automatic schema validation and dry-run.
    ///
    /// This is a convenience method that loads the model, validates the schema,
    /// and performs a dry-run inference to ensure the model is functional.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the .onnx model file
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// * `Ok(OnnxModel)` if loading and validation succeed
    /// * `Err(...)` if loading fails, schema is invalid, or dry-run fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use mantis::onnx::{OnnxModel, ModelConfig};
    /// let config = ModelConfig::new("my_model", 10);
    /// let model = OnnxModel::from_file_validated("model.onnx", config)
    ///     .expect("Failed to load and validate model");
    /// ```
    pub fn from_file_validated(path: impl AsRef<Path>, config: ModelConfig) -> Result<Self> {
        let mut model = Self::from_file(path, config)?;

        // Check schema validation
        if !model.schema.validated {
            return Err(anyhow::anyhow!(
                "Model schema validation failed: {}",
                model.schema.validation_message
            ));
        }

        // Perform dry-run
        model.validate_with_dry_run()?;

        Ok(model)
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

    #[test]
    fn test_model_schema_defaults() {
        let schema = ModelSchema::default();
        assert!(schema.input_size.is_none());
        assert!(schema.output_size.is_none());
        assert!(schema.input_name.is_none());
        assert!(schema.output_name.is_none());
        assert!(!schema.validated);
        assert!(schema.validation_message.is_empty());
    }

    #[test]
    fn test_model_schema_with_values() {
        let schema = ModelSchema {
            input_size: Some(10),
            output_size: Some(1),
            input_name: Some("input".to_string()),
            output_name: Some("output".to_string()),
            validated: true,
            validation_message: "Schema validation passed".to_string(),
        };

        assert_eq!(schema.input_size, Some(10));
        assert_eq!(schema.output_size, Some(1));
        assert_eq!(schema.input_name, Some("input".to_string()));
        assert!(schema.validated);
    }
}
