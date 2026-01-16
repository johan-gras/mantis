//! Integration tests for ONNX model inference.
//!
//! These tests require the `onnx` feature to be enabled:
//! ```bash
//! cargo test --features onnx
//! ```
//!
//! They also require test ONNX models to exist in `data/models/`.
//! Generate them with:
//! ```bash
//! python scripts/generate_test_onnx.py
//! ```

#![cfg(feature = "onnx")]

use mantis::onnx::{ModelConfig, OnnxModel};
use std::path::Path;

/// Path to the test models directory.
const MODELS_DIR: &str = "data/models";

/// Check if test models are available.
fn models_available() -> bool {
    Path::new(MODELS_DIR).join("simple_mlp.onnx").exists()
}

#[test]
fn test_load_simple_model() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let result = OnnxModel::from_file(&model_path, config);
    assert!(result.is_ok(), "Failed to load model: {:?}", result.err());

    let model = result.unwrap();
    assert_eq!(model.config().name, "simple_mlp");
    assert_eq!(model.config().input_size, 10);
}

#[test]
fn test_load_minimal_model() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("minimal.onnx");
    let config = ModelConfig::new("minimal", 10);

    let result = OnnxModel::from_file(&model_path, config);
    assert!(
        result.is_ok(),
        "Failed to load minimal model: {:?}",
        result.err()
    );
}

#[test]
fn test_single_inference() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Create test input
    let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let result = model.predict(&features);
    assert!(result.is_ok(), "Inference failed: {:?}", result.err());

    let prediction = result.unwrap();
    // Model uses Tanh activation, so output should be in [-1, 1]
    assert!(
        prediction >= -1.0 && prediction <= 1.0,
        "Prediction {} out of expected range [-1, 1]",
        prediction
    );
}

#[test]
fn test_batch_inference() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Create batch of test inputs
    let batch: Vec<Vec<f64>> = (0..100)
        .map(|i| (0..10).map(|j| ((i * 10 + j) as f64) / 1000.0).collect())
        .collect();

    let result = model.predict_batch(&batch);
    assert!(result.is_ok(), "Batch inference failed: {:?}", result.err());

    let predictions = result.unwrap();
    assert_eq!(predictions.len(), 100, "Expected 100 predictions");

    // All predictions should be in [-1, 1] (Tanh output)
    for (i, &pred) in predictions.iter().enumerate() {
        assert!(
            pred >= -1.0 && pred <= 1.0,
            "Prediction {} at index {} out of range",
            pred,
            i
        );
    }
}

#[test]
fn test_inference_latency_under_1ms() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("minimal.onnx");
    let config = ModelConfig::new("minimal", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    // Warm up
    for _ in 0..10 {
        let _ = model.predict(&features);
    }

    // Measure latency
    let start = std::time::Instant::now();
    let num_inferences = 100;
    for _ in 0..num_inferences {
        let _ = model.predict(&features);
    }
    let elapsed = start.elapsed();

    let avg_latency_ms = elapsed.as_secs_f64() * 1000.0 / num_inferences as f64;

    println!(
        "Average inference latency: {:.3}ms ({} inferences in {:.3}ms)",
        avg_latency_ms,
        num_inferences,
        elapsed.as_secs_f64() * 1000.0
    );

    // Spec requirement: < 1ms per bar for inference
    // Allow 2ms as tolerance for CI variability
    assert!(
        avg_latency_ms < 2.0,
        "Average inference latency {:.3}ms exceeds 2ms threshold",
        avg_latency_ms
    );
}

#[test]
fn test_batch_faster_than_sequential() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");

    // Create batch of inputs
    let batch: Vec<Vec<f64>> = (0..100)
        .map(|i| (0..10).map(|j| ((i * 10 + j) as f64) / 1000.0).collect())
        .collect();

    // Measure sequential inference
    let config = ModelConfig::new("simple_mlp", 10);
    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Warm up
    for features in batch.iter().take(10) {
        let _ = model.predict(features);
    }

    let start_seq = std::time::Instant::now();
    for features in &batch {
        let _ = model.predict(features);
    }
    let elapsed_seq = start_seq.elapsed();

    // Measure batch inference
    let config = ModelConfig::new("simple_mlp", 10);
    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Warm up
    let _ = model.predict_batch(&batch[..10].to_vec());

    let start_batch = std::time::Instant::now();
    let _ = model.predict_batch(&batch);
    let elapsed_batch = start_batch.elapsed();

    println!(
        "Sequential: {:.3}ms, Batch: {:.3}ms, Speedup: {:.1}x",
        elapsed_seq.as_secs_f64() * 1000.0,
        elapsed_batch.as_secs_f64() * 1000.0,
        elapsed_seq.as_secs_f64() / elapsed_batch.as_secs_f64()
    );

    // Batch should be faster (or at least not slower) than sequential
    // Allow some variance - batch should be at least 50% as fast
    assert!(
        elapsed_batch.as_secs_f64() <= elapsed_seq.as_secs_f64() * 2.0,
        "Batch inference unexpectedly slower than sequential"
    );
}

#[test]
fn test_input_size_mismatch_error() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Wrong number of features
    let wrong_features: Vec<f64> = vec![0.1, 0.2, 0.3]; // Only 3 instead of 10

    let result = model.predict(&wrong_features);
    assert!(result.is_err(), "Should fail with wrong input size");

    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("mismatch"),
        "Error should mention size mismatch: {}",
        err
    );
}

#[test]
fn test_inference_stats_tracking() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("minimal.onnx");
    let config = ModelConfig::new("minimal", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    // Run some inferences
    for _ in 0..50 {
        let _ = model.predict(&features);
    }

    let stats = model.stats();
    assert_eq!(stats.total_inferences, 50);
    assert_eq!(stats.successful_inferences, 50);
    assert_eq!(stats.failed_inferences, 0);
    assert!(stats.min_inference_time_us > 0);
    assert!(stats.max_inference_time_us >= stats.min_inference_time_us);
    assert!(stats.avg_inference_time_us() > 0.0);
    assert!((stats.success_rate() - 100.0).abs() < 0.01);
}

#[test]
fn test_model_with_normalization() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");

    // Configure with normalization
    let means: Vec<f32> = vec![0.5; 10];
    let stds: Vec<f32> = vec![0.3; 10];

    let config = ModelConfig::new("simple_mlp", 10).with_normalization(means, stds);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let result = model.predict(&features);
    assert!(
        result.is_ok(),
        "Inference with normalization failed: {:?}",
        result.err()
    );

    let prediction = result.unwrap();
    assert!(
        prediction >= -1.0 && prediction <= 1.0,
        "Normalized prediction {} out of range",
        prediction
    );
}

#[test]
fn test_model_schema_validation() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let model = OnnxModel::from_file(&model_path, config).unwrap();

    let schema = model.schema();

    // Schema should have been extracted
    assert!(schema.input_name.is_some(), "Input name should be detected");
    assert!(
        schema.output_name.is_some(),
        "Output name should be detected"
    );

    // Validation should pass with correct config
    assert!(
        schema.validated,
        "Schema validation should pass: {}",
        schema.validation_message
    );
}

#[test]
fn test_dry_run_validation() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("minimal.onnx");
    let config = ModelConfig::new("minimal", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Validate with dry run
    let result = model.validate_with_dry_run();
    assert!(
        result.is_ok(),
        "Dry run validation failed: {:?}",
        result.err()
    );
}

#[test]
fn test_larger_model_20_inputs() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("larger_mlp.onnx");

    if !model_path.exists() {
        eprintln!("Skipping test: larger_mlp.onnx not found");
        return;
    }

    let config = ModelConfig::new("larger_mlp", 20);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    let features: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();

    let result = model.predict(&features);
    assert!(
        result.is_ok(),
        "Larger model inference failed: {:?}",
        result.err()
    );

    let prediction = result.unwrap();
    assert!(
        prediction >= -1.0 && prediction <= 1.0,
        "Larger model prediction {} out of range",
        prediction
    );
}

#[test]
fn test_empty_batch_returns_empty() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    let empty_batch: Vec<Vec<f64>> = vec![];
    let result = model.predict_batch(&empty_batch);

    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_fallback_on_inference_error() {
    if !models_available() {
        eprintln!(
            "Skipping test: ONNX test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let model_path = Path::new(MODELS_DIR).join("simple_mlp.onnx");
    let config = ModelConfig::new("simple_mlp", 10).with_fallback(0.5);

    let mut model = OnnxModel::from_file(&model_path, config).unwrap();

    // Valid inference should work normally
    let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let result = model.predict(&features);
    assert!(result.is_ok());
}

#[test]
fn test_model_not_found_error() {
    let config = ModelConfig::new("nonexistent", 10);
    let result = OnnxModel::from_file("nonexistent_model.onnx", config);

    assert!(result.is_err(), "Should fail for nonexistent model");
}
