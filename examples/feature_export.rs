//! Example: Feature Export for Deep Learning
//!
//! This example shows how to extract features and export them in formats
//! suitable for training deep learning models in Python (PyTorch, TensorFlow, JAX).
//!
//! The workflow:
//! 1. Load price data
//! 2. Configure feature extraction
//! 3. Export to CSV (for pandas/polars)
//! 4. Export sequences to JSON (for LSTM/Transformer)
//!
//! Run with: cargo run --example feature_export

use ralph_backtest::data::{load_csv, DataConfig};
use ralph_backtest::features::{FeatureConfig, FeatureExtractor, SequenceBuilder, TimeSeriesSplitter};
use ralph_backtest::types::Bar;
use chrono::{TimeZone, Utc};
use std::fs;

/// Generate synthetic data for demonstration.
fn generate_synthetic_data(days: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(days);
    let mut price = 100.0;

    for i in 0..days {
        let trend = 0.0003;
        let noise = ((i as f64 * 0.7).sin() * 2.0 + (i as f64 * 1.3).cos()) * 0.5;
        let daily_change = price * trend + noise;
        price += daily_change;
        price = price.max(50.0);

        let volatility = 1.0 + (noise.abs() * 0.3);
        let open = price - volatility * 0.3;
        let high = price + volatility;
        let low = price - volatility;
        let close = price;
        let volume = 1_000_000.0 + (noise.abs() * 500000.0);

        bars.push(Bar::new(
            Utc.with_ymd_and_hms(2022, 1, 1, 0, 0, 0).unwrap()
                + chrono::Duration::days(i as i64),
            open,
            high.max(open).max(close),
            low.min(open).min(close),
            close,
            volume.abs(),
        ));
    }

    bars
}

fn main() {
    println!("=== Feature Export for Deep Learning ===\n");

    // Create output directory
    let output_dir = "ml_data";
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // 1. Load data
    println!("1. Loading data...");
    let bars = if std::path::Path::new("data/sample.csv").exists() {
        load_csv("data/sample.csv", &DataConfig::default())
            .expect("Failed to load data")
    } else {
        println!("   Using synthetic data");
        generate_synthetic_data(1000)
    };
    println!("   Loaded {} bars\n", bars.len());

    // 2. Configure feature extraction
    println!("2. Configuring feature extraction...");

    // Comprehensive features for deep learning
    let config = FeatureConfig {
        return_periods: vec![1, 2, 3, 5, 10, 20],
        ma_periods: vec![5, 10, 20, 50],
        rsi_period: 14,
        macd_params: (12, 26, 9),
        atr_period: 14,
        bb_params: (20, 2.0),
        include_time_features: true,
        include_volume_features: true,
        normalize: true,
        normalize_window: 252,
    };

    let extractor = FeatureExtractor::new(config);
    println!("   Warmup period: {} bars", extractor.warmup_period());

    // 3. Split data
    println!("\n3. Splitting data (70/15/15)...");
    let splitter = TimeSeriesSplitter::new(0.7, 0.15).with_gap(5);
    let (train_bars, val_bars, test_bars) = splitter.split(&bars);
    println!("   Train: {} bars", train_bars.len());
    println!("   Validation: {} bars", val_bars.len());
    println!("   Test: {} bars", test_bars.len());

    // 4. Export training data as CSV
    println!("\n4. Exporting training data to CSV...");
    let target_horizon = 5; // Predict 5-day forward returns

    let train_csv = extractor.to_csv(&train_bars, Some(target_horizon));
    let train_path = format!("{}/train.csv", output_dir);
    fs::write(&train_path, &train_csv).expect("Failed to write train CSV");
    println!("   Saved: {} ({} bytes)", train_path, train_csv.len());

    let val_csv = extractor.to_csv(&val_bars, Some(target_horizon));
    let val_path = format!("{}/validation.csv", output_dir);
    fs::write(&val_path, &val_csv).expect("Failed to write validation CSV");
    println!("   Saved: {} ({} bytes)", val_path, val_csv.len());

    let test_csv = extractor.to_csv(&test_bars, Some(target_horizon));
    let test_path = format!("{}/test.csv", output_dir);
    fs::write(&test_path, &test_csv).expect("Failed to write test CSV");
    println!("   Saved: {} ({} bytes)", test_path, test_csv.len());

    // 5. Export sequences for LSTM/Transformer
    println!("\n5. Exporting sequences for LSTM/Transformer...");
    let sequence_length = 20; // 20-bar lookback for sequence models

    let train_rows = extractor.extract_with_target(&train_bars, target_horizon);
    let mut builder = SequenceBuilder::new(sequence_length);
    let seq_json = builder.to_json(&train_rows);

    let seq_path = format!("{}/train_sequences.json", output_dir);
    fs::write(&seq_path, &seq_json).expect("Failed to write sequences JSON");
    println!("   Saved: {} ({} bytes)", seq_path, seq_json.len());
    println!("   Sequence length: {}", sequence_length);
    println!("   Feature count: {}", builder.feature_names().len());

    // 6. Export feature names for reference
    println!("\n6. Exporting feature metadata...");
    let (_, feature_names) = extractor.extract_matrix(&train_bars);
    let metadata = serde_json::json!({
        "feature_names": feature_names,
        "num_features": feature_names.len(),
        "sequence_length": sequence_length,
        "target_horizon": target_horizon,
        "warmup_period": extractor.warmup_period(),
        "splits": {
            "train_bars": train_bars.len(),
            "validation_bars": val_bars.len(),
            "test_bars": test_bars.len(),
        }
    });

    let meta_path = format!("{}/metadata.json", output_dir);
    fs::write(&meta_path, serde_json::to_string_pretty(&metadata).unwrap())
        .expect("Failed to write metadata");
    println!("   Saved: {}", meta_path);

    // 7. Generate Python loading code
    println!("\n7. Generating Python loading code...");
    let python_code = r#"
"""
Load Ralph-exported features for PyTorch training.

Usage:
    python load_features.py
"""

import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

# Load CSV data for simple models
train_df = pd.read_csv('ml_data/train.csv')
val_df = pd.read_csv('ml_data/validation.csv')
test_df = pd.read_csv('ml_data/test.csv')

# Load metadata
with open('ml_data/metadata.json') as f:
    metadata = json.load(f)

print(f"Features: {metadata['num_features']}")
print(f"Train samples: {len(train_df)}")

# Prepare features and targets
feature_cols = [c for c in train_df.columns if c not in ['index', 'timestamp', 'target']]
X_train = train_df[feature_cols].values
y_train = train_df['target'].values

X_val = val_df[feature_cols].values
y_val = val_df['target'].values

# Handle NaN values
X_train = np.nan_to_num(X_train, nan=0.0)
X_val = np.nan_to_num(X_val, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
y_val = np.nan_to_num(y_val, nan=0.0)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)

print(f"Feature shape: {X_train_tensor.shape}")
print(f"Target shape: {y_train_tensor.shape}")

# For sequence models, load JSON data
with open('ml_data/train_sequences.json') as f:
    seq_data = json.load(f)

sequences = np.array(seq_data['sequences'])
targets = np.array([t if t is not None else np.nan for t in seq_data['targets']])

print(f"Sequence shape: {sequences.shape}")

# Create PyTorch Dataset for sequence models
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(np.nan_to_num(sequences, nan=0.0))
        self.targets = torch.FloatTensor(np.nan_to_num(targets, nan=0.0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

dataset = TimeSeriesDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"\nDataLoader ready with {len(dataset)} samples")

# Example LSTM model structure
class SimpleLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = SimpleLSTM(input_size=metadata['num_features'])
print(f"\nModel created: {model}")
"#;

    let python_path = format!("{}/load_features.py", output_dir);
    fs::write(&python_path, python_code).expect("Failed to write Python code");
    println!("   Saved: {}", python_path);

    // Summary
    println!("\n=== Export Complete ===");
    println!("Output directory: {}/", output_dir);
    println!("\nFiles created:");
    println!("  - train.csv           (CSV for pandas/polars)");
    println!("  - validation.csv      (CSV for pandas/polars)");
    println!("  - test.csv            (CSV for pandas/polars)");
    println!("  - train_sequences.json (JSON for LSTM/Transformer)");
    println!("  - metadata.json       (Feature and config info)");
    println!("  - load_features.py    (Python loading code)");
    println!("\nTo use with PyTorch:");
    println!("  cd {} && python load_features.py", output_dir);
}
