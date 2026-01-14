//! Example: Machine Learning Backtest Workflow
//!
//! This example demonstrates the complete ML workflow:
//! 1. Load historical data
//! 2. Extract features for model training
//! 3. Simulate model predictions (in practice, load from your ML framework)
//! 4. Run backtest with the predictions
//! 5. Analyze results
//!
//! Run with: cargo run --example ml_backtest

use mantis::data::{load_csv, DataConfig};
use mantis::engine::{BacktestConfig, Engine};
use mantis::features::{FeatureConfig, FeatureExtractor, TimeSeriesSplitter};
use mantis::strategies::ExternalSignalStrategy;
use mantis::analytics::ResultFormatter;
use mantis::types::Bar;
use chrono::{TimeZone, Utc};

/// Generate synthetic data for demonstration.
fn generate_synthetic_data(days: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(days);
    let mut price = 100.0;

    for i in 0..days {
        // Add trend and noise
        let trend = 0.0005; // Slight upward bias
        let noise = ((i as f64 * 0.7).sin() * 2.0 + (i as f64 * 1.3).cos()) * 0.5;
        let daily_change = price * trend + noise;
        price += daily_change;
        price = price.max(50.0); // Floor at $50

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

/// Simulate ML model predictions.
///
/// In a real workflow, you would:
/// 1. Export features to CSV/Parquet
/// 2. Train your model in Python/Julia/etc.
/// 3. Generate predictions
/// 4. Load predictions back here
fn simulate_ml_predictions(features_matrix: &[Vec<f64>]) -> Vec<f64> {
    // This is a simple moving average crossover "model" for demonstration
    // Replace this with your actual model predictions
    features_matrix
        .iter()
        .map(|row| {
            // Use first few features to simulate a prediction
            // In practice, your model would output these values
            let signal: f64 = row.iter().take(5).sum::<f64>() / 5.0;

            // Normalize to [-1, 1] range
            (signal * 0.1).tanh()
        })
        .collect()
}

fn main() {
    println!("=== Machine Learning Backtest Workflow ===\n");

    // 1. Load or generate data
    println!("1. Loading data...");
    let bars = if std::path::Path::new("data/sample.csv").exists() {
        load_csv("data/sample.csv", &DataConfig::default())
            .expect("Failed to load data")
    } else {
        println!("   Using synthetic data (data/sample.csv not found)");
        generate_synthetic_data(500)
    };
    println!("   Loaded {} bars\n", bars.len());

    // 2. Split data (train/validation/test)
    println!("2. Splitting data...");
    let splitter = TimeSeriesSplitter::new(0.6, 0.2).with_gap(5);
    let (train_bars, val_bars, test_bars) = splitter.split(&bars);
    println!("   Train: {} bars", train_bars.len());
    println!("   Validation: {} bars", val_bars.len());
    println!("   Test: {} bars\n", test_bars.len());

    // 3. Extract features
    println!("3. Extracting features...");
    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);

    // Extract features with target (5-day forward return)
    let train_features = extractor.extract_with_target(&train_bars, 5);
    let (train_matrix, feature_names) = extractor.extract_matrix(&train_bars);
    println!("   Features extracted: {}", feature_names.len());
    println!("   Feature names: {:?}\n", &feature_names[..5.min(feature_names.len())]);

    // 4. "Train" model (simulated)
    println!("4. Training model (simulated)...");
    let _train_predictions = simulate_ml_predictions(&train_matrix);
    println!("   Model trained on {} samples\n", train_features.len());

    // 5. Generate predictions for test set
    println!("5. Generating predictions for test set...");
    let (test_matrix, _) = extractor.extract_matrix(&test_bars);
    let test_predictions = simulate_ml_predictions(&test_matrix);
    println!("   Generated {} predictions\n", test_predictions.len());

    // 6. Run backtest with predictions
    println!("6. Running backtest...");

    // Calculate offset (warmup period for feature extraction)
    let warmup = extractor.warmup_period();

    // Create strategy with external signals
    let strategy = ExternalSignalStrategy::new(test_predictions, 0.3)
        .with_offset(warmup)
        .with_exit_threshold(0.1)
        .with_name("ML Model v1");

    // Configure backtest
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.95, // Use 95% of capital
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(backtest_config);
    engine.add_data("TEST".to_string(), test_bars);

    // Clone strategy to avoid borrow issues
    let mut strategy = strategy;
    let result = engine.run(&mut strategy, "TEST").expect("Backtest failed");

    // 7. Analyze results
    println!("\n7. Backtest Results:");
    ResultFormatter::print_report(&result);

    // Export results
    println!("\n8. Exporting results...");
    let json = ResultFormatter::to_json(&result);
    if let Ok(_) = std::fs::write("backtest_result.json", &json) {
        println!("   Results saved to backtest_result.json");
    }

    // 9. Summary
    println!("\n=== Summary ===");
    if result.total_return_pct > 0.0 && result.sharpe_ratio > 0.5 {
        println!("Model shows promise! Consider further optimization.");
    } else {
        println!("Model needs improvement. Consider:");
        println!("  - Feature engineering");
        println!("  - Different model architecture");
        println!("  - Hyperparameter tuning");
    }
}
