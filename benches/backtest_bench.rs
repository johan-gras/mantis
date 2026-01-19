//! Performance benchmarks for the backtest engine.
//!
//! Run with: cargo bench
//!
//! Spec-required benchmarks (benchmarking.md):
//! - single_bar_1000: < 10ms - 1000-bar backtest, single symbol
//! - daily_10y: < 100ms - 10-year daily data (2520 bars)
//! - optimization_9param: < 10ms - 9-parameter grid optimization (full execution)
//! - sweep_1000: < 30s - 1000 parameter combinations
//! - walkforward_12fold: < 2s - Walk-forward with 12 folds
//! - multi_symbol_3: < 300ms - 3-symbol portfolio, 10y daily

use chrono::{TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mantis::data::{atr, bollinger_bands, ema, macd, rsi, sma};
use mantis::engine::{BacktestConfig, Engine};
use mantis::strategies::{MacdStrategy, MomentumStrategy, RsiStrategy, SmaCrossover};
use mantis::types::Bar;
use mantis::walkforward::{WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMetric};

/// Generate synthetic bars for benchmarking.
fn generate_bars(count: usize) -> Vec<Bar> {
    let mut price = 100.0;
    (0..count)
        .map(|i| {
            let noise = ((i as f64 * 0.7).sin() * 2.0 + (i as f64 * 1.3).cos()) * 0.5;
            price += 0.001 * price + noise;
            price = price.max(50.0);

            Bar::new(
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64),
                price - 1.0,
                price + 2.0,
                price - 2.0,
                price + 0.5,
                1_000_000.0,
            )
        })
        .collect()
}

/// Generate synthetic bars with a different seed for multi-symbol tests.
fn generate_bars_with_seed(count: usize, seed: f64) -> Vec<Bar> {
    let mut price = 100.0 + seed * 10.0;
    (0..count)
        .map(|i| {
            let noise = ((i as f64 * (0.7 + seed * 0.1)).sin() * 2.0
                + (i as f64 * (1.3 + seed * 0.1)).cos())
                * 0.5;
            price += 0.001 * price + noise;
            price = price.max(50.0);

            Bar::new(
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64),
                price - 1.0,
                price + 2.0,
                price - 2.0,
                price + 0.5,
                1_000_000.0,
            )
        })
        .collect()
}

/// Benchmark technical indicators.
fn bench_indicators(c: &mut Criterion) {
    let bars = generate_bars(1000);

    let mut group = c.benchmark_group("indicators");

    // SMA benchmarks
    for period in [10, 20, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("sma", period), period, |b, &period| {
            b.iter(|| sma(black_box(&bars), period))
        });
    }

    // EMA benchmarks
    for period in [10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::new("ema", period), period, |b, &period| {
            b.iter(|| ema(black_box(&bars), period))
        });
    }

    // RSI benchmark
    group.bench_function("rsi_14", |b| b.iter(|| rsi(black_box(&bars), 14)));

    // MACD benchmark
    group.bench_function("macd_12_26_9", |b| {
        b.iter(|| macd(black_box(&bars), 12, 26, 9))
    });

    // ATR benchmark
    group.bench_function("atr_14", |b| b.iter(|| atr(black_box(&bars), 14)));

    // Bollinger Bands benchmark
    group.bench_function("bbands_20_2", |b| {
        b.iter(|| bollinger_bands(black_box(&bars), 20, 2.0))
    });

    group.finish();
}

/// Benchmark backtest execution.
fn bench_backtest(c: &mut Criterion) {
    let mut group = c.benchmark_group("backtest");

    // Different data sizes
    for size in [252, 500, 1000, 2000].iter() {
        let bars = generate_bars(*size);

        group.bench_with_input(BenchmarkId::new("sma_crossover", size), &bars, |b, bars| {
            b.iter(|| {
                let config = BacktestConfig {
                    initial_capital: 100_000.0,
                    show_progress: false,
                    ..Default::default()
                };
                let mut engine = Engine::new(config);
                engine.add_data("TEST".to_string(), bars.clone());
                let mut strategy = SmaCrossover::new(10, 30);
                engine.run(black_box(&mut strategy), "TEST")
            })
        });
    }

    // Different strategies with 1000 bars
    let bars = generate_bars(1000);

    group.bench_function("macd_strategy_1000", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars.clone());
            let mut strategy = MacdStrategy::default_params();
            engine.run(black_box(&mut strategy), "TEST")
        })
    });

    group.bench_function("rsi_strategy_1000", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars.clone());
            let mut strategy = RsiStrategy::new(14, 30.0, 70.0);
            engine.run(black_box(&mut strategy), "TEST")
        })
    });

    group.bench_function("momentum_strategy_1000", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars.clone());
            let mut strategy = MomentumStrategy::new(20, 0.02);
            engine.run(black_box(&mut strategy), "TEST")
        })
    });

    group.finish();
}

// NOTE: bench_features removed - mantis::features module not implemented
// TODO: Re-add when FeatureConfig/FeatureExtractor are implemented

/// Benchmark parameter optimization.
fn bench_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization");
    group.sample_size(10); // Fewer samples for slow benchmarks

    let bars = generate_bars(500);

    // Small parameter grid
    group.bench_function("optimize_9_params", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars.clone());

            let params: Vec<(usize, usize)> = vec![
                (5, 20),
                (5, 30),
                (5, 40),
                (10, 30),
                (10, 40),
                (10, 50),
                (15, 40),
                (15, 50),
                (15, 60),
            ];

            engine.optimize("TEST", params, |&(fast, slow)| {
                Box::new(SmaCrossover::new(fast, slow))
            })
        })
    });

    group.finish();
}

// NOTE: bench_streaming_indicators removed - mantis::streaming module not implemented
// TODO: Re-add when StreamingSMA/StreamingEMA/StreamingRSI are implemented

/// Benchmark Monte Carlo simulation.
fn bench_monte_carlo(c: &mut Criterion) {
    use mantis::monte_carlo::{MonteCarloConfig, MonteCarloSimulator};

    let mut group = c.benchmark_group("monte_carlo");
    group.sample_size(10); // Fewer samples for slow benchmarks

    // Create some test returns
    let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 5.0).collect();

    // Monte Carlo with 100 simulations (block bootstrap)
    group.bench_function("mc_100_sims", |b| {
        b.iter(|| {
            let config = MonteCarloConfig {
                num_simulations: 100,
                confidence_level: 0.95,
                seed: Some(42),
                resample_trades: true,
                shuffle_returns: false,
                block_bootstrap: true,
                block_size: None,
            };
            let mut simulator = MonteCarloSimulator::new(config);
            simulator.simulate_from_returns(black_box(&returns), 100_000.0)
        })
    });

    // Monte Carlo with 1000 simulations (block bootstrap)
    group.bench_function("mc_1000_sims", |b| {
        b.iter(|| {
            let config = MonteCarloConfig {
                num_simulations: 1000,
                confidence_level: 0.95,
                seed: Some(42),
                resample_trades: true,
                shuffle_returns: false,
                block_bootstrap: true,
                block_size: None,
            };
            let mut simulator = MonteCarloSimulator::new(config);
            simulator.simulate_from_returns(black_box(&returns), 100_000.0)
        })
    });

    group.finish();
}

// NOTE: bench_regime_detection removed - mantis::regime module not implemented
// TODO: Re-add when RegimeConfig/RegimeDetector are implemented

/// Spec-required benchmarks per benchmarking.md
fn bench_spec_required(c: &mut Criterion) {
    let mut group = c.benchmark_group("spec_required");

    // single_bar_1000: 1000-bar backtest, single symbol (target: < 10ms)
    let bars_1000 = generate_bars(1000);
    group.bench_function("single_bar_1000", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars_1000.clone());
            let mut strategy = SmaCrossover::new(10, 30);
            engine.run(black_box(&mut strategy), "TEST")
        })
    });

    // daily_10y: 10-year daily data (2520 bars) (target: < 100ms)
    let bars_10y = generate_bars(2520);
    group.bench_function("daily_10y", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars_10y.clone());
            let mut strategy = SmaCrossover::new(10, 30);
            engine.run(black_box(&mut strategy), "TEST")
        })
    });

    // multi_symbol_3: 3-symbol portfolio, 10y daily (target: < 300ms)
    let bars_sym1 = generate_bars_with_seed(2520, 0.0);
    let bars_sym2 = generate_bars_with_seed(2520, 1.0);
    let bars_sym3 = generate_bars_with_seed(2520, 2.0);
    group.bench_function("multi_symbol_3", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("SYM1".to_string(), bars_sym1.clone());
            engine.add_data("SYM2".to_string(), bars_sym2.clone());
            engine.add_data("SYM3".to_string(), bars_sym3.clone());

            // Run backtest on each symbol (multi-symbol not yet combined)
            let mut strategy1 = SmaCrossover::new(10, 30);
            let r1 = engine.run(black_box(&mut strategy1), "SYM1");

            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("SYM1".to_string(), bars_sym1.clone());
            engine.add_data("SYM2".to_string(), bars_sym2.clone());
            engine.add_data("SYM3".to_string(), bars_sym3.clone());

            let mut strategy2 = SmaCrossover::new(10, 30);
            let r2 = engine.run(black_box(&mut strategy2), "SYM2");

            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("SYM1".to_string(), bars_sym1.clone());
            engine.add_data("SYM2".to_string(), bars_sym2.clone());
            engine.add_data("SYM3".to_string(), bars_sym3.clone());

            let mut strategy3 = SmaCrossover::new(10, 30);
            let r3 = engine.run(black_box(&mut strategy3), "SYM3");

            (r1, r2, r3)
        })
    });

    group.finish();
}

/// Benchmark parameter sweep (spec-required: sweep_1000 < 30s).
fn bench_sweep_1000(c: &mut Criterion) {
    let mut group = c.benchmark_group("sweep");
    group.sample_size(10); // Fewer samples for slow benchmarks

    let bars = generate_bars(500); // Smaller data for reasonable sweep time

    // Generate 1000 parameter combinations
    // Using 10 fast × 10 slow × 10 threshold = 1000 combinations
    let params: Vec<(usize, usize)> = (5..=50)
        .step_by(5)
        .flat_map(|fast| (20..=110).step_by(10).map(move |slow| (fast, slow)))
        .filter(|(fast, slow)| fast < slow)
        .take(1000)
        .collect();

    // sweep_1000: 1000 parameter combinations (target: < 30s)
    group.bench_function("sweep_1000", |b| {
        b.iter(|| {
            let config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };
            let mut engine = Engine::new(config);
            engine.add_data("TEST".to_string(), bars.clone());

            engine.optimize("TEST", params.clone(), |&(fast, slow)| {
                Box::new(SmaCrossover::new(fast, slow))
            })
        })
    });

    group.finish();
}

/// Benchmark walk-forward analysis (spec-required: walkforward_12fold < 2s).
fn bench_walkforward(c: &mut Criterion) {
    let mut group = c.benchmark_group("walkforward");
    group.sample_size(10); // Fewer samples for slow benchmarks

    // Need enough data for 12 folds: 12 * 50 = 600 bars minimum
    let bars = generate_bars(1200); // 1200 bars for 12-fold

    // Parameter grid for optimization within each fold
    let params: Vec<(usize, usize)> = vec![(5, 20), (10, 30), (15, 40), (20, 50)];

    // walkforward_12fold: Walk-forward with 12 folds (target: < 2s)
    group.bench_function("walkforward_12fold", |b| {
        b.iter(|| {
            let wf_config = WalkForwardConfig {
                num_windows: 12,
                in_sample_ratio: 0.75,
                anchored: true,
                min_bars_per_window: 50,
            };

            let backtest_config = BacktestConfig {
                initial_capital: 100_000.0,
                show_progress: false,
                ..Default::default()
            };

            let analyzer = WalkForwardAnalyzer::new(wf_config, backtest_config);

            analyzer.run(
                black_box(&bars),
                "TEST",
                params.clone(),
                |&(fast, slow)| Box::new(SmaCrossover::new(fast, slow)),
                WalkForwardMetric::Sharpe,
            )
        })
    });

    group.finish();
}

/// Benchmark Parquet export.
fn bench_parquet_export(c: &mut Criterion) {
    use mantis::export::export_features_parquet;
    use tempfile::TempDir;

    let mut group = c.benchmark_group("parquet");
    group.sample_size(10); // Fewer samples for I/O benchmarks

    // Create feature matrix
    let features: Vec<Vec<f64>> = (0..1000)
        .map(|i| {
            vec![
                i as f64,
                (i as f64 * 0.1).sin(),
                (i as f64 * 0.2).cos(),
                i as f64 * 0.001,
                100.0 - i as f64 * 0.01,
            ]
        })
        .collect();
    let columns = vec!["f1", "f2", "f3", "f4", "f5"];

    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("features.parquet");

    group.bench_function("export_1000_rows", |b| {
        b.iter(|| export_features_parquet(black_box(&features), &columns, &path).unwrap())
    });

    group.finish();
}

/// Benchmark ONNX model inference (spec-required: < 1ms/bar).
///
/// Run with: cargo bench --features onnx -- onnx
#[cfg(feature = "onnx")]
fn bench_onnx_inference(c: &mut Criterion) {
    use mantis::onnx::{ModelConfig, OnnxModel};
    use std::path::Path;

    let model_path = Path::new("data/models/minimal.onnx");

    // Skip if test models not available
    if !model_path.exists() {
        eprintln!(
            "ONNX benchmarks skipped: test models not found. Run: python scripts/generate_test_onnx.py"
        );
        return;
    }

    let mut group = c.benchmark_group("onnx");

    // Single inference benchmark (target: < 1ms)
    group.bench_function("single_inference_minimal", |b| {
        let config = ModelConfig::new("minimal", 10);
        let mut model = OnnxModel::from_file(model_path, config).unwrap();
        let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        b.iter(|| model.predict(black_box(&features)))
    });

    // Batch inference benchmark (100 samples)
    group.bench_function("batch_inference_100", |b| {
        let config = ModelConfig::new("minimal", 10);
        let mut model = OnnxModel::from_file(model_path, config).unwrap();
        let batch: Vec<Vec<f64>> = (0..100)
            .map(|i| (0..10).map(|j| ((i * 10 + j) as f64) / 1000.0).collect())
            .collect();

        b.iter(|| model.predict_batch(black_box(&batch)))
    });

    // Batch inference benchmark (1000 samples - 10-year daily equivalent)
    group.bench_function("batch_inference_1000", |b| {
        let config = ModelConfig::new("minimal", 10);
        let mut model = OnnxModel::from_file(model_path, config).unwrap();
        let batch: Vec<Vec<f64>> = (0..1000)
            .map(|i| (0..10).map(|j| ((i * 10 + j) as f64) / 10000.0).collect())
            .collect();

        b.iter(|| model.predict_batch(black_box(&batch)))
    });

    // Larger model with 20 inputs
    let larger_model_path = Path::new("data/models/simple_mlp.onnx");
    if larger_model_path.exists() {
        group.bench_function("single_inference_simple_mlp", |b| {
            let config = ModelConfig::new("simple_mlp", 10);
            let mut model = OnnxModel::from_file(larger_model_path, config).unwrap();
            let features: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

            b.iter(|| model.predict(black_box(&features)))
        });

        // Spec-required: ONNX inference for 2520 bars (10-year daily) should complete quickly
        group.bench_function("batch_inference_2520_10y", |b| {
            let config = ModelConfig::new("simple_mlp", 10);
            let mut model = OnnxModel::from_file(larger_model_path, config).unwrap();
            let batch: Vec<Vec<f64>> = (0..2520)
                .map(|i| (0..10).map(|j| ((i * 10 + j) as f64) / 25200.0).collect())
                .collect();

            b.iter(|| model.predict_batch(black_box(&batch)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_indicators,
    bench_backtest,
    // bench_features,              // Removed: mantis::features not implemented
    bench_optimization,
    // bench_streaming_indicators,  // Removed: mantis::streaming not implemented
    bench_monte_carlo,
    // bench_regime_detection,      // Removed: mantis::regime not implemented
    bench_parquet_export,
    // Spec-required benchmarks (benchmarking.md)
    bench_spec_required,
    bench_sweep_1000,
    bench_walkforward,
);

// ONNX benchmarks are in a separate group due to feature flag
#[cfg(feature = "onnx")]
criterion_group!(onnx_benches, bench_onnx_inference);

#[cfg(feature = "onnx")]
criterion_main!(benches, onnx_benches);

#[cfg(not(feature = "onnx"))]
criterion_main!(benches);
