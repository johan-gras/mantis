//! Performance benchmarks for the backtest engine.
//!
//! Run with: cargo bench

use chrono::{TimeZone, Utc};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mantis::data::{atr, bollinger_bands, ema, macd, rsi, sma};
use mantis::engine::{BacktestConfig, Engine};
use mantis::features::{FeatureConfig, FeatureExtractor};
use mantis::strategies::{MacdStrategy, MomentumStrategy, RsiStrategy, SmaCrossover};
use mantis::types::Bar;

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

/// Benchmark feature extraction.
fn bench_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("features");

    // Different data sizes
    for size in [252, 500, 1000].iter() {
        let bars = generate_bars(*size);

        group.bench_with_input(
            BenchmarkId::new("extract_minimal", size),
            &bars,
            |b, bars| {
                let config = FeatureConfig::minimal();
                let extractor = FeatureExtractor::new(config);
                b.iter(|| extractor.extract(black_box(bars)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("extract_comprehensive", size),
            &bars,
            |b, bars| {
                let config = FeatureConfig::comprehensive();
                let extractor = FeatureExtractor::new(config);
                b.iter(|| extractor.extract(black_box(bars)))
            },
        );
    }

    // CSV export
    let bars = generate_bars(500);
    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);

    group.bench_function("to_csv_500", |b| {
        b.iter(|| extractor.to_csv(black_box(&bars), Some(5)))
    });

    group.finish();
}

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

/// Benchmark streaming indicators.
fn bench_streaming_indicators(c: &mut Criterion) {
    use mantis::streaming::{StreamingEMA, StreamingIndicator, StreamingRSI, StreamingSMA};

    let bars = generate_bars(1000);

    let mut group = c.benchmark_group("streaming");

    // Streaming SMA
    group.bench_function("streaming_sma_20", |b| {
        b.iter(|| {
            let mut sma = StreamingSMA::new(20);
            for bar in black_box(&bars) {
                sma.update(bar.close);
            }
            sma.value()
        })
    });

    // Streaming EMA
    group.bench_function("streaming_ema_20", |b| {
        b.iter(|| {
            let mut ema = StreamingEMA::new(20);
            for bar in black_box(&bars) {
                ema.update(bar.close);
            }
            ema.value()
        })
    });

    // Streaming RSI
    group.bench_function("streaming_rsi_14", |b| {
        b.iter(|| {
            let mut rsi = StreamingRSI::new(14);
            for bar in black_box(&bars) {
                rsi.update(bar.close);
            }
            rsi.value()
        })
    });

    group.finish();
}

/// Benchmark Monte Carlo simulation.
fn bench_monte_carlo(c: &mut Criterion) {
    use mantis::monte_carlo::{MonteCarloConfig, MonteCarloSimulator};

    let mut group = c.benchmark_group("monte_carlo");
    group.sample_size(10); // Fewer samples for slow benchmarks

    // Create some test returns
    let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 5.0).collect();

    // Monte Carlo with 100 simulations
    group.bench_function("mc_100_sims", |b| {
        b.iter(|| {
            let config = MonteCarloConfig {
                num_simulations: 100,
                confidence_level: 0.95,
                seed: Some(42),
                resample_trades: true,
                shuffle_returns: false,
            };
            let mut simulator = MonteCarloSimulator::new(config);
            simulator.simulate_from_returns(black_box(&returns), 100_000.0)
        })
    });

    // Monte Carlo with 1000 simulations
    group.bench_function("mc_1000_sims", |b| {
        b.iter(|| {
            let config = MonteCarloConfig {
                num_simulations: 1000,
                confidence_level: 0.95,
                seed: Some(42),
                resample_trades: true,
                shuffle_returns: false,
            };
            let mut simulator = MonteCarloSimulator::new(config);
            simulator.simulate_from_returns(black_box(&returns), 100_000.0)
        })
    });

    group.finish();
}

/// Benchmark regime detection.
fn bench_regime_detection(c: &mut Criterion) {
    use mantis::regime::{RegimeConfig, RegimeDetector};

    let mut group = c.benchmark_group("regime");

    // Different data sizes
    for size in [252, 500, 1000].iter() {
        let bars = generate_bars(*size);

        group.bench_with_input(BenchmarkId::new("detect", size), &bars, |b, bars| {
            let config = RegimeConfig::default();
            let detector = RegimeDetector::new(config);
            b.iter(|| detector.detect(black_box(bars)))
        });
    }

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

    group.bench_function("export_1000_rows", |b| {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("features.parquet");

        b.iter(|| export_features_parquet(black_box(&features), &columns, &path).unwrap())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_indicators,
    bench_backtest,
    bench_features,
    bench_optimization,
    bench_streaming_indicators,
    bench_monte_carlo,
    bench_regime_detection,
    bench_parquet_export,
);

criterion_main!(benches);
