//! Performance benchmarks for the backtest engine.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ralph_backtest::data::{atr, bollinger_bands, ema, macd, rsi, sma};
use ralph_backtest::engine::{BacktestConfig, Engine};
use ralph_backtest::features::{FeatureConfig, FeatureExtractor};
use ralph_backtest::strategies::{SmaCrossover, MacdStrategy, RsiStrategy, MomentumStrategy};
use ralph_backtest::types::Bar;
use chrono::{TimeZone, Utc};

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
        group.bench_with_input(
            BenchmarkId::new("sma", period),
            period,
            |b, &period| b.iter(|| sma(black_box(&bars), period)),
        );
    }

    // EMA benchmarks
    for period in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("ema", period),
            period,
            |b, &period| b.iter(|| ema(black_box(&bars), period)),
        );
    }

    // RSI benchmark
    group.bench_function("rsi_14", |b| {
        b.iter(|| rsi(black_box(&bars), 14))
    });

    // MACD benchmark
    group.bench_function("macd_12_26_9", |b| {
        b.iter(|| macd(black_box(&bars), 12, 26, 9))
    });

    // ATR benchmark
    group.bench_function("atr_14", |b| {
        b.iter(|| atr(black_box(&bars), 14))
    });

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

        group.bench_with_input(
            BenchmarkId::new("sma_crossover", size),
            &bars,
            |b, bars| {
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
            },
        );
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
                (5, 20), (5, 30), (5, 40),
                (10, 30), (10, 40), (10, 50),
                (15, 40), (15, 50), (15, 60),
            ];

            engine.optimize("TEST", params, |&(fast, slow)| {
                Box::new(SmaCrossover::new(fast, slow))
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_indicators,
    bench_backtest,
    bench_features,
    bench_optimization,
);

criterion_main!(benches);
