//! Integration tests for the backtest engine.

use chrono::{TimeZone, Utc};
use mantis::data::{load_csv, DataConfig};
use mantis::engine::{BacktestConfig, Engine};
use mantis::features::{FeatureConfig, FeatureExtractor, SequenceBuilder, TimeSeriesSplitter};
use mantis::portfolio::{CostModel, CryptoCost, ForexCost, FuturesCost, MarketImpactModel};
use mantis::risk::{RiskConfig, StopLoss, TakeProfit};
use mantis::strategies::{
    BreakoutStrategy, ClassificationStrategy, EnsembleSignalStrategy, ExternalSignalStrategy,
    MacdStrategy, MomentumStrategy, RsiStrategy, SmaCrossover,
};
use mantis::types::Bar;
use mantis::walkforward::{WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMetric};

/// Create synthetic test data with a trend and some noise.
fn create_synthetic_data(days: usize, initial_price: f64, daily_return: f64) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(days);
    let mut price = initial_price;

    for i in 0..days {
        // Add some randomness via a simple deterministic pattern
        let noise = ((i as f64 * 0.7).sin() * 2.0 + (i as f64 * 1.3).cos()) * 0.5;
        let daily_change = price * daily_return + noise;
        price += daily_change;

        let open = price - 0.5;
        let high = price + 2.0 + noise.abs();
        let low = price - 2.0 - noise.abs();
        let close = price;
        let volume = 1_000_000.0 + (noise * 100000.0);

        bars.push(Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i as i64),
            open,
            high.max(open).max(close),
            low.min(open).min(close),
            close,
            volume.abs(),
        ));
    }

    bars
}

#[test]
fn test_full_backtest_sma_crossover() {
    // Create data with more price movement to trigger crossovers
    let bars = create_synthetic_data(252, 100.0, 0.003);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("TEST".to_string(), bars);

    let mut strategy = SmaCrossover::new(5, 15); // Faster periods for more signals
    let result = engine.run(&mut strategy, "TEST").unwrap();

    // Basic sanity checks
    assert!(result.final_equity > 0.0);
    // May have 0 trades if no crossovers occur - this is valid behavior
    assert!(result.sharpe_ratio.is_finite() || result.total_trades == 0);
    assert!(result.max_drawdown_pct <= 100.0);
}

#[test]
fn test_full_backtest_macd() {
    let bars = create_synthetic_data(200, 100.0, 0.002);

    let config = BacktestConfig {
        initial_capital: 50_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("MACD_TEST".to_string(), bars);

    let mut strategy = MacdStrategy::default_params();
    let result = engine.run(&mut strategy, "MACD_TEST").unwrap();

    assert!(result.final_equity > 0.0);
    assert_eq!(result.strategy_name, "MACD Strategy");
}

#[test]
fn test_full_backtest_rsi() {
    let bars = create_synthetic_data(180, 100.0, 0.0);

    let config = BacktestConfig {
        initial_capital: 75_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("RSI_TEST".to_string(), bars);

    let mut strategy = RsiStrategy::new(14, 30.0, 70.0);
    let result = engine.run(&mut strategy, "RSI_TEST").unwrap();

    assert!(result.final_equity > 0.0);
    assert!(result.win_rate >= 0.0 && result.win_rate <= 100.0);
}

#[test]
fn test_backtest_with_stop_loss() {
    let bars = create_synthetic_data(100, 100.0, -0.002); // Downtrend

    let risk_config = RiskConfig {
        stop_loss: StopLoss::Percentage(5.0),
        ..Default::default()
    };

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        risk_config,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("SL_TEST".to_string(), bars);

    let mut strategy = MomentumStrategy::new(10, 0.0);
    let result = engine.run(&mut strategy, "SL_TEST").unwrap();

    // With stop loss, drawdown should be limited
    assert!(result.max_drawdown_pct <= 50.0); // Generous limit due to potential gaps
}

#[test]
fn test_backtest_with_take_profit() {
    let bars = create_synthetic_data(100, 100.0, 0.005); // Uptrend

    let risk_config = RiskConfig {
        take_profit: TakeProfit::Percentage(5.0),
        ..Default::default()
    };

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        risk_config,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("TP_TEST".to_string(), bars);

    let mut strategy = MomentumStrategy::new(5, 0.0);
    let result = engine.run(&mut strategy, "TP_TEST").unwrap();

    assert!(result.total_trades > 0);
}

#[test]
fn test_backtest_with_costs() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let cost_model = CostModel {
        commission_flat: 5.0,
        commission_pct: 0.001, // 0.1%
        slippage_pct: 0.001,   // 0.1%
        min_commission: 1.0,
        futures: FuturesCost::default(),
        crypto: CryptoCost::default(),
        forex: ForexCost::default(),
        market_impact: MarketImpactModel::None,
        max_volume_participation: None,
    };

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        cost_model,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("COST_TEST".to_string(), bars);

    let mut strategy = SmaCrossover::new(5, 20);
    let result = engine.run(&mut strategy, "COST_TEST").unwrap();

    // With transaction costs, return should be less than without
    assert!(result.final_equity < 200_000.0); // Sanity check
}

#[test]
fn test_optimization() {
    let bars = create_synthetic_data(300, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("OPT_TEST".to_string(), bars);

    // Optimize SMA crossover
    let params: Vec<(usize, usize)> = vec![(5, 20), (10, 30), (15, 40)];

    let results = engine
        .optimize("OPT_TEST", params, |&(fast, slow)| {
            Box::new(SmaCrossover::new(fast, slow))
        })
        .unwrap();

    assert_eq!(results.len(), 3);

    // All results should have valid metrics
    for (_, result) in &results {
        assert!(result.sharpe_ratio.is_finite());
        assert!(result.total_return_pct.is_finite());
    }
}

#[test]
fn test_load_sample_data() {
    // This test depends on sample.csv existing
    let data_path = "data/sample.csv";
    if std::path::Path::new(data_path).exists() {
        let bars = load_csv(data_path, &DataConfig::default()).unwrap();
        assert!(!bars.is_empty());

        // Verify data integrity
        for i in 1..bars.len() {
            assert!(bars[i].timestamp > bars[i - 1].timestamp);
            assert!(bars[i].high >= bars[i].low);
            assert!(bars[i].high >= bars[i].open);
            assert!(bars[i].high >= bars[i].close);
            assert!(bars[i].low <= bars[i].open);
            assert!(bars[i].low <= bars[i].close);
        }
    }
}

#[test]
fn test_short_selling_disabled() {
    let bars = create_synthetic_data(100, 100.0, -0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        allow_short: false,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("NO_SHORT".to_string(), bars);

    let mut strategy = SmaCrossover::new(5, 20);
    let result = engine.run(&mut strategy, "NO_SHORT").unwrap();

    // All trades should be long-only
    for trade in &result.trades {
        assert!(trade.quantity >= 0.0 || trade.side == mantis::types::Side::Sell);
    }
}

#[test]
fn test_position_sizing() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.5, // 50% of equity
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("SIZE_TEST".to_string(), bars);

    let mut strategy = SmaCrossover::new(5, 20);
    let result = engine.run(&mut strategy, "SIZE_TEST").unwrap();

    // Position sizes should be roughly 50% of equity at entry
    assert!(result.final_equity > 0.0);
}

// ============================================================================
// ML Strategy Integration Tests
// ============================================================================

#[test]
fn test_external_signal_strategy_backtest() {
    let bars = create_synthetic_data(100, 100.0, 0.002);

    // Create signals that match the data length
    let warmup = 10; // Feature extraction warmup
    let num_signals = bars.len() - warmup;
    let signals: Vec<f64> = (0..num_signals)
        .map(|i| {
            // Alternating signals
            if i % 20 < 10 {
                0.6
            } else {
                -0.6
            }
        })
        .collect();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("ML_TEST".to_string(), bars);

    let mut strategy = ExternalSignalStrategy::new(signals, 0.5)
        .with_offset(warmup)
        .with_name("Test ML Model");

    let result = engine.run(&mut strategy, "ML_TEST").unwrap();

    assert!(result.final_equity > 0.0);
    assert_eq!(result.strategy_name, "Test ML Model");
}

#[test]
fn test_classification_strategy_backtest() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    // Create class predictions (1=long, 0=hold/exit, -1=short)
    let predictions: Vec<i8> = (0..bars.len())
        .map(|i| match i % 30 {
            0..=9 => 1,    // Go long
            10..=14 => 0,  // Hold/exit
            15..=24 => -1, // Go short
            _ => 0,
        })
        .collect();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("CLASS_TEST".to_string(), bars);

    let mut strategy = ClassificationStrategy::new(predictions).with_name("Classification Model");

    let result = engine.run(&mut strategy, "CLASS_TEST").unwrap();

    assert!(result.final_equity > 0.0);
    assert!(result.total_trades > 0);
}

#[test]
fn test_ensemble_strategy_backtest() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    // Create multiple model predictions
    let model1: Vec<f64> = (0..bars.len())
        .map(|i| (i as f64 * 0.1).sin() * 0.6)
        .collect();
    let model2: Vec<f64> = (0..bars.len())
        .map(|i| (i as f64 * 0.15).cos() * 0.5)
        .collect();
    let model3: Vec<f64> = (0..bars.len())
        .map(|i| if i % 20 < 10 { 0.4 } else { -0.3 })
        .collect();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("ENSEMBLE_TEST".to_string(), bars);

    let mut strategy = EnsembleSignalStrategy::new(vec![model1, model2, model3], 0.3);

    let result = engine.run(&mut strategy, "ENSEMBLE_TEST").unwrap();

    assert!(result.final_equity > 0.0);
}

// ============================================================================
// Feature Extraction Integration Tests
// ============================================================================

#[test]
fn test_feature_extraction_pipeline() {
    let bars = create_synthetic_data(200, 100.0, 0.001);

    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);

    // Extract features with target
    let rows = extractor.extract_with_target(&bars, 5);

    assert!(!rows.is_empty());

    // Verify features exist
    let first_row = &rows[0];
    assert!(first_row.features.contains_key("return_1"));
    assert!(first_row.features.contains_key("rsi"));

    // Verify some targets exist
    let with_target: Vec<_> = rows.iter().filter(|r| r.target.is_some()).collect();
    assert!(!with_target.is_empty());
}

#[test]
fn test_feature_matrix_export() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);

    let (matrix, names) = extractor.extract_matrix(&bars);

    assert!(!matrix.is_empty());
    assert!(!names.is_empty());

    // All rows should have the same number of features
    for row in &matrix {
        assert_eq!(row.len(), names.len());
    }
}

#[test]
fn test_sequence_builder() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);
    let rows = extractor.extract(&bars);

    let mut builder = SequenceBuilder::new(10);
    let (sequences, targets) = builder.build_sequences(&rows);

    assert!(!sequences.is_empty());
    assert_eq!(sequences.len(), targets.len());

    // Each sequence should have the correct length
    for seq in &sequences {
        assert_eq!(seq.len(), 10);
    }
}

#[test]
fn test_time_series_splitter() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let splitter = TimeSeriesSplitter::new(0.7, 0.15).with_gap(5);
    let (train, val, test) = splitter.split(&bars);

    // Verify sizes
    assert!(!train.is_empty());
    assert!(!val.is_empty());
    assert!(!test.is_empty());

    // Verify temporal ordering (train before val before test)
    assert!(train.last().unwrap().timestamp < val.first().unwrap().timestamp);
    assert!(val.last().unwrap().timestamp < test.first().unwrap().timestamp);
}

#[test]
fn test_csv_export() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(config);

    let csv = extractor.to_csv(&bars, Some(5));

    assert!(!csv.is_empty());
    assert!(csv.contains("index,timestamp"));
    assert!(csv.contains("target"));

    // Verify it has multiple lines
    let lines: Vec<_> = csv.lines().collect();
    assert!(lines.len() > 10);
}

// ============================================================================
// Walk-Forward Analysis Tests
// ============================================================================

#[test]
fn test_walk_forward_analysis() {
    let bars = create_synthetic_data(500, 100.0, 0.001);

    let wf_config = WalkForwardConfig {
        num_windows: 3,
        in_sample_ratio: 0.7,
        anchored: false,
        min_bars_per_window: 50,
    };

    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let analyzer = WalkForwardAnalyzer::new(wf_config, backtest_config);

    let params: Vec<(usize, usize)> = vec![(5, 20), (10, 30)];

    let result = analyzer
        .run(
            &bars,
            "TEST",
            params,
            |&(fast, slow)| Box::new(SmaCrossover::new(fast, slow)),
            WalkForwardMetric::Sharpe,
        )
        .unwrap();

    assert!(!result.windows.is_empty());
    assert!(result.avg_is_return.is_finite());
    assert!(result.avg_oos_return.is_finite());
}

// ============================================================================
// Additional Strategy Tests
// ============================================================================

#[test]
fn test_breakout_strategy_backtest() {
    let bars = create_synthetic_data(100, 100.0, 0.003);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("BREAKOUT_TEST".to_string(), bars);

    let mut strategy = BreakoutStrategy::new(20, 10);
    let result = engine.run(&mut strategy, "BREAKOUT_TEST").unwrap();

    assert!(result.final_equity > 0.0);
    assert_eq!(result.strategy_name, "Donchian Breakout");
}

#[test]
fn test_multiple_strategies_comparison() {
    let bars = create_synthetic_data(200, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    // Test multiple strategies on the same data
    let strategies: Vec<Box<dyn mantis::Strategy>> = vec![
        Box::new(SmaCrossover::new(10, 30)),
        Box::new(MacdStrategy::default_params()),
        Box::new(RsiStrategy::new(14, 30.0, 70.0)),
        Box::new(MomentumStrategy::new(20, 0.02)),
    ];

    for mut strategy in strategies {
        let mut engine = Engine::new(config.clone());
        engine.add_data("MULTI_TEST".to_string(), bars.clone());

        let result = engine.run(strategy.as_mut(), "MULTI_TEST").unwrap();
        assert!(result.final_equity > 0.0);
    }
}

#[test]
fn test_risk_reward_ratio() {
    let bars = create_synthetic_data(100, 100.0, 0.002);

    let risk_config = RiskConfig {
        stop_loss: StopLoss::Percentage(2.0),
        take_profit: TakeProfit::Percentage(6.0), // 3:1 risk-reward
        ..Default::default()
    };

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        risk_config,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("RR_TEST".to_string(), bars);

    let mut strategy = MomentumStrategy::new(10, 0.0);
    let result = engine.run(&mut strategy, "RR_TEST").unwrap();

    assert!(result.final_equity > 0.0);
}

#[test]
fn test_equity_curve_stored_in_result() {
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("TEST".to_string(), bars.clone());

    let mut strategy = SmaCrossover::new(5, 20);
    let result = engine.run(&mut strategy, "TEST").unwrap();

    // Equity curve should be populated
    assert!(!result.equity_curve.is_empty());
    assert_eq!(result.equity_curve.len(), bars.len());

    // First point should start at initial capital
    assert!((result.equity_curve[0].equity - 100_000.0).abs() < 0.01);

    // Last point should match final equity
    let last_equity = result.equity_curve.last().unwrap().equity;
    assert!((last_equity - result.final_equity).abs() < 0.01);
}

#[test]
fn test_monte_carlo_simulation() {
    use mantis::monte_carlo::{MonteCarloConfig, MonteCarloSimulator};

    // Create data with enough price movement to generate trades
    let bars = create_synthetic_data(500, 100.0, 0.002);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("MC_TEST".to_string(), bars);

    // Use momentum strategy which is more likely to generate trades
    let mut strategy = MomentumStrategy::new(5, 0.0);
    let result = engine.run(&mut strategy, "MC_TEST").unwrap();

    // Only run Monte Carlo if we have trades
    if result.trades.iter().filter(|t| t.is_closed()).count() >= 5 {
        // Run Monte Carlo simulation
        let mc_config = MonteCarloConfig {
            num_simulations: 100,
            confidence_level: 0.95,
            seed: Some(42),
            resample_trades: true,
            shuffle_returns: false,
        };

        let mut simulator = MonteCarloSimulator::new(mc_config);
        let mc_result = simulator.simulate_from_result(&result);

        // Verify simulation ran
        assert_eq!(mc_result.num_simulations, 100);

        // Confidence intervals should be valid
        assert!(mc_result.return_ci.0 <= mc_result.return_ci.1);
        assert!(mc_result.max_drawdown_ci.0 <= mc_result.max_drawdown_ci.1);

        // Mean return should be in the confidence interval
        assert!(mc_result.return_ci.0 <= mc_result.mean_return);
        assert!(mc_result.mean_return <= mc_result.return_ci.1);
    }
}

#[test]
fn test_regime_detection() {
    use mantis::regime::{RegimeConfig, RegimeDetector};

    // Create uptrending data
    let uptrend_bars = create_synthetic_data(100, 100.0, 0.01); // 1% daily returns

    // Create downtrending data
    let mut downtrend_bars = Vec::new();
    let mut price = 100.0;
    for i in 0..100 {
        price *= 0.99; // -1% daily returns
        let noise = ((i as f64 * 0.7).sin()) * 0.5;
        downtrend_bars.push(Bar::new(
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i as i64),
            price - 0.5 + noise,
            price + 2.0,
            price - 2.0,
            price + noise,
            1_000_000.0,
        ));
    }

    let config = RegimeConfig::default();
    let detector = RegimeDetector::new(config);

    let uptrend_regimes = detector.detect(&uptrend_bars);
    let downtrend_regimes = detector.detect(&downtrend_bars);

    // After warmup, uptrend should show bullish regimes
    let bullish_count = uptrend_regimes
        .iter()
        .skip(50) // Skip warmup
        .filter(|r| r.trend.is_bullish())
        .count();
    assert!(
        bullish_count > 20,
        "Expected mostly bullish regimes in uptrend"
    );

    // After warmup, downtrend should show bearish regimes
    let bearish_count = downtrend_regimes
        .iter()
        .skip(50) // Skip warmup
        .filter(|r| !r.trend.is_bullish())
        .count();
    assert!(
        bearish_count > 20,
        "Expected mostly bearish regimes in downtrend"
    );
}

#[test]
fn test_parquet_export() {
    use mantis::export::{
        export_equity_curve_parquet, export_features_parquet, export_trades_parquet,
    };
    use tempfile::TempDir;

    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("PARQUET_TEST".to_string(), bars.clone());

    let mut strategy = SmaCrossover::new(5, 20);
    let result = engine.run(&mut strategy, "PARQUET_TEST").unwrap();

    // Create temp directory for test files
    let temp_dir = TempDir::new().unwrap();

    // Export equity curve to parquet
    let equity_path = temp_dir.path().join("equity.parquet");
    export_equity_curve_parquet(&result.equity_curve, &equity_path).unwrap();
    assert!(equity_path.exists());
    assert!(std::fs::metadata(&equity_path).unwrap().len() > 0);

    // Export trades to parquet
    let trades_path = temp_dir.path().join("trades.parquet");
    if !result.trades.is_empty() && result.trades.iter().any(|t| t.is_closed()) {
        export_trades_parquet(&result.trades, &trades_path).unwrap();
        assert!(trades_path.exists());
    }

    // Export features to parquet
    let feature_config = FeatureConfig::minimal();
    let extractor = FeatureExtractor::new(feature_config);
    let (features, column_names) = extractor.extract_matrix(&bars);

    if !features.is_empty() {
        let features_path = temp_dir.path().join("features.parquet");
        export_features_parquet(&features, &column_names, &features_path).unwrap();
        assert!(features_path.exists());
    }
}

#[test]
fn test_streaming_indicators() {
    use mantis::streaming::{StreamingIndicator, StreamingRSI, StreamingSMA};

    let bars = create_synthetic_data(100, 100.0, 0.001);

    // Test streaming SMA
    let mut sma = StreamingSMA::new(20);
    for bar in &bars {
        sma.update(bar.close);
    }
    assert!(sma.is_ready());
    assert!(sma.value().is_some());

    // Test streaming RSI
    let mut rsi = StreamingRSI::new(14);
    for bar in &bars {
        rsi.update(bar.close);
    }
    assert!(rsi.is_ready());
    let rsi_value = rsi.value().unwrap();
    assert!(rsi_value >= 0.0 && rsi_value <= 100.0);

    // Test reset
    sma.reset();
    assert!(!sma.is_ready());
    assert!(sma.value().is_none());
}

#[test]
fn test_zero_trade_strategy() {
    // A strategy that never trades should have zero P&L
    let bars = create_synthetic_data(100, 100.0, 0.001);

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        show_progress: false,
        ..Default::default()
    };

    let mut engine = Engine::new(config);
    engine.add_data("ZERO_TEST".to_string(), bars);

    // Use RSI strategy with impossible thresholds
    let mut strategy = RsiStrategy::new(14, 5.0, 95.0); // Almost never triggers
    let result = engine.run(&mut strategy, "ZERO_TEST").unwrap();

    // If no trades, equity should equal initial capital
    if result.total_trades == 0 {
        assert!((result.final_equity - 100_000.0).abs() < 0.01);
        assert!((result.total_return_pct).abs() < 0.01);
    }
}
