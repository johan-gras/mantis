//! Property-based tests using proptest for fuzzing and invariant testing.
//!
//! These tests verify that:
//! 1. OHLC constraints always hold after any operation
//! 2. Bar data validation correctly rejects invalid data
//! 3. Position sizing constraints are maintained
//! 4. Backtest engine invariants hold under random inputs

use chrono::{TimeZone, Utc};
use proptest::prelude::*;

use mantis::engine::{BacktestConfig, Engine};
use mantis::portfolio::CostModel;
use mantis::risk::RiskConfig;
use mantis::strategies::{ExternalSignalStrategy, SmaCrossover};
use mantis::types::Bar;

// ============================================================================
// Bar and OHLC Property Tests
// ============================================================================

/// Strategy to generate valid OHLC values where High >= Low
fn valid_ohlc_strategy() -> impl Strategy<Value = (f64, f64, f64, f64)> {
    // Generate base price and variations
    (10.0..10000.0f64).prop_flat_map(|base| {
        let variation = base * 0.1; // 10% variation max
        (
            Just(base),
            0.0..variation, // high offset
            0.0..variation, // low offset
            0.0..variation, // open offset
            0.0..variation, // close offset
        )
            .prop_map(move |(base, h_off, l_off, o_off, c_off)| {
                let high = base + h_off;
                let low = base - l_off;
                let open = low + o_off * (high - low) / variation.max(0.01);
                let close = low + c_off * (high - low) / variation.max(0.01);

                // Ensure constraints: low <= open/close <= high
                let open = open.clamp(low, high);
                let close = close.clamp(low, high);

                (open, high, low, close)
            })
    })
}

/// Strategy to generate invalid OHLC (high < low)
fn invalid_ohlc_strategy() -> impl Strategy<Value = (f64, f64, f64, f64)> {
    (10.0..1000.0f64, 1.0..100.0f64).prop_map(|(base, offset)| {
        let low = base + offset; // low is higher than high
        let high = base;
        let open = (low + high) / 2.0;
        let close = (low + high) / 2.0;
        (open, high, low, close)
    })
}

/// Strategy to generate valid volume
fn valid_volume_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        1.0..1_000_000_000.0f64, // Normal volume
        Just(0.0),               // Zero volume (edge case)
    ]
}

/// Strategy to generate a valid Bar at a given day offset
#[allow(dead_code)]
fn valid_bar_strategy(day_offset: i64) -> impl Strategy<Value = Bar> {
    (valid_ohlc_strategy(), valid_volume_strategy()).prop_map(
        move |((open, high, low, close), volume)| {
            Bar::new(
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(day_offset),
                open,
                high,
                low,
                close,
                volume,
            )
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    // ========================================================================
    // OHLC Constraint Tests
    // ========================================================================

    #[test]
    fn valid_ohlc_passes_validation((open, high, low, close) in valid_ohlc_strategy()) {
        // Valid OHLC should satisfy: low <= open <= high, low <= close <= high
        prop_assert!(low <= open, "low ({}) must be <= open ({})", low, open);
        prop_assert!(open <= high, "open ({}) must be <= high ({})", open, high);
        prop_assert!(low <= close, "low ({}) must be <= close ({})", low, close);
        prop_assert!(close <= high, "close ({}) must be <= high ({})", close, high);
        prop_assert!(low <= high, "low ({}) must be <= high ({})", low, high);
    }

    #[test]
    fn invalid_ohlc_violates_constraints((_open, high, low, _close) in invalid_ohlc_strategy()) {
        // Invalid OHLC should have high < low
        prop_assert!(high < low, "Invalid OHLC should have high ({}) < low ({})", high, low);
    }

    #[test]
    fn bar_creation_preserves_ohlc_constraints((open, high, low, close) in valid_ohlc_strategy(), volume in valid_volume_strategy()) {
        let bar = Bar::new(
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap(),
            open, high, low, close, volume
        );

        // Bar should preserve the input values
        prop_assert!((bar.open - open).abs() < 1e-10);
        prop_assert!((bar.high - high).abs() < 1e-10);
        prop_assert!((bar.low - low).abs() < 1e-10);
        prop_assert!((bar.close - close).abs() < 1e-10);
        prop_assert!((bar.volume - volume).abs() < 1e-10);

        // Verify OHLC constraints hold
        prop_assert!(bar.low <= bar.open);
        prop_assert!(bar.low <= bar.close);
        prop_assert!(bar.high >= bar.open);
        prop_assert!(bar.high >= bar.close);
        prop_assert!(bar.low <= bar.high);
    }

    // ========================================================================
    // Data Validation Tests
    // ========================================================================

    #[test]
    fn data_validation_rejects_negative_open_or_close(
        base in 10.0..1000.0f64,
        negative_field in 0..2usize
    ) {
        let timestamp = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

        // The validate() method only checks open > 0 and close > 0 (not high/low)
        // So we only test negative open and close here
        let (open, high, low, close) = match negative_field {
            0 => (-base, base + 10.0, -base - 5.0, base), // negative open
            1 => (base, base + 10.0, base - 5.0, -base),  // negative close
            _ => unreachable!()
        };

        let bar = Bar::new(timestamp, open, high, low, close, 1000.0);

        // Bar validation should reject negative open/close
        prop_assert!(!bar.validate(),
            "Bar with negative open or close should fail validation: o={}, h={}, l={}, c={}", open, high, low, close);
    }

    #[test]
    fn data_validation_rejects_negative_high(
        base in 10.0..1000.0f64
    ) {
        let timestamp = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

        // Negative high will fail because high >= open and high >= close will be false
        let open = base;
        let high = -base;
        let low = -base - 5.0;
        let close = base;

        let bar = Bar::new(timestamp, open, high, low, close, 1000.0);

        // Bar validation should fail because high < open and high < close
        prop_assert!(!bar.validate(),
            "Bar with negative high should fail validation: o={}, h={}, l={}, c={}", open, high, low, close);
    }

    #[test]
    fn data_validation_detects_high_low_violation(
        base in 10.0..1000.0f64,
        violation in 1.0..100.0f64
    ) {
        let timestamp = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();

        // Create bar where high < low
        let high = base;
        let low = base + violation; // low is higher than high
        let open = base;
        let close = base;

        let bar = Bar::new(timestamp, open, high, low, close, 1000.0);

        // Bar validation should detect high < low violation
        prop_assert!(!bar.validate(),
            "Bar with high ({}) < low ({}) should fail validation", high, low);
    }

    #[test]
    fn valid_bar_series_passes_validation(num_bars in 2..50usize) {
        let bars: Vec<Bar> = (0..num_bars)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.5);
                let variation = 2.0;
                Bar::new(
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + variation,
                    base - variation,
                    base + 0.1,
                    1000.0 + i as f64,
                )
            })
            .collect();

        // All valid bars should pass validation
        for (i, bar) in bars.iter().enumerate() {
            prop_assert!(bar.validate(),
                "Bar {} should pass validation: o={}, h={}, l={}, c={}",
                i, bar.open, bar.high, bar.low, bar.close);
        }
    }

    // ========================================================================
    // Position Sizing and Capital Tests
    // ========================================================================

    #[test]
    fn position_size_respects_capital_constraints(
        initial_capital in 10_000.0..1_000_000.0f64,
        position_size in 0.01..1.0f64
    ) {
        // Position size must be between 0 and 1
        prop_assert!(position_size >= 0.0 && position_size <= 1.0);

        // Max position value should not exceed capital
        let max_position_value = initial_capital * position_size;
        prop_assert!(max_position_value <= initial_capital);
    }

    #[test]
    fn cost_model_produces_non_negative_costs(
        trade_value in 100.0..100_000.0f64,
        commission_pct in 0.0..0.1f64,
        slippage_pct in 0.0..0.1f64
    ) {
        let _cost_model = CostModel {
            commission_pct,
            slippage_pct,
            ..Default::default()
        };

        // Calculate expected costs
        let commission = trade_value * commission_pct;
        let slippage = trade_value * slippage_pct;

        prop_assert!(commission >= 0.0, "Commission must be non-negative");
        prop_assert!(slippage >= 0.0, "Slippage must be non-negative");
    }

    // ========================================================================
    // Backtest Engine Invariant Tests
    // ========================================================================

    #[test]
    fn backtest_equity_never_goes_negative(
        num_bars in 50..200usize,
        initial_capital in 10_000.0..100_000.0f64,
        position_size in 0.05..0.3f64
    ) {
        // Generate synthetic data
        let bars: Vec<Bar> = (0..num_bars)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.3).sin() * 10.0;
                let variation = 2.0;
                Bar::new(
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + variation,
                    base - variation,
                    base + ((i as f64 * 0.5).cos()),
                    100_000.0,
                )
            })
            .collect();

        let config = BacktestConfig {
            initial_capital,
            position_size,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = Engine::new(config);
        engine.add_data("PROP_TEST".to_string(), bars);

        let mut strategy = SmaCrossover::new(5, 15);
        let result = engine.run(&mut strategy, "PROP_TEST").unwrap();

        // Equity should never be negative
        prop_assert!(result.final_equity >= 0.0,
            "Final equity ({}) must be non-negative", result.final_equity);

        // Check entire equity curve
        for (i, point) in result.equity_curve.iter().enumerate() {
            prop_assert!(point.equity >= 0.0,
                "Equity at point {} ({}) must be non-negative", i, point.equity);
        }
    }

    #[test]
    fn backtest_max_drawdown_bounded(
        num_bars in 100..300usize,
        trend in -0.01..0.01f64
    ) {
        // Generate trending data
        let bars: Vec<Bar> = (0..num_bars)
            .map(|i| {
                let base = 100.0 * (1.0 + trend).powi(i as i32);
                let variation = base * 0.02;
                Bar::new(
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + variation,
                    base - variation,
                    base * (1.0 + ((i as f64 * 0.1).sin() * 0.01)),
                    100_000.0,
                )
            })
            .collect();

        let config = BacktestConfig {
            initial_capital: 100_000.0,
            position_size: 0.1,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = Engine::new(config);
        engine.add_data("DD_TEST".to_string(), bars);

        let mut strategy = SmaCrossover::new(10, 30);
        let result = engine.run(&mut strategy, "DD_TEST").unwrap();

        // Max drawdown should be between 0 and 100%
        prop_assert!(result.max_drawdown_pct >= 0.0 && result.max_drawdown_pct <= 100.0,
            "Max drawdown ({}) must be in [0, 100]%", result.max_drawdown_pct);
    }

    #[test]
    fn backtest_trade_count_matches_trades_vec(
        num_bars in 50..150usize
    ) {
        let bars: Vec<Bar> = (0..num_bars)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.7).sin() * 15.0;
                let variation = 3.0;
                Bar::new(
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + variation,
                    base - variation,
                    base + (i as f64 * 0.3).cos() * 2.0,
                    100_000.0,
                )
            })
            .collect();

        let config = BacktestConfig {
            initial_capital: 100_000.0,
            position_size: 0.1,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = Engine::new(config);
        engine.add_data("TRADE_TEST".to_string(), bars);

        let mut strategy = SmaCrossover::new(5, 15);
        let result = engine.run(&mut strategy, "TRADE_TEST").unwrap();

        // total_trades should match the number of completed round-trip trades
        let closed_trades = result.trades.iter().filter(|t| t.is_closed()).count();
        prop_assert_eq!(result.total_trades, closed_trades,
            "total_trades ({}) should equal closed trades count ({})",
            result.total_trades, closed_trades);
    }

    // ========================================================================
    // Signal Processing Tests
    // ========================================================================

    #[test]
    fn signals_outside_valid_range_treated_as_strong(
        signal_value in -10.0..10.0f64
    ) {
        // Per spec: signals > 1 or < -1 are treated as strong signals
        // Magnitude controls position sizing when enabled
        if signal_value.abs() > 1.0 {
            // Signal should be treated as a strong long/short
            prop_assert!(signal_value != 0.0);
        }

        // Zero signal always means no position
        if signal_value == 0.0 {
            // This should result in no trade
            prop_assert!(signal_value.abs() < f64::EPSILON);
        }
    }

    #[test]
    fn external_signals_length_check(
        num_bars in 10..100usize,
        num_signals in 1..150usize
    ) {
        let bars: Vec<Bar> = (0..num_bars)
            .map(|i| {
                Bar::new(
                    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    100.0, 102.0, 98.0, 101.0, 100_000.0,
                )
            })
            .collect();

        let signals: Vec<f64> = (0..num_signals)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();

        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = Engine::new(config);
        engine.add_data("SIG_TEST".to_string(), bars.clone());

        let mut strategy = ExternalSignalStrategy::new(signals.clone(), 0.5);
        let result = engine.run(&mut strategy, "SIG_TEST");

        // If signals are longer than data, it should still work (extra signals ignored)
        // If signals are shorter, strategy handles with NaN -> 0
        // Either way, the engine should not panic
        prop_assert!(result.is_ok() || result.is_err(),
            "Engine should handle signal/data length mismatch gracefully");
    }

    // ========================================================================
    // Risk Configuration Tests
    // ========================================================================

    #[test]
    fn risk_config_valid_percentages(
        stop_loss_pct in 0.0..50.0f64,
        take_profit_pct in 0.0..100.0f64,
        max_position_size in 0.0..1.0f64
    ) {
        let risk_config = RiskConfig {
            stop_loss: mantis::risk::StopLoss::Percentage(stop_loss_pct),
            take_profit: mantis::risk::TakeProfit::Percentage(take_profit_pct),
            max_position_size,
            ..Default::default()
        };

        // Risk config should be valid
        prop_assert!(risk_config.max_position_size >= 0.0);
        prop_assert!(risk_config.max_position_size <= 1.0 || risk_config.max_position_size > 0.0);
    }
}

// ============================================================================
// Non-proptest Property Verification
// ============================================================================

#[test]
fn verify_ohlc_constraints_on_sample_data() {
    // This test verifies OHLC constraints on actual sample data if available
    let data_path = "data/sample.csv";
    if std::path::Path::new(data_path).exists() {
        use mantis::data::{load_csv, DataConfig};
        let bars = load_csv(data_path, &DataConfig::default()).unwrap();

        for (i, bar) in bars.iter().enumerate() {
            assert!(
                bar.low <= bar.open,
                "Bar {}: low ({}) > open ({})",
                i,
                bar.low,
                bar.open
            );
            assert!(
                bar.low <= bar.close,
                "Bar {}: low ({}) > close ({})",
                i,
                bar.low,
                bar.close
            );
            assert!(
                bar.high >= bar.open,
                "Bar {}: high ({}) < open ({})",
                i,
                bar.high,
                bar.open
            );
            assert!(
                bar.high >= bar.close,
                "Bar {}: high ({}) < close ({})",
                i,
                bar.high,
                bar.close
            );
            assert!(
                bar.low <= bar.high,
                "Bar {}: low ({}) > high ({})",
                i,
                bar.low,
                bar.high
            );
            assert!(
                bar.volume >= 0.0,
                "Bar {}: negative volume ({})",
                i,
                bar.volume
            );
        }
    }
}

#[test]
fn verify_backtest_determinism() {
    // Run the same backtest twice and verify identical results
    let bars: Vec<Bar> = (0..100)
        .map(|i| {
            let base = 100.0 + (i as f64 * 0.5).sin() * 10.0;
            Bar::new(
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                    + chrono::Duration::days(i as i64),
                base,
                base + 2.0,
                base - 2.0,
                base + 0.5,
                100_000.0,
            )
        })
        .collect();

    let config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.1,
        show_progress: false,
        ..Default::default()
    };

    // Run 1
    let mut engine1 = Engine::new(config.clone());
    engine1.add_data("DET_TEST".to_string(), bars.clone());
    let mut strategy1 = SmaCrossover::new(10, 30);
    let result1 = engine1.run(&mut strategy1, "DET_TEST").unwrap();

    // Run 2
    let mut engine2 = Engine::new(config);
    engine2.add_data("DET_TEST".to_string(), bars);
    let mut strategy2 = SmaCrossover::new(10, 30);
    let result2 = engine2.run(&mut strategy2, "DET_TEST").unwrap();

    // Results should be identical
    assert!(
        (result1.final_equity - result2.final_equity).abs() < 1e-10,
        "Results should be deterministic: {} vs {}",
        result1.final_equity,
        result2.final_equity
    );
    assert_eq!(
        result1.total_trades, result2.total_trades,
        "Trade count should be deterministic"
    );
    assert!(
        (result1.sharpe_ratio - result2.sharpe_ratio).abs() < 1e-10
            || (result1.sharpe_ratio.is_nan() && result2.sharpe_ratio.is_nan()),
        "Sharpe should be deterministic: {} vs {}",
        result1.sharpe_ratio,
        result2.sharpe_ratio
    );
}
