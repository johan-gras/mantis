//! Example: Walk-Forward Analysis
//!
//! Walk-forward analysis helps prevent overfitting by:
//! 1. Dividing data into multiple windows
//! 2. Optimizing on in-sample data
//! 3. Testing on out-of-sample data
//! 4. Rolling forward through time
//!
//! This provides a more realistic estimate of strategy performance.
//!
//! Run with: cargo run --example walkforward

use ralph_backtest::data::{load_csv, DataConfig};
use ralph_backtest::engine::BacktestConfig;
use ralph_backtest::strategies::SmaCrossover;
use ralph_backtest::types::Bar;
use ralph_backtest::walkforward::{WalkForwardAnalyzer, WalkForwardConfig, WalkForwardMetric};
use chrono::{TimeZone, Utc};

/// Generate synthetic data with varying trends.
fn generate_synthetic_data(days: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(days);
    let mut price = 100.0;

    for i in 0..days {
        // Regime changes
        let trend = if i < days / 4 {
            0.001 // Uptrend
        } else if i < days / 2 {
            -0.0005 // Downtrend
        } else if i < 3 * days / 4 {
            0.0008 // Uptrend
        } else {
            -0.0003 // Downtrend
        };

        let noise = ((i as f64 * 0.7).sin() * 2.0 + (i as f64 * 1.3).cos()) * 0.5;
        let daily_change = price * trend + noise;
        price += daily_change;
        price = price.max(50.0);

        let volatility = 1.0 + (noise.abs() * 0.3);
        bars.push(Bar::new(
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
                + chrono::Duration::days(i as i64),
            price - volatility * 0.3,
            price + volatility,
            price - volatility,
            price,
            1_000_000.0,
        ));
    }

    bars
}

fn main() {
    println!("=== Walk-Forward Analysis ===\n");

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

    // 2. Configure walk-forward analysis
    println!("2. Configuring walk-forward analysis...");
    let wf_config = WalkForwardConfig {
        num_windows: 5,         // 5 walk-forward windows
        in_sample_ratio: 0.7,   // 70% in-sample, 30% out-of-sample
        anchored: false,        // Rolling windows (not expanding)
        min_bars_per_window: 50,
    };
    println!("   Windows: {}", wf_config.num_windows);
    println!("   In-sample ratio: {:.0}%", wf_config.in_sample_ratio * 100.0);
    println!("   Mode: {}\n", if wf_config.anchored { "Anchored" } else { "Rolling" });

    // 3. Define parameter space for optimization
    println!("3. Defining parameter space...");
    let params: Vec<(usize, usize)> = vec![
        (5, 15),
        (5, 20),
        (5, 30),
        (10, 20),
        (10, 30),
        (10, 40),
        (15, 30),
        (15, 45),
        (20, 40),
        (20, 50),
    ];
    println!("   Testing {} parameter combinations\n", params.len());

    // 4. Configure backtest
    let backtest_config = BacktestConfig {
        initial_capital: 100_000.0,
        position_size: 0.95,
        show_progress: false,
        ..Default::default()
    };

    // 5. Run walk-forward analysis
    println!("4. Running walk-forward analysis...");
    let analyzer = WalkForwardAnalyzer::new(wf_config.clone(), backtest_config);

    // Strategy factory creates strategies from parameters
    let strategy_factory = |params: &(usize, usize)| -> Box<dyn ralph_backtest::Strategy> {
        Box::new(SmaCrossover::new(params.0, params.1))
    };

    let result = analyzer
        .run(
            &bars,
            "TEST",
            params,
            strategy_factory,
            WalkForwardMetric::Sharpe, // Optimize for Sharpe ratio
        )
        .expect("Walk-forward analysis failed");

    // 6. Display results
    println!("\n5. Walk-Forward Results:\n");
    println!("{}", "=".repeat(70));

    for (i, window) in result.windows.iter().enumerate() {
        println!("\nWindow {} of {}:", i + 1, result.windows.len());
        println!(
            "  In-Sample:     {:>8.2}% return, {:.2} Sharpe",
            window.in_sample_result.total_return_pct,
            window.in_sample_result.sharpe_ratio
        );
        println!(
            "  Out-of-Sample: {:>8.2}% return, {:.2} Sharpe",
            window.out_of_sample_result.total_return_pct,
            window.out_of_sample_result.sharpe_ratio
        );
        println!(
            "  Efficiency:    {:>8.2}%",
            window.efficiency_ratio * 100.0
        );
    }

    println!("\n{}", "=".repeat(70));
    println!("\n{}", result.summary());

    // 7. Robustness check
    println!("\n6. Robustness Check:");
    if result.is_robust(0.5) {
        println!("   PASSED: Strategy shows robust out-of-sample performance");
        println!("   Walk-forward efficiency: {:.1}%", result.walk_forward_efficiency * 100.0);
    } else {
        println!("   FAILED: Strategy may be overfit");
        println!("   Walk-forward efficiency: {:.1}%", result.walk_forward_efficiency * 100.0);
        println!("\n   Suggestions:");
        println!("   - Use simpler strategy");
        println!("   - Reduce number of parameters");
        println!("   - Use longer in-sample period");
        println!("   - Consider different market regime");
    }

    // 8. Anchored analysis comparison
    println!("\n7. Comparing with Anchored Windows...\n");

    let anchored_config = WalkForwardConfig {
        anchored: true,
        ..wf_config
    };

    let anchored_analyzer = WalkForwardAnalyzer::new(
        anchored_config,
        BacktestConfig {
            initial_capital: 100_000.0,
            position_size: 0.95,
            show_progress: false,
            ..Default::default()
        },
    );

    let anchored_params: Vec<(usize, usize)> = vec![
        (5, 20),
        (10, 30),
        (15, 45),
    ];

    if let Ok(anchored_result) = anchored_analyzer.run(
        &bars,
        "TEST",
        anchored_params,
        |params: &(usize, usize)| Box::new(SmaCrossover::new(params.0, params.1)),
        WalkForwardMetric::Sharpe,
    ) {
        println!("   Anchored WF Efficiency: {:.1}%", anchored_result.walk_forward_efficiency * 100.0);
        println!("   Anchored Avg OOS Return: {:.2}%", anchored_result.avg_oos_return);
    }

    println!("\n=== Analysis Complete ===");
}
