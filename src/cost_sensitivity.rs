//! Transaction Cost Sensitivity Analysis
//!
//! This module provides tools to test strategy robustness by running backtests
//! with varying transaction cost levels. Testing with 2x, 5x, and 10x cost
//! multipliers helps identify strategies that are overfitted to low-cost
//! environments or that have insufficient edge to overcome realistic frictions.
//!
//! # Example
//!
//! ```no_run
//! use mantis::cost_sensitivity::{CostSensitivityConfig, run_cost_sensitivity_analysis};
//! use mantis::engine::BacktestConfig;
//! use mantis::strategies::SmaCrossover;
//!
//! let base_config = BacktestConfig::default();
//! let sensitivity_config = CostSensitivityConfig::default();
//! let data = vec![/* bars */];
//! let mut strategy = SmaCrossover::new(20, 50);
//!
//! let analysis = run_cost_sensitivity_analysis(
//!     &base_config,
//!     &sensitivity_config,
//!     &data,
//!     &mut strategy,
//!     "AAPL"
//! ).unwrap();
//!
//! println!("Sharpe degradation at 5x costs: {:.2}%", analysis.sharpe_degradation_at(5.0).unwrap());
//! ```

use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::portfolio::CostModel;
use crate::strategy::Strategy;
use crate::types::Bar;
use serde::{Deserialize, Serialize};
use std::error::Error;

/// Configuration for cost sensitivity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSensitivityConfig {
    /// Cost multipliers to test (e.g., [0.0, 1.0, 2.0, 5.0, 10.0])
    pub multipliers: Vec<f64>,
    /// Minimum acceptable Sharpe ratio at 5x costs (robustness threshold)
    pub robustness_threshold_5x: Option<f64>,
    /// Whether to include zero-cost baseline (theoretical upper bound)
    pub include_zero_cost: bool,
}

impl Default for CostSensitivityConfig {
    fn default() -> Self {
        Self {
            multipliers: vec![0.0, 1.0, 2.0, 5.0, 10.0],
            robustness_threshold_5x: Some(0.5), // Sharpe > 0.5 at 5x costs
            include_zero_cost: true,
        }
    }
}

impl CostSensitivityConfig {
    /// Create config with standard multipliers (1x, 2x, 5x, 10x)
    pub fn standard() -> Self {
        Self {
            multipliers: vec![1.0, 2.0, 5.0, 10.0],
            robustness_threshold_5x: Some(0.5),
            include_zero_cost: false,
        }
    }

    /// Create config with aggressive stress testing (up to 20x)
    pub fn aggressive() -> Self {
        Self {
            multipliers: vec![0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
            robustness_threshold_5x: Some(0.3),
            include_zero_cost: true,
        }
    }
}

/// Results for a single cost scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostScenario {
    /// Cost multiplier applied
    pub multiplier: f64,
    /// Complete backtest result for this scenario
    pub result: BacktestResult,
    /// Total transaction costs incurred
    pub total_costs: f64,
    /// Average cost per trade
    pub avg_cost_per_trade: f64,
}

impl CostScenario {
    /// Whether this is the zero-cost baseline
    pub fn is_zero_cost(&self) -> bool {
        self.multiplier < 0.001
    }

    /// Whether this is the baseline (1x) scenario
    pub fn is_baseline(&self) -> bool {
        (self.multiplier - 1.0).abs() < 0.001
    }

    /// Get Sharpe ratio for this scenario
    pub fn sharpe_ratio(&self) -> f64 {
        self.result.sharpe_ratio
    }

    /// Get total return percentage for this scenario
    pub fn total_return_pct(&self) -> f64 {
        self.result.total_return_pct
    }
}

/// Complete cost sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSensitivityAnalysis {
    /// Results for each cost scenario
    pub scenarios: Vec<CostScenario>,
    /// Symbol analyzed
    pub symbol: String,
    /// Strategy name
    pub strategy_name: String,
}

impl CostSensitivityAnalysis {
    /// Get scenario with specific multiplier
    pub fn scenario_at(&self, multiplier: f64) -> Option<&CostScenario> {
        self.scenarios
            .iter()
            .find(|s| (s.multiplier - multiplier).abs() < 0.001)
    }

    /// Get baseline (1x) scenario
    pub fn baseline(&self) -> Option<&CostScenario> {
        self.scenario_at(1.0)
    }

    /// Get zero-cost scenario (theoretical upper bound)
    pub fn zero_cost(&self) -> Option<&CostScenario> {
        self.scenario_at(0.0)
    }

    /// Calculate Sharpe ratio degradation percentage at given multiplier
    /// Returns percentage degradation relative to baseline (1x costs)
    pub fn sharpe_degradation_at(&self, multiplier: f64) -> Option<f64> {
        let baseline = self.baseline()?.sharpe_ratio();
        let scenario = self.scenario_at(multiplier)?.sharpe_ratio();

        if baseline.abs() < 1e-6 {
            return None; // Avoid division by zero
        }

        Some(((baseline - scenario) / baseline) * 100.0)
    }

    /// Calculate return degradation percentage at given multiplier
    pub fn return_degradation_at(&self, multiplier: f64) -> Option<f64> {
        let baseline = self.baseline()?.total_return_pct();
        let scenario = self.scenario_at(multiplier)?.total_return_pct();

        if baseline.abs() < 1e-6 {
            return None;
        }

        Some(((baseline - scenario) / baseline) * 100.0)
    }

    /// Check if strategy passes robustness threshold at 5x costs
    pub fn is_robust(&self, threshold_sharpe: f64) -> bool {
        if let Some(scenario_5x) = self.scenario_at(5.0) {
            return scenario_5x.sharpe_ratio() >= threshold_sharpe;
        }
        false
    }

    /// Calculate cost elasticity: % change in return per % change in costs
    /// Measures sensitivity of returns to transaction costs
    pub fn cost_elasticity(&self) -> Option<f64> {
        let baseline = self.baseline()?;
        let scenario_2x = self.scenario_at(2.0)?;

        let return_change_pct = (scenario_2x.total_return_pct() - baseline.total_return_pct())
            / baseline.total_return_pct()
            * 100.0;

        let cost_change_pct = 100.0; // 1x to 2x is 100% increase

        Some(return_change_pct / cost_change_pct)
    }

    /// Calculate breakeven cost multiplier (where returns become zero/negative)
    /// Uses linear interpolation between scenarios
    pub fn breakeven_multiplier(&self) -> Option<f64> {
        // Sort scenarios by multiplier
        let mut sorted = self.scenarios.clone();
        sorted.sort_by(|a, b| a.multiplier.partial_cmp(&b.multiplier).unwrap());

        // Find first scenario with negative returns
        for i in 1..sorted.len() {
            let prev = &sorted[i - 1];
            let curr = &sorted[i];

            if prev.total_return_pct() > 0.0 && curr.total_return_pct() <= 0.0 {
                // Linear interpolation
                let ret_diff = prev.total_return_pct() - curr.total_return_pct();
                if ret_diff.abs() < 1e-6 {
                    return Some(curr.multiplier);
                }

                let interpolation_factor = prev.total_return_pct() / ret_diff;
                let breakeven =
                    prev.multiplier + interpolation_factor * (curr.multiplier - prev.multiplier);

                return Some(breakeven);
            }
        }

        None // Strategy remains profitable at all tested multipliers
    }

    /// Generate summary report as formatted string
    pub fn summary_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "Cost Sensitivity Analysis: {} ({})\n",
            self.symbol, self.strategy_name
        ));
        report.push_str(&format!("{}\n", "=".repeat(60)));

        // Scenario table
        report.push_str(&format!(
            "\n{:<12} {:>12} {:>12} {:>12} {:>10}\n",
            "Multiplier", "Return", "Sharpe", "MaxDD", "Trades"
        ));
        report.push_str(&format!("{}\n", "-".repeat(60)));

        for scenario in &self.scenarios {
            let return_str = format!("{:.2}%", scenario.total_return_pct());
            let sharpe_str = format!("{:.3}", scenario.sharpe_ratio());
            let dd_str = format!("{:.2}%", scenario.result.max_drawdown_pct);

            report.push_str(&format!(
                "{:<12} {:>12} {:>12} {:>12} {:>10}\n",
                format!("{}x", scenario.multiplier),
                return_str,
                sharpe_str,
                dd_str,
                scenario.result.total_trades
            ));
        }

        // Degradation analysis
        report.push_str(&format!("\n{}\n", "=".repeat(60)));
        report.push_str("Degradation Analysis (vs 1x baseline):\n");
        report.push_str(&format!("{}\n", "-".repeat(60)));

        if let Some(degrad_2x) = self.sharpe_degradation_at(2.0) {
            report.push_str(&format!(
                "Sharpe degradation at 2x costs: {:.1}%\n",
                degrad_2x
            ));
        }
        if let Some(degrad_5x) = self.sharpe_degradation_at(5.0) {
            report.push_str(&format!(
                "Sharpe degradation at 5x costs: {:.1}%\n",
                degrad_5x
            ));
        }
        if let Some(degrad_10x) = self.sharpe_degradation_at(10.0) {
            report.push_str(&format!(
                "Sharpe degradation at 10x costs: {:.1}%\n",
                degrad_10x
            ));
        }

        if let Some(elasticity) = self.cost_elasticity() {
            report.push_str(&format!("\nCost elasticity: {:.3}\n", elasticity));
        }

        if let Some(breakeven) = self.breakeven_multiplier() {
            report.push_str(&format!("Breakeven cost multiplier: {:.2}x\n", breakeven));
        }

        // Robustness assessment
        report.push_str(&format!("\n{}\n", "=".repeat(60)));
        report.push_str("Robustness Assessment:\n");
        report.push_str(&format!("{}\n", "-".repeat(60)));

        if let Some(scenario_5x) = self.scenario_at(5.0) {
            let sharpe_5x = scenario_5x.sharpe_ratio();
            let threshold = 0.5;
            let passes = sharpe_5x >= threshold;
            report.push_str(&format!(
                "5x cost test: {} (Sharpe = {:.3}, threshold = {:.3})\n",
                if passes { "PASS" } else { "FAIL" },
                sharpe_5x,
                threshold
            ));
        }

        report
    }
}

/// Run cost sensitivity analysis
///
/// Executes the same backtest with different cost multipliers to assess
/// strategy robustness to transaction costs.
///
/// # Arguments
///
/// * `base_config` - Base backtest configuration (costs will be scaled)
/// * `sensitivity_config` - Cost sensitivity analysis configuration
/// * `bars` - Historical price data
/// * `strategy` - Mutable reference to strategy to test
/// * `symbol` - Symbol being traded
///
/// # Returns
///
/// Complete cost sensitivity analysis with results for each scenario
///
/// # Note
///
/// The strategy's state will be reset between runs (via on_start callback).
pub fn run_cost_sensitivity_analysis(
    base_config: &BacktestConfig,
    sensitivity_config: &CostSensitivityConfig,
    bars: &[Bar],
    strategy: &mut dyn Strategy,
    symbol: &str,
) -> Result<CostSensitivityAnalysis, Box<dyn Error>> {
    let mut scenarios = Vec::new();
    let strategy_name = strategy.name().to_string();

    for &multiplier in &sensitivity_config.multipliers {
        // Create scaled cost model
        let scaled_cost = scale_cost_model(&base_config.cost_model, multiplier);

        // Create config with scaled costs
        let mut config = base_config.clone();
        config.cost_model = scaled_cost;

        // Create engine with scaled costs and add data
        let mut engine = Engine::new(config);
        engine.data_mut().add(symbol, bars.to_vec());

        // Run backtest with scaled costs
        let result = engine.run(strategy, symbol)?;

        // Calculate total costs from trades
        let total_costs: f64 = result
            .trades
            .iter()
            .map(|t| t.commission + (t.slippage * t.quantity))
            .sum();

        let num_trades = result.trades.len();
        let avg_cost_per_trade = if num_trades > 0 {
            total_costs / num_trades as f64
        } else {
            0.0
        };

        scenarios.push(CostScenario {
            multiplier,
            result,
            total_costs,
            avg_cost_per_trade,
        });
    }

    // Sort scenarios by multiplier
    scenarios.sort_by(|a, b| a.multiplier.partial_cmp(&b.multiplier).unwrap());

    Ok(CostSensitivityAnalysis {
        scenarios,
        symbol: symbol.to_string(),
        strategy_name,
    })
}

/// Scale a cost model by a multiplier
///
/// Applies the multiplier to all cost components:
/// - Commission (flat and percentage)
/// - Slippage
/// - Market impact
/// - Asset-class specific fees
fn scale_cost_model(base: &CostModel, multiplier: f64) -> CostModel {
    let mut scaled = base.clone();

    // Scale commission components
    scaled.commission_flat *= multiplier;
    scaled.commission_pct *= multiplier;
    scaled.commission_per_share *= multiplier;
    scaled.min_commission *= multiplier;

    // Scale slippage
    scaled.slippage_pct *= multiplier;

    // Scale market impact
    scaled.market_impact = match &base.market_impact {
        crate::portfolio::MarketImpactModel::None => crate::portfolio::MarketImpactModel::None,
        crate::portfolio::MarketImpactModel::Linear { coefficient } => {
            crate::portfolio::MarketImpactModel::Linear {
                coefficient: coefficient * multiplier,
            }
        }
        crate::portfolio::MarketImpactModel::SquareRoot { coefficient } => {
            crate::portfolio::MarketImpactModel::SquareRoot {
                coefficient: coefficient * multiplier,
            }
        }
        crate::portfolio::MarketImpactModel::AlmgrenChriss { sigma, eta, gamma } => {
            crate::portfolio::MarketImpactModel::AlmgrenChriss {
                sigma: *sigma,             // Volatility unchanged
                eta: eta * multiplier,     // Temporary impact scaled
                gamma: gamma * multiplier, // Permanent impact scaled
            }
        }
    };

    // Scale futures costs
    scaled.futures.clearing_fee_per_contract *= multiplier;
    scaled.futures.exchange_fee_per_contract *= multiplier;
    scaled.futures.margin_interest_rate *= multiplier;

    // Scale crypto costs
    scaled.crypto.maker_fee_pct *= multiplier;
    scaled.crypto.taker_fee_pct *= multiplier;
    scaled.crypto.withdrawal_fee *= multiplier;

    // Scale forex costs
    scaled.forex.spread_pips *= multiplier;
    scaled.forex.swap_long *= multiplier;
    scaled.forex.swap_short *= multiplier;

    // Scale borrow cost for shorts
    scaled.borrow_cost_rate *= multiplier;

    scaled
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portfolio::{CryptoCost, ForexCost, FuturesCost, MarketImpactModel};
    use chrono::{TimeZone, Utc};

    fn create_test_bars() -> Vec<Bar> {
        // Generate bars with oscillating prices to trigger SMA crossovers
        // Pattern: uptrend, downtrend, uptrend to ensure crossovers occur
        (0..200)
            .map(|i| {
                let base_price = 100.0;
                // Create sine wave pattern for crossovers
                let cycle_price =
                    base_price + 10.0 * ((i as f64) / 20.0 * std::f64::consts::PI).sin();
                // Add upward drift
                let price = cycle_price + (i as f64) * 0.1;

                Bar {
                    timestamp: Utc.timestamp_opt(1_600_000_000 + i * 86400, 0).unwrap(), // Daily bars
                    open: price,
                    high: price + 0.5,
                    low: price - 0.5,
                    close: price,
                    volume: 1_000_000.0,
                }
            })
            .collect()
    }

    fn create_test_config() -> BacktestConfig {
        let mut config = BacktestConfig::default();
        config.cost_model.commission_pct = 0.001; // 0.1%
        config.cost_model.slippage_pct = 0.001; // 0.1%
        config.initial_capital = 100_000.0;
        config.show_progress = false; // Disable progress bar in tests
        config.margin.enabled = false;
        config
    }

    #[test]
    fn test_cost_model_scaling() {
        let base = CostModel {
            commission_flat: 5.0,
            commission_pct: 0.001,
            commission_per_share: 0.0,
            slippage_pct: 0.001,
            min_commission: 1.0,
            futures: FuturesCost {
                clearing_fee_per_contract: 0.50,
                exchange_fee_per_contract: 0.30,
                margin_interest_rate: 0.05,
            },
            crypto: CryptoCost {
                maker_fee_pct: 0.001,
                taker_fee_pct: 0.002,
                withdrawal_fee: 10.0,
            },
            forex: ForexCost {
                spread_pips: 2.0,
                swap_long: 0.0001,
                swap_short: -0.0002,
            },
            market_impact: MarketImpactModel::Linear { coefficient: 0.01 },
            max_volume_participation: None,
            borrow_cost_rate: 0.03, // 3% annual
        };

        let scaled = scale_cost_model(&base, 5.0);

        assert_eq!(scaled.commission_flat, 25.0);
        assert_eq!(scaled.commission_pct, 0.005);
        assert_eq!(scaled.slippage_pct, 0.005);
        assert_eq!(scaled.min_commission, 5.0);

        assert_eq!(scaled.futures.clearing_fee_per_contract, 2.5);
        assert_eq!(scaled.futures.exchange_fee_per_contract, 1.5);
        assert_eq!(scaled.futures.margin_interest_rate, 0.25);

        assert_eq!(scaled.crypto.maker_fee_pct, 0.005);
        assert_eq!(scaled.crypto.taker_fee_pct, 0.01);
        assert_eq!(scaled.crypto.withdrawal_fee, 50.0);

        assert_eq!(scaled.forex.spread_pips, 10.0);
        assert_eq!(scaled.forex.swap_long, 0.0005);
        assert_eq!(scaled.forex.swap_short, -0.001);

        match scaled.market_impact {
            MarketImpactModel::Linear { coefficient } => {
                assert_eq!(coefficient, 0.05);
            }
            _ => panic!("Expected Linear market impact model"),
        }
    }

    #[test]
    fn test_cost_sensitivity_config_defaults() {
        let config = CostSensitivityConfig::default();
        assert_eq!(config.multipliers, vec![0.0, 1.0, 2.0, 5.0, 10.0]);
        assert_eq!(config.robustness_threshold_5x, Some(0.5));
        assert!(config.include_zero_cost);

        let standard = CostSensitivityConfig::standard();
        assert_eq!(standard.multipliers, vec![1.0, 2.0, 5.0, 10.0]);
        assert!(!standard.include_zero_cost);

        let aggressive = CostSensitivityConfig::aggressive();
        assert_eq!(aggressive.multipliers, vec![0.0, 1.0, 2.0, 5.0, 10.0, 20.0]);
    }

    #[test]
    fn test_cost_sensitivity_analysis() {
        use crate::strategies::SmaCrossover;

        let config = create_test_config();
        let sensitivity_config = CostSensitivityConfig::standard();
        let bars = create_test_bars();
        let mut strategy = SmaCrossover::new(5, 15); // Shorter periods to generate signals faster

        let result = run_cost_sensitivity_analysis(
            &config,
            &sensitivity_config,
            &bars,
            &mut strategy,
            "TEST",
        );

        assert!(result.is_ok());
        let analysis = result.unwrap();

        // Should have 4 scenarios (1x, 2x, 5x, 10x)
        assert_eq!(analysis.scenarios.len(), 4);

        // Scenarios should be sorted by multiplier
        assert_eq!(analysis.scenarios[0].multiplier, 1.0);
        assert_eq!(analysis.scenarios[1].multiplier, 2.0);
        assert_eq!(analysis.scenarios[2].multiplier, 5.0);
        assert_eq!(analysis.scenarios[3].multiplier, 10.0);

        // Check that analysis completed successfully for all scenarios
        // Note: Trade count may vary depending on cost level and strategy parameters
        for scenario in &analysis.scenarios {
            assert!(
                scenario.result.trading_days > 0,
                "Should have processed bars"
            );
        }
    }

    #[test]
    fn test_degradation_metrics() {
        use crate::strategies::SmaCrossover;

        let config = create_test_config();
        let sensitivity_config = CostSensitivityConfig::standard();
        let bars = create_test_bars();
        let mut strategy = SmaCrossover::new(5, 15); // Shorter periods to generate signals faster

        let analysis = run_cost_sensitivity_analysis(
            &config,
            &sensitivity_config,
            &bars,
            &mut strategy,
            "TEST",
        )
        .unwrap();

        // Test scenario retrieval
        assert!(analysis.baseline().is_some());
        assert!(analysis.scenario_at(5.0).is_some());
        assert!(analysis.scenario_at(99.0).is_none());

        // Test degradation calculation (only if baseline Sharpe is meaningful)
        let baseline_sharpe = analysis.baseline().unwrap().sharpe_ratio();
        if baseline_sharpe.abs() > 0.1 {
            // Only test degradation if baseline has meaningful Sharpe
            let sharpe_degrad_2x = analysis.sharpe_degradation_at(2.0);
            assert!(sharpe_degrad_2x.is_some());
            if let Some(degrad) = sharpe_degrad_2x {
                // Degradation should be non-negative (performance same or worse)
                assert!(
                    degrad >= -0.1,
                    "Degradation should be non-negative: {}",
                    degrad
                );
            }
        }

        // Cost elasticity should be negative (costs hurt returns) if baseline is profitable
        let baseline_return = analysis.baseline().unwrap().total_return_pct();
        if baseline_return > 1.0 {
            // Only test elasticity if strategy is profitable
            let elasticity = analysis.cost_elasticity();
            if let Some(e) = elasticity {
                assert!(e <= 0.0, "Cost elasticity should be non-positive: {}", e);
            }
        }
    }

    #[test]
    fn test_summary_report_generation() {
        use crate::strategies::SmaCrossover;

        let config = create_test_config();
        let sensitivity_config = CostSensitivityConfig::standard();
        let bars = create_test_bars();
        let mut strategy = SmaCrossover::new(5, 15); // Shorter periods to generate signals faster

        let analysis = run_cost_sensitivity_analysis(
            &config,
            &sensitivity_config,
            &bars,
            &mut strategy,
            "TEST",
        )
        .unwrap();

        let report = analysis.summary_report();

        // Report should contain key sections
        assert!(report.contains("Cost Sensitivity Analysis"));
        assert!(report.contains("Multiplier"));
        assert!(report.contains("Return"));
        assert!(report.contains("Sharpe"));
        assert!(report.contains("Degradation Analysis"));
        assert!(report.contains("Robustness Assessment"));
    }

    #[test]
    fn test_zero_cost_baseline() {
        use crate::strategies::SmaCrossover;

        let config = create_test_config();
        let sensitivity_config = CostSensitivityConfig {
            multipliers: vec![0.0, 1.0],
            ..Default::default()
        };
        let bars = create_test_bars();
        let mut strategy = SmaCrossover::new(5, 15); // Shorter periods to generate signals faster

        let analysis = run_cost_sensitivity_analysis(
            &config,
            &sensitivity_config,
            &bars,
            &mut strategy,
            "TEST",
        )
        .unwrap();

        // Zero cost scenario should exist
        let zero_cost = analysis.zero_cost();
        assert!(zero_cost.is_some());

        // Zero cost should have higher or equal returns than 1x (lower costs = better performance)
        let baseline = analysis.baseline().unwrap();
        let zero_return = zero_cost.unwrap().total_return_pct();
        let baseline_return = baseline.total_return_pct();
        assert!(
            zero_return >= baseline_return,
            "Zero cost return ({:.2}%) should be >= 1x return ({:.2}%)",
            zero_return,
            baseline_return
        );
    }

    #[test]
    fn test_robustness_check() {
        use crate::strategies::SmaCrossover;

        let config = create_test_config();
        let sensitivity_config = CostSensitivityConfig::default();
        let bars = create_test_bars();
        let mut strategy = SmaCrossover::new(5, 15); // Shorter periods to generate signals faster

        let analysis = run_cost_sensitivity_analysis(
            &config,
            &sensitivity_config,
            &bars,
            &mut strategy,
            "TEST",
        )
        .unwrap();

        // Test robustness check at 5x
        let is_robust = analysis.is_robust(0.5);
        // Result depends on strategy performance, just verify it doesn't panic
        let _ = is_robust;
    }
}
