//! Walk-Forward Analysis for robust strategy optimization.
//!
//! Walk-forward analysis divides the data into multiple in-sample/out-of-sample
//! windows, optimizing on in-sample data and testing on out-of-sample data.
//! This helps prevent overfitting and provides more realistic performance estimates.

use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::error::{BacktestError, Result};
use crate::strategy::Strategy;
use crate::types::{Bar, Verdict};
use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::{info, warn};

/// Configuration for walk-forward analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of walk-forward windows.
    pub num_windows: usize,
    /// In-sample ratio (e.g., 0.7 means 70% in-sample, 30% out-of-sample).
    pub in_sample_ratio: f64,
    /// Whether to use anchored windows (growing in-sample) or rolling windows.
    pub anchored: bool,
    /// Minimum bars required in each window.
    pub min_bars_per_window: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            num_windows: 12,
            in_sample_ratio: 0.75,
            anchored: true,
            min_bars_per_window: 50,
        }
    }
}

impl WalkForwardConfig {
    /// Create a new walk-forward config.
    pub fn new(num_windows: usize, in_sample_ratio: f64) -> Self {
        assert!(num_windows > 0, "Number of windows must be positive");
        assert!(
            in_sample_ratio > 0.0 && in_sample_ratio < 1.0,
            "In-sample ratio must be between 0 and 1"
        );

        Self {
            num_windows,
            in_sample_ratio,
            ..Default::default()
        }
    }

    /// Use anchored (growing) windows instead of rolling windows.
    pub fn with_anchored(mut self) -> Self {
        self.anchored = true;
        self
    }
}

/// A single walk-forward window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardWindow {
    /// Window index.
    pub index: usize,
    /// In-sample start date.
    pub is_start: DateTime<Utc>,
    /// In-sample end date.
    pub is_end: DateTime<Utc>,
    /// Out-of-sample start date.
    pub oos_start: DateTime<Utc>,
    /// Out-of-sample end date.
    pub oos_end: DateTime<Utc>,
    /// Number of in-sample bars.
    pub is_bars: usize,
    /// Number of out-of-sample bars.
    pub oos_bars: usize,
}

/// Results from a single walk-forward window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    /// Window configuration.
    pub window: WalkForwardWindow,
    /// Best in-sample result.
    pub in_sample_result: BacktestResult,
    /// Out-of-sample result using best in-sample parameters.
    pub out_of_sample_result: BacktestResult,
    /// Efficiency ratio (OOS return / IS return).
    pub efficiency_ratio: f64,
    /// Hash of best parameter values for stability tracking.
    pub parameter_hash: u64,
}

/// Complete walk-forward analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    /// Configuration used.
    pub config: WalkForwardConfig,
    /// Results for each window.
    pub windows: Vec<WindowResult>,
    /// Combined out-of-sample performance.
    pub combined_oos_return: f64,
    /// Average in-sample return.
    pub avg_is_return: f64,
    /// Average out-of-sample return.
    pub avg_oos_return: f64,
    /// Average efficiency ratio.
    pub avg_efficiency_ratio: f64,
    /// Walk-forward efficiency (combined OOS / combined IS).
    pub walk_forward_efficiency: f64,
    /// Average in-sample Sharpe ratio.
    pub avg_is_sharpe: f64,
    /// Average out-of-sample Sharpe ratio.
    pub avg_oos_sharpe: f64,
    /// OOS Sharpe threshold met (OOS >= threshold * IS).
    pub oos_sharpe_threshold_met: bool,
    /// Parameter stability score (0.0-1.0, higher is more stable).
    pub parameter_stability: f64,
}

impl WalkForwardResult {
    /// Get a summary of the walk-forward analysis.
    pub fn summary(&self) -> String {
        format!(
            "Walk-Forward Analysis Summary:\n\
             Windows: {}\n\
             Avg IS Return: {:.2}%\n\
             Avg OOS Return: {:.2}%\n\
             Avg Efficiency: {:.2}%\n\
             Combined OOS Return: {:.2}%\n\
             WF Efficiency: {:.2}%\n\
             Avg IS Sharpe: {:.3}\n\
             Avg OOS Sharpe: {:.3}\n\
             OOS Sharpe Threshold Met: {}\n\
             Parameter Stability: {:.2}%",
            self.windows.len(),
            self.avg_is_return,
            self.avg_oos_return,
            self.avg_efficiency_ratio * 100.0,
            self.combined_oos_return,
            self.walk_forward_efficiency * 100.0,
            self.avg_is_sharpe,
            self.avg_oos_sharpe,
            if self.oos_sharpe_threshold_met {
                "Yes"
            } else {
                "No"
            },
            self.parameter_stability * 100.0
        )
    }

    /// Check if the strategy passes the walk-forward test.
    /// Typically requires positive OOS returns and efficiency > 50%.
    pub fn is_robust(&self, min_efficiency: f64) -> bool {
        self.avg_oos_return > 0.0 && self.walk_forward_efficiency >= min_efficiency
    }

    /// Enhanced robustness check including OOS Sharpe ratio threshold.
    /// Requires positive OOS returns, efficiency >= min_efficiency,
    /// and OOS Sharpe >= min_oos_sharpe_ratio * IS Sharpe.
    pub fn is_robust_with_sharpe(&self, min_efficiency: f64, min_oos_sharpe_ratio: f64) -> bool {
        self.avg_oos_return > 0.0
            && self.walk_forward_efficiency >= min_efficiency
            && self.avg_oos_sharpe >= self.avg_is_sharpe * min_oos_sharpe_ratio
    }

    /// Get the OOS Sharpe degradation ratio (OOS Sharpe / IS Sharpe).
    /// Values closer to 1.0 indicate less overfitting.
    /// Values < 0.6 suggest significant performance degradation.
    pub fn oos_sharpe_ratio(&self) -> f64 {
        if self.avg_is_sharpe.abs() > 0.001 {
            self.avg_oos_sharpe / self.avg_is_sharpe
        } else {
            0.0
        }
    }

    /// Get the strategy verdict based on walk-forward analysis.
    ///
    /// The verdict classifies the strategy as:
    /// - `Robust`: OOS/IS degradation > 0.80 and positive OOS returns with good efficiency
    /// - `Borderline`: OOS/IS degradation 0.60-0.80 or moderate efficiency
    /// - `LikelyOverfit`: OOS/IS degradation < 0.60 or negative OOS returns
    ///
    /// This provides a simple, actionable classification of strategy robustness.
    pub fn verdict(&self) -> Verdict {
        let degradation_ratio = self.oos_sharpe_ratio();
        let oos_positive = self.avg_oos_return > 0.0;

        Verdict::from_criteria(
            degradation_ratio,
            oos_positive,
            self.walk_forward_efficiency,
        )
    }
}

/// Walk-forward analyzer.
pub struct WalkForwardAnalyzer {
    config: WalkForwardConfig,
    backtest_config: BacktestConfig,
}

impl WalkForwardAnalyzer {
    /// Create a new walk-forward analyzer.
    pub fn new(config: WalkForwardConfig, backtest_config: BacktestConfig) -> Self {
        Self {
            config,
            backtest_config,
        }
    }

    /// Calculate window boundaries for the data.
    pub fn calculate_windows(&self, bars: &[Bar]) -> Result<Vec<WalkForwardWindow>> {
        if bars.len() < self.config.min_bars_per_window * self.config.num_windows {
            return Err(BacktestError::DataError(
                "Not enough data for walk-forward analysis".to_string(),
            ));
        }

        let total_bars = bars.len();
        let window_size = total_bars / self.config.num_windows;
        let is_size = (window_size as f64 * self.config.in_sample_ratio) as usize;
        let oos_size = window_size - is_size;

        let mut windows = Vec::with_capacity(self.config.num_windows);

        for i in 0..self.config.num_windows {
            let window_start = if self.config.anchored {
                0
            } else {
                i * window_size
            };

            let window_end = (i + 1) * window_size;
            let is_end_idx = if self.config.anchored {
                window_end - oos_size
            } else {
                window_start + is_size
            };

            // Ensure we don't go out of bounds
            let is_end_idx = is_end_idx.min(total_bars - 1);
            let window_end = window_end.min(total_bars - 1);

            let window = WalkForwardWindow {
                index: i,
                is_start: bars[window_start].timestamp,
                is_end: bars[is_end_idx].timestamp,
                oos_start: bars[is_end_idx + 1].timestamp,
                oos_end: bars[window_end].timestamp,
                is_bars: is_end_idx - window_start + 1,
                oos_bars: window_end - is_end_idx,
            };

            if window.is_bars >= self.config.min_bars_per_window && window.oos_bars >= 10 {
                windows.push(window);
            } else {
                warn!("Window {} skipped due to insufficient bars", i);
            }
        }

        if windows.is_empty() {
            return Err(BacktestError::DataError(
                "No valid walk-forward windows could be created".to_string(),
            ));
        }

        Ok(windows)
    }

    /// Run walk-forward analysis with a strategy factory that creates strategies for optimization.
    pub fn run<P, F>(
        &self,
        bars: &[Bar],
        symbol: &str,
        params: Vec<P>,
        strategy_factory: F,
        metric: WalkForwardMetric,
    ) -> Result<WalkForwardResult>
    where
        P: Clone + Send + Sync + Hash,
        F: Fn(&P) -> Box<dyn Strategy> + Send + Sync,
    {
        let windows = self.calculate_windows(bars)?;
        info!(
            "Running walk-forward analysis with {} windows",
            windows.len()
        );

        let mut window_results = Vec::with_capacity(windows.len());

        for window in &windows {
            info!(
                "Processing window {}: IS {} bars, OOS {} bars",
                window.index, window.is_bars, window.oos_bars
            );

            // Get in-sample bars
            let is_bars: Vec<Bar> = bars
                .iter()
                .filter(|b| b.timestamp >= window.is_start && b.timestamp <= window.is_end)
                .cloned()
                .collect();

            // Get out-of-sample bars
            let oos_bars: Vec<Bar> = bars
                .iter()
                .filter(|b| b.timestamp >= window.oos_start && b.timestamp <= window.oos_end)
                .cloned()
                .collect();

            if is_bars.is_empty() || oos_bars.is_empty() {
                warn!("Window {} has empty IS or OOS data, skipping", window.index);
                continue;
            }

            // Run optimization on in-sample data
            let is_results: Vec<(P, BacktestResult)> = params
                .par_iter()
                .filter_map(|param| {
                    let mut strategy = strategy_factory(param);
                    let mut config = self.backtest_config.clone();
                    config.show_progress = false;

                    let mut engine = Engine::new(config);
                    engine.add_data(symbol.to_string(), is_bars.clone());

                    match engine.run(strategy.as_mut(), symbol) {
                        Ok(result) => Some((param.clone(), result)),
                        Err(_) => None,
                    }
                })
                .collect();

            if is_results.is_empty() {
                warn!("Window {} optimization failed, skipping", window.index);
                continue;
            }

            // Find best in-sample parameters
            let (best_param, best_is_result) = is_results
                .into_iter()
                .max_by(|(_, a), (_, b)| {
                    let metric_a = metric.extract(a);
                    let metric_b = metric.extract(b);
                    metric_a.partial_cmp(&metric_b).unwrap()
                })
                .unwrap();

            // Run out-of-sample backtest with best parameters
            let mut oos_strategy = strategy_factory(&best_param);
            let mut oos_config = self.backtest_config.clone();
            oos_config.show_progress = false;

            let mut engine = Engine::new(oos_config);
            engine.add_data(symbol.to_string(), oos_bars);

            let oos_result = match engine.run(oos_strategy.as_mut(), symbol) {
                Ok(result) => result,
                Err(e) => {
                    warn!("Window {} OOS backtest failed: {}", window.index, e);
                    continue;
                }
            };

            // Calculate efficiency
            let efficiency = if best_is_result.total_return_pct.abs() > 0.001 {
                oos_result.total_return_pct / best_is_result.total_return_pct
            } else {
                0.0
            };

            // Hash the best parameter for stability tracking
            let mut hasher = DefaultHasher::new();
            best_param.hash(&mut hasher);
            let param_hash = hasher.finish();

            window_results.push(WindowResult {
                window: window.clone(),
                in_sample_result: best_is_result,
                out_of_sample_result: oos_result,
                efficiency_ratio: efficiency,
                parameter_hash: param_hash,
            });
        }

        if window_results.is_empty() {
            return Err(BacktestError::DataError(
                "All walk-forward windows failed".to_string(),
            ));
        }

        // Calculate aggregate statistics
        let avg_is_return: f64 = window_results
            .iter()
            .map(|w| w.in_sample_result.total_return_pct)
            .sum::<f64>()
            / window_results.len() as f64;

        let avg_oos_return: f64 = window_results
            .iter()
            .map(|w| w.out_of_sample_result.total_return_pct)
            .sum::<f64>()
            / window_results.len() as f64;

        let avg_efficiency_ratio: f64 = window_results
            .iter()
            .map(|w| w.efficiency_ratio)
            .filter(|e| e.is_finite())
            .sum::<f64>()
            / window_results.len() as f64;

        // Calculate Sharpe ratio statistics
        let avg_is_sharpe: f64 = window_results
            .iter()
            .map(|w| w.in_sample_result.sharpe_ratio)
            .sum::<f64>()
            / window_results.len() as f64;

        let avg_oos_sharpe: f64 = window_results
            .iter()
            .map(|w| w.out_of_sample_result.sharpe_ratio)
            .sum::<f64>()
            / window_results.len() as f64;

        // Check if OOS Sharpe meets 60% threshold (Lopez de Prado recommendation)
        let oos_sharpe_threshold_met = if avg_is_sharpe.abs() > 0.001 {
            avg_oos_sharpe >= avg_is_sharpe * 0.6
        } else {
            false
        };

        // Calculate parameter stability
        // If all windows have the same parameter hash, stability = 1.0
        // If all different, stability approaches 0.0
        let unique_hashes: std::collections::HashSet<u64> =
            window_results.iter().map(|w| w.parameter_hash).collect();
        let parameter_stability = if window_results.len() > 1 {
            1.0 - (unique_hashes.len() - 1) as f64 / (window_results.len() - 1) as f64
        } else {
            1.0
        };

        // Calculate combined OOS return (compound the returns)
        let combined_oos_return: f64 = window_results.iter().fold(1.0, |acc, w| {
            acc * (1.0 + w.out_of_sample_result.total_return_pct / 100.0)
        }) - 1.0;

        let total_is_return: f64 = window_results.iter().fold(1.0, |acc, w| {
            acc * (1.0 + w.in_sample_result.total_return_pct / 100.0)
        }) - 1.0;

        let walk_forward_efficiency = if total_is_return.abs() > 0.001 {
            combined_oos_return / total_is_return
        } else {
            0.0
        };

        Ok(WalkForwardResult {
            config: self.config.clone(),
            windows: window_results,
            combined_oos_return: combined_oos_return * 100.0,
            avg_is_return,
            avg_oos_return,
            avg_efficiency_ratio,
            walk_forward_efficiency,
            avg_is_sharpe,
            avg_oos_sharpe,
            oos_sharpe_threshold_met,
            parameter_stability,
        })
    }
}

/// Metric to optimize during walk-forward analysis.
#[derive(Debug, Clone, Copy)]
pub enum WalkForwardMetric {
    Sharpe,
    Sortino,
    Return,
    Calmar,
    ProfitFactor,
}

impl WalkForwardMetric {
    /// Extract the metric value from a backtest result.
    pub fn extract(&self, result: &BacktestResult) -> f64 {
        match self {
            WalkForwardMetric::Sharpe => result.sharpe_ratio,
            WalkForwardMetric::Sortino => result.sortino_ratio,
            WalkForwardMetric::Return => result.total_return_pct,
            WalkForwardMetric::Calmar => result.calmar_ratio,
            WalkForwardMetric::ProfitFactor => result.profit_factor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_bars(count: usize) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1);
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    base,
                    base + 2.0,
                    base - 1.0,
                    base + 0.5,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_walk_forward_config() {
        let config = WalkForwardConfig::new(5, 0.7);
        assert_eq!(config.num_windows, 5);
        assert!((config.in_sample_ratio - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_window_calculation() {
        let bars = create_test_bars(500);
        let config = WalkForwardConfig::new(5, 0.7);
        let analyzer = WalkForwardAnalyzer::new(config, BacktestConfig::default());

        let windows = analyzer.calculate_windows(&bars).unwrap();
        assert_eq!(windows.len(), 5);

        // Each window should have valid date ranges
        for window in &windows {
            assert!(window.is_start < window.is_end);
            assert!(window.is_end < window.oos_start);
            assert!(window.oos_start < window.oos_end);
        }
    }

    #[test]
    fn test_anchored_windows() {
        let bars = create_test_bars(500);
        let config = WalkForwardConfig::new(5, 0.7).with_anchored();
        let analyzer = WalkForwardAnalyzer::new(config, BacktestConfig::default());

        let windows = analyzer.calculate_windows(&bars).unwrap();

        // All windows should start at the same point (anchored)
        let first_start = windows[0].is_start;
        for window in &windows {
            assert_eq!(window.is_start, first_start);
        }
    }

    #[test]
    fn test_insufficient_data() {
        let bars = create_test_bars(50); // Too few bars
        let config = WalkForwardConfig::new(10, 0.7);
        let analyzer = WalkForwardAnalyzer::new(config, BacktestConfig::default());

        let result = analyzer.calculate_windows(&bars);
        assert!(result.is_err());
    }

    #[test]
    fn test_walk_forward_metric_extraction() {
        use crate::engine::BacktestConfig;

        let result = BacktestResult {
            strategy_name: "Test".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 10.0,
            trading_days: 252,
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 60.0,
            avg_win: 200.0,
            avg_loss: -100.0,
            profit_factor: 2.0,
            max_drawdown_pct: 5.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            calmar_ratio: 2.0,
            trades: vec![],
            equity_curve: vec![],
            start_time: Utc::now(),
            end_time: Utc::now(),
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
            seed: None,
        };

        assert!((WalkForwardMetric::Sharpe.extract(&result) - 1.5).abs() < 0.001);
        assert!((WalkForwardMetric::Return.extract(&result) - 10.0).abs() < 0.001);
        assert!((WalkForwardMetric::ProfitFactor.extract(&result) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_result_summary() {
        let config = WalkForwardConfig::default();
        let result = WalkForwardResult {
            config,
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.75,
            avg_is_sharpe: 1.5,
            avg_oos_sharpe: 1.0,
            oos_sharpe_threshold_met: true,
            parameter_stability: 0.8,
        };

        let summary = result.summary();
        assert!(summary.contains("Walk-Forward Analysis"));
        assert!(summary.contains("15.00%")); // Combined OOS
    }

    #[test]
    fn test_robustness_check() {
        let config = WalkForwardConfig::default();
        let result = WalkForwardResult {
            config: config.clone(),
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.6,
            avg_is_sharpe: 1.5,
            avg_oos_sharpe: 1.0,
            oos_sharpe_threshold_met: true,
            parameter_stability: 0.8,
        };

        // Should be robust with 50% efficiency threshold
        assert!(result.is_robust(0.5));

        // Should not be robust with 70% efficiency threshold
        assert!(!result.is_robust(0.7));
    }

    #[test]
    fn test_oos_sharpe_threshold() {
        let config = WalkForwardConfig::default();

        // Test passing threshold (OOS >= 60% of IS)
        let result_pass = WalkForwardResult {
            config: config.clone(),
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.6,
            avg_is_sharpe: 1.5,
            avg_oos_sharpe: 1.0, // 1.0 / 1.5 = 0.667 > 0.6
            oos_sharpe_threshold_met: true,
            parameter_stability: 0.8,
        };

        assert!(result_pass.is_robust_with_sharpe(0.5, 0.6));
        assert!((result_pass.oos_sharpe_ratio() - 0.667).abs() < 0.01);

        // Test failing threshold (OOS < 60% of IS)
        let result_fail = WalkForwardResult {
            config: config.clone(),
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.6,
            avg_is_sharpe: 2.0,
            avg_oos_sharpe: 0.8, // 0.8 / 2.0 = 0.4 < 0.6
            oos_sharpe_threshold_met: false,
            parameter_stability: 0.8,
        };

        assert!(!result_fail.is_robust_with_sharpe(0.5, 0.6));
        assert!((result_fail.oos_sharpe_ratio() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_parameter_stability() {
        let config = WalkForwardConfig::default();

        // High stability (same parameters across windows)
        let high_stability = WalkForwardResult {
            config: config.clone(),
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.6,
            avg_is_sharpe: 1.5,
            avg_oos_sharpe: 1.0,
            oos_sharpe_threshold_met: true,
            parameter_stability: 1.0, // Perfect stability
        };

        assert!((high_stability.parameter_stability - 1.0).abs() < 0.001);

        // Low stability (different parameters each window)
        let low_stability = WalkForwardResult {
            config: config.clone(),
            windows: vec![],
            combined_oos_return: 15.0,
            avg_is_return: 20.0,
            avg_oos_return: 12.0,
            avg_efficiency_ratio: 0.6,
            walk_forward_efficiency: 0.6,
            avg_is_sharpe: 1.5,
            avg_oos_sharpe: 1.0,
            oos_sharpe_threshold_met: true,
            parameter_stability: 0.2, // Low stability
        };

        assert!(low_stability.parameter_stability < 0.5);
    }
}
