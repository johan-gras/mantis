//! Monte Carlo simulation for strategy robustness testing.
//!
//! This module provides Monte Carlo methods to assess strategy robustness:
//!
//! - **Trade resampling**: Resample historical trades with replacement
//! - **Return shuffling**: Shuffle daily returns to test path dependency
//! - **Bootstrap confidence intervals**: Calculate confidence intervals for metrics
//!
//! # Example
//!
//! ```ignore
//! use mantis::monte_carlo::{MonteCarloSimulator, MonteCarloConfig};
//! use mantis::engine::BacktestResult;
//!
//! let result: BacktestResult = /* from backtest */;
//! let config = MonteCarloConfig::default();
//! let simulator = MonteCarloSimulator::new(config);
//!
//! let mc_result = simulator.simulate_from_result(&result);
//! println!("95% CI for return: [{:.2}%, {:.2}%]",
//!     mc_result.return_ci.0, mc_result.return_ci.1);
//! ```

use crate::engine::BacktestResult;
use crate::types::{Trade, Verdict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of simulation iterations.
    pub num_simulations: usize,
    /// Confidence level for intervals (e.g., 0.95 for 95%).
    pub confidence_level: f64,
    /// Random seed for reproducibility (None for random).
    pub seed: Option<u64>,
    /// Whether to use trade resampling (bootstrap).
    pub resample_trades: bool,
    /// Whether to shuffle returns (destroy autocorrelation).
    pub shuffle_returns: bool,
    /// Whether to use block bootstrap (preserves serial correlation).
    /// When true, resamples blocks of returns instead of individual returns.
    /// Block size = floor(sqrt(n)) where n is number of observations.
    pub block_bootstrap: bool,
    /// Optional custom block size for block bootstrap.
    /// If None, uses floor(sqrt(n)).
    pub block_size: Option<usize>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_simulations: 1000,
            confidence_level: 0.95,
            seed: None,
            resample_trades: true,
            shuffle_returns: false,
            block_bootstrap: true, // Default to block bootstrap per spec
            block_size: None,      // Auto-calculate as floor(sqrt(n))
        }
    }
}

impl MonteCarloConfig {
    /// Create config for quick analysis.
    pub fn quick() -> Self {
        Self {
            num_simulations: 100,
            ..Default::default()
        }
    }

    /// Create config for thorough analysis.
    pub fn thorough() -> Self {
        Self {
            num_simulations: 10000,
            ..Default::default()
        }
    }

    /// Set number of simulations.
    pub fn with_simulations(mut self, n: usize) -> Self {
        self.num_simulations = n;
        self
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, level: f64) -> Self {
        assert!(level > 0.0 && level < 1.0);
        self.confidence_level = level;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable or disable block bootstrap.
    /// When enabled, resamples blocks of returns to preserve serial correlation.
    pub fn with_block_bootstrap(mut self, enabled: bool) -> Self {
        self.block_bootstrap = enabled;
        self
    }

    /// Set custom block size for block bootstrap.
    /// If not set, uses floor(sqrt(n)) where n is number of observations.
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }

    /// Use simple bootstrap (individual resampling, no blocks).
    /// This is the traditional bootstrap that may break serial correlation.
    pub fn simple_bootstrap(mut self) -> Self {
        self.block_bootstrap = false;
        self.resample_trades = true;
        self.shuffle_returns = false;
        self
    }
}

/// Results from Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// Configuration used.
    pub config: MonteCarloConfig,
    /// Number of simulations run.
    pub num_simulations: usize,
    /// Number of trades used.
    pub num_trades: usize,

    // Return statistics
    /// Mean simulated return.
    pub mean_return: f64,
    /// Median simulated return.
    pub median_return: f64,
    /// Standard deviation of returns.
    pub return_std: f64,
    /// Confidence interval for return (lower, upper).
    pub return_ci: (f64, f64),
    /// Probability of positive return.
    pub prob_positive_return: f64,

    // Drawdown statistics
    /// Mean max drawdown.
    pub mean_max_drawdown: f64,
    /// Median max drawdown.
    pub median_max_drawdown: f64,
    /// Confidence interval for max drawdown.
    pub max_drawdown_ci: (f64, f64),
    /// 95th percentile max drawdown (worst case).
    pub max_drawdown_95th: f64,

    // Sharpe statistics
    /// Mean Sharpe ratio.
    pub mean_sharpe: f64,
    /// Median Sharpe ratio.
    pub median_sharpe: f64,
    /// Confidence interval for Sharpe.
    pub sharpe_ci: (f64, f64),
    /// Probability of positive Sharpe.
    pub prob_positive_sharpe: f64,

    // Risk metrics
    /// Value at Risk (VaR) at confidence level.
    pub var: f64,
    /// Conditional VaR (Expected Shortfall).
    pub cvar: f64,

    // Distribution data
    /// All simulated returns (sorted).
    pub return_distribution: Vec<f64>,
    /// All simulated Sharpe ratios (sorted).
    pub sharpe_distribution: Vec<f64>,
    /// All simulated max drawdowns (sorted).
    pub drawdown_distribution: Vec<f64>,
    /// Percentile returns (5th, 10th, 25th, 50th, 75th, 90th, 95th).
    pub return_percentiles: HashMap<String, f64>,
}

impl MonteCarloResult {
    /// Check if strategy is robust based on common criteria.
    pub fn is_robust(&self) -> bool {
        self.prob_positive_return > 0.6
            && self.prob_positive_sharpe > 0.5
            && self.median_return > 0.0
    }

    /// Get risk-adjusted score combining multiple metrics.
    pub fn robustness_score(&self) -> f64 {
        let return_score = (self.prob_positive_return * 100.0).min(100.0);
        let sharpe_score = (self.prob_positive_sharpe * 100.0).min(100.0);
        let drawdown_score = (100.0 - self.max_drawdown_95th).max(0.0);

        (return_score + sharpe_score + drawdown_score) / 3.0
    }

    /// Get the strategy verdict based on Monte Carlo simulation results.
    ///
    /// The verdict classifies the strategy as:
    /// - `Robust`: High probability of positive returns and Sharpe, score > 70
    /// - `Borderline`: Moderate probability of positive returns, score 50-70
    /// - `LikelyOverfit`: Low probability of positive returns, score < 50
    ///
    /// This provides a simple, actionable classification of strategy robustness.
    pub fn verdict(&self) -> Verdict {
        let score = self.robustness_score();

        if self.prob_positive_return > 0.7 && self.prob_positive_sharpe > 0.6 && score > 70.0 {
            Verdict::Robust
        } else if self.prob_positive_return > 0.5 && self.median_return > 0.0 && score >= 50.0 {
            Verdict::Borderline
        } else {
            Verdict::LikelyOverfit
        }
    }

    /// Generate summary report.
    pub fn summary(&self) -> String {
        format!(
            r#"Monte Carlo Simulation Results
==============================
Simulations: {}
Trades used: {}

Returns:
  Mean: {:.2}%
  Median: {:.2}%
  Std Dev: {:.2}%
  95% CI: [{:.2}%, {:.2}%]
  P(Return > 0): {:.1}%

Risk:
  Mean Max DD: {:.2}%
  95th %ile DD: {:.2}%
  VaR ({}%): {:.2}%
  CVaR: {:.2}%

Sharpe Ratio:
  Mean: {:.2}
  Median: {:.2}
  95% CI: [{:.2}, {:.2}]
  P(Sharpe > 0): {:.1}%

Robustness Score: {:.1}/100
Verdict: {}"#,
            self.num_simulations,
            self.num_trades,
            self.mean_return,
            self.median_return,
            self.return_std,
            self.return_ci.0,
            self.return_ci.1,
            self.prob_positive_return * 100.0,
            self.mean_max_drawdown,
            self.max_drawdown_95th,
            (self.config.confidence_level * 100.0) as i32,
            self.var,
            self.cvar,
            self.mean_sharpe,
            self.median_sharpe,
            self.sharpe_ci.0,
            self.sharpe_ci.1,
            self.prob_positive_sharpe * 100.0,
            self.robustness_score(),
            self.verdict().label().to_uppercase()
        )
    }
}

/// Monte Carlo simulator for strategy analysis.
pub struct MonteCarloSimulator {
    config: MonteCarloConfig,
    rng_state: u64,
}

impl MonteCarloSimulator {
    /// Create a new simulator.
    pub fn new(config: MonteCarloConfig) -> Self {
        let rng_state = config.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345)
        });

        Self { config, rng_state }
    }

    /// Simple LCG random number generator for reproducibility.
    fn next_random(&mut self) -> f64 {
        // Linear congruential generator
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_state >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Generate random index in range [0, max).
    fn random_index(&mut self, max: usize) -> usize {
        (self.next_random() * max as f64) as usize % max
    }

    /// Shuffle a slice in place.
    fn shuffle<T: Clone>(&mut self, data: &mut [T]) {
        for i in (1..data.len()).rev() {
            let j = self.random_index(i + 1);
            data.swap(i, j);
        }
    }

    /// Calculate block size for block bootstrap.
    /// Per spec: block_size = floor(sqrt(n))
    fn calculate_block_size(&self, n: usize) -> usize {
        if let Some(size) = self.config.block_size {
            return size.max(1).min(n);
        }
        // Default: floor(sqrt(n))
        let block_size = (n as f64).sqrt().floor() as usize;
        block_size.max(1) // Ensure at least 1
    }

    /// Perform block bootstrap resampling.
    /// Divides returns into blocks, samples blocks with replacement,
    /// and concatenates to form simulated return series.
    fn block_bootstrap_resample(&mut self, returns: &[f64]) -> Vec<f64> {
        let n = returns.len();
        let block_size = self.calculate_block_size(n);

        // Calculate number of blocks needed to reach target length
        let num_blocks_needed = n.div_ceil(block_size);
        let num_possible_blocks = n.saturating_sub(block_size - 1).max(1);

        let mut result = Vec::with_capacity(n);

        for _ in 0..num_blocks_needed {
            // Randomly select a block start position
            let start = self.random_index(num_possible_blocks);
            let end = (start + block_size).min(n);

            // Add the block to the result
            result.extend_from_slice(&returns[start..end]);

            if result.len() >= n {
                break;
            }
        }

        // Trim to exact target length
        result.truncate(n);
        result
    }

    /// Run Monte Carlo simulation from backtest result.
    pub fn simulate_from_result(&mut self, result: &BacktestResult) -> MonteCarloResult {
        let closed_trades: Vec<&Trade> = result.trades.iter().filter(|t| t.is_closed()).collect();

        if closed_trades.is_empty() {
            return self.empty_result();
        }

        let trade_returns: Vec<f64> = closed_trades
            .iter()
            .filter_map(|t| t.return_pct())
            .collect();

        self.simulate_from_returns(&trade_returns, result.initial_capital)
    }

    /// Run Monte Carlo simulation from trade returns.
    pub fn simulate_from_returns(
        &mut self,
        returns: &[f64],
        initial_capital: f64,
    ) -> MonteCarloResult {
        if returns.is_empty() {
            return self.empty_result();
        }

        let num_trades = returns.len();
        let mut all_total_returns = Vec::with_capacity(self.config.num_simulations);
        let mut all_max_drawdowns = Vec::with_capacity(self.config.num_simulations);
        let mut all_sharpes = Vec::with_capacity(self.config.num_simulations);

        // Run simulations
        for _ in 0..self.config.num_simulations {
            let sim_returns = if self.config.resample_trades {
                if self.config.block_bootstrap {
                    // Block bootstrap: resample blocks to preserve serial correlation
                    // Per spec: block_size = floor(sqrt(n))
                    self.block_bootstrap_resample(returns)
                } else {
                    // Simple bootstrap: resample individual returns with replacement
                    (0..num_trades)
                        .map(|_| returns[self.random_index(num_trades)])
                        .collect::<Vec<f64>>()
                }
            } else if self.config.shuffle_returns {
                // Shuffle: permute returns
                let mut shuffled = returns.to_vec();
                self.shuffle(&mut shuffled);
                shuffled
            } else {
                returns.to_vec()
            };

            // Calculate metrics for this simulation
            let (total_return, max_dd, sharpe) =
                self.calculate_simulation_metrics(&sim_returns, initial_capital);

            all_total_returns.push(total_return);
            all_max_drawdowns.push(max_dd);
            all_sharpes.push(sharpe);
        }

        // Sort for percentile calculations
        all_total_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_max_drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all_sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate statistics
        let mean_return =
            all_total_returns.iter().sum::<f64>() / self.config.num_simulations as f64;
        let median_return = self.percentile(&all_total_returns, 0.5);
        let return_std = self.std_dev(&all_total_returns);
        let alpha = (1.0 - self.config.confidence_level) / 2.0;
        let return_ci = (
            self.percentile(&all_total_returns, alpha),
            self.percentile(&all_total_returns, 1.0 - alpha),
        );

        let prob_positive_return = all_total_returns.iter().filter(|&&r| r > 0.0).count() as f64
            / self.config.num_simulations as f64;

        let mean_max_drawdown =
            all_max_drawdowns.iter().sum::<f64>() / self.config.num_simulations as f64;
        let median_max_drawdown = self.percentile(&all_max_drawdowns, 0.5);
        let max_drawdown_ci = (
            self.percentile(&all_max_drawdowns, alpha),
            self.percentile(&all_max_drawdowns, 1.0 - alpha),
        );
        let max_drawdown_95th = self.percentile(&all_max_drawdowns, 0.95);

        let mean_sharpe = all_sharpes.iter().sum::<f64>() / self.config.num_simulations as f64;
        let median_sharpe = self.percentile(&all_sharpes, 0.5);
        let sharpe_ci = (
            self.percentile(&all_sharpes, alpha),
            self.percentile(&all_sharpes, 1.0 - alpha),
        );
        let prob_positive_sharpe = all_sharpes.iter().filter(|&&s| s > 0.0).count() as f64
            / self.config.num_simulations as f64;

        // VaR and CVaR
        let var = -self.percentile(&all_total_returns, alpha);
        let var_idx = (alpha * self.config.num_simulations as f64) as usize;
        let cvar = if var_idx > 0 {
            -all_total_returns[..var_idx].iter().sum::<f64>() / var_idx as f64
        } else {
            var
        };

        // Percentiles
        let mut return_percentiles = HashMap::new();
        for (name, p) in [
            ("5th", 0.05),
            ("10th", 0.10),
            ("25th", 0.25),
            ("50th", 0.50),
            ("75th", 0.75),
            ("90th", 0.90),
            ("95th", 0.95),
        ] {
            return_percentiles.insert(name.to_string(), self.percentile(&all_total_returns, p));
        }

        MonteCarloResult {
            config: self.config.clone(),
            num_simulations: self.config.num_simulations,
            num_trades,
            mean_return,
            median_return,
            return_std,
            return_ci,
            prob_positive_return,
            mean_max_drawdown,
            median_max_drawdown,
            max_drawdown_ci,
            max_drawdown_95th,
            mean_sharpe,
            median_sharpe,
            sharpe_ci,
            prob_positive_sharpe,
            var,
            cvar,
            return_distribution: all_total_returns,
            sharpe_distribution: all_sharpes,
            drawdown_distribution: all_max_drawdowns,
            return_percentiles,
        }
    }

    /// Calculate metrics for a single simulation path.
    fn calculate_simulation_metrics(
        &self,
        returns: &[f64],
        initial_capital: f64,
    ) -> (f64, f64, f64) {
        if returns.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        // Calculate cumulative equity
        let mut equity = initial_capital;
        let mut peak = equity;
        let mut max_dd: f64 = 0.0;
        let mut equities = Vec::with_capacity(returns.len() + 1);
        equities.push(equity);

        for &ret in returns {
            equity *= 1.0 + ret / 100.0;
            equities.push(equity);
            peak = peak.max(equity);
            let dd = (peak - equity) / peak * 100.0;
            max_dd = max_dd.max(dd);
        }

        let total_return = (equity - initial_capital) / initial_capital * 100.0;

        // Calculate Sharpe (simplified using returns directly)
        let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        let sharpe = if std_dev > 0.0 {
            (mean_ret / std_dev) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        (total_return, max_dd, sharpe)
    }

    /// Calculate percentile of sorted data.
    fn percentile(&self, sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let idx = (p * (sorted_data.len() - 1) as f64) as usize;
        sorted_data[idx.min(sorted_data.len() - 1)]
    }

    /// Calculate standard deviation.
    fn std_dev(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    /// Create empty result when no trades available.
    fn empty_result(&self) -> MonteCarloResult {
        MonteCarloResult {
            config: self.config.clone(),
            num_simulations: 0,
            num_trades: 0,
            mean_return: 0.0,
            median_return: 0.0,
            return_std: 0.0,
            return_ci: (0.0, 0.0),
            prob_positive_return: 0.0,
            mean_max_drawdown: 0.0,
            median_max_drawdown: 0.0,
            max_drawdown_ci: (0.0, 0.0),
            max_drawdown_95th: 0.0,
            mean_sharpe: 0.0,
            median_sharpe: 0.0,
            sharpe_ci: (0.0, 0.0),
            prob_positive_sharpe: 0.0,
            var: 0.0,
            cvar: 0.0,
            return_distribution: vec![],
            sharpe_distribution: vec![],
            drawdown_distribution: vec![],
            return_percentiles: HashMap::new(),
        }
    }
}

/// Run bootstrap analysis on equity curve to estimate confidence bands.
pub fn bootstrap_equity_curve(
    equity_curve: &[f64],
    num_simulations: usize,
    confidence_level: f64,
) -> Vec<(f64, f64, f64)> {
    if equity_curve.len() < 2 {
        return vec![];
    }

    // Calculate daily returns
    let returns: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();

    let mut simulator = MonteCarloSimulator::new(
        MonteCarloConfig::default()
            .with_simulations(num_simulations)
            .with_confidence(confidence_level),
    );

    let mut all_paths: Vec<Vec<f64>> = Vec::with_capacity(num_simulations);

    for _ in 0..num_simulations {
        let mut shuffled_returns = returns.clone();
        simulator.shuffle(&mut shuffled_returns);

        // Reconstruct equity curve
        let mut equity = equity_curve[0];
        let mut path = vec![equity];
        for ret in &shuffled_returns {
            equity *= 1.0 + ret;
            path.push(equity);
        }
        all_paths.push(path);
    }

    // Calculate confidence bands at each time point
    let alpha = (1.0 - confidence_level) / 2.0;
    let n = equity_curve.len();

    (0..n)
        .map(|t| {
            let mut values: Vec<f64> = all_paths.iter().map(|p| p[t]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (alpha * values.len() as f64) as usize;
            let upper_idx = ((1.0 - alpha) * values.len() as f64) as usize;
            let median_idx = values.len() / 2;

            (
                values[lower_idx.min(values.len() - 1)],
                values[median_idx.min(values.len() - 1)],
                values[upper_idx.min(values.len() - 1)],
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_config() {
        let config = MonteCarloConfig::default();
        assert_eq!(config.num_simulations, 1000);
        assert!((config.confidence_level - 0.95).abs() < 0.001);

        let quick = MonteCarloConfig::quick();
        assert_eq!(quick.num_simulations, 100);

        let thorough = MonteCarloConfig::thorough();
        assert_eq!(thorough.num_simulations, 10000);
    }

    #[test]
    fn test_monte_carlo_simulation() {
        // Create synthetic trade returns
        let returns: Vec<f64> = vec![
            2.0, -1.0, 3.0, -0.5, 1.5, -2.0, 4.0, -1.5, 2.5, 0.5, 1.0, -0.8, 2.2, -1.2, 3.5, -0.3,
            1.8, -1.0, 2.0, 1.0,
        ];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // Basic sanity checks
        assert_eq!(result.num_simulations, 500);
        assert_eq!(result.num_trades, 20);
        assert!(result.mean_return.is_finite());
        assert!(result.median_return.is_finite());
        assert!(result.return_std >= 0.0);
        assert!(result.prob_positive_return >= 0.0 && result.prob_positive_return <= 1.0);
        assert!(result.mean_max_drawdown >= 0.0);
    }

    #[test]
    fn test_monte_carlo_reproducibility() {
        let returns: Vec<f64> = vec![1.0, -0.5, 2.0, -1.0, 1.5];

        let config = MonteCarloConfig::default()
            .with_seed(12345)
            .with_simulations(100);

        let mut sim1 = MonteCarloSimulator::new(config.clone());
        let result1 = sim1.simulate_from_returns(&returns, 100_000.0);

        let mut sim2 = MonteCarloSimulator::new(config);
        let result2 = sim2.simulate_from_returns(&returns, 100_000.0);

        // Same seed should produce same results
        assert!((result1.mean_return - result2.mean_return).abs() < 0.001);
        assert!((result1.median_return - result2.median_return).abs() < 0.001);
    }

    #[test]
    fn test_positive_returns_mostly_positive() {
        // All positive returns should give high probability of positive outcome
        let returns: Vec<f64> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // With all positive returns, probability should be very high
        assert!(result.prob_positive_return > 0.9);
        assert!(result.mean_return > 0.0);
    }

    #[test]
    fn test_negative_returns_mostly_negative() {
        // All negative returns should give low probability of positive outcome
        let returns: Vec<f64> = vec![-1.0, -2.0, -1.5, -3.0, -2.5, -1.8, -2.2];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // With all negative returns, probability should be very low
        assert!(result.prob_positive_return < 0.1);
        assert!(result.mean_return < 0.0);
    }

    #[test]
    fn test_confidence_intervals() {
        let returns: Vec<f64> = vec![2.0, -1.0, 3.0, -0.5, 1.5, -2.0, 4.0, -1.5, 2.5, 0.5];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(1000);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // CI should contain mean and be ordered correctly
        assert!(result.return_ci.0 <= result.median_return);
        assert!(result.return_ci.1 >= result.median_return);
        assert!(result.return_ci.0 <= result.return_ci.1);
    }

    #[test]
    fn test_empty_returns() {
        let returns: Vec<f64> = vec![];

        let config = MonteCarloConfig::default();
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        assert_eq!(result.num_trades, 0);
        assert_eq!(result.mean_return, 0.0);
    }

    #[test]
    fn test_summary_report() {
        let returns: Vec<f64> = vec![1.0, -0.5, 2.0, -1.0, 1.5];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(100);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);
        let summary = result.summary();

        assert!(summary.contains("Monte Carlo Simulation"));
        assert!(summary.contains("Returns:"));
        assert!(summary.contains("Risk:"));
        assert!(summary.contains("Sharpe Ratio:"));
    }

    #[test]
    fn test_robustness_score() {
        let returns: Vec<f64> = vec![2.0, 1.0, 3.0, 1.5, 2.5, 1.8, 2.2];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // Score should be between 0 and 100
        let score = result.robustness_score();
        assert!((0.0..=100.0).contains(&score));
    }

    #[test]
    fn test_bootstrap_equity_curve() {
        let equity_curve: Vec<f64> = vec![100.0, 102.0, 101.0, 105.0, 104.0, 108.0];

        let bands = bootstrap_equity_curve(&equity_curve, 100, 0.95);

        assert_eq!(bands.len(), equity_curve.len());

        // Each band should have lower <= median <= upper
        for (lower, median, upper) in &bands {
            assert!(lower <= median);
            assert!(median <= upper);
        }
    }

    #[test]
    fn test_block_bootstrap_block_size_calculation() {
        // Test block size calculation: floor(sqrt(n))
        let returns: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(10);
        let simulator = MonteCarloSimulator::new(config);

        // For n=100, block_size should be floor(sqrt(100)) = 10
        let block_size = simulator.calculate_block_size(returns.len());
        assert_eq!(block_size, 10);

        // For n=2520 (10 years daily), block_size should be floor(sqrt(2520)) = 50
        let returns_2520: Vec<f64> = (0..2520).map(|i| (i as f64) * 0.01).collect();
        let block_size_2520 = simulator.calculate_block_size(returns_2520.len());
        assert_eq!(block_size_2520, 50);
    }

    #[test]
    fn test_block_bootstrap_preserves_length() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(10);
        let mut simulator = MonteCarloSimulator::new(config);

        // Block bootstrap should return same length as input
        let resampled = simulator.block_bootstrap_resample(&returns);
        assert_eq!(resampled.len(), returns.len());
    }

    #[test]
    fn test_block_bootstrap_vs_simple_bootstrap() {
        // Block bootstrap should preserve more serial correlation
        let returns: Vec<f64> = vec![
            1.0, 1.1, 1.2, 1.3, 1.4, // Trend up
            -0.5, -0.6, -0.7, -0.8, -0.9, // Trend down
            0.2, 0.3, 0.4, 0.5, 0.6, // Trend up
            -0.1, -0.2, -0.3, -0.4, -0.5, // Trend down
        ];

        // Run both block and simple bootstrap
        let config_block = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500)
            .with_block_bootstrap(true);

        let config_simple = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(500)
            .simple_bootstrap();

        let mut sim_block = MonteCarloSimulator::new(config_block);
        let mut sim_simple = MonteCarloSimulator::new(config_simple);

        let result_block = sim_block.simulate_from_returns(&returns, 100_000.0);
        let result_simple = sim_simple.simulate_from_returns(&returns, 100_000.0);

        // Both should produce valid results
        assert!(result_block.mean_return.is_finite());
        assert!(result_simple.mean_return.is_finite());
        assert_eq!(result_block.num_trades, 20);
        assert_eq!(result_simple.num_trades, 20);
    }

    #[test]
    fn test_block_bootstrap_custom_block_size() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64) * 0.1).collect();

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(10)
            .with_block_size(20);
        let mut simulator = MonteCarloSimulator::new(config);

        // Custom block size should override default
        let block_size = simulator.calculate_block_size(returns.len());
        assert_eq!(block_size, 20);

        // Resample should work with custom block size
        let resampled = simulator.block_bootstrap_resample(&returns);
        assert_eq!(resampled.len(), returns.len());
    }

    #[test]
    fn test_block_bootstrap_small_data() {
        // Edge case: very small data should still work
        let returns: Vec<f64> = vec![1.0, -0.5, 2.0];

        let config = MonteCarloConfig::default()
            .with_seed(42)
            .with_simulations(100)
            .with_block_bootstrap(true);
        let mut simulator = MonteCarloSimulator::new(config);

        let result = simulator.simulate_from_returns(&returns, 100_000.0);

        // Should complete without panic
        assert!(result.mean_return.is_finite());
        assert_eq!(result.num_trades, 3);
    }

    #[test]
    fn test_block_bootstrap_reproducibility() {
        let returns: Vec<f64> = vec![1.0, -0.5, 2.0, -1.0, 1.5, -0.8, 2.2, -1.2, 1.8, -0.6];

        // Same seed should produce same results with block bootstrap
        let config1 = MonteCarloConfig::default()
            .with_seed(12345)
            .with_simulations(100)
            .with_block_bootstrap(true);

        let config2 = MonteCarloConfig::default()
            .with_seed(12345)
            .with_simulations(100)
            .with_block_bootstrap(true);

        let mut sim1 = MonteCarloSimulator::new(config1);
        let mut sim2 = MonteCarloSimulator::new(config2);

        let result1 = sim1.simulate_from_returns(&returns, 100_000.0);
        let result2 = sim2.simulate_from_returns(&returns, 100_000.0);

        // Same seed should produce same results
        assert!((result1.mean_return - result2.mean_return).abs() < 0.001);
        assert!((result1.median_return - result2.median_return).abs() < 0.001);
    }
}
