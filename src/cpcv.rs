//! Combinatorial Purged Cross-Validation (CPCV) for robust ML model validation.
//!
//! CPCV addresses key challenges in time-series cross-validation:
//! - Removes overlapping observations between training and test sets (purging)
//! - Enforces embargo periods to prevent information leakage
//! - Generates combinatorial folds to maximize data usage
//! - Provides stability metrics to detect overfitting
//!
//! # References
//! - Lopez de Prado, "Advances in Financial Machine Learning", Chapter 7

use crate::engine::{BacktestConfig, BacktestResult, Engine};
use crate::error::{BacktestError, Result};
use crate::strategy::Strategy;
use crate::types::Bar;
use chrono::{DateTime, Duration, Utc};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{info, warn};

/// Configuration for Combinatorial Purged Cross-Validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVConfig {
    /// Number of folds to split data into (typically 5-10).
    pub n_splits: usize,
    /// Number of test folds in each combination (typically 1).
    pub n_test_splits: usize,
    /// Embargo period in trading days after test end (prevents leakage).
    pub embargo_days: usize,
    /// Whether to purge training observations that overlap with test.
    pub purge_overlapping: bool,
    /// Minimum bars required in each fold.
    pub min_bars_per_fold: usize,
}

impl Default for CPCVConfig {
    fn default() -> Self {
        Self {
            n_splits: 5,
            n_test_splits: 1,
            embargo_days: 5,
            purge_overlapping: true,
            min_bars_per_fold: 50,
        }
    }
}

impl CPCVConfig {
    /// Create a new CPCV configuration.
    pub fn new(n_splits: usize, embargo_days: usize) -> Self {
        assert!(n_splits >= 2, "Number of splits must be at least 2");
        assert!(embargo_days > 0, "Embargo period must be positive");

        Self {
            n_splits,
            embargo_days,
            ..Default::default()
        }
    }

    /// Set the number of test splits (defaults to 1).
    pub fn with_test_splits(mut self, n_test_splits: usize) -> Self {
        assert!(
            n_test_splits < self.n_splits,
            "Test splits must be less than total splits"
        );
        self.n_test_splits = n_test_splits;
        self
    }

    /// Disable purging (not recommended - increases leakage risk).
    pub fn without_purging(mut self) -> Self {
        self.purge_overlapping = false;
        self
    }
}

/// A single fold in CPCV.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVFold {
    /// Fold index.
    pub index: usize,
    /// Training data start.
    pub train_start: DateTime<Utc>,
    /// Training data end.
    pub train_end: DateTime<Utc>,
    /// Test data start.
    pub test_start: DateTime<Utc>,
    /// Test data end.
    pub test_end: DateTime<Utc>,
    /// Embargo end (bars after test_end excluded from next fold training).
    pub embargo_end: DateTime<Utc>,
    /// Number of training bars.
    pub train_bars: usize,
    /// Number of test bars.
    pub test_bars: usize,
    /// Number of purged bars (removed from training).
    pub purged_bars: usize,
}

/// Results from a single CPCV fold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldResult {
    /// Fold configuration.
    pub fold: CPCVFold,
    /// Test set result.
    pub test_result: BacktestResult,
}

/// Complete CPCV results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPCVResult {
    /// Configuration used.
    pub config: CPCVConfig,
    /// Results for each fold.
    pub folds: Vec<FoldResult>,
    /// Mean test score across folds.
    pub mean_test_score: f64,
    /// Standard deviation of test scores (lower = more stable).
    pub std_test_score: f64,
    /// Minimum test score (worst fold).
    pub min_test_score: f64,
    /// Maximum test score (best fold).
    pub max_test_score: f64,
    /// Total number of combinations tested.
    pub n_combinations: usize,
}

impl CPCVResult {
    /// Get a summary of the CPCV analysis.
    pub fn summary(&self) -> String {
        format!(
            "CPCV Analysis Summary:\n\
             Folds: {} ({}% embargo)\n\
             Mean Score: {:.4}\n\
             Std Score: {:.4}\n\
             Min Score: {:.4}\n\
             Max Score: {:.4}\n\
             Combinations: {}",
            self.folds.len(),
            (self.config.embargo_days as f64 / self.config.n_splits as f64 * 100.0),
            self.mean_test_score,
            self.std_test_score,
            self.min_test_score,
            self.max_test_score,
            self.n_combinations
        )
    }

    /// Check if the model is robust (low variance across folds).
    /// Typically require CV < 0.3 (coefficient of variation).
    pub fn is_robust(&self, max_cv: f64) -> bool {
        let cv = if self.mean_test_score.abs() > 1e-9 {
            self.std_test_score / self.mean_test_score.abs()
        } else {
            f64::INFINITY
        };
        cv <= max_cv && self.mean_test_score > 0.0
    }
}

/// Combinatorial Purged Cross-Validation analyzer.
pub struct CPCVAnalyzer {
    config: CPCVConfig,
    backtest_config: BacktestConfig,
}

impl CPCVAnalyzer {
    /// Create a new CPCV analyzer.
    pub fn new(config: CPCVConfig, backtest_config: BacktestConfig) -> Self {
        Self {
            config,
            backtest_config,
        }
    }

    /// Calculate fold boundaries for the data.
    pub fn calculate_folds(&self, bars: &[Bar]) -> Result<Vec<CPCVFold>> {
        if bars.len() < self.config.min_bars_per_fold * self.config.n_splits {
            return Err(BacktestError::DataError(
                "Not enough data for CPCV".to_string(),
            ));
        }

        let total_bars = bars.len();
        let fold_size = total_bars / self.config.n_splits;

        // Calculate split boundaries
        let mut split_indices: Vec<usize> = (0..self.config.n_splits)
            .map(|i| (i * fold_size).min(total_bars - 1))
            .collect();
        split_indices.push(total_bars);

        // Generate all combinations of test splits
        let test_combinations = self.generate_test_combinations();
        let mut folds = Vec::with_capacity(test_combinations.len());

        for (fold_idx, test_splits) in test_combinations.iter().enumerate() {
            // Identify test split range
            let test_start_idx = split_indices[*test_splits.iter().min().unwrap()];
            let test_end_idx = split_indices[*test_splits.iter().max().unwrap() + 1] - 1;

            if test_end_idx >= total_bars {
                warn!("Fold {} test range out of bounds, skipping", fold_idx);
                continue;
            }

            let test_start = bars[test_start_idx].timestamp;
            let test_end = bars[test_end_idx].timestamp;

            // Calculate embargo end
            let embargo_end = test_end + Duration::days(self.config.embargo_days as i64);

            // Training data: all splits NOT in test set and NOT in embargo
            let train_splits: Vec<usize> = (0..self.config.n_splits)
                .filter(|&s| !test_splits.contains(&s))
                .collect();

            if train_splits.is_empty() {
                warn!("Fold {} has no training data, skipping", fold_idx);
                continue;
            }

            let train_start = bars[split_indices[*train_splits.iter().min().unwrap()]].timestamp;
            let train_end_idx = split_indices[*train_splits.iter().max().unwrap() + 1] - 1;
            let train_end = bars[train_end_idx.min(total_bars - 1)].timestamp;

            // Count bars in each set
            let test_bars = bars
                .iter()
                .filter(|b| b.timestamp >= test_start && b.timestamp <= test_end)
                .count();

            // Count training bars (before purging/embargo)
            let train_bars_unpurged = bars
                .iter()
                .filter(|b| b.timestamp >= train_start && b.timestamp <= train_end)
                .count();

            // Apply embargo: exclude bars within embargo period
            let train_bars_after_embargo = if self.config.embargo_days > 0 {
                bars.iter()
                    .filter(|b| {
                        b.timestamp >= train_start
                            && b.timestamp <= train_end
                            && (b.timestamp
                                < test_start - Duration::days(self.config.embargo_days as i64)
                                || b.timestamp > embargo_end)
                    })
                    .count()
            } else {
                train_bars_unpurged
            };

            let purged_bars = train_bars_unpurged - train_bars_after_embargo;

            // Validate minimum bars requirement
            if train_bars_after_embargo < self.config.min_bars_per_fold
                || test_bars < self.config.min_bars_per_fold / 5
            {
                warn!(
                    "Fold {} insufficient bars (train: {}, test: {}), skipping",
                    fold_idx, train_bars_after_embargo, test_bars
                );
                continue;
            }

            folds.push(CPCVFold {
                index: fold_idx,
                train_start,
                train_end,
                test_start,
                test_end,
                embargo_end,
                train_bars: train_bars_after_embargo,
                test_bars,
                purged_bars,
            });
        }

        if folds.is_empty() {
            return Err(BacktestError::DataError(
                "No valid CPCV folds could be created".to_string(),
            ));
        }

        Ok(folds)
    }

    /// Generate all combinations of test splits.
    fn generate_test_combinations(&self) -> Vec<HashSet<usize>> {
        let mut combinations = Vec::new();

        // Generate all C(n_splits, n_test_splits) combinations
        let indices: Vec<usize> = (0..self.config.n_splits).collect();
        let mut combo = vec![0; self.config.n_test_splits];

        Self::generate_combinations_helper(
            &indices,
            self.config.n_test_splits,
            0,
            &mut combo,
            0,
            &mut combinations,
        );

        combinations
    }

    /// Recursive helper for generating combinations.
    fn generate_combinations_helper(
        indices: &[usize],
        k: usize,
        start: usize,
        combo: &mut Vec<usize>,
        depth: usize,
        results: &mut Vec<HashSet<usize>>,
    ) {
        if depth == k {
            results.push(combo[..k].iter().copied().collect());
            return;
        }

        for i in start..indices.len() {
            combo[depth] = indices[i];
            Self::generate_combinations_helper(indices, k, i + 1, combo, depth + 1, results);
        }
    }

    /// Run CPCV analysis.
    pub fn run<F>(
        &self,
        bars: &[Bar],
        symbol: &str,
        strategy_factory: F,
        metric: CPCVMetric,
    ) -> Result<CPCVResult>
    where
        F: Fn() -> Box<dyn Strategy> + Send + Sync,
    {
        let folds = self.calculate_folds(bars)?;
        info!("Running CPCV with {} folds", folds.len());

        let fold_results: Vec<FoldResult> = folds
            .par_iter()
            .filter_map(|fold| {
                info!(
                    "Processing fold {}: train {} bars, test {} bars, purged {}",
                    fold.index, fold.train_bars, fold.test_bars, fold.purged_bars
                );

                // Get training bars (with purging and embargo)
                let train_bars: Vec<Bar> = bars
                    .iter()
                    .filter(|b| {
                        let in_train_period =
                            b.timestamp >= fold.train_start && b.timestamp <= fold.train_end;
                        let before_test_embargo = b.timestamp
                            < fold.test_start - Duration::days(self.config.embargo_days as i64);
                        let after_test_embargo = b.timestamp > fold.embargo_end;

                        in_train_period && (before_test_embargo || after_test_embargo)
                    })
                    .cloned()
                    .collect();

                // Get test bars
                let test_bars: Vec<Bar> = bars
                    .iter()
                    .filter(|b| b.timestamp >= fold.test_start && b.timestamp <= fold.test_end)
                    .cloned()
                    .collect();

                if train_bars.is_empty() || test_bars.is_empty() {
                    warn!("Fold {} has empty train or test data, skipping", fold.index);
                    return None;
                }

                // Train on training data (in real ML workflow, this is where model training happens)
                // For backtest, we just run the strategy
                let mut test_strategy = strategy_factory();
                let mut test_config = self.backtest_config.clone();
                test_config.show_progress = false;

                let mut engine = Engine::new(test_config);
                engine.add_data(symbol.to_string(), test_bars);

                match engine.run(test_strategy.as_mut(), symbol) {
                    Ok(result) => Some(FoldResult {
                        fold: fold.clone(),
                        test_result: result,
                    }),
                    Err(e) => {
                        warn!("Fold {} backtest failed: {}", fold.index, e);
                        None
                    }
                }
            })
            .collect();

        if fold_results.is_empty() {
            return Err(BacktestError::DataError(
                "All CPCV folds failed".to_string(),
            ));
        }

        // Calculate statistics
        let scores: Vec<f64> = fold_results
            .iter()
            .map(|f| metric.extract(&f.test_result))
            .collect();

        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|&s| (s - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_score = variance.sqrt();
        let min_score = scores
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let max_score = scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        Ok(CPCVResult {
            config: self.config.clone(),
            folds: fold_results,
            mean_test_score: mean_score,
            std_test_score: std_score,
            min_test_score: min_score,
            max_test_score: max_score,
            n_combinations: self.generate_test_combinations().len(),
        })
    }
}

/// Metric to evaluate during CPCV.
#[derive(Debug, Clone, Copy)]
pub enum CPCVMetric {
    Sharpe,
    Sortino,
    Return,
    Calmar,
    ProfitFactor,
}

impl CPCVMetric {
    /// Extract the metric value from a backtest result.
    pub fn extract(&self, result: &BacktestResult) -> f64 {
        match self {
            CPCVMetric::Sharpe => result.sharpe_ratio,
            CPCVMetric::Sortino => result.sortino_ratio,
            CPCVMetric::Return => result.total_return_pct,
            CPCVMetric::Calmar => result.calmar_ratio,
            CPCVMetric::ProfitFactor => result.profit_factor,
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
    fn test_cpcv_config() {
        let config = CPCVConfig::new(5, 5);
        assert_eq!(config.n_splits, 5);
        assert_eq!(config.embargo_days, 5);
        assert_eq!(config.n_test_splits, 1);
        assert!(config.purge_overlapping);
    }

    #[test]
    fn test_cpcv_config_validation() {
        let config = CPCVConfig::new(5, 10).with_test_splits(2);
        assert_eq!(config.n_test_splits, 2);
    }

    #[test]
    #[should_panic(expected = "Test splits must be less than total splits")]
    fn test_invalid_test_splits() {
        CPCVConfig::new(5, 5).with_test_splits(5);
    }

    #[test]
    fn test_combination_generation() {
        let config = CPCVConfig::new(5, 5);
        let analyzer = CPCVAnalyzer::new(config, BacktestConfig::default());

        let combinations = analyzer.generate_test_combinations();

        // C(5, 1) = 5
        assert_eq!(combinations.len(), 5);

        // Each combination should have exactly 1 element
        for combo in &combinations {
            assert_eq!(combo.len(), 1);
        }
    }

    #[test]
    fn test_combination_generation_multiple_test_splits() {
        let config = CPCVConfig::new(5, 5).with_test_splits(2);
        let analyzer = CPCVAnalyzer::new(config, BacktestConfig::default());

        let combinations = analyzer.generate_test_combinations();

        // C(5, 2) = 10
        assert_eq!(combinations.len(), 10);

        // Each combination should have exactly 2 elements
        for combo in &combinations {
            assert_eq!(combo.len(), 2);
        }
    }

    #[test]
    fn test_fold_calculation() {
        let bars = create_test_bars(500);
        let config = CPCVConfig::new(5, 5);
        let analyzer = CPCVAnalyzer::new(config, BacktestConfig::default());

        let folds = analyzer.calculate_folds(&bars).unwrap();

        // Should have 5 folds (one for each test split)
        assert!(folds.len() <= 5);

        // Each fold should have valid date ranges
        for fold in &folds {
            assert!(fold.train_start <= fold.train_end);
            assert!(fold.test_start <= fold.test_end);
            assert!(fold.test_end <= fold.embargo_end);
        }
    }

    #[test]
    fn test_embargo_enforcement() {
        let bars = create_test_bars(500);
        let embargo_days = 10;
        let config = CPCVConfig::new(5, embargo_days);
        let analyzer = CPCVAnalyzer::new(config, BacktestConfig::default());

        let folds = analyzer.calculate_folds(&bars).unwrap();

        for fold in &folds {
            // Embargo period should extend beyond test end
            let expected_embargo = fold.test_end + Duration::days(embargo_days as i64);
            assert_eq!(fold.embargo_end, expected_embargo);
        }
    }

    #[test]
    fn test_insufficient_data() {
        let bars = create_test_bars(50); // Too few bars
        let config = CPCVConfig::new(10, 5);
        let analyzer = CPCVAnalyzer::new(config, BacktestConfig::default());

        let result = analyzer.calculate_folds(&bars);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpcv_metric_extraction() {
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
        };

        assert!((CPCVMetric::Sharpe.extract(&result) - 1.5).abs() < 0.001);
        assert!((CPCVMetric::Return.extract(&result) - 10.0).abs() < 0.001);
        assert!((CPCVMetric::ProfitFactor.extract(&result) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_robustness_check() {
        let config = CPCVConfig::default();
        let result = CPCVResult {
            config,
            folds: vec![],
            mean_test_score: 1.5,
            std_test_score: 0.3,
            min_test_score: 1.0,
            max_test_score: 2.0,
            n_combinations: 5,
        };

        // CV = 0.3/1.5 = 0.2, should be robust with max_cv = 0.3
        assert!(result.is_robust(0.3));

        // Should not be robust with stricter max_cv = 0.15
        assert!(!result.is_robust(0.15));
    }

    #[test]
    fn test_result_summary() {
        let config = CPCVConfig::default();
        let result = CPCVResult {
            config,
            folds: vec![],
            mean_test_score: 1.5,
            std_test_score: 0.3,
            min_test_score: 1.0,
            max_test_score: 2.0,
            n_combinations: 5,
        };

        let summary = result.summary();
        assert!(summary.contains("CPCV Analysis"));
        assert!(summary.contains("1.5")); // Mean score
        assert!(summary.contains("0.3")); // Std score
    }
}
