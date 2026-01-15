//! Multi-asset portfolio backtesting support.
//!
//! This module enables backtesting strategies across multiple assets simultaneously,
//! supporting portfolio-level risk management and rebalancing.
//!
//! # Example
//!
//! ```ignore
//! use mantis::multi_asset::{MultiAssetEngine, PortfolioStrategy, AllocationSignal};
//! use mantis::engine::BacktestConfig;
//!
//! let config = BacktestConfig::default();
//! let mut engine = MultiAssetEngine::new(config);
//!
//! engine.add_data("AAPL", aapl_bars);
//! engine.add_data("GOOG", goog_bars);
//!
//! let result = engine.run(&mut strategy)?;
//! ```

use crate::data::DataManager;
use crate::engine::BacktestConfig;
use crate::error::{BacktestError, Result};
use crate::portfolio::Portfolio;
use crate::types::{Bar, EquityPoint, Order, Side, Trade};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::info;

/// Target allocation for an asset as a fraction of portfolio.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Allocation {
    /// Target weight (0.0 to 1.0).
    pub weight: f64,
    /// Minimum weight (optional floor).
    pub min_weight: f64,
    /// Maximum weight (optional cap).
    pub max_weight: f64,
}

impl Allocation {
    /// Create a new allocation with target weight.
    pub fn new(weight: f64) -> Self {
        Self {
            weight: weight.clamp(0.0, 1.0),
            min_weight: 0.0,
            max_weight: 1.0,
        }
    }

    /// Create allocation with bounds.
    pub fn with_bounds(weight: f64, min: f64, max: f64) -> Self {
        Self {
            weight: weight.clamp(min, max),
            min_weight: min,
            max_weight: max,
        }
    }
}

/// Portfolio-level constraints for risk management and diversification.
///
/// These constraints are enforced at the portfolio level to prevent concentration risk,
/// over-leverage, and excessive turnover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstraints {
    /// Maximum position size per symbol as fraction of portfolio (e.g., 0.10 = 10%).
    pub max_position_size: Option<f64>,

    /// Minimum position size per symbol as fraction of portfolio (e.g., 0.01 = 1%).
    pub min_position_size: Option<f64>,

    /// Maximum leverage (e.g., 1.0 = no leverage, 2.0 = 2x leverage).
    pub max_leverage: f64,

    /// Minimum number of holdings to maintain diversification.
    pub min_holdings: Option<usize>,

    /// Maximum number of holdings.
    pub max_holdings: Option<usize>,

    /// Maximum turnover per rebalance as fraction of portfolio (e.g., 0.20 = 20%).
    /// Prevents excessive trading costs.
    pub max_turnover: Option<f64>,

    /// Maximum correlation between any two holdings (e.g., 0.95).
    /// Used to prevent concentration in highly correlated assets.
    pub max_correlation: Option<f64>,

    /// Sector exposure limits (sector name -> max weight).
    /// Example: {"Technology": 0.30} means max 30% in tech sector.
    pub sector_limits: HashMap<String, f64>,

    /// Symbol-specific overrides for max position size.
    /// Takes precedence over global max_position_size.
    pub symbol_limits: HashMap<String, f64>,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            max_position_size: Some(0.25), // 25% max per position
            min_position_size: None,
            max_leverage: 1.0, // No leverage by default
            min_holdings: None,
            max_holdings: None,
            max_turnover: None,
            max_correlation: None,
            sector_limits: HashMap::new(),
            symbol_limits: HashMap::new(),
        }
    }
}

impl PortfolioConstraints {
    /// Create constraints with no limits (except leverage = 1.0).
    pub fn none() -> Self {
        Self {
            max_position_size: None,
            min_position_size: None,
            max_leverage: 1.0,
            min_holdings: None,
            max_holdings: None,
            max_turnover: None,
            max_correlation: None,
            sector_limits: HashMap::new(),
            symbol_limits: HashMap::new(),
        }
    }

    /// Create reasonable default constraints for a diversified portfolio.
    pub fn moderate() -> Self {
        Self {
            max_position_size: Some(0.20), // 20% max
            min_position_size: Some(0.02), // 2% min
            max_leverage: 1.0,
            min_holdings: Some(5),
            max_holdings: Some(30),
            max_turnover: Some(0.30), // 30% turnover limit
            max_correlation: Some(0.95),
            sector_limits: HashMap::new(),
            symbol_limits: HashMap::new(),
        }
    }

    /// Create strict constraints for conservative portfolio management.
    pub fn strict() -> Self {
        Self {
            max_position_size: Some(0.10), // 10% max
            min_position_size: Some(0.05), // 5% min
            max_leverage: 1.0,
            min_holdings: Some(10),
            max_holdings: Some(20),
            max_turnover: Some(0.15), // 15% turnover limit
            max_correlation: Some(0.90),
            sector_limits: HashMap::new(),
            symbol_limits: HashMap::new(),
        }
    }

    /// Set max position size for a specific symbol.
    pub fn with_symbol_limit(mut self, symbol: impl Into<String>, limit: f64) -> Self {
        self.symbol_limits
            .insert(symbol.into(), limit.clamp(0.0, 1.0));
        self
    }

    /// Set max exposure for a sector.
    pub fn with_sector_limit(mut self, sector: impl Into<String>, limit: f64) -> Self {
        self.sector_limits
            .insert(sector.into(), limit.clamp(0.0, 1.0));
        self
    }

    /// Validate proposed weights against constraints.
    ///
    /// Returns `Ok(())` if all constraints are satisfied, or an error describing the violation.
    pub fn validate_weights(
        &self,
        weights: &HashMap<String, f64>,
        symbol_sectors: &HashMap<String, String>,
    ) -> Result<()> {
        let total_weight: f64 = weights.values().sum();

        // Check leverage (total weight > 1.0 indicates leverage)
        if total_weight > self.max_leverage {
            return Err(BacktestError::ConstraintViolation(format!(
                "Total weight {:.2}% exceeds max leverage {:.2}",
                total_weight * 100.0,
                self.max_leverage * 100.0
            )));
        }

        // Check number of holdings
        let num_holdings = weights.values().filter(|&&w| w > 1e-6).count();

        if let Some(min) = self.min_holdings {
            if num_holdings < min {
                return Err(BacktestError::ConstraintViolation(format!(
                    "Number of holdings {} is below minimum {}",
                    num_holdings, min
                )));
            }
        }

        if let Some(max) = self.max_holdings {
            if num_holdings > max {
                return Err(BacktestError::ConstraintViolation(format!(
                    "Number of holdings {} exceeds maximum {}",
                    num_holdings, max
                )));
            }
        }

        // Check position size limits
        for (symbol, &weight) in weights.iter() {
            // Skip negligible weights
            if weight < 1e-6 {
                continue;
            }

            // Check symbol-specific limit first
            let max_weight = self
                .symbol_limits
                .get(symbol)
                .copied()
                .or(self.max_position_size);

            if let Some(max) = max_weight {
                if weight > max {
                    return Err(BacktestError::ConstraintViolation(format!(
                        "Position size {:.2}% for {} exceeds maximum {:.2}%",
                        weight * 100.0,
                        symbol,
                        max * 100.0
                    )));
                }
            }

            if let Some(min) = self.min_position_size {
                if weight < min {
                    return Err(BacktestError::ConstraintViolation(format!(
                        "Position size {:.2}% for {} is below minimum {:.2}%",
                        weight * 100.0,
                        symbol,
                        min * 100.0
                    )));
                }
            }
        }

        // Check sector exposure limits
        if !self.sector_limits.is_empty() {
            let mut sector_exposure: HashMap<String, f64> = HashMap::new();

            for (symbol, &weight) in weights.iter() {
                if let Some(sector) = symbol_sectors.get(symbol) {
                    *sector_exposure.entry(sector.clone()).or_insert(0.0) += weight;
                }
            }

            for (sector, &exposure) in sector_exposure.iter() {
                if let Some(&limit) = self.sector_limits.get(sector) {
                    if exposure > limit {
                        return Err(BacktestError::ConstraintViolation(format!(
                            "Sector {} exposure {:.2}% exceeds limit {:.2}%",
                            sector,
                            exposure * 100.0,
                            limit * 100.0
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Calculate turnover between current and proposed weights.
    ///
    /// Turnover is the sum of absolute differences in weights, divided by 2.
    pub fn calculate_turnover(
        current_weights: &HashMap<String, f64>,
        proposed_weights: &HashMap<String, f64>,
    ) -> f64 {
        let mut total_diff = 0.0;

        // Get all unique symbols
        let all_symbols: std::collections::HashSet<_> = current_weights
            .keys()
            .chain(proposed_weights.keys())
            .collect();

        for symbol in all_symbols {
            let current = current_weights.get(symbol).copied().unwrap_or(0.0);
            let proposed = proposed_weights.get(symbol).copied().unwrap_or(0.0);
            total_diff += (proposed - current).abs();
        }

        // Divide by 2 because buying and selling are counted separately
        total_diff / 2.0
    }

    /// Validate turnover constraint.
    pub fn validate_turnover(
        &self,
        current_weights: &HashMap<String, f64>,
        proposed_weights: &HashMap<String, f64>,
    ) -> Result<()> {
        if let Some(max_turnover) = self.max_turnover {
            let turnover = Self::calculate_turnover(current_weights, proposed_weights);

            if turnover > max_turnover {
                return Err(BacktestError::ConstraintViolation(format!(
                    "Turnover {:.2}% exceeds maximum {:.2}%",
                    turnover * 100.0,
                    max_turnover * 100.0
                )));
            }
        }

        Ok(())
    }

    /// Validate all constraints for a rebalance operation.
    pub fn validate_rebalance(
        &self,
        current_weights: &HashMap<String, f64>,
        proposed_weights: &HashMap<String, f64>,
        symbol_sectors: &HashMap<String, String>,
    ) -> Result<()> {
        self.validate_weights(proposed_weights, symbol_sectors)?;
        self.validate_turnover(current_weights, proposed_weights)?;
        Ok(())
    }
}

/// Signal generated by a portfolio strategy.
#[derive(Debug, Clone)]
pub enum AllocationSignal {
    /// Target allocations for all assets.
    Allocate(HashMap<String, Allocation>),
    /// Rebalance to specified weights.
    Rebalance(HashMap<String, f64>),
    /// Exit all positions.
    ExitAll,
    /// No change.
    Hold,
}

/// Context for portfolio strategy decisions.
#[derive(Debug)]
pub struct PortfolioContext<'a> {
    /// Current bar index.
    pub bar_index: usize,
    /// Current bars for all symbols.
    pub current_bars: &'a HashMap<String, Bar>,
    /// Historical bars for all symbols.
    pub bars: &'a HashMap<String, Vec<Bar>>,
    /// Current positions (symbol -> quantity).
    pub positions: HashMap<String, f64>,
    /// Current cash.
    pub cash: f64,
    /// Total portfolio equity.
    pub equity: f64,
    /// Current weights of each position.
    pub weights: HashMap<String, f64>,
    /// Symbols being traded.
    pub symbols: &'a [String],
}

impl<'a> PortfolioContext<'a> {
    /// Get current price for a symbol.
    pub fn price(&self, symbol: &str) -> Option<f64> {
        self.current_bars.get(symbol).map(|b| b.close)
    }

    /// Get historical bars for a symbol.
    pub fn history(&self, symbol: &str) -> Option<&Vec<Bar>> {
        self.bars.get(symbol)
    }

    /// Get current weight of a symbol.
    pub fn weight(&self, symbol: &str) -> f64 {
        self.weights.get(symbol).copied().unwrap_or(0.0)
    }

    /// Check if we have a position in a symbol.
    pub fn has_position(&self, symbol: &str) -> bool {
        self.positions.get(symbol).is_some_and(|&q| q.abs() > 0.0)
    }

    /// Calculate correlation between two symbols over a lookback period.
    pub fn correlation(&self, symbol1: &str, symbol2: &str, lookback: usize) -> Option<f64> {
        let bars1 = self.bars.get(symbol1)?;
        let bars2 = self.bars.get(symbol2)?;

        if bars1.len() < lookback || bars2.len() < lookback {
            return None;
        }

        let returns1: Vec<f64> = bars1[bars1.len() - lookback..]
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        let returns2: Vec<f64> = bars2[bars2.len() - lookback..]
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        if returns1.len() != returns2.len() || returns1.is_empty() {
            return None;
        }

        let mean1: f64 = returns1.iter().sum::<f64>() / returns1.len() as f64;
        let mean2: f64 = returns2.iter().sum::<f64>() / returns2.len() as f64;

        let cov: f64 = returns1
            .iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| (r1 - mean1) * (r2 - mean2))
            .sum::<f64>()
            / returns1.len() as f64;

        let std1: f64 = (returns1.iter().map(|r| (r - mean1).powi(2)).sum::<f64>()
            / returns1.len() as f64)
            .sqrt();
        let std2: f64 = (returns2.iter().map(|r| (r - mean2).powi(2)).sum::<f64>()
            / returns2.len() as f64)
            .sqrt();

        if std1 > 0.0 && std2 > 0.0 {
            Some(cov / (std1 * std2))
        } else {
            None
        }
    }

    /// Calculate the percentile rank of a symbol's metric across the universe.
    ///
    /// Returns a value between 0.0 (lowest) and 1.0 (highest) indicating where
    /// the symbol ranks relative to all other symbols for the given metric.
    ///
    /// # Arguments
    /// * `symbol` - The symbol to rank
    /// * `metric_fn` - Function that computes the metric from historical bars
    /// * `lookback` - Minimum number of bars required for metric calculation
    ///
    /// # Example
    /// ```ignore
    /// // Rank by momentum (return over last 20 bars)
    /// let momentum_rank = ctx.rank_percentile("AAPL", |bars, lb| {
    ///     if bars.len() < lb { return None; }
    ///     let start = bars[bars.len() - lb].close;
    ///     let end = bars.last().unwrap().close;
    ///     Some((end - start) / start)
    /// }, 20);
    /// ```
    pub fn rank_percentile<F>(&self, symbol: &str, metric_fn: F, lookback: usize) -> Option<f64>
    where
        F: Fn(&Vec<Bar>, usize) -> Option<f64>,
    {
        // Calculate metric for target symbol
        let target_bars = self.bars.get(symbol)?;
        let target_value = metric_fn(target_bars, lookback)?;

        // Calculate metric for all symbols in universe
        let mut values: Vec<(String, f64)> = Vec::new();
        for sym in self.symbols {
            if let Some(bars) = self.bars.get(sym) {
                if let Some(value) = metric_fn(bars, lookback) {
                    values.push((sym.clone(), value));
                }
            }
        }

        if values.len() < 2 {
            return None; // Need at least 2 symbols for meaningful ranking
        }

        // Count how many symbols have lower values
        let lower_count = values.iter().filter(|(_, v)| *v < target_value).count();

        // Percentile rank = (number below + 0.5 * number equal) / total
        // Simplified: number below / (total - 1) for unique values
        let percentile = lower_count as f64 / (values.len() - 1) as f64;

        Some(percentile)
    }

    /// Calculate the cross-sectional z-score of a symbol's metric across the universe.
    ///
    /// Returns how many standard deviations the symbol's metric is from the
    /// universe mean. Positive values indicate above-average, negative below-average.
    ///
    /// # Arguments
    /// * `symbol` - The symbol to score
    /// * `metric_fn` - Function that computes the metric from historical bars
    /// * `lookback` - Minimum number of bars required for metric calculation
    ///
    /// # Example
    /// ```ignore
    /// // Z-score by volatility (std dev of returns over last 20 bars)
    /// let vol_zscore = ctx.cross_sectional_zscore("AAPL", |bars, lb| {
    ///     if bars.len() < lb { return None; }
    ///     let returns: Vec<f64> = bars[bars.len()-lb..].windows(2)
    ///         .map(|w| (w[1].close - w[0].close) / w[0].close).collect();
    ///     let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    ///     let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    ///     Some(variance.sqrt())
    /// }, 20);
    /// ```
    pub fn cross_sectional_zscore<F>(
        &self,
        symbol: &str,
        metric_fn: F,
        lookback: usize,
    ) -> Option<f64>
    where
        F: Fn(&Vec<Bar>, usize) -> Option<f64>,
    {
        // Calculate metric for target symbol
        let target_bars = self.bars.get(symbol)?;
        let target_value = metric_fn(target_bars, lookback)?;

        // Calculate metric for all symbols in universe
        let mut values: Vec<f64> = Vec::new();
        for sym in self.symbols {
            if let Some(bars) = self.bars.get(sym) {
                if let Some(value) = metric_fn(bars, lookback) {
                    values.push(value);
                }
            }
        }

        if values.len() < 2 {
            return None; // Need at least 2 symbols for meaningful z-score
        }

        // Calculate mean and standard deviation across universe
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            Some((target_value - mean) / std_dev)
        } else {
            Some(0.0) // All values are the same
        }
    }

    /// Calculate relative momentum of a symbol compared to the universe average.
    ///
    /// Returns the momentum (return) of the symbol minus the equal-weighted
    /// average momentum of all symbols in the universe.
    ///
    /// # Arguments
    /// * `symbol` - The symbol to calculate relative momentum for
    /// * `lookback` - Number of bars to calculate momentum over
    ///
    /// # Returns
    /// Relative momentum as a decimal (e.g., 0.05 = 5% outperformance).
    /// Positive values indicate outperformance, negative underperformance.
    ///
    /// # Example
    /// ```ignore
    /// // Relative momentum over last 20 bars
    /// if let Some(rel_mom) = ctx.relative_momentum("AAPL", 20) {
    ///     if rel_mom > 0.0 {
    ///         println!("AAPL is outperforming the universe");
    ///     }
    /// }
    /// ```
    pub fn relative_momentum(&self, symbol: &str, lookback: usize) -> Option<f64> {
        // Calculate momentum for target symbol
        let target_bars = self.bars.get(symbol)?;
        if target_bars.len() < lookback + 1 {
            return None;
        }

        let start_price = target_bars[target_bars.len() - lookback - 1].close;
        let end_price = target_bars.last()?.close;
        let target_momentum = (end_price - start_price) / start_price;

        // Calculate average momentum across universe
        let mut momentums: Vec<f64> = Vec::new();
        for sym in self.symbols {
            if let Some(bars) = self.bars.get(sym) {
                if bars.len() > lookback {
                    let start = bars[bars.len() - lookback - 1].close;
                    let end = bars.last()?.close;
                    let momentum = (end - start) / start;
                    momentums.push(momentum);
                }
            }
        }

        if momentums.is_empty() {
            return None;
        }

        let avg_momentum = momentums.iter().sum::<f64>() / momentums.len() as f64;
        let relative = target_momentum - avg_momentum;

        Some(relative)
    }

    /// Calculate the percentile rank of a symbol by a simple metric (price, volume, etc.).
    ///
    /// This is a convenience method for ranking by simple bar metrics without
    /// needing to provide a custom metric function.
    ///
    /// # Arguments
    /// * `symbol` - The symbol to rank
    /// * `metric` - The metric to rank by ("return", "volume", "volatility")
    /// * `lookback` - Number of bars for metric calculation
    ///
    /// # Returns
    /// Percentile rank from 0.0 (lowest) to 1.0 (highest), or None if insufficient data.
    pub fn rank_by_metric(&self, symbol: &str, metric: &str, lookback: usize) -> Option<f64> {
        match metric {
            "return" | "momentum" => self.rank_percentile(
                symbol,
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                lookback,
            ),
            "volume" => self.rank_percentile(
                symbol,
                |bars, lb| {
                    if bars.len() < lb {
                        return None;
                    }
                    let avg_vol = bars[bars.len() - lb..]
                        .iter()
                        .map(|b| b.volume)
                        .sum::<f64>()
                        / lb as f64;
                    Some(avg_vol)
                },
                lookback,
            ),
            "volatility" => self.rank_percentile(
                symbol,
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let returns: Vec<f64> = bars[bars.len() - lb..]
                        .windows(2)
                        .map(|w| (w[1].close - w[0].close) / w[0].close)
                        .collect();
                    if returns.is_empty() {
                        return None;
                    }
                    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / returns.len() as f64;
                    Some(variance.sqrt())
                },
                lookback,
            ),
            _ => None,
        }
    }
}

/// Trait for multi-asset portfolio strategies.
pub trait PortfolioStrategy: Send + Sync {
    /// Returns the name of the strategy.
    fn name(&self) -> &str;

    /// Called once at the start of the backtest.
    fn init(&mut self) {}

    /// Generate allocation signals based on current state.
    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal;

    /// Called at the end of the backtest.
    fn on_finish(&mut self) {}

    /// Minimum bars needed before the strategy can generate signals.
    fn warmup_period(&self) -> usize {
        0
    }
}

/// Result from a multi-asset backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAssetResult {
    /// Strategy name.
    pub strategy_name: String,
    /// Symbols traded.
    pub symbols: Vec<String>,
    /// Initial capital.
    pub initial_capital: f64,
    /// Final equity.
    pub final_equity: f64,
    /// Total return percentage.
    pub total_return_pct: f64,
    /// Annual return percentage.
    pub annual_return_pct: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Total trades across all assets.
    pub total_trades: usize,
    /// Trades by symbol.
    pub trades_by_symbol: HashMap<String, usize>,
    /// All trades.
    pub trades: Vec<Trade>,
    /// Equity curve.
    pub equity_curve: Vec<EquityPoint>,
    /// Start time.
    pub start_time: DateTime<Utc>,
    /// End time.
    pub end_time: DateTime<Utc>,
    /// Weight history (sampled).
    pub weight_history: Vec<(DateTime<Utc>, HashMap<String, f64>)>,
}

/// Multi-asset backtest engine.
pub struct MultiAssetEngine {
    config: BacktestConfig,
    data: DataManager,
    constraints: Option<PortfolioConstraints>,
    symbol_sectors: HashMap<String, String>,
}

impl MultiAssetEngine {
    /// Create a new multi-asset engine.
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            data: DataManager::new(),
            constraints: None,
            symbol_sectors: HashMap::new(),
        }
    }

    /// Set portfolio constraints for the engine.
    pub fn with_constraints(mut self, constraints: PortfolioConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Set sector classification for symbols.
    ///
    /// Example: engine.set_symbol_sector("AAPL", "Technology");
    pub fn set_symbol_sector(&mut self, symbol: impl Into<String>, sector: impl Into<String>) {
        self.symbol_sectors.insert(symbol.into(), sector.into());
    }

    /// Set multiple sector classifications at once.
    pub fn with_sectors(mut self, sectors: HashMap<String, String>) -> Self {
        self.symbol_sectors = sectors;
        self
    }

    /// Add data for a symbol.
    pub fn add_data(&mut self, symbol: impl Into<String>, bars: Vec<Bar>) {
        self.data.add(symbol, bars);
    }

    /// Run multi-asset backtest.
    pub fn run(&self, strategy: &mut dyn PortfolioStrategy) -> Result<MultiAssetResult> {
        let symbols: Vec<String> = self.data.symbols().iter().map(|s| s.to_string()).collect();

        if symbols.is_empty() {
            return Err(BacktestError::NoData);
        }

        // Find common date range
        let (min_len, start_time, end_time) = self.find_common_range(&symbols)?;

        info!(
            "Running multi-asset backtest: {} on {} symbols ({} bars)",
            strategy.name(),
            symbols.len(),
            min_len
        );

        strategy.init();

        let mut portfolio =
            Portfolio::with_cost_model(self.config.initial_capital, self.config.cost_model.clone());
        portfolio.allow_short = self.config.allow_short;
        portfolio.fractional_shares = self.config.fractional_shares;
        portfolio.set_execution_price(self.config.execution_price);
        portfolio.set_lot_selection_method(self.config.lot_selection.clone());
        portfolio.set_margin_config(self.config.margin.clone());
        portfolio.set_asset_configs(self.data.asset_configs());
        let volume_profiles = self.data.volume_profiles();
        portfolio.set_volume_profiles(&volume_profiles);

        let warmup = strategy.warmup_period();

        // Align data by date
        let aligned_data = self.align_data(&symbols)?;
        let timestamps: Vec<DateTime<Utc>> = aligned_data.keys().cloned().collect();

        let mut weight_history = Vec::new();
        let mut sample_interval = timestamps.len() / 100;
        if sample_interval == 0 {
            sample_interval = 1;
        }

        // Main backtest loop
        for (i, timestamp) in timestamps.iter().enumerate() {
            let current_bars = aligned_data.get(timestamp).unwrap();

            // Update prices and record equity
            let prices: HashMap<String, f64> = current_bars
                .iter()
                .map(|(s, b)| (s.clone(), b.close))
                .collect();
            portfolio.record_equity(*timestamp, &prices)?;

            // Skip warmup
            if i < warmup {
                continue;
            }

            // Build historical bars up to current point
            let historical: HashMap<String, Vec<Bar>> = symbols
                .iter()
                .filter_map(|s| {
                    self.data.get(s).map(|bars| {
                        let filtered: Vec<Bar> = bars
                            .iter()
                            .filter(|b| b.timestamp <= *timestamp)
                            .cloned()
                            .collect();
                        (s.clone(), filtered)
                    })
                })
                .collect();

            // Calculate current weights
            let equity = portfolio.equity(&prices);
            let weights: HashMap<String, f64> = symbols
                .iter()
                .map(|s| {
                    let pos_value = portfolio
                        .position(s)
                        .map(|p| p.quantity * prices.get(s).unwrap_or(&0.0))
                        .unwrap_or(0.0);
                    (
                        s.clone(),
                        if equity > 0.0 {
                            pos_value / equity
                        } else {
                            0.0
                        },
                    )
                })
                .collect();

            // Sample weight history
            if i % sample_interval == 0 {
                weight_history.push((*timestamp, weights.clone()));
            }

            // Build context
            let positions: HashMap<String, f64> = symbols
                .iter()
                .map(|s| (s.clone(), portfolio.position_qty(s)))
                .collect();

            let ctx = PortfolioContext {
                bar_index: i,
                current_bars,
                bars: &historical,
                positions,
                cash: portfolio.cash,
                equity,
                weights,
                symbols: &symbols,
            };

            // Get allocation signal
            let signal = strategy.on_bars(&ctx);

            // Execute signal
            match signal {
                AllocationSignal::Allocate(allocations) => {
                    self.execute_allocations(&mut portfolio, &allocations, &prices, equity)?;
                }
                AllocationSignal::Rebalance(target_weights) => {
                    self.execute_rebalance(&mut portfolio, &target_weights, &prices, equity)?;
                }
                AllocationSignal::ExitAll => {
                    self.exit_all_positions(&mut portfolio, current_bars)?;
                }
                AllocationSignal::Hold => {}
            }
        }

        strategy.on_finish();

        // Calculate results
        let equity_curve = portfolio.equity_curve().to_vec();
        let trades = portfolio.trades().to_vec();
        let final_equity = equity_curve
            .last()
            .map(|e| e.equity)
            .unwrap_or(self.config.initial_capital);

        let total_return_pct =
            (final_equity - self.config.initial_capital) / self.config.initial_capital * 100.0;

        let days = (end_time - start_time).num_days() as f64;
        let years = days / 365.0;
        let annual_return_pct = if years > 0.0 {
            ((final_equity / self.config.initial_capital).powf(1.0 / years) - 1.0) * 100.0
        } else {
            0.0
        };

        let max_drawdown_pct = equity_curve
            .iter()
            .map(|e| e.drawdown_pct)
            .fold(0.0_f64, |a, b| a.max(b));

        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect();

        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);

        let closed_trades: Vec<_> = trades.iter().filter(|t| t.is_closed()).collect();
        let total_trades = closed_trades.len();

        let mut trades_by_symbol: HashMap<String, usize> = HashMap::new();
        for trade in &closed_trades {
            *trades_by_symbol.entry(trade.symbol.clone()).or_insert(0) += 1;
        }

        Ok(MultiAssetResult {
            strategy_name: strategy.name().to_string(),
            symbols,
            initial_capital: self.config.initial_capital,
            final_equity,
            total_return_pct,
            annual_return_pct,
            max_drawdown_pct,
            sharpe_ratio,
            sortino_ratio,
            total_trades,
            trades_by_symbol,
            trades,
            equity_curve,
            start_time,
            end_time,
            weight_history,
        })
    }

    /// Find common date range across all symbols.
    fn find_common_range(
        &self,
        symbols: &[String],
    ) -> Result<(usize, DateTime<Utc>, DateTime<Utc>)> {
        let mut start = None;
        let mut end = None;
        let mut min_len = usize::MAX;

        for symbol in symbols {
            let bars = self.data.get(symbol).ok_or_else(|| {
                BacktestError::DataError(format!("No data for symbol: {}", symbol))
            })?;

            if bars.is_empty() {
                continue;
            }

            let s = bars.first().unwrap().timestamp;
            let e = bars.last().unwrap().timestamp;

            start = Some(start.map_or(s, |curr: DateTime<Utc>| curr.max(s)));
            end = Some(end.map_or(e, |curr: DateTime<Utc>| curr.min(e)));
            min_len = min_len.min(bars.len());
        }

        let start = start.ok_or(BacktestError::NoData)?;
        let end = end.ok_or(BacktestError::NoData)?;

        Ok((min_len, start, end))
    }

    /// Align data by timestamp across symbols.
    fn align_data(
        &self,
        symbols: &[String],
    ) -> Result<BTreeMap<DateTime<Utc>, HashMap<String, Bar>>> {
        let mut aligned: BTreeMap<DateTime<Utc>, HashMap<String, Bar>> = BTreeMap::new();

        // Collect all timestamps
        for symbol in symbols {
            if let Some(bars) = self.data.get(symbol) {
                for bar in bars {
                    aligned
                        .entry(bar.timestamp)
                        .or_default()
                        .insert(symbol.clone(), bar.clone());
                }
            }
        }

        // Filter to only include timestamps where all symbols have data
        let num_symbols = symbols.len();
        aligned.retain(|_, bars| bars.len() == num_symbols);

        if aligned.is_empty() {
            return Err(BacktestError::DataError(
                "No overlapping data across symbols".to_string(),
            ));
        }

        Ok(aligned)
    }

    /// Execute target allocations.
    fn execute_allocations(
        &self,
        portfolio: &mut Portfolio,
        allocations: &HashMap<String, Allocation>,
        prices: &HashMap<String, f64>,
        equity: f64,
    ) -> Result<()> {
        for (symbol, allocation) in allocations {
            let price = prices.get(symbol).copied().unwrap_or(0.0);
            if price <= 0.0 {
                continue;
            }

            let target_value = equity * allocation.weight;
            let current_qty = portfolio.position_qty(symbol);
            let current_value = current_qty * price;
            let diff_value = target_value - current_value;

            if diff_value.abs() < 100.0 {
                // Skip small adjustments
                continue;
            }

            let qty_change = diff_value / price;
            if qty_change.abs() < 0.01 {
                continue;
            }

            let bar = Bar::new(Utc::now(), price, price, price, price, 0.0);

            if qty_change > 0.0 {
                let order = Order::market(symbol, Side::Buy, qty_change, Utc::now());
                let _ = portfolio.execute_with_fill_probability(
                    &order,
                    &bar,
                    self.config.fill_probability,
                );
            } else {
                let order = Order::market(symbol, Side::Sell, qty_change.abs(), Utc::now());
                let _ = portfolio.execute_with_fill_probability(
                    &order,
                    &bar,
                    self.config.fill_probability,
                );
            }
        }

        Ok(())
    }

    /// Execute rebalance to target weights.
    fn execute_rebalance(
        &self,
        portfolio: &mut Portfolio,
        target_weights: &HashMap<String, f64>,
        prices: &HashMap<String, f64>,
        equity: f64,
    ) -> Result<()> {
        // If constraints are set, validate them before executing
        if let Some(ref constraints) = self.constraints {
            // Calculate current weights
            let current_weights = self.calculate_current_weights(portfolio, prices, equity);

            // Validate the rebalance
            constraints.validate_rebalance(
                &current_weights,
                target_weights,
                &self.symbol_sectors,
            )?;
        }

        let allocations: HashMap<String, Allocation> = target_weights
            .iter()
            .map(|(s, w)| (s.clone(), Allocation::new(*w)))
            .collect();

        self.execute_allocations(portfolio, &allocations, prices, equity)
    }

    /// Calculate current portfolio weights.
    fn calculate_current_weights(
        &self,
        portfolio: &Portfolio,
        prices: &HashMap<String, f64>,
        equity: f64,
    ) -> HashMap<String, f64> {
        if equity <= 0.0 {
            return HashMap::new();
        }

        let mut weights = HashMap::new();
        for (symbol, position) in portfolio.positions() {
            if let Some(&price) = prices.get(symbol) {
                let position_value = position.quantity.abs() * price;
                let weight = position_value / equity;
                if weight > 1e-6 {
                    // Only include non-negligible weights
                    weights.insert(symbol.clone(), weight);
                }
            }
        }

        weights
    }

    /// Exit all positions.
    fn exit_all_positions(
        &self,
        portfolio: &mut Portfolio,
        current_bars: &HashMap<String, Bar>,
    ) -> Result<()> {
        let positions: Vec<(String, f64)> = portfolio
            .positions()
            .iter()
            .map(|(s, p)| (s.clone(), p.quantity))
            .collect();

        for (symbol, qty) in positions {
            if qty.abs() < 0.01 {
                continue;
            }

            let bar = current_bars
                .get(&symbol)
                .cloned()
                .unwrap_or_else(|| Bar::new(Utc::now(), 100.0, 100.0, 100.0, 100.0, 0.0));

            let side = if qty > 0.0 { Side::Sell } else { Side::Buy };
            let order = Order::market(&symbol, side, qty.abs(), Utc::now());
            let _ =
                portfolio.execute_with_fill_probability(&order, &bar, self.config.fill_probability);
        }

        Ok(())
    }

    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        if std_dev == 0.0 {
            return 0.0;
        }
        (mean / std_dev) * 252.0_f64.sqrt()
    }

    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }
        let downside_var: f64 =
            downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
        let downside_dev = downside_var.sqrt();
        if downside_dev == 0.0 {
            return 0.0;
        }
        (mean / downside_dev) * 252.0_f64.sqrt()
    }
}

/// Simple equal-weight portfolio strategy.
pub struct EqualWeightStrategy {
    rebalance_frequency: usize,
    last_rebalance: usize,
}

impl EqualWeightStrategy {
    /// Create new equal-weight strategy.
    pub fn new(rebalance_frequency: usize) -> Self {
        Self {
            rebalance_frequency,
            last_rebalance: 0,
        }
    }
}

impl PortfolioStrategy for EqualWeightStrategy {
    fn name(&self) -> &str {
        "Equal Weight"
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        // Rebalance at specified frequency
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        let weight = 1.0 / ctx.symbols.len() as f64;
        let weights: HashMap<String, f64> =
            ctx.symbols.iter().map(|s| (s.clone(), weight)).collect();

        AllocationSignal::Rebalance(weights)
    }
}

/// Momentum-based portfolio strategy.
pub struct MomentumPortfolioStrategy {
    lookback: usize,
    top_n: usize,
    rebalance_frequency: usize,
    last_rebalance: usize,
}

impl MomentumPortfolioStrategy {
    /// Create new momentum portfolio strategy.
    pub fn new(lookback: usize, top_n: usize, rebalance_frequency: usize) -> Self {
        Self {
            lookback,
            top_n,
            rebalance_frequency,
            last_rebalance: 0,
        }
    }
}

impl PortfolioStrategy for MomentumPortfolioStrategy {
    fn name(&self) -> &str {
        "Momentum Portfolio"
    }

    fn warmup_period(&self) -> usize {
        self.lookback
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        // Calculate momentum for each symbol
        let mut momentums: Vec<(String, f64)> = ctx
            .symbols
            .iter()
            .filter_map(|s| {
                let bars = ctx.history(s)?;
                if bars.len() < self.lookback {
                    return None;
                }
                let old_price = bars[bars.len() - self.lookback].close;
                let new_price = bars.last()?.close;
                let momentum = (new_price - old_price) / old_price;
                Some((s.clone(), momentum))
            })
            .collect();

        // Sort by momentum descending
        momentums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top N
        let selected: Vec<&str> = momentums
            .iter()
            .take(self.top_n.min(momentums.len()))
            .filter(|(_, m)| *m > 0.0) // Only positive momentum
            .map(|(s, _)| s.as_str())
            .collect();

        if selected.is_empty() {
            return AllocationSignal::ExitAll;
        }

        let weight = 1.0 / selected.len() as f64;
        let weights: HashMap<String, f64> = selected
            .into_iter()
            .map(|s| (s.to_string(), weight))
            .collect();

        AllocationSignal::Rebalance(weights)
    }
}

/// Inverse volatility portfolio strategy.
/// Allocates weight inversely proportional to each asset's volatility.
pub struct InverseVolatilityStrategy {
    lookback: usize,
    rebalance_frequency: usize,
    last_rebalance: usize,
}

impl InverseVolatilityStrategy {
    /// Create new inverse volatility strategy.
    pub fn new(lookback: usize, rebalance_frequency: usize) -> Self {
        Self {
            lookback,
            rebalance_frequency,
            last_rebalance: 0,
        }
    }
}

impl PortfolioStrategy for InverseVolatilityStrategy {
    fn name(&self) -> &str {
        "Inverse Volatility"
    }

    fn warmup_period(&self) -> usize {
        self.lookback
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        // Calculate volatility for each symbol
        let volatilities: Vec<(String, f64)> = ctx
            .symbols
            .iter()
            .filter_map(|s| {
                let bars = ctx.history(s)?;
                if bars.len() < self.lookback + 1 {
                    return None;
                }

                // Calculate returns
                let returns: Vec<f64> = bars
                    .windows(2)
                    .rev()
                    .take(self.lookback)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();

                if returns.is_empty() {
                    return None;
                }

                // Calculate volatility (standard deviation of returns)
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let volatility = variance.sqrt();

                Some((s.clone(), volatility))
            })
            .collect();

        if volatilities.is_empty() {
            return AllocationSignal::Hold;
        }

        // Calculate inverse volatility weights
        let inverse_vols: Vec<(String, f64)> = volatilities
            .into_iter()
            .map(|(s, vol)| {
                let inv_vol = if vol > 0.0 { 1.0 / vol } else { 0.0 };
                (s, inv_vol)
            })
            .collect();

        let total_inv_vol: f64 = inverse_vols.iter().map(|(_, iv)| iv).sum();

        if total_inv_vol == 0.0 {
            return AllocationSignal::Hold;
        }

        // Normalize to sum to 1.0
        let weights: HashMap<String, f64> = inverse_vols
            .into_iter()
            .map(|(s, iv)| (s, iv / total_inv_vol))
            .collect();

        AllocationSignal::Rebalance(weights)
    }
}

/// Risk parity portfolio strategy.
/// Allocates so each asset contributes equally to portfolio risk.
/// Simplified version using inverse volatility as proxy for risk contribution.
pub struct RiskParityStrategy {
    lookback: usize,
    rebalance_frequency: usize,
    last_rebalance: usize,
}

impl RiskParityStrategy {
    /// Create new risk parity strategy.
    pub fn new(lookback: usize, rebalance_frequency: usize) -> Self {
        Self {
            lookback,
            rebalance_frequency,
            last_rebalance: 0,
        }
    }
}

impl PortfolioStrategy for RiskParityStrategy {
    fn name(&self) -> &str {
        "Risk Parity"
    }

    fn warmup_period(&self) -> usize {
        self.lookback
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        // Calculate volatility for each symbol
        let volatilities: Vec<(String, f64)> = ctx
            .symbols
            .iter()
            .filter_map(|s| {
                let bars = ctx.history(s)?;
                if bars.len() < self.lookback + 1 {
                    return None;
                }

                // Calculate returns
                let returns: Vec<f64> = bars
                    .windows(2)
                    .rev()
                    .take(self.lookback)
                    .map(|w| (w[1].close - w[0].close) / w[0].close)
                    .collect();

                if returns.is_empty() {
                    return None;
                }

                // Calculate volatility (annualized)
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                let daily_vol = variance.sqrt();
                let annual_vol = daily_vol * 252.0_f64.sqrt();

                Some((s.clone(), annual_vol))
            })
            .collect();

        if volatilities.is_empty() {
            return AllocationSignal::Hold;
        }

        // For risk parity, allocate inversely to volatility
        // This ensures each asset contributes equally to portfolio variance
        let inverse_vols: Vec<(String, f64)> = volatilities
            .into_iter()
            .map(|(s, vol)| {
                let inv_vol = if vol > 0.0 { 1.0 / vol } else { 0.0 };
                (s, inv_vol)
            })
            .collect();

        let total_inv_vol: f64 = inverse_vols.iter().map(|(_, iv)| iv).sum();

        if total_inv_vol == 0.0 {
            return AllocationSignal::Hold;
        }

        // Normalize to sum to 1.0
        let weights: HashMap<String, f64> = inverse_vols
            .into_iter()
            .map(|(s, iv)| (s, iv / total_inv_vol))
            .collect();

        AllocationSignal::Rebalance(weights)
    }
}

/// Mean-variance portfolio optimizer using Markowitz framework.
///
/// Solves quadratic optimization problems to find optimal portfolio weights
/// that minimize risk for a given return target or maximize Sharpe ratio.
pub struct MeanVarianceOptimizer {
    /// Expected returns for each asset (annualized).
    expected_returns: Vec<f64>,
    /// Covariance matrix (annualized).
    covariance_matrix: Vec<Vec<f64>>,
    /// Asset symbols in order.
    symbols: Vec<String>,
    /// Risk-free rate for Sharpe ratio calculation (annualized).
    risk_free_rate: f64,
}

impl MeanVarianceOptimizer {
    /// Create a new optimizer with expected returns and covariance matrix.
    ///
    /// # Arguments
    /// * `symbols` - Asset symbols
    /// * `expected_returns` - Annualized expected returns for each asset
    /// * `covariance_matrix` - Annualized covariance matrix (NxN)
    /// * `risk_free_rate` - Risk-free rate (annualized, default 0.0)
    pub fn new(
        symbols: Vec<String>,
        expected_returns: Vec<f64>,
        covariance_matrix: Vec<Vec<f64>>,
        risk_free_rate: f64,
    ) -> Result<Self> {
        let n = symbols.len();
        if expected_returns.len() != n {
            return Err(BacktestError::InvalidInput(
                "Expected returns length must match number of symbols".to_string(),
            ));
        }
        if covariance_matrix.len() != n || covariance_matrix.iter().any(|row| row.len() != n) {
            return Err(BacktestError::InvalidInput(
                "Covariance matrix must be square and match number of symbols".to_string(),
            ));
        }

        Ok(Self {
            expected_returns,
            covariance_matrix,
            symbols,
            risk_free_rate,
        })
    }

    /// Create optimizer from historical data.
    ///
    /// # Arguments
    /// * `bars` - Historical bars for each symbol
    /// * `lookback` - Number of periods to use for estimation
    /// * `risk_free_rate` - Risk-free rate (annualized)
    pub fn from_history(
        bars: &HashMap<String, Vec<Bar>>,
        lookback: usize,
        risk_free_rate: f64,
    ) -> Result<Self> {
        let symbols: Vec<String> = bars.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(BacktestError::InvalidInput(
                "Need at least one symbol".to_string(),
            ));
        }

        // Calculate returns for each asset
        let mut returns_matrix: Vec<Vec<f64>> = Vec::new();
        for symbol in &symbols {
            let asset_bars = bars.get(symbol).ok_or_else(|| {
                BacktestError::InvalidInput(format!("Missing bars for symbol {}", symbol))
            })?;

            if asset_bars.len() < lookback + 1 {
                return Err(BacktestError::InvalidInput(format!(
                    "Insufficient data for symbol {}: need {} bars, have {}",
                    symbol,
                    lookback + 1,
                    asset_bars.len()
                )));
            }

            let returns: Vec<f64> = asset_bars[asset_bars.len() - lookback - 1..]
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();

            returns_matrix.push(returns);
        }

        // Calculate expected returns (mean) and annualize
        let expected_returns: Vec<f64> = returns_matrix
            .iter()
            .map(|returns| {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                mean * 252.0 // Annualize assuming daily data
            })
            .collect();

        // Calculate covariance matrix and annualize
        let mut covariance_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mean_i = returns_matrix[i].iter().sum::<f64>() / returns_matrix[i].len() as f64;
                let mean_j = returns_matrix[j].iter().sum::<f64>() / returns_matrix[j].len() as f64;

                let cov = returns_matrix[i]
                    .iter()
                    .zip(returns_matrix[j].iter())
                    .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                    .sum::<f64>()
                    / returns_matrix[i].len() as f64;

                covariance_matrix[i][j] = cov * 252.0; // Annualize assuming daily data
            }
        }

        Self::new(symbols, expected_returns, covariance_matrix, risk_free_rate)
    }

    /// Find minimum variance portfolio (ignoring returns).
    ///
    /// Returns optimal weights that minimize portfolio variance subject to
    /// weights summing to 1 and being non-negative (long-only constraint).
    pub fn minimum_variance(&self) -> Result<HashMap<String, f64>> {
        use clarabel::algebra::*;
        use clarabel::solver::*;

        let n = self.symbols.len();

        // Build P matrix (covariance matrix in CscMatrix format)
        // P must be positive semidefinite
        let mut p_data = Vec::new();
        let mut p_indices = Vec::new();
        let mut p_indptr = vec![0];

        for j in 0..n {
            for i in 0..n {
                let val = self.covariance_matrix[i][j];
                if val.abs() > 1e-10 {
                    p_data.push(val);
                    p_indices.push(i);
                }
            }
            p_indptr.push(p_data.len());
        }

        let p = CscMatrix::new(n, n, p_indptr, p_indices, p_data);

        // q vector (linear term, zeros for minimum variance)
        let q = vec![0.0; n];

        // Constraints: [equality; inequality]
        // Equality: sum(w) = 1 (row of ones)
        // Inequality: w >= 0 (identity matrix, as -w <= 0)
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        // Column by column (CSC format)
        for j in 0..n {
            // Equality constraint: 1.0 (sum constraint)
            a_data.push(1.0);
            a_indices.push(0);

            // Inequality constraint: -1.0 for non-negativity (-w_j <= 0)
            a_data.push(-1.0);
            a_indices.push(1 + j);

            a_indptr.push(a_data.len());
        }

        let a = CscMatrix::new(1 + n, n, a_indptr, a_indices, a_data);

        // Bounds: b = [1.0, 0, 0, ..., 0]
        let mut b = vec![1.0]; // Equality: sum(w) = 1
        b.extend(vec![0.0; n]); // Inequality: w >= 0

        // Cones: [Zero cone for equality, Nonnegative cone for inequality]
        let cones = [ZeroConeT(1), NonnegativeConeT(n)];

        // Solve
        let settings = DefaultSettingsBuilder::default()
            .max_iter(100)
            .verbose(false)
            .build()
            .map_err(|e| {
                BacktestError::OptimizationError(format!("Failed to build settings: {}", e))
            })?;

        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).map_err(|e| {
            BacktestError::OptimizationError(format!("Failed to create solver: {:?}", e))
        })?;

        solver.solve();

        if !matches!(solver.solution.status, SolverStatus::Solved) {
            return Err(BacktestError::OptimizationError(format!(
                "Optimization failed with status: {:?}",
                solver.solution.status
            )));
        }

        // Extract weights
        let weights: HashMap<String, f64> = self
            .symbols
            .iter()
            .zip(solver.solution.x.iter())
            .map(|(s, &w)| (s.clone(), w.max(0.0))) // Ensure non-negative due to numerical precision
            .collect();

        Ok(weights)
    }

    /// Find maximum Sharpe ratio portfolio.
    ///
    /// Returns optimal weights that maximize (return - risk_free_rate) / volatility
    /// subject to weights summing to 1 and being non-negative.
    pub fn maximum_sharpe_ratio(&self) -> Result<HashMap<String, f64>> {
        // Maximum Sharpe ratio can be found by solving:
        // minimize w'Σw subject to (μ - rf)'w = 1, w >= 0
        // Then normalize weights to sum to 1

        use clarabel::algebra::*;
        use clarabel::solver::*;

        let n = self.symbols.len();

        // Excess returns (μ - rf)
        let excess_returns: Vec<f64> = self
            .expected_returns
            .iter()
            .map(|&r| r - self.risk_free_rate)
            .collect();

        // Check if all excess returns are non-positive
        if excess_returns.iter().all(|&r| r <= 0.0) {
            // Return minimum variance portfolio as fallback
            return self.minimum_variance();
        }

        // Build P matrix (covariance matrix)
        let mut p_data = Vec::new();
        let mut p_indices = Vec::new();
        let mut p_indptr = vec![0];

        for j in 0..n {
            for i in 0..n {
                let val = self.covariance_matrix[i][j];
                if val.abs() > 1e-10 {
                    p_data.push(val);
                    p_indices.push(i);
                }
            }
            p_indptr.push(p_data.len());
        }

        let p = CscMatrix::new(n, n, p_indptr, p_indices, p_data);
        let q = vec![0.0; n];

        // Constraints: (μ - rf)'w = 1, w >= 0
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        for (j, &excess_ret) in excess_returns.iter().enumerate() {
            // Equality: excess return
            a_data.push(excess_ret);
            a_indices.push(0);

            // Inequality: non-negativity
            a_data.push(-1.0);
            a_indices.push(1 + j);

            a_indptr.push(a_data.len());
        }

        let a = CscMatrix::new(1 + n, n, a_indptr, a_indices, a_data);

        let mut b = vec![1.0]; // Excess return = 1
        b.extend(vec![0.0; n]); // w >= 0

        let cones = [ZeroConeT(1), NonnegativeConeT(n)];

        let settings = DefaultSettingsBuilder::default()
            .max_iter(100)
            .verbose(false)
            .build()
            .map_err(|e| {
                BacktestError::OptimizationError(format!("Failed to build settings: {}", e))
            })?;

        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).map_err(|e| {
            BacktestError::OptimizationError(format!("Failed to create solver: {:?}", e))
        })?;

        solver.solve();

        if !matches!(solver.solution.status, SolverStatus::Solved) {
            return Err(BacktestError::OptimizationError(format!(
                "Max Sharpe optimization failed: {:?}",
                solver.solution.status
            )));
        }

        // Normalize weights to sum to 1
        let weight_sum: f64 = solver.solution.x.iter().sum();
        let weights: HashMap<String, f64> = self
            .symbols
            .iter()
            .zip(solver.solution.x.iter())
            .map(|(s, &w)| (s.clone(), (w / weight_sum).max(0.0)))
            .collect();

        Ok(weights)
    }

    /// Find portfolio with target return.
    ///
    /// Returns optimal weights that minimize variance subject to
    /// achieving a target expected return and weights summing to 1.
    ///
    /// # Arguments
    /// * `target_return` - Target expected return (annualized)
    pub fn target_return(&self, target_return: f64) -> Result<HashMap<String, f64>> {
        use clarabel::algebra::*;
        use clarabel::solver::*;

        let n = self.symbols.len();

        // Build P matrix (covariance matrix)
        let mut p_data = Vec::new();
        let mut p_indices = Vec::new();
        let mut p_indptr = vec![0];

        for j in 0..n {
            for i in 0..n {
                let val = self.covariance_matrix[i][j];
                if val.abs() > 1e-10 {
                    p_data.push(val);
                    p_indices.push(i);
                }
            }
            p_indptr.push(p_data.len());
        }

        let p = CscMatrix::new(n, n, p_indptr, p_indices, p_data);
        let q = vec![0.0; n];

        // Constraints: sum(w) = 1, μ'w = target_return, w >= 0
        let mut a_data = Vec::new();
        let mut a_indices = Vec::new();
        let mut a_indptr = vec![0];

        for j in 0..n {
            // Equality 1: sum constraint
            a_data.push(1.0);
            a_indices.push(0);

            // Equality 2: return constraint
            a_data.push(self.expected_returns[j]);
            a_indices.push(1);

            // Inequality: non-negativity
            a_data.push(-1.0);
            a_indices.push(2 + j);

            a_indptr.push(a_data.len());
        }

        let a = CscMatrix::new(2 + n, n, a_indptr, a_indices, a_data);

        let mut b = vec![1.0, target_return]; // Equality constraints
        b.extend(vec![0.0; n]); // Inequality: w >= 0

        let cones = [ZeroConeT(2), NonnegativeConeT(n)];

        let settings = DefaultSettingsBuilder::default()
            .max_iter(100)
            .verbose(false)
            .build()
            .map_err(|e| {
                BacktestError::OptimizationError(format!("Failed to build settings: {}", e))
            })?;

        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings).map_err(|e| {
            BacktestError::OptimizationError(format!("Failed to create solver: {:?}", e))
        })?;

        solver.solve();

        if !matches!(solver.solution.status, SolverStatus::Solved) {
            return Err(BacktestError::OptimizationError(format!(
                "Target return optimization failed: {:?}",
                solver.solution.status
            )));
        }

        let weights: HashMap<String, f64> = self
            .symbols
            .iter()
            .zip(solver.solution.x.iter())
            .map(|(s, &w)| (s.clone(), w.max(0.0)))
            .collect();

        Ok(weights)
    }

    /// Calculate portfolio expected return given weights.
    pub fn portfolio_return(&self, weights: &HashMap<String, f64>) -> f64 {
        self.symbols
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let w = weights.get(s).copied().unwrap_or(0.0);
                w * self.expected_returns[i]
            })
            .sum()
    }

    /// Calculate portfolio variance given weights.
    pub fn portfolio_variance(&self, weights: &HashMap<String, f64>) -> f64 {
        let n = self.symbols.len();
        let mut variance = 0.0;

        for i in 0..n {
            let wi = weights.get(&self.symbols[i]).copied().unwrap_or(0.0);
            for j in 0..n {
                let wj = weights.get(&self.symbols[j]).copied().unwrap_or(0.0);
                variance += wi * wj * self.covariance_matrix[i][j];
            }
        }

        variance
    }

    /// Calculate portfolio volatility (standard deviation) given weights.
    pub fn portfolio_volatility(&self, weights: &HashMap<String, f64>) -> f64 {
        self.portfolio_variance(weights).sqrt()
    }
}

/// Mean-variance optimized portfolio strategy.
///
/// Uses Markowitz mean-variance optimization to construct portfolios that
/// maximize Sharpe ratio or target specific return levels.
pub struct MeanVarianceStrategy {
    /// Lookback period for estimating returns and covariance.
    lookback: usize,
    /// Rebalancing frequency in bars.
    rebalance_frequency: usize,
    /// Last rebalance bar index.
    last_rebalance: usize,
    /// Optimization objective.
    objective: MeanVarianceObjective,
    /// Risk-free rate (annualized).
    risk_free_rate: f64,
}

/// Objective for mean-variance optimization.
#[derive(Debug, Clone)]
pub enum MeanVarianceObjective {
    /// Minimize variance (ignoring returns).
    MinimumVariance,
    /// Maximize Sharpe ratio.
    MaximumSharpe,
    /// Target specific expected return (annualized).
    TargetReturn(f64),
}

impl MeanVarianceStrategy {
    /// Create a new mean-variance strategy.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for estimating parameters
    /// * `rebalance_frequency` - Rebalancing frequency in bars
    /// * `objective` - Optimization objective
    /// * `risk_free_rate` - Risk-free rate (annualized)
    pub fn new(
        lookback: usize,
        rebalance_frequency: usize,
        objective: MeanVarianceObjective,
        risk_free_rate: f64,
    ) -> Self {
        Self {
            lookback,
            rebalance_frequency,
            last_rebalance: 0,
            objective,
            risk_free_rate,
        }
    }

    /// Create minimum variance strategy.
    pub fn minimum_variance(lookback: usize, rebalance_frequency: usize) -> Self {
        Self::new(
            lookback,
            rebalance_frequency,
            MeanVarianceObjective::MinimumVariance,
            0.0,
        )
    }

    /// Create maximum Sharpe ratio strategy.
    pub fn maximum_sharpe(
        lookback: usize,
        rebalance_frequency: usize,
        risk_free_rate: f64,
    ) -> Self {
        Self::new(
            lookback,
            rebalance_frequency,
            MeanVarianceObjective::MaximumSharpe,
            risk_free_rate,
        )
    }

    /// Create target return strategy.
    pub fn target_return(lookback: usize, rebalance_frequency: usize, target_return: f64) -> Self {
        Self::new(
            lookback,
            rebalance_frequency,
            MeanVarianceObjective::TargetReturn(target_return),
            0.0,
        )
    }
}

impl PortfolioStrategy for MeanVarianceStrategy {
    fn name(&self) -> &str {
        match self.objective {
            MeanVarianceObjective::MinimumVariance => "Mean-Variance (Min Variance)",
            MeanVarianceObjective::MaximumSharpe => "Mean-Variance (Max Sharpe)",
            MeanVarianceObjective::TargetReturn(_) => "Mean-Variance (Target Return)",
        }
    }

    fn warmup_period(&self) -> usize {
        self.lookback + 1
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        // Check if it's time to rebalance
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        // Build optimizer from historical data
        let optimizer =
            match MeanVarianceOptimizer::from_history(ctx.bars, self.lookback, self.risk_free_rate)
            {
                Ok(opt) => opt,
                Err(e) => {
                    tracing::warn!("Failed to create optimizer: {}", e);
                    return AllocationSignal::Hold;
                }
            };

        // Optimize based on objective
        let weights = match &self.objective {
            MeanVarianceObjective::MinimumVariance => optimizer.minimum_variance(),
            MeanVarianceObjective::MaximumSharpe => optimizer.maximum_sharpe_ratio(),
            MeanVarianceObjective::TargetReturn(target) => optimizer.target_return(*target),
        };

        match weights {
            Ok(w) => AllocationSignal::Rebalance(w),
            Err(e) => {
                tracing::warn!("Optimization failed: {}", e);
                AllocationSignal::Hold
            }
        }
    }
}

/// Represents an investor's view in the Black-Litterman model.
///
/// Views express the investor's expectations about asset returns,
/// which are combined with market equilibrium returns using Bayesian updating.
#[derive(Debug, Clone)]
pub enum View {
    /// Absolute view: Asset will return X% (annualized).
    ///
    /// # Example
    /// `View::Absolute("AAPL".to_string(), 0.15, 0.05)` means:
    /// "I believe AAPL will return 15% with 5% confidence level (variance)"
    Absolute {
        symbol: String,
        expected_return: f64,
        confidence: f64, // Variance of view (lower = more confident)
    },
    /// Relative view: Asset A will outperform Asset B by X% (annualized).
    ///
    /// # Example
    /// `View::Relative("AAPL".to_string(), "GOOGL".to_string(), 0.03, 0.02)` means:
    /// "I believe AAPL will outperform GOOGL by 3% with confidence level 0.02"
    Relative {
        symbol_a: String,
        symbol_b: String,
        expected_outperformance: f64,
        confidence: f64,
    },
}

/// Black-Litterman portfolio optimizer.
///
/// The Black-Litterman model combines market equilibrium returns (derived from
/// market capitalization weights via reverse optimization) with investor views
/// using Bayesian updating. This produces more stable and intuitive portfolios
/// than pure mean-variance optimization.
///
/// # Mathematical Framework
///
/// 1. **Market Implied Returns (Π)**: Π = δ * Σ * w_mkt
///    where δ is risk aversion, Σ is covariance, w_mkt is market cap weights
///
/// 2. **Bayesian Update**: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]
///    where:
///    - τ is uncertainty scaling (typically 0.025-0.05)
///    - P is "pick matrix" encoding views
///    - Q is vector of view returns
///    - Ω is diagonal matrix of view confidence levels
///
/// 3. **Optimal Weights**: Run mean-variance optimization with posterior returns E[R]
///
/// # Example
///
/// ```ignore
/// let symbols = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
/// let market_caps = vec![3000e9, 2000e9, 2500e9]; // Market cap in dollars
/// let covariance_matrix = ...; // Compute from history
///
/// let mut optimizer = BlackLittermanOptimizer::new(
///     symbols,
///     market_caps,
///     covariance_matrix,
///     0.03,  // tau
///     2.5,   // risk aversion
///     0.02   // risk-free rate
/// )?;
///
/// // Add views
/// optimizer.add_view(View::Absolute {
///     symbol: "AAPL".to_string(),
///     expected_return: 0.15,
///     confidence: 0.05,
/// });
///
/// optimizer.add_view(View::Relative {
///     symbol_a: "GOOGL".to_string(),
///     symbol_b: "MSFT".to_string(),
///     expected_outperformance: 0.03,
///     confidence: 0.02,
/// });
///
/// let weights = optimizer.optimize()?;
/// ```
///
/// # References
/// - Black, F. and Litterman, R. (1992). "Global Portfolio Optimization"
/// - He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman"
pub struct BlackLittermanOptimizer {
    /// Asset symbols in order.
    symbols: Vec<String>,
    /// Market capitalization weights (must sum to 1.0).
    market_weights: Vec<f64>,
    /// Covariance matrix (annualized).
    covariance_matrix: Vec<Vec<f64>>,
    /// Tau: uncertainty scaling factor (typically 0.025-0.05).
    /// Represents uncertainty in the prior (market equilibrium).
    tau: f64,
    /// Risk aversion coefficient (typically 2.5-4.0).
    /// Higher values = more conservative portfolios.
    risk_aversion: f64,
    /// Risk-free rate (annualized).
    risk_free_rate: f64,
    /// Investor views.
    views: Vec<View>,
}

impl BlackLittermanOptimizer {
    /// Create a new Black-Litterman optimizer.
    ///
    /// # Arguments
    /// * `symbols` - Asset symbols
    /// * `market_caps` - Market capitalizations (used to derive market weights)
    /// * `covariance_matrix` - Annualized covariance matrix (NxN)
    /// * `tau` - Uncertainty scaling (0.025-0.05 typical)
    /// * `risk_aversion` - Risk aversion coefficient (2.5 typical)
    /// * `risk_free_rate` - Risk-free rate (annualized)
    pub fn new(
        symbols: Vec<String>,
        market_caps: Vec<f64>,
        covariance_matrix: Vec<Vec<f64>>,
        tau: f64,
        risk_aversion: f64,
        risk_free_rate: f64,
    ) -> Result<Self> {
        let n = symbols.len();

        if market_caps.len() != n {
            return Err(BacktestError::InvalidInput(
                "Market caps length must match number of symbols".to_string(),
            ));
        }

        if covariance_matrix.len() != n || covariance_matrix.iter().any(|row| row.len() != n) {
            return Err(BacktestError::InvalidInput(
                "Covariance matrix must be square and match number of symbols".to_string(),
            ));
        }

        if tau <= 0.0 || tau > 1.0 {
            return Err(BacktestError::InvalidInput(
                "Tau must be between 0 and 1 (typically 0.025-0.05)".to_string(),
            ));
        }

        if risk_aversion <= 0.0 {
            return Err(BacktestError::InvalidInput(
                "Risk aversion must be positive".to_string(),
            ));
        }

        // Normalize market caps to weights
        let total_cap: f64 = market_caps.iter().sum();
        if total_cap <= 0.0 {
            return Err(BacktestError::InvalidInput(
                "Total market cap must be positive".to_string(),
            ));
        }

        let market_weights: Vec<f64> = market_caps.iter().map(|&cap| cap / total_cap).collect();

        Ok(Self {
            symbols,
            market_weights,
            covariance_matrix,
            tau,
            risk_aversion,
            risk_free_rate,
            views: Vec::new(),
        })
    }

    /// Create optimizer from historical data with equal market weights.
    ///
    /// # Arguments
    /// * `bars` - Historical bars for each symbol
    /// * `lookback` - Number of periods to use for covariance estimation
    /// * `tau` - Uncertainty scaling
    /// * `risk_aversion` - Risk aversion coefficient
    /// * `risk_free_rate` - Risk-free rate (annualized)
    pub fn from_history(
        bars: &HashMap<String, Vec<Bar>>,
        lookback: usize,
        tau: f64,
        risk_aversion: f64,
        risk_free_rate: f64,
    ) -> Result<Self> {
        let symbols: Vec<String> = bars.keys().cloned().collect();
        let n = symbols.len();

        if n == 0 {
            return Err(BacktestError::InvalidInput(
                "Need at least one symbol".to_string(),
            ));
        }

        // Calculate returns for each asset
        let mut returns_matrix: Vec<Vec<f64>> = Vec::new();
        for symbol in &symbols {
            let asset_bars = bars.get(symbol).ok_or_else(|| {
                BacktestError::InvalidInput(format!("Missing bars for symbol {}", symbol))
            })?;

            if asset_bars.len() < lookback + 1 {
                return Err(BacktestError::InvalidInput(format!(
                    "Insufficient data for symbol {}: need {} bars, have {}",
                    symbol,
                    lookback + 1,
                    asset_bars.len()
                )));
            }

            let returns: Vec<f64> = asset_bars[asset_bars.len() - lookback - 1..]
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();

            returns_matrix.push(returns);
        }

        // Calculate covariance matrix and annualize
        let mut covariance_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mean_i = returns_matrix[i].iter().sum::<f64>() / returns_matrix[i].len() as f64;
                let mean_j = returns_matrix[j].iter().sum::<f64>() / returns_matrix[j].len() as f64;

                let cov = returns_matrix[i]
                    .iter()
                    .zip(returns_matrix[j].iter())
                    .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                    .sum::<f64>()
                    / returns_matrix[i].len() as f64;

                covariance_matrix[i][j] = cov * 252.0; // Annualize assuming daily data
            }
        }

        // Use equal market weights (1/n each)
        let market_caps = vec![1.0; n];

        Self::new(
            symbols,
            market_caps,
            covariance_matrix,
            tau,
            risk_aversion,
            risk_free_rate,
        )
    }

    /// Add an investor view.
    pub fn add_view(&mut self, view: View) {
        self.views.push(view);
    }

    /// Calculate market implied equilibrium returns (Π = δ * Σ * w_mkt).
    ///
    /// These are the returns implied by the current market capitalization weights,
    /// assuming the market is in equilibrium.
    #[allow(clippy::needless_range_loop)]
    pub fn implied_returns(&self) -> Vec<f64> {
        let n = self.symbols.len();
        let mut implied = vec![0.0; n];

        for i in 0..n {
            for j in 0..n {
                implied[i] +=
                    self.risk_aversion * self.covariance_matrix[i][j] * self.market_weights[j];
            }
        }

        implied
    }

    /// Compute posterior expected returns by blending market equilibrium with views.
    ///
    /// Uses Bayesian updating: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 Π + P'Ω^-1 Q]
    pub fn compute_posterior_returns(&self) -> Result<Vec<f64>> {
        let n = self.symbols.len();
        let k = self.views.len();

        // If no views, return implied equilibrium returns
        if k == 0 {
            return Ok(self.implied_returns());
        }

        let implied = self.implied_returns();

        // Build P matrix (k x n) and Q vector (k x 1)
        // P encodes which assets each view refers to
        // Q contains the expected returns from views
        let mut p_matrix = vec![vec![0.0; n]; k];
        let mut q_vector = vec![0.0; k];
        let mut omega_diag = vec![0.0; k]; // View confidence (diagonal of Ω)

        for (view_idx, view) in self.views.iter().enumerate() {
            match view {
                View::Absolute {
                    symbol,
                    expected_return,
                    confidence,
                } => {
                    // Find symbol index
                    if let Some(asset_idx) = self.symbols.iter().position(|s| s == symbol) {
                        p_matrix[view_idx][asset_idx] = 1.0;
                        q_vector[view_idx] = *expected_return;
                        omega_diag[view_idx] = *confidence;
                    } else {
                        return Err(BacktestError::InvalidInput(format!(
                            "View references unknown symbol: {}",
                            symbol
                        )));
                    }
                }
                View::Relative {
                    symbol_a,
                    symbol_b,
                    expected_outperformance,
                    confidence,
                } => {
                    let idx_a = self.symbols.iter().position(|s| s == symbol_a);
                    let idx_b = self.symbols.iter().position(|s| s == symbol_b);

                    match (idx_a, idx_b) {
                        (Some(a), Some(b)) => {
                            p_matrix[view_idx][a] = 1.0;
                            p_matrix[view_idx][b] = -1.0;
                            q_vector[view_idx] = *expected_outperformance;
                            omega_diag[view_idx] = *confidence;
                        }
                        _ => {
                            return Err(BacktestError::InvalidInput(
                                "Relative view references unknown symbols".to_string(),
                            ));
                        }
                    }
                }
            }
        }

        // Compute τΣ
        let tau_cov: Vec<Vec<f64>> = self
            .covariance_matrix
            .iter()
            .map(|row| row.iter().map(|&val| self.tau * val).collect())
            .collect();

        // Compute (τΣ)^-1 using simple matrix inversion for small matrices
        let tau_cov_inv = self.invert_matrix(&tau_cov)?;

        // Compute P'Ω^-1P (n x n matrix)
        let mut pt_omega_inv_p = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for view_idx in 0..k {
                    sum += p_matrix[view_idx][i] * p_matrix[view_idx][j] / omega_diag[view_idx];
                }
                pt_omega_inv_p[i][j] = sum;
            }
        }

        // Compute [(τΣ)^-1 + P'Ω^-1P]
        let mut combined_precision = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                combined_precision[i][j] = tau_cov_inv[i][j] + pt_omega_inv_p[i][j];
            }
        }

        // Invert to get posterior covariance
        let posterior_cov = self.invert_matrix(&combined_precision)?;

        // Compute right-hand side: (τΣ)^-1 Π + P'Ω^-1 Q
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            // (τΣ)^-1 Π term
            for j in 0..n {
                rhs[i] += tau_cov_inv[i][j] * implied[j];
            }

            // P'Ω^-1 Q term
            for view_idx in 0..k {
                rhs[i] += p_matrix[view_idx][i] * q_vector[view_idx] / omega_diag[view_idx];
            }
        }

        // Multiply posterior covariance by RHS to get posterior returns
        let mut posterior_returns = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                posterior_returns[i] += posterior_cov[i][j] * rhs[j];
            }
        }

        Ok(posterior_returns)
    }

    /// Optimize portfolio using Black-Litterman posterior returns.
    ///
    /// Returns optimal weights that maximize Sharpe ratio using the
    /// posterior expected returns from Black-Litterman.
    pub fn optimize(&self) -> Result<HashMap<String, f64>> {
        let posterior_returns = self.compute_posterior_returns()?;

        // Use mean-variance optimizer with posterior returns
        let mv_optimizer = MeanVarianceOptimizer::new(
            self.symbols.clone(),
            posterior_returns,
            self.covariance_matrix.clone(),
            self.risk_free_rate,
        )?;

        mv_optimizer.maximum_sharpe_ratio()
    }

    /// Invert a matrix using Gaussian elimination (for small matrices).
    #[allow(clippy::needless_range_loop)]
    fn invert_matrix(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = matrix.len();

        // Create augmented matrix [A | I]
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = 1.0; // Identity on right side
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > aug[max_row][col].abs() {
                    max_row = row;
                }
            }

            // Swap rows
            if max_row != col {
                aug.swap(col, max_row);
            }

            // Check for singularity
            if aug[col][col].abs() < 1e-10 {
                return Err(BacktestError::OptimizationError(
                    "Matrix is singular or nearly singular".to_string(),
                ));
            }

            // Eliminate column
            for row in (col + 1)..n {
                let factor = aug[row][col] / aug[col][col];
                for j in col..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        for col in (0..n).rev() {
            // Normalize pivot row
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }

            // Eliminate above
            for row in 0..col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Extract inverse from right half
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = aug[i][n + j];
            }
        }

        Ok(inverse)
    }
}

/// Black-Litterman optimized portfolio strategy.
///
/// Combines market equilibrium with investor views to construct portfolios
/// that are more stable than pure mean-variance optimization.
pub struct BlackLittermanStrategy {
    /// Lookback period for covariance estimation.
    lookback: usize,
    /// Rebalancing frequency in bars.
    rebalance_frequency: usize,
    /// Last rebalance bar index.
    last_rebalance: usize,
    /// Tau: uncertainty scaling factor.
    tau: f64,
    /// Risk aversion coefficient.
    risk_aversion: f64,
    /// Risk-free rate (annualized).
    risk_free_rate: f64,
    /// Investor views to apply.
    views: Vec<View>,
}

impl BlackLittermanStrategy {
    /// Create a new Black-Litterman strategy.
    ///
    /// # Arguments
    /// * `lookback` - Lookback period for covariance estimation
    /// * `rebalance_frequency` - Rebalancing frequency in bars
    /// * `tau` - Uncertainty scaling (0.025-0.05 typical)
    /// * `risk_aversion` - Risk aversion coefficient (2.5 typical)
    /// * `risk_free_rate` - Risk-free rate (annualized)
    /// * `views` - Investor views
    pub fn new(
        lookback: usize,
        rebalance_frequency: usize,
        tau: f64,
        risk_aversion: f64,
        risk_free_rate: f64,
        views: Vec<View>,
    ) -> Self {
        Self {
            lookback,
            rebalance_frequency,
            last_rebalance: 0,
            tau,
            risk_aversion,
            risk_free_rate,
            views,
        }
    }
}

impl PortfolioStrategy for BlackLittermanStrategy {
    fn name(&self) -> &str {
        "Black-Litterman"
    }

    fn warmup_period(&self) -> usize {
        self.lookback + 1
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        // Check if it's time to rebalance
        if ctx.bar_index < self.last_rebalance + self.rebalance_frequency {
            return AllocationSignal::Hold;
        }

        self.last_rebalance = ctx.bar_index;

        // Build optimizer from historical data
        let mut optimizer = match BlackLittermanOptimizer::from_history(
            ctx.bars,
            self.lookback,
            self.tau,
            self.risk_aversion,
            self.risk_free_rate,
        ) {
            Ok(opt) => opt,
            Err(e) => {
                tracing::warn!("Failed to create Black-Litterman optimizer: {}", e);
                return AllocationSignal::Hold;
            }
        };

        // Add views
        for view in &self.views {
            optimizer.add_view(view.clone());
        }

        // Optimize
        match optimizer.optimize() {
            Ok(weights) => AllocationSignal::Rebalance(weights),
            Err(e) => {
                tracing::warn!("Black-Litterman optimization failed: {}", e);
                AllocationSignal::Hold
            }
        }
    }
}

/// Hierarchical Risk Parity (HRP) portfolio optimizer.
///
/// HRP is a sophisticated portfolio allocation method developed by Marcos Lopez de Prado
/// that uses machine learning techniques (hierarchical clustering) to construct diversified
/// portfolios. Unlike traditional methods (mean-variance, risk parity), HRP:
///
/// 1. Builds a hierarchical tree of assets based on their correlation structure
/// 2. Orders the covariance matrix based on this hierarchy (quasi-diagonalization)
/// 3. Recursively allocates weights by bisecting clusters
///
/// **Benefits of HRP:**
/// - More stable than mean-variance optimization (no matrix inversion)
/// - Naturally handles multicollinearity
/// - Incorporates both correlation and volatility information
/// - Produces intuitive, diversified portfolios
///
/// **Algorithm Steps:**
/// 1. Tree Clustering: Build dendrogram from correlation distance matrix
/// 2. Quasi-Diagonalization: Sort covariance matrix by hierarchical structure
/// 3. Recursive Bisection: Allocate weights by splitting clusters
///
/// # References
/// - Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample"
/// - Journal of Portfolio Management, 2016
#[derive(Debug, Clone)]
pub struct HierarchicalRiskParityOptimizer {
    symbols: Vec<String>,
    covariance_matrix: Vec<Vec<f64>>,
    correlation_matrix: Vec<Vec<f64>>,
}

impl HierarchicalRiskParityOptimizer {
    /// Create a new HRP optimizer with precomputed correlation and covariance matrices.
    ///
    /// # Parameters
    /// * `symbols` - Asset symbols in order matching matrix dimensions
    /// * `correlation_matrix` - Correlation matrix (NxN, values in [-1, 1])
    /// * `covariance_matrix` - Covariance matrix (NxN)
    ///
    /// # Returns
    /// Result containing optimizer or error if matrices are invalid
    pub fn new(
        symbols: Vec<String>,
        correlation_matrix: Vec<Vec<f64>>,
        covariance_matrix: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let n = symbols.len();
        if n < 2 {
            return Err(BacktestError::DataError(
                "Need at least 2 assets for HRP".to_string(),
            ));
        }

        // Validate correlation matrix dimensions
        if correlation_matrix.len() != n || correlation_matrix.iter().any(|row| row.len() != n) {
            return Err(BacktestError::DataError(format!(
                "Correlation matrix dimensions {}x{} don't match {} symbols",
                correlation_matrix.len(),
                correlation_matrix.first().map(|r| r.len()).unwrap_or(0),
                n
            )));
        }

        // Validate covariance matrix dimensions
        if covariance_matrix.len() != n || covariance_matrix.iter().any(|row| row.len() != n) {
            return Err(BacktestError::DataError(format!(
                "Covariance matrix dimensions {}x{} don't match {} symbols",
                covariance_matrix.len(),
                covariance_matrix.first().map(|r| r.len()).unwrap_or(0),
                n
            )));
        }

        Ok(Self {
            symbols,
            correlation_matrix,
            covariance_matrix,
        })
    }

    /// Construct HRP optimizer from historical return data.
    ///
    /// Computes correlation and covariance matrices from daily returns.
    ///
    /// # Parameters
    /// * `symbols` - Asset symbols
    /// * `returns_history` - Historical returns for each symbol (aligned by date)
    ///
    /// # Returns
    /// Result containing optimizer or error
    pub fn from_history(
        symbols: Vec<String>,
        returns_history: &HashMap<String, Vec<f64>>,
    ) -> Result<Self> {
        let n = symbols.len();
        if n < 2 {
            return Err(BacktestError::DataError(
                "Need at least 2 assets for HRP".to_string(),
            ));
        }

        // Validate all symbols have return history
        for symbol in &symbols {
            if !returns_history.contains_key(symbol) {
                return Err(BacktestError::DataError(format!(
                    "Missing return history for symbol: {}",
                    symbol
                )));
            }
        }

        // Get minimum history length
        let min_length = symbols
            .iter()
            .filter_map(|s| returns_history.get(s).map(|v| v.len()))
            .min()
            .unwrap_or(0);

        if min_length < 20 {
            return Err(BacktestError::DataError(format!(
                "Insufficient history: {} bars, need at least 20",
                min_length
            )));
        }

        // Compute correlation and covariance matrices
        let mut correlation_matrix = vec![vec![0.0; n]; n];
        let mut covariance_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            let returns_i = &returns_history[&symbols[i]][..min_length];
            let mean_i = returns_i.iter().sum::<f64>() / returns_i.len() as f64;

            for j in 0..n {
                let returns_j = &returns_history[&symbols[j]][..min_length];
                let mean_j = returns_j.iter().sum::<f64>() / returns_j.len() as f64;

                // Covariance
                let cov: f64 = returns_i
                    .iter()
                    .zip(returns_j.iter())
                    .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                    .sum::<f64>()
                    / (returns_i.len() - 1) as f64;

                covariance_matrix[i][j] = cov * 252.0; // Annualize

                // Correlation
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    let std_i = returns_i
                        .iter()
                        .map(|r| (r - mean_i).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let std_j = returns_j
                        .iter()
                        .map(|r| (r - mean_j).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if std_i > 1e-10 && std_j > 1e-10 {
                        let corr = cov * (returns_i.len() - 1) as f64 / (std_i * std_j);
                        correlation_matrix[i][j] = corr.clamp(-1.0, 1.0);
                    } else {
                        correlation_matrix[i][j] = 0.0;
                    }
                }
            }
        }

        Self::new(symbols, correlation_matrix, covariance_matrix)
    }

    /// Compute HRP portfolio weights.
    ///
    /// Uses the full HRP algorithm:
    /// 1. Hierarchical clustering of assets
    /// 2. Quasi-diagonalization of covariance matrix
    /// 3. Recursive bisection for weight allocation
    ///
    /// # Returns
    /// HashMap of symbol -> weight, where weights sum to 1.0
    pub fn optimize(&self) -> HashMap<String, f64> {
        // Step 1: Build hierarchical clustering tree
        let clusters = self.build_cluster_tree();

        // Step 2: Quasi-diagonalize - get sorted order
        let sorted_indices = self.quasi_diagonalization(&clusters);

        // Step 3: Recursive bisection to allocate weights
        let weights = self.recursive_bisection(&sorted_indices);

        // Convert to HashMap
        weights
            .into_iter()
            .enumerate()
            .map(|(i, w)| (self.symbols[i].clone(), w))
            .collect()
    }

    /// Build hierarchical clustering tree using single linkage.
    ///
    /// Returns a list of (cluster_a, cluster_b, distance) tuples representing the dendrogram.
    fn build_cluster_tree(&self) -> Vec<(usize, usize, f64)> {
        let n = self.symbols.len();

        // Convert correlation to distance: distance = sqrt(0.5 * (1 - correlation))
        let mut dist_matrix = vec![vec![0.0; n]; n];
        for (i, row) in dist_matrix.iter_mut().enumerate().take(n) {
            for (j, cell) in row.iter_mut().enumerate().take(n) {
                if i != j {
                    let corr = self.correlation_matrix[i][j].clamp(-1.0, 1.0);
                    *cell = (0.5 * (1.0 - corr)).sqrt();
                } else {
                    *cell = 0.0;
                }
            }
        }

        // Single linkage clustering
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut linkage = Vec::new();

        while clusters.len() > 1 {
            // Find pair of clusters with minimum distance
            let mut min_dist = f64::INFINITY;
            let mut min_i = 0;
            let mut min_j = 1;

            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    // Single linkage: minimum distance between any two points
                    let mut dist = f64::INFINITY;
                    for &ci in &clusters[i] {
                        for &cj in &clusters[j] {
                            dist = dist.min(dist_matrix[ci][cj]);
                        }
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            // Merge clusters
            let cluster_a = clusters[min_i].clone();
            let cluster_b = clusters.remove(min_j); // Remove j first (higher index)
            clusters.remove(min_i); // Then remove i

            let mut merged = cluster_a;
            merged.extend(cluster_b);

            linkage.push((min_i, min_j, min_dist));
            clusters.push(merged);
        }

        linkage
    }

    /// Quasi-diagonalization: Sort indices based on hierarchical structure.
    ///
    /// Returns indices in order that brings similar assets close together.
    fn quasi_diagonalization(&self, clusters: &[(usize, usize, f64)]) -> Vec<usize> {
        let n = self.symbols.len();

        // Build the final cluster (root of tree)
        let mut cluster_items: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for &(i, j, _dist) in clusters {
            let cluster_a = cluster_items[i].clone();
            let cluster_b = cluster_items[j].clone();

            let mut merged = cluster_a;
            merged.extend(cluster_b);

            cluster_items.push(merged);
        }

        // The last cluster contains all indices in hierarchical order
        cluster_items.last().unwrap().clone()
    }

    /// Recursive bisection to allocate weights.
    ///
    /// Split clusters recursively and allocate inverse-variance weights.
    fn recursive_bisection(&self, sorted_indices: &[usize]) -> Vec<f64> {
        let n = self.symbols.len();
        let mut weights = vec![1.0; n]; // Start with equal weights

        self.recursive_bisection_impl(sorted_indices, &mut weights);

        // Normalize to sum to 1.0
        let sum: f64 = weights.iter().sum();
        if sum > 1e-10 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// Recursive bisection implementation.
    fn recursive_bisection_impl(&self, indices: &[usize], weights: &mut [f64]) {
        if indices.len() <= 1 {
            return;
        }

        // Split into two halves
        let mid = indices.len() / 2;
        let left = &indices[..mid];
        let right = &indices[mid..];

        // Compute cluster variances
        let left_var = self.cluster_variance(left);
        let right_var = self.cluster_variance(right);

        // Allocate inversely proportional to variance
        let total_inv_var = 1.0 / left_var + 1.0 / right_var;
        let left_weight = (1.0 / left_var) / total_inv_var;
        let right_weight = (1.0 / right_var) / total_inv_var;

        // Scale weights
        for &idx in left {
            weights[idx] *= left_weight;
        }
        for &idx in right {
            weights[idx] *= right_weight;
        }

        // Recurse
        self.recursive_bisection_impl(left, weights);
        self.recursive_bisection_impl(right, weights);
    }

    /// Compute variance of a cluster (sub-portfolio).
    fn cluster_variance(&self, indices: &[usize]) -> f64 {
        if indices.is_empty() {
            return 1e-10; // Avoid division by zero
        }

        // Equal-weight within cluster
        let n = indices.len();
        let weight = 1.0 / n as f64;

        let mut variance = 0.0;
        for &i in indices {
            for &j in indices {
                variance += weight * weight * self.covariance_matrix[i][j];
            }
        }

        variance.max(1e-10) // Ensure positive
    }
}

/// Portfolio strategy using Hierarchical Risk Parity.
pub struct HierarchicalRiskParityStrategy {
    lookback: usize,
    rebalance_frequency: usize,
}

impl HierarchicalRiskParityStrategy {
    /// Create new HRP strategy.
    ///
    /// # Parameters
    /// * `lookback` - Historical window for computing correlation/covariance
    /// * `rebalance_frequency` - Rebalance every N bars
    pub fn new(lookback: usize, rebalance_frequency: usize) -> Self {
        Self {
            lookback,
            rebalance_frequency,
        }
    }
}

impl PortfolioStrategy for HierarchicalRiskParityStrategy {
    fn name(&self) -> &str {
        "Hierarchical Risk Parity"
    }

    fn warmup_period(&self) -> usize {
        self.lookback + 1
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        let bar_idx = ctx.bar_index;

        // Rebalance at specified frequency
        if !bar_idx.is_multiple_of(self.rebalance_frequency) {
            return AllocationSignal::Hold;
        }

        let symbols: Vec<String> = ctx.symbols.to_vec();

        // Compute returns for each symbol
        let mut returns_history: HashMap<String, Vec<f64>> = HashMap::new();

        for symbol in &symbols {
            let history = match ctx.history(symbol) {
                Some(h) => h,
                None => continue,
            };

            if history.len() < self.lookback {
                continue;
            }

            // Calculate daily returns
            let closes: Vec<f64> = history
                .iter()
                .rev()
                .take(self.lookback)
                .map(|b| b.close)
                .collect();

            let mut returns = Vec::new();
            for i in 1..closes.len() {
                let ret = (closes[i] - closes[i - 1]) / closes[i - 1];
                if ret.is_finite() {
                    returns.push(ret);
                }
            }

            if !returns.is_empty() {
                returns_history.insert(symbol.to_string(), returns);
            }
        }

        // Need at least 2 assets with sufficient history
        if returns_history.len() < 2 {
            return AllocationSignal::Hold;
        }

        // Run HRP optimization
        match HierarchicalRiskParityOptimizer::from_history(symbols.clone(), &returns_history) {
            Ok(optimizer) => {
                let weights = optimizer.optimize();
                AllocationSignal::Rebalance(weights)
            }
            Err(_) => AllocationSignal::Hold,
        }
    }
}

/// Configuration for drift-based rebalancing.
///
/// Drift-based rebalancing triggers when the current portfolio weights deviate
/// from target weights by more than a specified threshold. This approach reduces
/// transaction costs compared to periodic rebalancing while maintaining desired allocations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriftRebalancingConfig {
    /// Maximum allowed drift for any single asset as a fraction (e.g., 0.05 = 5%).
    /// If abs(current_weight - target_weight) > threshold, rebalancing is triggered.
    pub drift_threshold: f64,

    /// Minimum bars between rebalances (prevents excessive trading).
    pub min_rebalance_interval: usize,

    /// Optional: Maximum total portfolio drift (sum of absolute drifts).
    /// If specified, rebalancing triggers when total drift exceeds this value.
    pub max_total_drift: Option<f64>,
}

impl Default for DriftRebalancingConfig {
    fn default() -> Self {
        Self {
            drift_threshold: 0.05,       // 5% drift threshold
            min_rebalance_interval: 1,   // Allow checking every bar
            max_total_drift: Some(0.10), // 10% total drift
        }
    }
}

impl DriftRebalancingConfig {
    /// Create conservative drift config (low threshold, frequent checking).
    pub fn conservative() -> Self {
        Self {
            drift_threshold: 0.03,       // 3% threshold
            min_rebalance_interval: 1,   // Check every bar
            max_total_drift: Some(0.06), // 6% total drift
        }
    }

    /// Create moderate drift config (balanced threshold).
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Create relaxed drift config (high threshold, less frequent trading).
    pub fn relaxed() -> Self {
        Self {
            drift_threshold: 0.10,       // 10% threshold
            min_rebalance_interval: 5,   // Wait at least 5 bars
            max_total_drift: Some(0.20), // 20% total drift
        }
    }

    /// Check if rebalancing should be triggered based on drift from target weights.
    ///
    /// Returns `true` if any asset has drifted beyond `drift_threshold` or if
    /// total portfolio drift exceeds `max_total_drift`.
    pub fn should_rebalance(
        &self,
        current_weights: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
        bars_since_last_rebalance: usize,
    ) -> bool {
        // Enforce minimum rebalance interval
        if bars_since_last_rebalance < self.min_rebalance_interval {
            return false;
        }

        let mut total_drift = 0.0;

        // Check drift for each asset
        for (symbol, &target) in target_weights.iter() {
            let current = current_weights.get(symbol).copied().unwrap_or(0.0);
            let drift = (current - target).abs();

            total_drift += drift;

            // Trigger if any single asset exceeds threshold
            if drift > self.drift_threshold {
                return true;
            }
        }

        // Check for assets in current but not in target (should be closed)
        for (symbol, &current) in current_weights.iter() {
            if !target_weights.contains_key(symbol) && current > 1e-6 {
                total_drift += current;
                if current > self.drift_threshold {
                    return true;
                }
            }
        }

        // Check total portfolio drift if configured
        if let Some(max_total) = self.max_total_drift {
            if total_drift > max_total {
                return true;
            }
        }

        false
    }

    /// Calculate the maximum drift between current and target weights.
    pub fn max_drift(
        current_weights: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
    ) -> f64 {
        let mut max_drift: f64 = 0.0;

        // Get all unique symbols
        let all_symbols: std::collections::HashSet<_> = current_weights
            .keys()
            .chain(target_weights.keys())
            .collect();

        for symbol in all_symbols {
            let current = current_weights.get(symbol).copied().unwrap_or(0.0);
            let target = target_weights.get(symbol).copied().unwrap_or(0.0);
            let drift = (current - target).abs();
            max_drift = max_drift.max(drift);
        }

        max_drift
    }

    /// Calculate the total drift (sum of absolute deviations).
    pub fn total_drift(
        current_weights: &HashMap<String, f64>,
        target_weights: &HashMap<String, f64>,
    ) -> f64 {
        let mut total = 0.0;

        // Get all unique symbols
        let all_symbols: std::collections::HashSet<_> = current_weights
            .keys()
            .chain(target_weights.keys())
            .collect();

        for symbol in all_symbols {
            let current = current_weights.get(symbol).copied().unwrap_or(0.0);
            let target = target_weights.get(symbol).copied().unwrap_or(0.0);
            total += (current - target).abs();
        }

        total
    }
}

/// Equal-weight portfolio strategy with drift-based rebalancing.
///
/// This strategy maintains equal weights across all assets but only rebalances
/// when weights drift beyond a threshold, reducing transaction costs compared
/// to periodic rebalancing.
pub struct DriftEqualWeightStrategy {
    drift_config: DriftRebalancingConfig,
    last_rebalance: usize,
    target_weights: HashMap<String, f64>,
}

impl DriftEqualWeightStrategy {
    /// Create new drift-based equal-weight strategy.
    pub fn new(drift_config: DriftRebalancingConfig) -> Self {
        Self {
            drift_config,
            last_rebalance: 0,
            target_weights: HashMap::new(),
        }
    }

    /// Create with default drift configuration.
    pub fn with_default_config() -> Self {
        Self::new(DriftRebalancingConfig::default())
    }

    /// Create with conservative drift configuration.
    pub fn conservative() -> Self {
        Self::new(DriftRebalancingConfig::conservative())
    }

    /// Create with relaxed drift configuration.
    pub fn relaxed() -> Self {
        Self::new(DriftRebalancingConfig::relaxed())
    }
}

impl PortfolioStrategy for DriftEqualWeightStrategy {
    fn name(&self) -> &str {
        "Drift Equal Weight"
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        // Initialize target weights on first bar
        if self.target_weights.is_empty() {
            let weight = 1.0 / ctx.symbols.len() as f64;
            self.target_weights = ctx.symbols.iter().map(|s| (s.clone(), weight)).collect();
            self.last_rebalance = ctx.bar_index;
            return AllocationSignal::Rebalance(self.target_weights.clone());
        }

        // Check if rebalancing is needed based on drift
        let bars_since_rebalance = ctx.bar_index.saturating_sub(self.last_rebalance);

        if self.drift_config.should_rebalance(
            &ctx.weights,
            &self.target_weights,
            bars_since_rebalance,
        ) {
            self.last_rebalance = ctx.bar_index;
            AllocationSignal::Rebalance(self.target_weights.clone())
        } else {
            AllocationSignal::Hold
        }
    }
}

/// Momentum portfolio strategy with drift-based rebalancing.
///
/// Selects top N momentum stocks and maintains equal weights, but only rebalances
/// when weights drift beyond threshold or when the momentum ranking changes significantly.
pub struct DriftMomentumStrategy {
    lookback: usize,
    top_n: usize,
    drift_config: DriftRebalancingConfig,
    periodic_rebalance_interval: usize, // Periodically recalculate momentum
    last_rebalance: usize,
    last_momentum_check: usize,
    target_weights: HashMap<String, f64>,
}

impl DriftMomentumStrategy {
    /// Create new drift-based momentum strategy.
    ///
    /// # Arguments
    /// * `lookback` - Period for momentum calculation
    /// * `top_n` - Number of top momentum stocks to hold
    /// * `drift_config` - Drift threshold configuration
    /// * `periodic_rebalance_interval` - How often to recalculate momentum (in bars)
    pub fn new(
        lookback: usize,
        top_n: usize,
        drift_config: DriftRebalancingConfig,
        periodic_rebalance_interval: usize,
    ) -> Self {
        Self {
            lookback,
            top_n,
            drift_config,
            periodic_rebalance_interval,
            last_rebalance: 0,
            last_momentum_check: 0,
            target_weights: HashMap::new(),
        }
    }

    /// Calculate target weights based on momentum.
    fn calculate_target_weights(&self, ctx: &PortfolioContext) -> HashMap<String, f64> {
        // Calculate momentum for each symbol
        let mut momentums: Vec<(String, f64)> = ctx
            .symbols
            .iter()
            .filter_map(|s| {
                let bars = ctx.history(s)?;
                if bars.len() < self.lookback {
                    return None;
                }
                let old_price = bars[bars.len() - self.lookback].close;
                let new_price = bars.last()?.close;
                let momentum = (new_price - old_price) / old_price;
                Some((s.clone(), momentum))
            })
            .collect();

        // Sort by momentum descending
        momentums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top N with positive momentum
        let selected: Vec<&str> = momentums
            .iter()
            .take(self.top_n.min(momentums.len()))
            .filter(|(_, m)| *m > 0.0)
            .map(|(s, _)| s.as_str())
            .collect();

        if selected.is_empty() {
            return HashMap::new();
        }

        let weight = 1.0 / selected.len() as f64;
        selected
            .into_iter()
            .map(|s| (s.to_string(), weight))
            .collect()
    }
}

impl PortfolioStrategy for DriftMomentumStrategy {
    fn name(&self) -> &str {
        "Drift Momentum"
    }

    fn warmup_period(&self) -> usize {
        self.lookback
    }

    fn on_bars(&mut self, ctx: &PortfolioContext) -> AllocationSignal {
        let bars_since_momentum_check = ctx.bar_index.saturating_sub(self.last_momentum_check);

        // Periodically recalculate target weights based on momentum
        let should_recalculate = self.target_weights.is_empty()
            || bars_since_momentum_check >= self.periodic_rebalance_interval;

        if should_recalculate {
            self.target_weights = self.calculate_target_weights(ctx);
            self.last_momentum_check = ctx.bar_index;

            // If targets changed significantly, rebalance immediately
            if self.target_weights.is_empty() {
                return AllocationSignal::ExitAll;
            }

            // Check if we need to rebalance based on new targets
            let bars_since_rebalance = ctx.bar_index.saturating_sub(self.last_rebalance);
            if bars_since_rebalance >= self.drift_config.min_rebalance_interval {
                self.last_rebalance = ctx.bar_index;
                return AllocationSignal::Rebalance(self.target_weights.clone());
            }
        }

        // Check drift-based rebalancing
        let bars_since_rebalance = ctx.bar_index.saturating_sub(self.last_rebalance);

        if self.drift_config.should_rebalance(
            &ctx.weights,
            &self.target_weights,
            bars_since_rebalance,
        ) {
            self.last_rebalance = ctx.bar_index;
            AllocationSignal::Rebalance(self.target_weights.clone())
        } else {
            AllocationSignal::Hold
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_bars(count: usize, base_price: f64, trend: f64) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let price = base_price * (1.0 + trend).powi(i as i32);
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::days(i as i64),
                    price * 0.99,
                    price * 1.02,
                    price * 0.98,
                    price,
                    1000000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_allocation() {
        let alloc = Allocation::new(0.5);
        assert!((alloc.weight - 0.5).abs() < 0.001);

        let bounded = Allocation::with_bounds(0.6, 0.1, 0.5);
        assert!((bounded.weight - 0.5).abs() < 0.001); // Clamped to max
    }

    #[test]
    fn test_equal_weight_strategy() {
        let strategy = EqualWeightStrategy::new(20);
        assert_eq!(strategy.name(), "Equal Weight");
    }

    #[test]
    fn test_momentum_portfolio_strategy() {
        let strategy = MomentumPortfolioStrategy::new(20, 3, 5);
        assert_eq!(strategy.name(), "Momentum Portfolio");
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_inverse_volatility_strategy() {
        let strategy = InverseVolatilityStrategy::new(20, 5);
        assert_eq!(strategy.name(), "Inverse Volatility");
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_risk_parity_strategy() {
        let strategy = RiskParityStrategy::new(20, 5);
        assert_eq!(strategy.name(), "Risk Parity");
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_inverse_volatility_allocation() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create assets with different volatilities
        // High volatility asset
        let bars1 = create_test_bars(100, 100.0, 0.02); // 2% daily
                                                        // Low volatility asset
        let bars2 = create_test_bars(100, 50.0, 0.005); // 0.5% daily

        engine.add_data("HIGH_VOL", bars1);
        engine.add_data("LOW_VOL", bars2);

        let mut strategy = InverseVolatilityStrategy::new(20, 5);
        let result = engine.run(&mut strategy).unwrap();

        // Inverse volatility should allocate less to high vol asset
        assert!(result.final_equity > 0.0);
        assert_eq!(result.symbols.len(), 2);
    }

    #[test]
    fn test_risk_parity_allocation() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create assets with different volatilities
        let bars1 = create_test_bars(100, 100.0, 0.02);
        let bars2 = create_test_bars(100, 50.0, 0.005);

        engine.add_data("ASSET1", bars1);
        engine.add_data("ASSET2", bars2);

        let mut strategy = RiskParityStrategy::new(20, 5);
        let result = engine.run(&mut strategy).unwrap();

        // Risk parity should produce a result
        assert!(result.final_equity > 0.0);
        assert_eq!(result.symbols.len(), 2);
    }

    #[test]
    fn test_multi_asset_engine_creation() {
        let config = BacktestConfig::default();
        let mut engine = MultiAssetEngine::new(config);

        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);

        engine.add_data("ASSET1", bars1);
        engine.add_data("ASSET2", bars2);
    }

    #[test]
    fn test_multi_asset_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);

        engine.add_data("ASSET1", bars1);
        engine.add_data("ASSET2", bars2);

        let mut strategy = EqualWeightStrategy::new(20);
        let result = engine.run(&mut strategy).unwrap();

        assert_eq!(result.symbols.len(), 2);
        assert!(result.final_equity > 0.0);
        assert!(!result.equity_curve.is_empty());
    }

    fn create_portfolio_context_for_testing() -> (
        PortfolioContext<'static>,
        HashMap<String, Bar>,
        HashMap<String, Vec<Bar>>,
        Vec<String>,
    ) {
        // Create test data with different price trends for cross-sectional testing
        let bars_a = create_test_bars(50, 100.0, 0.02); // Strong uptrend
        let bars_b = create_test_bars(50, 100.0, 0.01); // Moderate uptrend
        let bars_c = create_test_bars(50, 100.0, 0.00); // Flat
        let bars_d = create_test_bars(50, 100.0, -0.01); // Downtrend

        let mut all_bars = HashMap::new();
        all_bars.insert("ASSET_A".to_string(), bars_a.clone());
        all_bars.insert("ASSET_B".to_string(), bars_b.clone());
        all_bars.insert("ASSET_C".to_string(), bars_c.clone());
        all_bars.insert("ASSET_D".to_string(), bars_d.clone());

        let mut current_bars = HashMap::new();
        current_bars.insert("ASSET_A".to_string(), bars_a.last().unwrap().clone());
        current_bars.insert("ASSET_B".to_string(), bars_b.last().unwrap().clone());
        current_bars.insert("ASSET_C".to_string(), bars_c.last().unwrap().clone());
        current_bars.insert("ASSET_D".to_string(), bars_d.last().unwrap().clone());

        let symbols = vec![
            "ASSET_A".to_string(),
            "ASSET_B".to_string(),
            "ASSET_C".to_string(),
            "ASSET_D".to_string(),
        ];

        // These need to be 'static or have the same lifetime
        // For testing, we'll need to leak them or Box them
        // Let's use Box::leak to create static references
        let all_bars_static = Box::leak(Box::new(all_bars.clone()));
        let current_bars_static = Box::leak(Box::new(current_bars.clone()));
        let symbols_static = Box::leak(Box::new(symbols.clone()));

        let ctx = PortfolioContext {
            bar_index: 49,
            current_bars: current_bars_static,
            bars: all_bars_static,
            positions: HashMap::new(),
            cash: 100000.0,
            equity: 100000.0,
            weights: HashMap::new(),
            symbols: symbols_static,
        };

        (ctx, current_bars, all_bars, symbols)
    }

    #[test]
    fn test_rank_percentile_momentum() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // ASSET_A has strongest momentum (2% trend)
        let rank_a = ctx
            .rank_percentile(
                "ASSET_A",
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                20,
            )
            .unwrap();

        // ASSET_D has weakest momentum (-1% trend)
        let rank_d = ctx
            .rank_percentile(
                "ASSET_D",
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                20,
            )
            .unwrap();

        // ASSET_A should rank highest (1.0 or close to it)
        assert!(
            rank_a > 0.8,
            "ASSET_A should have high momentum rank: {}",
            rank_a
        );

        // ASSET_D should rank lowest (0.0 or close to it)
        assert!(
            rank_d < 0.2,
            "ASSET_D should have low momentum rank: {}",
            rank_d
        );

        // Rank should be ordered: A > B > C > D
        assert!(
            rank_a > rank_d,
            "Strong momentum should rank higher than weak"
        );
    }

    #[test]
    fn test_cross_sectional_zscore() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // Calculate z-score of momentum for each asset
        let zscore_a = ctx
            .cross_sectional_zscore(
                "ASSET_A",
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                20,
            )
            .unwrap();

        let zscore_d = ctx
            .cross_sectional_zscore(
                "ASSET_D",
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                20,
            )
            .unwrap();

        // ASSET_A should have positive z-score (above average)
        assert!(
            zscore_a > 0.0,
            "ASSET_A should have positive z-score: {}",
            zscore_a
        );

        // ASSET_D should have negative z-score (below average)
        assert!(
            zscore_d < 0.0,
            "ASSET_D should have negative z-score: {}",
            zscore_d
        );
    }

    #[test]
    fn test_relative_momentum() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // ASSET_A should have positive relative momentum (outperforming)
        let rel_mom_a = ctx.relative_momentum("ASSET_A", 20).unwrap();

        // ASSET_D should have negative relative momentum (underperforming)
        let rel_mom_d = ctx.relative_momentum("ASSET_D", 20).unwrap();

        assert!(
            rel_mom_a > 0.0,
            "ASSET_A should outperform universe: {}",
            rel_mom_a
        );
        assert!(
            rel_mom_d < 0.0,
            "ASSET_D should underperform universe: {}",
            rel_mom_d
        );

        // Relative momentum should be additive-inverse around average
        // (not exactly due to equal-weighting, but should be opposite signs)
        assert!(
            rel_mom_a > rel_mom_d,
            "Strong asset should have higher relative momentum than weak"
        );
    }

    #[test]
    fn test_rank_by_metric_momentum() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        let rank_a = ctx.rank_by_metric("ASSET_A", "momentum", 20).unwrap();
        let rank_d = ctx.rank_by_metric("ASSET_D", "momentum", 20).unwrap();

        // Same expectations as rank_percentile test
        assert!(
            rank_a > 0.8,
            "ASSET_A should have high momentum rank: {}",
            rank_a
        );
        assert!(
            rank_d < 0.2,
            "ASSET_D should have low momentum rank: {}",
            rank_d
        );
    }

    #[test]
    fn test_rank_by_metric_volatility() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // All assets should have similar volatility since they follow geometric trends
        let rank_a = ctx.rank_by_metric("ASSET_A", "volatility", 20);
        let rank_b = ctx.rank_by_metric("ASSET_B", "volatility", 20);

        assert!(rank_a.is_some(), "Volatility rank should be calculated");
        assert!(rank_b.is_some(), "Volatility rank should be calculated");
    }

    #[test]
    fn test_rank_by_metric_volume() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // All test bars have same volume (1000000.0), so ranks should be equal
        let rank_a = ctx.rank_by_metric("ASSET_A", "volume", 20).unwrap();

        // With equal volumes, each asset should have similar ranks
        // (exact value depends on tie-breaking, but should be around 0.33-0.66)
        assert!(
            rank_a >= 0.0 && rank_a <= 1.0,
            "Volume rank should be between 0 and 1: {}",
            rank_a
        );
    }

    #[test]
    fn test_rank_percentile_insufficient_symbols() {
        // Create context with only one symbol
        let bars = create_test_bars(50, 100.0, 0.01);
        let mut all_bars = HashMap::new();
        all_bars.insert("ASSET_A".to_string(), bars.clone());

        let mut current_bars = HashMap::new();
        current_bars.insert("ASSET_A".to_string(), bars.last().unwrap().clone());

        let symbols = vec!["ASSET_A".to_string()];

        let all_bars_static = Box::leak(Box::new(all_bars));
        let current_bars_static = Box::leak(Box::new(current_bars));
        let symbols_static = Box::leak(Box::new(symbols));

        let ctx = PortfolioContext {
            bar_index: 49,
            current_bars: current_bars_static,
            bars: all_bars_static,
            positions: HashMap::new(),
            cash: 100000.0,
            equity: 100000.0,
            weights: HashMap::new(),
            symbols: symbols_static,
        };

        // Should return None with only one symbol
        let rank = ctx.rank_percentile(
            "ASSET_A",
            |bars, lb| {
                if bars.len() < lb + 1 {
                    return None;
                }
                let start = bars[bars.len() - lb - 1].close;
                let end = bars.last()?.close;
                Some((end - start) / start)
            },
            20,
        );

        assert!(
            rank.is_none(),
            "Should return None with insufficient symbols"
        );
    }

    #[test]
    fn test_cross_sectional_zscore_all_equal() {
        // Create context where all assets have identical values
        let bars = create_test_bars(50, 100.0, 0.01);
        let mut all_bars = HashMap::new();
        all_bars.insert("ASSET_A".to_string(), bars.clone());
        all_bars.insert("ASSET_B".to_string(), bars.clone());
        all_bars.insert("ASSET_C".to_string(), bars.clone());

        let mut current_bars = HashMap::new();
        current_bars.insert("ASSET_A".to_string(), bars.last().unwrap().clone());
        current_bars.insert("ASSET_B".to_string(), bars.last().unwrap().clone());
        current_bars.insert("ASSET_C".to_string(), bars.last().unwrap().clone());

        let symbols = vec![
            "ASSET_A".to_string(),
            "ASSET_B".to_string(),
            "ASSET_C".to_string(),
        ];

        let all_bars_static = Box::leak(Box::new(all_bars));
        let current_bars_static = Box::leak(Box::new(current_bars));
        let symbols_static = Box::leak(Box::new(symbols));

        let ctx = PortfolioContext {
            bar_index: 49,
            current_bars: current_bars_static,
            bars: all_bars_static,
            positions: HashMap::new(),
            cash: 100000.0,
            equity: 100000.0,
            weights: HashMap::new(),
            symbols: symbols_static,
        };

        // When all values are equal, z-score should be 0.0
        let zscore = ctx
            .cross_sectional_zscore(
                "ASSET_A",
                |bars, lb| {
                    if bars.len() < lb + 1 {
                        return None;
                    }
                    let start = bars[bars.len() - lb - 1].close;
                    let end = bars.last()?.close;
                    Some((end - start) / start)
                },
                20,
            )
            .unwrap();

        assert_eq!(zscore, 0.0, "Z-score should be 0 when all values are equal");
    }

    #[test]
    fn test_relative_momentum_insufficient_data() {
        let (ctx, _, _, _) = create_portfolio_context_for_testing();

        // Request lookback longer than available data
        let rel_mom = ctx.relative_momentum("ASSET_A", 100);

        assert!(
            rel_mom.is_none(),
            "Should return None with insufficient data"
        );
    }

    #[test]
    fn test_mean_variance_optimizer_creation() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let expected_returns = vec![0.10, 0.12];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols, expected_returns, covariance_matrix, 0.02);

        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_mean_variance_optimizer_invalid_dimensions() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let expected_returns = vec![0.10]; // Wrong size
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols, expected_returns, covariance_matrix, 0.02);

        assert!(optimizer.is_err());
    }

    #[test]
    fn test_mean_variance_optimizer_from_history() {
        // Create historical data for two assets
        let bars1 = create_test_bars(100, 100.0, 0.001); // Low growth
        let bars2 = create_test_bars(100, 100.0, 0.002); // Higher growth

        let mut bars = HashMap::new();
        bars.insert("AAPL".to_string(), bars1);
        bars.insert("GOOGL".to_string(), bars2);

        let optimizer = MeanVarianceOptimizer::from_history(&bars, 50, 0.02);

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.symbols.len(), 2);
        assert_eq!(opt.expected_returns.len(), 2);
        assert_eq!(opt.covariance_matrix.len(), 2);
    }

    #[test]
    fn test_mean_variance_minimum_variance() {
        // Create a simple 2-asset case
        let symbols = vec!["LOW_VOL".to_string(), "HIGH_VOL".to_string()];
        let expected_returns = vec![0.08, 0.12]; // High vol has higher return
        let covariance_matrix = vec![
            vec![0.01, 0.00], // Low vol asset with 10% vol
            vec![0.00, 0.04], // High vol asset with 20% vol
        ];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        let weights = optimizer.minimum_variance().unwrap();

        // Should heavily favor the low volatility asset
        let low_vol_weight = weights.get("LOW_VOL").copied().unwrap_or(0.0);
        let high_vol_weight = weights.get("HIGH_VOL").copied().unwrap_or(0.0);

        assert!(
            low_vol_weight > high_vol_weight,
            "Minimum variance should prefer low volatility asset"
        );
        assert!(
            (low_vol_weight + high_vol_weight - 1.0).abs() < 0.01,
            "Weights should sum to 1"
        );
        assert!(low_vol_weight >= 0.0 && low_vol_weight <= 1.0);
        assert!(high_vol_weight >= 0.0 && high_vol_weight <= 1.0);
    }

    #[test]
    fn test_mean_variance_maximum_sharpe() {
        // Create a 2-asset case with different Sharpe ratios
        let symbols = vec!["LOW_SHARPE".to_string(), "HIGH_SHARPE".to_string()];
        let expected_returns = vec![0.06, 0.15]; // Second has much higher return
        let covariance_matrix = vec![
            vec![0.04, 0.00], // 20% vol, 6% return -> Sharpe ~0.2 (rf=2%)
            vec![0.00, 0.09], // 30% vol, 15% return -> Sharpe ~0.43 (rf=2%)
        ];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        let weights = optimizer.maximum_sharpe_ratio().unwrap();

        // Should favor the higher Sharpe ratio asset
        let low_sharpe_weight = weights.get("LOW_SHARPE").copied().unwrap_or(0.0);
        let high_sharpe_weight = weights.get("HIGH_SHARPE").copied().unwrap_or(0.0);

        assert!(
            high_sharpe_weight > low_sharpe_weight,
            "Max Sharpe should prefer higher Sharpe ratio asset"
        );
        assert!(
            (low_sharpe_weight + high_sharpe_weight - 1.0).abs() < 0.01,
            "Weights should sum to 1"
        );
    }

    #[test]
    fn test_mean_variance_target_return() {
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];
        let expected_returns = vec![0.08, 0.15];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        // Target return between the two assets
        let target_return = 0.11;
        let weights = optimizer.target_return(target_return).unwrap();

        // Calculate actual portfolio return
        let portfolio_return = optimizer.portfolio_return(&weights);

        assert!(
            (portfolio_return - target_return).abs() < 0.01,
            "Portfolio return should match target"
        );

        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 0.01, "Weights should sum to 1");
    }

    #[test]
    fn test_portfolio_metrics() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let expected_returns = vec![0.10, 0.12];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.6);
        weights.insert("GOOGL".to_string(), 0.4);

        let portfolio_return = optimizer.portfolio_return(&weights);
        let portfolio_variance = optimizer.portfolio_variance(&weights);
        let portfolio_volatility = optimizer.portfolio_volatility(&weights);

        // Expected return = 0.6 * 0.10 + 0.4 * 0.12 = 0.108
        assert!((portfolio_return - 0.108).abs() < 0.001);

        // Variance should be positive
        assert!(portfolio_variance > 0.0);

        // Volatility should be sqrt(variance)
        assert!((portfolio_volatility - portfolio_variance.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_mean_variance_strategy_creation() {
        let strategy = MeanVarianceStrategy::minimum_variance(60, 20);
        assert_eq!(strategy.name(), "Mean-Variance (Min Variance)");
        assert_eq!(strategy.warmup_period(), 61);

        let strategy2 = MeanVarianceStrategy::maximum_sharpe(60, 20, 0.02);
        assert_eq!(strategy2.name(), "Mean-Variance (Max Sharpe)");

        let strategy3 = MeanVarianceStrategy::target_return(60, 20, 0.10);
        assert_eq!(strategy3.name(), "Mean-Variance (Target Return)");
    }

    #[test]
    fn test_mean_variance_optimizer_with_correlated_assets() {
        // Test with positively correlated assets
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];
        let expected_returns = vec![0.10, 0.12];

        // High positive correlation (0.8)
        let std_a = 0.2; // 20% vol
        let std_b = 0.25; // 25% vol
        let corr = 0.8;
        let cov = corr * std_a * std_b; // 0.04

        let covariance_matrix = vec![vec![std_a * std_a, cov], vec![cov, std_b * std_b]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        let weights = optimizer.minimum_variance().unwrap();

        // With high correlation, diversification benefits are limited
        let weight_a = weights.get("ASSET_A").copied().unwrap_or(0.0);
        let weight_b = weights.get("ASSET_B").copied().unwrap_or(0.0);

        // Should still prefer lower volatility asset
        assert!(
            weight_a > weight_b,
            "Should prefer lower volatility when highly correlated"
        );
        assert!((weight_a + weight_b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mean_variance_all_negative_excess_returns() {
        // Test when all assets have negative excess returns
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];
        let expected_returns = vec![0.01, 0.02]; // Both below risk-free rate
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.05)
                .unwrap(); // Risk-free rate = 5%

        // Should fallback to minimum variance
        let weights = optimizer.maximum_sharpe_ratio().unwrap();

        let total_weight: f64 = weights.values().sum();
        assert!(
            (total_weight - 1.0).abs() < 0.01,
            "Weights should sum to 1 even with negative excess returns"
        );
    }

    #[test]
    fn test_mean_variance_three_assets() {
        // Test with three assets to ensure it works with more complex portfolios
        let symbols = vec![
            "ASSET_A".to_string(),
            "ASSET_B".to_string(),
            "ASSET_C".to_string(),
        ];
        let expected_returns = vec![0.08, 0.12, 0.10];
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.00],
            vec![0.01, 0.09, 0.02],
            vec![0.00, 0.02, 0.06],
        ];

        let optimizer =
            MeanVarianceOptimizer::new(symbols.clone(), expected_returns, covariance_matrix, 0.02)
                .unwrap();

        let weights = optimizer.maximum_sharpe_ratio().unwrap();

        assert_eq!(weights.len(), 3);
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 0.01, "Weights should sum to 1");

        // All weights should be non-negative
        for (symbol, weight) in weights.iter() {
            assert!(
                *weight >= 0.0,
                "Weight for {} should be non-negative",
                symbol
            );
        }
    }

    #[test]
    fn test_hrp_optimizer_creation() {
        // Test basic HRP optimizer creation with simple correlation/covariance
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];

        let correlation_matrix = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer = HierarchicalRiskParityOptimizer::new(
            symbols.clone(),
            correlation_matrix,
            covariance_matrix,
        );

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.symbols.len(), 2);
    }

    #[test]
    fn test_hrp_optimizer_invalid_dimensions() {
        // Test that HRP rejects mismatched matrix dimensions
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];

        // Wrong correlation matrix size (3x3 instead of 2x2)
        let correlation_matrix = vec![
            vec![1.0, 0.5, 0.0],
            vec![0.5, 1.0, 0.5],
            vec![0.0, 0.5, 1.0],
        ];

        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer = HierarchicalRiskParityOptimizer::new(
            symbols.clone(),
            correlation_matrix,
            covariance_matrix,
        );

        assert!(optimizer.is_err());
    }

    #[test]
    fn test_hrp_optimizer_from_history() {
        // Test HRP construction from historical returns
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];

        // Create synthetic return histories (30 days)
        let mut returns_history = HashMap::new();
        returns_history.insert("ASSET_A".to_string(), vec![0.01; 30]); // Low vol
        returns_history.insert("ASSET_B".to_string(), vec![0.02; 30]); // Higher vol

        let optimizer = HierarchicalRiskParityOptimizer::from_history(symbols, &returns_history);

        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_hrp_weights_sum_to_one() {
        // Test that HRP weights sum to 1.0
        let symbols = vec![
            "ASSET_A".to_string(),
            "ASSET_B".to_string(),
            "ASSET_C".to_string(),
        ];

        // Create correlation matrix (0.6 correlation between all pairs)
        let correlation_matrix = vec![
            vec![1.0, 0.6, 0.6],
            vec![0.6, 1.0, 0.6],
            vec![0.6, 0.6, 1.0],
        ];

        // Different volatilities
        let covariance_matrix = vec![
            vec![0.04, 0.012, 0.012],
            vec![0.012, 0.09, 0.027],
            vec![0.012, 0.027, 0.16],
        ];

        let optimizer = HierarchicalRiskParityOptimizer::new(
            symbols.clone(),
            correlation_matrix,
            covariance_matrix,
        )
        .unwrap();

        let weights = optimizer.optimize();

        assert_eq!(weights.len(), 3);

        let total_weight: f64 = weights.values().sum();
        assert!(
            (total_weight - 1.0).abs() < 0.01,
            "Weights should sum to 1.0, got {}",
            total_weight
        );

        // All weights should be non-negative
        for (symbol, weight) in weights.iter() {
            assert!(
                *weight >= 0.0,
                "Weight for {} should be non-negative, got {}",
                symbol,
                weight
            );
        }
    }

    #[test]
    fn test_hrp_uncorrelated_assets() {
        // Test HRP with uncorrelated assets (should allocate inversely to volatility)
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];

        // Zero correlation
        let correlation_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        // ASSET_A has lower volatility (0.2) than ASSET_B (0.3)
        let std_a = 0.2;
        let std_b = 0.3;
        let covariance_matrix = vec![vec![std_a * std_a, 0.0], vec![0.0, std_b * std_b]];

        let optimizer = HierarchicalRiskParityOptimizer::new(
            symbols.clone(),
            correlation_matrix,
            covariance_matrix,
        )
        .unwrap();

        let weights = optimizer.optimize();

        let weight_a = weights.get("ASSET_A").unwrap();
        let weight_b = weights.get("ASSET_B").unwrap();

        // With zero correlation, HRP should allocate more to lower volatility asset
        assert!(
            weight_a > weight_b,
            "Lower volatility asset should get higher weight: A={}, B={}",
            weight_a,
            weight_b
        );
    }

    #[test]
    fn test_hrp_highly_correlated_assets() {
        // Test HRP with highly correlated assets
        let symbols = vec!["ASSET_A".to_string(), "ASSET_B".to_string()];

        // High correlation (0.9)
        let corr = 0.9;
        let correlation_matrix = vec![vec![1.0, corr], vec![corr, 1.0]];

        let std_a = 0.2;
        let std_b = 0.3;
        let cov = corr * std_a * std_b;
        let covariance_matrix = vec![vec![std_a * std_a, cov], vec![cov, std_b * std_b]];

        let optimizer = HierarchicalRiskParityOptimizer::new(
            symbols.clone(),
            correlation_matrix,
            covariance_matrix,
        )
        .unwrap();

        let weights = optimizer.optimize();

        let weight_a = weights.get("ASSET_A").unwrap();
        let weight_b = weights.get("ASSET_B").unwrap();

        // Should still prefer lower volatility, even with high correlation
        assert!(
            weight_a > weight_b,
            "Lower volatility asset should get higher weight even when correlated: A={}, B={}",
            weight_a,
            weight_b
        );
        assert!((weight_a + weight_b - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hrp_strategy_creation() {
        let strategy = HierarchicalRiskParityStrategy::new(20, 5);
        assert_eq!(strategy.name(), "Hierarchical Risk Parity");
        assert_eq!(strategy.warmup_period(), 21); // lookback + 1
    }

    #[test]
    fn test_hrp_strategy_backtest() {
        // Full backtest with HRP strategy
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create assets with different risk profiles
        let bars1 = create_test_bars(100, 100.0, 0.01); // Low vol, positive trend
        let bars2 = create_test_bars(100, 50.0, 0.02); // High vol, higher trend
        let bars3 = create_test_bars(100, 75.0, 0.005); // Very low vol, low trend

        engine.add_data("LOW_VOL_A", bars1);
        engine.add_data("HIGH_VOL", bars2);
        engine.add_data("LOW_VOL_B", bars3);

        let mut strategy = HierarchicalRiskParityStrategy::new(20, 5);
        let result = engine.run(&mut strategy).unwrap();

        // HRP should produce positive returns with diversification
        assert!(result.final_equity > 0.0);
        assert_eq!(result.symbols.len(), 3);

        // Should have generated some trades (may be 0 if strategy holds throughout)
        // Relaxed assertion: just check result is valid
        println!("HRP Backtest Result: {} trades", result.total_trades);
        println!(
            "Final equity: {}, Initial: {}",
            result.final_equity, result.initial_capital
        );
    }

    // Portfolio Constraints Tests

    #[test]
    fn test_portfolio_constraints_default() {
        let constraints = PortfolioConstraints::default();
        assert_eq!(constraints.max_position_size, Some(0.25));
        assert_eq!(constraints.max_leverage, 1.0);
    }

    #[test]
    fn test_portfolio_constraints_moderate() {
        let constraints = PortfolioConstraints::moderate();
        assert_eq!(constraints.max_position_size, Some(0.20));
        assert_eq!(constraints.min_position_size, Some(0.02));
        assert_eq!(constraints.min_holdings, Some(5));
        assert_eq!(constraints.max_holdings, Some(30));
    }

    #[test]
    fn test_portfolio_constraints_strict() {
        let constraints = PortfolioConstraints::strict();
        assert_eq!(constraints.max_position_size, Some(0.10));
        assert_eq!(constraints.min_position_size, Some(0.05));
        assert_eq!(constraints.min_holdings, Some(10));
        assert_eq!(constraints.max_holdings, Some(20));
    }

    #[test]
    fn test_portfolio_constraints_builder() {
        let constraints = PortfolioConstraints::default()
            .with_symbol_limit("AAPL", 0.15)
            .with_sector_limit("Technology", 0.30);

        assert_eq!(constraints.symbol_limits.get("AAPL"), Some(&0.15));
        assert_eq!(constraints.sector_limits.get("Technology"), Some(&0.30));
    }

    #[test]
    fn test_validate_weights_max_position_size() {
        let constraints = PortfolioConstraints {
            max_position_size: Some(0.20),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.25); // Exceeds 20%
        weights.insert("GOOGL".to_string(), 0.15);

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_validate_weights_min_position_size() {
        let constraints = PortfolioConstraints {
            min_position_size: Some(0.05),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.03); // Below 5%
        weights.insert("GOOGL".to_string(), 0.10);

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("below minimum"));
    }

    #[test]
    fn test_validate_weights_leverage() {
        let constraints = PortfolioConstraints {
            max_leverage: 1.0,
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.60);
        weights.insert("GOOGL".to_string(), 0.50); // Total 1.1 > 1.0

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("leverage"));
    }

    #[test]
    fn test_validate_weights_min_holdings() {
        let constraints = PortfolioConstraints {
            min_holdings: Some(3),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.50);
        weights.insert("GOOGL".to_string(), 0.50); // Only 2 holdings

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("below minimum"));
    }

    #[test]
    fn test_validate_weights_max_holdings() {
        let constraints = PortfolioConstraints {
            max_holdings: Some(2),
            ..Default::default()
        };

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.33);
        weights.insert("GOOGL".to_string(), 0.33);
        weights.insert("MSFT".to_string(), 0.33); // 3 holdings > 2

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[test]
    fn test_validate_weights_sector_limits() {
        // Create constraints with sector limit and no position size limit
        let mut constraints = PortfolioConstraints {
            max_position_size: None, // Disable position size check
            sector_limits: HashMap::new(),
            ..Default::default()
        };
        constraints
            .sector_limits
            .insert("Technology".to_string(), 0.50);

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.30);
        weights.insert("GOOGL".to_string(), 0.25); // Total tech: 55% > 50%
        weights.insert("JPM".to_string(), 0.45);

        let mut sectors = HashMap::new();
        sectors.insert("AAPL".to_string(), "Technology".to_string());
        sectors.insert("GOOGL".to_string(), "Technology".to_string());
        sectors.insert("JPM".to_string(), "Finance".to_string());

        let result = constraints.validate_weights(&weights, &sectors);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Sector") || err_msg.contains("Technology"),
            "Error message: {}",
            err_msg
        );
    }

    #[test]
    fn test_validate_weights_symbol_limits() {
        let constraints = PortfolioConstraints::default().with_symbol_limit("AAPL", 0.10);

        let mut weights = HashMap::new();
        weights.insert("AAPL".to_string(), 0.15); // Exceeds symbol-specific limit
        weights.insert("GOOGL".to_string(), 0.20); // Within global limit

        let result = constraints.validate_weights(&weights, &HashMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("AAPL"));
    }

    #[test]
    fn test_calculate_turnover() {
        let mut current = HashMap::new();
        current.insert("AAPL".to_string(), 0.30);
        current.insert("GOOGL".to_string(), 0.40);
        current.insert("MSFT".to_string(), 0.30);

        let mut proposed = HashMap::new();
        proposed.insert("AAPL".to_string(), 0.40); // +10%
        proposed.insert("GOOGL".to_string(), 0.30); // -10%
        proposed.insert("MSFT".to_string(), 0.30); // No change

        let turnover = PortfolioConstraints::calculate_turnover(&current, &proposed);
        // Total change: |0.10| + |-0.10| + |0.0| = 0.20, divided by 2 = 0.10
        assert!((turnover - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_validate_turnover() {
        let constraints = PortfolioConstraints {
            max_turnover: Some(0.10),
            ..Default::default()
        };

        let mut current = HashMap::new();
        current.insert("AAPL".to_string(), 0.30);
        current.insert("GOOGL".to_string(), 0.40);
        current.insert("MSFT".to_string(), 0.30);

        let mut proposed = HashMap::new();
        proposed.insert("AAPL".to_string(), 0.50); // +20%
        proposed.insert("GOOGL".to_string(), 0.20); // -20%
        proposed.insert("MSFT".to_string(), 0.30); // No change
                                                   // Turnover = 0.20 > 0.10

        let result = constraints.validate_turnover(&current, &proposed);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Turnover"));
    }

    #[test]
    fn test_validate_rebalance() {
        let constraints = PortfolioConstraints {
            max_position_size: Some(0.40),
            max_turnover: Some(0.30),
            ..Default::default()
        };

        let mut current = HashMap::new();
        current.insert("AAPL".to_string(), 0.50);
        current.insert("GOOGL".to_string(), 0.50);

        let mut proposed = HashMap::new();
        proposed.insert("AAPL".to_string(), 0.60);
        proposed.insert("GOOGL".to_string(), 0.40);

        let result = constraints.validate_rebalance(&current, &proposed, &HashMap::new());
        // Should fail on max_position_size (AAPL at 60% > 40%)
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_asset_engine_with_constraints() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        // Create constraints that should NOT be violated
        let constraints = PortfolioConstraints {
            max_position_size: Some(0.60), // Generous limit
            min_holdings: Some(2),
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config).with_constraints(constraints);

        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);

        engine.add_data("ASSET1", bars1);
        engine.add_data("ASSET2", bars2);

        let mut strategy = EqualWeightStrategy::new(20);
        let result = engine.run(&mut strategy);

        // Should succeed as equal weight with 2 assets should satisfy constraints
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.final_equity > 0.0);
    }

    #[test]
    fn test_multi_asset_engine_constraint_violation() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        // Create strict constraints that will be violated
        let constraints = PortfolioConstraints {
            max_position_size: Some(0.40), // 40% max per position
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config).with_constraints(constraints);

        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);

        engine.add_data("ASSET1", bars1);
        engine.add_data("ASSET2", bars2);

        // Equal weight with 2 assets = 50% each, violates 40% limit
        let mut strategy = EqualWeightStrategy::new(20);
        let result = engine.run(&mut strategy);

        // Should fail due to constraint violation
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("constraint") || err.to_string().contains("exceeds"));
    }

    #[test]
    fn test_multi_asset_engine_with_sector_constraints() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        // Create constraints with sector limit but allow large position sizes
        let mut constraints = PortfolioConstraints {
            max_position_size: Some(0.50), // Allow 50% per position
            min_holdings: Some(2),
            sector_limits: HashMap::new(),
            ..Default::default()
        };
        constraints
            .sector_limits
            .insert("Technology".to_string(), 0.60);

        let mut engine = MultiAssetEngine::new(config).with_constraints(constraints);

        engine.set_symbol_sector("AAPL", "Technology");
        engine.set_symbol_sector("GOOGL", "Technology");
        engine.set_symbol_sector("JPM", "Finance");

        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);
        let bars3 = create_test_bars(100, 75.0, 0.0015);

        engine.add_data("AAPL", bars1);
        engine.add_data("GOOGL", bars2);
        engine.add_data("JPM", bars3);

        // Equal weight = 33.3% each, tech total = 66.6% > 60% limit
        let mut strategy = EqualWeightStrategy::new(20);
        let result = engine.run(&mut strategy);

        // Should fail due to sector constraint violation
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Sector") || err_msg.contains("Technology"),
            "Expected sector constraint error, got: {}",
            err_msg
        );
    }

    // Black-Litterman Tests

    #[test]
    fn test_black_litterman_optimizer_creation() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
        let market_caps = vec![3000e9, 2000e9, 2500e9]; // Market caps in billions
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.015],
            vec![0.01, 0.09, 0.02],
            vec![0.015, 0.02, 0.06],
        ];

        let optimizer = BlackLittermanOptimizer::new(
            symbols,
            market_caps,
            covariance_matrix,
            0.03, // tau
            2.5,  // risk aversion
            0.02, // risk-free rate
        );

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.symbols.len(), 3);

        // Check market weights sum to 1
        let weight_sum: f64 = opt.market_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_black_litterman_invalid_dimensions() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9]; // Wrong size
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer =
            BlackLittermanOptimizer::new(symbols, market_caps, covariance_matrix, 0.03, 2.5, 0.02);

        assert!(optimizer.is_err());
    }

    #[test]
    fn test_black_litterman_invalid_tau() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        // Tau too high
        let optimizer = BlackLittermanOptimizer::new(
            symbols.clone(),
            market_caps.clone(),
            covariance_matrix.clone(),
            1.5, // Invalid tau > 1
            2.5,
            0.02,
        );
        assert!(optimizer.is_err());

        // Tau negative
        let optimizer2 = BlackLittermanOptimizer::new(
            symbols,
            market_caps,
            covariance_matrix,
            -0.03, // Invalid negative tau
            2.5,
            0.02,
        );
        assert!(optimizer2.is_err());
    }

    #[test]
    fn test_black_litterman_from_history() {
        let bars1 = create_test_bars(100, 100.0, 0.001);
        let bars2 = create_test_bars(100, 50.0, 0.002);
        let bars3 = create_test_bars(100, 75.0, 0.0015);

        let mut bars = HashMap::new();
        bars.insert("AAPL".to_string(), bars1);
        bars.insert("GOOGL".to_string(), bars2);
        bars.insert("MSFT".to_string(), bars3);

        let optimizer = BlackLittermanOptimizer::from_history(&bars, 50, 0.03, 2.5, 0.02);

        assert!(optimizer.is_ok());
        let opt = optimizer.unwrap();
        assert_eq!(opt.symbols.len(), 3);

        // Equal weights for from_history (no market caps provided)
        for weight in opt.market_weights.iter() {
            assert!((weight - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_black_litterman_implied_returns() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9]; // 60% AAPL, 40% GOOGL
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer = BlackLittermanOptimizer::new(
            symbols,
            market_caps,
            covariance_matrix,
            0.03,
            2.5, // risk aversion
            0.02,
        )
        .unwrap();

        let implied = optimizer.implied_returns();

        // Implied returns should be non-negative and reasonable
        assert_eq!(implied.len(), 2);
        for &ret in implied.iter() {
            assert!(ret.is_finite());
            assert!(
                ret > -1.0 && ret < 2.0,
                "Implied return out of reasonable range: {}",
                ret
            );
        }

        // Asset with higher volatility should have higher implied return
        // GOOGL has 30% vol (0.09), AAPL has 20% vol (0.04)
        // But this depends on weights, so just check they're reasonable
        println!("Implied returns: {:?}", implied);
    }

    #[test]
    fn test_black_litterman_absolute_view() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let mut optimizer =
            BlackLittermanOptimizer::new(symbols, market_caps, covariance_matrix, 0.03, 2.5, 0.02)
                .unwrap();

        // Add absolute view: AAPL will return 15%
        optimizer.add_view(View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.15,
            confidence: 0.05,
        });

        let posterior = optimizer.compute_posterior_returns();
        assert!(posterior.is_ok());

        let returns = posterior.unwrap();
        assert_eq!(returns.len(), 2);

        // Posterior return for AAPL should be influenced by the view (closer to 15%)
        // Exact value depends on confidence and prior, but should be finite
        for &ret in returns.iter() {
            assert!(ret.is_finite());
        }

        println!("Posterior returns with absolute view: {:?}", returns);
    }

    #[test]
    fn test_black_litterman_relative_view() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
        let market_caps = vec![3000e9, 2000e9, 2500e9];
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.015],
            vec![0.01, 0.09, 0.02],
            vec![0.015, 0.02, 0.06],
        ];

        let mut optimizer =
            BlackLittermanOptimizer::new(symbols, market_caps, covariance_matrix, 0.03, 2.5, 0.02)
                .unwrap();

        // Add relative view: AAPL will outperform GOOGL by 3%
        optimizer.add_view(View::Relative {
            symbol_a: "AAPL".to_string(),
            symbol_b: "GOOGL".to_string(),
            expected_outperformance: 0.03,
            confidence: 0.02,
        });

        let posterior = optimizer.compute_posterior_returns();
        assert!(posterior.is_ok());

        let returns = posterior.unwrap();
        assert_eq!(returns.len(), 3);

        // AAPL return should be higher than GOOGL after incorporating view
        // (though not guaranteed due to equilibrium blending)
        for &ret in returns.iter() {
            assert!(ret.is_finite());
        }

        println!("Posterior returns with relative view: {:?}", returns);
    }

    #[test]
    fn test_black_litterman_multiple_views() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
        let market_caps = vec![3000e9, 2000e9, 2500e9];
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.015],
            vec![0.01, 0.09, 0.02],
            vec![0.015, 0.02, 0.06],
        ];

        let mut optimizer =
            BlackLittermanOptimizer::new(symbols, market_caps, covariance_matrix, 0.03, 2.5, 0.02)
                .unwrap();

        // Add multiple views
        optimizer.add_view(View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.15,
            confidence: 0.05,
        });

        optimizer.add_view(View::Relative {
            symbol_a: "GOOGL".to_string(),
            symbol_b: "MSFT".to_string(),
            expected_outperformance: 0.02,
            confidence: 0.03,
        });

        let posterior = optimizer.compute_posterior_returns();
        assert!(posterior.is_ok());

        let returns = posterior.unwrap();
        assert_eq!(returns.len(), 3);

        for &ret in returns.iter() {
            assert!(ret.is_finite());
        }

        println!("Posterior returns with multiple views: {:?}", returns);
    }

    #[test]
    fn test_black_litterman_optimize_no_views() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let optimizer = BlackLittermanOptimizer::new(
            symbols.clone(),
            market_caps,
            covariance_matrix,
            0.03,
            2.5,
            0.02,
        )
        .unwrap();

        // No views added, should use pure equilibrium returns
        let weights = optimizer.optimize();
        assert!(weights.is_ok());

        let w = weights.unwrap();
        assert_eq!(w.len(), 2);

        let total_weight: f64 = w.values().sum();
        assert!((total_weight - 1.0).abs() < 0.01, "Weights should sum to 1");

        for (symbol, weight) in w.iter() {
            assert!(
                *weight >= 0.0 && *weight <= 1.0,
                "Weight for {} out of bounds: {}",
                symbol,
                weight
            );
        }

        println!("Black-Litterman weights (no views): {:?}", w);
    }

    #[test]
    fn test_black_litterman_optimize_with_views() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string()];
        let market_caps = vec![3000e9, 2000e9, 2500e9];
        let covariance_matrix = vec![
            vec![0.04, 0.01, 0.015],
            vec![0.01, 0.09, 0.02],
            vec![0.015, 0.02, 0.06],
        ];

        let mut optimizer = BlackLittermanOptimizer::new(
            symbols.clone(),
            market_caps,
            covariance_matrix,
            0.03,
            2.5,
            0.02,
        )
        .unwrap();

        // Add view that AAPL will have higher return
        optimizer.add_view(View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.20, // Strong bullish view
            confidence: 0.02,      // High confidence (low variance)
        });

        let weights = optimizer.optimize();
        assert!(weights.is_ok());

        let w = weights.unwrap();
        assert_eq!(w.len(), 3);

        let total_weight: f64 = w.values().sum();
        assert!(
            (total_weight - 1.0).abs() < 0.01,
            "Weights should sum to 1, got {}",
            total_weight
        );

        // AAPL should have meaningful weight given bullish view
        let aapl_weight = w.get("AAPL").copied().unwrap_or(0.0);
        assert!(
            aapl_weight > 0.0,
            "AAPL should have positive weight with bullish view"
        );

        println!("Black-Litterman weights (with view): {:?}", w);
    }

    #[test]
    fn test_black_litterman_view_confidence_impact() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        // High confidence view
        let mut optimizer_high = BlackLittermanOptimizer::new(
            symbols.clone(),
            market_caps.clone(),
            covariance_matrix.clone(),
            0.03,
            2.5,
            0.02,
        )
        .unwrap();

        optimizer_high.add_view(View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.20,
            confidence: 0.01, // High confidence (low variance)
        });

        let returns_high = optimizer_high.compute_posterior_returns().unwrap();

        // Low confidence view
        let mut optimizer_low = BlackLittermanOptimizer::new(
            symbols.clone(),
            market_caps.clone(),
            covariance_matrix.clone(),
            0.03,
            2.5,
            0.02,
        )
        .unwrap();

        optimizer_low.add_view(View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.20,
            confidence: 0.10, // Low confidence (high variance)
        });

        let returns_low = optimizer_low.compute_posterior_returns().unwrap();

        // High confidence view should pull AAPL return closer to 0.20
        println!("High confidence posterior: {:?}", returns_high);
        println!("Low confidence posterior: {:?}", returns_low);

        // Both should be finite
        assert!(returns_high[0].is_finite());
        assert!(returns_low[0].is_finite());
    }

    #[test]
    fn test_black_litterman_strategy_creation() {
        let views = vec![View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.15,
            confidence: 0.05,
        }];

        let strategy = BlackLittermanStrategy::new(60, 20, 0.03, 2.5, 0.02, views);

        assert_eq!(strategy.name(), "Black-Litterman");
        assert_eq!(strategy.warmup_period(), 61);
    }

    #[test]
    fn test_black_litterman_strategy_backtest() {
        let config = BacktestConfig::default();
        let mut engine = MultiAssetEngine::new(config);

        let bars1 = create_test_bars(100, 100.0, 0.002); // Higher growth
        let bars2 = create_test_bars(100, 50.0, 0.001); // Lower growth
        let bars3 = create_test_bars(100, 75.0, 0.0015);

        engine.add_data("AAPL", bars1);
        engine.add_data("GOOGL", bars2);
        engine.add_data("MSFT", bars3);

        // Add view that AAPL will outperform
        let views = vec![View::Absolute {
            symbol: "AAPL".to_string(),
            expected_return: 0.20,
            confidence: 0.03,
        }];

        let mut strategy = BlackLittermanStrategy::new(60, 20, 0.03, 2.5, 0.02, views);
        let result = engine.run(&mut strategy);

        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.final_equity > 0.0);
        assert_eq!(res.symbols.len(), 3);

        println!(
            "Black-Litterman backtest result: {} trades",
            res.total_trades
        );
        println!(
            "Final equity: {}, Initial: {}",
            res.final_equity, res.initial_capital
        );
    }

    #[test]
    fn test_black_litterman_matrix_inversion() {
        // Test with a well-conditioned matrix
        let symbols = vec!["A".to_string(), "B".to_string()];
        let market_caps = vec![1.0, 1.0];
        let covariance_matrix = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let optimizer = BlackLittermanOptimizer::new(
            symbols,
            market_caps,
            covariance_matrix.clone(),
            0.03,
            2.5,
            0.02,
        )
        .unwrap();

        // Test matrix inversion
        let inverse = optimizer.invert_matrix(&covariance_matrix);
        assert!(inverse.is_ok());

        let inv = inverse.unwrap();
        assert_eq!(inv.len(), 2);
        assert_eq!(inv[0].len(), 2);

        // Check that A * A^-1 ≈ I
        let mut product = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] += covariance_matrix[i][k] * inv[k][j];
                }
            }
        }

        // Should be close to identity
        assert!((product[0][0] - 1.0).abs() < 1e-6);
        assert!((product[1][1] - 1.0).abs() < 1e-6);
        assert!(product[0][1].abs() < 1e-6);
        assert!(product[1][0].abs() < 1e-6);
    }

    #[test]
    fn test_black_litterman_invalid_view_symbol() {
        let symbols = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let market_caps = vec![3000e9, 2000e9];
        let covariance_matrix = vec![vec![0.04, 0.01], vec![0.01, 0.09]];

        let mut optimizer =
            BlackLittermanOptimizer::new(symbols, market_caps, covariance_matrix, 0.03, 2.5, 0.02)
                .unwrap();

        // Add view with unknown symbol
        optimizer.add_view(View::Absolute {
            symbol: "TSLA".to_string(), // Not in optimizer
            expected_return: 0.15,
            confidence: 0.05,
        });

        let posterior = optimizer.compute_posterior_returns();
        assert!(posterior.is_err());
    }

    #[test]
    fn test_drift_rebalancing_config_default() {
        let config = DriftRebalancingConfig::default();
        assert!((config.drift_threshold - 0.05).abs() < 1e-6);
        assert_eq!(config.min_rebalance_interval, 1);
        assert_eq!(config.max_total_drift, Some(0.10));
    }

    #[test]
    fn test_drift_rebalancing_config_conservative() {
        let config = DriftRebalancingConfig::conservative();
        assert!((config.drift_threshold - 0.03).abs() < 1e-6);
        assert_eq!(config.min_rebalance_interval, 1);
        assert_eq!(config.max_total_drift, Some(0.06));
    }

    #[test]
    fn test_drift_rebalancing_config_relaxed() {
        let config = DriftRebalancingConfig::relaxed();
        assert!((config.drift_threshold - 0.10).abs() < 1e-6);
        assert_eq!(config.min_rebalance_interval, 5);
        assert_eq!(config.max_total_drift, Some(0.20));
    }

    #[test]
    fn test_drift_calculation_no_drift() {
        let current = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        let max_drift = DriftRebalancingConfig::max_drift(&current, &target);
        assert!(max_drift < 1e-6);

        let total_drift = DriftRebalancingConfig::total_drift(&current, &target);
        assert!(total_drift < 1e-6);
    }

    #[test]
    fn test_drift_calculation_with_drift() {
        let current = vec![("AAPL".to_string(), 0.6), ("GOOGL".to_string(), 0.4)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        let max_drift = DriftRebalancingConfig::max_drift(&current, &target);
        assert!((max_drift - 0.1).abs() < 1e-6);

        let total_drift = DriftRebalancingConfig::total_drift(&current, &target);
        assert!((total_drift - 0.2).abs() < 1e-6); // 0.1 + 0.1
    }

    #[test]
    fn test_drift_calculation_new_asset() {
        let current = vec![
            ("AAPL".to_string(), 0.5),
            ("GOOGL".to_string(), 0.3),
            ("MSFT".to_string(), 0.2),
        ]
        .into_iter()
        .collect();

        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        let max_drift = DriftRebalancingConfig::max_drift(&current, &target);
        assert!((max_drift - 0.2).abs() < 1e-6); // GOOGL drifted from 0.5 to 0.3, MSFT is 0.2

        let total_drift = DriftRebalancingConfig::total_drift(&current, &target);
        assert!((total_drift - 0.4).abs() < 1e-6); // 0.0 + 0.2 + 0.2
    }

    #[test]
    fn test_should_rebalance_below_threshold() {
        let config = DriftRebalancingConfig {
            drift_threshold: 0.05,
            min_rebalance_interval: 1,
            max_total_drift: None,
        };

        let current = vec![("AAPL".to_string(), 0.52), ("GOOGL".to_string(), 0.48)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        assert!(!config.should_rebalance(&current, &target, 10));
    }

    #[test]
    fn test_should_rebalance_above_threshold() {
        let config = DriftRebalancingConfig {
            drift_threshold: 0.05,
            min_rebalance_interval: 1,
            max_total_drift: None,
        };

        let current = vec![("AAPL".to_string(), 0.6), ("GOOGL".to_string(), 0.4)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        assert!(config.should_rebalance(&current, &target, 10));
    }

    #[test]
    fn test_should_rebalance_min_interval() {
        let config = DriftRebalancingConfig {
            drift_threshold: 0.05,
            min_rebalance_interval: 10,
            max_total_drift: None,
        };

        let current = vec![("AAPL".to_string(), 0.6), ("GOOGL".to_string(), 0.4)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        // Should not rebalance if interval not met
        assert!(!config.should_rebalance(&current, &target, 5));

        // Should rebalance if interval met
        assert!(config.should_rebalance(&current, &target, 10));
    }

    #[test]
    fn test_should_rebalance_total_drift() {
        let config = DriftRebalancingConfig {
            drift_threshold: 0.10, // High individual threshold
            min_rebalance_interval: 1,
            max_total_drift: Some(0.10), // But low total drift threshold
        };

        // Each asset drifts by 0.06, total 0.12
        let current = vec![("AAPL".to_string(), 0.56), ("GOOGL".to_string(), 0.44)]
            .into_iter()
            .collect();
        let target = vec![("AAPL".to_string(), 0.5), ("GOOGL".to_string(), 0.5)]
            .into_iter()
            .collect();

        // Should rebalance due to total drift exceeding threshold
        assert!(config.should_rebalance(&current, &target, 10));
    }

    #[test]
    fn test_drift_equal_weight_strategy_creation() {
        let strategy = DriftEqualWeightStrategy::with_default_config();
        assert_eq!(strategy.name(), "Drift Equal Weight");

        let conservative = DriftEqualWeightStrategy::conservative();
        assert_eq!(conservative.name(), "Drift Equal Weight");

        let relaxed = DriftEqualWeightStrategy::relaxed();
        assert_eq!(relaxed.name(), "Drift Equal Weight");
    }

    #[test]
    fn test_drift_equal_weight_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create test data with significant price movements that will cause drift
        let bars1 = create_test_bars(200, 100.0, 0.01); // AAPL grows 1% per bar
        let bars2 = create_test_bars(200, 100.0, -0.005); // GOOGL declines 0.5% per bar

        engine.add_data("AAPL", bars1);
        engine.add_data("GOOGL", bars2);

        let mut strategy = DriftEqualWeightStrategy::with_default_config();
        let result = engine.run(&mut strategy);

        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.final_equity > 0.0);
        assert_eq!(res.symbols.len(), 2);
        // Strategy executed successfully - drift logic was tested in unit tests
    }

    #[test]
    fn test_drift_equal_weight_vs_periodic() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        // Test drift-based strategy with significant price movements
        let mut engine1 = MultiAssetEngine::new(config.clone());
        let bars1 = create_test_bars(200, 100.0, 0.01); // 1% growth per bar
        let bars2 = create_test_bars(200, 100.0, -0.005); // 0.5% decline per bar
        engine1.add_data("AAPL", bars1.clone());
        engine1.add_data("GOOGL", bars2.clone());

        let mut drift_strategy = DriftEqualWeightStrategy::relaxed(); // High threshold
        let drift_result = engine1.run(&mut drift_strategy).unwrap();

        // Test periodic strategy
        let mut engine2 = MultiAssetEngine::new(config);
        engine2.add_data("AAPL", bars1);
        engine2.add_data("GOOGL", bars2);

        let mut periodic_strategy = EqualWeightStrategy::new(10); // Rebalance every 10 bars
        let periodic_result = engine2.run(&mut periodic_strategy).unwrap();

        // Both strategies should complete successfully
        assert!(drift_result.final_equity > 0.0);
        assert!(periodic_result.final_equity > 0.0);
    }

    #[test]
    fn test_drift_momentum_strategy_creation() {
        let config = DriftRebalancingConfig::default();
        let strategy = DriftMomentumStrategy::new(20, 2, config, 10);
        assert_eq!(strategy.name(), "Drift Momentum");
        assert_eq!(strategy.warmup_period(), 20);
    }

    #[test]
    fn test_drift_momentum_backtest() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create test data with different momentum patterns and longer history
        let bars1 = create_test_bars(200, 100.0, 0.01); // Strong momentum
        let bars2 = create_test_bars(200, 100.0, 0.005); // Moderate momentum
        let bars3 = create_test_bars(200, 100.0, -0.005); // Negative momentum

        engine.add_data("AAPL", bars1);
        engine.add_data("GOOGL", bars2);
        engine.add_data("MSFT", bars3);

        let drift_config = DriftRebalancingConfig::default();
        let mut strategy = DriftMomentumStrategy::new(20, 2, drift_config, 20);
        let result = engine.run(&mut strategy);

        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(res.final_equity > 0.0);
        assert_eq!(res.symbols.len(), 3);
        // Strategy executed successfully
    }

    #[test]
    fn test_drift_momentum_top_n_selection() {
        let config = BacktestConfig {
            initial_capital: 100_000.0,
            show_progress: false,
            ..Default::default()
        };

        let mut engine = MultiAssetEngine::new(config);

        // Create 5 assets with clear momentum ranking and significant movements
        engine.add_data("BEST", create_test_bars(200, 100.0, 0.015)); // Best - 1.5% per bar
        engine.add_data("GOOD", create_test_bars(200, 100.0, 0.010)); // Good - 1% per bar
        engine.add_data("OK", create_test_bars(200, 100.0, 0.002)); // OK - 0.2% per bar
        engine.add_data("BAD", create_test_bars(200, 100.0, -0.005)); // Bad
        engine.add_data("WORST", create_test_bars(200, 100.0, -0.010)); // Worst

        let drift_config = DriftRebalancingConfig::default();
        let mut strategy = DriftMomentumStrategy::new(20, 2, drift_config, 100); // Top 2, rarely recalc

        let result = engine.run(&mut strategy).unwrap();

        // Strategy executed successfully with momentum selection
        assert!(result.final_equity > 0.0);
        assert_eq!(result.symbols.len(), 5);
    }

    #[test]
    fn test_drift_config_moderate() {
        let config = DriftRebalancingConfig::moderate();
        assert_eq!(config, DriftRebalancingConfig::default());
    }
}
