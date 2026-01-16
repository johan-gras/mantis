//! Risk management utilities for trading.
//!
//! This module provides stop-loss, take-profit, and position sizing functionality.

use crate::types::Side;
use serde::{Deserialize, Serialize};

/// Position sizing method for determining trade size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizingMethod {
    /// Fixed percentage of equity per trade (default: 10%).
    PercentOfEquity(f64),
    /// Fixed dollar amount per trade.
    FixedDollar(f64),
    /// Volatility-targeted sizing: scale position size inversely with asset volatility.
    /// Parameters: (target_volatility, lookback_period)
    VolatilityTargeted { target_vol: f64, lookback: usize },
    /// Signal-scaled sizing: use signal magnitude to scale position size.
    /// Parameters: base_size as fraction of equity
    SignalScaled { base_size: f64 },
    /// Risk-based sizing using ATR for stop-loss distance.
    /// Parameters: (risk_per_trade as percentage, stop_atr_multiplier, atr_period)
    RiskBased {
        risk_per_trade: f64,
        stop_atr: f64,
        atr_period: usize,
    },
}

impl Default for PositionSizingMethod {
    fn default() -> Self {
        PositionSizingMethod::PercentOfEquity(0.1) // 10% of equity
    }
}

impl PositionSizingMethod {
    /// Create percent of equity sizing (default method).
    pub fn percent(pct: f64) -> Self {
        PositionSizingMethod::PercentOfEquity(pct)
    }

    /// Create fixed dollar amount sizing.
    pub fn fixed_dollar(amount: f64) -> Self {
        PositionSizingMethod::FixedDollar(amount)
    }

    /// Create volatility-targeted sizing.
    pub fn volatility(target_vol: f64, lookback: usize) -> Self {
        PositionSizingMethod::VolatilityTargeted {
            target_vol,
            lookback,
        }
    }

    /// Create signal-scaled sizing.
    pub fn signal_scaled(base_size: f64) -> Self {
        PositionSizingMethod::SignalScaled { base_size }
    }

    /// Create risk-based sizing.
    pub fn risk_based(risk_per_trade: f64, stop_atr: f64, atr_period: usize) -> Self {
        PositionSizingMethod::RiskBased {
            risk_per_trade,
            stop_atr,
            atr_period,
        }
    }
}

/// Stop-loss configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum StopLoss {
    /// Fixed percentage stop loss from entry price.
    Percentage(f64),
    /// Fixed price amount stop loss.
    Fixed(f64),
    /// ATR-based stop loss (multiplier of ATR).
    Atr { multiplier: f64, atr_value: f64 },
    /// Trailing stop loss percentage.
    Trailing(f64),
    /// No stop loss.
    #[default]
    None,
}

impl StopLoss {
    /// Calculate stop loss price for a position.
    pub fn stop_price(&self, entry_price: f64, side: Side) -> Option<f64> {
        match self {
            StopLoss::Percentage(pct) => {
                let distance = entry_price * (pct / 100.0);
                match side {
                    Side::Buy => Some(entry_price - distance),
                    Side::Sell => Some(entry_price + distance),
                }
            }
            StopLoss::Fixed(amount) => match side {
                Side::Buy => Some(entry_price - amount),
                Side::Sell => Some(entry_price + amount),
            },
            StopLoss::Atr {
                multiplier,
                atr_value,
            } => {
                let distance = multiplier * atr_value;
                match side {
                    Side::Buy => Some(entry_price - distance),
                    Side::Sell => Some(entry_price + distance),
                }
            }
            StopLoss::Trailing(pct) => {
                let distance = entry_price * (pct / 100.0);
                match side {
                    Side::Buy => Some(entry_price - distance),
                    Side::Sell => Some(entry_price + distance),
                }
            }
            StopLoss::None => None,
        }
    }

    /// Check if stop loss is triggered.
    pub fn is_triggered(&self, entry_price: f64, current_price: f64, side: Side) -> bool {
        match self.stop_price(entry_price, side) {
            Some(stop) => match side {
                Side::Buy => current_price <= stop,
                Side::Sell => current_price >= stop,
            },
            None => false,
        }
    }
}

/// Take-profit configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum TakeProfit {
    /// Fixed percentage take profit from entry price.
    Percentage(f64),
    /// Fixed price amount take profit.
    Fixed(f64),
    /// Risk-reward ratio based take profit (multiplier of stop loss distance).
    RiskReward { ratio: f64, stop_distance: f64 },
    /// ATR-based take profit.
    Atr { multiplier: f64, atr_value: f64 },
    /// No take profit.
    #[default]
    None,
}

impl TakeProfit {
    /// Calculate take profit price for a position.
    pub fn target_price(&self, entry_price: f64, side: Side) -> Option<f64> {
        match self {
            TakeProfit::Percentage(pct) => {
                let distance = entry_price * (pct / 100.0);
                match side {
                    Side::Buy => Some(entry_price + distance),
                    Side::Sell => Some(entry_price - distance),
                }
            }
            TakeProfit::Fixed(amount) => match side {
                Side::Buy => Some(entry_price + amount),
                Side::Sell => Some(entry_price - amount),
            },
            TakeProfit::RiskReward {
                ratio,
                stop_distance,
            } => {
                let profit_distance = stop_distance * ratio;
                match side {
                    Side::Buy => Some(entry_price + profit_distance),
                    Side::Sell => Some(entry_price - profit_distance),
                }
            }
            TakeProfit::Atr {
                multiplier,
                atr_value,
            } => {
                let distance = multiplier * atr_value;
                match side {
                    Side::Buy => Some(entry_price + distance),
                    Side::Sell => Some(entry_price - distance),
                }
            }
            TakeProfit::None => None,
        }
    }

    /// Check if take profit is triggered.
    pub fn is_triggered(&self, entry_price: f64, current_price: f64, side: Side) -> bool {
        match self.target_price(entry_price, side) {
            Some(target) => match side {
                Side::Buy => current_price >= target,
                Side::Sell => current_price <= target,
            },
            None => false,
        }
    }
}

/// Risk management configuration for trades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Stop loss configuration.
    pub stop_loss: StopLoss,
    /// Take profit configuration.
    pub take_profit: TakeProfit,
    /// Maximum position size as fraction of equity.
    pub max_position_size: f64,
    /// Maximum number of concurrent positions.
    pub max_positions: usize,
    /// Maximum drawdown before halting trading (percentage).
    pub max_drawdown_pct: Option<f64>,
    /// Risk per trade as percentage of equity.
    pub risk_per_trade_pct: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            stop_loss: StopLoss::None,
            take_profit: TakeProfit::None,
            max_position_size: 1.0,
            max_positions: 1,
            max_drawdown_pct: None,
            risk_per_trade_pct: 2.0, // 2% risk per trade
        }
    }
}

impl RiskConfig {
    /// Create a risk config with percentage-based stop loss.
    pub fn with_stop_loss_pct(stop_pct: f64) -> Self {
        Self {
            stop_loss: StopLoss::Percentage(stop_pct),
            ..Default::default()
        }
    }

    /// Create a risk config with stop loss and take profit.
    pub fn with_stop_and_target(stop_pct: f64, target_pct: f64) -> Self {
        Self {
            stop_loss: StopLoss::Percentage(stop_pct),
            take_profit: TakeProfit::Percentage(target_pct),
            ..Default::default()
        }
    }

    /// Create a risk config with risk-reward ratio.
    pub fn with_risk_reward(stop_pct: f64, risk_reward_ratio: f64) -> Self {
        Self {
            stop_loss: StopLoss::Percentage(stop_pct),
            take_profit: TakeProfit::RiskReward {
                ratio: risk_reward_ratio,
                stop_distance: 0.0, // Will be calculated when position is opened
            },
            ..Default::default()
        }
    }
}

/// Trailing stop tracker for a position.
#[derive(Debug, Clone)]
pub struct TrailingStop {
    /// Initial entry price.
    pub entry_price: f64,
    /// Position side.
    pub side: Side,
    /// Trailing distance as percentage.
    pub trail_pct: f64,
    /// Highest price seen (for longs) or lowest price seen (for shorts).
    pub extreme_price: f64,
    /// Current stop price.
    pub stop_price: f64,
}

impl TrailingStop {
    /// Create a new trailing stop.
    pub fn new(entry_price: f64, side: Side, trail_pct: f64) -> Self {
        let stop_price = match side {
            Side::Buy => entry_price * (1.0 - trail_pct / 100.0),
            Side::Sell => entry_price * (1.0 + trail_pct / 100.0),
        };

        Self {
            entry_price,
            side,
            trail_pct,
            extreme_price: entry_price,
            stop_price,
        }
    }

    /// Update the trailing stop with current price.
    pub fn update(&mut self, current_price: f64) {
        match self.side {
            Side::Buy => {
                if current_price > self.extreme_price {
                    self.extreme_price = current_price;
                    self.stop_price = current_price * (1.0 - self.trail_pct / 100.0);
                }
            }
            Side::Sell => {
                if current_price < self.extreme_price {
                    self.extreme_price = current_price;
                    self.stop_price = current_price * (1.0 + self.trail_pct / 100.0);
                }
            }
        }
    }

    /// Check if stop is triggered.
    pub fn is_triggered(&self, current_price: f64) -> bool {
        match self.side {
            Side::Buy => current_price <= self.stop_price,
            Side::Sell => current_price >= self.stop_price,
        }
    }
}

/// Position sizing calculator.
pub struct PositionSizer;

impl PositionSizer {
    /// Calculate position size based on risk per trade.
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `risk_pct` - Risk per trade as percentage of equity
    /// * `entry_price` - Expected entry price
    /// * `stop_price` - Stop loss price
    ///
    /// # Returns
    /// Number of shares/contracts to trade
    pub fn size_by_risk(equity: f64, risk_pct: f64, entry_price: f64, stop_price: f64) -> f64 {
        let risk_amount = equity * (risk_pct / 100.0);
        let risk_per_share = (entry_price - stop_price).abs();

        if risk_per_share <= 0.0 {
            return 0.0;
        }

        risk_amount / risk_per_share
    }

    /// Calculate position size based on volatility (ATR).
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `risk_pct` - Risk per trade as percentage of equity
    /// * `atr` - Average True Range value
    /// * `atr_multiplier` - Multiplier for ATR to determine risk
    pub fn size_by_volatility(equity: f64, risk_pct: f64, atr: f64, atr_multiplier: f64) -> f64 {
        let risk_amount = equity * (risk_pct / 100.0);
        let risk_per_share = atr * atr_multiplier;

        if risk_per_share <= 0.0 {
            return 0.0;
        }

        risk_amount / risk_per_share
    }

    /// Calculate position size with Kelly Criterion.
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `win_rate` - Historical win rate (0.0 to 1.0)
    /// * `avg_win` - Average winning trade amount
    /// * `avg_loss` - Average losing trade amount (positive number)
    /// * `max_kelly_fraction` - Maximum fraction of Kelly to use (e.g., 0.5 for half-Kelly)
    pub fn size_by_kelly(
        equity: f64,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        max_kelly_fraction: f64,
    ) -> f64 {
        if avg_loss <= 0.0 || win_rate <= 0.0 || win_rate >= 1.0 {
            return 0.0;
        }

        let b = avg_win / avg_loss; // Odds ratio
        let kelly = win_rate - (1.0 - win_rate) / b;

        if kelly <= 0.0 {
            return 0.0;
        }

        let adjusted_kelly = kelly * max_kelly_fraction;
        equity * adjusted_kelly.min(max_kelly_fraction) // Cap at max fraction
    }

    /// Calculate position size for fixed dollar amount.
    ///
    /// # Arguments
    /// * `amount` - Fixed dollar amount to trade
    /// * `price` - Current price of the asset
    ///
    /// # Returns
    /// Number of shares/contracts to trade
    pub fn size_fixed_dollar(amount: f64, price: f64) -> f64 {
        if price <= 0.0 {
            return 0.0;
        }
        amount / price
    }

    /// Calculate position size scaled by signal magnitude.
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `base_size` - Base position size as fraction of equity (e.g., 0.10 for 10%)
    /// * `signal_magnitude` - Absolute value of the signal (scales the position)
    ///
    /// # Returns
    /// Dollar amount to trade (not shares - divide by price for shares)
    pub fn size_by_signal(equity: f64, base_size: f64, signal_magnitude: f64) -> f64 {
        equity * base_size * signal_magnitude.abs()
    }

    /// Calculate position size targeting a specific portfolio volatility.
    ///
    /// This scales position size inversely with asset volatility to maintain
    /// a target overall portfolio volatility.
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `target_vol` - Target annual portfolio volatility (e.g., 0.15 for 15%)
    /// * `asset_vol` - Current asset's annualized volatility
    ///
    /// # Returns
    /// Dollar amount to allocate to this position
    ///
    /// # Formula
    /// Position size = (equity × target_vol) / (asset_vol × √252)
    /// For daily data, asset_vol should already be annualized or we annualize here.
    pub fn size_by_volatility_target(equity: f64, target_vol: f64, asset_vol: f64) -> f64 {
        if asset_vol <= 0.0 {
            return 0.0;
        }
        // If asset_vol is already annualized, use it directly
        // Position = equity × target_vol / asset_vol
        equity * target_vol / asset_vol
    }

    /// Calculate percent of equity position size.
    ///
    /// # Arguments
    /// * `equity` - Current account equity
    /// * `percent` - Percentage of equity to allocate (e.g., 0.10 for 10%)
    ///
    /// # Returns
    /// Dollar amount to allocate
    pub fn size_percent_of_equity(equity: f64, percent: f64) -> f64 {
        equity * percent
    }
}

/// Result of a risk check.
#[derive(Debug, Clone)]
pub enum RiskCheckResult {
    /// Position can be opened.
    Ok,
    /// Position would exceed max position size.
    MaxPositionSizeExceeded,
    /// Maximum drawdown has been reached.
    MaxDrawdownReached,
    /// Maximum number of positions reached.
    MaxPositionsReached,
    /// Insufficient funds for the position.
    InsufficientFunds,
}

/// Risk manager for checking trade validity.
pub struct RiskManager {
    config: RiskConfig,
    peak_equity: f64,
    current_drawdown_pct: f64,
}

impl RiskManager {
    pub fn new(config: RiskConfig, initial_equity: f64) -> Self {
        Self {
            config,
            peak_equity: initial_equity,
            current_drawdown_pct: 0.0,
        }
    }

    /// Update drawdown tracking.
    pub fn update_equity(&mut self, equity: f64) {
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        self.current_drawdown_pct = (self.peak_equity - equity) / self.peak_equity * 100.0;
    }

    /// Check if a new position can be opened.
    pub fn check_new_position(
        &self,
        equity: f64,
        position_value: f64,
        current_positions: usize,
    ) -> RiskCheckResult {
        // Check max drawdown
        if let Some(max_dd) = self.config.max_drawdown_pct {
            if self.current_drawdown_pct >= max_dd {
                return RiskCheckResult::MaxDrawdownReached;
            }
        }

        // Check max positions
        if current_positions >= self.config.max_positions {
            return RiskCheckResult::MaxPositionsReached;
        }

        // Check position size
        let position_size_pct = position_value / equity;
        if position_size_pct > self.config.max_position_size {
            return RiskCheckResult::MaxPositionSizeExceeded;
        }

        // Check funds
        if position_value > equity {
            return RiskCheckResult::InsufficientFunds;
        }

        RiskCheckResult::Ok
    }

    /// Get the current drawdown percentage.
    pub fn current_drawdown(&self) -> f64 {
        self.current_drawdown_pct
    }

    /// Get the risk configuration.
    pub fn config(&self) -> &RiskConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_loss_percentage() {
        let stop = StopLoss::Percentage(5.0);

        // Long position
        let stop_price = stop.stop_price(100.0, Side::Buy).unwrap();
        assert!((stop_price - 95.0).abs() < 0.001);

        // Short position
        let stop_price = stop.stop_price(100.0, Side::Sell).unwrap();
        assert!((stop_price - 105.0).abs() < 0.001);
    }

    #[test]
    fn test_stop_loss_triggered() {
        let stop = StopLoss::Percentage(5.0);

        // Long position - stop triggered
        assert!(stop.is_triggered(100.0, 94.0, Side::Buy));

        // Long position - stop not triggered
        assert!(!stop.is_triggered(100.0, 96.0, Side::Buy));

        // Short position - stop triggered
        assert!(stop.is_triggered(100.0, 106.0, Side::Sell));
    }

    #[test]
    fn test_take_profit_percentage() {
        let tp = TakeProfit::Percentage(10.0);

        // Long position
        let target = tp.target_price(100.0, Side::Buy).unwrap();
        assert!((target - 110.0).abs() < 0.001);

        // Short position
        let target = tp.target_price(100.0, Side::Sell).unwrap();
        assert!((target - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_take_profit_triggered() {
        let tp = TakeProfit::Percentage(10.0);

        // Long position - target hit
        assert!(tp.is_triggered(100.0, 111.0, Side::Buy));

        // Long position - target not hit
        assert!(!tp.is_triggered(100.0, 109.0, Side::Buy));
    }

    #[test]
    fn test_trailing_stop() {
        let mut trailing = TrailingStop::new(100.0, Side::Buy, 5.0);
        assert!((trailing.stop_price - 95.0).abs() < 0.001);

        // Price moves up
        trailing.update(110.0);
        assert!((trailing.extreme_price - 110.0).abs() < 0.001);
        assert!((trailing.stop_price - 104.5).abs() < 0.001);

        // Price moves down but not to stop
        trailing.update(108.0);
        assert!((trailing.extreme_price - 110.0).abs() < 0.001); // Unchanged
        assert!(!trailing.is_triggered(108.0));

        // Price triggers stop
        assert!(trailing.is_triggered(104.0));
    }

    #[test]
    fn test_position_size_by_risk() {
        let size = PositionSizer::size_by_risk(100000.0, 2.0, 100.0, 95.0);
        // Risk $2000, risk per share = $5, size = 400 shares
        assert!((size - 400.0).abs() < 0.001);
    }

    #[test]
    fn test_position_size_by_volatility() {
        let size = PositionSizer::size_by_volatility(100000.0, 2.0, 5.0, 2.0);
        // Risk $2000, risk per share = $10 (ATR*2), size = 200 shares
        assert!((size - 200.0).abs() < 0.001);
    }

    #[test]
    fn test_kelly_criterion() {
        // 60% win rate, avg win = $150, avg loss = $100
        // Kelly = 0.6 - 0.4/1.5 = 0.6 - 0.267 = 0.333
        let size = PositionSizer::size_by_kelly(100000.0, 0.6, 150.0, 100.0, 0.5);
        // Half-Kelly, so ~16.65% of equity
        assert!(size > 0.0);
        assert!(size < 50000.0); // Capped at 50% for half-Kelly
    }

    #[test]
    fn test_risk_manager_drawdown() {
        let config = RiskConfig {
            max_drawdown_pct: Some(10.0),
            ..Default::default()
        };
        let mut rm = RiskManager::new(config, 100000.0);

        // Update to lower equity
        rm.update_equity(95000.0);
        assert!((rm.current_drawdown() - 5.0).abs() < 0.001);

        // Check position - should be OK
        let result = rm.check_new_position(95000.0, 10000.0, 0);
        assert!(matches!(result, RiskCheckResult::Ok));

        // Simulate larger drawdown
        rm.update_equity(89000.0);
        let result = rm.check_new_position(89000.0, 10000.0, 0);
        assert!(matches!(result, RiskCheckResult::MaxDrawdownReached));
    }

    #[test]
    fn test_size_fixed_dollar() {
        // Fixed $10,000 at $50/share = 200 shares
        let shares = PositionSizer::size_fixed_dollar(10000.0, 50.0);
        assert!((shares - 200.0).abs() < 0.001);

        // Fixed $10,000 at $100/share = 100 shares
        let shares = PositionSizer::size_fixed_dollar(10000.0, 100.0);
        assert!((shares - 100.0).abs() < 0.001);

        // Edge case: zero price
        let shares = PositionSizer::size_fixed_dollar(10000.0, 0.0);
        assert_eq!(shares, 0.0);
    }

    #[test]
    fn test_size_by_signal() {
        // Base 10% of equity, signal magnitude 1.0
        let value = PositionSizer::size_by_signal(100000.0, 0.1, 1.0);
        assert!((value - 10000.0).abs() < 0.001);

        // Base 10% of equity, signal magnitude 2.0 = 20% position
        let value = PositionSizer::size_by_signal(100000.0, 0.1, 2.0);
        assert!((value - 20000.0).abs() < 0.001);

        // Base 10% of equity, signal magnitude 0.5 = 5% position
        let value = PositionSizer::size_by_signal(100000.0, 0.1, 0.5);
        assert!((value - 5000.0).abs() < 0.001);

        // Negative signal (absolute value used)
        let value = PositionSizer::size_by_signal(100000.0, 0.1, -2.0);
        assert!((value - 20000.0).abs() < 0.001);
    }

    #[test]
    fn test_size_by_volatility_target() {
        // $100k equity, target 15% vol, asset has 30% vol -> 50% position
        let value = PositionSizer::size_by_volatility_target(100000.0, 0.15, 0.30);
        assert!((value - 50000.0).abs() < 0.001);

        // $100k equity, target 15% vol, asset has 15% vol -> 100% position
        let value = PositionSizer::size_by_volatility_target(100000.0, 0.15, 0.15);
        assert!((value - 100000.0).abs() < 0.001);

        // $100k equity, target 15% vol, asset has 7.5% vol -> 200% position (leverage)
        let value = PositionSizer::size_by_volatility_target(100000.0, 0.15, 0.075);
        assert!((value - 200000.0).abs() < 0.001);

        // Edge case: zero volatility
        let value = PositionSizer::size_by_volatility_target(100000.0, 0.15, 0.0);
        assert_eq!(value, 0.0);
    }

    #[test]
    fn test_size_percent_of_equity() {
        // 10% of $100,000 = $10,000
        let value = PositionSizer::size_percent_of_equity(100000.0, 0.1);
        assert!((value - 10000.0).abs() < 0.001);

        // 25% of $100,000 = $25,000
        let value = PositionSizer::size_percent_of_equity(100000.0, 0.25);
        assert!((value - 25000.0).abs() < 0.001);
    }

    #[test]
    fn test_position_sizing_method_default() {
        let method = PositionSizingMethod::default();
        assert!(matches!(
            method,
            PositionSizingMethod::PercentOfEquity(pct) if (pct - 0.1).abs() < 0.001
        ));
    }

    #[test]
    fn test_position_sizing_method_constructors() {
        let percent = PositionSizingMethod::percent(0.2);
        assert!(matches!(
            percent,
            PositionSizingMethod::PercentOfEquity(p) if (p - 0.2).abs() < 0.001
        ));

        let fixed = PositionSizingMethod::fixed_dollar(5000.0);
        assert!(matches!(
            fixed,
            PositionSizingMethod::FixedDollar(a) if (a - 5000.0).abs() < 0.001
        ));

        let volatility = PositionSizingMethod::volatility(0.12, 30);
        assert!(matches!(
            volatility,
            PositionSizingMethod::VolatilityTargeted { target_vol, lookback }
                if (target_vol - 0.12).abs() < 0.001 && lookback == 30
        ));

        let signal = PositionSizingMethod::signal_scaled(0.15);
        assert!(matches!(
            signal,
            PositionSizingMethod::SignalScaled { base_size }
                if (base_size - 0.15).abs() < 0.001
        ));

        let risk = PositionSizingMethod::risk_based(1.0, 2.5, 20);
        assert!(matches!(
            risk,
            PositionSizingMethod::RiskBased { risk_per_trade, stop_atr, atr_period }
                if (risk_per_trade - 1.0).abs() < 0.001
                && (stop_atr - 2.5).abs() < 0.001
                && atr_period == 20
        ));
    }
}
