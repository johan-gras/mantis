//! Backtest execution engine.

use crate::data::DataManager;
use crate::error::{BacktestError, Result};
use crate::metadata::{compute_config_hash, generate_experiment_id, GitInfo};
use crate::portfolio::{CostModel, MarginConfig, Portfolio};
use crate::risk::{PositionSizer, PositionSizingMethod, RiskConfig, StopLoss, TrailingStop};
use crate::strategy::{Strategy, StrategyContext};
use crate::timeframe::TimeframeManager;
use crate::types::{
    Bar, DataFrequency, EquityPoint, ExecutionPrice, LotSelectionMethod, Order, OrderType, Side,
    Signal, Trade, VolumeProfile,
};
use chrono::{DateTime, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Configuration for the backtest engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital for the backtest.
    pub initial_capital: f64,
    /// Trading costs configuration.
    pub cost_model: CostModel,
    /// Margin configuration for leverage and borrowing rules.
    #[serde(default)]
    pub margin: MarginConfig,
    /// Position sizing as fraction of equity (0.0 to 1.0).
    /// This is used as fallback when `position_sizing_method` is `PercentOfEquity`.
    pub position_size: f64,
    /// Position sizing method. Determines how position sizes are calculated.
    /// If not specified, defaults to PercentOfEquity using the `position_size` field.
    #[serde(default)]
    pub position_sizing_method: Option<PositionSizingMethod>,
    /// Allow short selling.
    pub allow_short: bool,
    /// Allow fractional shares.
    pub fractional_shares: bool,
    /// Show progress bar during backtest.
    pub show_progress: bool,
    /// Start date filter (optional).
    pub start_date: Option<DateTime<Utc>>,
    /// End date filter (optional).
    pub end_date: Option<DateTime<Utc>>,
    /// Risk management configuration.
    #[serde(default)]
    pub risk_config: RiskConfig,
    /// Execution price model for market orders.
    #[serde(default)]
    pub execution_price: ExecutionPrice,
    /// Probability that an order fully fills on a given attempt.
    #[serde(default = "default_fill_probability")]
    pub fill_probability: f64,
    /// Number of bars before pending limit orders expire (None = GTC).
    #[serde(default)]
    pub limit_order_ttl_bars: Option<usize>,
    /// Default tax-lot selection policy when closing positions.
    #[serde(default)]
    pub lot_selection: LotSelectionMethod,
    /// Random seed for reproducible execution (None = deterministic from timestamps).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Data frequency for proper annualization of metrics.
    /// If None, frequency is auto-detected from bar timestamps.
    #[serde(default)]
    pub data_frequency: Option<DataFrequency>,
    /// Whether to use 24/7 trading hours (for crypto markets).
    /// If None, this is auto-detected based on weekend bars.
    #[serde(default)]
    pub trading_hours_24: Option<bool>,
    /// Default order type for signal-generated orders.
    /// If true, uses limit orders with offset from close price.
    /// If false (default), uses market orders.
    #[serde(default)]
    pub use_limit_orders: bool,
    /// Limit order offset as a fraction of close price (e.g., 0.01 for 1%).
    /// For buys: limit_price = close * (1 - limit_offset)
    /// For sells: limit_price = close * (1 + limit_offset)
    #[serde(default)]
    pub limit_offset: f64,
    /// Annual risk-free rate for Sharpe/Sortino calculation (as decimal, e.g., 0.02 for 2%).
    /// Default is 0.0 (no risk-free rate adjustment).
    #[serde(default)]
    pub risk_free_rate: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            cost_model: CostModel::default(),
            margin: MarginConfig::default(),
            position_size: 0.1,
            position_sizing_method: None, // Uses position_size field as fallback
            allow_short: true,
            fractional_shares: false, // Default: whole shares (per spec)
            show_progress: true,
            start_date: None,
            end_date: None,
            risk_config: RiskConfig::default(),
            execution_price: ExecutionPrice::Open,
            fill_probability: default_fill_probability(),
            limit_order_ttl_bars: Some(5),
            lot_selection: LotSelectionMethod::default(),
            seed: None,
            data_frequency: None,   // Auto-detect from data
            trading_hours_24: None, // Auto-detect from data
            use_limit_orders: false,
            limit_offset: 0.0,
            risk_free_rate: 0.0, // No risk-free rate by default (per spec note)
        }
    }
}

fn default_fill_probability() -> f64 {
    1.0
}

/// Results from a backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Strategy name.
    pub strategy_name: String,
    /// Symbol(s) traded.
    pub symbols: Vec<String>,
    /// Configuration used.
    pub config: BacktestConfig,
    /// Initial capital.
    pub initial_capital: f64,
    /// Final equity.
    pub final_equity: f64,
    /// Total return percentage.
    pub total_return_pct: f64,
    /// Annual return percentage (CAGR).
    pub annual_return_pct: f64,
    /// Number of trading days.
    pub trading_days: usize,
    /// Total number of trades.
    pub total_trades: usize,
    /// Number of winning trades.
    pub winning_trades: usize,
    /// Number of losing trades.
    pub losing_trades: usize,
    /// Win rate percentage.
    pub win_rate: f64,
    /// Average winning trade P&L.
    pub avg_win: f64,
    /// Average losing trade P&L.
    pub avg_loss: f64,
    /// Profit factor (gross wins / gross losses).
    pub profit_factor: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown_pct: f64,
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized).
    pub sortino_ratio: f64,
    /// Calmar ratio (CAGR / max drawdown).
    pub calmar_ratio: f64,
    /// All trades.
    pub trades: Vec<Trade>,
    /// Equity curve.
    pub equity_curve: Vec<EquityPoint>,
    /// Start timestamp.
    pub start_time: DateTime<Utc>,
    /// End timestamp.
    pub end_time: DateTime<Utc>,
    /// Unique experiment identifier for reproducibility.
    #[serde(default = "generate_experiment_id")]
    pub experiment_id: Uuid,
    /// Git repository state (commit SHA, branch, dirty flag).
    #[serde(default)]
    pub git_info: Option<GitInfo>,
    /// SHA256 hash of the configuration for change detection.
    #[serde(default)]
    pub config_hash: String,
    /// Data file checksums (symbol -> SHA256 hash).
    #[serde(default)]
    pub data_checksums: HashMap<String, String>,
    /// Random seed used for execution (None = deterministic from timestamps).
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct PendingOrder {
    order: Order,
    created_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    remaining_quantity: f64,
    /// Original signal that generated this order (for entry tracking)
    signal: Option<Signal>,
}

impl PendingOrder {
    fn is_expired(&self, timestamp: DateTime<Utc>) -> bool {
        self.expires_at
            .map(|expires| timestamp >= expires)
            .unwrap_or(false)
    }
}

/// The main backtest engine.
pub struct Engine {
    config: BacktestConfig,
    data: DataManager,
}

impl Engine {
    /// Create a new backtest engine.
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            data: DataManager::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Set the configuration.
    pub fn set_config(&mut self, config: BacktestConfig) {
        self.config = config;
    }

    /// Get a mutable reference to the data manager.
    pub fn data_mut(&mut self) -> &mut DataManager {
        &mut self.data
    }

    /// Get a reference to the data manager.
    pub fn data(&self) -> &DataManager {
        &self.data
    }

    /// Add data for a symbol.
    pub fn add_data(&mut self, symbol: impl Into<String>, bars: Vec<Bar>) {
        self.data.add(symbol, bars);
    }

    /// Run a backtest on a single symbol.
    pub fn run(&self, strategy: &mut dyn Strategy, symbol: &str) -> Result<BacktestResult> {
        let bars = self
            .data
            .get(symbol)
            .ok_or_else(|| BacktestError::DataError(format!("No data for symbol: {}", symbol)))?;

        if bars.is_empty() {
            return Err(BacktestError::NoData);
        }

        // Apply date filters
        let bars: Vec<Bar> = bars
            .iter()
            .filter(|b| {
                self.config
                    .start_date
                    .is_none_or(|start| b.timestamp >= start)
                    && self.config.end_date.is_none_or(|end| b.timestamp <= end)
            })
            .cloned()
            .collect();

        if bars.is_empty() {
            return Err(BacktestError::DataError(
                "No data in specified date range".to_string(),
            ));
        }

        let volume_profile = VolumeProfile::from_bars(&bars);

        info!(
            "Running backtest: {} on {} ({} bars)",
            strategy.name(),
            symbol,
            bars.len()
        );

        // Initialize strategy
        strategy.init();

        // Check if strategy requests additional timeframes
        let requested_timeframes = strategy.requested_timeframes();
        let timeframe_manager = if !requested_timeframes.is_empty() {
            info!(
                "Strategy requested {} additional timeframes: {:?}",
                requested_timeframes.len(),
                requested_timeframes
            );
            let mut manager = TimeframeManager::new(bars.clone());
            for interval in requested_timeframes {
                manager.request_timeframe(interval)?;
            }
            Some(manager)
        } else {
            None
        };

        // Create portfolio
        let mut portfolio =
            Portfolio::with_cost_model(self.config.initial_capital, self.config.cost_model.clone());
        portfolio.allow_short = self.config.allow_short;
        portfolio.fractional_shares = self.config.fractional_shares;
        portfolio.set_execution_price(self.config.execution_price);
        portfolio.set_lot_selection_method(self.config.lot_selection.clone());
        portfolio.set_margin_config(self.config.margin.clone());
        portfolio.set_asset_configs(self.data.asset_configs());
        portfolio.set_rng_seed(self.config.seed);
        if let Some(profile) = volume_profile {
            portfolio.set_volume_profile(symbol.to_string(), profile);
        }

        // Setup progress bar
        let progress = if self.config.show_progress {
            let pb = ProgressBar::new(bars.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        let warmup = strategy.warmup_period();

        // Track entry prices and trailing stops for risk management
        let mut entry_prices: HashMap<String, f64> = HashMap::new();
        let mut trailing_stops: HashMap<String, TrailingStop> = HashMap::new();
        let mut pending_orders: Vec<PendingOrder> = Vec::new();
        // Track symbols with pending exit orders to avoid double-buffering
        let mut pending_exits: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Main backtest loop
        for i in 0..bars.len() {
            let bar = &bars[i];

            // Record equity
            let mut prices = HashMap::new();
            prices.insert(symbol.to_string(), bar.close);
            portfolio.record_equity(bar.timestamp, &prices)?;

            // Skip warmup period
            if i < warmup {
                if let Some(ref pb) = progress {
                    pb.inc(1);
                }
                continue;
            }

            // Check stop-loss and take-profit for existing positions
            if let Some(position) = portfolio.position(symbol) {
                if let Some(&entry_price) = entry_prices.get(symbol) {
                    let position_side = position.side;
                    let current_price = bar.close;

                    // Check stop-loss
                    let stop_triggered = self.config.risk_config.stop_loss.is_triggered(
                        entry_price,
                        current_price,
                        position_side,
                    );

                    // Check take-profit
                    let take_profit_triggered = self.config.risk_config.take_profit.is_triggered(
                        entry_price,
                        current_price,
                        position_side,
                    );

                    // Check trailing stop if configured
                    let trailing_triggered = if let Some(ts) = trailing_stops.get_mut(symbol) {
                        ts.update(current_price);
                        ts.is_triggered(current_price)
                    } else {
                        false
                    };

                    // Buffer exit order if any stop/target is hit (fills at next bar open)
                    // Only buffer if no exit is already pending for this symbol
                    if (stop_triggered || take_profit_triggered || trailing_triggered)
                        && !pending_exits.contains(symbol)
                    {
                        let exit_reason = if stop_triggered {
                            "stop-loss"
                        } else if take_profit_triggered {
                            "take-profit"
                        } else {
                            "trailing-stop"
                        };

                        let exit_side = match position_side {
                            Side::Buy => Side::Sell,
                            Side::Sell => Side::Buy,
                        };

                        let exit_order =
                            Order::market(symbol, exit_side, position.quantity, bar.timestamp);

                        // Buffer exit for next bar to prevent lookahead bias
                        self.buffer_order_for_next_bar(
                            &mut pending_orders,
                            exit_order,
                            None,
                            &bars,
                            i,
                        );
                        pending_exits.insert(symbol.to_string());

                        debug!(
                            "{} triggered for {}: buffered exit for next bar @ {:.2}",
                            exit_reason, symbol, current_price
                        );

                        // Clean up entry tracking (will be stale but harmless)
                        entry_prices.remove(symbol);
                        trailing_stops.remove(symbol);

                        if let Some(ref pb) = progress {
                            pb.inc(1);
                        }
                        continue; // Skip strategy signal this bar
                    }
                }
            }

            // Create strategy context
            let ctx = StrategyContext {
                bar_index: i,
                bars: &bars,
                position: portfolio.position_qty(symbol),
                cash: portfolio.cash,
                equity: portfolio.equity(&prices),
                symbol,
                volume_profile,
                timeframe_manager: timeframe_manager.as_ref(),
            };

            self.handle_pending_orders(
                &mut pending_orders,
                &mut pending_exits,
                &mut entry_prices,
                &mut trailing_stops,
                bar,
                strategy,
                &ctx,
                &mut portfolio,
            )?;

            // Check for custom orders first
            if let Some(orders) = strategy.generate_orders(&ctx) {
                // Buffer orders for execution at next bar's open (prevents lookahead bias)
                for order in orders {
                    self.buffer_order_for_next_bar(&mut pending_orders, order, None, &bars, i);
                }
            } else {
                // Get signal and convert to order
                let signal = strategy.on_bar(&ctx);
                if let Some(order) = self.signal_to_order(signal, symbol, bar, &bars, i, &portfolio)
                {
                    // Buffer order for execution at next bar's open (prevents lookahead bias)
                    // Entry tracking (entry_prices, trailing_stops) will be handled in
                    // handle_pending_orders when the order actually fills
                    self.buffer_order_for_next_bar(
                        &mut pending_orders,
                        order,
                        Some(signal),
                        &bars,
                        i,
                    );
                }
            }

            if let Some(ref pb) = progress {
                pb.inc(1);
            }
        }

        if let Some(pb) = progress {
            pb.finish_with_message("Backtest complete");
        }

        // Finish strategy
        strategy.on_finish();

        // Calculate results
        let result = self.calculate_results(strategy.name(), symbol, &portfolio, &bars);

        info!(
            "Backtest complete: {:.2}% return, {:.2}% max DD, {:.2} Sharpe",
            result.total_return_pct, result.max_drawdown_pct, result.sharpe_ratio
        );

        Ok(result)
    }

    /// Convert a signal to an order.
    fn signal_to_order(
        &self,
        signal: Signal,
        symbol: &str,
        bar: &Bar,
        bars: &[Bar],
        bar_index: usize,
        portfolio: &Portfolio,
    ) -> Option<Order> {
        let current_position = portfolio.position_qty(symbol);
        let equity = portfolio.cash
            + portfolio
                .positions()
                .values()
                .map(|p| p.quantity * p.avg_entry_price)
                .sum::<f64>();

        match signal {
            Signal::Long => {
                if current_position <= 0.0 {
                    // Close short if exists, then go long
                    let close_qty = current_position.abs();
                    let long_value =
                        self.calculate_position_value(equity, bar.close, bars, bar_index, 1.0);
                    let long_qty = if self.config.fractional_shares {
                        long_value / bar.close
                    } else {
                        (long_value / bar.close).floor()
                    };

                    let total_qty = close_qty + long_qty;
                    if total_qty > 0.0 {
                        return Some(self.create_order(symbol, Side::Buy, total_qty, bar));
                    }
                }
            }
            Signal::Short => {
                if self.config.allow_short && current_position >= 0.0 {
                    // Close long if exists, then go short
                    let close_qty = current_position;
                    let short_value =
                        self.calculate_position_value(equity, bar.close, bars, bar_index, 1.0);
                    let short_qty = if self.config.fractional_shares {
                        short_value / bar.close
                    } else {
                        (short_value / bar.close).floor()
                    };

                    let total_qty = close_qty + short_qty;
                    if total_qty > 0.0 {
                        return Some(self.create_order(symbol, Side::Sell, total_qty, bar));
                    }
                }
            }
            Signal::Exit => {
                if current_position > 0.0 {
                    return Some(self.create_order(symbol, Side::Sell, current_position, bar));
                } else if current_position < 0.0 {
                    return Some(self.create_order(symbol, Side::Buy, current_position.abs(), bar));
                }
            }
            Signal::Hold => {}
        }

        None
    }

    /// Create an order based on config (market or limit).
    fn create_order(&self, symbol: &str, side: Side, quantity: f64, bar: &Bar) -> Order {
        if self.config.use_limit_orders && self.config.limit_offset > 0.0 {
            // Calculate limit price based on side and offset
            // For buys: limit below close (better price for buyer)
            // For sells: limit above close (better price for seller)
            let limit_price = match side {
                Side::Buy => bar.close * (1.0 - self.config.limit_offset),
                Side::Sell => bar.close * (1.0 + self.config.limit_offset),
            };
            Order::limit(symbol, side, quantity, limit_price, bar.timestamp)
        } else {
            Order::market(symbol, side, quantity, bar.timestamp)
        }
    }

    /// Calculate position value based on the configured sizing method.
    fn calculate_position_value(
        &self,
        equity: f64,
        price: f64,
        bars: &[Bar],
        bar_index: usize,
        signal_magnitude: f64,
    ) -> f64 {
        let method = self.config.position_sizing_method.clone().unwrap_or(
            PositionSizingMethod::PercentOfEquity(self.config.position_size),
        );

        match method {
            PositionSizingMethod::PercentOfEquity(pct) => {
                PositionSizer::size_percent_of_equity(equity, pct)
            }
            PositionSizingMethod::FixedDollar(amount) => {
                // Return dollar amount directly (caller converts to shares)
                amount
            }
            PositionSizingMethod::VolatilityTargeted {
                target_vol,
                lookback,
            } => {
                let asset_vol = self.calculate_volatility(bars, bar_index, lookback);
                if asset_vol > 0.0 {
                    PositionSizer::size_by_volatility_target(equity, target_vol, asset_vol)
                } else {
                    // Fallback to percent of equity if volatility can't be calculated
                    warn!("Cannot calculate volatility, falling back to percent sizing");
                    PositionSizer::size_percent_of_equity(equity, self.config.position_size)
                }
            }
            PositionSizingMethod::SignalScaled { base_size } => {
                PositionSizer::size_by_signal(equity, base_size, signal_magnitude)
            }
            PositionSizingMethod::RiskBased {
                risk_per_trade,
                stop_atr,
                atr_period,
            } => {
                let atr = self.calculate_atr(bars, bar_index, atr_period);
                if atr > 0.0 {
                    // Position size in shares, convert to dollars
                    let shares =
                        PositionSizer::size_by_volatility(equity, risk_per_trade, atr, stop_atr);
                    shares * price
                } else {
                    // Fallback to percent of equity if ATR can't be calculated
                    warn!("Cannot calculate ATR, falling back to percent sizing");
                    PositionSizer::size_percent_of_equity(equity, self.config.position_size)
                }
            }
        }
    }

    /// Calculate annualized volatility from historical bars.
    fn calculate_volatility(&self, bars: &[Bar], bar_index: usize, lookback: usize) -> f64 {
        if bar_index < lookback || lookback < 2 {
            return 0.0;
        }

        let start = bar_index.saturating_sub(lookback);
        let returns: Vec<f64> = bars[start..=bar_index]
            .windows(2)
            .map(|w| (w[1].close / w[0].close).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let daily_vol = variance.sqrt();

        // Annualize (assuming 252 trading days)
        daily_vol * 252.0_f64.sqrt()
    }

    /// Calculate Average True Range (ATR) from historical bars.
    fn calculate_atr(&self, bars: &[Bar], bar_index: usize, period: usize) -> f64 {
        if bar_index < period || period == 0 {
            return 0.0;
        }

        let start = bar_index.saturating_sub(period);
        let mut tr_sum = 0.0;

        for i in (start + 1)..=bar_index {
            let bar = &bars[i];
            let prev_close = bars[i - 1].close;

            // True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
            let tr = (bar.high - bar.low)
                .max((bar.high - prev_close).abs())
                .max((bar.low - prev_close).abs());
            tr_sum += tr;
        }

        tr_sum / period as f64
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_pending_orders(
        &self,
        pending_orders: &mut Vec<PendingOrder>,
        pending_exits: &mut std::collections::HashSet<String>,
        entry_prices: &mut HashMap<String, f64>,
        trailing_stops: &mut HashMap<String, TrailingStop>,
        bar: &Bar,
        strategy: &mut dyn Strategy,
        ctx: &StrategyContext,
        portfolio: &mut Portfolio,
    ) -> Result<()> {
        if pending_orders.is_empty() {
            return Ok(());
        }

        let mut still_pending = Vec::new();
        for mut pending in pending_orders.drain(..) {
            if pending.is_expired(bar.timestamp) {
                debug!(
                    "Pending order expired after {:?}",
                    bar.timestamp - pending.created_at
                );
                // Clear pending exit tracking for expired orders
                pending_exits.remove(&pending.order.symbol);
                continue;
            }

            let mut pending_order = pending.order.clone();
            pending_order.quantity = pending.remaining_quantity;
            let signal = pending.signal;
            // Per spec: Insufficient cash should skip trade with warning, not fail backtest
            let execution_result = portfolio.execute_with_fill_probability(
                &pending_order,
                bar,
                self.config.fill_probability,
            );
            match execution_result {
                Err(BacktestError::InsufficientFunds {
                    required,
                    available,
                }) => {
                    warn!(
                        "Skipping trade for {}: insufficient funds (required: ${:.2}, available: ${:.2})",
                        pending_order.symbol, required, available
                    );
                    pending_exits.remove(&pending_order.symbol);
                    continue;
                }
                Err(e) => return Err(e),
                Ok(Some(fill)) => {
                    if let Some(trade) = fill.trade.as_ref() {
                        strategy.on_trade(ctx, &pending_order);
                        debug!("Trade executed: {:?}", trade);

                        // Handle entry tracking for Long/Short signals
                        if matches!(signal, Some(Signal::Long) | Some(Signal::Short)) {
                            entry_prices.insert(pending_order.symbol.clone(), trade.entry_price);

                            if let StopLoss::Trailing(trail_pct) = self.config.risk_config.stop_loss
                            {
                                let ts =
                                    TrailingStop::new(trade.entry_price, trade.side, trail_pct);
                                trailing_stops.insert(pending_order.symbol.clone(), ts);
                            }
                        } else if matches!(signal, Some(Signal::Exit))
                            && portfolio.position(&pending_order.symbol).is_none()
                        {
                            entry_prices.remove(&pending_order.symbol);
                            trailing_stops.remove(&pending_order.symbol);
                        }
                    }

                    if fill.partial && fill.remaining_quantity > f64::EPSILON {
                        pending.remaining_quantity = fill.remaining_quantity;
                        still_pending.push(pending);
                    } else {
                        // Order fully filled, clear pending exit tracking
                        pending_exits.remove(&pending_order.symbol);
                    }
                }
                Ok(None) => still_pending.push(pending),
            }
        }

        *pending_orders = still_pending;
        Ok(())
    }

    /// Buffer an order for execution at the next bar's open.
    /// This prevents lookahead bias by ensuring orders generated from bar[i]
    /// data fill at bar[i+1].open, not bar[i].open.
    fn buffer_order_for_next_bar(
        &self,
        pending_orders: &mut Vec<PendingOrder>,
        order: Order,
        signal: Option<Signal>,
        bars: &[Bar],
        index: usize,
    ) {
        // Market orders get no expiry (execute at next bar)
        // Limit orders get TTL-based expiry
        let expires_at = match order.order_type {
            OrderType::Market => None,
            _ => self.pending_expiry_for(bars, index),
        };

        pending_orders.push(PendingOrder {
            order: order.clone(),
            created_at: bars[index].timestamp,
            expires_at,
            remaining_quantity: order.quantity,
            signal,
        });
    }

    /// Calculate backtest results.
    fn calculate_results(
        &self,
        strategy_name: &str,
        symbol: &str,
        portfolio: &Portfolio,
        bars: &[Bar],
    ) -> BacktestResult {
        let equity_curve = portfolio.equity_curve();
        let trades = portfolio.trades();
        let closed_trades: Vec<_> = trades.iter().filter(|t| t.is_closed()).collect();

        let final_equity = equity_curve
            .last()
            .map(|e| e.equity)
            .unwrap_or(self.config.initial_capital);
        let total_return_pct =
            (final_equity - self.config.initial_capital) / self.config.initial_capital * 100.0;

        // Calculate time period
        let start_time = bars.first().map(|b| b.timestamp).unwrap_or_else(Utc::now);
        let end_time = bars.last().map(|b| b.timestamp).unwrap_or_else(Utc::now);
        let days = (end_time - start_time).num_days() as f64;
        let years = days / 365.0;

        // Annual return (CAGR)
        let annual_return_pct = if years > 0.0 {
            ((final_equity / self.config.initial_capital).powf(1.0 / years) - 1.0) * 100.0
        } else {
            0.0
        };

        // Win/loss statistics
        let winning: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) > 0.0)
            .collect();
        let losing: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) < 0.0)
            .collect();

        // Per spec: Zero trades returns NaN for trade-based metrics
        let win_rate = if !closed_trades.is_empty() {
            winning.len() as f64 / closed_trades.len() as f64 * 100.0
        } else {
            f64::NAN
        };

        let avg_win = if !winning.is_empty() {
            winning.iter().filter_map(|t| t.net_pnl()).sum::<f64>() / winning.len() as f64
        } else {
            f64::NAN
        };

        let avg_loss = if !losing.is_empty() {
            losing.iter().filter_map(|t| t.net_pnl()).sum::<f64>() / losing.len() as f64
        } else {
            f64::NAN
        };

        let gross_wins: f64 = winning.iter().filter_map(|t| t.net_pnl()).sum();
        let gross_losses: f64 = losing
            .iter()
            .filter_map(|t| t.net_pnl())
            .map(|p| p.abs())
            .sum();
        // Per spec: Zero trades returns NaN for trade-based metrics
        let profit_factor = if closed_trades.is_empty() {
            f64::NAN
        } else if gross_losses > 0.0 {
            gross_wins / gross_losses
        } else if gross_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown_pct = equity_curve
            .iter()
            .map(|e| e.drawdown_pct)
            .fold(0.0_f64, |a, b| a.max(b));

        // Calculate returns for Sharpe/Sortino
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect();

        // Determine annualization factor based on data frequency
        let frequency = self
            .config
            .data_frequency
            .unwrap_or_else(|| DataFrequency::detect(bars));
        let trading_24h = self
            .config
            .trading_hours_24
            .unwrap_or_else(|| DataFrequency::is_likely_crypto(bars));
        let annualization_factor = frequency.annualization_factor(trading_24h);

        // Log frequency detection for transparency
        debug!(
            "Using {} data frequency with annualization factor {:.1} (24/7: {})",
            frequency, annualization_factor, trading_24h
        );

        let sharpe_ratio =
            calculate_sharpe(&returns, annualization_factor, self.config.risk_free_rate);
        let sortino_ratio =
            calculate_sortino(&returns, annualization_factor, self.config.risk_free_rate);

        let calmar_ratio = if max_drawdown_pct > 0.0 {
            annual_return_pct / max_drawdown_pct
        } else if annual_return_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Capture metadata for reproducibility
        let experiment_id = generate_experiment_id();
        let git_info = GitInfo::capture();
        let config_hash = compute_config_hash(&self.config);
        let data_checksums = self.data.checksums().clone();

        BacktestResult {
            strategy_name: strategy_name.to_string(),
            symbols: vec![symbol.to_string()],
            config: self.config.clone(),
            initial_capital: self.config.initial_capital,
            final_equity,
            total_return_pct,
            annual_return_pct,
            trading_days: bars.len(),
            total_trades: closed_trades.len(),
            winning_trades: winning.len(),
            losing_trades: losing.len(),
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            max_drawdown_pct,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            trades: trades.to_vec(),
            equity_curve: equity_curve.to_vec(),
            start_time,
            end_time,
            experiment_id,
            git_info,
            config_hash,
            data_checksums,
            seed: self.config.seed,
        }
    }

    /// Run parameter optimization in parallel.
    pub fn optimize<P, F>(
        &self,
        symbol: &str,
        params: Vec<P>,
        strategy_factory: F,
    ) -> Result<Vec<(P, BacktestResult)>>
    where
        P: Clone + Send + Sync,
        F: Fn(&P) -> Box<dyn Strategy> + Send + Sync,
    {
        let bars = self
            .data
            .get(symbol)
            .ok_or_else(|| BacktestError::DataError(format!("No data for symbol: {}", symbol)))?
            .clone();

        let config = self.config.clone();

        let results: Vec<(P, BacktestResult)> = params
            .par_iter()
            .filter_map(|param| {
                let mut strategy = strategy_factory(param);
                let engine = Engine::new(config.clone());
                let mut engine = engine;
                engine.add_data(symbol.to_string(), bars.clone());

                // Disable progress for parallel runs
                let mut config = engine.config.clone();
                config.show_progress = false;
                engine.set_config(config);

                match engine.run(strategy.as_mut(), symbol) {
                    Ok(result) => Some((param.clone(), result)),
                    Err(e) => {
                        warn!("Optimization run failed: {}", e);
                        None
                    }
                }
            })
            .collect();

        Ok(results)
    }

    fn pending_expiry_for(&self, bars: &[Bar], index: usize) -> Option<DateTime<Utc>> {
        self.config.limit_order_ttl_bars.and_then(|ttl| {
            if ttl == 0 || bars.is_empty() {
                None
            } else {
                let target = (index + ttl).min(bars.len().saturating_sub(1));
                bars.get(target).map(|b| b.timestamp)
            }
        })
    }
}

/// Calculate Sharpe ratio from returns.
///
/// Per spec (performance-metrics.md):
/// - sharpe = (mean(returns) - risk_free_rate) / std(returns) × √periods_per_year
/// - Empty returns → NaN
/// - Zero volatility with positive excess mean → inf
/// - Zero volatility with negative excess mean → -inf
/// - Zero volatility with zero excess mean → NaN
///
/// # Arguments
/// * `returns` - Array of period returns (e.g., daily returns)
/// * `annualization_factor` - Number of periods per year (e.g., 252 for daily)
/// * `annual_risk_free_rate` - Annual risk-free rate as decimal (e.g., 0.02 for 2%)
fn calculate_sharpe(returns: &[f64], annualization_factor: f64, annual_risk_free_rate: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }

    // Convert annual risk-free rate to per-period rate
    let period_risk_free_rate = annual_risk_free_rate / annualization_factor;

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_mean = mean - period_risk_free_rate;

    let variance: f64 =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 || std_dev.is_nan() {
        return if excess_mean > 0.0 {
            f64::INFINITY
        } else if excess_mean < 0.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }

    (excess_mean / std_dev) * annualization_factor.sqrt()
}

/// Calculate Sortino ratio from returns.
///
/// Per spec (performance-metrics.md):
/// - sortino = (mean(returns) - risk_free_rate) / downside_std × √periods_per_year
/// - Empty returns → NaN
/// - No downside returns with positive excess mean → inf
/// - No downside returns with non-positive excess mean → NaN
/// - Zero downside deviation with positive excess mean → inf
/// - Zero downside deviation with non-positive excess mean → NaN
///
/// # Arguments
/// * `returns` - Array of period returns (e.g., daily returns)
/// * `annualization_factor` - Number of periods per year (e.g., 252 for daily)
/// * `annual_risk_free_rate` - Annual risk-free rate as decimal (e.g., 0.02 for 2%)
fn calculate_sortino(
    returns: &[f64],
    annualization_factor: f64,
    annual_risk_free_rate: f64,
) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }

    // Convert annual risk-free rate to per-period rate
    let period_risk_free_rate = annual_risk_free_rate / annualization_factor;

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_mean = mean - period_risk_free_rate;

    // Only consider negative returns for downside deviation
    // Note: downside deviation is relative to 0 (or could be relative to target return)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside_returns.is_empty() {
        // No downside returns: if excess mean is positive, infinite Sortino; otherwise undefined
        return if excess_mean > 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }

    let downside_variance: f64 =
        downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
    let downside_dev = downside_variance.sqrt();

    if downside_dev == 0.0 || downside_dev.is_nan() {
        return if excess_mean > 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }

    (excess_mean / downside_dev) * annualization_factor.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;
    use chrono::{TimeZone, Utc};

    struct SimpleStrategy {
        buy_on_bar: usize,
        sell_on_bar: usize,
    }

    impl Strategy for SimpleStrategy {
        fn name(&self) -> &str {
            "SimpleStrategy"
        }

        fn on_bar(&mut self, ctx: &StrategyContext) -> Signal {
            if ctx.bar_index == self.buy_on_bar && ctx.is_flat() {
                Signal::Long
            } else if ctx.bar_index == self.sell_on_bar && ctx.is_long() {
                Signal::Exit
            } else {
                Signal::Hold
            }
        }
    }

    fn create_test_bars() -> Vec<Bar> {
        (0..20)
            .map(|i| {
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, i + 1, 0, 0, 0).unwrap(),
                    100.0 + i as f64,
                    105.0 + i as f64,
                    98.0 + i as f64,
                    102.0 + i as f64,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_engine_creation() {
        let engine = Engine::with_defaults();
        assert_eq!(engine.config.initial_capital, 100_000.0);
    }

    #[test]
    fn test_simple_backtest() {
        let mut config = BacktestConfig::default();
        config.show_progress = false;
        config.cost_model = CostModel::zero();

        let mut engine = Engine::new(config);
        engine.add_data("TEST", create_test_bars());

        let mut strategy = SimpleStrategy {
            buy_on_bar: 5,
            sell_on_bar: 15,
        };

        let result = engine.run(&mut strategy, "TEST").unwrap();

        assert_eq!(result.strategy_name, "SimpleStrategy");
        assert!(result.total_trades > 0);
        assert_eq!(result.trading_days, 20);
    }

    #[test]
    fn test_buffer_order_for_next_bar() {
        use crate::types::Signal;

        let mut config = BacktestConfig::default();
        config.limit_order_ttl_bars = Some(3);
        config.show_progress = false;
        let engine = Engine::new(config);
        let bars = create_test_bars();
        let mut pending = Vec::new();

        // Limit order should be buffered with TTL-based expiry
        let limit_order = Order::limit("TEST", Side::Buy, 10.0, 99.0, bars[0].timestamp);
        engine.buffer_order_for_next_bar(
            &mut pending,
            limit_order.clone(),
            Some(Signal::Long),
            &bars,
            0,
        );
        assert_eq!(pending.len(), 1);
        assert!((pending[0].remaining_quantity - 10.0).abs() < f64::EPSILON);
        assert!(pending[0].expires_at.is_some());
        assert!(matches!(pending[0].signal, Some(Signal::Long)));

        // Market order should be buffered with no expiry (executes next bar)
        let market = Order::market("TEST", Side::Buy, 5.0, bars[0].timestamp);
        engine.buffer_order_for_next_bar(&mut pending, market, Some(Signal::Long), &bars, 0);
        assert_eq!(pending.len(), 2);
        assert!((pending[1].remaining_quantity - 5.0).abs() < f64::EPSILON);
        assert!(pending[1].expires_at.is_none()); // Market orders have no expiry
    }

    #[test]
    fn test_sharpe_calculation() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0);
        // Should be a reasonable number
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_sharpe_empty_returns() {
        let returns: Vec<f64> = vec![];
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0);
        assert!(sharpe.is_nan());
    }

    #[test]
    fn test_sharpe_zero_volatility_positive_mean() {
        // All identical positive returns = zero volatility with positive mean → inf
        let returns = vec![0.01, 0.01, 0.01, 0.01];
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0);
        assert!(sharpe.is_infinite() && sharpe > 0.0);
    }

    #[test]
    fn test_sharpe_zero_volatility_negative_mean() {
        // All identical negative returns = zero volatility with negative mean → -inf
        let returns = vec![-0.01, -0.01, -0.01, -0.01];
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0);
        assert!(sharpe.is_infinite() && sharpe < 0.0);
    }

    #[test]
    fn test_sharpe_zero_volatility_zero_mean() {
        // All zero returns = zero volatility with zero mean → NaN
        let returns = vec![0.0, 0.0, 0.0, 0.0];
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0);
        assert!(sharpe.is_nan());
    }

    #[test]
    fn test_sharpe_with_risk_free_rate() {
        // Test that risk-free rate affects the Sharpe ratio
        let returns = vec![0.01, 0.02, 0.015, 0.005, 0.01, 0.02];
        let sharpe_no_rf = calculate_sharpe(&returns, 252.0, 0.0);
        let sharpe_with_rf = calculate_sharpe(&returns, 252.0, 0.02); // 2% annual risk-free rate
                                                                      // With positive risk-free rate, Sharpe should be lower
        assert!(sharpe_with_rf < sharpe_no_rf);
    }

    #[test]
    fn test_sharpe_risk_free_rate_conversion() {
        // Test the exact calculation with risk-free rate
        // Mean return = 0.01 per day
        // If risk-free rate = 2.52% annual (0.0252), that's 0.0001 per day (0.0252 / 252)
        // Excess return = 0.01 - 0.0001 = 0.0099
        let returns = vec![0.01, 0.01, 0.01, 0.01]; // Zero vol, mean = 0.01
        let sharpe = calculate_sharpe(&returns, 252.0, 0.0252);
        // Zero volatility with positive excess mean should still be inf
        assert!(sharpe.is_infinite() && sharpe > 0.0);
    }

    #[test]
    fn test_sortino_calculation() {
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let sortino = calculate_sortino(&returns, 252.0, 0.0);
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_sortino_empty_returns() {
        let returns: Vec<f64> = vec![];
        let sortino = calculate_sortino(&returns, 252.0, 0.0);
        assert!(sortino.is_nan());
    }

    #[test]
    fn test_sortino_no_downside_positive_mean() {
        // All positive returns = no downside returns with positive mean → inf
        let returns = vec![0.01, 0.02, 0.015, 0.005];
        let sortino = calculate_sortino(&returns, 252.0, 0.0);
        assert!(sortino.is_infinite() && sortino > 0.0);
    }

    #[test]
    fn test_sortino_no_downside_zero_mean() {
        // All zero returns = no downside returns with zero mean → NaN
        let returns = vec![0.0, 0.0, 0.0, 0.0];
        let sortino = calculate_sortino(&returns, 252.0, 0.0);
        assert!(sortino.is_nan());
    }

    #[test]
    fn test_sortino_with_risk_free_rate() {
        // Test that risk-free rate affects the Sortino ratio
        let returns = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02];
        let sortino_no_rf = calculate_sortino(&returns, 252.0, 0.0);
        let sortino_with_rf = calculate_sortino(&returns, 252.0, 0.02); // 2% annual risk-free rate
                                                                        // With positive risk-free rate, Sortino should be lower
        assert!(sortino_with_rf < sortino_no_rf);
    }
}
