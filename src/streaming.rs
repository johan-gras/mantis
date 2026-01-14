//! Streaming/incremental indicator calculations.
//!
//! This module provides streaming versions of technical indicators that
//! maintain state and update incrementally with each new data point,
//! avoiding recalculation of the entire history.
//!
//! # Example
//!
//! ```ignore
//! use mantis::streaming::{StreamingSMA, StreamingRSI, StreamingIndicator};
//!
//! let mut sma = StreamingSMA::new(20);
//! let mut rsi = StreamingRSI::new(14);
//!
//! for bar in bars {
//!     sma.update(bar.close);
//!     rsi.update(bar.close);
//!
//!     if let (Some(sma_val), Some(rsi_val)) = (sma.value(), rsi.value()) {
//!         // Use indicator values
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Trait for streaming indicators.
pub trait StreamingIndicator {
    /// Update indicator with new price.
    fn update(&mut self, price: f64);

    /// Get current indicator value.
    fn value(&self) -> Option<f64>;

    /// Check if indicator has warmed up.
    fn is_ready(&self) -> bool;

    /// Reset indicator state.
    fn reset(&mut self);

    /// Get warmup period.
    fn warmup_period(&self) -> usize;
}

/// Streaming Simple Moving Average.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingSMA {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl StreamingSMA {
    /// Create a new streaming SMA.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
        }
    }
}

impl StreamingIndicator for StreamingSMA {
    fn update(&mut self, price: f64) {
        self.buffer.push_back(price);
        self.sum += price;

        if self.buffer.len() > self.period {
            self.sum -= self.buffer.pop_front().unwrap();
        }
    }

    fn value(&self) -> Option<f64> {
        if self.buffer.len() >= self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    fn is_ready(&self) -> bool {
        self.buffer.len() >= self.period
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }

    fn warmup_period(&self) -> usize {
        self.period
    }
}

/// Streaming Exponential Moving Average.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingEMA {
    period: usize,
    multiplier: f64,
    ema: Option<f64>,
    count: usize,
}

impl StreamingEMA {
    /// Create a new streaming EMA.
    pub fn new(period: usize) -> Self {
        let multiplier = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            multiplier,
            ema: None,
            count: 0,
        }
    }
}

impl StreamingIndicator for StreamingEMA {
    fn update(&mut self, price: f64) {
        self.count += 1;

        match self.ema {
            Some(prev) => {
                self.ema = Some(price * self.multiplier + prev * (1.0 - self.multiplier));
            }
            None => {
                self.ema = Some(price);
            }
        }
    }

    fn value(&self) -> Option<f64> {
        if self.count >= self.period {
            self.ema
        } else {
            None
        }
    }

    fn is_ready(&self) -> bool {
        self.count >= self.period
    }

    fn reset(&mut self) {
        self.ema = None;
        self.count = 0;
    }

    fn warmup_period(&self) -> usize {
        self.period
    }
}

/// Streaming RSI (Relative Strength Index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingRSI {
    period: usize,
    prev_price: Option<f64>,
    avg_gain: f64,
    avg_loss: f64,
    count: usize,
    initial_gains: Vec<f64>,
    initial_losses: Vec<f64>,
}

impl StreamingRSI {
    /// Create a new streaming RSI.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_price: None,
            avg_gain: 0.0,
            avg_loss: 0.0,
            count: 0,
            initial_gains: Vec::with_capacity(period),
            initial_losses: Vec::with_capacity(period),
        }
    }
}

impl StreamingIndicator for StreamingRSI {
    fn update(&mut self, price: f64) {
        if let Some(prev) = self.prev_price {
            let change = price - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            self.count += 1;

            if self.count <= self.period {
                // Collecting initial period
                self.initial_gains.push(gain);
                self.initial_losses.push(loss);

                if self.count == self.period {
                    // Calculate initial averages
                    self.avg_gain = self.initial_gains.iter().sum::<f64>() / self.period as f64;
                    self.avg_loss = self.initial_losses.iter().sum::<f64>() / self.period as f64;
                }
            } else {
                // Smoothed average (Wilder's method)
                self.avg_gain =
                    (self.avg_gain * (self.period - 1) as f64 + gain) / self.period as f64;
                self.avg_loss =
                    (self.avg_loss * (self.period - 1) as f64 + loss) / self.period as f64;
            }
        }

        self.prev_price = Some(price);
    }

    fn value(&self) -> Option<f64> {
        if self.count >= self.period {
            if self.avg_loss == 0.0 {
                Some(100.0)
            } else {
                let rs = self.avg_gain / self.avg_loss;
                Some(100.0 - 100.0 / (1.0 + rs))
            }
        } else {
            None
        }
    }

    fn is_ready(&self) -> bool {
        self.count >= self.period
    }

    fn reset(&mut self) {
        self.prev_price = None;
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.count = 0;
        self.initial_gains.clear();
        self.initial_losses.clear();
    }

    fn warmup_period(&self) -> usize {
        self.period + 1 // +1 because we need prev_price
    }
}

/// Streaming MACD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingMACD {
    fast_ema: StreamingEMA,
    slow_ema: StreamingEMA,
    signal_ema: StreamingEMA,
    count: usize,
}

impl StreamingMACD {
    /// Create a new streaming MACD with default parameters (12, 26, 9).
    pub fn new() -> Self {
        Self::with_params(12, 26, 9)
    }

    /// Create with custom parameters.
    pub fn with_params(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_ema: StreamingEMA::new(fast),
            slow_ema: StreamingEMA::new(slow),
            signal_ema: StreamingEMA::new(signal),
            count: 0,
        }
    }

    /// Get MACD line value.
    pub fn macd(&self) -> Option<f64> {
        match (self.fast_ema.value(), self.slow_ema.value()) {
            (Some(fast), Some(slow)) => Some(fast - slow),
            _ => None,
        }
    }

    /// Get signal line value.
    pub fn signal(&self) -> Option<f64> {
        self.signal_ema.value()
    }

    /// Get histogram value.
    pub fn histogram(&self) -> Option<f64> {
        match (self.macd(), self.signal()) {
            (Some(macd), Some(signal)) => Some(macd - signal),
            _ => None,
        }
    }
}

impl Default for StreamingMACD {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingIndicator for StreamingMACD {
    fn update(&mut self, price: f64) {
        self.fast_ema.update(price);
        self.slow_ema.update(price);
        self.count += 1;

        // Update signal EMA with MACD value once available
        if let Some(macd) = self.macd() {
            self.signal_ema.update(macd);
        }
    }

    fn value(&self) -> Option<f64> {
        self.macd()
    }

    fn is_ready(&self) -> bool {
        self.slow_ema.is_ready() && self.signal_ema.is_ready()
    }

    fn reset(&mut self) {
        self.fast_ema.reset();
        self.slow_ema.reset();
        self.signal_ema.reset();
        self.count = 0;
    }

    fn warmup_period(&self) -> usize {
        26 + 9 // slow period + signal period
    }
}

/// Streaming Bollinger Bands.
#[derive(Debug, Clone)]
pub struct StreamingBollinger {
    period: usize,
    std_dev_mult: f64,
    buffer: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl StreamingBollinger {
    /// Create new Bollinger Bands indicator.
    pub fn new(period: usize, std_dev_mult: f64) -> Self {
        Self {
            period,
            std_dev_mult,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Get middle band (SMA).
    pub fn middle(&self) -> Option<f64> {
        if self.buffer.len() >= self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }

    /// Get standard deviation.
    pub fn std_dev(&self) -> Option<f64> {
        if self.buffer.len() >= self.period {
            let mean = self.sum / self.period as f64;
            let variance = self.sum_sq / self.period as f64 - mean * mean;
            Some(variance.max(0.0).sqrt())
        } else {
            None
        }
    }

    /// Get upper band.
    pub fn upper(&self) -> Option<f64> {
        match (self.middle(), self.std_dev()) {
            (Some(mid), Some(std)) => Some(mid + self.std_dev_mult * std),
            _ => None,
        }
    }

    /// Get lower band.
    pub fn lower(&self) -> Option<f64> {
        match (self.middle(), self.std_dev()) {
            (Some(mid), Some(std)) => Some(mid - self.std_dev_mult * std),
            _ => None,
        }
    }

    /// Get %B (position within bands).
    pub fn percent_b(&self, price: f64) -> Option<f64> {
        match (self.upper(), self.lower()) {
            (Some(upper), Some(lower)) => {
                if (upper - lower).abs() < f64::EPSILON {
                    Some(0.5)
                } else {
                    Some((price - lower) / (upper - lower))
                }
            }
            _ => None,
        }
    }

    /// Get bandwidth (normalized width).
    pub fn bandwidth(&self) -> Option<f64> {
        match (self.middle(), self.std_dev()) {
            (Some(mid), Some(std)) if mid > 0.0 => Some(4.0 * self.std_dev_mult * std / mid),
            _ => None,
        }
    }
}

impl StreamingIndicator for StreamingBollinger {
    fn update(&mut self, price: f64) {
        self.buffer.push_back(price);
        self.sum += price;
        self.sum_sq += price * price;

        if self.buffer.len() > self.period {
            let old = self.buffer.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }
    }

    fn value(&self) -> Option<f64> {
        self.middle()
    }

    fn is_ready(&self) -> bool {
        self.buffer.len() >= self.period
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    fn warmup_period(&self) -> usize {
        self.period
    }
}

/// Streaming ATR (Average True Range).
#[derive(Debug, Clone)]
pub struct StreamingATR {
    period: usize,
    prev_close: Option<f64>,
    atr: Option<f64>,
    count: usize,
    initial_trs: Vec<f64>,
}

impl StreamingATR {
    /// Create new ATR indicator.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prev_close: None,
            atr: None,
            count: 0,
            initial_trs: Vec::with_capacity(period),
        }
    }

    /// Update with OHLC data.
    pub fn update_ohlc(&mut self, high: f64, low: f64, close: f64) {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };

        self.count += 1;

        if self.count <= self.period {
            self.initial_trs.push(tr);
            if self.count == self.period {
                self.atr = Some(self.initial_trs.iter().sum::<f64>() / self.period as f64);
            }
        } else if let Some(prev_atr) = self.atr {
            // Wilder's smoothing
            self.atr = Some((prev_atr * (self.period - 1) as f64 + tr) / self.period as f64);
        }

        self.prev_close = Some(close);
    }
}

impl StreamingIndicator for StreamingATR {
    fn update(&mut self, price: f64) {
        // For single price updates, estimate TR from price change
        if let Some(prev) = self.prev_close {
            let change = (price - prev).abs();
            self.count += 1;

            if self.count <= self.period {
                self.initial_trs.push(change);
                if self.count == self.period {
                    self.atr = Some(self.initial_trs.iter().sum::<f64>() / self.period as f64);
                }
            } else if let Some(prev_atr) = self.atr {
                self.atr =
                    Some((prev_atr * (self.period - 1) as f64 + change) / self.period as f64);
            }
        }

        self.prev_close = Some(price);
    }

    fn value(&self) -> Option<f64> {
        self.atr
    }

    fn is_ready(&self) -> bool {
        self.count >= self.period
    }

    fn reset(&mut self) {
        self.prev_close = None;
        self.atr = None;
        self.count = 0;
        self.initial_trs.clear();
    }

    fn warmup_period(&self) -> usize {
        self.period
    }
}

/// Streaming standard deviation.
#[derive(Debug, Clone)]
pub struct StreamingStdDev {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl StreamingStdDev {
    /// Create new streaming standard deviation.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            buffer: VecDeque::with_capacity(period),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Get variance.
    pub fn variance(&self) -> Option<f64> {
        if self.buffer.len() >= self.period {
            let mean = self.sum / self.period as f64;
            Some((self.sum_sq / self.period as f64 - mean * mean).max(0.0))
        } else {
            None
        }
    }

    /// Get mean.
    pub fn mean(&self) -> Option<f64> {
        if self.buffer.len() >= self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }
}

impl StreamingIndicator for StreamingStdDev {
    fn update(&mut self, price: f64) {
        self.buffer.push_back(price);
        self.sum += price;
        self.sum_sq += price * price;

        if self.buffer.len() > self.period {
            let old = self.buffer.pop_front().unwrap();
            self.sum -= old;
            self.sum_sq -= old * old;
        }
    }

    fn value(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    fn is_ready(&self) -> bool {
        self.buffer.len() >= self.period
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    fn warmup_period(&self) -> usize {
        self.period
    }
}

/// Collection of streaming indicators for a single symbol.
#[derive(Debug)]
pub struct StreamingIndicatorSet {
    pub sma_short: StreamingSMA,
    pub sma_long: StreamingSMA,
    pub ema_short: StreamingEMA,
    pub ema_long: StreamingEMA,
    pub rsi: StreamingRSI,
    pub macd: StreamingMACD,
    pub bollinger: StreamingBollinger,
    pub atr: StreamingATR,
    pub std_dev: StreamingStdDev,
}

impl StreamingIndicatorSet {
    /// Create a new indicator set with common parameters.
    pub fn new() -> Self {
        Self {
            sma_short: StreamingSMA::new(10),
            sma_long: StreamingSMA::new(20),
            ema_short: StreamingEMA::new(12),
            ema_long: StreamingEMA::new(26),
            rsi: StreamingRSI::new(14),
            macd: StreamingMACD::new(),
            bollinger: StreamingBollinger::new(20, 2.0),
            atr: StreamingATR::new(14),
            std_dev: StreamingStdDev::new(20),
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        sma_short: usize,
        sma_long: usize,
        rsi_period: usize,
        bb_period: usize,
    ) -> Self {
        Self {
            sma_short: StreamingSMA::new(sma_short),
            sma_long: StreamingSMA::new(sma_long),
            ema_short: StreamingEMA::new(12),
            ema_long: StreamingEMA::new(26),
            rsi: StreamingRSI::new(rsi_period),
            macd: StreamingMACD::new(),
            bollinger: StreamingBollinger::new(bb_period, 2.0),
            atr: StreamingATR::new(14),
            std_dev: StreamingStdDev::new(bb_period),
        }
    }

    /// Update all indicators with new price.
    pub fn update(&mut self, price: f64) {
        self.sma_short.update(price);
        self.sma_long.update(price);
        self.ema_short.update(price);
        self.ema_long.update(price);
        self.rsi.update(price);
        self.macd.update(price);
        self.bollinger.update(price);
        self.atr.update(price);
        self.std_dev.update(price);
    }

    /// Update with OHLC data for more accurate ATR.
    pub fn update_ohlc(&mut self, high: f64, low: f64, close: f64) {
        self.sma_short.update(close);
        self.sma_long.update(close);
        self.ema_short.update(close);
        self.ema_long.update(close);
        self.rsi.update(close);
        self.macd.update(close);
        self.bollinger.update(close);
        self.atr.update_ohlc(high, low, close);
        self.std_dev.update(close);
    }

    /// Check if all indicators are ready.
    pub fn is_ready(&self) -> bool {
        self.sma_long.is_ready()
            && self.rsi.is_ready()
            && self.macd.is_ready()
            && self.bollinger.is_ready()
    }

    /// Reset all indicators.
    pub fn reset(&mut self) {
        self.sma_short.reset();
        self.sma_long.reset();
        self.ema_short.reset();
        self.ema_long.reset();
        self.rsi.reset();
        self.macd.reset();
        self.bollinger.reset();
        self.atr.reset();
        self.std_dev.reset();
    }

    /// Get warmup period (max of all indicators).
    pub fn warmup_period(&self) -> usize {
        self.sma_long
            .warmup_period()
            .max(self.rsi.warmup_period())
            .max(self.macd.warmup_period())
            .max(self.bollinger.warmup_period())
    }
}

impl Default for StreamingIndicatorSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_prices(count: usize) -> Vec<f64> {
        (0..count)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.02)
            .collect()
    }

    #[test]
    fn test_streaming_sma() {
        let mut sma = StreamingSMA::new(5);
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        for (i, &price) in prices.iter().enumerate() {
            sma.update(price);
            if i < 4 {
                assert!(!sma.is_ready());
                assert!(sma.value().is_none());
            } else {
                assert!(sma.is_ready());
                assert!(sma.value().is_some());
            }
        }

        // Last 5 values: 3, 4, 5, 6, 7 -> average = 5.0
        assert!((sma.value().unwrap() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_streaming_ema() {
        let mut ema = StreamingEMA::new(10);
        let prices = generate_prices(20);

        for &price in &prices {
            ema.update(price);
        }

        assert!(ema.is_ready());
        let value = ema.value().unwrap();
        assert!(value > 0.0);
    }

    #[test]
    fn test_streaming_rsi() {
        let mut rsi = StreamingRSI::new(14);
        let prices = generate_prices(50);

        for &price in &prices {
            rsi.update(price);
        }

        assert!(rsi.is_ready());
        let value = rsi.value().unwrap();
        assert!(value >= 0.0 && value <= 100.0);
    }

    #[test]
    fn test_streaming_rsi_overbought() {
        let mut rsi = StreamingRSI::new(14);

        // Consistently rising prices should give high RSI
        for i in 0..30 {
            rsi.update(100.0 + i as f64);
        }

        assert!(rsi.is_ready());
        let value = rsi.value().unwrap();
        assert!(value > 70.0); // Should be overbought
    }

    #[test]
    fn test_streaming_macd() {
        let mut macd = StreamingMACD::new();
        let prices = generate_prices(50);

        for &price in &prices {
            macd.update(price);
        }

        assert!(macd.is_ready());
        assert!(macd.macd().is_some());
        assert!(macd.signal().is_some());
        assert!(macd.histogram().is_some());
    }

    #[test]
    fn test_streaming_bollinger() {
        let mut bb = StreamingBollinger::new(20, 2.0);
        let prices = generate_prices(30);

        for &price in &prices {
            bb.update(price);
        }

        assert!(bb.is_ready());

        let upper = bb.upper().unwrap();
        let middle = bb.middle().unwrap();
        let lower = bb.lower().unwrap();

        assert!(upper > middle);
        assert!(middle > lower);
    }

    #[test]
    fn test_streaming_atr() {
        let mut atr = StreamingATR::new(14);

        for i in 0..30 {
            let base = 100.0 + i as f64 * 0.1;
            atr.update_ohlc(base + 1.0, base - 1.0, base);
        }

        assert!(atr.is_ready());
        let value = atr.value().unwrap();
        assert!(value > 0.0);
    }

    #[test]
    fn test_streaming_std_dev() {
        let mut std_dev = StreamingStdDev::new(10);

        // Update with same value - std dev should be 0
        for _ in 0..15 {
            std_dev.update(100.0);
        }

        assert!(std_dev.is_ready());
        assert!((std_dev.value().unwrap() - 0.0).abs() < 0.001);

        // Reset and update with varying values
        std_dev.reset();
        for i in 0..15 {
            std_dev.update(100.0 + i as f64);
        }

        assert!(std_dev.value().unwrap() > 0.0);
    }

    #[test]
    fn test_indicator_set() {
        let mut set = StreamingIndicatorSet::new();
        let prices = generate_prices(100);

        for &price in &prices {
            set.update(price);
        }

        assert!(set.is_ready());
        assert!(set.sma_short.value().is_some());
        assert!(set.sma_long.value().is_some());
        assert!(set.rsi.value().is_some());
        assert!(set.macd.macd().is_some());
        assert!(set.bollinger.middle().is_some());
    }

    #[test]
    fn test_indicator_reset() {
        let mut sma = StreamingSMA::new(5);

        for i in 0..10 {
            sma.update(i as f64);
        }

        assert!(sma.is_ready());

        sma.reset();

        assert!(!sma.is_ready());
        assert!(sma.value().is_none());
    }

    #[test]
    fn test_bollinger_percent_b() {
        let mut bb = StreamingBollinger::new(20, 2.0);

        for i in 0..25 {
            bb.update(100.0 + (i as f64 * 0.3).sin());
        }

        assert!(bb.is_ready());

        let middle = bb.middle().unwrap();
        let percent_b = bb.percent_b(middle).unwrap();

        // At middle band, %B should be around 0.5
        assert!((percent_b - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_bollinger_bandwidth() {
        let mut bb = StreamingBollinger::new(20, 2.0);

        for i in 0..25 {
            bb.update(100.0 + i as f64 * 0.1);
        }

        assert!(bb.is_ready());
        let bandwidth = bb.bandwidth().unwrap();
        assert!(bandwidth > 0.0);
    }
}
