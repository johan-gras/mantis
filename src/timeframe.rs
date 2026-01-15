//! Multi-timeframe management for strategies.
//!
//! This module provides the `TimeframeManager` which maintains multiple resampled
//! bar series at different timeframes, enabling strategies to access data at
//! multiple time scales simultaneously (e.g., daily trend + hourly entries).

use crate::data::{resample, ResampleInterval};
use crate::types::Bar;
use std::collections::HashMap;
use std::sync::Arc;

/// Manages multiple timeframes for strategy access.
///
/// The manager maintains a base (highest frequency) bar series and lazily
/// resamples to coarser timeframes on demand. All timeframes are automatically
/// aligned to the current base bar timestamp.
#[derive(Debug, Clone)]
pub struct TimeframeManager {
    /// Base (highest frequency) bars
    base_bars: Arc<Vec<Bar>>,
    /// Cached resampled bars: interval -> bars
    timeframes: HashMap<ResampleInterval, Arc<Vec<Bar>>>,
}

impl TimeframeManager {
    /// Create a new manager from base bars.
    ///
    /// # Arguments
    /// * `bars` - The base (highest frequency) bars
    ///
    /// # Example
    /// ```
    /// use mantis::timeframe::TimeframeManager;
    /// use mantis::types::Bar;
    /// use chrono::Utc;
    ///
    /// let bars = vec![/* minute bars */];
    /// let manager = TimeframeManager::new(bars);
    /// ```
    pub fn new(bars: Vec<Bar>) -> Self {
        Self {
            base_bars: Arc::new(bars),
            timeframes: HashMap::new(),
        }
    }

    /// Request a specific timeframe, computing it lazily if needed.
    ///
    /// # Arguments
    /// * `interval` - The timeframe to request
    ///
    /// # Returns
    /// Ok(()) if successful, Err if resampling fails
    ///
    /// # Example
    /// ```
    /// use mantis::data::ResampleInterval;
    /// # use mantis::timeframe::TimeframeManager;
    /// # use mantis::types::Bar;
    /// # let bars = vec![];
    /// # let mut manager = TimeframeManager::new(bars);
    ///
    /// // Request hourly bars (will resample from base if not cached)
    /// manager.request_timeframe(ResampleInterval::Hour(1))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn request_timeframe(&mut self, interval: ResampleInterval) -> crate::error::Result<()> {
        // Skip if already cached
        if self.timeframes.contains_key(&interval) {
            return Ok(());
        }

        // Resample from base bars
        let resampled = resample(&self.base_bars, interval);
        self.timeframes.insert(interval, Arc::new(resampled));

        Ok(())
    }

    /// Get bars at a specific timeframe up to a base bar index.
    ///
    /// Returns all bars at the requested timeframe whose timestamps are
    /// less than or equal to the timestamp of `base_bars[base_index]`.
    ///
    /// # Arguments
    /// * `interval` - The timeframe to query
    /// * `base_index` - The current index in the base bar series
    ///
    /// # Returns
    /// Slice of bars at the requested timeframe, or None if timeframe not requested
    ///
    /// # Example
    /// ```
    /// use mantis::data::ResampleInterval;
    /// # use mantis::timeframe::TimeframeManager;
    /// # use mantis::types::Bar;
    /// # use chrono::Utc;
    /// # let bars = vec![];
    /// # let mut manager = TimeframeManager::new(bars);
    /// # manager.request_timeframe(ResampleInterval::Hour(1))?;
    ///
    /// // Get hourly bars up to current base bar
    /// if let Some(hourly_bars) = manager.get_bars_up_to(ResampleInterval::Hour(1), 100) {
    ///     println!("Found {} hourly bars", hourly_bars.len());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_bars_up_to(&self, interval: ResampleInterval, base_index: usize) -> Option<&[Bar]> {
        let resampled_bars = self.timeframes.get(&interval)?;

        // Get timestamp of current base bar
        let current_timestamp = self.base_bars.get(base_index)?.timestamp;

        // Find last resampled bar where timestamp <= current_timestamp
        let end_index = resampled_bars
            .iter()
            .position(|bar| bar.timestamp > current_timestamp)
            .unwrap_or(resampled_bars.len());

        if end_index == 0 {
            None
        } else {
            Some(&resampled_bars[..end_index])
        }
    }

    /// Get the current bar at a specific timeframe.
    ///
    /// Returns the most recent bar at the requested timeframe whose timestamp
    /// is less than or equal to the current base bar timestamp.
    ///
    /// # Arguments
    /// * `interval` - The timeframe to query
    /// * `base_index` - The current index in the base bar series
    ///
    /// # Returns
    /// Reference to the current bar, or None if not available
    ///
    /// # Example
    /// ```
    /// use mantis::data::ResampleInterval;
    /// # use mantis::timeframe::TimeframeManager;
    /// # use mantis::types::Bar;
    /// # let bars = vec![];
    /// # let mut manager = TimeframeManager::new(bars);
    /// # manager.request_timeframe(ResampleInterval::Hour(1))?;
    ///
    /// // Get current hourly bar
    /// if let Some(hourly_bar) = manager.get_current_bar(ResampleInterval::Hour(1), 100) {
    ///     println!("Current hourly close: {}", hourly_bar.close);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_current_bar(&self, interval: ResampleInterval, base_index: usize) -> Option<&Bar> {
        let bars = self.get_bars_up_to(interval, base_index)?;
        bars.last()
    }

    /// Get a bar at a specific timeframe with lookback.
    ///
    /// # Arguments
    /// * `interval` - The timeframe to query
    /// * `base_index` - The current index in the base bar series
    /// * `lookback` - How many bars to look back (0 = current, 1 = previous, etc.)
    ///
    /// # Returns
    /// Reference to the bar, or None if not available
    ///
    /// # Example
    /// ```
    /// use mantis::data::ResampleInterval;
    /// # use mantis::timeframe::TimeframeManager;
    /// # use mantis::types::Bar;
    /// # let bars = vec![];
    /// # let mut manager = TimeframeManager::new(bars);
    /// # manager.request_timeframe(ResampleInterval::Hour(1))?;
    ///
    /// // Get hourly bar from 2 hours ago
    /// if let Some(bar) = manager.get_bar_at(ResampleInterval::Hour(1), 100, 2) {
    ///     println!("Bar 2 hours ago: close = {}", bar.close);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_bar_at(
        &self,
        interval: ResampleInterval,
        base_index: usize,
        lookback: usize,
    ) -> Option<&Bar> {
        let bars = self.get_bars_up_to(interval, base_index)?;
        if lookback < bars.len() {
            Some(&bars[bars.len() - 1 - lookback])
        } else {
            None
        }
    }

    /// Get the base (highest frequency) bars.
    pub fn base_bars(&self) -> &[Bar] {
        &self.base_bars
    }

    /// Get the number of bars at a specific timeframe.
    pub fn timeframe_len(&self, interval: ResampleInterval) -> Option<usize> {
        self.timeframes.get(&interval).map(|bars| bars.len())
    }

    /// Check if a timeframe has been requested and cached.
    pub fn has_timeframe(&self, interval: ResampleInterval) -> bool {
        self.timeframes.contains_key(&interval)
    }

    /// Get all registered timeframes.
    pub fn registered_timeframes(&self) -> Vec<ResampleInterval> {
        self.timeframes.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    fn create_minute_bars(count: usize) -> Vec<Bar> {
        let base_time = Utc.with_ymd_and_hms(2024, 1, 1, 9, 0, 0).unwrap();
        (0..count)
            .map(|i| Bar {
                timestamp: base_time + chrono::Duration::minutes(i as i64),
                open: 100.0 + i as f64,
                high: 101.0 + i as f64,
                low: 99.0 + i as f64,
                close: 100.5 + i as f64,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_timeframe_manager_creation() {
        let bars = create_minute_bars(100);
        let manager = TimeframeManager::new(bars.clone());
        assert_eq!(manager.base_bars().len(), 100);
        assert_eq!(manager.registered_timeframes().len(), 0);
    }

    #[test]
    fn test_request_timeframe() {
        let bars = create_minute_bars(60);
        let mut manager = TimeframeManager::new(bars);

        // Request 5-minute timeframe
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();
        assert!(manager.has_timeframe(ResampleInterval::Minute(5)));

        // Should have 12 five-minute bars from 60 one-minute bars
        assert_eq!(
            manager.timeframe_len(ResampleInterval::Minute(5)),
            Some(12)
        );
    }

    #[test]
    fn test_get_bars_up_to() {
        let bars = create_minute_bars(60);
        let mut manager = TimeframeManager::new(bars);
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();

        // At base_index=9 (10th minute), we should have 2 complete 5-minute bars
        let bars_at_10 = manager
            .get_bars_up_to(ResampleInterval::Minute(5), 9)
            .unwrap();
        assert_eq!(bars_at_10.len(), 2);

        // At base_index=29 (30th minute), we should have 6 complete 5-minute bars
        let bars_at_30 = manager
            .get_bars_up_to(ResampleInterval::Minute(5), 29)
            .unwrap();
        assert_eq!(bars_at_30.len(), 6);
    }

    #[test]
    fn test_get_current_bar() {
        let bars = create_minute_bars(60);
        let mut manager = TimeframeManager::new(bars);
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();

        // Get current 5-minute bar at base_index=14
        let current = manager
            .get_current_bar(ResampleInterval::Minute(5), 14)
            .unwrap();

        // Should be the 3rd five-minute bar (bars 10-14)
        assert_eq!(current.timestamp, Utc.with_ymd_and_hms(2024, 1, 1, 9, 10, 0).unwrap());
    }

    #[test]
    fn test_get_bar_at_with_lookback() {
        let bars = create_minute_bars(60);
        let mut manager = TimeframeManager::new(bars);
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();

        // At base_index=29, lookback=1 should give us the previous 5-minute bar
        let prev_bar = manager
            .get_bar_at(ResampleInterval::Minute(5), 29, 1)
            .unwrap();

        // At index 29 we have 6 bars, so lookback=1 should give us bar[4] (the 5th bar)
        assert_eq!(prev_bar.timestamp, Utc.with_ymd_and_hms(2024, 1, 1, 9, 20, 0).unwrap());
    }

    #[test]
    fn test_multiple_timeframes() {
        let bars = create_minute_bars(120);
        let mut manager = TimeframeManager::new(bars);

        // Request multiple timeframes
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();
        manager
            .request_timeframe(ResampleInterval::Minute(15))
            .unwrap();
        manager.request_timeframe(ResampleInterval::Hour(1)).unwrap();

        assert_eq!(manager.registered_timeframes().len(), 3);

        // Verify counts at base_index=119 (all bars)
        assert_eq!(
            manager.timeframe_len(ResampleInterval::Minute(5)),
            Some(24)
        );
        assert_eq!(
            manager.timeframe_len(ResampleInterval::Minute(15)),
            Some(8)
        );
        assert_eq!(manager.timeframe_len(ResampleInterval::Hour(1)), Some(2));
    }

    #[test]
    fn test_lazy_evaluation() {
        let bars = create_minute_bars(60);
        let mut manager = TimeframeManager::new(bars);

        // Initially, no timeframes cached
        assert!(!manager.has_timeframe(ResampleInterval::Minute(5)));

        // Request timeframe (lazy compute)
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();

        // Now it's cached
        assert!(manager.has_timeframe(ResampleInterval::Minute(5)));

        // Requesting again should be a no-op
        manager
            .request_timeframe(ResampleInterval::Minute(5))
            .unwrap();
        assert_eq!(manager.registered_timeframes().len(), 1);
    }

    #[test]
    fn test_none_when_timeframe_not_requested() {
        let bars = create_minute_bars(60);
        let manager = TimeframeManager::new(bars);

        // Should return None if timeframe not requested
        assert!(manager
            .get_bars_up_to(ResampleInterval::Minute(5), 10)
            .is_none());
        assert!(manager
            .get_current_bar(ResampleInterval::Minute(5), 10)
            .is_none());
    }
}
