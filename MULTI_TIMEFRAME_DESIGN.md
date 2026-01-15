# Multi-Timeframe Strategy Interface Design

## Goal
Enable strategies to access bars at multiple timeframes simultaneously (e.g., daily trend + hourly entries).

## Current State
- ✅ Resampling function works (`resample()` in `data.rs`)
- ✅ CLI resampling tool
- ❌ Strategies can only access ONE timeframe (via `ctx.bars`)
- ❌ No automatic alignment or multi-timeframe management

## Architecture

### 1. TimeframeManager

New struct to maintain multiple resampled bar series:

```rust
pub struct TimeframeManager {
    /// Base (highest frequency) bars
    base_bars: Arc<Vec<Bar>>,
    /// Base timeframe interval (e.g., Minute(1))
    base_interval: ResampleInterval,
    /// Cached resampled bars: interval -> bars
    timeframes: HashMap<ResampleInterval, Arc<Vec<Bar>>>,
    /// Current base bar index
    current_base_index: usize,
}
```

**Key Methods:**
- `new(bars: Vec<Bar>, base_interval: ResampleInterval)` - Create manager
- `request_timeframe(&mut self, interval: ResampleInterval)` - Lazy compute timeframe
- `get_bars_up_to(&self, interval: ResampleInterval, base_index: usize) -> &[Bar]` - Get aligned bars
- `get_current_bar(&self, interval: ResampleInterval, base_index: usize) -> Option<&Bar>` - Get current bar at timeframe

### 2. Extended StrategyContext

Add multi-timeframe access while maintaining backward compatibility:

```rust
pub struct StrategyContext<'a> {
    // Existing fields (unchanged for compatibility)
    pub bar_index: usize,
    pub bars: &'a [Bar],  // Base timeframe bars
    pub position: f64,
    pub cash: f64,
    pub equity: f64,
    pub symbol: &'a str,
    pub volume_profile: Option<VolumeProfile>,

    // New: optional multi-timeframe manager
    timeframe_manager: Option<&'a TimeframeManager>,
}

impl<'a> StrategyContext<'a> {
    /// Get bars at a specific timeframe (up to current timestamp)
    pub fn bars_at(&self, interval: ResampleInterval) -> Option<&[Bar]>

    /// Get current bar at a specific timeframe
    pub fn current_bar_at(&self, interval: ResampleInterval) -> Option<&Bar>

    /// Get historical bar at timeframe with lookback
    pub fn bar_at_timeframe(&self, interval: ResampleInterval, lookback: usize) -> Option<&Bar>
}
```

### 3. Engine Integration

Modify `Engine::run()` to:
1. Create `TimeframeManager` if strategy requests it
2. Pass manager reference in `StrategyContext`
3. Update manager each iteration

**Strategy Registration:**
```rust
pub trait Strategy {
    // New optional method
    fn requested_timeframes(&self) -> Vec<ResampleInterval> {
        vec![]  // Default: no additional timeframes
    }
}
```

## Alignment Logic

**Temporal Alignment:**
- At base bar index `i`, provide all resampled bars up to timestamp `base_bars[i].timestamp`
- Higher timeframes may have fewer bars (e.g., 1000 1m bars → ~16 1h bars)
- Current resampled bar may be "partial" (not yet complete)

**Example:**
```
Base (1m): [09:00, 09:01, 09:02, 09:03, 09:04, 09:05]
5m bars:   [09:00 (complete), 09:05 (forming)]

At base_index=3 (09:03):
- ctx.bars_at(Minute(5)) returns [09:00 bar] (complete 5m bar)
- ctx.current_bar_at(Minute(5)) returns partial 09:00-09:03 bar

At base_index=5 (09:05):
- ctx.bars_at(Minute(5)) returns [09:00 bar (complete), 09:05 bar (just started)]
```

## Implementation Plan

1. ✅ Design document (this file)
2. Create `TimeframeManager` struct in new `src/timeframe.rs`
3. Extend `StrategyContext` with optional timeframe manager
4. Add `requested_timeframes()` method to `Strategy` trait
5. Modify `Engine::run()` to create and use `TimeframeManager`
6. Add comprehensive tests
7. Create example strategy using multi-timeframe
8. Update documentation

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing strategies use `ctx.bars` (unchanged)
- New strategies opt-in by implementing `requested_timeframes()`
- `timeframe_manager` is `Option<&TimeframeManager>` - None for single-timeframe

## Testing Strategy

1. Unit tests for `TimeframeManager` alignment logic
2. Integration test: strategy using 1m + 5m + 1h bars
3. Verify correct bar counts at each timeframe
4. Test partial bar handling
5. Test with different base frequencies

## Performance

- **Lazy evaluation**: Only resample requested timeframes
- **Caching**: Resampled bars cached, not recomputed each iteration
- **Arc<Vec<Bar>>**: Shared ownership, no clones
- **Target overhead**: <10% vs single-timeframe backtest
