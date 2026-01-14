//! Data loading and management for the backtest engine.

use crate::error::{BacktestError, Result};
use crate::types::Bar;
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use tracing::{debug, info, warn};

/// Raw CSV row with flexible date parsing.
#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(
        alias = "Date",
        alias = "date",
        alias = "Timestamp",
        alias = "timestamp",
        alias = "Time",
        alias = "time",
        alias = "datetime"
    )]
    date: String,
    #[serde(alias = "Open", alias = "open", alias = "o")]
    open: f64,
    #[serde(alias = "High", alias = "high", alias = "h")]
    high: f64,
    #[serde(alias = "Low", alias = "low", alias = "l")]
    low: f64,
    #[serde(alias = "Close", alias = "close", alias = "c", alias = "Adj Close")]
    close: f64,
    #[serde(alias = "Volume", alias = "volume", alias = "v", default)]
    volume: f64,
}

/// Data source configuration.
#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Date format string for parsing (e.g., "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S").
    pub date_format: Option<String>,
    /// Whether the CSV has headers.
    pub has_headers: bool,
    /// CSV delimiter character.
    pub delimiter: u8,
    /// Skip invalid rows instead of failing.
    pub skip_invalid: bool,
    /// Validate bar data (high >= low, etc.).
    pub validate_bars: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            date_format: None,
            has_headers: true,
            delimiter: b',',
            skip_invalid: true,
            validate_bars: true,
        }
    }
}

/// Parse a date string with multiple format attempts.
fn parse_datetime(s: &str, format: Option<&str>) -> Result<DateTime<Utc>> {
    // Try explicit format first if provided
    if let Some(fmt) = format {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&dt));
        }
        if let Ok(d) = NaiveDate::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()));
        }
    }

    // Try common datetime formats
    let datetime_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%.fZ",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    ];

    for fmt in &datetime_formats {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&dt));
        }
    }

    // Try date-only formats
    let date_formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"];

    for fmt in &date_formats {
        if let Ok(d) = NaiveDate::parse_from_str(s, fmt) {
            return Ok(Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()));
        }
    }

    // Try parsing as Unix timestamp
    if let Ok(ts) = s.parse::<i64>() {
        if let Some(dt) = DateTime::from_timestamp(ts, 0) {
            return Ok(dt);
        }
    }

    Err(BacktestError::DataError(format!(
        "Could not parse date: '{}'",
        s
    )))
}

/// Load OHLCV data from a CSV file.
pub fn load_csv(path: impl AsRef<Path>, config: &DataConfig) -> Result<Vec<Bar>> {
    let path = path.as_ref();
    info!("Loading data from: {}", path.display());

    let mut reader = ReaderBuilder::new()
        .has_headers(config.has_headers)
        .delimiter(config.delimiter)
        .flexible(true)
        .from_path(path)?;

    let mut bars = Vec::new();
    let mut skipped = 0;
    let mut row_num = 0;

    for result in reader.deserialize() {
        row_num += 1;
        let row: CsvRow = match result {
            Ok(r) => r,
            Err(e) => {
                if config.skip_invalid {
                    debug!("Skipping row {}: {}", row_num, e);
                    skipped += 1;
                    continue;
                } else {
                    return Err(BacktestError::CsvError(e));
                }
            }
        };

        let timestamp = match parse_datetime(&row.date, config.date_format.as_deref()) {
            Ok(ts) => ts,
            Err(e) => {
                if config.skip_invalid {
                    debug!("Skipping row {} due to date parse error: {}", row_num, e);
                    skipped += 1;
                    continue;
                } else {
                    return Err(e);
                }
            }
        };

        let bar = Bar::new(
            timestamp, row.open, row.high, row.low, row.close, row.volume,
        );

        if config.validate_bars && !bar.validate() {
            if config.skip_invalid {
                debug!(
                    "Skipping row {} due to invalid bar data: {:?}",
                    row_num, bar
                );
                skipped += 1;
                continue;
            } else {
                return Err(BacktestError::DataError(format!(
                    "Invalid bar data at row {}: {:?}",
                    row_num, bar
                )));
            }
        }

        bars.push(bar);
    }

    if skipped > 0 {
        warn!("Skipped {} invalid rows", skipped);
    }

    // Sort by timestamp
    bars.sort_by_key(|b| b.timestamp);

    // Check for duplicates
    let original_len = bars.len();
    bars.dedup_by_key(|b| b.timestamp);
    if bars.len() < original_len {
        warn!("Removed {} duplicate timestamps", original_len - bars.len());
    }

    info!(
        "Loaded {} bars from {} to {}",
        bars.len(),
        bars.first()
            .map(|b| b.timestamp.to_string())
            .unwrap_or_default(),
        bars.last()
            .map(|b| b.timestamp.to_string())
            .unwrap_or_default()
    );

    if bars.is_empty() {
        return Err(BacktestError::NoData);
    }

    Ok(bars)
}

/// Load OHLCV data from a Parquet file.
///
/// Supports multiple column naming conventions:
/// - Date/timestamp columns: "timestamp", "date", "time", "datetime", "Date", "Timestamp"
/// - OHLCV columns: standard variations (Open/open/o, High/high/h, etc.)
///
/// Timestamp handling:
/// - Arrow Timestamp types (milliseconds, microseconds, nanoseconds)
/// - Unix timestamps (milliseconds or seconds as Int64)
/// - ISO 8601 strings
pub fn load_parquet(path: impl AsRef<Path>, config: &DataConfig) -> Result<Vec<Bar>> {
    use arrow::array::{Array, Float64Array, RecordBatchReader};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let path = path.as_ref();
    info!("Loading Parquet data from: {}", path.display());

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| BacktestError::DataError(format!("Failed to open parquet file: {}", e)))?;

    let reader = builder
        .build()
        .map_err(|e| BacktestError::DataError(format!("Failed to build parquet reader: {}", e)))?;

    let schema = reader.schema();
    debug!("Parquet schema: {:?}", schema);

    // Find column indices by matching various common names
    let timestamp_names = [
        "timestamp",
        "date",
        "time",
        "datetime",
        "Date",
        "Timestamp",
        "Time",
        "Datetime",
    ];
    let open_names = ["open", "Open", "o", "O"];
    let high_names = ["high", "High", "h", "H"];
    let low_names = ["low", "Low", "l", "L"];
    let close_names = [
        "close",
        "Close",
        "c",
        "C",
        "adj_close",
        "Adj Close",
        "Adj_Close",
    ];
    let volume_names = ["volume", "Volume", "v", "V", "vol", "Vol"];

    fn find_column_index(schema: &arrow::datatypes::Schema, names: &[&str]) -> Option<usize> {
        for name in names {
            if let Ok(idx) = schema.index_of(name) {
                return Some(idx);
            }
        }
        None
    }

    let ts_idx = find_column_index(&schema, &timestamp_names).ok_or_else(|| {
        BacktestError::DataError("No timestamp column found in parquet file".to_string())
    })?;
    let open_idx = find_column_index(&schema, &open_names).ok_or_else(|| {
        BacktestError::DataError("No open column found in parquet file".to_string())
    })?;
    let high_idx = find_column_index(&schema, &high_names).ok_or_else(|| {
        BacktestError::DataError("No high column found in parquet file".to_string())
    })?;
    let low_idx = find_column_index(&schema, &low_names).ok_or_else(|| {
        BacktestError::DataError("No low column found in parquet file".to_string())
    })?;
    let close_idx = find_column_index(&schema, &close_names).ok_or_else(|| {
        BacktestError::DataError("No close column found in parquet file".to_string())
    })?;
    let volume_idx = find_column_index(&schema, &volume_names); // Volume is optional

    debug!(
        "Column indices - timestamp: {}, open: {}, high: {}, low: {}, close: {}, volume: {:?}",
        ts_idx, open_idx, high_idx, low_idx, close_idx, volume_idx
    );

    let mut bars = Vec::new();
    let mut skipped = 0;
    let mut row_num = 0;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| {
            BacktestError::DataError(format!("Failed to read parquet batch: {}", e))
        })?;

        let ts_col = batch.column(ts_idx);
        let open_col = batch.column(open_idx);
        let high_col = batch.column(high_idx);
        let low_col = batch.column(low_idx);
        let close_col = batch.column(close_idx);
        let volume_col = volume_idx.map(|idx| batch.column(idx));

        // Cast numeric columns to Float64
        let open_array = open_col.as_any().downcast_ref::<Float64Array>();
        let high_array = high_col.as_any().downcast_ref::<Float64Array>();
        let low_array = low_col.as_any().downcast_ref::<Float64Array>();
        let close_array = close_col.as_any().downcast_ref::<Float64Array>();
        let volume_array = volume_col.and_then(|col| col.as_any().downcast_ref::<Float64Array>());

        // Handle missing Float64 arrays by trying Int64 and casting
        let (open_vals, high_vals, low_vals, close_vals, vol_vals) = (
            get_f64_values(open_col.as_ref()),
            get_f64_values(high_col.as_ref()),
            get_f64_values(low_col.as_ref()),
            get_f64_values(close_col.as_ref()),
            volume_col.map(|col| get_f64_values(col.as_ref())),
        );

        for i in 0..batch.num_rows() {
            row_num += 1;

            // Parse timestamp based on column type
            let timestamp =
                match parse_arrow_timestamp(ts_col.as_ref(), i, config.date_format.as_deref()) {
                    Ok(ts) => ts,
                    Err(e) => {
                        if config.skip_invalid {
                            debug!("Skipping row {} due to timestamp error: {}", row_num, e);
                            skipped += 1;
                            continue;
                        } else {
                            return Err(e);
                        }
                    }
                };

            let open = open_vals.get(i).copied().or_else(|| {
                open_array.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
            });
            let high = high_vals.get(i).copied().or_else(|| {
                high_array.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
            });
            let low = low_vals.get(i).copied().or_else(|| {
                low_array.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
            });
            let close = close_vals.get(i).copied().or_else(|| {
                close_array.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
            });
            let volume = vol_vals
                .as_ref()
                .and_then(|v| v.get(i).copied())
                .or_else(|| {
                    volume_array.and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) })
                })
                .unwrap_or(0.0);

            let (open, high, low, close) = match (open, high, low, close) {
                (Some(o), Some(h), Some(l), Some(c)) => (o, h, l, c),
                _ => {
                    if config.skip_invalid {
                        debug!("Skipping row {} due to missing OHLC values", row_num);
                        skipped += 1;
                        continue;
                    } else {
                        return Err(BacktestError::DataError(format!(
                            "Missing OHLC values at row {}",
                            row_num
                        )));
                    }
                }
            };

            let bar = Bar::new(timestamp, open, high, low, close, volume);

            if config.validate_bars && !bar.validate() {
                if config.skip_invalid {
                    debug!(
                        "Skipping row {} due to invalid bar data: {:?}",
                        row_num, bar
                    );
                    skipped += 1;
                    continue;
                } else {
                    return Err(BacktestError::DataError(format!(
                        "Invalid bar data at row {}: {:?}",
                        row_num, bar
                    )));
                }
            }

            bars.push(bar);
        }
    }

    if skipped > 0 {
        warn!("Skipped {} invalid rows", skipped);
    }

    // Sort by timestamp
    bars.sort_by_key(|b| b.timestamp);

    // Check for duplicates
    let original_len = bars.len();
    bars.dedup_by_key(|b| b.timestamp);
    if bars.len() < original_len {
        warn!("Removed {} duplicate timestamps", original_len - bars.len());
    }

    info!(
        "Loaded {} bars from {} to {}",
        bars.len(),
        bars.first()
            .map(|b| b.timestamp.to_string())
            .unwrap_or_default(),
        bars.last()
            .map(|b| b.timestamp.to_string())
            .unwrap_or_default()
    );

    if bars.is_empty() {
        return Err(BacktestError::NoData);
    }

    Ok(bars)
}

/// Extract f64 values from an Arrow array (supports Float64 and Int64).
fn get_f64_values(array: &dyn arrow::array::Array) -> Vec<f64> {
    use arrow::array::{Array, Float64Array, Int64Array};

    if let Some(f64_arr) = array.as_any().downcast_ref::<Float64Array>() {
        (0..f64_arr.len())
            .filter_map(|i| {
                if f64_arr.is_null(i) {
                    None
                } else {
                    Some(f64_arr.value(i))
                }
            })
            .collect()
    } else if let Some(i64_arr) = array.as_any().downcast_ref::<Int64Array>() {
        (0..i64_arr.len())
            .filter_map(|i| {
                if i64_arr.is_null(i) {
                    None
                } else {
                    Some(i64_arr.value(i) as f64)
                }
            })
            .collect()
    } else {
        Vec::new()
    }
}

/// Parse a timestamp from an Arrow array at the given index.
fn parse_arrow_timestamp(
    array: &dyn arrow::array::Array,
    idx: usize,
    date_format: Option<&str>,
) -> Result<DateTime<Utc>> {
    use arrow::array::{
        Int64Array, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    };

    if array.is_null(idx) {
        return Err(BacktestError::DataError(format!(
            "Null timestamp at index {}",
            idx
        )));
    }

    // Try Arrow timestamp types first
    if let Some(ts_millis) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        let millis = ts_millis.value(idx);
        return DateTime::from_timestamp_millis(millis).ok_or_else(|| {
            BacktestError::DataError(format!("Invalid timestamp millis: {}", millis))
        });
    }

    if let Some(ts_micros) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        let micros = ts_micros.value(idx);
        let secs = micros / 1_000_000;
        let nanos = ((micros % 1_000_000) * 1000) as u32;
        return DateTime::from_timestamp(secs, nanos).ok_or_else(|| {
            BacktestError::DataError(format!("Invalid timestamp micros: {}", micros))
        });
    }

    if let Some(ts_nanos) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        let nanos = ts_nanos.value(idx);
        let secs = nanos / 1_000_000_000;
        let subsec_nanos = (nanos % 1_000_000_000) as u32;
        return DateTime::from_timestamp(secs, subsec_nanos).ok_or_else(|| {
            BacktestError::DataError(format!("Invalid timestamp nanos: {}", nanos))
        });
    }

    if let Some(ts_secs) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        let secs = ts_secs.value(idx);
        return DateTime::from_timestamp(secs, 0).ok_or_else(|| {
            BacktestError::DataError(format!("Invalid timestamp seconds: {}", secs))
        });
    }

    // Try Int64 as Unix timestamp (assume milliseconds if > 1e12, else seconds)
    if let Some(int_arr) = array.as_any().downcast_ref::<Int64Array>() {
        let val = int_arr.value(idx);
        if val > 1_000_000_000_000 {
            // Milliseconds
            return DateTime::from_timestamp_millis(val).ok_or_else(|| {
                BacktestError::DataError(format!("Invalid timestamp millis: {}", val))
            });
        } else {
            // Seconds
            return DateTime::from_timestamp(val, 0).ok_or_else(|| {
                BacktestError::DataError(format!("Invalid timestamp seconds: {}", val))
            });
        }
    }

    // Try string parsing
    if let Some(str_arr) = array.as_any().downcast_ref::<StringArray>() {
        let s = str_arr.value(idx);
        return parse_datetime(s, date_format);
    }

    Err(BacktestError::DataError(format!(
        "Unsupported timestamp column type: {:?}",
        array.data_type()
    )))
}

/// Detect input file format based on extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Csv,
    Parquet,
}

impl DataFormat {
    /// Detect format from file extension.
    pub fn from_path(path: impl AsRef<Path>) -> Option<Self> {
        let ext = path.as_ref().extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "csv" => Some(DataFormat::Csv),
            "parquet" | "pq" => Some(DataFormat::Parquet),
            _ => None,
        }
    }
}

/// Load OHLCV data from a file, auto-detecting format based on extension.
///
/// Supports CSV and Parquet formats. Format is determined by file extension:
/// - `.csv` -> CSV format
/// - `.parquet` or `.pq` -> Parquet format
pub fn load_data(path: impl AsRef<Path>, config: &DataConfig) -> Result<Vec<Bar>> {
    let path = path.as_ref();
    let format = DataFormat::from_path(path).ok_or_else(|| {
        BacktestError::DataError(format!(
            "Unknown file format for: {}. Supported: .csv, .parquet, .pq",
            path.display()
        ))
    })?;

    match format {
        DataFormat::Csv => load_csv(path, config),
        DataFormat::Parquet => load_parquet(path, config),
    }
}

/// Multi-symbol data container.
#[derive(Debug, Default)]
pub struct DataManager {
    data: HashMap<String, Vec<Bar>>,
}

impl DataManager {
    /// Create a new data manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load data for a symbol from a file (auto-detects CSV or Parquet format).
    pub fn load(&mut self, symbol: impl Into<String>, path: impl AsRef<Path>) -> Result<()> {
        let bars = load_data(path, &DataConfig::default())?;
        self.data.insert(symbol.into(), bars);
        Ok(())
    }

    /// Load data with custom configuration (auto-detects CSV or Parquet format).
    pub fn load_with_config(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
        config: &DataConfig,
    ) -> Result<()> {
        let bars = load_data(path, config)?;
        self.data.insert(symbol.into(), bars);
        Ok(())
    }

    /// Load data for a symbol from a CSV file.
    pub fn load_csv(&mut self, symbol: impl Into<String>, path: impl AsRef<Path>) -> Result<()> {
        let bars = load_csv(path, &DataConfig::default())?;
        self.data.insert(symbol.into(), bars);
        Ok(())
    }

    /// Load data for a symbol from a Parquet file.
    pub fn load_parquet(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let bars = load_parquet(path, &DataConfig::default())?;
        self.data.insert(symbol.into(), bars);
        Ok(())
    }

    /// Load Parquet data with custom configuration.
    pub fn load_parquet_with_config(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
        config: &DataConfig,
    ) -> Result<()> {
        let bars = load_parquet(path, config)?;
        self.data.insert(symbol.into(), bars);
        Ok(())
    }

    /// Add pre-loaded data for a symbol.
    pub fn add(&mut self, symbol: impl Into<String>, bars: Vec<Bar>) {
        self.data.insert(symbol.into(), bars);
    }

    /// Get data for a symbol.
    pub fn get(&self, symbol: &str) -> Option<&Vec<Bar>> {
        self.data.get(symbol)
    }

    /// Get all symbols.
    pub fn symbols(&self) -> Vec<&String> {
        self.data.keys().collect()
    }

    /// Check if a symbol exists.
    pub fn contains(&self, symbol: &str) -> bool {
        self.data.contains_key(symbol)
    }

    /// Get the number of symbols.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the date range across all symbols.
    pub fn date_range(&self) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
        let mut min_date: Option<DateTime<Utc>> = None;
        let mut max_date: Option<DateTime<Utc>> = None;

        for bars in self.data.values() {
            if let Some(first) = bars.first() {
                min_date = Some(min_date.map_or(first.timestamp, |m| m.min(first.timestamp)));
            }
            if let Some(last) = bars.last() {
                max_date = Some(max_date.map_or(last.timestamp, |m| m.max(last.timestamp)));
            }
        }

        min_date.zip(max_date)
    }
}

/// Calculate simple moving average over a slice of bars.
pub fn sma(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }
    let sum: f64 = bars[bars.len() - period..].iter().map(|b| b.close).sum();
    Some(sum / period as f64)
}

/// Calculate exponential moving average over a slice of bars.
pub fn ema(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }

    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_val = sma(&bars[..period], period)?;

    for bar in bars[period..].iter() {
        ema_val = (bar.close - ema_val) * multiplier + ema_val;
    }

    Some(ema_val)
}

/// Calculate standard deviation of closes.
pub fn std_dev(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }

    let mean = sma(bars, period)?;
    let variance: f64 = bars[bars.len() - period..]
        .iter()
        .map(|b| (b.close - mean).powi(2))
        .sum::<f64>()
        / period as f64;

    Some(variance.sqrt())
}

/// Calculate the Average True Range (ATR).
pub fn atr(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period + 1 || period == 0 {
        return None;
    }

    let true_ranges: Vec<f64> = bars
        .windows(2)
        .map(|w| {
            let prev = &w[0];
            let curr = &w[1];
            let hl = curr.high - curr.low;
            let hc = (curr.high - prev.close).abs();
            let lc = (curr.low - prev.close).abs();
            hl.max(hc).max(lc)
        })
        .collect();

    if true_ranges.len() < period {
        return None;
    }

    let sum: f64 = true_ranges[true_ranges.len() - period..].iter().sum();
    Some(sum / period as f64)
}

/// Calculate Relative Strength Index (RSI).
pub fn rsi(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period + 1 || period == 0 {
        return None;
    }

    let changes: Vec<f64> = bars.windows(2).map(|w| w[1].close - w[0].close).collect();

    if changes.len() < period {
        return None;
    }

    let (gains, losses): (Vec<f64>, Vec<f64>) = changes[changes.len() - period..]
        .iter()
        .map(|&c| if c > 0.0 { (c, 0.0) } else { (0.0, -c) })
        .unzip();

    let avg_gain: f64 = gains.iter().sum::<f64>() / period as f64;
    let avg_loss: f64 = losses.iter().sum::<f64>() / period as f64;

    if avg_loss == 0.0 {
        return Some(100.0);
    }

    let rs = avg_gain / avg_loss;
    Some(100.0 - 100.0 / (1.0 + rs))
}

/// Calculate Bollinger Bands (returns middle, upper, lower).
pub fn bollinger_bands(bars: &[Bar], period: usize, num_std: f64) -> Option<(f64, f64, f64)> {
    let middle = sma(bars, period)?;
    let std = std_dev(bars, period)?;
    let upper = middle + num_std * std;
    let lower = middle - num_std * std;
    Some((middle, upper, lower))
}

/// Calculate MACD (Moving Average Convergence Divergence).
/// Returns (macd_line, signal_line, histogram).
pub fn macd(
    bars: &[Bar],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> Option<(f64, f64, f64)> {
    if bars.len() < slow_period + signal_period {
        return None;
    }

    let fast_ema = ema(bars, fast_period)?;
    let slow_ema = ema(bars, slow_period)?;
    let macd_line = fast_ema - slow_ema;

    // Calculate signal line (EMA of MACD)
    // We need to calculate MACD values for the signal period
    let mut macd_values = Vec::with_capacity(signal_period);
    for i in (bars.len() - signal_period)..bars.len() {
        let slice = &bars[..=i];
        if let (Some(f), Some(s)) = (ema(slice, fast_period), ema(slice, slow_period)) {
            macd_values.push(f - s);
        }
    }

    if macd_values.len() < signal_period {
        return None;
    }

    // Calculate EMA of MACD values for signal line
    let multiplier = 2.0 / (signal_period as f64 + 1.0);
    let mut signal = macd_values.iter().take(signal_period).sum::<f64>() / signal_period as f64;
    for &val in macd_values.iter().skip(signal_period) {
        signal = (val - signal) * multiplier + signal;
    }

    // Use last MACD value for signal calculation
    let signal_line = signal;
    let histogram = macd_line - signal_line;

    Some((macd_line, signal_line, histogram))
}

/// Calculate Williams %R oscillator.
/// Returns a value between -100 (oversold) and 0 (overbought).
pub fn williams_r(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }

    let recent = &bars[bars.len() - period..];
    let highest_high = recent
        .iter()
        .map(|b| b.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest_low = recent.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
    let close = bars.last()?.close;

    if (highest_high - lowest_low).abs() < f64::EPSILON {
        return Some(-50.0); // Neutral when no range
    }

    Some(-100.0 * (highest_high - close) / (highest_high - lowest_low))
}

/// Calculate Stochastic Oscillator (%K and %D).
/// Returns (%K, %D) where %K is the fast line and %D is the slow (signal) line.
pub fn stochastic(bars: &[Bar], k_period: usize, d_period: usize) -> Option<(f64, f64)> {
    if bars.len() < k_period + d_period || k_period == 0 || d_period == 0 {
        return None;
    }

    // Calculate %K values for the last d_period bars
    let mut k_values = Vec::with_capacity(d_period);
    for i in (bars.len() - d_period)..bars.len() {
        if i >= k_period {
            let slice = &bars[i + 1 - k_period..=i];
            let highest_high = slice
                .iter()
                .map(|b| b.high)
                .fold(f64::NEG_INFINITY, f64::max);
            let lowest_low = slice.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
            let close = slice.last()?.close;

            if (highest_high - lowest_low).abs() > f64::EPSILON {
                let k = 100.0 * (close - lowest_low) / (highest_high - lowest_low);
                k_values.push(k);
            }
        }
    }

    if k_values.is_empty() {
        return None;
    }

    let percent_k = *k_values.last()?;
    let percent_d = k_values.iter().sum::<f64>() / k_values.len() as f64;

    Some((percent_k, percent_d))
}

/// Calculate On-Balance Volume (OBV).
pub fn obv(bars: &[Bar]) -> f64 {
    if bars.len() < 2 {
        return 0.0;
    }

    let mut obv_value = 0.0;
    for i in 1..bars.len() {
        if bars[i].close > bars[i - 1].close {
            obv_value += bars[i].volume;
        } else if bars[i].close < bars[i - 1].close {
            obv_value -= bars[i].volume;
        }
        // If close == prev close, OBV stays the same
    }
    obv_value
}

/// Calculate Volume Weighted Average Price (VWAP) for the period.
pub fn vwap(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }

    let recent = &bars[bars.len() - period..];
    let mut cum_tp_vol = 0.0;
    let mut cum_vol = 0.0;

    for bar in recent {
        let typical_price = bar.typical_price();
        cum_tp_vol += typical_price * bar.volume;
        cum_vol += bar.volume;
    }

    if cum_vol == 0.0 {
        return None;
    }

    Some(cum_tp_vol / cum_vol)
}

/// Calculate Commodity Channel Index (CCI).
pub fn cci(bars: &[Bar], period: usize) -> Option<f64> {
    if bars.len() < period || period == 0 {
        return None;
    }

    let recent = &bars[bars.len() - period..];
    let typical_prices: Vec<f64> = recent.iter().map(|b| b.typical_price()).collect();
    let mean_tp: f64 = typical_prices.iter().sum::<f64>() / period as f64;

    // Calculate mean deviation
    let mean_deviation: f64 = typical_prices
        .iter()
        .map(|tp| (tp - mean_tp).abs())
        .sum::<f64>()
        / period as f64;

    if mean_deviation.abs() < f64::EPSILON {
        return Some(0.0);
    }

    let current_tp = bars.last()?.typical_price();
    Some((current_tp - mean_tp) / (0.015 * mean_deviation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, TimeZone, Timelike};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "Date,Open,High,Low,Close,Volume").unwrap();
        writeln!(file, "2024-01-01,100,105,98,102,1000").unwrap();
        writeln!(file, "2024-01-02,102,108,101,107,1200").unwrap();
        writeln!(file, "2024-01-03,107,110,105,108,1100").unwrap();
        writeln!(file, "2024-01-04,108,109,103,104,900").unwrap();
        writeln!(file, "2024-01-05,104,106,100,105,1000").unwrap();
        file
    }

    fn create_test_bars() -> Vec<Bar> {
        vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0,
                105.0,
                98.0,
                102.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                102.0,
                108.0,
                101.0,
                107.0,
                1200.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                107.0,
                110.0,
                105.0,
                108.0,
                1100.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
                108.0,
                109.0,
                103.0,
                104.0,
                900.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
                104.0,
                106.0,
                100.0,
                105.0,
                1000.0,
            ),
        ]
    }

    #[test]
    fn test_load_csv() {
        let file = create_test_csv();
        let bars = load_csv(file.path(), &DataConfig::default()).unwrap();

        assert_eq!(bars.len(), 5);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[4].close, 105.0);
    }

    #[test]
    fn test_date_parsing() {
        // ISO format
        let dt = parse_datetime("2024-01-15", None).unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);

        // With time
        let dt2 = parse_datetime("2024-01-15 09:30:00", None).unwrap();
        assert_eq!(dt2.hour(), 9);
        assert_eq!(dt2.minute(), 30);
    }

    #[test]
    fn test_sma() {
        let bars = create_test_bars();

        let sma_3 = sma(&bars, 3).unwrap();
        // (108 + 104 + 105) / 3 = 105.67
        assert!((sma_3 - 105.666666).abs() < 0.001);

        // Not enough data
        assert!(sma(&bars[..2], 3).is_none());
    }

    #[test]
    fn test_std_dev() {
        let bars = create_test_bars();
        let std = std_dev(&bars, 5).unwrap();
        // Standard deviation of [102, 107, 108, 104, 105]
        assert!(std > 0.0);
    }

    #[test]
    fn test_rsi() {
        let bars = create_test_bars();
        let rsi_val = rsi(&bars, 3).unwrap();
        // RSI should be between 0 and 100
        assert!(rsi_val >= 0.0 && rsi_val <= 100.0);
    }

    #[test]
    fn test_data_manager() {
        let bars = create_test_bars();
        let mut dm = DataManager::new();
        dm.add("TEST", bars);

        assert!(dm.contains("TEST"));
        assert!(!dm.contains("OTHER"));
        assert_eq!(dm.len(), 1);
        assert_eq!(dm.get("TEST").unwrap().len(), 5);
    }

    #[test]
    fn test_ema() {
        let bars = create_test_bars();
        let ema_val = ema(&bars, 3).unwrap();
        // EMA should be close to recent prices
        assert!(ema_val > 100.0 && ema_val < 110.0);
    }

    #[test]
    fn test_atr() {
        let bars = create_test_bars();
        let atr_val = atr(&bars, 3).unwrap();
        // ATR should be positive
        assert!(atr_val > 0.0);
    }

    #[test]
    fn test_williams_r() {
        let bars = create_test_bars();
        let wr = williams_r(&bars, 5).unwrap();
        // Williams %R should be between -100 and 0
        assert!(wr >= -100.0 && wr <= 0.0);
    }

    #[test]
    fn test_stochastic() {
        let bars = create_test_bars();
        let result = stochastic(&bars, 3, 2);
        assert!(result.is_some());
        let (k, d) = result.unwrap();
        // Stochastic should be between 0 and 100
        assert!(k >= 0.0 && k <= 100.0);
        assert!(d >= 0.0 && d <= 100.0);
    }

    #[test]
    fn test_obv() {
        let bars = create_test_bars();
        let obv_val = obv(&bars);
        // OBV should be non-zero with varying prices
        assert!(obv_val.abs() > 0.0 || bars.windows(2).all(|w| w[0].close == w[1].close));
    }

    #[test]
    fn test_vwap() {
        let bars = create_test_bars();
        let vwap_val = vwap(&bars, 3).unwrap();
        // VWAP should be in a reasonable price range
        assert!(vwap_val > 90.0 && vwap_val < 120.0);
    }

    #[test]
    fn test_cci() {
        let bars = create_test_bars();
        let cci_val = cci(&bars, 3).unwrap();
        // CCI is unbounded but should be a finite number
        assert!(cci_val.is_finite());
    }

    #[test]
    fn test_macd() {
        // Need more bars for MACD calculation
        let bars: Vec<Bar> = (0..50)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.5);
                Bar::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i),
                    base,
                    base + 2.0,
                    base - 1.0,
                    base + 0.5,
                    1000.0,
                )
            })
            .collect();

        let result = macd(&bars, 12, 26, 9);
        assert!(result.is_some());
        let (macd_line, signal_line, histogram) = result.unwrap();
        assert!(macd_line.is_finite());
        assert!(signal_line.is_finite());
        assert!((histogram - (macd_line - signal_line)).abs() < 0.0001);
    }

    /// Helper to create a parquet file with OHLCV data for testing.
    fn create_test_parquet() -> tempfile::NamedTempFile {
        use arrow::array::{Float64Array, TimestampMillisecondArray};
        use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let file = tempfile::NamedTempFile::with_suffix(".parquet").unwrap();

        // Create schema
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                false,
            ),
            Field::new("open", DataType::Float64, false),
            Field::new("high", DataType::Float64, false),
            Field::new("low", DataType::Float64, false),
            Field::new("close", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        // Create data arrays
        let timestamps: Vec<i64> = vec![
            Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0)
                .unwrap()
                .timestamp_millis(),
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0)
                .unwrap()
                .timestamp_millis(),
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0)
                .unwrap()
                .timestamp_millis(),
            Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0)
                .unwrap()
                .timestamp_millis(),
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0)
                .unwrap()
                .timestamp_millis(),
        ];

        let arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(TimestampMillisecondArray::from(timestamps)),
            Arc::new(Float64Array::from(vec![100.0, 102.0, 107.0, 108.0, 104.0])),
            Arc::new(Float64Array::from(vec![105.0, 108.0, 110.0, 109.0, 106.0])),
            Arc::new(Float64Array::from(vec![98.0, 101.0, 105.0, 103.0, 100.0])),
            Arc::new(Float64Array::from(vec![102.0, 107.0, 108.0, 104.0, 105.0])),
            Arc::new(Float64Array::from(vec![
                1000.0, 1200.0, 1100.0, 900.0, 1000.0,
            ])),
        ];

        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();

        // Write to parquet
        let file_handle = File::create(file.path()).unwrap();
        let mut writer = ArrowWriter::try_new(file_handle, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        file
    }

    #[test]
    fn test_load_parquet() {
        let file = create_test_parquet();
        let bars = load_parquet(file.path(), &DataConfig::default()).unwrap();

        assert_eq!(bars.len(), 5);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[0].high, 105.0);
        assert_eq!(bars[0].low, 98.0);
        assert_eq!(bars[0].close, 102.0);
        assert_eq!(bars[0].volume, 1000.0);
        assert_eq!(bars[4].close, 105.0);
    }

    #[test]
    fn test_load_parquet_matches_csv() {
        // Test that parquet loading gives same results as CSV loading
        let csv_file = create_test_csv();
        let parquet_file = create_test_parquet();

        let csv_bars = load_csv(csv_file.path(), &DataConfig::default()).unwrap();
        let parquet_bars = load_parquet(parquet_file.path(), &DataConfig::default()).unwrap();

        assert_eq!(csv_bars.len(), parquet_bars.len());
        for (csv_bar, parquet_bar) in csv_bars.iter().zip(parquet_bars.iter()) {
            assert_eq!(csv_bar.open, parquet_bar.open);
            assert_eq!(csv_bar.high, parquet_bar.high);
            assert_eq!(csv_bar.low, parquet_bar.low);
            assert_eq!(csv_bar.close, parquet_bar.close);
            assert_eq!(csv_bar.volume, parquet_bar.volume);
        }
    }

    #[test]
    fn test_load_data_auto_detect() {
        // Test CSV auto-detection
        let csv_file = create_test_csv();
        let csv_bars = load_data(csv_file.path(), &DataConfig::default()).unwrap();
        assert_eq!(csv_bars.len(), 5);

        // Test Parquet auto-detection
        let parquet_file = create_test_parquet();
        let parquet_bars = load_data(parquet_file.path(), &DataConfig::default()).unwrap();
        assert_eq!(parquet_bars.len(), 5);
    }

    #[test]
    fn test_data_format_detection() {
        assert_eq!(DataFormat::from_path("data.csv"), Some(DataFormat::Csv));
        assert_eq!(
            DataFormat::from_path("data.parquet"),
            Some(DataFormat::Parquet)
        );
        assert_eq!(DataFormat::from_path("data.pq"), Some(DataFormat::Parquet));
        assert_eq!(DataFormat::from_path("data.txt"), None);
        assert_eq!(DataFormat::from_path("data"), None);
    }

    #[test]
    fn test_data_manager_parquet() {
        let parquet_file = create_test_parquet();
        let mut dm = DataManager::new();
        dm.load_parquet("TEST", parquet_file.path()).unwrap();

        assert!(dm.contains("TEST"));
        assert_eq!(dm.get("TEST").unwrap().len(), 5);
    }

    #[test]
    fn test_data_manager_auto_detect() {
        // Test with parquet file using auto-detect load
        let parquet_file = create_test_parquet();
        let mut dm = DataManager::new();
        dm.load("TEST_PARQUET", parquet_file.path()).unwrap();

        assert!(dm.contains("TEST_PARQUET"));
        assert_eq!(dm.get("TEST_PARQUET").unwrap().len(), 5);

        // Test with CSV file using auto-detect load
        let csv_file = create_test_csv();
        dm.load("TEST_CSV", csv_file.path()).unwrap();

        assert!(dm.contains("TEST_CSV"));
        assert_eq!(dm.get("TEST_CSV").unwrap().len(), 5);
    }
}
