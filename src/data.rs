//! Data loading and management for the backtest engine.

use crate::error::{BacktestError, Result};
use crate::types::{AssetConfig, Bar, VolumeProfile};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tracing::{debug, info, warn};

/// Raw CSV row with flexible date parsing.
#[derive(Debug, Deserialize)]
struct CsvRow {
    #[serde(
        alias = "Date",
        alias = "date",
        alias = "DATE",
        alias = "Timestamp",
        alias = "timestamp",
        alias = "Time",
        alias = "time",
        alias = "datetime",
        alias = "Datetime"
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
    #[serde(
        alias = "Volume",
        alias = "volume",
        alias = "v",
        alias = "vol",
        alias = "Vol",
        default
    )]
    volume: f64,
}

/// Data source configuration.
#[derive(Debug, Clone)]
pub struct DataConfig {
    /// Date format string for parsing (e.g., "%Y-%m-%d" or "%Y-%m-%d %H:%M:%S").
    pub date_format: Option<String>,
    /// Whether the CSV has headers.
    pub has_headers: bool,
    /// CSV delimiter character. If None, delimiter is auto-detected.
    pub delimiter: Option<u8>,
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
            delimiter: None, // Auto-detect by default
            skip_invalid: true,
            validate_bars: true,
        }
    }
}

/// Detect the CSV delimiter by analyzing the first few lines of the file.
///
/// Tries common delimiters (comma, tab, semicolon, pipe) and returns the one
/// that produces the most consistent column count across lines.
fn detect_delimiter(path: &Path) -> Result<u8> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read first 5 lines for analysis (or less if file is smaller)
    let lines: Vec<String> = reader.lines().take(5).filter_map(|l| l.ok()).collect();

    if lines.is_empty() {
        return Ok(b','); // Default to comma for empty files
    }

    // Common delimiters to try (comma, tab, semicolon, pipe)
    let delimiters = [b',', b'\t', b';', b'|'];

    let mut best_delimiter = b',';
    let mut best_score = 0;

    for &delim in &delimiters {
        // Count fields per line for this delimiter
        let counts: Vec<usize> = lines
            .iter()
            .map(|line| line.as_bytes().iter().filter(|&&b| b == delim).count() + 1)
            .collect();

        if counts.is_empty() {
            continue;
        }

        // Calculate score: consistency (all lines have same count) + minimum 5 fields for OHLCV
        let first_count = counts[0];
        let all_consistent = counts.iter().all(|&c| c == first_count);
        let has_enough_fields = first_count >= 5; // At least date,O,H,L,C,V

        if all_consistent && has_enough_fields {
            // Score higher for more fields (tie-breaker)
            let score = first_count;
            if score > best_score {
                best_score = score;
                best_delimiter = delim;
            }
        }
    }

    // If no delimiter produced consistent results, try to find any that produces >= 5 fields
    if best_score == 0 {
        for &delim in &delimiters {
            let first_line = &lines[0];
            let count = first_line
                .as_bytes()
                .iter()
                .filter(|&&b| b == delim)
                .count()
                + 1;
            if count >= 5 {
                debug!(
                    "Detected delimiter {:?} with {} fields",
                    delim as char, count
                );
                return Ok(delim);
            }
        }
    }

    debug!(
        "Detected delimiter {:?} with score {}",
        best_delimiter as char, best_score
    );
    Ok(best_delimiter)
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
    let date_formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",  // 15-Jan-2024
        "%d %b %Y",  // 15 Jan 2024
        "%b %d, %Y", // Jan 15, 2024
    ];

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

    // Auto-detect delimiter if not specified
    let delimiter = match config.delimiter {
        Some(d) => d,
        None => {
            let detected = detect_delimiter(path)?;
            debug!("Auto-detected delimiter: {:?}", char::from(detected));
            detected
        }
    };

    let mut reader = ReaderBuilder::new()
        .has_headers(config.has_headers)
        .delimiter(delimiter)
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

// =============================================================================
// Bundled Sample Data
// =============================================================================

/// Bundled sample CSV data for AAPL (10 years daily OHLCV, 2014-2024).
const SAMPLE_AAPL_CSV: &str = include_str!("../data/samples/AAPL.csv");

/// Bundled sample CSV data for SPY (10 years daily OHLCV, 2014-2024).
const SAMPLE_SPY_CSV: &str = include_str!("../data/samples/SPY.csv");

/// Bundled sample CSV data for BTC (10 years daily OHLCV including weekends, 2014-2024).
const SAMPLE_BTC_CSV: &str = include_str!("../data/samples/BTC.csv");

/// Available sample data names.
///
/// Returns a list of sample data identifiers that can be used with [`load_sample`].
///
/// # Example
/// ```
/// use mantis::data::list_samples;
///
/// let samples = list_samples();
/// assert!(samples.contains(&"AAPL"));
/// assert!(samples.contains(&"SPY"));
/// assert!(samples.contains(&"BTC"));
/// ```
pub fn list_samples() -> Vec<&'static str> {
    vec!["AAPL", "SPY", "BTC"]
}

/// Load bundled sample data by name.
///
/// Sample data is embedded in the binary and requires no external files or internet access.
/// Available samples:
/// - `"AAPL"` - Apple Inc. stock (10 years daily, 2014-2024)
/// - `"SPY"` - S&P 500 ETF (10 years daily, 2014-2024)
/// - `"BTC"` - Bitcoin (10 years daily including weekends, 2014-2024)
///
/// # Arguments
/// * `name` - Sample data identifier (case-insensitive)
///
/// # Returns
/// `Ok(Vec<Bar>)` containing the sample OHLCV data
///
/// # Errors
/// Returns `BacktestError::DataError` if the sample name is not recognized.
///
/// # Example
/// ```
/// use mantis::data::load_sample;
///
/// // Load Apple stock data
/// let aapl_data = load_sample("AAPL").unwrap();
/// assert!(aapl_data.len() > 2500); // ~10 years of daily data
///
/// // Case-insensitive
/// let spy_data = load_sample("spy").unwrap();
/// ```
pub fn load_sample(name: &str) -> Result<Vec<Bar>> {
    let name_upper = name.to_uppercase();
    let csv_content = match name_upper.as_str() {
        "AAPL" => SAMPLE_AAPL_CSV,
        "SPY" => SAMPLE_SPY_CSV,
        "BTC" => SAMPLE_BTC_CSV,
        _ => {
            return Err(BacktestError::DataError(format!(
                "Unknown sample: '{}'. Available samples: {:?}",
                name,
                list_samples()
            )))
        }
    };

    load_csv_from_string(csv_content, &DataConfig::default())
}

/// Parse CSV data from a string (used for embedded sample data).
fn load_csv_from_string(csv_content: &str, config: &DataConfig) -> Result<Vec<Bar>> {
    // For string-based loading, default to comma if not specified
    let delimiter = config.delimiter.unwrap_or(b',');

    let mut reader = ReaderBuilder::new()
        .has_headers(config.has_headers)
        .delimiter(delimiter)
        .flexible(true)
        .from_reader(csv_content.as_bytes());

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

    debug!(
        "Loaded {} bars from embedded sample data ({} to {})",
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

/// Load multiple symbols from a map of symbol -> file path.
///
/// # Example
/// ```ignore
/// use std::collections::HashMap;
/// use mantis::data::{load_multi, DataConfig};
///
/// let paths: HashMap<&str, &str> = [
///     ("AAPL", "data/AAPL.csv"),
///     ("GOOGL", "data/GOOGL.csv"),
///     ("MSFT", "data/MSFT.csv"),
/// ].into_iter().collect();
///
/// let data = load_multi(&paths, &DataConfig::default())?;
/// assert!(data.contains_key("AAPL"));
/// ```
pub fn load_multi<S, P>(
    paths: &HashMap<S, P>,
    config: &DataConfig,
) -> Result<HashMap<String, Vec<Bar>>>
where
    S: AsRef<str>,
    P: AsRef<Path>,
{
    let mut result = HashMap::new();

    for (symbol, path) in paths {
        let symbol = symbol.as_ref().to_string();
        let path = path.as_ref();

        info!("Loading {} from {}", symbol, path.display());

        let bars = load_data(path, config)?;
        result.insert(symbol, bars);
    }

    info!("Loaded {} symbols", result.len());
    Ok(result)
}

/// Load all matching files from a directory using a glob pattern.
///
/// The symbol name is derived from the file stem (filename without extension).
///
/// # Example
/// ```ignore
/// use mantis::data::{load_dir, DataConfig};
///
/// // Load all CSV files from the data directory
/// let data = load_dir("data/", "*.csv", &DataConfig::default())?;
///
/// // Load all Parquet files matching a pattern
/// let data = load_dir("data/prices/", "*.parquet", &DataConfig::default())?;
/// ```
pub fn load_dir(
    dir: impl AsRef<Path>,
    pattern: &str,
    config: &DataConfig,
) -> Result<HashMap<String, Vec<Bar>>> {
    let dir = dir.as_ref();

    if !dir.is_dir() {
        return Err(BacktestError::DataError(format!(
            "Not a directory: {}",
            dir.display()
        )));
    }

    // Construct full glob pattern
    let glob_pattern = dir.join(pattern);
    let glob_pattern_str = glob_pattern.to_string_lossy();

    info!("Loading files matching: {}", glob_pattern_str);

    let paths = glob::glob(&glob_pattern_str).map_err(|e| {
        BacktestError::DataError(format!("Invalid glob pattern '{}': {}", pattern, e))
    })?;

    let mut result = HashMap::new();
    let mut loaded = 0;
    let mut errors = 0;

    for entry in paths {
        match entry {
            Ok(path) => {
                // Extract symbol from filename stem
                let symbol = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| {
                        BacktestError::DataError(format!(
                            "Could not extract symbol from path: {}",
                            path.display()
                        ))
                    })?;

                debug!("Loading {} from {}", symbol, path.display());

                match load_data(&path, config) {
                    Ok(bars) => {
                        result.insert(symbol, bars);
                        loaded += 1;
                    }
                    Err(e) => {
                        warn!("Failed to load {}: {}", path.display(), e);
                        errors += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Glob error: {}", e);
                errors += 1;
            }
        }
    }

    if loaded == 0 && errors > 0 {
        return Err(BacktestError::DataError(format!(
            "No files loaded from {}. {} errors occurred.",
            dir.display(),
            errors
        )));
    }

    info!(
        "Loaded {} symbols from {} ({} errors)",
        loaded,
        dir.display(),
        errors
    );
    Ok(result)
}

/// Multi-symbol data container.
#[derive(Debug, Default)]
pub struct DataManager {
    data: HashMap<String, Vec<Bar>>,
    asset_configs: HashMap<String, AssetConfig>,
    /// SHA256 checksums of data files (symbol -> checksum).
    checksums: HashMap<String, String>,
}

impl DataManager {
    /// Create a new data manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load data for a symbol from a file (auto-detects CSV or Parquet format).
    pub fn load(&mut self, symbol: impl Into<String>, path: impl AsRef<Path>) -> Result<()> {
        let symbol = symbol.into();
        let path = path.as_ref();

        // Compute checksum for reproducibility
        if let Ok(checksum) = crate::metadata::compute_file_checksum(path) {
            self.checksums.insert(symbol.clone(), checksum);
        }

        let bars = load_data(path, &DataConfig::default())?;
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
        Ok(())
    }

    /// Load data with custom configuration (auto-detects CSV or Parquet format).
    pub fn load_with_config(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
        config: &DataConfig,
    ) -> Result<()> {
        let symbol = symbol.into();
        let path = path.as_ref();

        // Compute checksum for reproducibility
        if let Ok(checksum) = crate::metadata::compute_file_checksum(path) {
            self.checksums.insert(symbol.clone(), checksum);
        }

        let bars = load_data(path, config)?;
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
        Ok(())
    }

    /// Load data for a symbol from a CSV file.
    pub fn load_csv(&mut self, symbol: impl Into<String>, path: impl AsRef<Path>) -> Result<()> {
        let symbol = symbol.into();
        let path = path.as_ref();

        // Compute checksum for reproducibility
        if let Ok(checksum) = crate::metadata::compute_file_checksum(path) {
            self.checksums.insert(symbol.clone(), checksum);
        }

        let bars = load_csv(path, &DataConfig::default())?;
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
        Ok(())
    }

    /// Load data for a symbol from a Parquet file.
    pub fn load_parquet(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let symbol = symbol.into();
        let path = path.as_ref();

        // Compute checksum for reproducibility
        if let Ok(checksum) = crate::metadata::compute_file_checksum(path) {
            self.checksums.insert(symbol.clone(), checksum);
        }

        let bars = load_parquet(path, &DataConfig::default())?;
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
        Ok(())
    }

    /// Load Parquet data with custom configuration.
    pub fn load_parquet_with_config(
        &mut self,
        symbol: impl Into<String>,
        path: impl AsRef<Path>,
        config: &DataConfig,
    ) -> Result<()> {
        let symbol = symbol.into();
        let path = path.as_ref();

        // Compute checksum for reproducibility
        if let Ok(checksum) = crate::metadata::compute_file_checksum(path) {
            self.checksums.insert(symbol.clone(), checksum);
        }

        let bars = load_parquet(path, config)?;
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
        Ok(())
    }

    /// Add pre-loaded data for a symbol.
    pub fn add(&mut self, symbol: impl Into<String>, bars: Vec<Bar>) {
        let symbol = symbol.into();
        self.ensure_asset_config(&symbol);
        self.data.insert(symbol, bars);
    }

    /// Load multiple symbols from a map of symbol -> file path.
    ///
    /// # Example
    /// ```ignore
    /// use std::collections::HashMap;
    /// use mantis::data::DataManager;
    ///
    /// let mut manager = DataManager::new();
    /// let paths: HashMap<&str, &str> = [
    ///     ("AAPL", "data/AAPL.csv"),
    ///     ("GOOGL", "data/GOOGL.csv"),
    /// ].into_iter().collect();
    ///
    /// manager.load_multi(&paths)?;
    /// assert!(manager.contains("AAPL"));
    /// ```
    pub fn load_multi<S, P>(&mut self, paths: &HashMap<S, P>) -> Result<()>
    where
        S: AsRef<str>,
        P: AsRef<Path>,
    {
        let data = load_multi(paths, &DataConfig::default())?;
        for (symbol, bars) in data {
            self.ensure_asset_config(&symbol);
            self.data.insert(symbol, bars);
        }
        Ok(())
    }

    /// Load all matching files from a directory using a glob pattern.
    ///
    /// The symbol name is derived from the file stem (filename without extension).
    ///
    /// # Example
    /// ```ignore
    /// use mantis::data::DataManager;
    ///
    /// let mut manager = DataManager::new();
    /// manager.load_dir("data/", "*.csv")?;
    /// println!("Loaded symbols: {:?}", manager.symbols());
    /// ```
    pub fn load_dir(&mut self, dir: impl AsRef<Path>, pattern: &str) -> Result<()> {
        let data = load_dir(dir, pattern, &DataConfig::default())?;
        for (symbol, bars) in data {
            self.ensure_asset_config(&symbol);
            self.data.insert(symbol, bars);
        }
        Ok(())
    }

    /// Load multiple symbols from a map with custom configuration.
    pub fn load_multi_with_config<S, P>(
        &mut self,
        paths: &HashMap<S, P>,
        config: &DataConfig,
    ) -> Result<()>
    where
        S: AsRef<str>,
        P: AsRef<Path>,
    {
        let data = load_multi(paths, config)?;
        for (symbol, bars) in data {
            self.ensure_asset_config(&symbol);
            self.data.insert(symbol, bars);
        }
        Ok(())
    }

    /// Load all matching files from a directory with custom configuration.
    pub fn load_dir_with_config(
        &mut self,
        dir: impl AsRef<Path>,
        pattern: &str,
        config: &DataConfig,
    ) -> Result<()> {
        let data = load_dir(dir, pattern, config)?;
        for (symbol, bars) in data {
            self.ensure_asset_config(&symbol);
            self.data.insert(symbol, bars);
        }
        Ok(())
    }

    /// Register or override the asset configuration for a symbol.
    pub fn set_asset_config(&mut self, config: AssetConfig) {
        self.asset_configs.insert(config.symbol.clone(), config);
    }

    /// Get the configured asset metadata for a symbol.
    pub fn asset_config(&self, symbol: &str) -> Option<&AssetConfig> {
        self.asset_configs.get(symbol)
    }

    /// Return the asset configurations for all loaded symbols.
    pub fn asset_configs(&self) -> &HashMap<String, AssetConfig> {
        &self.asset_configs
    }

    /// Compute the volume profile for a specific symbol.
    pub fn volume_profile(&self, symbol: &str) -> Option<VolumeProfile> {
        self.data
            .get(symbol)
            .and_then(|bars| VolumeProfile::from_bars(bars))
    }

    /// Compute volume profiles for all loaded symbols.
    pub fn volume_profiles(&self) -> HashMap<String, VolumeProfile> {
        self.data
            .iter()
            .filter_map(|(symbol, bars)| {
                VolumeProfile::from_bars(bars).map(|profile| (symbol.clone(), profile))
            })
            .collect()
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

    /// Get SHA256 checksums of loaded data files.
    ///
    /// Returns a map of symbol -> checksum for all data loaded from files.
    /// Pre-loaded data (via `add()`) will not have checksums.
    pub fn checksums(&self) -> &HashMap<String, String> {
        &self.checksums
    }
}

impl DataManager {
    fn ensure_asset_config(&mut self, symbol: &str) {
        self.asset_configs
            .entry(symbol.to_string())
            .or_insert_with(|| AssetConfig::equity(symbol));
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

// =============================================================================
// Time-Series Resampling
// =============================================================================

/// Interval for resampling time-series data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResampleInterval {
    /// Minute intervals (e.g., 5, 15, 30 minutes).
    Minute(u32),
    /// Hourly intervals (e.g., 1, 4 hours).
    Hour(u32),
    /// Daily intervals.
    Day,
    /// Weekly intervals.
    Week,
    /// Monthly intervals.
    Month,
}

impl ResampleInterval {
    /// Get the interval duration in seconds (for minute/hour/day intervals).
    /// Returns None for week/month as they vary in actual duration.
    pub fn to_seconds(&self) -> Option<i64> {
        match self {
            ResampleInterval::Minute(m) => Some(*m as i64 * 60),
            ResampleInterval::Hour(h) => Some(*h as i64 * 3600),
            ResampleInterval::Day => Some(86400),
            ResampleInterval::Week => Some(7 * 86400),
            ResampleInterval::Month => None, // Variable duration
        }
    }

    /// Calculate the bucket key for a given timestamp.
    /// This determines which resampled bar a timestamp belongs to.
    fn bucket_key(&self, timestamp: DateTime<Utc>) -> i64 {
        use chrono::Datelike;

        match self {
            ResampleInterval::Minute(m) => {
                let secs = timestamp.timestamp();
                let interval_secs = *m as i64 * 60;
                secs / interval_secs
            }
            ResampleInterval::Hour(h) => {
                let secs = timestamp.timestamp();
                let interval_secs = *h as i64 * 3600;
                secs / interval_secs
            }
            ResampleInterval::Day => {
                // Use ordinal day from epoch
                timestamp.timestamp() / 86400
            }
            ResampleInterval::Week => {
                // ISO week number combined with year
                let iso_week = timestamp.iso_week();
                iso_week.year() as i64 * 100 + iso_week.week() as i64
            }
            ResampleInterval::Month => {
                // Year * 12 + month
                let year = timestamp.year() as i64;
                let month = timestamp.month() as i64;
                year * 12 + month
            }
        }
    }

    /// Get the start timestamp for a bucket.
    fn bucket_start(&self, timestamp: DateTime<Utc>) -> DateTime<Utc> {
        use chrono::{Datelike, NaiveDate, TimeZone};

        match self {
            ResampleInterval::Minute(m) => {
                let secs = timestamp.timestamp();
                let interval_secs = *m as i64 * 60;
                let bucket_start_secs = (secs / interval_secs) * interval_secs;
                DateTime::from_timestamp(bucket_start_secs, 0).unwrap_or(timestamp)
            }
            ResampleInterval::Hour(h) => {
                let secs = timestamp.timestamp();
                let interval_secs = *h as i64 * 3600;
                let bucket_start_secs = (secs / interval_secs) * interval_secs;
                DateTime::from_timestamp(bucket_start_secs, 0).unwrap_or(timestamp)
            }
            ResampleInterval::Day => {
                let date = timestamp.date_naive();
                Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
            }
            ResampleInterval::Week => {
                // Start of ISO week (Monday)
                let iso_week = timestamp.iso_week();
                let monday = NaiveDate::from_isoywd_opt(
                    iso_week.year(),
                    iso_week.week(),
                    chrono::Weekday::Mon,
                )
                .unwrap_or(timestamp.date_naive());
                Utc.from_utc_datetime(&monday.and_hms_opt(0, 0, 0).unwrap())
            }
            ResampleInterval::Month => {
                let date = NaiveDate::from_ymd_opt(timestamp.year(), timestamp.month(), 1).unwrap();
                Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap())
            }
        }
    }
}

/// Resample bars to a coarser time interval.
///
/// Aggregates OHLCV data using standard rules:
/// - Open: first bar's open in the period
/// - High: maximum high across all bars
/// - Low: minimum low across all bars
/// - Close: last bar's close in the period
/// - Volume: sum of all volumes
///
/// # Arguments
/// * `bars` - Input bars (must be sorted by timestamp)
/// * `target` - Target resampling interval
///
/// # Returns
/// Resampled bars at the target interval.
pub fn resample(bars: &[Bar], target: ResampleInterval) -> Vec<Bar> {
    if bars.is_empty() {
        return Vec::new();
    }

    // Group bars by their bucket
    let mut buckets: HashMap<i64, Vec<&Bar>> = HashMap::new();
    for bar in bars {
        let key = target.bucket_key(bar.timestamp);
        buckets.entry(key).or_default().push(bar);
    }

    // Aggregate each bucket into a single bar
    let mut result: Vec<Bar> = buckets
        .into_iter()
        .filter_map(|(_, bucket_bars)| {
            if bucket_bars.is_empty() {
                return None;
            }

            // Bars in bucket should be sorted by time
            let mut sorted_bars = bucket_bars;
            sorted_bars.sort_by_key(|b| b.timestamp);

            let first = sorted_bars.first()?;
            let last = sorted_bars.last()?;

            let open = first.open;
            let close = last.close;
            let high = sorted_bars
                .iter()
                .map(|b| b.high)
                .fold(f64::NEG_INFINITY, f64::max);
            let low = sorted_bars
                .iter()
                .map(|b| b.low)
                .fold(f64::INFINITY, f64::min);
            let volume: f64 = sorted_bars.iter().map(|b| b.volume).sum();

            // Use bucket start time as the timestamp
            let timestamp = target.bucket_start(first.timestamp);

            Some(Bar::new(timestamp, open, high, low, close, volume))
        })
        .collect();

    // Sort result by timestamp
    result.sort_by_key(|b| b.timestamp);
    result
}

// =============================================================================
// Missing Data Handling
// =============================================================================

/// Represents a gap in time-series data.
#[derive(Debug, Clone, PartialEq)]
pub struct DataGap {
    /// Start of the gap (last known timestamp before gap).
    pub start: DateTime<Utc>,
    /// End of the gap (first timestamp after gap).
    pub end: DateTime<Utc>,
    /// Number of expected bars missing.
    pub expected_bars: usize,
}

impl DataGap {
    /// Duration of the gap.
    pub fn duration(&self) -> chrono::Duration {
        self.end.signed_duration_since(self.start)
    }
}

/// Methods for filling missing data gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FillMethod {
    /// Forward fill: use the last known values.
    #[default]
    ForwardFill,
    /// Backward fill: use the next known values.
    BackwardFill,
    /// Linear interpolation for prices, zero for volume.
    Linear,
    /// Fill with zeros (mainly useful for volume).
    Zero,
}

/// Report on data quality issues.
#[derive(Debug, Clone, Default)]
pub struct DataQualityReport {
    /// Total number of bars in the dataset.
    pub total_bars: usize,
    /// Detected gaps in the data.
    pub gaps: Vec<DataGap>,
    /// Percentage of expected data that is missing.
    pub gap_percentage: f64,
    /// Number of duplicate timestamps found (and removed).
    pub duplicate_timestamps: usize,
    /// Number of bars with invalid data (e.g., high < low).
    pub invalid_bars: usize,
    /// Expected time interval between bars (for gap detection).
    pub expected_interval_seconds: i64,
}

impl DataQualityReport {
    /// Check if the data quality is acceptable (no major issues).
    pub fn is_acceptable(&self) -> bool {
        self.gaps.is_empty() && self.duplicate_timestamps == 0 && self.invalid_bars == 0
    }

    /// Get a summary string of the report.
    pub fn summary(&self) -> String {
        format!(
            "Bars: {}, Gaps: {}, Gap%: {:.2}%, Duplicates: {}, Invalid: {}",
            self.total_bars,
            self.gaps.len(),
            self.gap_percentage,
            self.duplicate_timestamps,
            self.invalid_bars
        )
    }
}

/// Detect gaps in time-series data.
///
/// A gap is detected when the time between consecutive bars exceeds
/// the expected interval by more than 50%.
///
/// # Arguments
/// * `bars` - Input bars (should be sorted by timestamp)
/// * `expected_interval_seconds` - Expected seconds between bars
///
/// # Returns
/// List of detected gaps.
pub fn detect_gaps(bars: &[Bar], expected_interval_seconds: i64) -> Vec<DataGap> {
    if bars.len() < 2 || expected_interval_seconds <= 0 {
        return Vec::new();
    }

    let threshold = (expected_interval_seconds as f64 * 1.5) as i64;
    let mut gaps = Vec::new();

    for window in bars.windows(2) {
        let elapsed = window[1]
            .timestamp
            .signed_duration_since(window[0].timestamp)
            .num_seconds();

        if elapsed > threshold {
            let expected_bars = ((elapsed / expected_interval_seconds) - 1) as usize;
            gaps.push(DataGap {
                start: window[0].timestamp,
                end: window[1].timestamp,
                expected_bars,
            });
        }
    }

    gaps
}

/// Fill gaps in time-series data.
///
/// # Arguments
/// * `bars` - Input bars (should be sorted by timestamp)
/// * `expected_interval_seconds` - Expected seconds between bars
/// * `method` - Fill method to use
///
/// # Returns
/// Bars with gaps filled according to the specified method.
pub fn fill_gaps(bars: &[Bar], expected_interval_seconds: i64, method: FillMethod) -> Vec<Bar> {
    if bars.len() < 2 || expected_interval_seconds <= 0 {
        return bars.to_vec();
    }

    let mut result = Vec::with_capacity(bars.len());
    let threshold = (expected_interval_seconds as f64 * 1.5) as i64;

    for window in bars.windows(2) {
        result.push(window[0].clone());

        let elapsed = window[1]
            .timestamp
            .signed_duration_since(window[0].timestamp)
            .num_seconds();

        if elapsed > threshold {
            // Calculate number of bars to insert
            let num_fill = ((elapsed / expected_interval_seconds) - 1) as usize;

            for j in 1..=num_fill {
                let fill_timestamp = window[0].timestamp
                    + chrono::Duration::seconds(expected_interval_seconds * j as i64);

                let fill_bar = match method {
                    FillMethod::ForwardFill => Bar::new(
                        fill_timestamp,
                        window[0].close, // Use previous close as open
                        window[0].close,
                        window[0].close,
                        window[0].close,
                        0.0, // Zero volume for filled bars
                    ),
                    FillMethod::BackwardFill => Bar::new(
                        fill_timestamp,
                        window[1].open, // Use next open
                        window[1].open,
                        window[1].open,
                        window[1].open,
                        0.0,
                    ),
                    FillMethod::Linear => {
                        let t = (j as f64) / ((num_fill + 1) as f64);
                        let interp_price = window[0].close * (1.0 - t) + window[1].open * t;
                        Bar::new(
                            fill_timestamp,
                            interp_price,
                            interp_price,
                            interp_price,
                            interp_price,
                            0.0,
                        )
                    }
                    FillMethod::Zero => Bar::new(fill_timestamp, 0.0, 0.0, 0.0, 0.0, 0.0),
                };

                result.push(fill_bar);
            }
        }
    }

    // Add the last bar
    if let Some(last) = bars.last() {
        result.push(last.clone());
    }

    result
}

/// Generate a data quality report for the given bars.
///
/// # Arguments
/// * `bars` - Input bars
/// * `expected_interval_seconds` - Expected seconds between bars
///
/// # Returns
/// A comprehensive data quality report.
pub fn data_quality_report(bars: &[Bar], expected_interval_seconds: i64) -> DataQualityReport {
    let total_bars = bars.len();

    // Detect gaps
    let gaps = detect_gaps(bars, expected_interval_seconds);

    // Calculate gap percentage
    let total_missing: usize = gaps.iter().map(|g| g.expected_bars).sum();
    let expected_total = total_bars + total_missing;
    let gap_percentage = if expected_total > 0 {
        (total_missing as f64 / expected_total as f64) * 100.0
    } else {
        0.0
    };

    // Count invalid bars
    let invalid_bars = bars.iter().filter(|b| !b.validate()).count();

    // Note: duplicates are removed during loading, so we can't count them here
    // This would need to be tracked during the loading process

    DataQualityReport {
        total_bars,
        gaps,
        gap_percentage,
        duplicate_timestamps: 0, // Would be set during loading
        invalid_bars,
        expected_interval_seconds,
    }
}

// =============================================================================
// Multi-Symbol Time-Series Alignment
// =============================================================================

/// Mode for aligning multiple time series.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignMode {
    /// Inner join: only timestamps present in all series.
    #[default]
    Inner,
    /// Outer join with forward fill for missing values.
    OuterForwardFill,
    /// Outer join - missing values result in None.
    OuterNone,
}

/// Aligned data point for a single timestamp across multiple symbols.
#[derive(Debug, Clone)]
pub struct AlignedBars {
    pub timestamp: DateTime<Utc>,
    pub bars: HashMap<String, Option<Bar>>,
}

impl AlignedBars {
    /// Check if all symbols have data at this timestamp.
    pub fn is_complete(&self) -> bool {
        self.bars.values().all(|b| b.is_some())
    }

    /// Get the bar for a specific symbol.
    pub fn get(&self, symbol: &str) -> Option<&Bar> {
        self.bars.get(symbol).and_then(|b| b.as_ref())
    }
}

/// Align multiple time series to common timestamps.
///
/// # Arguments
/// * `series` - Slice of (symbol, bars) pairs
/// * `mode` - Alignment mode to use
///
/// # Returns
/// Vector of aligned bars at each timestamp.
pub fn align_series(series: &[(&str, &[Bar])], mode: AlignMode) -> Vec<AlignedBars> {
    if series.is_empty() {
        return Vec::new();
    }

    // Collect all unique timestamps
    let mut all_timestamps: Vec<DateTime<Utc>> = series
        .iter()
        .flat_map(|(_, bars)| bars.iter().map(|b| b.timestamp))
        .collect();
    all_timestamps.sort();
    all_timestamps.dedup();

    // Build index maps for quick lookup
    let symbol_maps: HashMap<&str, HashMap<DateTime<Utc>, &Bar>> = series
        .iter()
        .map(|(symbol, bars)| {
            let map: HashMap<DateTime<Utc>, &Bar> = bars.iter().map(|b| (b.timestamp, b)).collect();
            (*symbol, map)
        })
        .collect();

    let symbols: Vec<&str> = series.iter().map(|(s, _)| *s).collect();

    // Track last known bars for forward fill
    let mut last_bars: HashMap<&str, &Bar> = HashMap::new();

    let mut result = Vec::new();

    for ts in all_timestamps {
        let mut bars_at_ts: HashMap<String, Option<Bar>> = HashMap::new();
        let mut all_present = true;

        for &symbol in &symbols {
            let bar = symbol_maps.get(symbol).and_then(|m| m.get(&ts).copied());

            match bar {
                Some(b) => {
                    last_bars.insert(symbol, b);
                    bars_at_ts.insert(symbol.to_string(), Some(b.clone()));
                }
                None => {
                    all_present = false;
                    match mode {
                        AlignMode::Inner => {
                            bars_at_ts.insert(symbol.to_string(), None);
                        }
                        AlignMode::OuterForwardFill => {
                            let filled = last_bars.get(symbol).map(|&prev| {
                                // Create forward-filled bar with updated timestamp
                                Bar::new(ts, prev.close, prev.close, prev.close, prev.close, 0.0)
                            });
                            bars_at_ts.insert(symbol.to_string(), filled);
                        }
                        AlignMode::OuterNone => {
                            bars_at_ts.insert(symbol.to_string(), None);
                        }
                    }
                }
            }
        }

        // For inner join, only include if all symbols have data
        if mode == AlignMode::Inner && !all_present {
            continue;
        }

        result.push(AlignedBars {
            timestamp: ts,
            bars: bars_at_ts,
        });
    }

    result
}

/// Convert aligned bars back to separate series.
///
/// This is useful after alignment to get back individual bar vectors.
pub fn unalign_series(aligned: &[AlignedBars], symbol: &str) -> Vec<Bar> {
    aligned
        .iter()
        .filter_map(|ab| ab.get(symbol).cloned())
        .collect()
}

// =============================================================================
// Corporate Actions Adjustment
// =============================================================================

use crate::types::{CorporateAction, CorporateActionType, DividendAdjustMethod, DividendType};

/// Adjust bars for stock splits.
///
/// This function adjusts historical prices by dividing by the split ratio for all
/// bars before the ex-date. This makes pre-split prices comparable to post-split prices.
///
/// For a 4-for-1 split, pre-split prices are divided by 4.
/// For a 1-for-10 reverse split (ratio = 0.1), pre-split prices are divided by 0.1 (multiplied by 10).
///
/// # Arguments
/// * `bars` - Mutable slice of bars to adjust (should be sorted by timestamp)
/// * `splits` - Slice of corporate actions (only Split and ReverseSplit are processed)
///
/// # Example
/// ```
/// use mantis::data::adjust_for_splits;
/// use mantis::types::{Bar, CorporateAction};
/// use chrono::{TimeZone, Utc};
///
/// let mut bars = vec![
///     Bar::new(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), 400.0, 410.0, 390.0, 405.0, 1000.0),
///     Bar::new(Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(), 100.0, 105.0, 98.0, 102.0, 4000.0),
/// ];
///
/// let splits = vec![
///     CorporateAction::split("AAPL", 4.0, Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap()),
/// ];
///
/// adjust_for_splits(&mut bars, &splits);
/// // Pre-split bar (Jan 1) prices are now divided by 4
/// assert!((bars[0].close - 101.25).abs() < 0.01);
/// ```
pub fn adjust_for_splits(bars: &mut [Bar], splits: &[CorporateAction]) {
    // Filter to only split/reverse split actions
    let split_actions: Vec<&CorporateAction> = splits
        .iter()
        .filter(|a| a.requires_price_adjustment())
        .collect();

    if split_actions.is_empty() || bars.is_empty() {
        return;
    }

    // Sort actions by ex_date descending (process most recent first)
    let mut sorted_actions = split_actions;
    sorted_actions.sort_by(|a, b| b.ex_date.cmp(&a.ex_date));

    for action in sorted_actions {
        let ratio = action.adjustment_factor();
        if (ratio - 1.0).abs() < f64::EPSILON || ratio <= 0.0 {
            continue;
        }

        // Adjust all bars before the ex-date
        for bar in bars.iter_mut() {
            if bar.timestamp < action.ex_date {
                bar.open /= ratio;
                bar.high /= ratio;
                bar.low /= ratio;
                bar.close /= ratio;
                // Volume is multiplied by ratio (more shares = more volume)
                bar.volume *= ratio;
            }
        }
    }
}

/// Adjust bars for dividend payments.
///
/// This function adjusts historical prices to account for dividend payments,
/// making pre-dividend prices comparable to post-dividend prices.
///
/// # Arguments
/// * `bars` - Mutable slice of bars to adjust (should be sorted by timestamp)
/// * `dividends` - Slice of corporate actions (only Dividend types are processed)
/// * `method` - Method to use for adjustment
///
/// # Adjustment Methods
/// * `Proportional` - Standard method: multiply pre-ex-date prices by (1 - dividend/close_before_ex)
/// * `Absolute` - Subtract the dividend amount from all pre-ex-date prices
/// * `None` - No adjustment
pub fn adjust_for_dividends(
    bars: &mut [Bar],
    dividends: &[CorporateAction],
    method: DividendAdjustMethod,
) {
    if method == DividendAdjustMethod::None {
        return;
    }

    // Filter to only dividend actions
    let div_actions: Vec<&CorporateAction> = dividends.iter().filter(|a| a.is_dividend()).collect();

    if div_actions.is_empty() || bars.is_empty() {
        return;
    }

    // Sort actions by ex_date descending (process most recent first)
    let mut sorted_actions = div_actions;
    sorted_actions.sort_by(|a, b| b.ex_date.cmp(&a.ex_date));

    for action in sorted_actions {
        let dividend = match action.dividend_amount() {
            Some(d) if d > 0.0 => d,
            _ => continue,
        };

        // Find the close price just before the ex-date for proportional adjustment
        let close_before_ex = match method {
            DividendAdjustMethod::Proportional => {
                // Find the last bar before ex-date by searching from the end
                bars.iter()
                    .rev()
                    .find(|b| b.timestamp < action.ex_date)
                    .map(|b| b.close)
            }
            _ => None,
        };

        // Calculate adjustment factor for proportional method
        let adjustment_factor = match (method, close_before_ex) {
            (DividendAdjustMethod::Proportional, Some(close)) if close > dividend => {
                (close - dividend) / close
            }
            (DividendAdjustMethod::Proportional, _) => 1.0, // Skip if close not found or dividend >= close
            _ => 1.0,
        };

        // Adjust all bars before the ex-date
        for bar in bars.iter_mut() {
            if bar.timestamp < action.ex_date {
                match method {
                    DividendAdjustMethod::Proportional => {
                        bar.open *= adjustment_factor;
                        bar.high *= adjustment_factor;
                        bar.low *= adjustment_factor;
                        bar.close *= adjustment_factor;
                    }
                    DividendAdjustMethod::Absolute => {
                        bar.open -= dividend;
                        bar.high -= dividend;
                        bar.low -= dividend;
                        bar.close -= dividend;
                    }
                    DividendAdjustMethod::None => {}
                }
                // Volume is not affected by dividend adjustments
            }
        }
    }
}

/// Apply a list of adjustment factors to bars.
///
/// Each factor is applied to all bars before the specified timestamp.
/// Factors are cumulative - if there are multiple factors, they are multiplied together.
///
/// # Arguments
/// * `bars` - Mutable slice of bars to adjust
/// * `factors` - Slice of (timestamp, factor) pairs. Bars before each timestamp are divided by the factor.
///
/// # Example
/// ```
/// use mantis::data::apply_adjustment_factor;
/// use mantis::types::Bar;
/// use chrono::{TimeZone, Utc};
///
/// let mut bars = vec![
///     Bar::new(Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(), 100.0, 105.0, 95.0, 102.0, 1000.0),
/// ];
///
/// let factors = vec![
///     (Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(), 2.0),
/// ];
///
/// apply_adjustment_factor(&mut bars, &factors);
/// assert!((bars[0].close - 51.0).abs() < 0.01);
/// ```
pub fn apply_adjustment_factor(bars: &mut [Bar], factors: &[(chrono::DateTime<chrono::Utc>, f64)]) {
    if factors.is_empty() || bars.is_empty() {
        return;
    }

    // Sort factors by timestamp descending (process most recent first)
    let mut sorted_factors: Vec<_> = factors.to_vec();
    sorted_factors.sort_by(|a, b| b.0.cmp(&a.0));

    for (timestamp, factor) in sorted_factors {
        if (factor - 1.0).abs() < f64::EPSILON || factor <= 0.0 {
            continue;
        }

        for bar in bars.iter_mut() {
            if bar.timestamp < timestamp {
                bar.open /= factor;
                bar.high /= factor;
                bar.low /= factor;
                bar.close /= factor;
                bar.volume *= factor;
            }
        }
    }
}

/// CSV row for corporate actions file.
#[derive(Debug, Deserialize)]
struct CorporateActionRow {
    #[serde(alias = "Symbol", alias = "symbol", alias = "ticker", alias = "Ticker")]
    symbol: String,
    #[serde(alias = "Type", alias = "type", alias = "action", alias = "Action")]
    action_type: String,
    #[serde(
        alias = "ExDate",
        alias = "ex_date",
        alias = "Date",
        alias = "date",
        alias = "ex-date"
    )]
    ex_date: String,
    #[serde(
        alias = "Value",
        alias = "value",
        alias = "amount",
        alias = "Amount",
        alias = "ratio",
        alias = "Ratio",
        default
    )]
    value: f64,
    #[serde(
        alias = "RecordDate",
        alias = "record_date",
        alias = "record-date",
        default
    )]
    record_date: Option<String>,
    #[serde(alias = "PayDate", alias = "pay_date", alias = "pay-date", default)]
    pay_date: Option<String>,
    #[serde(
        alias = "NewSymbol",
        alias = "new_symbol",
        alias = "new-symbol",
        default
    )]
    new_symbol: Option<String>,
}

/// Load corporate actions from a CSV file.
///
/// Expected CSV columns:
/// - symbol: Stock symbol
/// - action_type: One of "split", "reverse_split", "dividend", "special_dividend", "stock_dividend", "spinoff"
/// - ex_date: Ex-date in parseable date format
/// - value: Split ratio or dividend amount
/// - record_date: Optional record date
/// - pay_date: Optional pay date
/// - new_symbol: New symbol for spin-offs
///
/// # Example CSV
/// ```csv
/// symbol,type,ex_date,value
/// AAPL,split,2020-08-31,4.0
/// MSFT,dividend,2024-02-14,0.75
/// GE,spinoff,2024-04-02,0.25
/// ```
pub fn load_corporate_actions(path: impl AsRef<Path>) -> Result<Vec<CorporateAction>> {
    let path = path.as_ref();
    info!("Loading corporate actions from: {}", path.display());

    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_path(path)?;

    let mut actions = Vec::new();

    for (row_num, result) in reader.deserialize().enumerate() {
        let row: CorporateActionRow = match result {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping row {}: {}", row_num + 1, e);
                continue;
            }
        };

        let ex_date = match parse_datetime(&row.ex_date, None) {
            Ok(d) => d,
            Err(e) => {
                warn!("Skipping row {} - invalid ex_date: {}", row_num + 1, e);
                continue;
            }
        };

        let record_date = row
            .record_date
            .as_ref()
            .and_then(|s| parse_datetime(s, None).ok());
        let pay_date = row
            .pay_date
            .as_ref()
            .and_then(|s| parse_datetime(s, None).ok());

        let action_type = row.action_type.to_lowercase();
        let action = match action_type.as_str() {
            "split" | "stock_split" | "stock-split" => {
                if row.value <= 0.0 {
                    warn!(
                        "Skipping row {} - invalid split ratio: {}",
                        row_num + 1,
                        row.value
                    );
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::Split { ratio: row.value },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            "reverse_split" | "reverse-split" | "reversesplit" => {
                if row.value <= 0.0 {
                    warn!(
                        "Skipping row {} - invalid reverse split ratio: {}",
                        row_num + 1,
                        row.value
                    );
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::ReverseSplit { ratio: row.value },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            "dividend" | "cash_dividend" | "cash-dividend" => {
                if row.value <= 0.0 {
                    warn!(
                        "Skipping row {} - invalid dividend amount: {}",
                        row_num + 1,
                        row.value
                    );
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::Dividend {
                        amount: row.value,
                        div_type: DividendType::Cash,
                    },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            "special_dividend" | "special-dividend" | "specialdividend" => {
                if row.value <= 0.0 {
                    warn!(
                        "Skipping row {} - invalid special dividend amount: {}",
                        row_num + 1,
                        row.value
                    );
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::Dividend {
                        amount: row.value,
                        div_type: DividendType::Special,
                    },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            "stock_dividend" | "stock-dividend" | "stockdividend" => {
                if row.value <= 0.0 {
                    warn!(
                        "Skipping row {} - invalid stock dividend amount: {}",
                        row_num + 1,
                        row.value
                    );
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::Dividend {
                        amount: row.value,
                        div_type: DividendType::Stock,
                    },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            "spinoff" | "spin_off" | "spin-off" => {
                let new_symbol = row.new_symbol.unwrap_or_default();
                if new_symbol.is_empty() {
                    warn!("Skipping row {} - spinoff missing new_symbol", row_num + 1);
                    continue;
                }
                CorporateAction {
                    symbol: row.symbol,
                    action_type: CorporateActionType::SpinOff {
                        ratio: row.value,
                        new_symbol,
                    },
                    ex_date,
                    record_date,
                    pay_date,
                }
            }
            _ => {
                warn!(
                    "Skipping row {} - unknown action type: {}",
                    row_num + 1,
                    row.action_type
                );
                continue;
            }
        };

        actions.push(action);
    }

    // Sort by ex_date
    actions.sort_by_key(|a| a.ex_date);

    info!("Loaded {} corporate actions", actions.len());
    Ok(actions)
}

/// Filter corporate actions for a specific symbol.
pub fn filter_actions_for_symbol<'a>(
    actions: &'a [CorporateAction],
    symbol: &str,
) -> Vec<&'a CorporateAction> {
    actions.iter().filter(|a| a.symbol == symbol).collect()
}

/// Calculate cumulative adjustment factor for a bar based on all subsequent corporate actions.
///
/// This is useful for converting an unadjusted price to an adjusted price.
pub fn cumulative_adjustment_factor(
    bar_timestamp: chrono::DateTime<chrono::Utc>,
    actions: &[CorporateAction],
) -> f64 {
    actions
        .iter()
        .filter(|a| a.requires_price_adjustment() && a.ex_date > bar_timestamp)
        .map(|a| a.adjustment_factor())
        .product()
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

        // Named month format: 15-Jan-2024
        let dt3 = parse_datetime("15-Jan-2024", None).unwrap();
        assert_eq!(dt3.year(), 2024);
        assert_eq!(dt3.month(), 1);
        assert_eq!(dt3.day(), 15);

        // Named month with spaces: 15 Jan 2024
        let dt4 = parse_datetime("15 Jan 2024", None).unwrap();
        assert_eq!(dt4.year(), 2024);
        assert_eq!(dt4.month(), 1);
        assert_eq!(dt4.day(), 15);

        // US named month format: Jan 15, 2024
        let dt5 = parse_datetime("Jan 15, 2024", None).unwrap();
        assert_eq!(dt5.year(), 2024);
        assert_eq!(dt5.month(), 1);
        assert_eq!(dt5.day(), 15);
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
        assert!((0.0..=100.0).contains(&rsi_val));
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
        assert!((-100.0..=0.0).contains(&wr));
    }

    #[test]
    fn test_stochastic() {
        let bars = create_test_bars();
        let result = stochastic(&bars, 3, 2);
        assert!(result.is_some());
        let (k, d) = result.unwrap();
        // Stochastic should be between 0 and 100
        assert!((0.0..=100.0).contains(&k));
        assert!((0.0..=100.0).contains(&d));
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

    // =========================================================================
    // Resampling Tests
    // =========================================================================

    fn create_minute_bars() -> Vec<Bar> {
        // Create 60 minute bars (1 hour of data)
        (0..60)
            .map(|i| {
                let ts = Utc.with_ymd_and_hms(2024, 1, 1, 9, i, 0).unwrap();
                let base_price = 100.0 + (i as f64 * 0.1);
                Bar::new(
                    ts,
                    base_price,
                    base_price + 0.5,
                    base_price - 0.3,
                    base_price + 0.2,
                    1000.0 + (i as f64 * 10.0),
                )
            })
            .collect()
    }

    #[test]
    fn test_resample_minute_to_5min() {
        let bars = create_minute_bars();
        let resampled = resample(&bars, ResampleInterval::Minute(5));

        // 60 minutes -> 12 five-minute bars
        assert_eq!(resampled.len(), 12);

        // Check first bar aggregation
        let first = &resampled[0];
        assert_eq!(first.open, 100.0); // First minute's open
        assert_eq!(first.close, 100.0 + 0.2 + 4.0 * 0.1); // Fifth minute's close (i=4)

        // High should be max of first 5 bars
        let expected_high = (0..5)
            .map(|i| 100.0 + (i as f64 * 0.1) + 0.5)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((first.high - expected_high).abs() < 0.001);

        // Volume should be sum
        let expected_vol: f64 = (0..5).map(|i| 1000.0 + (i as f64 * 10.0)).sum();
        assert!((first.volume - expected_vol).abs() < 0.001);
    }

    #[test]
    fn test_resample_minute_to_hour() {
        let bars = create_minute_bars();
        let resampled = resample(&bars, ResampleInterval::Hour(1));

        // 60 minutes -> 1 hour bar
        assert_eq!(resampled.len(), 1);

        let bar = &resampled[0];
        assert_eq!(bar.open, 100.0);
        assert!((bar.close - (100.0 + 0.2 + 59.0 * 0.1)).abs() < 0.001);

        // Volume should be sum of all 60 bars
        let expected_vol: f64 = (0..60).map(|i| 1000.0 + (i as f64 * 10.0)).sum();
        assert!((bar.volume - expected_vol).abs() < 0.001);
    }

    #[test]
    fn test_resample_daily_to_weekly() {
        // Create 14 daily bars (2 weeks)
        let bars: Vec<Bar> = (0..14)
            .map(|i| {
                let ts =
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap() + chrono::Duration::days(i);
                Bar::new(
                    ts,
                    100.0 + i as f64,
                    105.0 + i as f64,
                    95.0 + i as f64,
                    102.0 + i as f64,
                    1000.0,
                )
            })
            .collect();

        let resampled = resample(&bars, ResampleInterval::Week);

        // Should have 3 weeks (partial week at start, full week, partial at end)
        // Week 1 of 2024 starts Jan 1 (Monday)
        assert!(resampled.len() >= 2);

        // Each resampled bar should have proper OHLC aggregation
        for bar in &resampled {
            assert!(bar.high >= bar.low);
            assert!(bar.high >= bar.open);
            assert!(bar.high >= bar.close);
        }
    }

    #[test]
    fn test_resample_empty() {
        let bars: Vec<Bar> = Vec::new();
        let resampled = resample(&bars, ResampleInterval::Day);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_resample_interval_to_seconds() {
        assert_eq!(ResampleInterval::Minute(5).to_seconds(), Some(300));
        assert_eq!(ResampleInterval::Hour(1).to_seconds(), Some(3600));
        assert_eq!(ResampleInterval::Day.to_seconds(), Some(86400));
        assert_eq!(ResampleInterval::Week.to_seconds(), Some(604800));
        assert_eq!(ResampleInterval::Month.to_seconds(), None);
    }

    // =========================================================================
    // Gap Detection Tests
    // =========================================================================

    fn create_bars_with_gap() -> Vec<Bar> {
        // Create bars with a 3-day gap
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
            // Gap: Jan 3, 4, 5 missing
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 6, 0, 0, 0).unwrap(),
                108.0,
                112.0,
                106.0,
                110.0,
                900.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 7, 0, 0, 0).unwrap(),
                110.0,
                115.0,
                108.0,
                112.0,
                1100.0,
            ),
        ]
    }

    #[test]
    fn test_detect_gaps() {
        let bars = create_bars_with_gap();
        let gaps = detect_gaps(&bars, 86400); // Daily interval

        assert_eq!(gaps.len(), 1);

        let gap = &gaps[0];
        assert_eq!(
            gap.start,
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap()
        );
        assert_eq!(gap.end, Utc.with_ymd_and_hms(2024, 1, 6, 0, 0, 0).unwrap());
        assert_eq!(gap.expected_bars, 3); // Jan 3, 4, 5
    }

    #[test]
    fn test_detect_gaps_no_gaps() {
        let bars = create_test_bars();
        let gaps = detect_gaps(&bars, 86400); // Daily interval

        assert!(gaps.is_empty());
    }

    #[test]
    fn test_fill_gaps_forward_fill() {
        let bars = create_bars_with_gap();
        let filled = fill_gaps(&bars, 86400, FillMethod::ForwardFill);

        assert_eq!(filled.len(), 7); // Original 4 + 3 filled

        // Check filled bars use previous close
        let filled_bar = &filled[2]; // First filled bar (Jan 3)
        assert_eq!(filled_bar.open, 107.0); // Previous close
        assert_eq!(filled_bar.close, 107.0);
        assert_eq!(filled_bar.volume, 0.0); // Zero volume for fills
    }

    #[test]
    fn test_fill_gaps_backward_fill() {
        let bars = create_bars_with_gap();
        let filled = fill_gaps(&bars, 86400, FillMethod::BackwardFill);

        assert_eq!(filled.len(), 7);

        // Check filled bars use next open
        let filled_bar = &filled[2]; // First filled bar
        assert_eq!(filled_bar.open, 108.0); // Next open
        assert_eq!(filled_bar.close, 108.0);
    }

    #[test]
    fn test_fill_gaps_linear() {
        let bars = create_bars_with_gap();
        let filled = fill_gaps(&bars, 86400, FillMethod::Linear);

        assert_eq!(filled.len(), 7);

        // Check interpolated values
        // Gap from close=107 to open=108
        // With 3 fill bars, t = 1/4, 2/4, 3/4
        let filled_bar = &filled[2]; // t = 0.25
        let expected_price = 107.0 * 0.75 + 108.0 * 0.25;
        assert!((filled_bar.close - expected_price).abs() < 0.001);
    }

    #[test]
    fn test_data_quality_report() {
        let bars = create_bars_with_gap();
        let report = data_quality_report(&bars, 86400);

        assert_eq!(report.total_bars, 4);
        assert_eq!(report.gaps.len(), 1);
        assert!(report.gap_percentage > 0.0);
        assert_eq!(report.invalid_bars, 0);
        assert!(!report.is_acceptable()); // Has gaps
    }

    #[test]
    fn test_data_quality_report_clean_data() {
        let bars = create_test_bars();
        let report = data_quality_report(&bars, 86400);

        assert!(report.is_acceptable());
        assert!(report.gaps.is_empty());
    }

    // =========================================================================
    // Alignment Tests
    // =========================================================================

    fn create_series_a() -> Vec<Bar> {
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
        ]
    }

    fn create_series_b() -> Vec<Bar> {
        vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                50.0,
                55.0,
                48.0,
                52.0,
                500.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                52.0,
                58.0,
                51.0,
                57.0,
                600.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0).unwrap(),
                57.0,
                60.0,
                55.0,
                58.0,
                550.0,
            ),
        ]
    }

    #[test]
    fn test_align_series_inner() {
        let series_a = create_series_a();
        let series_b = create_series_b();

        let aligned = align_series(&[("A", &series_a), ("B", &series_b)], AlignMode::Inner);

        // Only Jan 2 and Jan 3 are common
        assert_eq!(aligned.len(), 2);

        // Check first aligned point (Jan 2)
        let first = &aligned[0];
        assert_eq!(
            first.timestamp,
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap()
        );
        assert!(first.is_complete());
        assert_eq!(first.get("A").unwrap().close, 107.0);
        assert_eq!(first.get("B").unwrap().close, 52.0);
    }

    #[test]
    fn test_align_series_outer_none() {
        let series_a = create_series_a();
        let series_b = create_series_b();

        let aligned = align_series(&[("A", &series_a), ("B", &series_b)], AlignMode::OuterNone);

        // All 4 unique dates
        assert_eq!(aligned.len(), 4);

        // Jan 1 - only A has data
        let jan1 = &aligned[0];
        assert!(jan1.get("A").is_some());
        assert!(jan1.get("B").is_none());

        // Jan 4 - only B has data
        let jan4 = &aligned[3];
        assert!(jan4.get("A").is_none());
        assert!(jan4.get("B").is_some());
    }

    #[test]
    fn test_align_series_outer_forward_fill() {
        let series_a = create_series_a();
        let series_b = create_series_b();

        let aligned = align_series(
            &[("A", &series_a), ("B", &series_b)],
            AlignMode::OuterForwardFill,
        );

        assert_eq!(aligned.len(), 4);

        // Jan 1 - B should have None (no previous value to forward fill)
        let jan1 = &aligned[0];
        assert!(jan1.get("A").is_some());
        assert!(jan1.get("B").is_none());

        // Jan 4 - A should be forward filled from Jan 3
        let jan4 = &aligned[3];
        let a_bar = jan4.get("A").unwrap();
        assert_eq!(a_bar.close, 108.0); // Forward filled from Jan 3 close
        assert_eq!(a_bar.volume, 0.0); // Filled bars have zero volume
    }

    #[test]
    fn test_unalign_series() {
        let series_a = create_series_a();
        let series_b = create_series_b();

        let aligned = align_series(&[("A", &series_a), ("B", &series_b)], AlignMode::Inner);
        let unaligned_a = unalign_series(&aligned, "A");

        assert_eq!(unaligned_a.len(), 2);
        assert_eq!(unaligned_a[0].close, 107.0);
        assert_eq!(unaligned_a[1].close, 108.0);
    }

    #[test]
    fn test_align_series_empty() {
        let aligned = align_series(&[], AlignMode::Inner);
        assert!(aligned.is_empty());
    }

    #[test]
    fn test_aligned_bars_is_complete() {
        let series_a = create_series_a();
        let series_b = create_series_b();

        let aligned = align_series(&[("A", &series_a), ("B", &series_b)], AlignMode::OuterNone);

        // Jan 2 should be complete
        let jan2 = &aligned[1];
        assert!(jan2.is_complete());

        // Jan 1 should not be complete
        let jan1 = &aligned[0];
        assert!(!jan1.is_complete());
    }

    // =========================================================================
    // Corporate Actions Tests
    // =========================================================================

    fn create_split_test_bars() -> Vec<Bar> {
        // Create bars spanning a 4-for-1 split on Jan 5
        vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                400.0,
                420.0,
                390.0,
                410.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                410.0,
                430.0,
                400.0,
                420.0,
                1100.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                420.0,
                440.0,
                410.0,
                432.0,
                1200.0,
            ),
            // After split on Jan 5
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 8, 0, 0, 0).unwrap(),
                108.0,
                112.0,
                105.0,
                110.0,
                4500.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 9, 0, 0, 0).unwrap(),
                110.0,
                115.0,
                108.0,
                112.0,
                4800.0,
            ),
        ]
    }

    #[test]
    fn test_adjust_for_splits() {
        let mut bars = create_split_test_bars();
        let splits = vec![CorporateAction::split(
            "AAPL",
            4.0,
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
        )];

        adjust_for_splits(&mut bars, &splits);

        // Pre-split bars (Jan 1-3) should be divided by 4
        assert!((bars[0].close - 102.5).abs() < 0.01); // 410 / 4
        assert!((bars[1].close - 105.0).abs() < 0.01); // 420 / 4
        assert!((bars[2].close - 108.0).abs() < 0.01); // 432 / 4

        // Pre-split volume should be multiplied by 4
        assert!((bars[0].volume - 4000.0).abs() < 0.01);

        // Post-split bars should remain unchanged
        assert!((bars[3].close - 110.0).abs() < 0.01);
        assert!((bars[4].close - 112.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_for_reverse_split() {
        let mut bars = vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                1.0,
                1.2,
                0.9,
                1.1,
                10000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
                10.0,
                12.0,
                9.0,
                11.0,
                1000.0,
            ),
        ];

        let splits = vec![CorporateAction::reverse_split(
            "SIRI",
            0.1, // 1-for-10 reverse split
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        )];

        adjust_for_splits(&mut bars, &splits);

        // Pre-split price should be divided by 0.1 (multiplied by 10)
        assert!((bars[0].close - 11.0).abs() < 0.01); // 1.1 / 0.1
                                                      // Pre-split volume should be multiplied by 0.1
        assert!((bars[0].volume - 1000.0).abs() < 0.01); // 10000 * 0.1

        // Post-split should remain unchanged
        assert!((bars[1].close - 11.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_for_dividends_proportional() {
        let mut bars = vec![
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
                106.0,
                1100.0,
            ),
            // Ex-date is Jan 3
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                105.0,
                110.0,
                104.0,
                108.0,
                1200.0,
            ),
        ];

        let dividends = vec![CorporateAction::cash_dividend(
            "MSFT",
            2.0, // $2 dividend per share
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        )];

        adjust_for_dividends(&mut bars, &dividends, DividendAdjustMethod::Proportional);

        // Adjustment factor = (106 - 2) / 106 = 0.9811...
        let factor = (106.0 - 2.0) / 106.0;

        // Pre-ex bars should be adjusted
        assert!((bars[0].close - 102.0 * factor).abs() < 0.01);
        assert!((bars[1].close - 106.0 * factor).abs() < 0.01);

        // Post-ex bar should remain unchanged
        assert!((bars[2].close - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_for_dividends_absolute() {
        let mut bars = vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0,
                105.0,
                98.0,
                102.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                105.0,
                110.0,
                104.0,
                108.0,
                1200.0,
            ),
        ];

        let dividends = vec![CorporateAction::cash_dividend(
            "MSFT",
            2.0,
            Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
        )];

        adjust_for_dividends(&mut bars, &dividends, DividendAdjustMethod::Absolute);

        // Pre-ex bar should have $2 subtracted
        assert!((bars[0].close - 100.0).abs() < 0.01); // 102 - 2

        // Post-ex bar should remain unchanged
        assert!((bars[1].close - 108.0).abs() < 0.01);
    }

    #[test]
    fn test_adjust_for_dividends_none() {
        let mut bars = create_test_bars();
        let original_close = bars[0].close;

        let dividends = vec![CorporateAction::cash_dividend(
            "MSFT",
            2.0,
            Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
        )];

        adjust_for_dividends(&mut bars, &dividends, DividendAdjustMethod::None);

        // No adjustment should be made
        assert!((bars[0].close - original_close).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apply_adjustment_factor() {
        let mut bars = vec![
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                100.0,
                105.0,
                95.0,
                102.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(),
                50.0,
                55.0,
                48.0,
                52.0,
                2000.0,
            ),
        ];

        let factors = vec![(Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(), 2.0)];

        apply_adjustment_factor(&mut bars, &factors);

        // Bar before factor date should be divided by 2
        assert!((bars[0].close - 51.0).abs() < 0.01);
        assert!((bars[0].volume - 2000.0).abs() < 0.01);

        // Bar after factor date should remain unchanged
        assert!((bars[1].close - 52.0).abs() < 0.01);
    }

    #[test]
    fn test_cumulative_adjustment_factor() {
        let actions = vec![
            CorporateAction::split(
                "AAPL",
                4.0,
                Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(),
            ),
            CorporateAction::split(
                "AAPL",
                7.0,
                Utc.with_ymd_and_hms(2020, 8, 31, 0, 0, 0).unwrap(),
            ),
        ];

        // Before both splits
        let factor = cumulative_adjustment_factor(
            Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap(),
            &actions,
        );
        assert!((factor - 28.0).abs() < 0.01); // 4 * 7

        // After first split, before second
        let factor = cumulative_adjustment_factor(
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            &actions,
        );
        assert!((factor - 4.0).abs() < 0.01); // Only the 2024 split

        // After both splits
        let factor = cumulative_adjustment_factor(
            Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).unwrap(),
            &actions,
        );
        assert!((factor - 1.0).abs() < 0.01); // No adjustment
    }

    #[test]
    fn test_filter_actions_for_symbol() {
        let actions = vec![
            CorporateAction::split(
                "AAPL",
                4.0,
                Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            ),
            CorporateAction::cash_dividend(
                "MSFT",
                0.75,
                Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
            ),
            CorporateAction::split(
                "AAPL",
                2.0,
                Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
            ),
        ];

        let aapl_actions = filter_actions_for_symbol(&actions, "AAPL");
        assert_eq!(aapl_actions.len(), 2);

        let msft_actions = filter_actions_for_symbol(&actions, "MSFT");
        assert_eq!(msft_actions.len(), 1);

        let goog_actions = filter_actions_for_symbol(&actions, "GOOG");
        assert_eq!(goog_actions.len(), 0);
    }

    #[test]
    fn test_load_corporate_actions() {
        let mut file = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(file, "symbol,type,ex_date,value,new_symbol").unwrap();
        writeln!(file, "AAPL,split,2020-08-31,4.0,").unwrap();
        writeln!(file, "MSFT,dividend,2024-02-14,0.75,").unwrap();
        writeln!(file, "GE,spinoff,2024-04-02,0.25,GEV").unwrap();
        writeln!(file, "SIRI,reverse_split,2024-03-15,0.1,").unwrap();

        let actions = load_corporate_actions(file.path()).unwrap();

        assert_eq!(actions.len(), 4);

        // Check split
        assert_eq!(actions[0].symbol, "AAPL");
        assert!(actions[0].requires_price_adjustment());
        assert!((actions[0].adjustment_factor() - 4.0).abs() < f64::EPSILON);

        // Check dividend
        assert_eq!(actions[1].symbol, "MSFT");
        assert!(actions[1].is_dividend());
        assert!((actions[1].dividend_amount().unwrap() - 0.75).abs() < f64::EPSILON);

        // Check reverse split
        assert_eq!(actions[2].symbol, "SIRI");
        assert!(actions[2].requires_price_adjustment());
        assert!((actions[2].adjustment_factor() - 0.1).abs() < f64::EPSILON);

        // Check spinoff
        assert_eq!(actions[3].symbol, "GE");
        match &actions[3].action_type {
            CorporateActionType::SpinOff { ratio, new_symbol } => {
                assert!((*ratio - 0.25).abs() < f64::EPSILON);
                assert_eq!(new_symbol, "GEV");
            }
            _ => panic!("Expected SpinOff action type"),
        }
    }

    #[test]
    fn test_adjust_for_splits_empty() {
        let mut bars = create_test_bars();
        let original_close = bars[0].close;

        // Empty actions
        adjust_for_splits(&mut bars, &[]);
        assert!((bars[0].close - original_close).abs() < f64::EPSILON);

        // Only dividend actions (no splits)
        let dividends = vec![CorporateAction::cash_dividend(
            "AAPL",
            1.0,
            Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(),
        )];
        adjust_for_splits(&mut bars, &dividends);
        assert!((bars[0].close - original_close).abs() < f64::EPSILON);
    }

    #[test]
    fn test_multiple_splits() {
        let mut bars = vec![
            Bar::new(
                Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap(),
                280.0,
                290.0,
                275.0,
                285.0,
                1000.0,
            ),
            Bar::new(
                Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap(),
                180.0,
                185.0,
                175.0,
                182.0,
                5000.0,
            ),
        ];

        // Two splits: 7-for-1 in 2020, then 4-for-1 in 2024
        let splits = vec![
            CorporateAction::split(
                "AAPL",
                7.0,
                Utc.with_ymd_and_hms(2020, 8, 31, 0, 0, 0).unwrap(),
            ),
            CorporateAction::split(
                "AAPL",
                4.0,
                Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(),
            ),
        ];

        adjust_for_splits(&mut bars, &splits);

        // Bar from 2020 should be adjusted for both splits: 285 / 7 / 4 = 10.18
        assert!((bars[0].close - (285.0 / 7.0 / 4.0)).abs() < 0.01);

        // Bar from July 2024 (after the June 2024 split) should remain unchanged
        assert!((bars[1].close - 182.0).abs() < 0.01);
    }

    #[test]
    fn test_load_multi() {
        use std::io::Write;

        // Create temp directory with test CSV files
        let temp_dir = tempfile::tempdir().unwrap();
        let aapl_path = temp_dir.path().join("AAPL.csv");
        let msft_path = temp_dir.path().join("MSFT.csv");

        // Write test data
        let csv_content = "Date,Open,High,Low,Close,Volume
2024-01-02,100.0,105.0,99.0,103.0,1000000
2024-01-03,103.0,107.0,102.0,106.0,1200000";

        std::fs::File::create(&aapl_path)
            .unwrap()
            .write_all(csv_content.as_bytes())
            .unwrap();
        std::fs::File::create(&msft_path)
            .unwrap()
            .write_all(csv_content.as_bytes())
            .unwrap();

        // Test load_multi
        let paths: HashMap<&str, std::path::PathBuf> = [("AAPL", aapl_path), ("MSFT", msft_path)]
            .into_iter()
            .collect();

        let result = load_multi(&paths, &DataConfig::default()).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("AAPL"));
        assert!(result.contains_key("MSFT"));
        assert_eq!(result["AAPL"].len(), 2);
        assert_eq!(result["MSFT"].len(), 2);
    }

    #[test]
    fn test_load_dir() {
        use std::io::Write;

        // Create temp directory with test CSV files
        let temp_dir = tempfile::tempdir().unwrap();

        let csv_content = "Date,Open,High,Low,Close,Volume
2024-01-02,100.0,105.0,99.0,103.0,1000000";

        // Create multiple CSV files
        for symbol in &["AAPL", "GOOGL", "MSFT"] {
            let path = temp_dir.path().join(format!("{}.csv", symbol));
            std::fs::File::create(&path)
                .unwrap()
                .write_all(csv_content.as_bytes())
                .unwrap();
        }

        // Also create a non-matching file
        let txt_path = temp_dir.path().join("readme.txt");
        std::fs::File::create(txt_path)
            .unwrap()
            .write_all(b"This is not a CSV")
            .unwrap();

        // Test load_dir with glob pattern
        let result = load_dir(temp_dir.path(), "*.csv", &DataConfig::default()).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("AAPL"));
        assert!(result.contains_key("GOOGL"));
        assert!(result.contains_key("MSFT"));
    }

    #[test]
    fn test_load_dir_not_a_directory() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let result = load_dir(temp_file.path(), "*.csv", &DataConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_data_manager_load_multi() {
        use std::io::Write;

        let temp_dir = tempfile::tempdir().unwrap();

        let csv_content = "Date,Open,High,Low,Close,Volume
2024-01-02,100.0,105.0,99.0,103.0,1000000";

        // Create test files
        let paths: HashMap<&str, std::path::PathBuf> = ["AAPL", "MSFT"]
            .iter()
            .map(|symbol| {
                let path = temp_dir.path().join(format!("{}.csv", symbol));
                std::fs::File::create(&path)
                    .unwrap()
                    .write_all(csv_content.as_bytes())
                    .unwrap();
                (*symbol, path)
            })
            .collect();

        let mut manager = DataManager::new();
        manager.load_multi(&paths).unwrap();

        assert_eq!(manager.len(), 2);
        assert!(manager.contains("AAPL"));
        assert!(manager.contains("MSFT"));
    }

    #[test]
    fn test_data_manager_load_dir() {
        use std::io::Write;

        let temp_dir = tempfile::tempdir().unwrap();

        let csv_content = "Date,Open,High,Low,Close,Volume
2024-01-02,100.0,105.0,99.0,103.0,1000000";

        // Create test files
        for symbol in &["SPY", "QQQ", "IWM"] {
            let path = temp_dir.path().join(format!("{}.csv", symbol));
            std::fs::File::create(&path)
                .unwrap()
                .write_all(csv_content.as_bytes())
                .unwrap();
        }

        let mut manager = DataManager::new();
        manager.load_dir(temp_dir.path(), "*.csv").unwrap();

        assert_eq!(manager.len(), 3);
        assert!(manager.contains("SPY"));
        assert!(manager.contains("QQQ"));
        assert!(manager.contains("IWM"));
    }

    // =========================================================================
    // Sample Data Loading Tests
    // =========================================================================

    #[test]
    fn test_list_samples() {
        let samples = list_samples();
        assert!(samples.contains(&"AAPL"));
        assert!(samples.contains(&"SPY"));
        assert!(samples.contains(&"BTC"));
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_load_sample_aapl() {
        let bars = load_sample("AAPL").unwrap();

        // Should have approximately 10 years of daily data
        assert!(
            bars.len() > 2500,
            "Expected ~2600 bars for 10 years of daily data"
        );

        // First bar should be from 2014
        let first = &bars[0];
        assert_eq!(first.timestamp.year(), 2014);
        assert_eq!(first.timestamp.month(), 1);

        // Last bar should be from 2024
        let last = bars.last().unwrap();
        assert_eq!(last.timestamp.year(), 2024);

        // Verify OHLC constraints are met
        for bar in &bars {
            assert!(bar.high >= bar.open, "High must be >= Open");
            assert!(bar.high >= bar.close, "High must be >= Close");
            assert!(bar.low <= bar.open, "Low must be <= Open");
            assert!(bar.low <= bar.close, "Low must be <= Close");
            assert!(bar.volume > 0.0, "Volume must be positive");
        }
    }

    #[test]
    fn test_load_sample_spy() {
        let bars = load_sample("SPY").unwrap();

        // Should have approximately 10 years of daily data
        assert!(
            bars.len() > 2500,
            "Expected ~2600 bars for 10 years of daily data"
        );

        // Verify all bars are sorted by timestamp
        for window in bars.windows(2) {
            assert!(
                window[0].timestamp < window[1].timestamp,
                "Bars should be sorted by timestamp"
            );
        }
    }

    #[test]
    fn test_load_sample_btc() {
        let bars = load_sample("BTC").unwrap();

        // BTC includes weekends so should have more bars than stocks
        assert!(
            bars.len() > 3500,
            "Expected ~3650 bars for 10 years of daily BTC data (including weekends)"
        );

        // Verify all bars have positive prices
        for bar in &bars {
            assert!(bar.open > 0.0);
            assert!(bar.high > 0.0);
            assert!(bar.low > 0.0);
            assert!(bar.close > 0.0);
        }
    }

    #[test]
    fn test_load_sample_case_insensitive() {
        // Test lowercase
        let bars_lower = load_sample("aapl").unwrap();
        // Test uppercase
        let bars_upper = load_sample("AAPL").unwrap();
        // Test mixed case
        let bars_mixed = load_sample("Aapl").unwrap();

        assert_eq!(bars_lower.len(), bars_upper.len());
        assert_eq!(bars_lower.len(), bars_mixed.len());
    }

    #[test]
    fn test_load_sample_unknown() {
        let result = load_sample("UNKNOWN");
        assert!(result.is_err());

        let err = result.unwrap_err();
        let err_str = format!("{}", err);
        assert!(err_str.contains("Unknown sample"));
        assert!(err_str.contains("AAPL")); // Should suggest available samples
    }

    #[test]
    fn test_detect_delimiter_comma() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp file with comma-separated data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "date,open,high,low,close,volume").unwrap();
        writeln!(file, "2024-01-01,100.0,105.0,99.0,104.0,1000").unwrap();
        writeln!(file, "2024-01-02,104.0,108.0,103.0,107.0,1500").unwrap();
        file.flush().unwrap();

        let delimiter = detect_delimiter(file.path()).unwrap();
        assert_eq!(delimiter, b',');
    }

    #[test]
    fn test_detect_delimiter_semicolon() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp file with semicolon-separated data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "date;open;high;low;close;volume").unwrap();
        writeln!(file, "2024-01-01;100.0;105.0;99.0;104.0;1000").unwrap();
        writeln!(file, "2024-01-02;104.0;108.0;103.0;107.0;1500").unwrap();
        file.flush().unwrap();

        let delimiter = detect_delimiter(file.path()).unwrap();
        assert_eq!(delimiter, b';');
    }

    #[test]
    fn test_detect_delimiter_tab() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp file with tab-separated data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "date\topen\thigh\tlow\tclose\tvolume").unwrap();
        writeln!(file, "2024-01-01\t100.0\t105.0\t99.0\t104.0\t1000").unwrap();
        writeln!(file, "2024-01-02\t104.0\t108.0\t103.0\t107.0\t1500").unwrap();
        file.flush().unwrap();

        let delimiter = detect_delimiter(file.path()).unwrap();
        assert_eq!(delimiter, b'\t');
    }

    #[test]
    fn test_load_csv_auto_detect_semicolon() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create temp file with semicolon-separated data
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "date;open;high;low;close;volume").unwrap();
        writeln!(file, "2024-01-01;100.0;105.0;99.0;104.0;1000").unwrap();
        writeln!(file, "2024-01-02;104.0;108.0;103.0;107.0;1500").unwrap();
        file.flush().unwrap();

        // Load with auto-detection (default config)
        let bars = load_csv(file.path(), &DataConfig::default()).unwrap();
        assert_eq!(bars.len(), 2);
        assert_eq!(bars[0].open, 100.0);
        assert_eq!(bars[1].close, 107.0);
    }
}
