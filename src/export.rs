//! Export utilities for backtest results.
//!
//! This module provides various export formats for:
//! - Equity curves (CSV, JSON)
//! - Trade logs (CSV, JSON)
//! - Performance reports (HTML, Markdown)
//! - Feature data for ML pipelines
//!
//! # Example
//!
//! ```ignore
//! use mantis::export::{ExportConfig, Exporter};
//! use mantis::engine::BacktestResult;
//!
//! let result: BacktestResult = /* from backtest */;
//! let exporter = Exporter::new(result);
//!
//! // Export to multiple formats
//! exporter.export_equity_csv("equity.csv")?;
//! exporter.export_trades_csv("trades.csv")?;
//! exporter.export_report_md("report.md")?;
//! ```

use crate::engine::BacktestResult;
use crate::error::{BacktestError, Result};
use crate::multi_asset::MultiAssetResult;
use crate::types::EquityPoint;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Configuration for exports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Date format for CSV exports.
    pub date_format: String,
    /// Decimal precision for numeric values.
    pub precision: usize,
    /// Whether to include headers in CSV.
    pub include_headers: bool,
    /// Delimiter for CSV files.
    pub delimiter: char,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            date_format: "%Y-%m-%d %H:%M:%S".to_string(),
            precision: 4,
            include_headers: true,
            delimiter: ',',
        }
    }
}

/// Exporter for backtest results.
pub struct Exporter {
    result: BacktestResult,
    config: ExportConfig,
}

impl Exporter {
    /// Create a new exporter.
    pub fn new(result: BacktestResult) -> Self {
        Self {
            result,
            config: ExportConfig::default(),
        }
    }

    /// Create exporter with custom config.
    pub fn with_config(result: BacktestResult, config: ExportConfig) -> Self {
        Self { result, config }
    }

    /// Export equity curve to CSV.
    pub fn export_equity_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        if self.config.include_headers {
            writeln!(
                writer,
                "timestamp,equity,cash,positions_value,drawdown,drawdown_pct"
            )?;
        }

        let prec = self.config.precision;
        for point in &self.result.equity_curve {
            writeln!(
                writer,
                "{},{:.prec$},{:.prec$},{:.prec$},{:.prec$},{:.prec$}",
                point.timestamp.format(&self.config.date_format),
                point.equity,
                point.cash,
                point.positions_value,
                point.drawdown,
                point.drawdown_pct,
            )?;
        }

        Ok(())
    }

    /// Export trades to CSV.
    pub fn export_trades_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        if self.config.include_headers {
            writeln!(
                writer,
                "trade_num,symbol,side,quantity,entry_price,entry_time,exit_price,exit_time,pnl,pnl_pct,duration_hours"
            )?;
        }

        let mut trade_num = 0;
        for trade in &self.result.trades {
            if !trade.is_closed() {
                continue;
            }
            trade_num += 1;

            let exit_price = trade.exit_price.unwrap_or(0.0);
            let exit_time = trade
                .exit_time
                .map(|t| t.format(&self.config.date_format).to_string())
                .unwrap_or_default();
            let pnl = trade.net_pnl().unwrap_or(0.0);
            let pnl_pct = trade.return_pct().unwrap_or(0.0);
            let duration = trade
                .exit_time
                .map(|exit| {
                    let dur = exit - trade.entry_time;
                    dur.num_hours() as f64 + dur.num_minutes() as f64 / 60.0
                })
                .unwrap_or(0.0);

            writeln!(
                writer,
                "{},{},{:?},{:.4},{:.4},{},{:.4},{},{:.4},{:.4},{:.2}",
                trade_num,
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.entry_price,
                trade.entry_time.format(&self.config.date_format),
                exit_price,
                exit_time,
                pnl,
                pnl_pct,
                duration,
            )?;
        }

        Ok(())
    }

    /// Export trades to JSON.
    pub fn export_trades_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let closed_trades: Vec<_> = self
            .result
            .trades
            .iter()
            .filter(|t| t.is_closed())
            .collect();
        serde_json::to_writer_pretty(file, &closed_trades)?;
        Ok(())
    }

    /// Generate performance summary as JSON.
    pub fn export_summary_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let summary = PerformanceSummary::from_result(&self.result);
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &summary)?;
        Ok(())
    }

    /// Generate markdown report.
    pub fn export_report_md(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let summary = PerformanceSummary::from_result(&self.result);

        writeln!(writer, "# Backtest Report: {}", self.result.strategy_name)?;
        writeln!(writer)?;
        writeln!(
            writer,
            "**Period:** {} to {}",
            self.result.start_time.format("%Y-%m-%d"),
            self.result.end_time.format("%Y-%m-%d")
        )?;
        writeln!(writer, "**Symbols:** {}", self.result.symbols.join(", "))?;
        writeln!(writer)?;

        writeln!(writer, "## Performance Summary")?;
        writeln!(writer)?;
        writeln!(writer, "| Metric | Value |")?;
        writeln!(writer, "|--------|-------|")?;
        writeln!(
            writer,
            "| Initial Capital | ${:.2} |",
            summary.initial_capital
        )?;
        writeln!(writer, "| Final Equity | ${:.2} |", summary.final_equity)?;
        writeln!(
            writer,
            "| Total Return | {:.2}% |",
            summary.total_return_pct
        )?;
        writeln!(
            writer,
            "| Annual Return | {:.2}% |",
            summary.annual_return_pct
        )?;
        writeln!(
            writer,
            "| Max Drawdown | {:.2}% |",
            summary.max_drawdown_pct
        )?;
        writeln!(writer, "| Sharpe Ratio | {:.2} |", summary.sharpe_ratio)?;
        writeln!(writer, "| Sortino Ratio | {:.2} |", summary.sortino_ratio)?;
        writeln!(writer, "| Calmar Ratio | {:.2} |", summary.calmar_ratio)?;
        writeln!(writer)?;

        writeln!(writer, "## Trade Statistics")?;
        writeln!(writer)?;
        writeln!(writer, "| Metric | Value |")?;
        writeln!(writer, "|--------|-------|")?;
        writeln!(writer, "| Total Trades | {} |", summary.total_trades)?;
        writeln!(writer, "| Winning Trades | {} |", summary.winning_trades)?;
        writeln!(writer, "| Losing Trades | {} |", summary.losing_trades)?;
        writeln!(writer, "| Win Rate | {:.1}% |", summary.win_rate)?;
        writeln!(writer, "| Average Win | ${:.2} |", summary.avg_win)?;
        writeln!(writer, "| Average Loss | ${:.2} |", summary.avg_loss)?;
        writeln!(writer, "| Profit Factor | {:.2} |", summary.profit_factor)?;
        writeln!(writer)?;

        writeln!(
            writer,
            "*Report generated on {}*",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        Ok(())
    }

    /// Get string representation of trades as CSV.
    pub fn trades_to_csv(&self) -> String {
        let mut output = String::new();

        if self.config.include_headers {
            output.push_str(
                "trade_num,symbol,side,quantity,entry_price,entry_time,exit_price,exit_time,pnl,pnl_pct\n",
            );
        }

        let mut trade_num = 0;
        for trade in &self.result.trades {
            if !trade.is_closed() {
                continue;
            }
            trade_num += 1;

            let line = format!(
                "{},{},{:?},{:.4},{:.4},{},{:.4},{},{:.4},{:.4}\n",
                trade_num,
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.entry_price,
                trade.entry_time.format(&self.config.date_format),
                trade.exit_price.unwrap_or(0.0),
                trade
                    .exit_time
                    .map(|t| t.format(&self.config.date_format).to_string())
                    .unwrap_or_default(),
                trade.net_pnl().unwrap_or(0.0),
                trade.return_pct().unwrap_or(0.0),
            );
            output.push_str(&line);
        }

        output
    }
}

/// Performance summary structure for JSON export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub strategy_name: String,
    pub symbols: Vec<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub trading_days: i64,
    pub initial_capital: f64,
    pub final_equity: f64,
    pub total_return_pct: f64,
    pub annual_return_pct: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
}

impl PerformanceSummary {
    /// Create summary from backtest result.
    pub fn from_result(result: &BacktestResult) -> Self {
        Self {
            strategy_name: result.strategy_name.clone(),
            symbols: result.symbols.clone(),
            start_time: result.start_time,
            end_time: result.end_time,
            trading_days: result.trading_days as i64,
            initial_capital: result.initial_capital,
            final_equity: result.final_equity,
            total_return_pct: result.total_return_pct,
            annual_return_pct: result.annual_return_pct,
            max_drawdown_pct: result.max_drawdown_pct,
            sharpe_ratio: result.sharpe_ratio,
            sortino_ratio: result.sortino_ratio,
            calmar_ratio: result.calmar_ratio,
            total_trades: result.total_trades,
            winning_trades: result.winning_trades,
            losing_trades: result.losing_trades,
            win_rate: result.win_rate,
            avg_win: result.avg_win,
            avg_loss: result.avg_loss,
            profit_factor: result.profit_factor,
        }
    }
}

/// Export equity curve data to CSV.
pub fn export_equity_curve_csv(equity_curve: &[EquityPoint], path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "timestamp,equity,drawdown_pct")?;

    for point in equity_curve {
        writeln!(
            writer,
            "{},{:.4},{:.4}",
            point.timestamp.format("%Y-%m-%d %H:%M:%S"),
            point.equity,
            point.drawdown_pct
        )?;
    }

    Ok(())
}

/// Export multiple backtest results comparison to CSV.
pub fn export_comparison_csv(
    results: &[(String, BacktestResult)],
    path: impl AsRef<Path>,
) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "name,total_return_pct,annual_return_pct,max_drawdown_pct,sharpe_ratio,sortino_ratio,calmar_ratio,total_trades,win_rate,profit_factor"
    )?;

    for (name, result) in results {
        writeln!(
            writer,
            "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{:.2},{:.4}",
            name,
            result.total_return_pct,
            result.annual_return_pct,
            result.max_drawdown_pct,
            result.sharpe_ratio,
            result.sortino_ratio,
            result.calmar_ratio,
            result.total_trades,
            result.win_rate,
            result.profit_factor
        )?;
    }

    Ok(())
}

/// Generate numpy-compatible binary export of feature matrix.
pub fn export_features_npy(features: &[Vec<f64>], path: impl AsRef<Path>) -> Result<()> {
    if features.is_empty() {
        return Err(BacktestError::DataError(
            "No features to export".to_string(),
        ));
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Simple NPY format header
    let rows = features.len();
    let cols = features[0].len();

    // NPY magic number and version
    writer.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y'])?;
    writer.write_all(&[0x01, 0x00])?; // Version 1.0

    // Header dict
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
        rows, cols
    );

    // Pad header to multiple of 64 bytes
    let header_len = header.len() + 1; // +1 for newline
    let padding = 64 - ((10 + header_len) % 64);
    let total_header_len = (header_len + padding) as u16;

    writer.write_all(&total_header_len.to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for _ in 0..padding {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    // Write data
    for row in features {
        for &val in row {
            writer.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Export feature matrix to Parquet format for efficient ML data loading.
///
/// Parquet is a columnar storage format that's efficient for ML workflows:
/// - Compressed storage (typically 2-5x smaller than CSV)
/// - Fast random access and column-wise operations
/// - Native support in pandas, PyArrow, and most ML frameworks
///
/// # Arguments
/// * `features` - 2D feature matrix (rows x columns)
/// * `column_names` - Names for each column (must match feature columns)
/// * `path` - Output file path (.parquet extension recommended)
///
/// # Example
/// ```ignore
/// let features = vec![
///     vec![1.0, 0.5, -0.2],
///     vec![1.1, 0.6, -0.1],
/// ];
/// let columns = vec!["returns", "rsi", "macd"];
/// export_features_parquet(&features, &columns, "features.parquet")?;
/// ```
pub fn export_features_parquet(
    features: &[Vec<f64>],
    column_names: &[impl AsRef<str>],
    path: impl AsRef<Path>,
) -> Result<()> {
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    if features.is_empty() {
        return Err(BacktestError::DataError(
            "No features to export".to_string(),
        ));
    }

    let num_cols = features[0].len();

    if column_names.len() != num_cols {
        return Err(BacktestError::DataError(format!(
            "Column names count ({}) doesn't match feature columns ({})",
            column_names.len(),
            num_cols
        )));
    }

    // Create schema
    let fields: Vec<Field> = column_names
        .iter()
        .map(|name| Field::new(name.as_ref(), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    // Create column arrays
    let arrays: Vec<Arc<dyn arrow::array::Array>> = (0..num_cols)
        .map(|col_idx| {
            let values: Vec<f64> = features.iter().map(|row| row[col_idx]).collect();
            Arc::new(Float64Array::from(values)) as Arc<dyn arrow::array::Array>
        })
        .collect();

    // Create record batch
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| BacktestError::DataError(format!("Failed to create record batch: {}", e)))?;

    // Write to parquet file
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| BacktestError::DataError(format!("Failed to create parquet writer: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| BacktestError::DataError(format!("Failed to write parquet: {}", e)))?;

    writer
        .close()
        .map_err(|e| BacktestError::DataError(format!("Failed to close parquet file: {}", e)))?;

    Ok(())
}

/// Export equity curve to Parquet format.
pub fn export_equity_curve_parquet(
    equity_curve: &[EquityPoint],
    path: impl AsRef<Path>,
) -> Result<()> {
    use arrow::array::{Float64Array, TimestampMillisecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    if equity_curve.is_empty() {
        return Err(BacktestError::DataError(
            "No equity curve data to export".to_string(),
        ));
    }

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "timestamp",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("equity", DataType::Float64, false),
        Field::new("cash", DataType::Float64, false),
        Field::new("positions_value", DataType::Float64, false),
        Field::new("drawdown", DataType::Float64, false),
        Field::new("drawdown_pct", DataType::Float64, false),
    ]));

    // Create arrays
    let timestamps: Vec<i64> = equity_curve
        .iter()
        .map(|p| p.timestamp.timestamp_millis())
        .collect();
    let equity: Vec<f64> = equity_curve.iter().map(|p| p.equity).collect();
    let cash: Vec<f64> = equity_curve.iter().map(|p| p.cash).collect();
    let positions_value: Vec<f64> = equity_curve.iter().map(|p| p.positions_value).collect();
    let drawdown: Vec<f64> = equity_curve.iter().map(|p| p.drawdown).collect();
    let drawdown_pct: Vec<f64> = equity_curve.iter().map(|p| p.drawdown_pct).collect();

    let arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(TimestampMillisecondArray::from(timestamps)),
        Arc::new(Float64Array::from(equity)),
        Arc::new(Float64Array::from(cash)),
        Arc::new(Float64Array::from(positions_value)),
        Arc::new(Float64Array::from(drawdown)),
        Arc::new(Float64Array::from(drawdown_pct)),
    ];

    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| BacktestError::DataError(format!("Failed to create record batch: {}", e)))?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| BacktestError::DataError(format!("Failed to create parquet writer: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| BacktestError::DataError(format!("Failed to write parquet: {}", e)))?;

    writer
        .close()
        .map_err(|e| BacktestError::DataError(format!("Failed to close parquet file: {}", e)))?;

    Ok(())
}

/// Export trades to Parquet format.
pub fn export_trades_parquet(trades: &[crate::types::Trade], path: impl AsRef<Path>) -> Result<()> {
    use arrow::array::{Float64Array, Int64Array, StringBuilder, TimestampMillisecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let closed_trades: Vec<_> = trades.iter().filter(|t| t.is_closed()).collect();

    if closed_trades.is_empty() {
        return Err(BacktestError::DataError(
            "No closed trades to export".to_string(),
        ));
    }

    // Create schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("entry_price", DataType::Float64, false),
        Field::new(
            "entry_time",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("exit_price", DataType::Float64, false),
        Field::new(
            "exit_time",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            false,
        ),
        Field::new("pnl", DataType::Float64, false),
        Field::new("pnl_pct", DataType::Float64, false),
        Field::new("commission", DataType::Float64, false),
        Field::new("duration_hours", DataType::Int64, false),
    ]));

    // Build arrays
    let mut symbols = StringBuilder::new();
    let mut sides = StringBuilder::new();
    let mut quantities = Vec::new();
    let mut entry_prices = Vec::new();
    let mut entry_times = Vec::new();
    let mut exit_prices = Vec::new();
    let mut exit_times = Vec::new();
    let mut pnls = Vec::new();
    let mut pnl_pcts = Vec::new();
    let mut commissions = Vec::new();
    let mut durations = Vec::new();

    for trade in &closed_trades {
        symbols.append_value(&trade.symbol);
        sides.append_value(format!("{:?}", trade.side));
        quantities.push(trade.quantity);
        entry_prices.push(trade.entry_price);
        entry_times.push(trade.entry_time.timestamp_millis());
        exit_prices.push(trade.exit_price.unwrap_or(0.0));
        exit_times.push(trade.exit_time.map(|t| t.timestamp_millis()).unwrap_or(0));
        pnls.push(trade.net_pnl().unwrap_or(0.0));
        pnl_pcts.push(trade.return_pct().unwrap_or(0.0));
        commissions.push(trade.commission);
        let duration_hours = trade.holding_period().map(|d| d.num_hours()).unwrap_or(0);
        durations.push(duration_hours);
    }

    let arrays: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(symbols.finish()),
        Arc::new(sides.finish()),
        Arc::new(Float64Array::from(quantities)),
        Arc::new(Float64Array::from(entry_prices)),
        Arc::new(TimestampMillisecondArray::from(entry_times)),
        Arc::new(Float64Array::from(exit_prices)),
        Arc::new(TimestampMillisecondArray::from(exit_times)),
        Arc::new(Float64Array::from(pnls)),
        Arc::new(Float64Array::from(pnl_pcts)),
        Arc::new(Float64Array::from(commissions)),
        Arc::new(Int64Array::from(durations)),
    ];

    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| BacktestError::DataError(format!("Failed to create record batch: {}", e)))?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|e| BacktestError::DataError(format!("Failed to create parquet writer: {}", e)))?;

    writer
        .write(&batch)
        .map_err(|e| BacktestError::DataError(format!("Failed to write parquet: {}", e)))?;

    writer
        .close()
        .map_err(|e| BacktestError::DataError(format!("Failed to close parquet file: {}", e)))?;

    Ok(())
}

/// Multi-asset result exporter.
pub struct MultiAssetExporter {
    result: MultiAssetResult,
    config: ExportConfig,
}

impl MultiAssetExporter {
    /// Create a new multi-asset exporter.
    pub fn new(result: MultiAssetResult) -> Self {
        Self {
            result,
            config: ExportConfig::default(),
        }
    }

    /// Create exporter with custom config.
    pub fn with_config(result: MultiAssetResult, config: ExportConfig) -> Self {
        Self { result, config }
    }

    /// Export weight history to CSV.
    pub fn export_weights_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        if self.config.include_headers {
            write!(writer, "timestamp")?;
            for symbol in &self.result.symbols {
                write!(writer, "{}{}", self.config.delimiter, symbol)?;
            }
            writeln!(writer)?;
        }

        // Write data
        for (timestamp, weights) in &self.result.weight_history {
            write!(writer, "{}", timestamp.format(&self.config.date_format))?;
            for symbol in &self.result.symbols {
                let weight = weights.get(symbol).copied().unwrap_or(0.0);
                write!(
                    writer,
                    "{}{:.prec$}",
                    self.config.delimiter,
                    weight,
                    prec = self.config.precision
                )?;
            }
            writeln!(writer)?;
        }

        Ok(())
    }

    /// Export equity curve to CSV.
    pub fn export_equity_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        export_equity_curve_csv(&self.result.equity_curve, path)
    }

    /// Generate markdown report.
    pub fn export_report_md(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        writeln!(
            writer,
            "# Multi-Asset Backtest Report: {}",
            self.result.strategy_name
        )?;
        writeln!(writer)?;
        writeln!(
            writer,
            "**Period:** {} to {}",
            self.result.start_time.format("%Y-%m-%d"),
            self.result.end_time.format("%Y-%m-%d")
        )?;
        writeln!(writer, "**Assets:** {}", self.result.symbols.join(", "))?;
        writeln!(writer)?;

        writeln!(writer, "## Performance Summary")?;
        writeln!(writer)?;
        writeln!(writer, "| Metric | Value |")?;
        writeln!(writer, "|--------|-------|")?;
        writeln!(
            writer,
            "| Initial Capital | ${:.2} |",
            self.result.initial_capital
        )?;
        writeln!(
            writer,
            "| Final Equity | ${:.2} |",
            self.result.final_equity
        )?;
        writeln!(
            writer,
            "| Total Return | {:.2}% |",
            self.result.total_return_pct
        )?;
        writeln!(
            writer,
            "| Annual Return | {:.2}% |",
            self.result.annual_return_pct
        )?;
        writeln!(
            writer,
            "| Max Drawdown | {:.2}% |",
            self.result.max_drawdown_pct
        )?;
        writeln!(writer, "| Sharpe Ratio | {:.2} |", self.result.sharpe_ratio)?;
        writeln!(
            writer,
            "| Sortino Ratio | {:.2} |",
            self.result.sortino_ratio
        )?;
        writeln!(writer, "| Total Trades | {} |", self.result.total_trades)?;
        writeln!(writer)?;

        writeln!(writer, "## Trades by Symbol")?;
        writeln!(writer)?;
        writeln!(writer, "| Symbol | Trades |")?;
        writeln!(writer, "|--------|--------|")?;
        for symbol in &self.result.symbols {
            let trades = self
                .result
                .trades_by_symbol
                .get(symbol)
                .copied()
                .unwrap_or(0);
            writeln!(writer, "| {} | {} |", symbol, trades)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BacktestConfig;
    use crate::types::{Side, Trade};
    use chrono::{Duration, TimeZone};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    fn create_test_result() -> BacktestResult {
        let mut trades = vec![];

        // Add a closed trade using the Trade::open constructor
        let mut trade = crate::types::Trade::open(
            "TEST",
            Side::Buy,
            100.0,
            100.0,
            Utc.with_ymd_and_hms(2024, 1, 1, 10, 0, 0).unwrap(),
            1.0, // commission
            0.0, // slippage
        );
        trade.close(
            110.0,
            Utc.with_ymd_and_hms(2024, 1, 5, 15, 0, 0).unwrap(),
            1.0,
        );
        trades.push(trade);

        BacktestResult {
            strategy_name: "Test Strategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 15.0,
            trading_days: 252,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 1000.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 5.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            calmar_ratio: 3.0,
            trades,
            equity_curve: vec![],
            start_time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            end_time: Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(),
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
        }
    }

    fn create_test_multi_asset_result() -> MultiAssetResult {
        let start_time = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end_time = Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap();

        let mut trade_aapl = Trade::open("AAPL", Side::Buy, 50.0, 200.0, start_time, 1.0, 0.0);
        trade_aapl.close(210.0, end_time, 1.0);

        let mut trade_msft = Trade::open("MSFT", Side::Sell, 25.0, 300.0, start_time, 1.0, 0.0);
        trade_msft.close(290.0, end_time, 1.0);

        let trades = vec![trade_aapl, trade_msft];

        let mut trades_by_symbol = HashMap::new();
        trades_by_symbol.insert("AAPL".to_string(), 1);
        trades_by_symbol.insert("MSFT".to_string(), 1);

        let equity_curve = vec![
            EquityPoint {
                timestamp: start_time,
                equity: 1_000_000.0,
                cash: 1_000_000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: start_time + Duration::days(1),
                equity: 1_030_000.0,
                cash: 600_000.0,
                positions_value: 430_000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut weights_day1 = HashMap::new();
        weights_day1.insert("AAPL".to_string(), 0.6);
        weights_day1.insert("MSFT".to_string(), 0.4);

        let mut weights_day2 = HashMap::new();
        weights_day2.insert("AAPL".to_string(), 0.5);
        weights_day2.insert("MSFT".to_string(), 0.5);

        let weight_history = vec![(start_time, weights_day1), (end_time, weights_day2)];

        MultiAssetResult {
            strategy_name: "MultiAsset Strategy".to_string(),
            symbols: vec!["AAPL".to_string(), "MSFT".to_string()],
            initial_capital: 1_000_000.0,
            final_equity: 1_030_000.0,
            total_return_pct: 3.0,
            annual_return_pct: 4.5,
            max_drawdown_pct: 1.2,
            sharpe_ratio: 1.1,
            sortino_ratio: 1.3,
            total_trades: trades.len(),
            trades_by_symbol,
            trades,
            equity_curve,
            start_time,
            end_time,
            weight_history,
        }
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.precision, 4);
        assert!(config.include_headers);
    }

    #[test]
    fn test_trades_to_csv() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let csv = exporter.trades_to_csv();
        assert!(csv.contains("trade_num,symbol,side"));
        assert!(csv.contains("TEST"));
        assert!(csv.contains("Buy"));
    }

    #[test]
    fn test_export_trades_csv() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_trades_csv(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("trade_num,symbol,side"));
    }

    #[test]
    fn test_export_trades_json() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_trades_json(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("TEST"));
    }

    #[test]
    fn test_export_summary_json() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_summary_json(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("sharpe_ratio"));
        assert!(content.contains("total_trades"));
    }

    #[test]
    fn test_export_report_md() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_md(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("# Backtest Report"));
        assert!(content.contains("Performance Summary"));
        assert!(content.contains("Trade Statistics"));
    }

    #[test]
    fn test_multi_asset_export_weights_csv() {
        let result = create_test_multi_asset_result();
        let exporter = MultiAssetExporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_weights_csv(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("timestamp,AAPL,MSFT"));
        assert!(content.contains("0.6000,0.4000"));
        assert!(content.contains("0.5000,0.5000"));
    }

    #[test]
    fn test_multi_asset_export_equity_csv() {
        let result = create_test_multi_asset_result();
        let exporter = MultiAssetExporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_equity_csv(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("timestamp,equity,drawdown_pct"));
        assert!(content.contains("1000000.0000"));
        assert!(content.contains("1030000.0000"));
    }

    #[test]
    fn test_multi_asset_export_report_md() {
        let result = create_test_multi_asset_result();
        let exporter = MultiAssetExporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_md(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("# Multi-Asset Backtest Report: MultiAsset Strategy"));
        assert!(content.contains("**Assets:** AAPL, MSFT"));
        assert!(content.contains("| Symbol | Trades |"));
        assert!(content.contains("| AAPL | 1 |"));
        assert!(content.contains("| MSFT | 1 |"));
    }

    #[test]
    fn test_performance_summary() {
        let result = create_test_result();
        let summary = PerformanceSummary::from_result(&result);

        assert_eq!(summary.strategy_name, "Test Strategy");
        assert!((summary.total_return_pct - 10.0).abs() < 0.001);
        assert_eq!(summary.total_trades, 1);
    }

    #[test]
    fn test_export_equity_curve() {
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                equity: 101000.0,
                cash: 1000.0,
                positions_value: 100000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                equity: 99000.0,
                cash: 1000.0,
                positions_value: 98000.0,
                drawdown: 2000.0,
                drawdown_pct: 1.98,
            },
        ];

        let file = NamedTempFile::new().unwrap();
        export_equity_curve_csv(&equity_curve, file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.contains("timestamp,equity,drawdown_pct"));
        assert!(content.contains("100000.0000"));
    }

    #[test]
    fn test_export_features_npy() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let file = NamedTempFile::new().unwrap();
        export_features_npy(&features, file.path()).unwrap();

        // Check file was created and has content
        let metadata = std::fs::metadata(file.path()).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_export_features_parquet() {
        let features = vec![
            vec![1.0, 0.5, -0.2],
            vec![1.1, 0.6, -0.1],
            vec![1.2, 0.4, 0.0],
        ];
        let columns = vec!["returns", "rsi", "macd"];

        let file = NamedTempFile::new().unwrap();
        export_features_parquet(&features, &columns, file.path()).unwrap();

        // Check file was created and has content
        let metadata = std::fs::metadata(file.path()).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_export_equity_curve_parquet() {
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 50000.0,
                positions_value: 50000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0).unwrap(),
                equity: 101000.0,
                cash: 51000.0,
                positions_value: 50000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let file = NamedTempFile::new().unwrap();
        export_equity_curve_parquet(&equity_curve, file.path()).unwrap();

        // Check file was created and has content
        let metadata = std::fs::metadata(file.path()).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_export_trades_parquet() {
        let mut trade = Trade::open(
            "TEST",
            Side::Buy,
            100.0,
            100.0,
            Utc.with_ymd_and_hms(2024, 1, 1, 10, 0, 0).unwrap(),
            1.0,
            0.0,
        );
        trade.close(
            110.0,
            Utc.with_ymd_and_hms(2024, 1, 5, 15, 0, 0).unwrap(),
            1.0,
        );

        let trades = vec![trade];

        let file = NamedTempFile::new().unwrap();
        export_trades_parquet(&trades, file.path()).unwrap();

        // Check file was created and has content
        let metadata = std::fs::metadata(file.path()).unwrap();
        assert!(metadata.len() > 0);
    }
}
