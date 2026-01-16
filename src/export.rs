//! Export utilities for backtest results.
//!
//! This module provides various export formats for:
//! - Equity curves (CSV, JSON, Parquet)
//! - Trade logs (CSV, JSON, Parquet)
//! - Performance reports (HTML)
//!
//! **DEPRECATION NOTICE**: The following export formats are deprecated:
//! - NPY export (`export_features_npy`) - Use Parquet instead, which is faster and more portable
//! - Markdown reports (`export_report_md`) - Use HTML reports instead, which support interactivity
//!
//! # Supported Formats
//!
//! | Format | Use Case |
//! |--------|----------|
//! | CSV | Universal compatibility, human-readable |
//! | JSON | API integration, structured data |
//! | Parquet | ML pipelines, columnar efficiency |
//! | HTML | Interactive reports with charts |
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
//! // Export to supported formats
//! exporter.export_equity_csv("equity.csv")?;
//! exporter.export_trades_csv("trades.csv")?;
//! exporter.export_report_html("report.html")?;  // Preferred over deprecated Markdown
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

/// Generate self-contained HTML report.
///
/// Creates an HTML file with embedded CSS and SVG charts that can be viewed
/// in any browser without external dependencies.
///
/// # Report Sections
/// - Summary metrics table
/// - Equity curve chart (SVG)
/// - Drawdown chart (SVG)
/// - Trade list
///
/// # Example
/// ```ignore
/// let exporter = Exporter::new(result);
/// exporter.export_report_html("report.html")?;
/// ```
impl Exporter {
    /// Export a self-contained HTML report.
    pub fn export_report_html(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let summary = PerformanceSummary::from_result(&self.result);

        // Write HTML header and styles
        write!(writer, "{}", html_header(&self.result.strategy_name))?;

        // Summary section
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Summary</h2>")?;
        writeln!(writer, "    <div class=\"meta\">")?;
        writeln!(
            writer,
            "      <p><strong>Period:</strong> {} to {}</p>",
            self.result.start_time.format("%Y-%m-%d"),
            self.result.end_time.format("%Y-%m-%d")
        )?;
        writeln!(
            writer,
            "      <p><strong>Symbols:</strong> {}</p>",
            self.result.symbols.join(", ")
        )?;
        writeln!(
            writer,
            "      <p><strong>Trading Days:</strong> {}</p>",
            summary.trading_days
        )?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Performance metrics
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Performance Metrics</h2>")?;
        writeln!(writer, "    <div class=\"metrics-grid\">")?;
        write_metric_card(
            &mut writer,
            "Initial Capital",
            &format!("${:.2}", summary.initial_capital),
            "neutral",
        )?;
        write_metric_card(
            &mut writer,
            "Final Equity",
            &format!("${:.2}", summary.final_equity),
            if summary.final_equity >= summary.initial_capital {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Total Return",
            &format!("{:.2}%", summary.total_return_pct),
            if summary.total_return_pct >= 0.0 {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Annual Return",
            &format!("{:.2}%", summary.annual_return_pct),
            if summary.annual_return_pct >= 0.0 {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Sharpe Ratio",
            &format!("{:.2}", summary.sharpe_ratio),
            sharpe_color(summary.sharpe_ratio),
        )?;
        write_metric_card(
            &mut writer,
            "Sortino Ratio",
            &format!("{:.2}", summary.sortino_ratio),
            if summary.sortino_ratio >= 1.0 {
                "positive"
            } else {
                "neutral"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Calmar Ratio",
            &format!("{:.2}", summary.calmar_ratio),
            if summary.calmar_ratio >= 1.0 {
                "positive"
            } else {
                "neutral"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Max Drawdown",
            &format!("{:.2}%", summary.max_drawdown_pct),
            "negative",
        )?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Trade statistics
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Trade Statistics</h2>")?;
        writeln!(writer, "    <div class=\"metrics-grid\">")?;
        write_metric_card(
            &mut writer,
            "Total Trades",
            &format!("{}", summary.total_trades),
            "neutral",
        )?;
        write_metric_card(
            &mut writer,
            "Winning Trades",
            &format!("{}", summary.winning_trades),
            "positive",
        )?;
        write_metric_card(
            &mut writer,
            "Losing Trades",
            &format!("{}", summary.losing_trades),
            "negative",
        )?;
        write_metric_card(
            &mut writer,
            "Win Rate",
            &format!("{:.1}%", summary.win_rate),
            if summary.win_rate >= 50.0 {
                "positive"
            } else {
                "neutral"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Avg Win",
            &format!("${:.2}", summary.avg_win),
            "positive",
        )?;
        write_metric_card(
            &mut writer,
            "Avg Loss",
            &format!("${:.2}", summary.avg_loss),
            "negative",
        )?;
        write_metric_card(
            &mut writer,
            "Profit Factor",
            &format!(
                "{:.2}",
                if summary.profit_factor.is_infinite() {
                    999.99
                } else {
                    summary.profit_factor
                }
            ),
            if summary.profit_factor >= 1.5 {
                "positive"
            } else {
                "neutral"
            },
        )?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Equity curve chart
        if !self.result.equity_curve.is_empty() {
            writeln!(writer, "  <div class=\"section\">")?;
            writeln!(writer, "    <h2>Equity Curve</h2>")?;
            writeln!(writer, "    <div class=\"chart-container\">")?;
            write_equity_curve_svg(&mut writer, &self.result.equity_curve)?;
            writeln!(writer, "    </div>")?;
            writeln!(writer, "  </div>")?;

            // Drawdown chart
            writeln!(writer, "  <div class=\"section\">")?;
            writeln!(writer, "    <h2>Drawdown</h2>")?;
            writeln!(writer, "    <div class=\"chart-container\">")?;
            write_drawdown_svg(&mut writer, &self.result.equity_curve)?;
            writeln!(writer, "    </div>")?;
            writeln!(writer, "  </div>")?;
        }

        // Trade list
        let closed_trades: Vec<_> = self
            .result
            .trades
            .iter()
            .filter(|t| t.is_closed())
            .collect();
        if !closed_trades.is_empty() {
            writeln!(writer, "  <div class=\"section\">")?;
            writeln!(writer, "    <h2>Trade List</h2>")?;
            writeln!(writer, "    <div class=\"table-container\">")?;
            writeln!(writer, "      <table>")?;
            writeln!(writer, "        <thead>")?;
            writeln!(writer, "          <tr>")?;
            writeln!(writer, "            <th>#</th>")?;
            writeln!(writer, "            <th>Symbol</th>")?;
            writeln!(writer, "            <th>Side</th>")?;
            writeln!(writer, "            <th>Qty</th>")?;
            writeln!(writer, "            <th>Entry Price</th>")?;
            writeln!(writer, "            <th>Entry Time</th>")?;
            writeln!(writer, "            <th>Exit Price</th>")?;
            writeln!(writer, "            <th>Exit Time</th>")?;
            writeln!(writer, "            <th>P&amp;L</th>")?;
            writeln!(writer, "            <th>Return %</th>")?;
            writeln!(writer, "          </tr>")?;
            writeln!(writer, "        </thead>")?;
            writeln!(writer, "        <tbody>")?;

            for (i, trade) in closed_trades.iter().enumerate() {
                let pnl = trade.net_pnl().unwrap_or(0.0);
                let pnl_pct = trade.return_pct().unwrap_or(0.0);
                let row_class = if pnl >= 0.0 { "win" } else { "loss" };

                writeln!(writer, "          <tr class=\"{}\">", row_class)?;
                writeln!(writer, "            <td>{}</td>", i + 1)?;
                writeln!(writer, "            <td>{}</td>", trade.symbol)?;
                writeln!(writer, "            <td>{:?}</td>", trade.side)?;
                writeln!(writer, "            <td>{:.4}</td>", trade.quantity)?;
                writeln!(writer, "            <td>${:.2}</td>", trade.entry_price)?;
                writeln!(
                    writer,
                    "            <td>{}</td>",
                    trade.entry_time.format("%Y-%m-%d %H:%M")
                )?;
                writeln!(
                    writer,
                    "            <td>${:.2}</td>",
                    trade.exit_price.unwrap_or(0.0)
                )?;
                writeln!(
                    writer,
                    "            <td>{}</td>",
                    trade
                        .exit_time
                        .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_default()
                )?;
                writeln!(
                    writer,
                    "            <td class=\"{}\">$<!-- -->{:.2}</td>",
                    if pnl >= 0.0 { "positive" } else { "negative" },
                    pnl
                )?;
                writeln!(
                    writer,
                    "            <td class=\"{}\">{}<!-- -->{:.2}%</td>",
                    if pnl_pct >= 0.0 {
                        "positive"
                    } else {
                        "negative"
                    },
                    if pnl_pct >= 0.0 { "+" } else { "" },
                    pnl_pct
                )?;
                writeln!(writer, "          </tr>")?;
            }

            writeln!(writer, "        </tbody>")?;
            writeln!(writer, "      </table>")?;
            writeln!(writer, "    </div>")?;
            writeln!(writer, "  </div>")?;
        }

        // Footer
        write!(writer, "{}", html_footer())?;

        Ok(())
    }
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

    /// Export a self-contained HTML report for multi-asset backtest.
    pub fn export_report_html(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write HTML header and styles
        write!(writer, "{}", html_header(&self.result.strategy_name))?;

        // Summary section
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Summary</h2>")?;
        writeln!(writer, "    <div class=\"meta\">")?;
        writeln!(
            writer,
            "      <p><strong>Period:</strong> {} to {}</p>",
            self.result.start_time.format("%Y-%m-%d"),
            self.result.end_time.format("%Y-%m-%d")
        )?;
        writeln!(
            writer,
            "      <p><strong>Assets:</strong> {}</p>",
            self.result.symbols.join(", ")
        )?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Performance metrics
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Performance Metrics</h2>")?;
        writeln!(writer, "    <div class=\"metrics-grid\">")?;
        write_metric_card(
            &mut writer,
            "Initial Capital",
            &format!("${:.2}", self.result.initial_capital),
            "neutral",
        )?;
        write_metric_card(
            &mut writer,
            "Final Equity",
            &format!("${:.2}", self.result.final_equity),
            if self.result.final_equity >= self.result.initial_capital {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Total Return",
            &format!("{:.2}%", self.result.total_return_pct),
            if self.result.total_return_pct >= 0.0 {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Annual Return",
            &format!("{:.2}%", self.result.annual_return_pct),
            if self.result.annual_return_pct >= 0.0 {
                "positive"
            } else {
                "negative"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Sharpe Ratio",
            &format!("{:.2}", self.result.sharpe_ratio),
            sharpe_color(self.result.sharpe_ratio),
        )?;
        write_metric_card(
            &mut writer,
            "Sortino Ratio",
            &format!("{:.2}", self.result.sortino_ratio),
            if self.result.sortino_ratio >= 1.0 {
                "positive"
            } else {
                "neutral"
            },
        )?;
        write_metric_card(
            &mut writer,
            "Max Drawdown",
            &format!("{:.2}%", self.result.max_drawdown_pct),
            "negative",
        )?;
        write_metric_card(
            &mut writer,
            "Total Trades",
            &format!("{}", self.result.total_trades),
            "neutral",
        )?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Trades by symbol
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Trades by Symbol</h2>")?;
        writeln!(writer, "    <div class=\"table-container\">")?;
        writeln!(writer, "      <table>")?;
        writeln!(writer, "        <thead>")?;
        writeln!(writer, "          <tr>")?;
        writeln!(writer, "            <th>Symbol</th>")?;
        writeln!(writer, "            <th>Trades</th>")?;
        writeln!(writer, "          </tr>")?;
        writeln!(writer, "        </thead>")?;
        writeln!(writer, "        <tbody>")?;
        for symbol in &self.result.symbols {
            let trades = self
                .result
                .trades_by_symbol
                .get(symbol)
                .copied()
                .unwrap_or(0);
            writeln!(writer, "          <tr>")?;
            writeln!(writer, "            <td>{}</td>", symbol)?;
            writeln!(writer, "            <td>{}</td>", trades)?;
            writeln!(writer, "          </tr>")?;
        }
        writeln!(writer, "        </tbody>")?;
        writeln!(writer, "      </table>")?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Equity curve chart
        if !self.result.equity_curve.is_empty() {
            writeln!(writer, "  <div class=\"section\">")?;
            writeln!(writer, "    <h2>Equity Curve</h2>")?;
            writeln!(writer, "    <div class=\"chart-container\">")?;
            write_equity_curve_svg(&mut writer, &self.result.equity_curve)?;
            writeln!(writer, "    </div>")?;
            writeln!(writer, "  </div>")?;

            // Drawdown chart
            writeln!(writer, "  <div class=\"section\">")?;
            writeln!(writer, "    <h2>Drawdown</h2>")?;
            writeln!(writer, "    <div class=\"chart-container\">")?;
            write_drawdown_svg(&mut writer, &self.result.equity_curve)?;
            writeln!(writer, "    </div>")?;
            writeln!(writer, "  </div>")?;
        }

        // Footer
        write!(writer, "{}", html_footer())?;

        Ok(())
    }
}

// ============================================================================
// HTML Report Helper Functions
// ============================================================================

/// Generate HTML header with embedded CSS.
fn html_header(title: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Backtest Report: {}</title>
  <style>
    :root {{
      --bg-color: #ffffff;
      --text-color: #1a1a2e;
      --card-bg: #f8f9fa;
      --border-color: #e9ecef;
      --positive: #28a745;
      --negative: #dc3545;
      --neutral: #6c757d;
      --accent: #007bff;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg-color: #1a1a2e;
        --text-color: #e9ecef;
        --card-bg: #16213e;
        --border-color: #0f3460;
        --positive: #4ade80;
        --negative: #f87171;
        --neutral: #94a3b8;
        --accent: #60a5fa;
      }}
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: var(--bg-color);
      color: var(--text-color);
      line-height: 1.6;
      padding: 2rem;
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 1.75rem;
      margin-bottom: 1.5rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--accent);
    }}
    h2 {{
      font-size: 1.25rem;
      margin-bottom: 1rem;
      color: var(--accent);
    }}
    .section {{
      margin-bottom: 2rem;
      padding: 1.5rem;
      background: var(--card-bg);
      border-radius: 8px;
      border: 1px solid var(--border-color);
    }}
    .meta p {{
      margin-bottom: 0.5rem;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 1rem;
    }}
    .metric-card {{
      background: var(--bg-color);
      padding: 1rem;
      border-radius: 6px;
      border: 1px solid var(--border-color);
    }}
    .metric-card .label {{
      font-size: 0.75rem;
      text-transform: uppercase;
      color: var(--neutral);
      margin-bottom: 0.25rem;
    }}
    .metric-card .value {{
      font-size: 1.25rem;
      font-weight: 600;
    }}
    .metric-card .value.positive {{ color: var(--positive); }}
    .metric-card .value.negative {{ color: var(--negative); }}
    .metric-card .value.neutral {{ color: var(--text-color); }}
    .chart-container {{
      background: var(--bg-color);
      border-radius: 6px;
      padding: 1rem;
      overflow-x: auto;
    }}
    .chart-container svg {{
      width: 100%;
      height: auto;
      min-height: 200px;
    }}
    .table-container {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    th, td {{
      padding: 0.75rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border-color);
    }}
    th {{
      background: var(--bg-color);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
      color: var(--neutral);
    }}
    tr:hover {{
      background: var(--bg-color);
    }}
    tr.win {{ background: rgba(40, 167, 69, 0.05); }}
    tr.loss {{ background: rgba(220, 53, 69, 0.05); }}
    td.positive {{ color: var(--positive); }}
    td.negative {{ color: var(--negative); }}
    .footer {{
      text-align: center;
      padding: 2rem;
      color: var(--neutral);
      font-size: 0.75rem;
    }}
  </style>
</head>
<body>
  <h1>Backtest Report: {}</h1>
"#,
        title, title
    )
}

/// Generate HTML footer.
fn html_footer() -> String {
    format!(
        r#"  <div class="footer">
    <p>Report generated on {} by Mantis Backtesting Engine</p>
  </div>
</body>
</html>
"#,
        Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    )
}

/// Write a metric card HTML.
fn write_metric_card<W: Write>(
    writer: &mut W,
    label: &str,
    value: &str,
    color_class: &str,
) -> Result<()> {
    writeln!(writer, "      <div class=\"metric-card\">")?;
    writeln!(writer, "        <div class=\"label\">{}</div>", label)?;
    writeln!(
        writer,
        "        <div class=\"value {}\">{}</div>",
        color_class, value
    )?;
    writeln!(writer, "      </div>")?;
    Ok(())
}

/// Determine color class for Sharpe ratio.
fn sharpe_color(sharpe: f64) -> &'static str {
    if sharpe >= 2.0 {
        "positive"
    } else if sharpe >= 1.0 {
        "neutral"
    } else {
        "negative"
    }
}

/// Write equity curve as SVG.
fn write_equity_curve_svg<W: Write>(writer: &mut W, equity_curve: &[EquityPoint]) -> Result<()> {
    if equity_curve.is_empty() {
        return Ok(());
    }

    let width = 800.0;
    let height = 300.0;
    let padding = 50.0;
    let chart_width = width - 2.0 * padding;
    let chart_height = height - 2.0 * padding;

    // Find min/max equity values
    let min_equity = equity_curve
        .iter()
        .map(|p| p.equity)
        .fold(f64::INFINITY, f64::min);
    let max_equity = equity_curve
        .iter()
        .map(|p| p.equity)
        .fold(f64::NEG_INFINITY, f64::max);

    // Add some padding to y-axis
    let equity_range = max_equity - min_equity;
    let y_min = min_equity - equity_range * 0.05;
    let y_max = max_equity + equity_range * 0.05;
    let y_range = y_max - y_min;

    // Downsample if too many points
    let points: Vec<_> = if equity_curve.len() > 500 {
        let step = equity_curve.len() / 500;
        equity_curve.iter().step_by(step).collect()
    } else {
        equity_curve.iter().collect()
    };

    // Build path
    let mut path_d = String::new();
    for (i, point) in points.iter().enumerate() {
        let x = padding + (i as f64 / (points.len() - 1).max(1) as f64) * chart_width;
        let y = padding + chart_height - ((point.equity - y_min) / y_range) * chart_height;

        if i == 0 {
            path_d.push_str(&format!("M {:.1} {:.1}", x, y));
        } else {
            path_d.push_str(&format!(" L {:.1} {:.1}", x, y));
        }
    }

    // Determine if the overall return is positive or negative
    let start_equity = equity_curve.first().map(|p| p.equity).unwrap_or(0.0);
    let end_equity = equity_curve.last().map(|p| p.equity).unwrap_or(0.0);
    let line_color = if end_equity >= start_equity {
        "#28a745"
    } else {
        "#dc3545"
    };

    // Color constants for SVG
    let grid_color = "#e0e0e0";
    let text_color = "#666";

    // Write SVG
    writeln!(
        writer,
        r##"      <svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"##,
        width, height
    )?;

    // Background
    writeln!(
        writer,
        r##"        <rect width="{}" height="{}" fill="transparent"/>"##,
        width, height
    )?;

    // Y-axis grid lines and labels
    let num_y_ticks = 5;
    for i in 0..=num_y_ticks {
        let y_val = y_min + (i as f64 / num_y_ticks as f64) * y_range;
        let y = padding + chart_height - (i as f64 / num_y_ticks as f64) * chart_height;

        // Grid line
        writeln!(
            writer,
            r##"        <line x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="{}" stroke-dasharray="4,4"/>"##,
            padding,
            y,
            width - padding,
            y,
            grid_color
        )?;

        // Y-axis label
        writeln!(
            writer,
            r##"        <text x="{}" y="{:.1}" font-size="10" fill="{}" text-anchor="end">${:.0}k</text>"##,
            padding - 5.0,
            y + 3.0,
            text_color,
            y_val / 1000.0
        )?;
    }

    // X-axis labels (start and end dates)
    if let (Some(first), Some(last)) = (equity_curve.first(), equity_curve.last()) {
        writeln!(
            writer,
            r##"        <text x="{}" y="{}" font-size="10" fill="{}" text-anchor="start">{}</text>"##,
            padding,
            height - 10.0,
            text_color,
            first.timestamp.format("%Y-%m-%d")
        )?;
        writeln!(
            writer,
            r##"        <text x="{}" y="{}" font-size="10" fill="{}" text-anchor="end">{}</text>"##,
            width - padding,
            height - 10.0,
            text_color,
            last.timestamp.format("%Y-%m-%d")
        )?;
    }

    // Equity line
    writeln!(
        writer,
        r##"        <path d="{}" fill="none" stroke="{}" stroke-width="2"/>"##,
        path_d, line_color
    )?;

    writeln!(writer, "      </svg>")?;

    Ok(())
}

/// Write drawdown chart as SVG.
fn write_drawdown_svg<W: Write>(writer: &mut W, equity_curve: &[EquityPoint]) -> Result<()> {
    if equity_curve.is_empty() {
        return Ok(());
    }

    let width = 800.0;
    let height = 200.0;
    let padding = 50.0;
    let chart_width = width - 2.0 * padding;
    let chart_height = height - 2.0 * padding;

    // Find max drawdown
    let max_dd = equity_curve
        .iter()
        .map(|p| p.drawdown_pct)
        .fold(0.0_f64, f64::max);

    // Y-axis range: 0 to max_dd (inverted, since drawdown is shown below zero line)
    let y_max = (max_dd * 1.1).max(1.0); // At least 1% for scale

    // Downsample if too many points
    let points: Vec<_> = if equity_curve.len() > 500 {
        let step = equity_curve.len() / 500;
        equity_curve.iter().step_by(step).collect()
    } else {
        equity_curve.iter().collect()
    };

    // Build area path (fill under the line)
    let mut area_d = format!("M {:.1} {:.1}", padding, padding);
    let mut line_d = String::new();

    for (i, point) in points.iter().enumerate() {
        let x = padding + (i as f64 / (points.len() - 1).max(1) as f64) * chart_width;
        let y = padding + (point.drawdown_pct / y_max) * chart_height;

        if i == 0 {
            line_d.push_str(&format!("M {:.1} {:.1}", x, y));
        } else {
            line_d.push_str(&format!(" L {:.1} {:.1}", x, y));
        }
        area_d.push_str(&format!(" L {:.1} {:.1}", x, y));
    }

    // Close the area path
    area_d.push_str(&format!(" L {:.1} {:.1} Z", padding + chart_width, padding));

    // Color constants for SVG
    let grid_color = "#e0e0e0";
    let text_color = "#666";
    let drawdown_color = "#dc3545";

    // Write SVG
    writeln!(
        writer,
        r##"      <svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"##,
        width, height
    )?;

    // Background
    writeln!(
        writer,
        r##"        <rect width="{}" height="{}" fill="transparent"/>"##,
        width, height
    )?;

    // Zero line
    writeln!(
        writer,
        r##"        <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"##,
        padding,
        padding,
        width - padding,
        padding,
        text_color
    )?;

    // Y-axis labels
    let num_y_ticks = 4;
    for i in 0..=num_y_ticks {
        let y_val = (i as f64 / num_y_ticks as f64) * y_max;
        let y = padding + (i as f64 / num_y_ticks as f64) * chart_height;

        // Grid line
        if i > 0 {
            writeln!(
                writer,
                r##"        <line x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="{}" stroke-dasharray="4,4"/>"##,
                padding,
                y,
                width - padding,
                y,
                grid_color
            )?;
        }

        // Y-axis label
        writeln!(
            writer,
            r##"        <text x="{}" y="{:.1}" font-size="10" fill="{}" text-anchor="end">-{:.1}%</text>"##,
            padding - 5.0,
            y + 3.0,
            text_color,
            y_val
        )?;
    }

    // X-axis labels
    if let (Some(first), Some(last)) = (equity_curve.first(), equity_curve.last()) {
        writeln!(
            writer,
            r##"        <text x="{}" y="{}" font-size="10" fill="{}" text-anchor="start">{}</text>"##,
            padding,
            height - 5.0,
            text_color,
            first.timestamp.format("%Y-%m-%d")
        )?;
        writeln!(
            writer,
            r##"        <text x="{}" y="{}" font-size="10" fill="{}" text-anchor="end">{}</text>"##,
            width - padding,
            height - 5.0,
            text_color,
            last.timestamp.format("%Y-%m-%d")
        )?;
    }

    // Drawdown area (filled)
    writeln!(
        writer,
        r##"        <path d="{}" fill="rgba(220, 53, 69, 0.2)"/>"##,
        area_d
    )?;

    // Drawdown line
    writeln!(
        writer,
        r##"        <path d="{}" fill="none" stroke="{}" stroke-width="1.5"/>"##,
        line_d, drawdown_color
    )?;

    writeln!(writer, "      </svg>")?;

    Ok(())
}

// ============================================================================
// Walk-Forward Report Export
// ============================================================================

use crate::walkforward::WalkForwardResult;

/// Export walk-forward validation results to a self-contained HTML report.
///
/// Creates an HTML file with embedded CSS and SVG charts showing:
/// - Summary metrics (IS/OOS Sharpe, degradation, verdict)
/// - Fold-by-fold performance comparison
/// - Efficiency metrics
pub fn export_walkforward_html(result: &WalkForwardResult, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write HTML header
    write!(writer, "{}", html_header("Walk-Forward Validation Report"))?;

    // Summary section
    writeln!(writer, "  <div class=\"section\">")?;
    writeln!(writer, "    <h2>Walk-Forward Summary</h2>")?;
    writeln!(writer, "    <div class=\"meta\">")?;
    writeln!(
        writer,
        "      <p><strong>Folds:</strong> {}</p>",
        result.windows.len()
    )?;
    writeln!(
        writer,
        "      <p><strong>Window Type:</strong> {}</p>",
        if result.config.anchored {
            "Anchored"
        } else {
            "Rolling"
        }
    )?;
    writeln!(
        writer,
        "      <p><strong>In-Sample Ratio:</strong> {:.0}%</p>",
        result.config.in_sample_ratio * 100.0
    )?;
    writeln!(writer, "    </div>")?;
    writeln!(writer, "  </div>")?;

    // Verdict section
    let verdict = result.verdict();
    let verdict_color = match verdict.label() {
        "robust" => "positive",
        "borderline" => "neutral",
        _ => "negative",
    };
    writeln!(writer, "  <div class=\"section\">")?;
    writeln!(writer, "    <h2>Verdict</h2>")?;
    writeln!(writer, "    <div class=\"metrics-grid\">")?;
    write_metric_card(
        &mut writer,
        "Classification",
        verdict.label(),
        verdict_color,
    )?;
    write_metric_card(
        &mut writer,
        "OOS/IS Degradation",
        &format!("{:.0}%", result.oos_sharpe_ratio() * 100.0),
        if result.oos_sharpe_ratio() >= 0.8 {
            "positive"
        } else if result.oos_sharpe_ratio() >= 0.6 {
            "neutral"
        } else {
            "negative"
        },
    )?;
    write_metric_card(
        &mut writer,
        "Walk-Forward Efficiency",
        &format!("{:.1}%", result.walk_forward_efficiency * 100.0),
        if result.walk_forward_efficiency >= 0.5 {
            "positive"
        } else {
            "negative"
        },
    )?;
    write_metric_card(
        &mut writer,
        "Parameter Stability",
        &format!("{:.1}%", result.parameter_stability * 100.0),
        if result.parameter_stability >= 0.7 {
            "positive"
        } else if result.parameter_stability >= 0.5 {
            "neutral"
        } else {
            "negative"
        },
    )?;
    writeln!(writer, "    </div>")?;
    writeln!(writer, "  </div>")?;

    // Performance metrics
    writeln!(writer, "  <div class=\"section\">")?;
    writeln!(writer, "    <h2>Performance Metrics</h2>")?;
    writeln!(writer, "    <div class=\"metrics-grid\">")?;
    write_metric_card(
        &mut writer,
        "Avg IS Sharpe",
        &format!("{:.2}", result.avg_is_sharpe),
        sharpe_color(result.avg_is_sharpe),
    )?;
    write_metric_card(
        &mut writer,
        "Avg OOS Sharpe",
        &format!("{:.2}", result.avg_oos_sharpe),
        sharpe_color(result.avg_oos_sharpe),
    )?;
    write_metric_card(
        &mut writer,
        "Avg IS Return",
        &format!("{:.2}%", result.avg_is_return),
        if result.avg_is_return >= 0.0 {
            "positive"
        } else {
            "negative"
        },
    )?;
    write_metric_card(
        &mut writer,
        "Avg OOS Return",
        &format!("{:.2}%", result.avg_oos_return),
        if result.avg_oos_return >= 0.0 {
            "positive"
        } else {
            "negative"
        },
    )?;
    write_metric_card(
        &mut writer,
        "Combined OOS Return",
        &format!("{:.2}%", result.combined_oos_return),
        if result.combined_oos_return >= 0.0 {
            "positive"
        } else {
            "negative"
        },
    )?;
    write_metric_card(
        &mut writer,
        "Avg Efficiency",
        &format!("{:.1}%", result.avg_efficiency_ratio * 100.0),
        if result.avg_efficiency_ratio >= 0.5 {
            "positive"
        } else {
            "neutral"
        },
    )?;
    writeln!(writer, "    </div>")?;
    writeln!(writer, "  </div>")?;

    // Fold-by-fold table
    if !result.windows.is_empty() {
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Fold-by-Fold Results</h2>")?;
        writeln!(writer, "    <div class=\"table-container\">")?;
        writeln!(writer, "      <table>")?;
        writeln!(writer, "        <thead>")?;
        writeln!(writer, "          <tr>")?;
        writeln!(writer, "            <th>Fold</th>")?;
        writeln!(writer, "            <th>IS Bars</th>")?;
        writeln!(writer, "            <th>OOS Bars</th>")?;
        writeln!(writer, "            <th>IS Sharpe</th>")?;
        writeln!(writer, "            <th>OOS Sharpe</th>")?;
        writeln!(writer, "            <th>IS Return</th>")?;
        writeln!(writer, "            <th>OOS Return</th>")?;
        writeln!(writer, "            <th>Efficiency</th>")?;
        writeln!(writer, "          </tr>")?;
        writeln!(writer, "        </thead>")?;
        writeln!(writer, "        <tbody>")?;

        for (i, window_result) in result.windows.iter().enumerate() {
            let is_sharpe = window_result.in_sample_result.sharpe_ratio;
            let oos_sharpe = window_result.out_of_sample_result.sharpe_ratio;
            let is_return = window_result.in_sample_result.total_return_pct;
            let oos_return = window_result.out_of_sample_result.total_return_pct;
            let efficiency = window_result.efficiency_ratio;

            let row_class = if oos_return >= 0.0 { "win" } else { "loss" };

            writeln!(writer, "          <tr class=\"{}\">", row_class)?;
            writeln!(writer, "            <td>{}</td>", i + 1)?;
            writeln!(
                writer,
                "            <td>{}</td>",
                window_result.window.is_bars
            )?;
            writeln!(
                writer,
                "            <td>{}</td>",
                window_result.window.oos_bars
            )?;
            writeln!(writer, "            <td>{:.2}</td>", is_sharpe)?;
            writeln!(
                writer,
                "            <td class=\"{}\">{:.2}</td>",
                if oos_sharpe >= 0.0 {
                    "positive"
                } else {
                    "negative"
                },
                oos_sharpe
            )?;
            writeln!(
                writer,
                "            <td class=\"{}\">{:.2}%</td>",
                if is_return >= 0.0 {
                    "positive"
                } else {
                    "negative"
                },
                is_return
            )?;
            writeln!(
                writer,
                "            <td class=\"{}\">{:.2}%</td>",
                if oos_return >= 0.0 {
                    "positive"
                } else {
                    "negative"
                },
                oos_return
            )?;
            writeln!(
                writer,
                "            <td class=\"{}\">{:.0}%</td>",
                if efficiency >= 0.5 {
                    "positive"
                } else {
                    "negative"
                },
                efficiency * 100.0
            )?;
            writeln!(writer, "          </tr>")?;
        }

        writeln!(writer, "        </tbody>")?;
        writeln!(writer, "      </table>")?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;

        // Bar chart visualization of IS vs OOS performance
        writeln!(writer, "  <div class=\"section\">")?;
        writeln!(writer, "    <h2>Performance Comparison</h2>")?;
        writeln!(writer, "    <div class=\"chart-container\">")?;
        write_walkforward_bar_chart(&mut writer, result)?;
        writeln!(writer, "    </div>")?;
        writeln!(writer, "  </div>")?;
    }

    // Footer
    write!(writer, "{}", html_footer())?;

    Ok(())
}

/// Write a bar chart comparing IS vs OOS performance across folds.
fn write_walkforward_bar_chart<W: Write>(writer: &mut W, result: &WalkForwardResult) -> Result<()> {
    if result.windows.is_empty() {
        return Ok(());
    }

    let width = 800.0;
    let height = 300.0;
    let padding = 60.0;
    let chart_width = width - 2.0 * padding;
    let chart_height = height - 2.0 * padding;

    let num_folds = result.windows.len();
    let bar_group_width = chart_width / num_folds as f64;
    let bar_width = bar_group_width * 0.35;
    let gap = bar_group_width * 0.1;

    // Find max/min returns for scaling
    let mut max_return = 0.0_f64;
    let mut min_return = 0.0_f64;
    for w in &result.windows {
        max_return = max_return.max(w.in_sample_result.total_return_pct);
        max_return = max_return.max(w.out_of_sample_result.total_return_pct);
        min_return = min_return.min(w.in_sample_result.total_return_pct);
        min_return = min_return.min(w.out_of_sample_result.total_return_pct);
    }

    // Add padding to y range
    let y_range = (max_return - min_return).max(1.0);
    max_return += y_range * 0.1;
    min_return -= y_range * 0.1;
    let y_range = max_return - min_return;

    // Zero line position
    let zero_y = padding + ((max_return - 0.0) / y_range) * chart_height;

    // Colors
    let is_color = "#3b82f6"; // Blue for in-sample
    let oos_color = "#10b981"; // Green for out-of-sample

    writeln!(
        writer,
        r##"      <svg viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">"##,
        width, height
    )?;

    // Background
    writeln!(
        writer,
        r##"        <rect width="{}" height="{}" fill="transparent"/>"##,
        width, height
    )?;

    // Zero line
    writeln!(
        writer,
        r##"        <line x1="{}" y1="{:.1}" x2="{}" y2="{:.1}" stroke="#666" stroke-width="1"/>"##,
        padding,
        zero_y,
        width - padding,
        zero_y
    )?;

    // Draw bars for each fold
    for (i, window_result) in result.windows.iter().enumerate() {
        let is_return = window_result.in_sample_result.total_return_pct;
        let oos_return = window_result.out_of_sample_result.total_return_pct;

        let x_center = padding + (i as f64 + 0.5) * bar_group_width;

        // IS bar
        let is_x = x_center - bar_width - gap / 2.0;
        let is_height = ((is_return - 0.0) / y_range).abs() * chart_height;
        let is_y = if is_return >= 0.0 {
            zero_y - is_height
        } else {
            zero_y
        };
        writeln!(
            writer,
            r##"        <rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" opacity="0.8"/>"##,
            is_x,
            is_y,
            bar_width,
            is_height.max(1.0),
            is_color
        )?;

        // OOS bar
        let oos_x = x_center + gap / 2.0;
        let oos_height = ((oos_return - 0.0) / y_range).abs() * chart_height;
        let oos_y = if oos_return >= 0.0 {
            zero_y - oos_height
        } else {
            zero_y
        };
        writeln!(
            writer,
            r##"        <rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{}" opacity="0.8"/>"##,
            oos_x,
            oos_y,
            bar_width,
            oos_height.max(1.0),
            oos_color
        )?;

        // X-axis label (fold number)
        writeln!(
            writer,
            r##"        <text x="{:.1}" y="{}" font-size="10" fill="#666" text-anchor="middle">{}</text>"##,
            x_center,
            height - 10.0,
            i + 1
        )?;
    }

    // Y-axis labels
    let num_y_ticks = 5;
    for i in 0..=num_y_ticks {
        let y_val = min_return + (i as f64 / num_y_ticks as f64) * y_range;
        let y = padding + chart_height - (i as f64 / num_y_ticks as f64) * chart_height;

        writeln!(
            writer,
            r##"        <text x="{}" y="{:.1}" font-size="10" fill="#666" text-anchor="end">{:.1}%</text>"##,
            padding - 5.0,
            y + 3.0,
            y_val
        )?;
    }

    // Legend
    writeln!(
        writer,
        r##"        <rect x="{}" y="10" width="12" height="12" fill="{}"/>"##,
        width - 150.0,
        is_color
    )?;
    writeln!(
        writer,
        r##"        <text x="{}" y="20" font-size="11" fill="#666">In-Sample</text>"##,
        width - 135.0
    )?;
    writeln!(
        writer,
        r##"        <rect x="{}" y="30" width="12" height="12" fill="{}"/>"##,
        width - 150.0,
        oos_color
    )?;
    writeln!(
        writer,
        r##"        <text x="{}" y="40" font-size="11" fill="#666">Out-of-Sample</text>"##,
        width - 135.0
    )?;

    writeln!(writer, "      </svg>")?;

    Ok(())
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
            seed: None,
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

    #[test]
    fn test_export_report_html() {
        let mut result = create_test_result();

        // Add equity curve data for chart generation
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        result.equity_curve = vec![
            EquityPoint {
                timestamp: start,
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: start + Duration::days(30),
                equity: 105000.0,
                cash: 5000.0,
                positions_value: 100000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: start + Duration::days(60),
                equity: 102000.0,
                cash: 2000.0,
                positions_value: 100000.0,
                drawdown: 3000.0,
                drawdown_pct: 2.86,
            },
            EquityPoint {
                timestamp: start + Duration::days(90),
                equity: 110000.0,
                cash: 10000.0,
                positions_value: 100000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_html(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();

        // Verify HTML structure
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("<html lang=\"en\">"));
        assert!(content.contains("Backtest Report: Test Strategy"));

        // Verify performance metrics
        assert!(content.contains("Performance Metrics"));
        assert!(content.contains("$100000.00")); // Initial capital
        assert!(content.contains("$110000.00")); // Final equity
        assert!(content.contains("10.00%")); // Total return

        // Verify trade statistics
        assert!(content.contains("Trade Statistics"));
        assert!(content.contains("Win Rate"));
        assert!(content.contains("100.0%")); // Win rate

        // Verify equity curve SVG
        assert!(content.contains("<svg"));
        assert!(content.contains("viewBox"));

        // Verify trade list
        assert!(content.contains("Trade List"));
        assert!(content.contains("TEST"));
        assert!(content.contains("Buy"));

        // Verify footer
        assert!(content.contains("Mantis Backtesting Engine"));
    }

    #[test]
    fn test_export_report_html_multi_asset() {
        let result = create_test_multi_asset_result();
        let exporter = MultiAssetExporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_html(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();

        // Verify HTML structure
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("Backtest Report: MultiAsset Strategy"));

        // Verify assets are listed
        assert!(content.contains("AAPL"));
        assert!(content.contains("MSFT"));

        // Verify trades by symbol section
        assert!(content.contains("Trades by Symbol"));

        // Verify equity curve SVG
        assert!(content.contains("<svg"));
    }

    #[test]
    fn test_html_header_contains_title() {
        let header = html_header("Test Strategy");
        assert!(header.contains("Test Strategy"));
        assert!(header.contains("<title>"));
        assert!(header.contains("<style>"));
    }

    #[test]
    fn test_sharpe_color() {
        assert_eq!(sharpe_color(2.5), "positive");
        assert_eq!(sharpe_color(1.5), "neutral");
        assert_eq!(sharpe_color(0.5), "negative");
    }

    #[test]
    fn test_export_report_html_empty_equity_curve() {
        let result = create_test_result();
        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_html(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();

        // Should still produce valid HTML without equity curve section
        assert!(content.contains("<!DOCTYPE html>"));
        assert!(content.contains("</html>"));
        // No SVG since equity curve is empty
        // (the test result has empty equity_curve by default)
    }

    #[test]
    fn test_export_report_html_negative_returns() {
        let mut result = create_test_result();
        result.total_return_pct = -15.0;
        result.final_equity = 85000.0;
        result.sharpe_ratio = -0.5;
        result.winning_trades = 0;
        result.losing_trades = 1;
        result.win_rate = 0.0;

        let exporter = Exporter::new(result);

        let file = NamedTempFile::new().unwrap();
        exporter.export_report_html(file.path()).unwrap();

        let content = std::fs::read_to_string(file.path()).unwrap();

        // Verify negative values are displayed
        assert!(content.contains("-15.00%"));
        assert!(content.contains("$85000.00"));
        // Verify negative class is applied
        assert!(content.contains("class=\"value negative\""));
    }
}
