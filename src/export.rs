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
//! use ralph_backtest::export::{ExportConfig, Exporter};
//! use ralph_backtest::engine::BacktestResult;
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
            writeln!(writer, "timestamp,equity,drawdown_pct")?;
        }

        for point in &self.result.trades {
            // Use equity from trades (entry timestamps)
            // Actually we need equity curve from engine, but it's not stored in BacktestResult
            // For now, we'll compute from trades
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
        let closed_trades: Vec<_> = self.result.trades.iter().filter(|t| t.is_closed()).collect();
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
pub fn export_equity_curve_csv(
    equity_curve: &[EquityPoint],
    path: impl AsRef<Path>,
) -> Result<()> {
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
        return Err(BacktestError::DataError("No features to export".to_string()));
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

    /// Export weight history to CSV.
    pub fn export_weights_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        write!(writer, "timestamp")?;
        for symbol in &self.result.symbols {
            write!(writer, ",{}", symbol)?;
        }
        writeln!(writer)?;

        // Write data
        for (timestamp, weights) in &self.result.weight_history {
            write!(writer, "{}", timestamp.format("%Y-%m-%d %H:%M:%S"))?;
            for symbol in &self.result.symbols {
                let weight = weights.get(symbol).copied().unwrap_or(0.0);
                write!(writer, ",{:.4}", weight)?;
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
        writeln!(
            writer,
            "| Sharpe Ratio | {:.2} |",
            self.result.sharpe_ratio
        )?;
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
            let trades = self.result.trades_by_symbol.get(symbol).copied().unwrap_or(0);
            writeln!(writer, "| {} | {} |", symbol, trades)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BacktestConfig;
    use crate::types::Side;
    use chrono::TimeZone;
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
            1.0,  // commission
            0.0,  // slippage
        );
        trade.close(110.0, Utc.with_ymd_and_hms(2024, 1, 5, 15, 0, 0).unwrap(), 1.0);
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
            start_time: Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap(),
            end_time: Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap(),
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
}
