//! Visualization utilities for backtest results.
//!
//! This module provides terminal-friendly visualization including:
//! - ASCII sparklines for quick equity curve visualization
//! - Strategy comparison tables
//! - SVG heatmaps for parameter sensitivity analysis
//!
//! # Example
//!
//! ```ignore
//! use mantis::viz::{sparkline, compare_strategies, SparklineConfig};
//! use mantis::engine::BacktestResult;
//!
//! // Generate ASCII sparkline for equity curve
//! let spark = sparkline(&equity_values, 40);
//! println!("Equity: {}", spark);
//!
//! // Compare multiple strategies
//! let comparison = compare_strategies(&[result1, result2, result3], &["LSTM", "Baseline", "Momentum"]);
//! println!("{}", comparison);
//! ```

use crate::engine::BacktestResult;
use crate::sensitivity::HeatmapData;
#[cfg(test)]
use crate::sensitivity::SensitivityMetric;
use crate::walkforward::WalkForwardResult;
use std::fmt::Write;

/// Characters used for sparkline rendering, ordered from low to high.
const SPARKLINE_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Configuration for sparkline generation.
#[derive(Debug, Clone)]
pub struct SparklineConfig {
    /// Maximum width in characters.
    pub width: usize,
    /// Whether to normalize values (min=0, max=1).
    pub normalize: bool,
    /// Custom minimum value (if not normalizing from data).
    pub min_value: Option<f64>,
    /// Custom maximum value (if not normalizing from data).
    pub max_value: Option<f64>,
}

impl Default for SparklineConfig {
    fn default() -> Self {
        Self {
            width: 40,
            normalize: true,
            min_value: None,
            max_value: None,
        }
    }
}

/// Generate an ASCII sparkline from a slice of values.
///
/// # Arguments
/// * `values` - The data values to visualize.
/// * `width` - Maximum character width of the sparkline.
///
/// # Returns
/// A string containing the sparkline characters.
///
/// # Example
/// ```ignore
/// let equity = vec![100.0, 102.0, 101.0, 105.0, 108.0, 107.0, 112.0];
/// let spark = sparkline(&equity, 7);
/// assert!(!spark.is_empty());
/// ```
pub fn sparkline(values: &[f64], width: usize) -> String {
    if values.is_empty() {
        return String::new();
    }

    sparkline_with_config(
        values,
        &SparklineConfig {
            width,
            ..Default::default()
        },
    )
}

/// Generate an ASCII sparkline with custom configuration.
pub fn sparkline_with_config(values: &[f64], config: &SparklineConfig) -> String {
    if values.is_empty() {
        return String::new();
    }

    // Downsample if needed
    let sampled = if values.len() > config.width {
        downsample(values, config.width)
    } else {
        values.to_vec()
    };

    // Find min/max for normalization
    let min_val = config
        .min_value
        .unwrap_or_else(|| sampled.iter().cloned().fold(f64::INFINITY, f64::min));
    let max_val = config
        .max_value
        .unwrap_or_else(|| sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    let range = max_val - min_val;

    // Build sparkline string
    let mut result = String::with_capacity(sampled.len() * 4); // UTF-8 chars can be up to 4 bytes

    for &val in &sampled {
        let normalized = if range > 0.0 {
            ((val - min_val) / range).clamp(0.0, 1.0)
        } else {
            0.5 // All values are the same
        };

        // Map to character index (0-7)
        let idx = ((normalized * 7.0).round() as usize).min(7);
        result.push(SPARKLINE_CHARS[idx]);
    }

    result
}

/// Downsample a slice of values to a target length using averaging.
fn downsample(values: &[f64], target_len: usize) -> Vec<f64> {
    if values.len() <= target_len {
        return values.to_vec();
    }

    let chunk_size = values.len() as f64 / target_len as f64;
    let mut result = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let start = (i as f64 * chunk_size).floor() as usize;
        let end = ((i + 1) as f64 * chunk_size).ceil() as usize;
        let end = end.min(values.len());

        if start < end {
            let sum: f64 = values[start..end].iter().sum();
            result.push(sum / (end - start) as f64);
        }
    }

    result
}

/// Generate a sparkline from a BacktestResult's equity curve.
pub fn equity_sparkline(result: &BacktestResult, width: usize) -> String {
    let equity_values: Vec<f64> = result.equity_curve.iter().map(|p| p.equity).collect();
    sparkline(&equity_values, width)
}

/// Format a number as a percentage with sign.
fn format_pct(value: f64) -> String {
    if value >= 0.0 {
        format!("+{:.1}%", value)
    } else {
        format!("{:.1}%", value)
    }
}

/// Format a number with specified precision.
fn format_num(value: f64, precision: usize) -> String {
    format!("{:.prec$}", value, prec = precision)
}

/// A row in the strategy comparison table.
#[derive(Debug, Clone)]
pub struct ComparisonRow {
    /// Strategy name.
    pub name: String,
    /// Total return percentage.
    pub total_return: f64,
    /// Sharpe ratio.
    pub sharpe: f64,
    /// Maximum drawdown percentage.
    pub max_drawdown: f64,
    /// Win rate percentage.
    pub win_rate: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Number of trades.
    pub total_trades: usize,
    /// Sortino ratio.
    pub sortino: f64,
    /// Calmar ratio.
    pub calmar: f64,
}

impl ComparisonRow {
    /// Create a comparison row from a BacktestResult.
    pub fn from_result(result: &BacktestResult, name: &str) -> Self {
        Self {
            name: name.to_string(),
            total_return: result.total_return_pct,
            sharpe: result.sharpe_ratio,
            max_drawdown: result.max_drawdown_pct,
            win_rate: result.win_rate,
            profit_factor: result.profit_factor,
            total_trades: result.total_trades,
            sortino: result.sortino_ratio,
            calmar: result.calmar_ratio,
        }
    }
}

/// Strategy comparison result with formatted output.
#[derive(Debug, Clone)]
pub struct StrategyComparison {
    /// Rows of strategy data.
    pub rows: Vec<ComparisonRow>,
    /// Whether to use extended format with more metrics.
    pub extended: bool,
}

impl StrategyComparison {
    /// Create a new comparison from backtest results.
    pub fn new(results: &[&BacktestResult], names: &[&str]) -> Self {
        let rows = results
            .iter()
            .zip(names.iter())
            .map(|(result, name)| ComparisonRow::from_result(result, name))
            .collect();

        Self {
            rows,
            extended: false,
        }
    }

    /// Enable extended format with additional metrics.
    pub fn with_extended(mut self, extended: bool) -> Self {
        self.extended = extended;
        self
    }

    /// Get the best strategy by a given metric.
    pub fn best_by_sharpe(&self) -> Option<&ComparisonRow> {
        self.rows.iter().max_by(|a, b| {
            a.sharpe
                .partial_cmp(&b.sharpe)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get the best strategy by total return.
    pub fn best_by_return(&self) -> Option<&ComparisonRow> {
        self.rows.iter().max_by(|a, b| {
            a.total_return
                .partial_cmp(&b.total_return)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Format as a table string.
    pub fn format_table(&self) -> String {
        if self.rows.is_empty() {
            return "No strategies to compare.".to_string();
        }

        let mut output = String::new();

        // Header
        writeln!(output, "Strategy Comparison").unwrap();
        writeln!(output, "{}", "─".repeat(70)).unwrap();

        if self.extended {
            writeln!(
                output,
                "{:15} {:>12} {:>8} {:>10} {:>10} {:>8} {:>8}",
                "Strategy", "Total Return", "Sharpe", "Max DD", "Win Rate", "PF", "Trades"
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "{:15} {:>12} {:>8} {:>10} {:>10}",
                "Strategy", "Total Return", "Sharpe", "Max DD", "Win Rate"
            )
            .unwrap();
        }

        writeln!(output, "{}", "─".repeat(70)).unwrap();

        for row in &self.rows {
            if self.extended {
                writeln!(
                    output,
                    "{:15} {:>12} {:>8} {:>10} {:>10} {:>8} {:>8}",
                    truncate_str(&row.name, 15),
                    format_pct(row.total_return),
                    format_num(row.sharpe, 2),
                    format_pct(-row.max_drawdown.abs()),
                    format_pct(row.win_rate),
                    format_num(row.profit_factor, 2),
                    row.total_trades
                )
                .unwrap();
            } else {
                writeln!(
                    output,
                    "{:15} {:>12} {:>8} {:>10} {:>10}",
                    truncate_str(&row.name, 15),
                    format_pct(row.total_return),
                    format_num(row.sharpe, 2),
                    format_pct(-row.max_drawdown.abs()),
                    format_pct(row.win_rate)
                )
                .unwrap();
            }
        }

        output
    }
}

impl std::fmt::Display for StrategyComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_table())
    }
}

/// Truncate a string to a maximum length, adding ellipsis if needed.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Compare multiple backtest results and return a formatted comparison.
///
/// # Arguments
/// * `results` - Slice of backtest results to compare.
/// * `names` - Names to label each strategy.
///
/// # Example
/// ```ignore
/// let comparison = compare_strategies(&[&result1, &result2], &["LSTM", "Baseline"]);
/// println!("{}", comparison);
/// ```
pub fn compare_strategies(results: &[&BacktestResult], names: &[&str]) -> StrategyComparison {
    StrategyComparison::new(results, names)
}

/// Generate a quick summary string for a backtest result.
///
/// Includes sparkline and key metrics in one line.
pub fn result_summary(result: &BacktestResult, width: usize) -> String {
    let spark = equity_sparkline(result, width);
    format!(
        "[{}] {} | Return: {} | Sharpe: {:.2} | Max DD: {}",
        spark,
        result.strategy_name,
        format_pct(result.total_return_pct),
        result.sharpe_ratio,
        format_pct(-result.max_drawdown_pct.abs())
    )
}

/// Generate a quick summary with verdict for terminal display.
pub fn result_with_verdict(result: &BacktestResult, width: usize) -> String {
    let spark = equity_sparkline(result, width);
    format!(
        "Backtest Results: {}\n[{}] equity curve\nTotal Return: {}  |  Sharpe: {:.2}  |  Max DD: {}",
        result.symbols.join(", "),
        spark,
        format_pct(result.total_return_pct),
        result.sharpe_ratio,
        format_pct(-result.max_drawdown_pct.abs())
    )
}

// ============================================================================
// Walk-Forward Visualization
// ============================================================================

/// Generate a bar chart representation of walk-forward fold returns.
pub fn walkforward_fold_chart(result: &WalkForwardResult, width: usize) -> String {
    let mut output = String::new();

    writeln!(output, "Walk-Forward Fold Performance").unwrap();
    writeln!(output, "{}", "─".repeat(60)).unwrap();
    writeln!(
        output,
        "{:>6} {:>10} {:>10} {:>12}",
        "Fold", "IS Return", "OOS Return", "Efficiency"
    )
    .unwrap();
    writeln!(output, "{}", "─".repeat(60)).unwrap();

    // Collect returns for bar chart
    let is_returns: Vec<f64> = result
        .windows
        .iter()
        .map(|w| w.in_sample_result.total_return_pct)
        .collect();
    let oos_returns: Vec<f64> = result
        .windows
        .iter()
        .map(|w| w.out_of_sample_result.total_return_pct)
        .collect();

    // Find max for scaling
    let max_return = is_returns
        .iter()
        .chain(oos_returns.iter())
        .cloned()
        .fold(1.0_f64, f64::max);

    for (i, window) in result.windows.iter().enumerate() {
        let is_ret = window.in_sample_result.total_return_pct;
        let oos_ret = window.out_of_sample_result.total_return_pct;
        let efficiency = window.efficiency_ratio * 100.0;

        // Simple bar representation
        let is_bar_len = if max_return > 0.0 {
            ((is_ret / max_return) * width as f64).max(0.0) as usize
        } else {
            0
        };
        let oos_bar_len = if max_return > 0.0 {
            ((oos_ret / max_return) * width as f64).max(0.0) as usize
        } else {
            0
        };

        let is_bar: String = "█".repeat(is_bar_len);
        let oos_bar: String = "▒".repeat(oos_bar_len);

        writeln!(
            output,
            "{:>6} {:>10} {:>10} {:>11.1}%",
            i + 1,
            format_pct(is_ret),
            format_pct(oos_ret),
            efficiency
        )
        .unwrap();

        writeln!(output, "       IS:  {}", is_bar).unwrap();
        writeln!(output, "       OOS: {}", oos_bar).unwrap();
    }

    writeln!(output, "{}", "─".repeat(60)).unwrap();
    writeln!(output, "Verdict: {:?}", result.verdict()).unwrap();

    output
}

/// Generate a compact walk-forward summary.
pub fn walkforward_summary(result: &WalkForwardResult) -> String {
    format!(
        "Walk-Forward: {} folds | Avg OOS: {} | Efficiency: {:.1}% | Verdict: {:?}",
        result.windows.len(),
        format_pct(result.avg_oos_return),
        result.walk_forward_efficiency * 100.0,
        result.verdict()
    )
}

// ============================================================================
// Heatmap SVG Generation
// ============================================================================

/// Configuration for SVG heatmap generation.
#[derive(Debug, Clone)]
pub struct HeatmapSvgConfig {
    /// Cell width in pixels.
    pub cell_width: usize,
    /// Cell height in pixels.
    pub cell_height: usize,
    /// Margin around the heatmap in pixels.
    pub margin: usize,
    /// Color for low values (RGB).
    pub color_low: (u8, u8, u8),
    /// Color for mid values (RGB).
    pub color_mid: (u8, u8, u8),
    /// Color for high values (RGB).
    pub color_high: (u8, u8, u8),
    /// Whether to show value labels in cells.
    pub show_labels: bool,
    /// Title for the heatmap.
    pub title: Option<String>,
}

impl Default for HeatmapSvgConfig {
    fn default() -> Self {
        Self {
            cell_width: 50,
            cell_height: 40,
            margin: 60,
            color_low: (220, 53, 69),   // Red for low/negative
            color_mid: (255, 255, 255), // White for neutral
            color_high: (40, 167, 69),  // Green for high/positive
            show_labels: true,
            title: None,
        }
    }
}

/// Interpolate between two RGB colors.
fn interpolate_color(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    (
        (c1.0 as f64 + (c2.0 as f64 - c1.0 as f64) * t) as u8,
        (c1.1 as f64 + (c2.1 as f64 - c1.1 as f64) * t) as u8,
        (c1.2 as f64 + (c2.2 as f64 - c1.2 as f64) * t) as u8,
    )
}

/// Get color for a normalized value (0.0 = low, 0.5 = mid, 1.0 = high).
fn value_to_color(value: f64, config: &HeatmapSvgConfig) -> String {
    let (r, g, b) = if value < 0.5 {
        interpolate_color(config.color_low, config.color_mid, value * 2.0)
    } else {
        interpolate_color(config.color_mid, config.color_high, (value - 0.5) * 2.0)
    };
    format!("rgb({},{},{})", r, g, b)
}

/// Generate SVG heatmap from HeatmapData.
///
/// # Arguments
/// * `heatmap` - The heatmap data to visualize.
/// * `config` - Configuration for rendering.
///
/// # Returns
/// A string containing the complete SVG document.
pub fn heatmap_to_svg(heatmap: &HeatmapData, config: &HeatmapSvgConfig) -> String {
    let x_len = heatmap.x_values.len();
    let y_len = heatmap.y_values.len();

    if x_len == 0 || y_len == 0 {
        return r#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#.to_string();
    }

    let grid_width = x_len * config.cell_width;
    let grid_height = y_len * config.cell_height;
    let total_width = grid_width + 2 * config.margin;
    let total_height = grid_height + 2 * config.margin + 30; // Extra for title

    // Find min/max for normalization (values is Vec<Vec<Option<f64>>>)
    let values: Vec<f64> = heatmap
        .values
        .iter()
        .flat_map(|row| row.iter())
        .filter_map(|&v| v)
        .collect();

    if values.is_empty() {
        return r#"<svg xmlns="http://www.w3.org/2000/svg"></svg>"#.to_string();
    }

    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() > f64::EPSILON {
        max_val - min_val
    } else {
        1.0
    };

    let mut svg = String::new();

    // SVG header
    writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        total_width, total_height, total_width, total_height
    )
    .unwrap();

    // Background
    writeln!(svg, r#"  <rect width="100%" height="100%" fill="white"/>"#).unwrap();

    // Title
    if let Some(title) = &config.title {
        writeln!(svg, r#"  <text x="{}" y="25" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold">{}</text>"#,
            total_width / 2, title).unwrap();
    }

    let title_offset = if config.title.is_some() { 30 } else { 0 };

    // Draw cells (values is Vec<Vec<Option<f64>>>, indexed as [y_idx][x_idx])
    for (y_idx, row) in heatmap.values.iter().enumerate() {
        for (x_idx, &value) in row.iter().enumerate() {
            let cell_x = config.margin + x_idx * config.cell_width;
            let cell_y = config.margin + title_offset + y_idx * config.cell_height;

            let fill_color = match value {
                Some(v) => {
                    let normalized = (v - min_val) / range;
                    value_to_color(normalized, config)
                }
                None => "rgb(200,200,200)".to_string(), // Gray for missing values
            };

            // Cell rectangle - escape the # in stroke color
            let stroke_color = "rgb(153,153,153)";
            writeln!(svg, r#"  <rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="1"/>"#,
                cell_x, cell_y, config.cell_width, config.cell_height, fill_color, stroke_color).unwrap();

            // Value label
            if config.show_labels {
                if let Some(v) = value {
                    let text_x = cell_x + config.cell_width / 2;
                    let text_y = cell_y + config.cell_height / 2 + 4;
                    let text_color = if (v - min_val) / range > 0.5 {
                        "white"
                    } else {
                        "black"
                    };
                    writeln!(svg, r#"  <text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="{}">{:.2}</text>"#,
                        text_x, text_y, text_color, v).unwrap();
                }
            }
        }
    }

    // X-axis labels
    for (i, x_val) in heatmap.x_values.iter().enumerate() {
        let label_x = config.margin + i * config.cell_width + config.cell_width / 2;
        let label_y = config.margin + title_offset + y_len * config.cell_height + 15;
        writeln!(svg, r#"  <text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="10">{:.2}</text>"#,
            label_x, label_y, x_val).unwrap();
    }

    // Y-axis labels
    for (i, y_val) in heatmap.y_values.iter().enumerate() {
        let label_x = config.margin - 5;
        let label_y =
            config.margin + title_offset + i * config.cell_height + config.cell_height / 2 + 4;
        writeln!(svg, r#"  <text x="{}" y="{}" text-anchor="end" font-family="sans-serif" font-size="10">{:.2}</text>"#,
            label_x, label_y, y_val).unwrap();
    }

    // Axis labels
    writeln!(svg, r#"  <text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold">{}</text>"#,
        config.margin + grid_width / 2,
        total_height - 5,
        &heatmap.x_param).unwrap();

    writeln!(svg, r#"  <text x="15" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" transform="rotate(-90 15 {})">{}</text>"#,
        config.margin + title_offset + grid_height / 2,
        config.margin + title_offset + grid_height / 2,
        &heatmap.y_param).unwrap();

    writeln!(svg, "</svg>").unwrap();

    svg
}

/// Export heatmap as SVG file.
pub fn export_heatmap_svg(heatmap: &HeatmapData, path: &str) -> std::io::Result<()> {
    let svg = heatmap_to_svg(heatmap, &HeatmapSvgConfig::default());
    std::fs::write(path, svg)
}

/// Export heatmap as SVG file with custom configuration.
pub fn export_heatmap_svg_with_config(
    heatmap: &HeatmapData,
    path: &str,
    config: &HeatmapSvgConfig,
) -> std::io::Result<()> {
    let svg = heatmap_to_svg(heatmap, config);
    std::fs::write(path, svg)
}

// ============================================================================
// Monte Carlo Visualization
// ============================================================================

use crate::monte_carlo::MonteCarloResult;

/// Generate an ASCII histogram of Monte Carlo simulation results.
///
/// Shows return, Sharpe, and drawdown distributions with key statistics.
///
/// # Arguments
/// * `result` - The Monte Carlo simulation result.
/// * `width` - Width of the histogram bars (default: 40).
///
/// # Returns
/// A string containing the formatted Monte Carlo visualization.
pub fn monte_carlo_chart(result: &MonteCarloResult, width: usize) -> String {
    let mut output = String::new();

    writeln!(output, "Monte Carlo Simulation ({} iterations)", result.num_simulations).unwrap();
    writeln!(output, "{}", "═".repeat(60)).unwrap();

    // Return distribution histogram
    if !result.return_distribution.is_empty() {
        writeln!(output, "\nReturn Distribution:").unwrap();
        output.push_str(&histogram(&result.return_distribution, width, "%"));

        writeln!(
            output,
            "  Mean: {:+.2}%  Median: {:+.2}%  Std: {:.2}%",
            result.mean_return, result.median_return, result.return_std
        )
        .unwrap();
        writeln!(
            output,
            "  95% CI: [{:+.2}%, {:+.2}%]  P(>0): {:.1}%",
            result.return_ci.0,
            result.return_ci.1,
            result.prob_positive_return * 100.0
        )
        .unwrap();
    }

    // Sharpe distribution histogram
    if !result.sharpe_distribution.is_empty() {
        writeln!(output, "\nSharpe Ratio Distribution:").unwrap();
        output.push_str(&histogram(&result.sharpe_distribution, width, ""));

        writeln!(
            output,
            "  Mean: {:.2}  Median: {:.2}",
            result.mean_sharpe, result.median_sharpe
        )
        .unwrap();
        writeln!(
            output,
            "  95% CI: [{:.2}, {:.2}]  P(>0): {:.1}%",
            result.sharpe_ci.0,
            result.sharpe_ci.1,
            result.prob_positive_sharpe * 100.0
        )
        .unwrap();
    }

    // Drawdown distribution histogram
    if !result.drawdown_distribution.is_empty() {
        writeln!(output, "\nMax Drawdown Distribution:").unwrap();
        output.push_str(&histogram(&result.drawdown_distribution, width, "%"));

        writeln!(
            output,
            "  Mean: {:.2}%  Median: {:.2}%  95th: {:.2}%",
            result.mean_max_drawdown, result.median_max_drawdown, result.max_drawdown_95th
        )
        .unwrap();
    }

    // Risk metrics
    writeln!(output, "\n{}", "─".repeat(60)).unwrap();
    writeln!(
        output,
        "Risk: VaR(95%): {:.2}%  CVaR: {:.2}%",
        result.var, result.cvar
    )
    .unwrap();
    writeln!(
        output,
        "Robustness Score: {:.0}/100  Verdict: {}",
        result.robustness_score(),
        result.verdict().label().to_uppercase()
    )
    .unwrap();

    output
}

/// Generate a simple ASCII histogram from distribution data.
fn histogram(data: &[f64], width: usize, suffix: &str) -> String {
    const NUM_BINS: usize = 10;

    if data.is_empty() {
        return String::new();
    }

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range <= 0.0 {
        return format!("  All values: {:.2}{}\n", min_val, suffix);
    }

    let bin_width = range / NUM_BINS as f64;

    // Count values in each bin
    let mut bins = [0usize; NUM_BINS];
    for &val in data {
        let bin_idx = ((val - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(NUM_BINS - 1);
        bins[bin_idx] += 1;
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut output = String::new();

    for (i, &count) in bins.iter().enumerate() {
        let bin_start = min_val + i as f64 * bin_width;
        let bar_len = if max_count > 0 {
            (count as f64 / max_count as f64 * width as f64) as usize
        } else {
            0
        };
        let bar: String = "█".repeat(bar_len);
        let percentage = count as f64 / data.len() as f64 * 100.0;

        writeln!(
            output,
            "  {:>7.1}{} |{:<width$}| {:>5.1}%",
            bin_start,
            suffix,
            bar,
            percentage,
            width = width
        )
        .unwrap();
    }

    output
}

// ============================================================================
// ASCII Heatmap (Terminal-friendly)
// ============================================================================

/// Characters for ASCII heatmap, ordered from low to high intensity.
const HEATMAP_CHARS: [char; 5] = ['░', '▒', '▓', '█', '█'];

/// Generate ASCII heatmap from HeatmapData.
pub fn heatmap_to_ascii(heatmap: &HeatmapData) -> String {
    let x_len = heatmap.x_values.len();
    let y_len = heatmap.y_values.len();

    if x_len == 0 || y_len == 0 {
        return "Empty heatmap".to_string();
    }

    // Find min/max for normalization (values is Vec<Vec<Option<f64>>>)
    let values: Vec<f64> = heatmap
        .values
        .iter()
        .flat_map(|row| row.iter())
        .filter_map(|&v| v)
        .collect();

    if values.is_empty() {
        return "No values in heatmap".to_string();
    }

    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() > f64::EPSILON {
        max_val - min_val
    } else {
        1.0
    };

    let mut output = String::new();

    // Header
    writeln!(
        output,
        "Parameter Sensitivity Heatmap: {} vs {}",
        heatmap.y_param, heatmap.x_param
    )
    .unwrap();
    writeln!(output, "{}", "─".repeat(x_len * 3 + 10)).unwrap();

    // X-axis labels (truncated)
    write!(output, "{:>8} ", "").unwrap();
    for x_val in &heatmap.x_values {
        write!(
            output,
            "{:>3}",
            format!("{:.0}", x_val).chars().take(3).collect::<String>()
        )
        .unwrap();
    }
    writeln!(output).unwrap();

    // Rows (values[y_idx][x_idx])
    for (y_idx, y_val) in heatmap.y_values.iter().enumerate() {
        write!(output, "{:>8} ", format!("{:.1}", y_val)).unwrap();

        for x_idx in 0..x_len {
            let ch = match heatmap
                .values
                .get(y_idx)
                .and_then(|row| row.get(x_idx))
                .and_then(|&v| v)
            {
                Some(v) => {
                    let normalized = (v - min_val) / range;
                    let char_idx = ((normalized * 4.0) as usize).min(4);
                    HEATMAP_CHARS[char_idx]
                }
                None => ' ',
            };
            write!(output, " {} ", ch).unwrap();
        }
        writeln!(output).unwrap();
    }

    // Legend
    writeln!(output, "\nLegend: ░=low ▒=med ▓=high █=highest").unwrap();
    writeln!(output, "Value range: {:.2} to {:.2}", min_val, max_val).unwrap();

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparkline_empty() {
        let result = sparkline(&[], 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sparkline_single_value() {
        let result = sparkline(&[100.0], 10);
        // Single value produces a single character (UTF-8 may be multi-byte)
        assert_eq!(result.chars().count(), 1);
    }

    #[test]
    fn test_sparkline_increasing() {
        let values: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let result = sparkline(&values, 8);
        assert_eq!(result.chars().count(), 8);
        // First char should be lowest, last should be highest
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARKLINE_CHARS[0]);
        assert_eq!(chars[7], SPARKLINE_CHARS[7]);
    }

    #[test]
    fn test_sparkline_decreasing() {
        let values: Vec<f64> = (0..8).rev().map(|i| i as f64).collect();
        let result = sparkline(&values, 8);
        let chars: Vec<char> = result.chars().collect();
        assert_eq!(chars[0], SPARKLINE_CHARS[7]);
        assert_eq!(chars[7], SPARKLINE_CHARS[0]);
    }

    #[test]
    fn test_sparkline_constant() {
        let values = vec![50.0; 10];
        let result = sparkline(&values, 10);
        // All values should be mid-level
        for ch in result.chars() {
            assert_eq!(ch, SPARKLINE_CHARS[4]); // Middle index for constant values
        }
    }

    #[test]
    fn test_sparkline_downsampling() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = sparkline(&values, 10);
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn test_downsample() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let downsampled = downsample(&values, 10);
        assert_eq!(downsampled.len(), 10);
        // First chunk should average to ~4.5, last chunk should average to ~94.5
        assert!(downsampled[0] < 10.0);
        assert!(downsampled[9] > 90.0);
    }

    #[test]
    fn test_format_pct() {
        assert_eq!(format_pct(10.5), "+10.5%");
        assert_eq!(format_pct(-5.3), "-5.3%");
        assert_eq!(format_pct(0.0), "+0.0%");
    }

    #[test]
    fn test_truncate_str() {
        assert_eq!(truncate_str("hello", 10), "hello");
        assert_eq!(truncate_str("hello world", 8), "hello...");
        assert_eq!(truncate_str("ab", 2), "ab");
    }

    #[test]
    fn test_interpolate_color() {
        let c1 = (0, 0, 0);
        let c2 = (255, 255, 255);

        let mid = interpolate_color(c1, c2, 0.5);
        assert_eq!(mid, (127, 127, 127));

        let start = interpolate_color(c1, c2, 0.0);
        assert_eq!(start, (0, 0, 0));

        let end = interpolate_color(c1, c2, 1.0);
        assert_eq!(end, (255, 255, 255));
    }

    #[test]
    fn test_heatmap_svg_empty() {
        let heatmap = HeatmapData {
            x_param: "x".to_string(),
            y_param: "y".to_string(),
            x_values: vec![],
            y_values: vec![],
            values: vec![],
            metric: SensitivityMetric::Sharpe,
        };
        let svg = heatmap_to_svg(&heatmap, &HeatmapSvgConfig::default());
        assert!(svg.contains("<svg"));
    }

    #[test]
    fn test_heatmap_svg_basic() {
        let heatmap = HeatmapData {
            x_param: "param_a".to_string(),
            y_param: "param_b".to_string(),
            x_values: vec![1.0, 2.0, 3.0],
            y_values: vec![10.0, 20.0],
            // 2D: 2 rows (y values) x 3 cols (x values)
            values: vec![
                vec![Some(0.5), Some(0.7), Some(0.9)],
                vec![Some(0.3), Some(0.6), Some(0.8)],
            ],
            metric: SensitivityMetric::Sharpe,
        };
        let svg = heatmap_to_svg(&heatmap, &HeatmapSvgConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("param_a"));
        assert!(svg.contains("param_b"));
    }

    #[test]
    fn test_heatmap_ascii_basic() {
        let heatmap = HeatmapData {
            x_param: "param_a".to_string(),
            y_param: "param_b".to_string(),
            x_values: vec![1.0, 2.0, 3.0],
            y_values: vec![10.0, 20.0],
            // 2D: 2 rows x 3 cols
            values: vec![
                vec![Some(0.1), Some(0.5), Some(0.9)],
                vec![Some(0.2), Some(0.6), Some(1.0)],
            ],
            metric: SensitivityMetric::Sharpe,
        };
        let ascii = heatmap_to_ascii(&heatmap);
        assert!(ascii.contains("param_a"));
        assert!(ascii.contains("param_b"));
        assert!(ascii.contains("Legend"));
    }
}
