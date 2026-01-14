//! Performance analytics and reporting.

use crate::engine::BacktestResult;
use crate::types::Trade;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use tabled::{builder::Builder, settings::Style};

/// Comprehensive performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Returns
    pub total_return_pct: f64,
    pub annual_return_pct: f64,
    pub monthly_return_avg: f64,

    // Risk metrics
    pub volatility_annual: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_duration_days: i64,
    pub avg_drawdown_pct: f64,
    pub ulcer_index: f64,

    // Risk-adjusted returns
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub omega_ratio: f64,

    // Trade statistics
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    pub avg_trade: f64,
    pub avg_holding_period_days: f64,

    // Expectancy
    pub expectancy: f64,
    pub expectancy_ratio: f64,

    // Consistency
    pub winning_months: usize,
    pub losing_months: usize,
    pub best_month_pct: f64,
    pub worst_month_pct: f64,

    // Recovery
    pub recovery_factor: f64,
    pub payoff_ratio: f64,
}

impl PerformanceMetrics {
    /// Calculate comprehensive metrics from backtest results.
    pub fn from_result(result: &BacktestResult) -> Self {
        let _returns = Self::calculate_returns(&result.trades, result.initial_capital);
        let daily_returns = Self::calculate_daily_returns(result);

        // Volatility
        let volatility_annual = Self::annualized_volatility(&daily_returns);

        // Drawdown analysis
        let (max_dd_duration, avg_dd) = Self::drawdown_analysis(result);

        // Ulcer index
        let ulcer_index = Self::ulcer_index(result);

        // Omega ratio
        let omega_ratio = Self::omega_ratio(&daily_returns, 0.0);

        // Trade analysis
        let closed_trades: Vec<_> = result.trades.iter().filter(|t| t.is_closed()).collect();
        let winning: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) > 0.0)
            .collect();
        let losing: Vec<_> = closed_trades
            .iter()
            .filter(|t| t.net_pnl().unwrap_or(0.0) < 0.0)
            .collect();

        let largest_win = winning
            .iter()
            .filter_map(|t| t.net_pnl())
            .fold(0.0_f64, |a, b| a.max(b));
        let largest_loss = losing
            .iter()
            .filter_map(|t| t.net_pnl())
            .fold(0.0_f64, |a, b| a.min(b));

        let avg_trade = if !closed_trades.is_empty() {
            closed_trades
                .iter()
                .filter_map(|t| t.net_pnl())
                .sum::<f64>()
                / closed_trades.len() as f64
        } else {
            0.0
        };

        let avg_holding_period_days = if !closed_trades.is_empty() {
            closed_trades
                .iter()
                .filter_map(|t| t.holding_period())
                .map(|d| d.num_hours() as f64 / 24.0)
                .sum::<f64>()
                / closed_trades.len() as f64
        } else {
            0.0
        };

        // Expectancy
        let expectancy = (result.win_rate / 100.0) * result.avg_win
            - (1.0 - result.win_rate / 100.0) * result.avg_loss.abs();
        let expectancy_ratio = if result.avg_loss.abs() > 0.0 {
            expectancy / result.avg_loss.abs()
        } else {
            0.0
        };

        // Monthly analysis
        let monthly_returns = Self::monthly_returns(result);
        let winning_months = monthly_returns.iter().filter(|&&r| r > 0.0).count();
        let losing_months = monthly_returns.iter().filter(|&&r| r < 0.0).count();
        let best_month_pct = monthly_returns
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_month_pct = monthly_returns.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let monthly_return_avg = if !monthly_returns.is_empty() {
            monthly_returns.iter().sum::<f64>() / monthly_returns.len() as f64
        } else {
            0.0
        };

        // Recovery factor
        let total_profit = result.final_equity - result.initial_capital;
        let recovery_factor = if result.max_drawdown_pct > 0.0 {
            total_profit / (result.initial_capital * result.max_drawdown_pct / 100.0)
        } else {
            0.0
        };

        // Payoff ratio
        let payoff_ratio = if result.avg_loss.abs() > 0.0 {
            result.avg_win / result.avg_loss.abs()
        } else {
            0.0
        };

        Self {
            total_return_pct: result.total_return_pct,
            annual_return_pct: result.annual_return_pct,
            monthly_return_avg,
            volatility_annual,
            max_drawdown_pct: result.max_drawdown_pct,
            max_drawdown_duration_days: max_dd_duration,
            avg_drawdown_pct: avg_dd,
            ulcer_index,
            sharpe_ratio: result.sharpe_ratio,
            sortino_ratio: result.sortino_ratio,
            calmar_ratio: result.calmar_ratio,
            omega_ratio,
            total_trades: result.total_trades,
            winning_trades: result.winning_trades,
            losing_trades: result.losing_trades,
            win_rate: result.win_rate,
            profit_factor: result.profit_factor,
            avg_win: result.avg_win,
            avg_loss: result.avg_loss,
            largest_win,
            largest_loss,
            avg_trade,
            avg_holding_period_days,
            expectancy,
            expectancy_ratio,
            winning_months,
            losing_months,
            best_month_pct: if best_month_pct == f64::NEG_INFINITY {
                0.0
            } else {
                best_month_pct
            },
            worst_month_pct: if worst_month_pct == f64::INFINITY {
                0.0
            } else {
                worst_month_pct
            },
            recovery_factor,
            payoff_ratio,
        }
    }

    fn calculate_returns(trades: &[Trade], _initial_capital: f64) -> Vec<f64> {
        trades
            .iter()
            .filter(|t| t.is_closed())
            .filter_map(|t| t.return_pct())
            .collect()
    }

    fn calculate_daily_returns(result: &BacktestResult) -> Vec<f64> {
        // Approximate daily returns from total return and trading days
        if result.trading_days <= 1 {
            return vec![];
        }
        let daily_return = (result.final_equity / result.initial_capital)
            .powf(1.0 / result.trading_days as f64)
            - 1.0;
        vec![daily_return; result.trading_days]
    }

    fn annualized_volatility(daily_returns: &[f64]) -> f64 {
        if daily_returns.is_empty() {
            return 0.0;
        }
        let mean: f64 = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance: f64 = daily_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / daily_returns.len() as f64;
        variance.sqrt() * 252.0_f64.sqrt() * 100.0
    }

    fn drawdown_analysis(result: &BacktestResult) -> (i64, f64) {
        // Simplified drawdown analysis
        let avg_dd = result.max_drawdown_pct / 2.0; // Rough approximation
        let max_dd_duration = (result.end_time - result.start_time).num_days() / 10; // Rough estimate
        (max_dd_duration.max(0), avg_dd)
    }

    fn ulcer_index(result: &BacktestResult) -> f64 {
        // Ulcer Index = sqrt(sum(drawdown^2) / n)
        // Using simplified calculation
        let squared_dd = result.max_drawdown_pct.powi(2);
        (squared_dd / 2.0).sqrt() // Approximation
    }

    fn omega_ratio(returns: &[f64], threshold: f64) -> f64 {
        if returns.is_empty() {
            return 1.0;
        }

        let gains: f64 = returns
            .iter()
            .filter(|&&r| r > threshold)
            .map(|r| r - threshold)
            .sum();
        let losses: f64 = returns
            .iter()
            .filter(|&&r| r < threshold)
            .map(|r| threshold - r)
            .sum();

        if losses == 0.0 {
            if gains > 0.0 {
                f64::INFINITY
            } else {
                1.0
            }
        } else {
            gains / losses
        }
    }

    fn monthly_returns(result: &BacktestResult) -> Vec<f64> {
        // Simplified monthly returns estimation
        let months = (result.end_time - result.start_time).num_days() / 30;
        if months <= 0 {
            return vec![];
        }

        let monthly_return = result.total_return_pct / months as f64;
        vec![monthly_return; months as usize]
    }
}

/// Format results for terminal display.
pub struct ResultFormatter;

impl ResultFormatter {
    /// Print a comprehensive results report to stdout.
    pub fn print_report(result: &BacktestResult) {
        let metrics = PerformanceMetrics::from_result(result);

        println!();
        println!("{}", "═".repeat(60).blue());
        println!("{}", " BACKTEST RESULTS ".bold().blue());
        println!("{}", "═".repeat(60).blue());
        println!();

        // Overview
        println!("{}", "Overview".bold().underline());
        println!("  Strategy:        {}", result.strategy_name);
        println!("  Symbol(s):       {}", result.symbols.join(", "));
        println!(
            "  Period:          {} to {}",
            result.start_time.format("%Y-%m-%d"),
            result.end_time.format("%Y-%m-%d")
        );
        println!("  Trading Days:    {}", result.trading_days);
        println!();

        // Performance
        println!("{}", "Performance".bold().underline());
        println!("  Initial Capital: ${:>12.2}", result.initial_capital);
        println!(
            "  Final Equity:    ${:>12.2}  {}",
            result.final_equity,
            Self::format_pct_change(result.total_return_pct)
        );
        println!("  Total Return:    {:>12.2}%", result.total_return_pct);
        println!("  Annual Return:   {:>12.2}%", result.annual_return_pct);
        println!();

        // Risk Metrics
        println!("{}", "Risk Metrics".bold().underline());
        println!("  Max Drawdown:    {:>12.2}%", -result.max_drawdown_pct);
        println!("  Volatility:      {:>12.2}%", metrics.volatility_annual);
        println!("  Sharpe Ratio:    {:>12.2}", result.sharpe_ratio);
        println!("  Sortino Ratio:   {:>12.2}", result.sortino_ratio);
        println!("  Calmar Ratio:    {:>12.2}", result.calmar_ratio);
        println!();

        // Trade Statistics
        println!("{}", "Trade Statistics".bold().underline());
        println!("  Total Trades:    {:>12}", result.total_trades);
        println!(
            "  Winning Trades:  {:>12}  ({:.1}%)",
            result.winning_trades, result.win_rate
        );
        println!(
            "  Losing Trades:   {:>12}  ({:.1}%)",
            result.losing_trades,
            100.0 - result.win_rate
        );
        println!("  Profit Factor:   {:>12.2}", result.profit_factor);
        println!();

        println!("{}", "Trade P&L".bold().underline());
        println!("  Average Win:     ${:>11.2}", result.avg_win);
        println!("  Average Loss:    ${:>11.2}", result.avg_loss);
        println!("  Largest Win:     ${:>11.2}", metrics.largest_win);
        println!("  Largest Loss:    ${:>11.2}", metrics.largest_loss);
        println!("  Expectancy:      ${:>11.2}", metrics.expectancy);
        println!();

        println!("{}", "═".repeat(60).blue());
    }

    /// Format percentage change with color.
    fn format_pct_change(pct: f64) -> String {
        if pct >= 0.0 {
            format!("(+{:.2}%)", pct).green().to_string()
        } else {
            format!("({:.2}%)", pct).red().to_string()
        }
    }

    /// Print results as a table.
    pub fn print_table(results: &[BacktestResult]) {
        let mut builder = Builder::new();
        builder.push_record([
            "Strategy", "Return %", "Annual %", "Max DD %", "Sharpe", "Trades", "Win Rate",
        ]);

        for result in results {
            builder.push_record([
                result.strategy_name.clone(),
                format!("{:.2}", result.total_return_pct),
                format!("{:.2}", result.annual_return_pct),
                format!("{:.2}", -result.max_drawdown_pct),
                format!("{:.2}", result.sharpe_ratio),
                result.total_trades.to_string(),
                format!("{:.1}%", result.win_rate),
            ]);
        }

        let table = builder.build().with(Style::rounded()).to_string();
        println!("{}", table);
    }

    /// Export results to JSON.
    pub fn to_json(result: &BacktestResult) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export results to CSV line.
    pub fn to_csv_line(result: &BacktestResult) -> String {
        format!(
            "{},{},{:.2},{:.2},{:.2},{:.2},{},{},{:.1},{:.2},{:.2}",
            result.strategy_name,
            result.symbols.join(";"),
            result.initial_capital,
            result.final_equity,
            result.total_return_pct,
            result.annual_return_pct,
            result.total_trades,
            result.winning_trades,
            result.win_rate,
            result.sharpe_ratio,
            result.max_drawdown_pct
        )
    }

    /// Get CSV header.
    pub fn csv_header() -> &'static str {
        "strategy,symbols,initial_capital,final_equity,total_return_pct,annual_return_pct,total_trades,winning_trades,win_rate,sharpe_ratio,max_drawdown_pct"
    }
}

/// Trade report generator.
pub struct TradeReport;

impl TradeReport {
    /// Print individual trades.
    pub fn print_trades(trades: &[Trade], limit: usize) {
        let closed_trades: Vec<_> = trades.iter().filter(|t| t.is_closed()).collect();

        if closed_trades.is_empty() {
            println!("No closed trades.");
            return;
        }

        let display_trades = if limit > 0 && limit < closed_trades.len() {
            &closed_trades[..limit]
        } else {
            &closed_trades
        };

        let mut builder = Builder::new();
        builder.push_record([
            "#", "Symbol", "Side", "Qty", "Entry", "Exit", "P&L", "Return %",
        ]);

        for (i, trade) in display_trades.iter().enumerate() {
            let pnl = trade.net_pnl().unwrap_or(0.0);
            let pnl_str = if pnl >= 0.0 {
                format!("+{:.2}", pnl)
            } else {
                format!("{:.2}", pnl)
            };

            let ret = trade.return_pct().unwrap_or(0.0);
            let ret_str = if ret >= 0.0 {
                format!("+{:.2}%", ret)
            } else {
                format!("{:.2}%", ret)
            };

            builder.push_record([
                (i + 1).to_string(),
                trade.symbol.clone(),
                trade.side.to_string(),
                format!("{:.2}", trade.quantity),
                format!("{:.2}", trade.entry_price),
                format!("{:.2}", trade.exit_price.unwrap_or(0.0)),
                pnl_str,
                ret_str,
            ]);
        }

        let table = builder.build().with(Style::rounded()).to_string();
        println!("{}", table);

        if limit > 0 && limit < closed_trades.len() {
            println!("... and {} more trades", closed_trades.len() - limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::BacktestConfig;
    use crate::types::Side;
    use chrono::{TimeZone, Utc};

    fn create_test_result() -> BacktestResult {
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(110.0, end, 1.0);

        BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 10.0,
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
            calmar_ratio: 2.0,
            trades: vec![trade],
            equity_curve: vec![],
            start_time: start,
            end_time: end,
        }
    }

    #[test]
    fn test_metrics_calculation() {
        let result = create_test_result();
        let metrics = PerformanceMetrics::from_result(&result);

        assert_eq!(metrics.total_trades, 1);
        assert_eq!(metrics.winning_trades, 1);
        assert_eq!(metrics.win_rate, 100.0);
    }

    #[test]
    fn test_json_export() {
        let result = create_test_result();
        let json = ResultFormatter::to_json(&result);
        assert!(json.contains("TestStrategy"));
    }

    #[test]
    fn test_csv_export() {
        let result = create_test_result();
        let csv = ResultFormatter::to_csv_line(&result);
        assert!(csv.contains("TestStrategy"));
        assert!(csv.contains("TEST"));
    }
}
