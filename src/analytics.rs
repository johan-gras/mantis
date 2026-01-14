//! Performance analytics and reporting.

use crate::engine::BacktestResult;
use crate::types::Trade;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use tabled::{builder::Builder, settings::Style};

/// Benchmark comparison metrics.
/// These metrics compare portfolio performance against a benchmark index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Benchmark name (e.g., "SPY", "QQQ").
    pub benchmark_name: String,
    /// Jensen's alpha - risk-adjusted excess return.
    /// Measures the portfolio's return above the expected return given its beta.
    pub alpha: f64,
    /// Portfolio beta - sensitivity to benchmark movements.
    /// Beta > 1 means more volatile than benchmark, < 1 means less volatile.
    pub beta: f64,
    /// Tracking error - standard deviation of excess returns.
    /// Measures how closely the portfolio follows the benchmark.
    pub tracking_error: f64,
    /// Information ratio - alpha divided by tracking error.
    /// Measures risk-adjusted excess return per unit of active risk.
    pub information_ratio: f64,
    /// Correlation coefficient with benchmark (-1 to 1).
    pub correlation: f64,
    /// Upside capture ratio - percentage of benchmark gains captured.
    /// > 100% means outperforming in up markets.
    pub up_capture: f64,
    /// Downside capture ratio - percentage of benchmark losses captured.
    /// < 100% means protecting better in down markets.
    pub down_capture: f64,
    /// Benchmark total return for the period.
    pub benchmark_return_pct: f64,
    /// Excess return (portfolio return - benchmark return).
    pub excess_return_pct: f64,
}

impl BenchmarkMetrics {
    /// Calculate benchmark comparison metrics from aligned return series.
    ///
    /// # Arguments
    /// * `benchmark_name` - Name of the benchmark
    /// * `portfolio_returns` - Daily portfolio returns (as decimals, e.g., 0.01 for 1%)
    /// * `benchmark_returns` - Daily benchmark returns (as decimals)
    /// * `risk_free_rate` - Annual risk-free rate (as decimal, e.g., 0.05 for 5%)
    ///
    /// Both return series must be aligned (same dates, same length).
    pub fn calculate(
        benchmark_name: impl Into<String>,
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
        risk_free_rate: f64,
    ) -> Option<Self> {
        if portfolio_returns.is_empty()
            || benchmark_returns.is_empty()
            || portfolio_returns.len() != benchmark_returns.len()
        {
            return None;
        }

        let n = portfolio_returns.len() as f64;

        // Calculate means
        let port_mean: f64 = portfolio_returns.iter().sum::<f64>() / n;
        let bench_mean: f64 = benchmark_returns.iter().sum::<f64>() / n;

        // Calculate beta: Cov(portfolio, benchmark) / Var(benchmark)
        let covariance: f64 = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(p, b)| (p - port_mean) * (b - bench_mean))
            .sum::<f64>()
            / n;

        let bench_variance: f64 = benchmark_returns
            .iter()
            .map(|b| (b - bench_mean).powi(2))
            .sum::<f64>()
            / n;

        let port_variance: f64 = portfolio_returns
            .iter()
            .map(|p| (p - port_mean).powi(2))
            .sum::<f64>()
            / n;

        let beta = if bench_variance > 0.0 {
            covariance / bench_variance
        } else {
            0.0
        };

        // Calculate alpha (annualized Jensen's alpha)
        // Alpha = (Portfolio Return - Rf) - Beta * (Benchmark Return - Rf)
        let annualized_port_return = (1.0 + port_mean).powf(252.0) - 1.0;
        let annualized_bench_return = (1.0 + bench_mean).powf(252.0) - 1.0;
        let alpha = (annualized_port_return - risk_free_rate)
            - beta * (annualized_bench_return - risk_free_rate);

        // Calculate correlation
        let port_std = port_variance.sqrt();
        let bench_std = bench_variance.sqrt();
        let correlation = if port_std > 0.0 && bench_std > 0.0 {
            covariance / (port_std * bench_std)
        } else {
            0.0
        };

        // Calculate excess returns and tracking error
        let excess_returns: Vec<f64> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .map(|(p, b)| p - b)
            .collect();

        let excess_mean: f64 = excess_returns.iter().sum::<f64>() / n;
        let tracking_error_daily: f64 = (excess_returns
            .iter()
            .map(|e| (e - excess_mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let tracking_error = tracking_error_daily * 252.0_f64.sqrt() * 100.0; // Annualized as percentage

        // Calculate information ratio (annualized)
        let information_ratio = if tracking_error > 0.0 {
            (excess_mean * 252.0 * 100.0) / tracking_error
        } else {
            0.0
        };

        // Calculate capture ratios
        let (up_capture, down_capture) =
            Self::calculate_capture_ratios(portfolio_returns, benchmark_returns);

        // Calculate total returns
        let portfolio_total_return: f64 =
            portfolio_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let benchmark_total_return: f64 =
            benchmark_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        Some(Self {
            benchmark_name: benchmark_name.into(),
            alpha: alpha * 100.0, // Convert to percentage
            beta,
            tracking_error,
            information_ratio,
            correlation,
            up_capture,
            down_capture,
            benchmark_return_pct: benchmark_total_return * 100.0,
            excess_return_pct: (portfolio_total_return - benchmark_total_return) * 100.0,
        })
    }

    /// Calculate upside and downside capture ratios.
    fn calculate_capture_ratios(
        portfolio_returns: &[f64],
        benchmark_returns: &[f64],
    ) -> (f64, f64) {
        let up_periods: Vec<(f64, f64)> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .filter(|(_, b)| **b > 0.0)
            .map(|(p, b)| (*p, *b))
            .collect();

        let down_periods: Vec<(f64, f64)> = portfolio_returns
            .iter()
            .zip(benchmark_returns.iter())
            .filter(|(_, b)| **b < 0.0)
            .map(|(p, b)| (*p, *b))
            .collect();

        let up_capture = if !up_periods.is_empty() {
            let port_up: f64 = up_periods.iter().map(|(p, _)| p).sum::<f64>();
            let bench_up: f64 = up_periods.iter().map(|(_, b)| b).sum::<f64>();
            if bench_up > 0.0 {
                (port_up / bench_up) * 100.0
            } else {
                100.0
            }
        } else {
            100.0
        };

        let down_capture = if !down_periods.is_empty() {
            let port_down: f64 = down_periods.iter().map(|(p, _)| p).sum::<f64>();
            let bench_down: f64 = down_periods.iter().map(|(_, b)| b).sum::<f64>();
            if bench_down < 0.0 {
                (port_down / bench_down) * 100.0
            } else {
                100.0
            }
        } else {
            100.0
        };

        (up_capture, down_capture)
    }

    /// Extract daily returns from an equity curve.
    /// This produces actual daily returns rather than synthetic uniform returns.
    pub fn extract_daily_returns(equity_curve: &[crate::types::EquityPoint]) -> Vec<f64> {
        if equity_curve.len() < 2 {
            return vec![];
        }

        equity_curve
            .windows(2)
            .map(|w| (w[1].equity - w[0].equity) / w[0].equity)
            .collect()
    }

    /// Extract daily returns from a price series (e.g., benchmark close prices).
    pub fn extract_returns_from_prices(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return vec![];
        }

        prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
    }
}

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

    /// Print benchmark comparison metrics.
    pub fn print_benchmark_comparison(metrics: &BenchmarkMetrics) {
        println!();
        println!(
            "{}",
            format!("Benchmark Comparison ({})", metrics.benchmark_name)
                .bold()
                .underline()
        );
        println!(
            "  Benchmark Return: {:>10.2}%",
            metrics.benchmark_return_pct
        );
        println!(
            "  Excess Return:    {:>10.2}%  {}",
            metrics.excess_return_pct,
            Self::format_pct_change(metrics.excess_return_pct)
        );
        println!("  Alpha:            {:>10.2}%", metrics.alpha);
        println!("  Beta:             {:>10.2}", metrics.beta);
        println!("  Correlation:      {:>10.2}", metrics.correlation);
        println!();
        println!("  Tracking Error:   {:>10.2}%", metrics.tracking_error);
        println!("  Information Ratio:{:>10.2}", metrics.information_ratio);
        println!();
        println!("  Up Capture:       {:>10.2}%", metrics.up_capture);
        println!("  Down Capture:     {:>10.2}%", metrics.down_capture);
        println!();
    }

    /// Print a comprehensive report including benchmark comparison.
    pub fn print_report_with_benchmark(
        result: &BacktestResult,
        benchmark_metrics: Option<&BenchmarkMetrics>,
    ) {
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

        // Benchmark Comparison (if available)
        if let Some(bm) = benchmark_metrics {
            Self::print_benchmark_comparison(bm);
        }

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

    #[test]
    fn test_benchmark_metrics_calculation() {
        // Simulated daily returns - portfolio outperforms benchmark
        let portfolio_returns = vec![0.01, -0.005, 0.015, 0.008, -0.012, 0.02, 0.005, -0.003];
        let benchmark_returns = vec![0.008, -0.003, 0.01, 0.005, -0.01, 0.015, 0.003, -0.002];

        let metrics =
            BenchmarkMetrics::calculate("SPY", &portfolio_returns, &benchmark_returns, 0.05);

        assert!(metrics.is_some());
        let m = metrics.unwrap();

        // Beta should be positive (correlated with market)
        assert!(m.beta > 0.0, "Beta should be positive: {}", m.beta);

        // Correlation should be positive
        assert!(
            m.correlation > 0.0,
            "Correlation should be positive: {}",
            m.correlation
        );
        assert!(
            m.correlation <= 1.0,
            "Correlation should be <= 1: {}",
            m.correlation
        );

        // Tracking error should be positive
        assert!(
            m.tracking_error > 0.0,
            "Tracking error should be positive: {}",
            m.tracking_error
        );

        // Excess return should be positive (portfolio outperformed)
        assert!(
            m.excess_return_pct > 0.0,
            "Excess return should be positive: {}",
            m.excess_return_pct
        );

        // Capture ratios should be reasonable
        assert!(
            m.up_capture > 0.0,
            "Up capture should be positive: {}",
            m.up_capture
        );
        assert!(
            m.down_capture > 0.0,
            "Down capture should be positive: {}",
            m.down_capture
        );

        // Benchmark name should be set
        assert_eq!(m.benchmark_name, "SPY");
    }

    #[test]
    fn test_benchmark_metrics_perfect_correlation() {
        // Portfolio returns match benchmark exactly
        let returns = vec![0.01, -0.005, 0.015, 0.008, -0.012];

        let metrics = BenchmarkMetrics::calculate("SPY", &returns, &returns, 0.0);

        assert!(metrics.is_some());
        let m = metrics.unwrap();

        // Perfect correlation
        assert!(
            (m.correlation - 1.0).abs() < 0.0001,
            "Correlation should be 1.0: {}",
            m.correlation
        );

        // Beta should be 1.0
        assert!(
            (m.beta - 1.0).abs() < 0.0001,
            "Beta should be 1.0: {}",
            m.beta
        );

        // Tracking error should be near zero
        assert!(
            m.tracking_error < 0.1,
            "Tracking error should be near zero: {}",
            m.tracking_error
        );

        // Excess return should be zero
        assert!(
            m.excess_return_pct.abs() < 0.0001,
            "Excess return should be zero: {}",
            m.excess_return_pct
        );
    }

    #[test]
    fn test_benchmark_metrics_empty_returns() {
        let empty: Vec<f64> = vec![];
        let returns = vec![0.01, -0.005];

        // Empty portfolio returns
        assert!(BenchmarkMetrics::calculate("SPY", &empty, &returns, 0.0).is_none());

        // Empty benchmark returns
        assert!(BenchmarkMetrics::calculate("SPY", &returns, &empty, 0.0).is_none());

        // Both empty
        assert!(BenchmarkMetrics::calculate("SPY", &empty, &empty, 0.0).is_none());
    }

    #[test]
    fn test_benchmark_metrics_mismatched_lengths() {
        let portfolio = vec![0.01, -0.005, 0.015];
        let benchmark = vec![0.01, -0.005]; // Different length

        let metrics = BenchmarkMetrics::calculate("SPY", &portfolio, &benchmark, 0.0);
        assert!(
            metrics.is_none(),
            "Should return None for mismatched lengths"
        );
    }

    #[test]
    fn test_extract_daily_returns() {
        use crate::types::EquityPoint;

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
                equity: 101000.0, // +1%
                cash: 0.0,
                positions_value: 101000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap(),
                equity: 100500.0, // -0.495%
                cash: 0.0,
                positions_value: 100500.0,
                drawdown: 500.0,
                drawdown_pct: 0.495,
            },
        ];

        let returns = BenchmarkMetrics::extract_daily_returns(&equity_curve);

        assert_eq!(returns.len(), 2);
        assert!((returns[0] - 0.01).abs() < 0.0001); // +1%
        assert!((returns[1] - (-0.00495)).abs() < 0.0001); // -0.495%
    }

    #[test]
    fn test_extract_returns_from_prices() {
        let prices = vec![100.0, 101.0, 100.5, 102.0];

        let returns = BenchmarkMetrics::extract_returns_from_prices(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.01).abs() < 0.0001); // 101/100 - 1 = 0.01
        assert!((returns[1] - (-0.00495)).abs() < 0.0001); // 100.5/101 - 1
        assert!((returns[2] - 0.01493).abs() < 0.0001); // 102/100.5 - 1
    }

    #[test]
    fn test_capture_ratios_all_up_market() {
        // All positive benchmark returns
        let portfolio = vec![0.02, 0.01, 0.015];
        let benchmark = vec![0.01, 0.005, 0.01];

        let metrics = BenchmarkMetrics::calculate("SPY", &portfolio, &benchmark, 0.0).unwrap();

        // Portfolio captured 200% of benchmark gains
        assert!(
            metrics.up_capture > 100.0,
            "Up capture should be > 100%: {}",
            metrics.up_capture
        );
        // Down capture should be 100% (no down periods)
        assert!(
            (metrics.down_capture - 100.0).abs() < 0.01,
            "Down capture should be 100%: {}",
            metrics.down_capture
        );
    }

    #[test]
    fn test_benchmark_serialization() {
        let portfolio_returns = vec![0.01, -0.005, 0.015];
        let benchmark_returns = vec![0.008, -0.003, 0.01];

        let metrics =
            BenchmarkMetrics::calculate("SPY", &portfolio_returns, &benchmark_returns, 0.0)
                .unwrap();

        // Test JSON serialization
        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("SPY"));
        assert!(json.contains("alpha"));
        assert!(json.contains("beta"));

        // Test deserialization
        let deserialized: BenchmarkMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.benchmark_name, "SPY");
        assert!((deserialized.beta - metrics.beta).abs() < 0.0001);
    }
}
