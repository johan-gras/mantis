//! Performance analytics and reporting.

use crate::engine::BacktestResult;
use crate::types::Trade;
use chrono::Datelike;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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

/// Represents a single drawdown period from peak to recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownPeriod {
    /// Start of drawdown (when equity dropped below peak)
    pub start: chrono::DateTime<chrono::Utc>,
    /// Time of maximum drawdown depth (trough)
    pub trough: chrono::DateTime<chrono::Utc>,
    /// End of drawdown (when equity recovered to peak, None if not recovered)
    pub end: Option<chrono::DateTime<chrono::Utc>>,
    /// Maximum drawdown depth as percentage
    pub depth_pct: f64,
    /// Duration from start to end (or current if not recovered) in days
    pub duration_days: i64,
}

/// Comprehensive drawdown analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    /// Maximum drawdown percentage
    pub max_drawdown_pct: f64,
    /// Duration of the maximum drawdown period in days
    pub max_drawdown_duration_days: i64,
    /// Average of all drawdown depths
    pub avg_drawdown_pct: f64,
    /// Ulcer Index - sqrt(mean(drawdown_pct^2))
    pub ulcer_index: f64,
    /// All drawdown periods
    pub periods: Vec<DrawdownPeriod>,
    /// Time underwater as percentage of total time
    pub time_underwater_pct: f64,
}

impl DrawdownAnalysis {
    /// Calculate comprehensive drawdown analysis from an equity curve.
    /// This provides actual drawdown statistics rather than approximations.
    pub fn from_equity_curve(equity_curve: &[crate::types::EquityPoint]) -> Self {
        if equity_curve.len() < 2 {
            return Self::empty();
        }

        let mut peak = equity_curve[0].equity;
        let mut peak_time = equity_curve[0].timestamp;
        let mut periods: Vec<DrawdownPeriod> = Vec::new();
        let mut current_period: Option<DrawdownPeriod> = None;
        let mut drawdown_pcts: Vec<f64> = Vec::with_capacity(equity_curve.len());

        for point in equity_curve.iter() {
            if point.equity >= peak {
                // New peak - drawdown is 0%, close any existing drawdown period
                drawdown_pcts.push(0.0);
                if let Some(mut period) = current_period.take() {
                    period.end = Some(point.timestamp);
                    period.duration_days = (point.timestamp - period.start).num_days();
                    periods.push(period);
                }
                peak = point.equity;
                peak_time = point.timestamp;
            } else {
                // In drawdown - calculate percentage below peak
                let drawdown_pct = if peak > 0.0 {
                    ((peak - point.equity) / peak) * 100.0
                } else {
                    0.0
                };
                drawdown_pcts.push(drawdown_pct);

                // Track drawdown period
                match &mut current_period {
                    None => {
                        // Start new drawdown period
                        current_period = Some(DrawdownPeriod {
                            start: peak_time,
                            trough: point.timestamp,
                            end: None,
                            depth_pct: drawdown_pct,
                            duration_days: (point.timestamp - peak_time).num_days(),
                        });
                    }
                    Some(period) => {
                        // Update existing period if deeper
                        if drawdown_pct > period.depth_pct {
                            period.trough = point.timestamp;
                            period.depth_pct = drawdown_pct;
                        }
                        period.duration_days = (point.timestamp - period.start).num_days();
                    }
                }
            }
        }

        // Handle ongoing drawdown at end of data
        if let Some(period) = current_period {
            periods.push(period);
        }

        // Calculate statistics
        let max_drawdown_pct = periods
            .iter()
            .map(|p| p.depth_pct)
            .fold(0.0_f64, |a, b| a.max(b));

        let max_drawdown_duration_days = periods.iter().map(|p| p.duration_days).max().unwrap_or(0);

        let avg_drawdown_pct = if !periods.is_empty() {
            periods.iter().map(|p| p.depth_pct).sum::<f64>() / periods.len() as f64
        } else {
            0.0
        };

        // Ulcer Index: sqrt(mean(drawdown_pct^2))
        let ulcer_index = if !drawdown_pcts.is_empty() {
            let sum_squared: f64 = drawdown_pcts.iter().map(|d| d.powi(2)).sum();
            (sum_squared / drawdown_pcts.len() as f64).sqrt()
        } else {
            0.0
        };

        // Time underwater percentage
        let total_days = if equity_curve.len() >= 2 {
            (equity_curve.last().unwrap().timestamp - equity_curve.first().unwrap().timestamp)
                .num_days()
        } else {
            0
        };
        let underwater_days: i64 = periods.iter().map(|p| p.duration_days).sum();
        let time_underwater_pct = if total_days > 0 {
            (underwater_days as f64 / total_days as f64) * 100.0
        } else {
            0.0
        };

        Self {
            max_drawdown_pct,
            max_drawdown_duration_days,
            avg_drawdown_pct,
            ulcer_index,
            periods,
            time_underwater_pct,
        }
    }

    /// Create an empty analysis for when equity curve is unavailable.
    pub fn empty() -> Self {
        Self {
            max_drawdown_pct: 0.0,
            max_drawdown_duration_days: 0,
            avg_drawdown_pct: 0.0,
            ulcer_index: 0.0,
            periods: Vec::new(),
            time_underwater_pct: 0.0,
        }
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

    // Distribution metrics
    pub skewness: f64,
    pub kurtosis: f64,
    pub tail_ratio: f64,

    // Overfitting detection metrics
    pub deflated_sharpe_ratio: f64,
    pub probabilistic_sharpe_ratio: f64,

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
    /// For optimization contexts with multiple parameter combinations,
    /// use `from_result_with_trials` to properly adjust the deflated Sharpe ratio.
    pub fn from_result(result: &BacktestResult) -> Self {
        Self::from_result_with_trials(result, 1)
    }

    /// Calculate comprehensive metrics from backtest results with multiple testing adjustment.
    ///
    /// # Arguments
    /// * `result` - The backtest result to analyze
    /// * `n_trials` - Number of parameter combinations tested (for deflated Sharpe ratio)
    ///                Set to 1 for single backtest, or to the number of parameter
    ///                combinations tested during optimization.
    pub fn from_result_with_trials(result: &BacktestResult, n_trials: usize) -> Self {
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

        // Distribution metrics
        let skewness = Self::skewness(&daily_returns);
        let kurtosis = Self::kurtosis(&daily_returns);
        let tail_ratio = Self::tail_ratio(&daily_returns);

        // Overfitting detection metrics with multiple testing adjustment
        let n_observations = daily_returns.len();
        let deflated_sharpe_ratio =
            Self::deflated_sharpe_ratio(result.sharpe_ratio, n_trials, n_observations);
        let probabilistic_sharpe_ratio = Self::probabilistic_sharpe_ratio(
            result.sharpe_ratio,
            skewness,
            kurtosis,
            n_observations,
            0.0, // Benchmark SR = 0
        );

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
            skewness,
            kurtosis,
            tail_ratio,
            deflated_sharpe_ratio,
            probabilistic_sharpe_ratio,
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

    /// Calculate actual daily returns from the equity curve.
    /// This provides meaningful volatility calculations by using real equity changes
    /// instead of synthetic uniform returns.
    fn calculate_daily_returns(result: &BacktestResult) -> Vec<f64> {
        // Use actual equity curve if available
        if result.equity_curve.len() >= 2 {
            // Group equity points by day and take the last equity value for each day.
            // This handles both daily and intraday data correctly.
            let mut daily_equity: BTreeMap<(i32, u32, u32), f64> = BTreeMap::new();
            for point in &result.equity_curve {
                let key = (
                    point.timestamp.year(),
                    point.timestamp.month(),
                    point.timestamp.day(),
                );
                // BTreeMap::insert overwrites, so we keep the last value for each day
                daily_equity.insert(key, point.equity);
            }

            // Convert to vector of daily equity values (sorted by date due to BTreeMap)
            let equity_values: Vec<f64> = daily_equity.values().copied().collect();

            // Calculate returns between consecutive days
            if equity_values.len() >= 2 {
                return equity_values
                    .windows(2)
                    .map(|w| (w[1] - w[0]) / w[0])
                    .collect();
            }
        }

        // Fallback to synthetic returns only if equity curve is unavailable or too small
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

    /// Calculate drawdown analysis from the equity curve.
    /// Returns (max_drawdown_duration_days, average_drawdown_pct)
    fn drawdown_analysis(result: &BacktestResult) -> (i64, f64) {
        if result.equity_curve.len() >= 2 {
            let analysis = DrawdownAnalysis::from_equity_curve(&result.equity_curve);
            (
                analysis.max_drawdown_duration_days,
                analysis.avg_drawdown_pct,
            )
        } else {
            // Fallback to approximations when equity curve unavailable
            let avg_dd = result.max_drawdown_pct / 2.0;
            let max_dd_duration = (result.end_time - result.start_time).num_days() / 10;
            (max_dd_duration.max(0), avg_dd)
        }
    }

    /// Calculate Ulcer Index from the equity curve.
    /// Ulcer Index = sqrt(mean(drawdown_pct^2)) - measures downside volatility
    fn ulcer_index(result: &BacktestResult) -> f64 {
        if result.equity_curve.len() >= 2 {
            let analysis = DrawdownAnalysis::from_equity_curve(&result.equity_curve);
            analysis.ulcer_index
        } else {
            // Fallback approximation when equity curve unavailable
            let squared_dd = result.max_drawdown_pct.powi(2);
            (squared_dd / 2.0).sqrt()
        }
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

    /// Calculate skewness of returns distribution.
    /// Skewness = E[(X - μ)³] / σ³
    /// Positive skewness: right tail is longer (more extreme positive returns)
    /// Negative skewness: left tail is longer (more extreme negative returns)
    /// Zero skewness: symmetric distribution
    fn skewness(returns: &[f64]) -> f64 {
        if returns.len() < 3 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Calculate variance
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

        if variance == 0.0 {
            return 0.0;
        }

        let std_dev = variance.sqrt();

        // Calculate third moment
        let third_moment: f64 = returns.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n;

        third_moment / std_dev.powi(3)
    }

    /// Calculate excess kurtosis of returns distribution.
    /// Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
    /// Positive kurtosis: heavy tails (more extreme events than normal distribution)
    /// Negative kurtosis: light tails (fewer extreme events)
    /// Zero kurtosis: same tail behavior as normal distribution
    fn kurtosis(returns: &[f64]) -> f64 {
        if returns.len() < 4 {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Calculate variance
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;

        if variance == 0.0 {
            return 0.0;
        }

        // Calculate fourth moment
        let fourth_moment: f64 = returns.iter().map(|r| (r - mean).powi(4)).sum::<f64>() / n;

        // Subtract 3 to get excess kurtosis (relative to normal distribution)
        (fourth_moment / variance.powi(2)) - 3.0
    }

    /// Calculate tail ratio: 95th percentile gain / 95th percentile loss.
    /// Higher values indicate asymmetric upside potential.
    /// Tail ratio > 1: larger gains than losses in the tails
    /// Tail ratio < 1: larger losses than gains in the tails
    fn tail_ratio(returns: &[f64]) -> f64 {
        if returns.len() < 20 {
            return 1.0;
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 95th percentile (95% of returns are below this)
        let p95_idx = ((sorted.len() as f64) * 0.95) as usize;
        let p95_gain = sorted
            .get(p95_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);

        // 5th percentile (5% of returns are below this - the losses)
        let p05_idx = ((sorted.len() as f64) * 0.05) as usize;
        let p05_loss = sorted.get(p05_idx).copied().unwrap_or(0.0);

        // Tail ratio is the ratio of absolute values
        // Handle edge cases:
        if p05_loss.abs() < 1e-10 && p95_gain.abs() < 1e-10 {
            // Both near zero
            1.0
        } else if p05_loss.abs() < 1e-10 {
            // No downside tail, only upside
            100.0
        } else if p95_gain.abs() < 1e-10 {
            // No upside tail, only downside
            0.01
        } else {
            // Normal case: ratio of upside to downside tail
            p95_gain.abs() / p05_loss.abs()
        }
    }

    /// Calculate Deflated Sharpe Ratio (DSR) - adjusts for multiple testing bias.
    /// Based on Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio".
    ///
    /// DSR = (SR - E[SR_max]) / sqrt(Var[SR_max])
    ///
    /// Where E[SR_max] is the expected maximum Sharpe ratio under the null hypothesis
    /// (all strategies are random) given the number of trials.
    ///
    /// # Arguments
    /// * `sharpe` - The observed Sharpe ratio
    /// * `n_trials` - Number of independent strategies/parameter combinations tested
    /// * `n_observations` - Number of return observations (e.g., trading days)
    ///
    /// # Returns
    /// Deflated Sharpe Ratio. Values < 0 suggest the observed SR is likely due to luck.
    fn deflated_sharpe_ratio(sharpe: f64, n_trials: usize, n_observations: usize) -> f64 {
        if n_trials <= 1 {
            // No multiple testing - return original Sharpe
            return sharpe;
        }

        if n_observations < 2 {
            return 0.0;
        }

        // Expected maximum Sharpe ratio under null hypothesis (random strategies)
        // Approximation: E[SR_max] ≈ sqrt(2 * ln(N)) for large N
        let n = n_trials as f64;
        let expected_max_sr = (2.0 * n.ln()).sqrt();

        // Variance of maximum SR under null: Var[SR_max] ≈ 1 for large samples
        // More accurate: Var[SR_max] = 1 + (1 - Euler-Mascheroni) / N
        let euler_mascheroni = 0.5772156649;
        let var_max_sr = 1.0 + (1.0 - euler_mascheroni) / n;

        // Deflated Sharpe Ratio
        (sharpe - expected_max_sr) / var_max_sr.sqrt()
    }

    /// Calculate Probabilistic Sharpe Ratio (PSR) - probability that SR > benchmark SR.
    /// Based on Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier".
    ///
    /// PSR = Φ(Z) where Z = (SR - SR*) × sqrt(T-1) / sqrt(1 - γ₃×SR + (γ₄-1)/4 × SR²)
    ///
    /// PSR = P[True SR > Benchmark SR | observed data]
    ///
    /// # Arguments
    /// * `sharpe` - The observed annualized Sharpe ratio
    /// * `skewness` - Skewness of returns (γ₃)
    /// * `kurtosis` - Excess kurtosis of returns (γ₄)
    /// * `n_observations` - Number of return observations (T)
    /// * `benchmark_sr` - Reference Sharpe ratio (typically 0)
    ///
    /// # Returns
    /// Probability between 0 and 1. PSR > 0.95 suggests high confidence the strategy
    /// has skill (true SR > benchmark). PSR < 0.5 suggests the observed SR is likely luck.
    fn probabilistic_sharpe_ratio(
        sharpe: f64,
        skewness: f64,
        kurtosis: f64,
        n_observations: usize,
        benchmark_sr: f64,
    ) -> f64 {
        if n_observations < 2 {
            return 0.5; // No information
        }

        let t = n_observations as f64;

        // Calculate the denominator: adjusted variance of SR estimate
        // Var[SR] = (1 - γ₃×SR + (γ₄-1)/4 × SR²) / (T-1)
        let variance_adjustment = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe.powi(2);

        // Protect against negative variance (can happen with extreme skew/kurtosis)
        let variance_adjustment = variance_adjustment.max(0.01);

        // Standard error of SR estimate
        let std_error = (variance_adjustment / (t - 1.0)).sqrt();

        // Test statistic: Z = (SR - SR_benchmark) / SE[SR]
        let z_score = if std_error > 0.0 {
            (sharpe - benchmark_sr) / std_error
        } else {
            0.0
        };

        // PSR = Φ(Z) - cumulative distribution function of standard normal
        // Using error function: Φ(z) = 0.5 × (1 + erf(z / sqrt(2)))
        let psr = 0.5 * (1.0 + erf(z_score / std::f64::consts::SQRT_2));

        psr.clamp(0.0, 1.0)
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

/// Error function (erf) approximation for normal CDF calculation.
/// Uses Abramowitz and Stegun approximation (max error: 1.5e-7).
fn erf(x: f64) -> f64 {
    // Constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    // Save the sign of x
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // A&S formula 7.1.26
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
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
        println!("  Omega Ratio:     {:>12.2}", metrics.omega_ratio);
        println!();

        // Distribution Metrics
        println!("{}", "Distribution Metrics".bold().underline());
        println!("  Skewness:        {:>12.2}", metrics.skewness);
        println!("  Kurtosis:        {:>12.2}", metrics.kurtosis);
        println!("  Tail Ratio:      {:>12.2}", metrics.tail_ratio);
        println!();

        // Overfitting Detection
        println!("{}", "Overfitting Detection".bold().underline());
        println!("  Deflated Sharpe: {:>12.2}", metrics.deflated_sharpe_ratio);
        println!(
            "  Prob. Sharpe:    {:>12.2}  ({})",
            metrics.probabilistic_sharpe_ratio,
            Self::format_psr_confidence(metrics.probabilistic_sharpe_ratio)
        );
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

    /// Format PSR confidence level with interpretation.
    fn format_psr_confidence(psr: f64) -> String {
        let pct = psr * 100.0;
        if psr >= 0.95 {
            format!("{:.1}% - High confidence", pct).green().to_string()
        } else if psr >= 0.75 {
            format!("{:.1}% - Moderate confidence", pct)
                .yellow()
                .to_string()
        } else if psr >= 0.50 {
            format!("{:.1}% - Low confidence", pct).yellow().to_string()
        } else {
            format!("{:.1}% - Likely luck", pct).red().to_string()
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
        println!("  Omega Ratio:     {:>12.2}", metrics.omega_ratio);
        println!();

        // Distribution Metrics
        println!("{}", "Distribution Metrics".bold().underline());
        println!("  Skewness:        {:>12.2}", metrics.skewness);
        println!("  Kurtosis:        {:>12.2}", metrics.kurtosis);
        println!("  Tail Ratio:      {:>12.2}", metrics.tail_ratio);
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
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
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

    #[test]
    fn test_calculate_daily_returns_from_equity_curve() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap();

        // Create an equity curve with varying daily returns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 50000.0,
                positions_value: 50000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // +2%
                cash: 51000.0,
                positions_value: 51000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 99000.0, // -2.94%
                cash: 49500.0,
                positions_value: 49500.0,
                drawdown: 3000.0,
                drawdown_pct: 2.94,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 103000.0, // +4.04%
                cash: 51500.0,
                positions_value: 51500.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 5, 16, 0, 0).unwrap(),
                equity: 101000.0, // -1.94%
                cash: 50500.0,
                positions_value: 50500.0,
                drawdown: 2000.0,
                drawdown_pct: 1.94,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(101.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 101000.0,
            total_return_pct: 1.0,
            annual_return_pct: 1.0,
            trading_days: 5,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 100.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 2.94,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 0.34,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
        };

        // Test that daily returns are calculated from actual equity curve
        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        assert_eq!(returns.len(), 4, "Should have 4 daily returns from 5 days");

        // Verify actual returns are calculated correctly
        // Day 1 to Day 2: (102000 - 100000) / 100000 = 0.02 (+2%)
        assert!(
            (returns[0] - 0.02).abs() < 0.0001,
            "First return should be ~2%: {}",
            returns[0]
        );

        // Day 2 to Day 3: (99000 - 102000) / 102000 = -0.0294 (-2.94%)
        assert!(
            (returns[1] - (-0.0294117647)).abs() < 0.0001,
            "Second return should be ~-2.94%: {}",
            returns[1]
        );

        // Returns should NOT be all identical (unlike synthetic returns)
        assert!(
            returns[0] != returns[1],
            "Returns should vary, not be synthetic uniform values"
        );
    }

    #[test]
    fn test_calculate_daily_returns_meaningful_volatility() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap();

        // Create an equity curve with significant variation
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 105000.0, // +5%
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 95000.0, // -9.5%
                cash: 0.0,
                positions_value: 95000.0,
                drawdown: 10000.0,
                drawdown_pct: 9.5,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 110000.0, // +15.8%
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(110.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 110000.0,
            total_return_pct: 10.0,
            annual_return_pct: 10.0,
            trading_days: 4,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 1000.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 9.5,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 1.0,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
        };

        let metrics = PerformanceMetrics::from_result(&result);

        // Volatility should be meaningful (non-zero) with varying returns
        assert!(
            metrics.volatility_annual > 0.0,
            "Volatility should be positive with varying returns: {}",
            metrics.volatility_annual
        );

        // Volatility should be significant given the large daily swings
        assert!(
            metrics.volatility_annual > 50.0,
            "Volatility should be significant given 5%, -9.5%, +15.8% daily moves: {}",
            metrics.volatility_annual
        );
    }

    #[test]
    fn test_calculate_daily_returns_empty_equity_curve_fallback() {
        // Test fallback to synthetic returns when equity curve is empty
        let result = create_test_result(); // has empty equity_curve

        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        // Should fall back to synthetic returns (252 identical values)
        assert_eq!(returns.len(), 252);

        // All values should be identical in fallback mode
        let first = returns[0];
        assert!(
            returns.iter().all(|&r| (r - first).abs() < 1e-10),
            "Fallback returns should all be identical"
        );
    }

    #[test]
    fn test_calculate_daily_returns_intraday_aggregation() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 3, 0, 0, 0).unwrap();

        // Create equity curve with multiple intraday points
        // Should aggregate to daily by taking last value of each day
        let equity_curve = vec![
            // Day 1 - multiple points, last should be used
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 9, 30, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 12, 0, 0).unwrap(),
                equity: 100500.0,
                cash: 0.0,
                positions_value: 100500.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 101000.0, // This should be used for Day 1
                cash: 0.0,
                positions_value: 101000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            // Day 2 - multiple points
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 9, 30, 0).unwrap(),
                equity: 100000.0,
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 1000.0,
                drawdown_pct: 0.99,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // This should be used for Day 2
                cash: 0.0,
                positions_value: 102000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            // Day 3 - single point
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 103000.0, // This should be used for Day 3
                cash: 0.0,
                positions_value: 103000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(103.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 103000.0,
            total_return_pct: 3.0,
            annual_return_pct: 3.0,
            trading_days: 3,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 300.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 0.99,
            sharpe_ratio: 1.0,
            sortino_ratio: 1.5,
            calmar_ratio: 3.0,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
        };

        let returns = PerformanceMetrics::calculate_daily_returns(&result);

        // Should have 2 daily returns from 3 days
        assert_eq!(returns.len(), 2);

        // Day 1 to Day 2: (102000 - 101000) / 101000 = 0.0099 (~0.99%)
        assert!(
            (returns[0] - 0.00990099).abs() < 0.0001,
            "First return should be ~0.99%: {}",
            returns[0]
        );

        // Day 2 to Day 3: (103000 - 102000) / 102000 = 0.0098 (~0.98%)
        assert!(
            (returns[1] - 0.00980392).abs() < 0.0001,
            "Second return should be ~0.98%: {}",
            returns[1]
        );
    }

    #[test]
    fn test_drawdown_analysis_single_drawdown() {
        use crate::types::EquityPoint;

        // Equity curve: 100 -> 110 -> 90 -> 105
        // Peak at 110, trough at 90 (18.18% drawdown), then partial recovery
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 110000.0, // New peak
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 90000.0, // Trough - 18.18% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 20000.0,
                drawdown_pct: 18.18,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 105000.0, // Partial recovery, still in drawdown
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 5000.0,
                drawdown_pct: 4.55,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Max drawdown should be ~18.18% ((110000-90000)/110000 * 100)
        assert!(
            (analysis.max_drawdown_pct - 18.18).abs() < 0.1,
            "Max drawdown should be ~18.18%: {}",
            analysis.max_drawdown_pct
        );

        // Should have 1 ongoing drawdown period
        assert_eq!(analysis.periods.len(), 1, "Should have 1 drawdown period");

        // Period should not be recovered (end is None)
        assert!(
            analysis.periods[0].end.is_none(),
            "Drawdown period should not be recovered"
        );
    }

    #[test]
    fn test_drawdown_analysis_multiple_drawdowns() {
        use crate::types::EquityPoint;

        // Equity curve with two separate drawdown periods
        // 100 -> 120 -> 100 -> 130 -> 100 -> 150
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 5, 16, 0, 0).unwrap(),
                equity: 120000.0, // First peak
                cash: 0.0,
                positions_value: 120000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 10, 16, 0, 0).unwrap(),
                equity: 100000.0, // First drawdown ~16.67%
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 20000.0,
                drawdown_pct: 16.67,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 15, 16, 0, 0).unwrap(),
                equity: 130000.0, // New peak, first drawdown recovered
                cash: 0.0,
                positions_value: 130000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 20, 16, 0, 0).unwrap(),
                equity: 100000.0, // Second drawdown ~23.08%
                cash: 0.0,
                positions_value: 100000.0,
                drawdown: 30000.0,
                drawdown_pct: 23.08,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 25, 16, 0, 0).unwrap(),
                equity: 150000.0, // New peak, second drawdown recovered
                cash: 0.0,
                positions_value: 150000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Max drawdown should be ~23.08%
        assert!(
            (analysis.max_drawdown_pct - 23.08).abs() < 0.1,
            "Max drawdown should be ~23.08%: {}",
            analysis.max_drawdown_pct
        );

        // Should have 2 drawdown periods
        assert_eq!(analysis.periods.len(), 2, "Should have 2 drawdown periods");

        // Both periods should be recovered
        assert!(
            analysis.periods[0].end.is_some(),
            "First period should be recovered"
        );
        assert!(
            analysis.periods[1].end.is_some(),
            "Second period should be recovered"
        );

        // Average drawdown should be average of ~16.67% and ~23.08%
        let expected_avg = (16.666666 + 23.076923) / 2.0;
        assert!(
            (analysis.avg_drawdown_pct - expected_avg).abs() < 0.1,
            "Avg drawdown should be ~{:.2}%: {}",
            expected_avg,
            analysis.avg_drawdown_pct
        );
    }

    #[test]
    fn test_drawdown_analysis_ulcer_index() {
        use crate::types::EquityPoint;

        // Simple case: constant 10% drawdown
        // Ulcer Index = sqrt(mean(10^2)) = sqrt(100) = 10
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 90000.0, // 10% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 10000.0,
                drawdown_pct: 10.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 90000.0, // Still 10% drawdown
                cash: 0.0,
                positions_value: 90000.0,
                drawdown: 10000.0,
                drawdown_pct: 10.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Ulcer index calculation:
        // Point 0: drawdown = 0%
        // Point 1: drawdown = 10%
        // Point 2: drawdown = 10%
        // Ulcer = sqrt((0^2 + 10^2 + 10^2) / 3) = sqrt(200/3) = 8.165
        assert!(
            (analysis.ulcer_index - 8.165).abs() < 0.1,
            "Ulcer index should be ~8.165: {}",
            analysis.ulcer_index
        );
    }

    #[test]
    fn test_drawdown_analysis_no_drawdown() {
        use crate::types::EquityPoint;

        // Steadily increasing equity - no drawdowns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 110000.0,
                cash: 0.0,
                positions_value: 110000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 120000.0,
                cash: 0.0,
                positions_value: 120000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        assert!(
            analysis.max_drawdown_pct.abs() < 0.0001,
            "Max drawdown should be 0%"
        );
        assert!(
            analysis.periods.is_empty(),
            "Should have no drawdown periods"
        );
        assert!(
            analysis.ulcer_index.abs() < 0.0001,
            "Ulcer index should be 0, got: {}",
            analysis.ulcer_index
        );
    }

    #[test]
    fn test_drawdown_analysis_empty_curve() {
        let analysis = DrawdownAnalysis::from_equity_curve(&[]);

        assert_eq!(analysis.max_drawdown_pct, 0.0);
        assert_eq!(analysis.avg_drawdown_pct, 0.0);
        assert_eq!(analysis.ulcer_index, 0.0);
        assert!(analysis.periods.is_empty());
    }

    #[test]
    fn test_drawdown_analysis_serialization() {
        use crate::types::EquityPoint;

        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 95000.0,
                cash: 0.0,
                positions_value: 95000.0,
                drawdown: 5000.0,
                drawdown_pct: 5.0,
            },
        ];

        let analysis = DrawdownAnalysis::from_equity_curve(&equity_curve);

        // Test JSON serialization
        let json = serde_json::to_string(&analysis).unwrap();
        assert!(json.contains("max_drawdown_pct"));
        assert!(json.contains("ulcer_index"));

        // Test deserialization
        let deserialized: DrawdownAnalysis = serde_json::from_str(&json).unwrap();
        assert!((deserialized.max_drawdown_pct - analysis.max_drawdown_pct).abs() < 0.0001);
    }

    #[test]
    fn test_skewness_symmetric() {
        // Symmetric distribution should have near-zero skewness
        let returns = vec![-0.02, -0.01, 0.0, 0.01, 0.02];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew.abs() < 0.1,
            "Symmetric distribution should have near-zero skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_positive() {
        // Right-skewed: more extreme positive values
        let returns = vec![-0.01, -0.01, 0.0, 0.0, 0.05, 0.10];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew > 0.0,
            "Right-skewed distribution should have positive skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_negative() {
        // Left-skewed: more extreme negative values
        let returns = vec![-0.10, -0.05, 0.0, 0.0, 0.01, 0.01];
        let skew = PerformanceMetrics::skewness(&returns);
        assert!(
            skew < 0.0,
            "Left-skewed distribution should have negative skewness: {}",
            skew
        );
    }

    #[test]
    fn test_skewness_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02];
        let skew = PerformanceMetrics::skewness(&returns);
        assert_eq!(skew, 0.0, "Should return 0.0 for insufficient data");
    }

    #[test]
    fn test_kurtosis_normal() {
        // Approximately normal distribution should have near-zero excess kurtosis
        let returns = vec![-0.02, -0.01, 0.0, 0.0, 0.01, 0.02];
        let kurt = PerformanceMetrics::kurtosis(&returns);
        // For small samples, kurtosis can vary, so we just check it's calculated
        assert!(kurt.is_finite(), "Kurtosis should be finite: {}", kurt);
    }

    #[test]
    fn test_kurtosis_heavy_tails() {
        // Heavy tails: extreme values should produce positive excess kurtosis
        // Use more data points with clear heavy tails
        let mut returns = vec![0.0; 100];
        // Add extreme outliers in the tails
        returns[0] = -1.0;
        returns[1] = -0.8;
        returns[98] = 0.8;
        returns[99] = 1.0;
        let kurt = PerformanceMetrics::kurtosis(&returns);
        assert!(
            kurt > 0.0,
            "Heavy-tailed distribution should have positive excess kurtosis: {}",
            kurt
        );
    }

    #[test]
    fn test_kurtosis_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02, 0.03];
        let kurt = PerformanceMetrics::kurtosis(&returns);
        assert_eq!(kurt, 0.0, "Should return 0.0 for insufficient data");
    }

    #[test]
    fn test_tail_ratio_symmetric() {
        // Symmetric distribution should have tail ratio near 1.0
        let returns = vec![
            -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
        ];
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            (tail - 1.0).abs() < 0.5,
            "Symmetric distribution should have tail ratio near 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_positive_bias() {
        // Larger positive tail
        let mut returns = vec![0.0; 20];
        returns[0] = -0.02; // 5th percentile
        returns[19] = 0.10; // 95th percentile
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            tail > 1.0,
            "Positive bias should have tail ratio > 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_negative_bias() {
        // Larger negative tail
        // Create distribution with larger losses than gains
        let returns = vec![
            -0.10, -0.08, -0.06, -0.04, -0.02, // Bottom 25% - losses
            -0.01, -0.01, 0.0, 0.0, 0.0, // Middle 25%
            0.0, 0.0, 0.0, 0.01, 0.01, // Middle 25%
            0.01, 0.01, 0.02, 0.02, 0.02, // Top 25% - smaller gains
        ];
        // p05 (5th percentile) should be around -0.10 to -0.08
        // p95 (95th percentile) should be around 0.02
        // tail_ratio = 0.02 / 0.10 = 0.2 < 1.0
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert!(
            tail < 1.0,
            "Negative bias should have tail ratio < 1.0: {}",
            tail
        );
    }

    #[test]
    fn test_tail_ratio_insufficient_data() {
        // Too few data points
        let returns = vec![0.01, 0.02, 0.03];
        let tail = PerformanceMetrics::tail_ratio(&returns);
        assert_eq!(tail, 1.0, "Should return 1.0 for insufficient data");
    }

    #[test]
    fn test_distribution_metrics_integration() {
        use crate::types::EquityPoint;

        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap();

        // Create equity curve with varying returns
        let equity_curve = vec![
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 1, 16, 0, 0).unwrap(),
                equity: 100000.0,
                cash: 100000.0,
                positions_value: 0.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 2, 16, 0, 0).unwrap(),
                equity: 102000.0, // +2%
                cash: 0.0,
                positions_value: 102000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 3, 16, 0, 0).unwrap(),
                equity: 98000.0, // -3.9%
                cash: 0.0,
                positions_value: 98000.0,
                drawdown: 4000.0,
                drawdown_pct: 3.9,
            },
            EquityPoint {
                timestamp: Utc.with_ymd_and_hms(2024, 1, 4, 16, 0, 0).unwrap(),
                equity: 105000.0, // +7.1%
                cash: 0.0,
                positions_value: 105000.0,
                drawdown: 0.0,
                drawdown_pct: 0.0,
            },
        ];

        let mut trade = Trade::open("TEST", Side::Buy, 100.0, 100.0, start, 1.0, 0.0);
        trade.close(105.0, end, 1.0);

        let result = BacktestResult {
            strategy_name: "TestStrategy".to_string(),
            symbols: vec!["TEST".to_string()],
            config: BacktestConfig::default(),
            initial_capital: 100000.0,
            final_equity: 105000.0,
            total_return_pct: 5.0,
            annual_return_pct: 5.0,
            trading_days: 4,
            total_trades: 1,
            winning_trades: 1,
            losing_trades: 0,
            win_rate: 100.0,
            avg_win: 500.0,
            avg_loss: 0.0,
            profit_factor: f64::INFINITY,
            max_drawdown_pct: 3.9,
            sharpe_ratio: 0.5,
            sortino_ratio: 0.7,
            calmar_ratio: 1.28,
            trades: vec![trade],
            equity_curve,
            start_time: start,
            end_time: end,
            experiment_id: uuid::Uuid::new_v4(),
            git_info: None,
            config_hash: String::new(),
            data_checksums: std::collections::HashMap::new(),
        };

        let metrics = PerformanceMetrics::from_result(&result);

        // All distribution metrics should be calculated
        assert!(
            metrics.skewness.is_finite(),
            "Skewness should be calculated"
        );
        assert!(
            metrics.kurtosis.is_finite(),
            "Kurtosis should be calculated"
        );
        assert!(metrics.tail_ratio > 0.0, "Tail ratio should be positive");
    }

    #[test]
    fn test_deflated_sharpe_ratio_no_trials() {
        // With n_trials = 1 (no multiple testing), DSR should equal SR
        let sharpe = 1.5;
        let n_trials = 1;
        let n_observations = 252;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert_eq!(dsr, sharpe, "DSR should equal SR when n_trials = 1");
    }

    #[test]
    fn test_deflated_sharpe_ratio_multiple_trials() {
        // With multiple trials, DSR should be lower than SR
        let sharpe = 2.0;
        let n_trials = 100; // Tested 100 parameter combinations
        let n_observations = 252;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert!(
            dsr < sharpe,
            "DSR should be less than SR with multiple trials"
        );
        assert!(dsr.is_finite(), "DSR should be finite");

        // For 100 trials, expected max SR ≈ sqrt(2*ln(100)) ≈ 3.03
        // So a SR of 2.0 should deflate to a negative value
        assert!(
            dsr < 0.0,
            "DSR should be negative for SR=2.0 with 100 trials"
        );
    }

    #[test]
    fn test_deflated_sharpe_ratio_high_sharpe() {
        // Very high Sharpe ratio should survive deflation
        let sharpe = 5.0;
        let n_trials = 10;
        let n_observations = 500;

        let dsr = PerformanceMetrics::deflated_sharpe_ratio(sharpe, n_trials, n_observations);

        assert!(dsr.is_finite(), "DSR should be finite");
        // With only 10 trials, a SR of 5.0 is still impressive
        assert!(dsr > 2.0, "High SR should remain positive after deflation");
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_zero_sharpe() {
        // SR = 0 should give PSR ≈ 0.5 (50% confidence)
        let sharpe = 0.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!(psr >= 0.0 && psr <= 1.0, "PSR should be between 0 and 1");
        assert!(
            (psr - 0.5).abs() < 0.05,
            "PSR should be close to 0.5 for SR=0: {}",
            psr
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_positive_sharpe() {
        // High positive SR should give high PSR (>95%)
        let sharpe = 3.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!(psr >= 0.0 && psr <= 1.0, "PSR should be between 0 and 1");
        assert!(psr > 0.95, "PSR should be > 0.95 for high SR: {}", psr);
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_negative_sharpe() {
        // Negative SR should give low PSR (<5%)
        let sharpe = -2.0;
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 252;
        let benchmark_sr = 0.0;

        let psr = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            benchmark_sr,
        );

        assert!(psr >= 0.0 && psr <= 1.0, "PSR should be between 0 and 1");
        assert!(psr < 0.05, "PSR should be < 0.05 for negative SR: {}", psr);
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_with_skewness() {
        // Negative skewness should reduce PSR (use fewer observations to see effect)
        let sharpe = 0.8; // Lower Sharpe to avoid saturation
        let n_observations = 30; // Fewer observations to see skewness effect
        let benchmark_sr = 0.0;

        // PSR with zero skewness
        let psr_zero_skew = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            0.0,
            n_observations,
            benchmark_sr,
        );

        // PSR with negative skewness (bad tail behavior)
        let psr_neg_skew = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            -1.5,
            0.0,
            n_observations,
            benchmark_sr,
        );

        assert!(
            psr_zero_skew > psr_neg_skew,
            "Negative skewness should reduce PSR: {} vs {}",
            psr_zero_skew,
            psr_neg_skew
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_with_kurtosis() {
        // High kurtosis (fat tails) should reduce PSR (use fewer observations to see effect)
        let sharpe = 0.8; // Lower Sharpe to avoid saturation
        let n_observations = 30; // Fewer observations to see kurtosis effect
        let benchmark_sr = 0.0;

        // PSR with normal kurtosis
        let psr_normal_kurt = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            0.0,
            n_observations,
            benchmark_sr,
        );

        // PSR with high excess kurtosis (fat tails - more extreme events)
        let psr_high_kurt = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            0.0,
            8.0,
            n_observations,
            benchmark_sr,
        );

        assert!(
            psr_normal_kurt > psr_high_kurt,
            "High kurtosis should reduce PSR: {} vs {}",
            psr_normal_kurt,
            psr_high_kurt
        );
    }

    #[test]
    fn test_probabilistic_sharpe_ratio_benchmark_comparison() {
        // PSR should measure probability of exceeding benchmark
        let sharpe = 0.8; // Use moderate SR to avoid saturation
        let skewness = 0.0;
        let kurtosis = 0.0;
        let n_observations = 50; // Fewer observations to see benchmark effect

        // PSR vs zero benchmark
        let psr_vs_zero = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            0.0,
        );

        // PSR vs higher benchmark (harder to beat)
        let psr_vs_high = PerformanceMetrics::probabilistic_sharpe_ratio(
            sharpe,
            skewness,
            kurtosis,
            n_observations,
            0.5,
        );

        assert!(
            psr_vs_zero > psr_vs_high,
            "PSR should be lower vs higher benchmark: {} vs {}",
            psr_vs_zero,
            psr_vs_high
        );
    }

    #[test]
    fn test_overfitting_metrics_in_backtest_result() {
        // Ensure DSR and PSR are calculated in actual backtest results
        let result = create_test_result();
        let metrics = PerformanceMetrics::from_result(&result);

        assert!(
            metrics.deflated_sharpe_ratio.is_finite(),
            "Deflated Sharpe Ratio should be calculated"
        );
        assert!(
            metrics.probabilistic_sharpe_ratio >= 0.0 && metrics.probabilistic_sharpe_ratio <= 1.0,
            "PSR should be between 0 and 1: {}",
            metrics.probabilistic_sharpe_ratio
        );

        // With a decent SR of 1.5 and no multiple testing (n_trials=1),
        // DSR should equal SR
        assert_eq!(
            metrics.deflated_sharpe_ratio, result.sharpe_ratio,
            "DSR should equal SR when n_trials=1"
        );

        // With SR=1.5, PSR should be quite high
        assert!(
            metrics.probabilistic_sharpe_ratio > 0.80,
            "PSR should be > 0.80 for SR=1.5: {}",
            metrics.probabilistic_sharpe_ratio
        );
    }
}
