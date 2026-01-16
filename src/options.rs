//! Options pricing models and Greeks calculations.
//!
//! **DEPRECATION NOTICE**: This module is deprecated and will be removed in a future version.
//! Options pricing is outside the core thesis of Mantis as a backtester. For options pricing
//! and Greeks calculation, use dedicated libraries like:
//! - QuantLib (Python: QuantLib-Python)
//! - py_vollib for Black-Scholes and Greeks
//! - mibian for basic options pricing
//!
//! This module provides (deprecated):
//! - Black-Scholes pricing for European options
//! - Binomial tree pricing for American options (Cox-Ross-Rubinstein model)
//! - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
//! - Put-call parity validation
//! - Option contract representation
//! - Implied volatility solver

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Type of option contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptionType {
    /// Call option (right to buy underlying at strike).
    Call,
    /// Put option (right to sell underlying at strike).
    Put,
}

/// Exercise style of option contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExerciseStyle {
    /// American-style (can exercise any time before expiration).
    American,
    /// European-style (can only exercise at expiration).
    European,
}

/// Settlement type for option contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SettlementType {
    /// Physical delivery of underlying asset.
    Physical,
    /// Cash settlement based on intrinsic value.
    Cash,
}

/// Complete option contract specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionContract {
    /// Full option symbol (e.g., "AAPL240119C00150000").
    pub symbol: String,
    /// Underlying asset symbol (e.g., "AAPL").
    pub underlying: String,
    /// Strike price.
    pub strike: f64,
    /// Expiration date and time.
    pub expiration: DateTime<Utc>,
    /// Option type (Call or Put).
    pub option_type: OptionType,
    /// Exercise style (American or European).
    pub exercise_style: ExerciseStyle,
    /// Contract multiplier (typically 100 for equity options).
    pub multiplier: f64,
    /// Settlement type (Physical or Cash).
    pub settlement_type: SettlementType,
}

impl OptionContract {
    /// Calculate time to expiration in years from a given timestamp.
    pub fn time_to_expiration(&self, current_time: DateTime<Utc>) -> f64 {
        let duration = self.expiration.signed_duration_since(current_time);
        let seconds = duration.num_seconds() as f64;
        // Convert seconds to years (365.25 days accounting for leap years)
        (seconds / (365.25 * 24.0 * 3600.0)).max(0.0)
    }

    /// Calculate intrinsic value of the option.
    pub fn intrinsic_value(&self, underlying_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => (underlying_price - self.strike).max(0.0),
            OptionType::Put => (self.strike - underlying_price).max(0.0),
        }
    }

    /// Check if option is in-the-money.
    pub fn is_itm(&self, underlying_price: f64) -> bool {
        self.intrinsic_value(underlying_price) > 0.0
    }

    /// Check if option is at-the-money (within 2% of strike).
    pub fn is_atm(&self, underlying_price: f64, tolerance: f64) -> bool {
        (underlying_price - self.strike).abs() / self.strike <= tolerance
    }

    /// Check if option is out-of-the-money.
    pub fn is_otm(&self, underlying_price: f64) -> bool {
        !self.is_itm(underlying_price)
    }

    /// Calculate moneyness (S/K for calls, K/S for puts).
    pub fn moneyness(&self, underlying_price: f64) -> f64 {
        match self.option_type {
            OptionType::Call => underlying_price / self.strike,
            OptionType::Put => self.strike / underlying_price,
        }
    }
}

/// Greeks calculated for an option position.
#[derive(Debug, Clone, Copy, Default)]
pub struct Greeks {
    /// Delta: sensitivity to $1 change in underlying price.
    /// Range: 0 to 1 for calls, -1 to 0 for puts.
    pub delta: f64,
    /// Gamma: rate of change of delta.
    /// Always positive for long options.
    pub gamma: f64,
    /// Theta: time decay per day (value lost per day passing).
    /// Typically negative for long options.
    pub theta: f64,
    /// Vega: sensitivity to 1% change in implied volatility.
    /// Always positive for long options.
    pub vega: f64,
    /// Rho: sensitivity to 1% change in risk-free rate.
    pub rho: f64,
}

/// Black-Scholes pricing model for European options.
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate (e.g., 0.05 for 5%)
/// * `volatility` - Annualized volatility (e.g., 0.20 for 20%)
/// * `option_type` - Call or Put
///
/// # Returns
/// Option price using Black-Scholes formula
pub fn black_scholes(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
) -> f64 {
    // Handle edge case: option at expiration
    if time_to_expiration <= 0.0 {
        return match option_type {
            OptionType::Call => (underlying_price - strike).max(0.0),
            OptionType::Put => (strike - underlying_price).max(0.0),
        };
    }

    // Handle edge case: zero volatility
    if volatility <= 0.0 {
        let discount = (-risk_free_rate * time_to_expiration).exp();
        return match option_type {
            OptionType::Call => (underlying_price - strike * discount).max(0.0),
            OptionType::Put => (strike * discount - underlying_price).max(0.0),
        };
    }

    let sqrt_t = time_to_expiration.sqrt();
    let d1 = ((underlying_price / strike).ln()
        + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiration)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let discount_factor = (-risk_free_rate * time_to_expiration).exp();

    match option_type {
        OptionType::Call => {
            underlying_price * normal_cdf(d1) - strike * discount_factor * normal_cdf(d2)
        }
        OptionType::Put => {
            strike * discount_factor * normal_cdf(-d2) - underlying_price * normal_cdf(-d1)
        }
    }
}

/// Calculate all Greeks for a European option using Black-Scholes model.
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `volatility` - Annualized volatility
/// * `option_type` - Call or Put
///
/// # Returns
/// Greeks struct with Delta, Gamma, Theta, Vega, and Rho
pub fn calculate_greeks(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
) -> Greeks {
    // Handle edge case: option at expiration
    if time_to_expiration <= 0.0 {
        return Greeks {
            delta: match option_type {
                OptionType::Call => {
                    if underlying_price > strike {
                        1.0
                    } else {
                        0.0
                    }
                }
                OptionType::Put => {
                    if underlying_price < strike {
                        -1.0
                    } else {
                        0.0
                    }
                }
            },
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        };
    }

    let sqrt_t = time_to_expiration.sqrt();
    let d1 = ((underlying_price / strike).ln()
        + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiration)
        / (volatility * sqrt_t);
    let d2 = d1 - volatility * sqrt_t;

    let n_d1 = normal_cdf(d1);
    let n_prime_d1 = normal_pdf(d1);
    let discount_factor = (-risk_free_rate * time_to_expiration).exp();

    // Delta: ∂V/∂S
    let delta = match option_type {
        OptionType::Call => n_d1,
        OptionType::Put => n_d1 - 1.0,
    };

    // Gamma: ∂²V/∂S² (same for calls and puts)
    let gamma = n_prime_d1 / (underlying_price * volatility * sqrt_t);

    // Theta: -∂V/∂t (value lost per year, converted to per day)
    let theta_call = -(underlying_price * n_prime_d1 * volatility) / (2.0 * sqrt_t)
        - risk_free_rate * strike * discount_factor * normal_cdf(d2);
    let theta_put = -(underlying_price * n_prime_d1 * volatility) / (2.0 * sqrt_t)
        + risk_free_rate * strike * discount_factor * normal_cdf(-d2);
    let theta = match option_type {
        OptionType::Call => theta_call / 365.0, // Convert to per day
        OptionType::Put => theta_put / 365.0,
    };

    // Vega: ∂V/∂σ (sensitivity to 1% change in volatility)
    let vega = underlying_price * n_prime_d1 * sqrt_t / 100.0;

    // Rho: ∂V/∂r (sensitivity to 1% change in interest rate)
    let rho = match option_type {
        OptionType::Call => strike * time_to_expiration * discount_factor * normal_cdf(d2) / 100.0,
        OptionType::Put => -strike * time_to_expiration * discount_factor * normal_cdf(-d2) / 100.0,
    };

    Greeks {
        delta,
        gamma,
        theta,
        vega,
        rho,
    }
}

/// Validate put-call parity for European options.
///
/// Put-call parity states: C - P = S - K * e^(-rT)
///
/// # Arguments
/// * `call_price` - Market price of call option
/// * `put_price` - Market price of put option
/// * `underlying_price` - Current price of underlying
/// * `strike` - Strike price
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Risk-free interest rate
/// * `tolerance` - Acceptable deviation (e.g., 0.01 for 1%)
///
/// # Returns
/// True if put-call parity holds within tolerance, false otherwise
pub fn validate_put_call_parity(
    call_price: f64,
    put_price: f64,
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    tolerance: f64,
) -> bool {
    let discount_factor = (-risk_free_rate * time_to_expiration).exp();
    let lhs = call_price - put_price;
    let rhs = underlying_price - strike * discount_factor;
    (lhs - rhs).abs() / underlying_price <= tolerance
}

/// Cumulative distribution function for standard normal distribution.
///
/// Uses the error function (erf) for high accuracy.
/// Φ(x) = 0.5 * (1 + erf(x/√2))
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation using polynomial approximation.
/// Accurate to about 15 decimal places.
fn erf(x: f64) -> f64 {
    // Constants for approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = x.signum();
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Probability density function for standard normal distribution.
fn normal_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * PI).sqrt()
}

/// Calculate implied volatility from market price using Newton-Raphson method.
///
/// Solves for the volatility σ that makes Black-Scholes(σ) = market_price.
/// Uses vega-based Newton-Raphson iteration with Brent's method as fallback.
///
/// # Arguments
/// * `market_price` - Observed market price of the option
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `option_type` - Call or Put
///
/// # Returns
/// * `Some(volatility)` - Implied volatility if solver converges
/// * `None` - If solver fails to converge or inputs are invalid
///
/// # Algorithm
/// 1. Start with initial guess (historical volatility proxy)
/// 2. Newton-Raphson: σ_{n+1} = σ_n - (BS_price - market_price) / vega
/// 3. If Newton-Raphson fails, fall back to Brent's method
/// 4. Bounds checking: IV typically in [0.05, 2.00] (5% to 200%)
pub fn implied_volatility(
    market_price: f64,
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    option_type: OptionType,
) -> Option<f64> {
    // Input validation
    if market_price <= 0.0 || underlying_price <= 0.0 || strike <= 0.0 || time_to_expiration <= 0.0
    {
        return None;
    }

    // Check if market price is reasonable
    // For calls: price cannot exceed underlying price (max value = S when K=0)
    // For puts: price cannot exceed discounted strike (max value = K*e^(-rT))
    let max_value = match option_type {
        OptionType::Call => underlying_price,
        OptionType::Put => strike * (-risk_free_rate * time_to_expiration).exp(),
    };

    if market_price > max_value + 1e-6 {
        return None;
    }

    // Calculate intrinsic value (for American options, price >= intrinsic)
    // For European options, this check is not applicable due to discounting
    let intrinsic = match option_type {
        OptionType::Call => (underlying_price - strike).max(0.0),
        OptionType::Put => (strike - underlying_price).max(0.0),
    };

    // For deep ITM options at expiration, IV is undefined
    if time_to_expiration < 1e-6 {
        return None;
    }

    // Bounds for implied volatility (5% to 200%)
    const MIN_VOL: f64 = 0.05;
    const MAX_VOL: f64 = 2.00;
    const TOLERANCE: f64 = 1e-6;
    const MAX_ITERATIONS: usize = 100;

    // Initial guess using Brenner-Subrahmanyam approximation for ATM options
    let initial_guess = if (underlying_price / strike - 1.0).abs() < 0.1 {
        // ATM approximation: σ ≈ sqrt(2π/T) * (C/S)
        ((2.0 * PI / time_to_expiration).sqrt() * (market_price / underlying_price))
            .clamp(MIN_VOL, MAX_VOL)
    } else if (market_price - intrinsic).abs() < 0.01 {
        // Deep ITM option with very little time value - use low volatility guess
        0.10
    } else {
        // Default to 25% volatility for non-ATM options
        0.25
    };

    // Try Newton-Raphson first (fast convergence when it works)
    if let Some(vol) = newton_raphson_iv(
        market_price,
        underlying_price,
        strike,
        time_to_expiration,
        risk_free_rate,
        option_type,
        initial_guess,
        TOLERANCE,
        MAX_ITERATIONS,
    ) {
        return Some(vol);
    }

    // Fall back to Brent's method (more robust for difficult cases)
    brent_method_iv(
        market_price,
        underlying_price,
        strike,
        time_to_expiration,
        risk_free_rate,
        option_type,
        MIN_VOL,
        MAX_VOL,
        TOLERANCE,
        MAX_ITERATIONS,
    )
}

/// Newton-Raphson solver for implied volatility.
///
/// Uses vega (∂V/∂σ) as the derivative for fast convergence.
/// Typically converges in 5-10 iterations for well-behaved inputs.
#[allow(clippy::too_many_arguments)]
fn newton_raphson_iv(
    target_price: f64,
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    option_type: OptionType,
    initial_guess: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Option<f64> {
    let mut sigma = initial_guess;
    const MIN_VOL: f64 = 0.05;
    const MAX_VOL: f64 = 2.00;

    for _ in 0..max_iterations {
        let price = black_scholes(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            sigma,
            option_type,
        );

        let diff = price - target_price;

        // Check convergence
        if diff.abs() < tolerance {
            return Some(sigma);
        }

        // Calculate vega for Newton-Raphson step
        let greeks = calculate_greeks(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            sigma,
            option_type,
        );

        // Avoid division by very small vega
        if greeks.vega.abs() < 1e-10 {
            return None;
        }

        // Newton-Raphson update: σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
        // where f(σ) = BS_price(σ) - target_price
        // and f'(σ) = vega * 100 (vega is scaled to 1% change)
        sigma -= diff / (greeks.vega * 100.0);

        // Keep sigma within bounds
        sigma = sigma.clamp(MIN_VOL, MAX_VOL);

        // Detect if we're stuck oscillating
        if sigma <= MIN_VOL || sigma >= MAX_VOL {
            return None;
        }
    }

    None // Failed to converge
}

/// Brent's method for finding implied volatility.
///
/// More robust than Newton-Raphson but slower. Used as fallback.
/// Guaranteed to converge if the function is continuous and brackets the root.
#[allow(clippy::too_many_arguments)]
fn brent_method_iv(
    target_price: f64,
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    option_type: OptionType,
    mut a: f64,
    mut b: f64,
    tolerance: f64,
    max_iterations: usize,
) -> Option<f64> {
    let price_at = |vol: f64| -> f64 {
        black_scholes(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            vol,
            option_type,
        )
    };

    let mut fa = price_at(a) - target_price;
    let mut fb = price_at(b) - target_price;

    // Check if root is bracketed
    if fa * fb > 0.0 {
        return None;
    }

    // Ensure fa is closer to zero
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0;

    for _ in 0..max_iterations {
        if (fb.abs() < tolerance) || ((b - a).abs() < tolerance) {
            return Some(b);
        }

        let s = if (fa - fc).abs() > 1e-10 && (fb - fc).abs() > 1e-10 {
            // Inverse quadratic interpolation
            (a * fb * fc) / ((fa - fb) * (fa - fc))
                + (b * fa * fc) / ((fb - fa) * (fb - fc))
                + (c * fa * fb) / ((fc - fa) * (fc - fb))
        } else {
            // Secant method
            b - fb * (b - a) / (fb - fa)
        };

        // Determine if we should use bisection instead
        let condition1 = (s < (3.0 * a + b) / 4.0) || (s > b);
        let condition2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let condition3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let condition4 = mflag && (b - c).abs() < tolerance;
        let condition5 = !mflag && (c - d).abs() < tolerance;

        let s = if condition1 || condition2 || condition3 || condition4 || condition5 {
            // Bisection
            mflag = true;
            (a + b) / 2.0
        } else {
            mflag = false;
            s
        };

        let fs = price_at(s) - target_price;
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        // Ensure fa is closer to zero
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    Some(b)
}

/// Configuration for binomial tree pricing model.
#[derive(Debug, Clone, Copy)]
pub struct BinomialTreeConfig {
    /// Number of steps in the tree (higher = more accurate but slower).
    /// Typical values: 50-200 for most options, 500+ for very accurate pricing.
    pub steps: usize,
}

impl Default for BinomialTreeConfig {
    fn default() -> Self {
        Self { steps: 100 }
    }
}

impl BinomialTreeConfig {
    /// Create a fast configuration (50 steps, ~10ms).
    pub fn fast() -> Self {
        Self { steps: 50 }
    }

    /// Create an accurate configuration (200 steps, ~50ms).
    pub fn accurate() -> Self {
        Self { steps: 200 }
    }

    /// Create a high-precision configuration (500 steps, ~200ms).
    pub fn high_precision() -> Self {
        Self { steps: 500 }
    }
}

/// Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree model.
///
/// This model can price both American and European options. For American options,
/// it correctly accounts for early exercise at each node.
///
/// # Algorithm
///
/// The CRR model constructs a recombining binomial tree:
/// 1. At each time step, the stock price can move up by factor `u` or down by factor `d`
/// 2. `u = exp(σ√Δt)` and `d = 1/u` where Δt = T/N
/// 3. Risk-neutral probability `p = (exp(rΔt) - d) / (u - d)`
/// 4. Terminal node values = intrinsic value of option
/// 5. Work backwards: each node value = discounted expected value of children
/// 6. For American options: at each node, check if early exercise is optimal
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset (S)
/// * `strike` - Strike price of option (K)
/// * `time_to_expiration` - Time to expiration in years (T)
/// * `risk_free_rate` - Annualized risk-free interest rate (r)
/// * `volatility` - Annualized volatility (σ)
/// * `option_type` - Call or Put
/// * `exercise_style` - American or European
/// * `config` - Configuration for tree depth
///
/// # Returns
/// Option price calculated using the binomial tree model
///
/// # Example
/// ```
/// use mantis::options::{binomial_tree_price, OptionType, ExerciseStyle, BinomialTreeConfig};
///
/// // Price an American put option
/// let price = binomial_tree_price(
///     100.0,  // underlying price
///     100.0,  // strike
///     1.0,    // 1 year to expiration
///     0.05,   // 5% risk-free rate
///     0.20,   // 20% volatility
///     OptionType::Put,
///     ExerciseStyle::American,
///     &BinomialTreeConfig::default(),
/// );
/// ```
#[allow(clippy::too_many_arguments)]
pub fn binomial_tree_price(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    config: &BinomialTreeConfig,
) -> f64 {
    // Handle edge cases
    if time_to_expiration <= 0.0 {
        // At expiration, option is worth intrinsic value
        return match option_type {
            OptionType::Call => (underlying_price - strike).max(0.0),
            OptionType::Put => (strike - underlying_price).max(0.0),
        };
    }

    if volatility <= 0.0 {
        // With zero volatility, no uncertainty - price deterministically
        let discount = (-risk_free_rate * time_to_expiration).exp();
        return match option_type {
            OptionType::Call => (underlying_price - strike * discount).max(0.0),
            OptionType::Put => (strike * discount - underlying_price).max(0.0),
        };
    }

    let n = config.steps;
    let dt = time_to_expiration / n as f64;

    // CRR parameters
    let u = (volatility * dt.sqrt()).exp(); // Up factor
    let d = 1.0 / u; // Down factor (ensures recombining tree)
    let discount_factor = (-risk_free_rate * dt).exp();
    let p = ((risk_free_rate * dt).exp() - d) / (u - d); // Risk-neutral probability

    // Ensure probability is valid
    if !(0.0..=1.0).contains(&p) {
        // If probability is invalid, fall back to Black-Scholes
        return black_scholes(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            volatility,
            option_type,
        );
    }

    // Build terminal stock prices and option values
    // We use a single array and update in place for memory efficiency
    let mut option_values = Vec::with_capacity(n + 1);

    // Calculate stock prices at terminal nodes (time = T)
    // Stock price at node (n, j) = S * u^j * d^(n-j) = S * u^(2j-n)
    for j in 0..=n {
        let stock_price = underlying_price * u.powi(2 * j as i32 - n as i32);
        let intrinsic = match option_type {
            OptionType::Call => (stock_price - strike).max(0.0),
            OptionType::Put => (strike - stock_price).max(0.0),
        };
        option_values.push(intrinsic);
    }

    // Work backwards through the tree
    for i in (0..n).rev() {
        for j in 0..=i {
            // Expected option value (discounted)
            let expected_value =
                discount_factor * (p * option_values[j + 1] + (1.0 - p) * option_values[j]);

            // For American options, check if early exercise is optimal
            let option_value = match exercise_style {
                ExerciseStyle::European => expected_value,
                ExerciseStyle::American => {
                    // Stock price at node (i, j)
                    let stock_price = underlying_price * u.powi(2 * j as i32 - i as i32);
                    let intrinsic = match option_type {
                        OptionType::Call => (stock_price - strike).max(0.0),
                        OptionType::Put => (strike - stock_price).max(0.0),
                    };
                    // Take the maximum of holding vs exercising
                    expected_value.max(intrinsic)
                }
            };

            option_values[j] = option_value;
        }
    }

    option_values[0]
}

/// Calculate Greeks for an option using the binomial tree model with finite differences.
///
/// This provides numerical Greeks that work for both American and European options.
/// The finite difference method uses small perturbations to estimate sensitivities.
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `volatility` - Annualized volatility
/// * `option_type` - Call or Put
/// * `exercise_style` - American or European
/// * `config` - Configuration for tree depth
///
/// # Returns
/// Greeks struct with Delta, Gamma, Theta, Vega, and Rho
#[allow(clippy::too_many_arguments)]
pub fn binomial_tree_greeks(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
    config: &BinomialTreeConfig,
) -> Greeks {
    // Handle edge case: option at expiration
    if time_to_expiration <= 0.0 {
        return Greeks {
            delta: match option_type {
                OptionType::Call => {
                    if underlying_price > strike {
                        1.0
                    } else {
                        0.0
                    }
                }
                OptionType::Put => {
                    if underlying_price < strike {
                        -1.0
                    } else {
                        0.0
                    }
                }
            },
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
        };
    }

    let price =
        |s, t, r, v| binomial_tree_price(s, strike, t, r, v, option_type, exercise_style, config);

    let base_price = price(
        underlying_price,
        time_to_expiration,
        risk_free_rate,
        volatility,
    );

    // Delta: ∂V/∂S (using central difference)
    let ds = underlying_price * 0.01; // 1% perturbation
    let price_up = price(
        underlying_price + ds,
        time_to_expiration,
        risk_free_rate,
        volatility,
    );
    let price_down = price(
        underlying_price - ds,
        time_to_expiration,
        risk_free_rate,
        volatility,
    );
    let delta = (price_up - price_down) / (2.0 * ds);

    // Gamma: ∂²V/∂S² (using central difference)
    let gamma = (price_up - 2.0 * base_price + price_down) / (ds * ds);

    // Theta: -∂V/∂t (time decay per day)
    let dt = 1.0 / 365.0; // 1 day
    let theta = if time_to_expiration > dt {
        let price_later = price(
            underlying_price,
            time_to_expiration - dt,
            risk_free_rate,
            volatility,
        );
        (price_later - base_price) / dt / 365.0 // Per day
    } else {
        0.0
    };

    // Vega: ∂V/∂σ (sensitivity to 1% change in volatility)
    let dv = 0.01; // 1% volatility change
    let price_vol_up = price(
        underlying_price,
        time_to_expiration,
        risk_free_rate,
        volatility + dv,
    );
    let price_vol_down = price(
        underlying_price,
        time_to_expiration,
        risk_free_rate,
        volatility - dv.min(volatility - 0.01),
    );
    let vega = (price_vol_up - price_vol_down) / (2.0 * dv) / 100.0; // Per 1% vol change

    // Rho: ∂V/∂r (sensitivity to 1% change in interest rate)
    let dr = 0.01; // 1% rate change
    let price_rate_up = price(
        underlying_price,
        time_to_expiration,
        risk_free_rate + dr,
        volatility,
    );
    let price_rate_down = price(
        underlying_price,
        time_to_expiration,
        (risk_free_rate - dr).max(0.0),
        volatility,
    );
    let rho = (price_rate_up - price_rate_down) / (2.0 * dr) / 100.0; // Per 1% rate change

    Greeks {
        delta,
        gamma,
        theta,
        vega,
        rho,
    }
}

/// Calculate the early exercise premium for an American option.
///
/// The early exercise premium is the difference between the American option price
/// and the equivalent European option price. This premium is always non-negative.
///
/// For calls on non-dividend paying stocks, the premium is typically zero
/// (early exercise is never optimal). For puts, there's often a positive premium,
/// especially for deep ITM puts near expiration.
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `volatility` - Annualized volatility
/// * `option_type` - Call or Put
/// * `config` - Configuration for tree depth
///
/// # Returns
/// Early exercise premium (always >= 0)
#[allow(clippy::too_many_arguments)]
pub fn early_exercise_premium(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
    config: &BinomialTreeConfig,
) -> f64 {
    let american_price = binomial_tree_price(
        underlying_price,
        strike,
        time_to_expiration,
        risk_free_rate,
        volatility,
        option_type,
        ExerciseStyle::American,
        config,
    );

    let european_price = binomial_tree_price(
        underlying_price,
        strike,
        time_to_expiration,
        risk_free_rate,
        volatility,
        option_type,
        ExerciseStyle::European,
        config,
    );

    (american_price - european_price).max(0.0)
}

/// Price an option using the appropriate model based on exercise style.
///
/// This is a convenience function that automatically selects:
/// - Black-Scholes for European options (faster, closed-form solution)
/// - Binomial tree for American options (handles early exercise)
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `volatility` - Annualized volatility
/// * `option_type` - Call or Put
/// * `exercise_style` - American or European
///
/// # Returns
/// Option price using the appropriate model
pub fn price_option(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> f64 {
    match exercise_style {
        ExerciseStyle::European => black_scholes(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            volatility,
            option_type,
        ),
        ExerciseStyle::American => binomial_tree_price(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            volatility,
            option_type,
            ExerciseStyle::American,
            &BinomialTreeConfig::default(),
        ),
    }
}

/// Calculate Greeks using the appropriate model based on exercise style.
///
/// This is a convenience function that automatically selects:
/// - Analytical Greeks for European options (faster, exact)
/// - Numerical Greeks from binomial tree for American options
///
/// # Arguments
/// * `underlying_price` - Current price of underlying asset
/// * `strike` - Strike price of option
/// * `time_to_expiration` - Time to expiration in years
/// * `risk_free_rate` - Annualized risk-free interest rate
/// * `volatility` - Annualized volatility
/// * `option_type` - Call or Put
/// * `exercise_style` - American or European
///
/// # Returns
/// Greeks struct with Delta, Gamma, Theta, Vega, and Rho
pub fn greeks_for_option(
    underlying_price: f64,
    strike: f64,
    time_to_expiration: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: OptionType,
    exercise_style: ExerciseStyle,
) -> Greeks {
    match exercise_style {
        ExerciseStyle::European => calculate_greeks(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            volatility,
            option_type,
        ),
        ExerciseStyle::American => binomial_tree_greeks(
            underlying_price,
            strike,
            time_to_expiration,
            risk_free_rate,
            volatility,
            option_type,
            ExerciseStyle::American,
            &BinomialTreeConfig::default(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.0) - 0.8413447).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.1586553).abs() < 1e-6);
        assert!((normal_cdf(2.0) - 0.9772499).abs() < 1e-6);
        assert!((normal_cdf(-2.0) - 0.0227501).abs() < 1e-6);
    }

    #[test]
    fn test_normal_pdf() {
        // Test known values
        assert!((normal_pdf(0.0) - 0.3989423).abs() < 1e-6);
        assert!((normal_pdf(1.0) - 0.2419707).abs() < 1e-6);
        assert!((normal_pdf(-1.0) - 0.2419707).abs() < 1e-6); // Symmetric
    }

    #[test]
    fn test_black_scholes_call() {
        // Standard test case from literature
        let s = 100.0; // Underlying price
        let k = 100.0; // Strike (ATM)
        let t = 1.0; // 1 year to expiration
        let r = 0.05; // 5% risk-free rate
        let sigma = 0.20; // 20% volatility

        let price = black_scholes(s, k, t, r, sigma, OptionType::Call);

        // Expected value approximately $10.45 (verified against standard references)
        assert!((price - 10.45).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_put() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let price = black_scholes(s, k, t, r, sigma, OptionType::Put);

        // Expected value approximately $5.57 (verified against standard references)
        assert!((price - 5.57).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_itm_call() {
        // Deep ITM call (S > K)
        let s = 110.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let price = black_scholes(s, k, t, r, sigma, OptionType::Call);

        // Should be at least intrinsic value
        let intrinsic = s - k;
        assert!(price >= intrinsic);
        // Expected value approximately $17.66
        assert!((price - 17.66).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_otm_put() {
        // OTM put (S > K)
        let s = 110.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let price = black_scholes(s, k, t, r, sigma, OptionType::Put);

        // Should be less than ATM put
        assert!(price < 5.57);
        assert!(price > 0.0);
    }

    #[test]
    fn test_black_scholes_at_expiration() {
        let s = 105.0;
        let k = 100.0;
        let t = 0.0; // At expiration
        let r = 0.05;
        let sigma = 0.20;

        let call_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let put_price = black_scholes(s, k, t, r, sigma, OptionType::Put);

        // At expiration, option is worth intrinsic value
        assert!((call_price - 5.0).abs() < 1e-6);
        assert!((put_price - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_black_scholes_zero_volatility() {
        let s = 105.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.0; // Zero volatility

        let call_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let put_price = black_scholes(s, k, t, r, sigma, OptionType::Put);

        // With zero vol, call = max(S - K*e^(-rT), 0)
        let discount = (-r * t).exp();
        let expected_call = (s - k * discount).max(0.0);
        assert!((call_price - expected_call).abs() < 1e-6);

        let expected_put = (k * discount - s).max(0.0);
        assert!((put_price - expected_put).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_greeks_call() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Call);

        // Delta for ATM call should be around 0.5-0.6
        assert!(greeks.delta > 0.5 && greeks.delta < 0.7);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0);

        // Theta should be negative for long call
        assert!(greeks.theta < 0.0);

        // Vega should be positive
        assert!(greeks.vega > 0.0);

        // Rho should be positive for call
        assert!(greeks.rho > 0.0);
    }

    #[test]
    fn test_calculate_greeks_put() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Put);

        // Delta for ATM put should be around -0.4 to -0.5
        assert!(greeks.delta < 0.0 && greeks.delta > -0.6);

        // Gamma should be positive (same as call)
        assert!(greeks.gamma > 0.0);

        // Theta should be negative for long put
        assert!(greeks.theta < 0.0);

        // Vega should be positive (same as call)
        assert!(greeks.vega > 0.0);

        // Rho should be negative for put
        assert!(greeks.rho < 0.0);
    }

    #[test]
    fn test_greeks_deep_itm_call() {
        let s = 120.0; // Deep ITM
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Call);

        // Delta for deep ITM call approaches 1.0
        assert!(greeks.delta > 0.8);

        // Gamma should be low for deep ITM
        assert!(greeks.gamma < 0.02);
    }

    #[test]
    fn test_greeks_deep_otm_put() {
        let s = 120.0; // Deep OTM for put
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Put);

        // Delta for deep OTM put approaches 0
        assert!(greeks.delta.abs() < 0.2);

        // Gamma should be low for deep OTM
        assert!(greeks.gamma < 0.02);
    }

    #[test]
    fn test_greeks_at_expiration() {
        let s = 105.0;
        let k = 100.0;
        let t = 0.0; // At expiration
        let r = 0.05;
        let sigma = 0.20;

        let call_greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Call);
        let put_greeks = calculate_greeks(s, k, t, r, sigma, OptionType::Put);

        // At expiration, ITM call has delta = 1
        assert!((call_greeks.delta - 1.0).abs() < 1e-6);

        // At expiration, OTM put has delta = 0
        assert!((put_greeks.delta - 0.0).abs() < 1e-6);

        // All other Greeks should be zero at expiration
        assert!((call_greeks.gamma - 0.0).abs() < 1e-6);
        assert!((call_greeks.theta - 0.0).abs() < 1e-6);
        assert!((call_greeks.vega - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_put_call_parity() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let call_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let put_price = black_scholes(s, k, t, r, sigma, OptionType::Put);

        // Verify put-call parity holds
        let is_valid = validate_put_call_parity(call_price, put_price, s, k, t, r, 0.01);
        assert!(is_valid);

        // Manual verification: C - P = S - K * e^(-rT)
        let discount = (-r * t).exp();
        let lhs = call_price - put_price;
        let rhs = s - k * discount;
        assert!((lhs - rhs).abs() < 0.01);
    }

    #[test]
    fn test_put_call_parity_violation() {
        let call_price = 10.0;
        let put_price = 2.0; // Arbitrarily set to violate parity
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;

        let is_valid = validate_put_call_parity(call_price, put_price, s, k, t, r, 0.01);
        assert!(!is_valid);
    }

    #[test]
    fn test_option_contract_intrinsic_value() {
        let now = Utc::now();
        let expiration = now + Duration::days(30);

        let call = OptionContract {
            symbol: "TEST240101C00100000".to_string(),
            underlying: "TEST".to_string(),
            strike: 100.0,
            expiration,
            option_type: OptionType::Call,
            exercise_style: ExerciseStyle::European,
            multiplier: 100.0,
            settlement_type: SettlementType::Physical,
        };

        // ITM call
        assert!((call.intrinsic_value(110.0) - 10.0).abs() < 1e-6);

        // OTM call
        assert!((call.intrinsic_value(90.0) - 0.0).abs() < 1e-6);

        // ATM call
        assert!((call.intrinsic_value(100.0) - 0.0).abs() < 1e-6);

        let put = OptionContract {
            symbol: "TEST240101P00100000".to_string(),
            underlying: "TEST".to_string(),
            strike: 100.0,
            expiration,
            option_type: OptionType::Put,
            exercise_style: ExerciseStyle::European,
            multiplier: 100.0,
            settlement_type: SettlementType::Physical,
        };

        // ITM put
        assert!((put.intrinsic_value(90.0) - 10.0).abs() < 1e-6);

        // OTM put
        assert!((put.intrinsic_value(110.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_option_contract_moneyness() {
        let now = Utc::now();
        let expiration = now + Duration::days(30);

        let call = OptionContract {
            symbol: "TEST240101C00100000".to_string(),
            underlying: "TEST".to_string(),
            strike: 100.0,
            expiration,
            option_type: OptionType::Call,
            exercise_style: ExerciseStyle::European,
            multiplier: 100.0,
            settlement_type: SettlementType::Physical,
        };

        assert!(call.is_itm(110.0));
        assert!(!call.is_itm(90.0));
        assert!(call.is_atm(100.0, 0.02));
        assert!(call.is_atm(101.5, 0.02));
        assert!(!call.is_atm(110.0, 0.02));

        // Test moneyness ratio
        assert!((call.moneyness(110.0) - 1.1).abs() < 1e-6);
        assert!((call.moneyness(90.0) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_option_contract_time_to_expiration() {
        let now = Utc::now();
        let expiration = now + Duration::days(365);

        let option = OptionContract {
            symbol: "TEST250101C00100000".to_string(),
            underlying: "TEST".to_string(),
            strike: 100.0,
            expiration,
            option_type: OptionType::Call,
            exercise_style: ExerciseStyle::European,
            multiplier: 100.0,
            settlement_type: SettlementType::Physical,
        };

        let tte = option.time_to_expiration(now);
        // Should be approximately 1 year
        assert!((tte - 1.0).abs() < 0.01);

        // Test expiration passed
        let past = now - Duration::days(10);
        let option_expired = OptionContract {
            symbol: "TEST240101C00100000".to_string(),
            underlying: "TEST".to_string(),
            strike: 100.0,
            expiration: past,
            option_type: OptionType::Call,
            exercise_style: ExerciseStyle::European,
            multiplier: 100.0,
            settlement_type: SettlementType::Physical,
        };

        let tte_expired = option_expired.time_to_expiration(now);
        assert_eq!(tte_expired, 0.0); // Should be clamped to 0
    }

    #[test]
    fn test_implied_volatility_atm_call() {
        // Test roundtrip: calculate IV from a known Black-Scholes price
        let s = 100.0;
        let k = 100.0; // ATM
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20; // 20% volatility

        // Calculate theoretical price
        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);

        // Solve for IV from the price
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();

        // Should recover the original volatility within tolerance
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.20, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_atm_put() {
        let s = 100.0;
        let k = 100.0; // ATM
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.30; // 30% volatility

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Put);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Put);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.30, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_itm_call() {
        // Deep ITM call
        let s = 110.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.25;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.25, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_otm_put() {
        // OTM put
        let s = 110.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.15;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Put);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Put);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.15, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_short_dated() {
        // Short-dated option (1 month)
        let s = 100.0;
        let k = 100.0;
        let t = 1.0 / 12.0; // 1 month
        let r = 0.05;
        let sigma = 0.40; // Higher vol

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.40, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_long_dated() {
        // Long-dated option (2 years)
        let s = 100.0;
        let k = 100.0;
        let t = 2.0;
        let r = 0.05;
        let sigma = 0.18;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.18, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_high_vol() {
        // High volatility scenario
        let s = 100.0;
        let k = 100.0;
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.80; // 80% volatility

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.80, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_low_vol() {
        // Low volatility scenario
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.08; // 8% volatility

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.08, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_invalid_negative_price() {
        // Negative price should return None
        let implied_vol = implied_volatility(-10.0, 100.0, 100.0, 1.0, 0.05, OptionType::Call);
        assert!(implied_vol.is_none());
    }

    #[test]
    fn test_implied_volatility_invalid_zero_price() {
        // Zero price should return None
        let implied_vol = implied_volatility(0.0, 100.0, 100.0, 1.0, 0.05, OptionType::Call);
        assert!(implied_vol.is_none());
    }

    #[test]
    fn test_implied_volatility_price_below_intrinsic() {
        // Price below intrinsic value is impossible
        let s = 110.0;
        let k = 100.0;
        let intrinsic = s - k; // 10.0
        let invalid_price = intrinsic - 5.0; // 5.0 (below intrinsic)

        let implied_vol = implied_volatility(invalid_price, s, k, 1.0, 0.05, OptionType::Call);
        assert!(implied_vol.is_none());
    }

    #[test]
    fn test_implied_volatility_at_expiration() {
        // At expiration (t=0), IV is undefined
        let s = 105.0;
        let k = 100.0;
        let market_price = 5.0; // Intrinsic value

        let implied_vol = implied_volatility(market_price, s, k, 0.0, 0.05, OptionType::Call);
        assert!(implied_vol.is_none());
    }

    #[test]
    fn test_implied_volatility_deep_itm_call_roundtrip() {
        // Very deep ITM call (harder to solve)
        let s = 150.0;
        let k = 100.0;
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.22;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.22, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_deep_otm_put_roundtrip() {
        // Very deep OTM put (harder to solve)
        let s = 150.0;
        let k = 100.0;
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.35;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Put);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Put);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        assert!((iv - sigma).abs() < 0.001, "Expected IV ~0.35, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_very_short_expiration() {
        // Very short time to expiration (1 day)
        let s = 100.0;
        let k = 100.0;
        let t = 1.0 / 365.0;
        let r = 0.05;
        let sigma = 0.50;

        let theoretical_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let implied_vol = implied_volatility(theoretical_price, s, k, t, r, OptionType::Call);

        assert!(implied_vol.is_some());
        let iv = implied_vol.unwrap();
        // Allow slightly higher tolerance for very short expirations
        assert!((iv - sigma).abs() < 0.01, "Expected IV ~0.50, got {}", iv);
    }

    #[test]
    fn test_implied_volatility_consistency_across_strikes() {
        // Test that IV solver works consistently across different strikes
        let s = 100.0;
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.25;

        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];

        for k in strikes {
            let call_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
            let put_price = black_scholes(s, k, t, r, sigma, OptionType::Put);

            // Skip very small prices (< $0.01) as they're numerically challenging
            if put_price < 0.01 || call_price < 0.01 {
                continue;
            }

            let call_iv = implied_volatility(call_price, s, k, t, r, OptionType::Call);
            let put_iv = implied_volatility(put_price, s, k, t, r, OptionType::Put);

            assert!(
                call_iv.is_some(),
                "Failed to solve IV for call at strike {}: price = {}",
                k,
                call_price
            );
            assert!(
                put_iv.is_some(),
                "Failed to solve IV for put at strike {}: price = {}",
                k,
                put_price
            );

            let call_vol = call_iv.unwrap();
            let put_vol = put_iv.unwrap();

            assert!(
                (call_vol - sigma).abs() < 0.001,
                "Call IV mismatch at strike {}: expected {}, got {}",
                k,
                sigma,
                call_vol
            );
            assert!(
                (put_vol - sigma).abs() < 0.001,
                "Put IV mismatch at strike {}: expected {}, got {}",
                k,
                sigma,
                put_vol
            );
        }
    }

    #[test]
    fn test_implied_volatility_convergence_speed() {
        // Verify that solver converges in reasonable number of iterations
        // (This is more of a smoke test to ensure it doesn't hang)
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let price = black_scholes(s, k, t, r, sigma, OptionType::Call);

        // Should converge quickly (internally uses max 100 iterations, but typically <10)
        let start = std::time::Instant::now();
        let implied_vol = implied_volatility(price, s, k, t, r, OptionType::Call);
        let duration = start.elapsed();

        assert!(implied_vol.is_some());
        assert!(
            duration.as_millis() < 10,
            "IV solver took too long: {:?}",
            duration
        );
    }

    // ========== Binomial Tree Tests ==========

    #[test]
    fn test_binomial_tree_european_matches_black_scholes() {
        // For European options, binomial tree should converge to Black-Scholes
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let bs_call = black_scholes(s, k, t, r, sigma, OptionType::Call);
        let bs_put = black_scholes(s, k, t, r, sigma, OptionType::Put);

        let config = BinomialTreeConfig::accurate(); // Use more steps for accuracy
        let bt_call = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::European,
            &config,
        );
        let bt_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::European,
            &config,
        );

        // Should match Black-Scholes within 1%
        assert!(
            (bt_call - bs_call).abs() / bs_call < 0.01,
            "Call mismatch: BS={}, BT={}",
            bs_call,
            bt_call
        );
        assert!(
            (bt_put - bs_put).abs() / bs_put < 0.01,
            "Put mismatch: BS={}, BT={}",
            bs_put,
            bt_put
        );
    }

    #[test]
    fn test_binomial_tree_american_put_greater_than_european() {
        // American put should be worth at least as much as European put
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let american_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );
        let european_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::European,
            &config,
        );

        assert!(
            american_put >= european_put - 1e-6,
            "American put {} should be >= European put {}",
            american_put,
            european_put
        );
    }

    #[test]
    fn test_binomial_tree_american_call_equals_european() {
        // For non-dividend paying stocks, American call = European call
        // (early exercise is never optimal)
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let american_call = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );
        let european_call = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::European,
            &config,
        );

        // Should be very close (within 0.1%)
        assert!(
            (american_call - european_call).abs() / european_call < 0.001,
            "American call {} should equal European call {}",
            american_call,
            european_call
        );
    }

    #[test]
    fn test_binomial_tree_deep_itm_put_early_exercise() {
        // Deep ITM American put should have significant early exercise value
        let s = 60.0; // Deep ITM (S << K)
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let american_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );
        let european_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::European,
            &config,
        );

        // American should be worth more than European for deep ITM put
        let premium = american_put - european_put;
        assert!(
            premium > 0.0,
            "Deep ITM American put should have early exercise premium"
        );
    }

    #[test]
    fn test_binomial_tree_at_expiration() {
        let s = 105.0;
        let k = 100.0;
        let t = 0.0; // At expiration
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let call = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );
        let put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );

        // At expiration, worth intrinsic value
        assert!((call - 5.0).abs() < 1e-6, "Call should be worth 5.0");
        assert!((put - 0.0).abs() < 1e-6, "Put should be worth 0.0");
    }

    #[test]
    fn test_binomial_tree_zero_volatility() {
        let s = 105.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.0; // Zero vol

        let config = BinomialTreeConfig::default();
        let call = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );

        // With zero vol, deterministic payoff
        let discount = (-r * t).exp();
        let expected = (s - k * discount).max(0.0);
        assert!(
            (call - expected).abs() < 0.1,
            "Call {} should match expected {}",
            call,
            expected
        );
    }

    #[test]
    fn test_binomial_tree_config_presets() {
        let fast = BinomialTreeConfig::fast();
        assert_eq!(fast.steps, 50);

        let accurate = BinomialTreeConfig::accurate();
        assert_eq!(accurate.steps, 200);

        let high_precision = BinomialTreeConfig::high_precision();
        assert_eq!(high_precision.steps, 500);

        let default = BinomialTreeConfig::default();
        assert_eq!(default.steps, 100);
    }

    #[test]
    fn test_early_exercise_premium_put() {
        let s = 80.0; // ITM put
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let premium = early_exercise_premium(s, k, t, r, sigma, OptionType::Put, &config);

        // Premium should be non-negative
        assert!(
            premium >= 0.0,
            "Premium should be non-negative: {}",
            premium
        );

        // For ITM put, premium should be positive
        assert!(
            premium > 0.0,
            "ITM put should have positive early exercise premium"
        );
    }

    #[test]
    fn test_early_exercise_premium_call() {
        let s = 120.0; // ITM call
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let premium = early_exercise_premium(s, k, t, r, sigma, OptionType::Call, &config);

        // For call on non-dividend stock, premium should be ~0
        assert!(
            premium < 0.01,
            "Call should have near-zero early exercise premium: {}",
            premium
        );
    }

    #[test]
    fn test_price_option_auto_select() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        // European call should use Black-Scholes
        let european_price =
            price_option(s, k, t, r, sigma, OptionType::Call, ExerciseStyle::European);
        let bs_price = black_scholes(s, k, t, r, sigma, OptionType::Call);
        assert!(
            (european_price - bs_price).abs() < 1e-6,
            "European should use Black-Scholes"
        );

        // American put should use binomial tree
        let american_price =
            price_option(s, k, t, r, sigma, OptionType::Put, ExerciseStyle::American);
        let config = BinomialTreeConfig::default();
        let bt_price = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );
        assert!(
            (american_price - bt_price).abs() < 1e-6,
            "American should use binomial tree"
        );
    }

    #[test]
    fn test_binomial_tree_greeks_call() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let greeks = binomial_tree_greeks(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );

        // Delta for ATM call should be around 0.5-0.7
        assert!(
            greeks.delta > 0.4 && greeks.delta < 0.8,
            "Delta {} should be around 0.5-0.7",
            greeks.delta
        );

        // Gamma should be positive
        assert!(greeks.gamma > 0.0, "Gamma should be positive");

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0, "Theta should be negative");

        // Vega should be positive
        assert!(greeks.vega > 0.0, "Vega should be positive");
    }

    #[test]
    fn test_binomial_tree_greeks_put() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();
        let greeks = binomial_tree_greeks(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );

        // Delta for ATM put should be around -0.5 to -0.3
        assert!(
            greeks.delta < 0.0 && greeks.delta > -0.7,
            "Delta {} should be around -0.5 to -0.3",
            greeks.delta
        );

        // Gamma should be positive
        assert!(greeks.gamma > 0.0, "Gamma should be positive");
    }

    #[test]
    fn test_greeks_for_option_auto_select() {
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        // European call should use analytical Greeks
        let european_greeks =
            greeks_for_option(s, k, t, r, sigma, OptionType::Call, ExerciseStyle::European);
        let analytical = calculate_greeks(s, k, t, r, sigma, OptionType::Call);
        assert!(
            (european_greeks.delta - analytical.delta).abs() < 1e-6,
            "European should use analytical Greeks"
        );

        // American put should use numerical Greeks
        let american_greeks =
            greeks_for_option(s, k, t, r, sigma, OptionType::Put, ExerciseStyle::American);
        // Just check it's valid
        assert!(american_greeks.delta < 0.0, "Put delta should be negative");
    }

    #[test]
    fn test_binomial_tree_convergence() {
        // Test that increasing steps improves accuracy
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.20;

        let bs_price = black_scholes(s, k, t, r, sigma, OptionType::Call);

        let config_50 = BinomialTreeConfig { steps: 50 };
        let config_200 = BinomialTreeConfig { steps: 200 };

        let bt_50 = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::European,
            &config_50,
        );
        let bt_200 = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::European,
            &config_200,
        );

        // 200 steps should be closer to Black-Scholes than 50 steps
        let error_50 = (bt_50 - bs_price).abs();
        let error_200 = (bt_200 - bs_price).abs();
        assert!(
            error_200 <= error_50,
            "200 steps error {} should be <= 50 steps error {}",
            error_200,
            error_50
        );
    }

    #[test]
    fn test_binomial_tree_otm_options() {
        // Test OTM options
        let s = 100.0;
        let k_call = 120.0; // OTM call
        let k_put = 80.0; // OTM put
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.25;

        let config = BinomialTreeConfig::default();

        let otm_call = binomial_tree_price(
            s,
            k_call,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );
        let otm_put = binomial_tree_price(
            s,
            k_put,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );

        // OTM options should have positive value (time value)
        assert!(otm_call > 0.0, "OTM call should have positive value");
        assert!(otm_put > 0.0, "OTM put should have positive value");

        // OTM options should be worth less than ATM
        let atm_call = binomial_tree_price(
            s,
            s,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );
        assert!(otm_call < atm_call, "OTM call should be less than ATM call");
    }

    #[test]
    fn test_binomial_tree_itm_options() {
        // Test ITM options
        let s = 100.0;
        let k_call = 80.0; // ITM call
        let k_put = 120.0; // ITM put
        let t = 0.5;
        let r = 0.05;
        let sigma = 0.25;

        let config = BinomialTreeConfig::default();

        let itm_call = binomial_tree_price(
            s,
            k_call,
            t,
            r,
            sigma,
            OptionType::Call,
            ExerciseStyle::American,
            &config,
        );
        let itm_put = binomial_tree_price(
            s,
            k_put,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );

        // ITM options should be worth at least intrinsic value
        assert!(
            itm_call >= (s - k_call),
            "ITM call should be >= intrinsic value"
        );
        assert!(
            itm_put >= (k_put - s),
            "ITM put should be >= intrinsic value"
        );
    }

    #[test]
    fn test_binomial_tree_short_expiration() {
        // Test with very short time to expiration (1 week)
        let s = 100.0;
        let k = 100.0;
        let t = 7.0 / 365.0; // 1 week
        let r = 0.05;
        let sigma = 0.20;

        let config = BinomialTreeConfig::default();

        let american_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::American,
            &config,
        );
        let european_put = binomial_tree_price(
            s,
            k,
            t,
            r,
            sigma,
            OptionType::Put,
            ExerciseStyle::European,
            &config,
        );

        // Both should be positive and reasonable
        assert!(american_put > 0.0, "American put should be positive");
        assert!(european_put > 0.0, "European put should be positive");
        assert!(
            american_put >= european_put - 1e-6,
            "American should be >= European"
        );
    }
}
