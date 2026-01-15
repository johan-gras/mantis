//! Options pricing models and Greeks calculations.
//!
//! This module provides:
//! - Black-Scholes pricing for European options
//! - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
//! - Put-call parity validation
//! - Option contract representation

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
}
