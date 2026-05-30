//! One-Euro filter for cursor smoothing.
//!
//! Reference: Casiez, Roussel & Vogel, "1€ Filter: A Simple Speed-based
//! Low-pass Filter for Noisy Input in Interactive Systems" (CHI 2012).
//!
//! Defaults follow `PLAN.md` §6 Phase 7: `min_cutoff = 1.0`, `beta = 0.007`.
//! Hand-rolled per `PLAN.md` §2.7 ("~30 LoC, beats Kalman for cursor
//! smoothing").

/// A single-channel One-Euro filter.
///
/// Construct one filter per scalar signal (e.g. one each for `x` and `y`).
/// Feed timestamped samples to [`OneEuro::filter`]; the filter adapts its
/// cutoff frequency to the signal speed, trading lag for jitter the faster
/// the signal moves.
#[derive(Debug, Clone)]
pub struct OneEuro {
    /// Minimum cutoff frequency (Hz). Lower = smoother but laggier when still.
    min_cutoff: f64,
    /// Speed coefficient. Higher = less lag when moving fast.
    beta: f64,
    /// Cutoff frequency (Hz) for the derivative low-pass.
    d_cutoff: f64,

    /// Previous filtered value, `None` until the first sample.
    x_prev: Option<f64>,
    /// Previous filtered derivative.
    dx_prev: f64,
    /// Timestamp (seconds) of the previous sample.
    t_prev: Option<f64>,
}

impl OneEuro {
    /// Create a filter with the plan defaults (`min_cutoff = 1.0`,
    /// `beta = 0.007`, `d_cutoff = 1.0`).
    pub fn new() -> Self {
        Self::with_params(1.0, 0.007, 1.0)
    }

    /// Create a filter with explicit parameters.
    ///
    /// # Panics
    /// Panics if `min_cutoff` or `d_cutoff` is not strictly positive, since a
    /// non-positive cutoff yields a meaningless filter alpha.
    pub fn with_params(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        assert!(min_cutoff > 0.0, "min_cutoff must be > 0");
        assert!(d_cutoff > 0.0, "d_cutoff must be > 0");
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: None,
            dx_prev: 0.0,
            t_prev: None,
        }
    }

    /// Reset the filter to its initial (uninitialised) state.
    ///
    /// The next sample is passed through unfiltered, as on first use.
    pub fn reset(&mut self) {
        self.x_prev = None;
        self.dx_prev = 0.0;
        self.t_prev = None;
    }

    /// Smoothing factor for a cutoff frequency `cutoff` (Hz) at timestep
    /// `dt` (seconds).
    #[inline]
    fn alpha(cutoff: f64, dt: f64) -> f64 {
        let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        1.0 / (1.0 + tau / dt)
    }

    /// Filter one sample taken at absolute time `t` (seconds).
    ///
    /// Returns the smoothed value. Samples must be supplied in non-decreasing
    /// time order; a non-positive timestep (duplicate or out-of-order
    /// timestamp) is treated as "no time elapsed" and the raw value is
    /// returned to avoid a divide-by-zero blow-up.
    pub fn filter(&mut self, x: f64, t: f64) -> f64 {
        let (x_prev, t_prev) = match (self.x_prev, self.t_prev) {
            (Some(xp), Some(tp)) => (xp, tp),
            // First sample: seed state, pass through.
            _ => {
                self.x_prev = Some(x);
                self.t_prev = Some(t);
                self.dx_prev = 0.0;
                return x;
            }
        };

        let dt = t - t_prev;
        if dt <= 0.0 {
            // Out-of-order / duplicate timestamp: keep state, return raw.
            return x;
        }

        // Low-pass the derivative.
        let dx = (x - x_prev) / dt;
        let a_d = Self::alpha(self.d_cutoff, dt);
        let dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev;

        // Speed-adaptive cutoff, then low-pass the value.
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = Self::alpha(cutoff, dt);
        let x_hat = a * x + (1.0 - a) * x_prev;

        self.x_prev = Some(x_hat);
        self.dx_prev = dx_hat;
        self.t_prev = Some(t);
        x_hat
    }
}

impl Default for OneEuro {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_passes_through() {
        let mut f = OneEuro::new();
        assert_eq!(f.filter(3.5, 0.0), 3.5);
    }

    #[test]
    fn constant_signal_converges_to_value() {
        let mut f = OneEuro::new();
        let mut y = 0.0;
        for i in 0..200 {
            y = f.filter(10.0, i as f64 / 60.0);
        }
        assert!((y - 10.0).abs() < 1e-3, "converged to {y}");
    }

    #[test]
    fn output_lags_a_step_input() {
        // A sudden jump should be smoothed: the first filtered value after the
        // step lands strictly between the old and new level.
        let mut f = OneEuro::new();
        f.filter(0.0, 0.0);
        let y = f.filter(100.0, 1.0 / 60.0);
        assert!(y > 0.0 && y < 100.0, "expected smoothing, got {y}");
    }

    #[test]
    fn reduces_jitter_variance() {
        // Noisy signal around a constant mean: filtered variance < raw.
        let raw = [0.0, 2.0, -2.0, 1.5, -1.0, 2.0, -1.5, 0.5, -2.0, 1.0];
        let mut f = OneEuro::with_params(1.0, 0.0, 1.0); // beta=0 => pure low-pass
        let mut filtered = Vec::new();
        for (i, &v) in raw.iter().enumerate() {
            filtered.push(f.filter(v, i as f64 / 60.0));
        }
        let var = |xs: &[f64]| {
            let m = xs.iter().sum::<f64>() / xs.len() as f64;
            xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64
        };
        assert!(var(&filtered) < var(&raw), "filter did not reduce variance");
    }

    #[test]
    fn out_of_order_timestamp_returns_raw_without_panic() {
        let mut f = OneEuro::new();
        f.filter(1.0, 1.0);
        assert_eq!(f.filter(5.0, 0.5), 5.0); // dt <= 0 => raw
    }

    #[test]
    fn reset_restores_passthrough() {
        let mut f = OneEuro::new();
        f.filter(1.0, 0.0);
        f.filter(2.0, 0.1);
        f.reset();
        assert_eq!(f.filter(9.0, 0.2), 9.0);
    }
}
