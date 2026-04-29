//! Config loading + live-reload.
//!
//! Watches `~/.config/eye-tracker-rust/config.toml` via `notify`.
//! Reloads smoothing params and calibration matrix without restarting capture.
//!
//! See `PLAN.md` §2.5 (Config + hot reload).
//!
//! Status: scaffold only.
