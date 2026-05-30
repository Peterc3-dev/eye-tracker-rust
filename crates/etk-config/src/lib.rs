//! Config loading + live-reload.
//!
//! Defines the typed [`Config`] tree that the daemon reads from
//! `~/.config/eye-tracker-rust/config.toml`, plus helpers to load it from a
//! path or a string. Every field has a documented default drawn from
//! `PLAN.md`, so a missing or partial file still yields a working config.
//!
//! Live reload (watching the file via `notify` and pushing updated smoothing
//! params / calibration into the running pipeline) is a daemon-event-loop
//! concern — see the `TODO` on [`config_path`]. This crate provides the data
//! model and parsing; the watcher is wired up in the `eye-tracker` binary.
//!
//! See `PLAN.md` §2.5 (Config + hot reload).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Errors from loading or parsing config.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("reading config file {path}: {source}")]
    Io {
        /// The path that failed to read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The config file was not valid TOML / did not match the schema.
    #[error("parsing config: {0}")]
    Parse(#[from] toml::de::Error),
}

/// Top-level configuration.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// UDP capture settings.
    pub capture: CaptureConfig,
    /// One-Euro cursor smoothing (`PLAN.md` §6 Phase 7).
    pub smoothing: SmoothingConfig,
    /// Dwell-to-click behaviour (`PLAN.md` §6 Phase 7).
    pub dwell: DwellConfig,
    /// Binocular fusion (`PLAN.md` §6 Phase 6).
    pub fusion: FusionConfig,
}

/// UDP capture configuration (`PLAN.md` §2.1, §6 Phase 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaptureConfig {
    /// UDP port the left-eye module streams to.
    pub left_port: u16,
    /// UDP port the right-eye module streams to.
    pub right_port: u16,
    /// Per-eye ring-buffer capacity in frames (`PLAN.md` §1 ArrayQueue cap=4).
    pub ring_capacity: usize,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            left_port: 5005,
            right_port: 5006,
            ring_capacity: 4,
        }
    }
}

/// One-Euro filter parameters.
///
/// Defaults match `PLAN.md` §6 Phase 7 (`min_cutoff = 1.0`, `beta = 0.007`).
/// These map directly onto [`etk_geom::OneEuro::with_params`] in the
/// `etk-geom` crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SmoothingConfig {
    /// Minimum cutoff frequency (Hz).
    pub min_cutoff: f64,
    /// Speed coefficient.
    pub beta: f64,
    /// Derivative cutoff frequency (Hz).
    pub d_cutoff: f64,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

/// Dwell-click configuration.
///
/// Defaults match `PLAN.md` §6 Phase 7: "hold within 30 px circle for 0.7s
/// → emit `BTN_LEFT`".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DwellConfig {
    /// Whether dwell-click is enabled.
    pub enabled: bool,
    /// Radius (screen px) the gaze must stay within to count as dwelling.
    pub radius_px: f64,
    /// Hold time (seconds) before a click is emitted.
    pub hold_secs: f64,
}

impl Default for DwellConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            radius_px: 30.0,
            hold_secs: 0.7,
        }
    }
}

/// Binocular fusion configuration.
///
/// Default `min_confidence` matches `PLAN.md` §6 Phase 6 ("if one eye's
/// confidence < 0.6, ignore").
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FusionConfig {
    /// Confidence-weighted binocular average if `true`; dominant-eye only if
    /// `false` (the Tobii Eye X strategy, `PLAN.md` §7 risk 5).
    pub binocular: bool,
    /// Minimum per-eye confidence to include an eye in the fusion.
    pub min_confidence: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            binocular: true,
            min_confidence: 0.6,
        }
    }
}

impl Config {
    /// Parse a [`Config`] from a TOML string.
    ///
    /// Missing tables/fields fall back to their documented defaults.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        Ok(toml::from_str(s)?)
    }

    /// Serialise this config to a pretty TOML string.
    pub fn to_toml_string(&self) -> String {
        // `Config` is a plain struct of primitives; serialisation cannot fail.
        toml::to_string_pretty(self).expect("Config serialises to TOML")
    }

    /// Load a [`Config`] from a file path.
    ///
    /// If the file does not exist, returns [`Config::default`] rather than an
    /// error — a fresh install with no config file should still run.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(s) => Self::from_toml_str(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }
}

/// The canonical config path: `$XDG_CONFIG_HOME/eye-tracker-rust/config.toml`,
/// falling back to `$HOME/.config/eye-tracker-rust/config.toml`.
///
/// Returns `None` if neither `XDG_CONFIG_HOME` nor `HOME` is set.
///
/// TODO(Phase 7): the daemon should `notify`-watch this path and hot-reload
/// [`SmoothingConfig`], [`DwellConfig`], and the calibration matrix without
/// restarting capture (`PLAN.md` §2.5). The watcher belongs in the
/// `eye-tracker` binary's event loop, not in this data-model crate.
pub fn config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("eye-tracker-rust").join("config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_plan() {
        let c = Config::default();
        assert_eq!(c.smoothing.min_cutoff, 1.0);
        assert_eq!(c.smoothing.beta, 0.007);
        assert_eq!(c.dwell.radius_px, 30.0);
        assert_eq!(c.dwell.hold_secs, 0.7);
        assert_eq!(c.fusion.min_confidence, 0.6);
        assert_eq!(c.capture.ring_capacity, 4);
    }

    #[test]
    fn empty_string_yields_defaults() {
        let c = Config::from_toml_str("").unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn partial_config_merges_with_defaults() {
        let c = Config::from_toml_str("[smoothing]\nbeta = 0.05\n").unwrap();
        assert_eq!(c.smoothing.beta, 0.05); // overridden
        assert_eq!(c.smoothing.min_cutoff, 1.0); // default retained
        assert_eq!(c.dwell.hold_secs, 0.7); // unrelated table defaulted
    }

    #[test]
    fn round_trips_through_toml() {
        let mut c = Config::default();
        c.smoothing.beta = 0.02;
        c.dwell.enabled = false;
        c.capture.left_port = 7000;
        let s = c.to_toml_string();
        let back = Config::from_toml_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = Config::from_toml_str("[smoothing]\nbogus = 1\n");
        assert!(err.is_err());
    }

    #[test]
    fn load_missing_file_returns_default() {
        let c = Config::load("/nonexistent/eye-tracker-rust/config.toml").unwrap();
        assert_eq!(c, Config::default());
    }
}
