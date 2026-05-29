//! Ellipse fitting, glint detection, and calibration math.
//!
//! Scope of this crate (per `PLAN.md` §6 Phase 3 and §2.7):
//!
//! * [`Ellipse`] — the pupil ellipse parameterisation produced by the model's
//!   regression head and consumed by the screen mapper.
//! * [`OneEuro`] — the One-Euro filter used to smooth the cursor signal
//!   (`PLAN.md` §6 Phase 7, defaults `min_cutoff = 1.0`, `beta = 0.007`).
//! * [`Homography`] — the 2D 9-point homography that maps a gaze feature in
//!   normalised camera space to screen coordinates (`PLAN.md` §6 Phase 3 & 5).
//!
//! These three pieces are pure CPU math with no camera, GPU, or Wayland
//! dependency, so they are implemented and unit-tested here in full. The
//! GPU-bound mask/ellipse *inference* lives in `etk-infer`; this crate only
//! does the geometry on top of the engine's output.
//!
//! See `PLAN.md` §6 Phase 3 (Geometry).

mod ellipse;
mod homography;
mod one_euro;

pub use ellipse::Ellipse;
pub use homography::{Homography, HomographyError};
pub use one_euro::OneEuro;

/// A 2D point in `f64` precision.
///
/// Used throughout the geometry layer for pupil centroids, glint positions,
/// gaze features, and screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /// Construct a point.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    #[inline]
    pub fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl From<(f64, f64)> for Point {
    fn from((x, y): (f64, f64)) -> Self {
        Point::new(x, y)
    }
}
