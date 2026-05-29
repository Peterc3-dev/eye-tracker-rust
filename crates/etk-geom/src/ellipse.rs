//! Pupil ellipse parameterisation.
//!
//! `EllSeg`'s regression head emits an ellipse directly; this type is the
//! interchange representation the geometry and mapping stages consume.
//! (`PLAN.md` §6 Phase 3.)

use crate::Point;

/// An axis-parameterised 2D ellipse.
///
/// `angle` is the rotation of the major axis in radians, measured
/// counter-clockwise from the positive `x` axis. `a` and `b` are the
/// semi-axis lengths; by convention `a` is the semi-major axis (`a >= b`),
/// but the type does not enforce this so it can faithfully round-trip
/// whatever the regression head produces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// Centre of the ellipse.
    pub center: Point,
    /// First semi-axis length (semi-major by convention).
    pub a: f64,
    /// Second semi-axis length (semi-minor by convention).
    pub b: f64,
    /// Rotation of the `a` axis, radians, CCW from +x.
    pub angle: f64,
}

impl Ellipse {
    /// Construct an ellipse from centre, semi-axes, and rotation.
    #[inline]
    pub const fn new(center: Point, a: f64, b: f64, angle: f64) -> Self {
        Self {
            center,
            a,
            b,
            angle,
        }
    }

    /// Area of the ellipse, `π·a·b`.
    #[inline]
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.a * self.b
    }

    /// Eccentricity in `[0, 1)`.
    ///
    /// Returns `0` for a circle. Computed from the larger and smaller of the
    /// two semi-axes so it is independent of which axis is stored in `a`.
    pub fn eccentricity(&self) -> f64 {
        let major = self.a.abs().max(self.b.abs());
        let minor = self.a.abs().min(self.b.abs());
        if major == 0.0 {
            return 0.0;
        }
        let ratio = minor / major;
        (1.0 - ratio * ratio).max(0.0).sqrt()
    }

    /// Sample the ellipse boundary at parameter `t` (radians).
    ///
    /// `t` runs `0..2π` around the un-rotated ellipse before the `angle`
    /// rotation and centre translation are applied.
    pub fn point_at(&self, t: f64) -> Point {
        let (st, ct) = t.sin_cos();
        let ux = self.a * ct;
        let uy = self.b * st;
        let (sa, ca) = self.angle.sin_cos();
        Point::new(
            self.center.x + ux * ca - uy * sa,
            self.center.y + ux * sa + uy * ca,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn circle_has_zero_eccentricity() {
        let e = Ellipse::new(Point::new(0.0, 0.0), 5.0, 5.0, 0.0);
        approx(e.eccentricity(), 0.0);
    }

    #[test]
    fn area_matches_formula() {
        let e = Ellipse::new(Point::new(1.0, 2.0), 3.0, 2.0, 0.0);
        approx(e.area(), std::f64::consts::PI * 6.0);
    }

    #[test]
    fn eccentricity_is_axis_order_independent() {
        let wide = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.0, 0.0);
        let tall = Ellipse::new(Point::new(0.0, 0.0), 2.0, 4.0, 0.0);
        approx(wide.eccentricity(), tall.eccentricity());
    }

    #[test]
    fn point_at_zero_is_on_major_axis() {
        let e = Ellipse::new(Point::new(10.0, 20.0), 4.0, 2.0, 0.0);
        let p = e.point_at(0.0);
        approx(p.x, 14.0);
        approx(p.y, 20.0);
    }

    #[test]
    fn point_at_respects_rotation() {
        // 90° rotation: the +a axis now points along +y.
        let e = Ellipse::new(Point::new(0.0, 0.0), 4.0, 2.0, std::f64::consts::FRAC_PI_2);
        let p = e.point_at(0.0);
        approx(p.x, 0.0);
        approx(p.y, 4.0);
    }
}
