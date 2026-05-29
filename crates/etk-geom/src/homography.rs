//! 2D homography for gaze-feature → screen mapping.
//!
//! `PLAN.md` §6 Phase 3 & 5: a 9-point calibration fits a planar homography
//! that maps a gaze feature (glint-relative pupil position, in normalised
//! camera coordinates) onto screen pixels.
//!
//! A homography has 8 degrees of freedom and needs ≥4 point correspondences;
//! the 3×3 grid gives 9, so the system is over-determined and solved in the
//! least-squares sense. We use the Direct Linear Transform: each
//! correspondence contributes two rows to a linear system in the 8 unknowns
//! (`h33` is fixed to 1), which is then solved via the normal equations
//! (`AᵀA h = Aᵀb`) with a small hand-rolled Gaussian-elimination solver — no
//! external linear-algebra dependency, keeping this crate light per
//! `PLAN.md` §2 (lean deps).

use crate::Point;

/// Errors that can arise while fitting or applying a homography.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HomographyError {
    /// Fewer than 4 correspondences were supplied (8 DoF needs ≥4 points).
    TooFewPoints {
        /// Number of correspondences provided.
        got: usize,
    },
    /// `src` and `dst` slices had different lengths.
    LengthMismatch {
        /// Length of the source slice.
        src: usize,
        /// Length of the destination slice.
        dst: usize,
    },
    /// The normal-equations system was singular (degenerate / collinear
    /// calibration points). No unique homography exists.
    Singular,
}

impl std::fmt::Display for HomographyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HomographyError::TooFewPoints { got } => {
                write!(f, "need at least 4 correspondences, got {got}")
            }
            HomographyError::LengthMismatch { src, dst } => {
                write!(f, "src/dst length mismatch: {src} vs {dst}")
            }
            HomographyError::Singular => {
                write!(f, "degenerate correspondences: system is singular")
            }
        }
    }
}

impl std::error::Error for HomographyError {}

/// A 2D projective transform stored as a row-major 3×3 matrix with `h[8] = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    /// Row-major 3×3 matrix entries `[h11, h12, h13, h21, h22, h23, h31, h32, h33]`.
    pub h: [f64; 9],
}

impl Homography {
    /// The identity homography.
    pub const fn identity() -> Self {
        Self {
            h: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Apply the homography to a point.
    ///
    /// Returns `None` if the projective denominator collapses to zero (point
    /// maps to infinity).
    pub fn map(&self, p: Point) -> Option<Point> {
        let h = &self.h;
        let w = h[6] * p.x + h[7] * p.y + h[8];
        if w.abs() < 1e-12 {
            return None;
        }
        Some(Point::new(
            (h[0] * p.x + h[1] * p.y + h[2]) / w,
            (h[3] * p.x + h[4] * p.y + h[5]) / w,
        ))
    }

    /// Fit a homography from `src → dst` correspondences (least squares, DLT).
    ///
    /// Requires `src.len() == dst.len() >= 4`. For the 9-point calibration in
    /// `PLAN.md` §6 Phase 5 this is called with 9 correspondences.
    pub fn fit(src: &[Point], dst: &[Point]) -> Result<Self, HomographyError> {
        if src.len() != dst.len() {
            return Err(HomographyError::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        if src.len() < 4 {
            return Err(HomographyError::TooFewPoints { got: src.len() });
        }

        // Build the 2N×8 system A·h = b, with h = [h11..h32], h33 = 1.
        //   x' = (h11·x + h12·y + h13) / (h31·x + h32·y + 1)
        //   y' = (h21·x + h22·y + h23) / (h31·x + h32·y + 1)
        // Rearranged to linear form:
        //   h11·x + h12·y + h13 - h31·x·x' - h32·y·x' = x'
        //   h21·x + h22·y + h23 - h31·x·y' - h32·y·y' = y'
        let n = src.len();
        let mut a = vec![[0.0f64; 8]; 2 * n];
        let mut b = vec![0.0f64; 2 * n];

        for i in 0..n {
            let (x, y) = (src[i].x, src[i].y);
            let (xp, yp) = (dst[i].x, dst[i].y);

            let r0 = 2 * i;
            a[r0] = [x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp];
            b[r0] = xp;

            let r1 = 2 * i + 1;
            a[r1] = [0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp];
            b[r1] = yp;
        }

        // Normal equations: (AᵀA) h = Aᵀb  → 8×8 symmetric system.
        let mut ata = [[0.0f64; 8]; 8];
        let mut atb = [0.0f64; 8];
        for row in 0..2 * n {
            for i in 0..8 {
                atb[i] += a[row][i] * b[row];
                for j in 0..8 {
                    ata[i][j] += a[row][i] * a[row][j];
                }
            }
        }

        let sol = solve_8x8(ata, atb).ok_or(HomographyError::Singular)?;

        Ok(Homography {
            h: [
                sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], 1.0,
            ],
        })
    }
}

impl Default for Homography {
    fn default() -> Self {
        Self::identity()
    }
}

/// Solve an 8×8 linear system `A x = b` via Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
fn solve_8x8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    const N: usize = 8;
    for col in 0..N {
        // Partial pivot: find the row with the largest magnitude in `col`.
        let (pivot, best) = a
            .iter()
            .enumerate()
            .skip(col)
            .map(|(r, row)| (r, row[col].abs()))
            .fold(
                (col, 0.0f64),
                |acc, (r, v)| if v > acc.1 { (r, v) } else { acc },
            );
        if best < 1e-12 {
            return None; // singular
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        // Eliminate below the pivot. Copy the pivot row out so the borrow
        // checker (and clippy) are happy mutating the rows below it.
        let pivot_row = a[col];
        let pivot_b = b[col];
        let pivot_diag = pivot_row[col];
        for (r, row) in a.iter_mut().enumerate().skip(col + 1) {
            let factor = row[col] / pivot_diag;
            if factor != 0.0 {
                for c in col..N {
                    row[c] -= factor * pivot_row[c];
                }
                b[r] -= factor * pivot_b;
            }
        }
    }

    // Back-substitution.
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_pt(a: Point, b: Point, tol: f64) {
        assert!(
            (a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol,
            "{a:?} != {b:?}"
        );
    }

    #[test]
    fn identity_maps_points_unchanged() {
        let h = Homography::identity();
        let p = Point::new(3.0, 7.0);
        approx_pt(h.map(p).unwrap(), p, 1e-12);
    }

    #[test]
    fn rejects_too_few_points() {
        let src = [Point::new(0.0, 0.0); 3];
        let dst = [Point::new(0.0, 0.0); 3];
        assert_eq!(
            Homography::fit(&src, &dst),
            Err(HomographyError::TooFewPoints { got: 3 })
        );
    }

    #[test]
    fn rejects_length_mismatch() {
        let src = [Point::new(0.0, 0.0); 5];
        let dst = [Point::new(0.0, 0.0); 4];
        assert!(matches!(
            Homography::fit(&src, &dst),
            Err(HomographyError::LengthMismatch { src: 5, dst: 4 })
        ));
    }

    #[test]
    fn recovers_affine_scale_and_translate() {
        // Map [0,1]² gaze space onto a 1920×1080 screen: x' = 1920x, y' = 1080y.
        let src = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(0.0, 1.0),
            Point::new(1.0, 1.0),
            Point::new(0.5, 0.5),
        ];
        let dst: Vec<Point> = src
            .iter()
            .map(|p| Point::new(p.x * 1920.0, p.y * 1080.0))
            .collect();
        let h = Homography::fit(&src, &dst).unwrap();
        approx_pt(
            h.map(Point::new(0.25, 0.75)).unwrap(),
            Point::new(480.0, 810.0),
            1e-6,
        );
    }

    #[test]
    fn recovers_a_true_projective_map() {
        // A known non-affine homography (note nonzero h31/h32).
        let truth = Homography {
            h: [1.2, 0.1, 5.0, -0.05, 0.9, 3.0, 0.001, 0.002, 1.0],
        };
        // 9-point 3×3 grid in gaze space.
        let mut src = Vec::new();
        for gy in 0..3 {
            for gx in 0..3 {
                src.push(Point::new(gx as f64, gy as f64));
            }
        }
        let dst: Vec<Point> = src.iter().map(|p| truth.map(*p).unwrap()).collect();
        let fitted = Homography::fit(&src, &dst).unwrap();
        // Fitted map reproduces a held-out interior point.
        let test = Point::new(1.5, 0.5);
        approx_pt(fitted.map(test).unwrap(), truth.map(test).unwrap(), 1e-4);
    }

    #[test]
    fn collinear_points_are_singular() {
        // All source points on a line → degenerate.
        let src = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
            Point::new(4.0, 4.0),
        ];
        let dst = [
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(2.0, 0.0),
            Point::new(3.0, 0.0),
            Point::new(4.0, 0.0),
        ];
        assert_eq!(Homography::fit(&src, &dst), Err(HomographyError::Singular));
    }
}
