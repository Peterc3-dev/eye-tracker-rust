//! Phase 2-3 — capture + inference with overlay.
//!
//! ROI-crop a 256×256 patch around the previous-frame's pupil centroid;
//! draw predicted mask in red and fitted ellipse in green.
//! Pass criteria: stable mask under blinks/head-movement; ≥30 FPS sustained.
//!
//! See `PLAN.md` §6 Phase 2 + Phase 3.

fn main() {
    println!("track-preview scaffold — see PLAN.md §6 Phase 2-3");
}
