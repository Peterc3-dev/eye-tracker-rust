//! Final daemon — full pipeline:
//! capture → preprocess → inference → geometry → calibration map →
//! smoothing → cursor sink (libei or uinput).
//!
//! Hotkey pause, dwell-click, autostart via systemd user service.
//!
//! See `PLAN.md` §6 Phase 4-7.

fn main() {
    println!("eye-tracker scaffold — see PLAN.md §6 Phase 4-7");
}
