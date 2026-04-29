//! Cursor injection sinks.
//!
//! Primary: `reis` + `ashpd` over the `org.freedesktop.portal.RemoteDesktop`
//! portal — KWin under Plasma 6 implements that portal and exposes libei.
//!
//! Fallback: `evdev` writing absolute-pointer events to `/dev/uinput`.
//! Cross-DE, also works under X11. Requires user in the `input` group.
//!
//! Auto-detect at runtime: try libei first, fall back to uinput.
//!
//! See `PLAN.md` §2.4 (Cursor injection) and §6 Phase 4.
//!
//! Status: scaffold only.
