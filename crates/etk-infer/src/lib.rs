//! Pupil/glint inference engine.
//!
//! Primary backend: `burn-wgpu` (Vulkan via wgpu).
//! Alternate backend: `wonnx` (also Vulkan via wgpu) — used if `burn-import`
//! struggles with EllSeg's exact ONNX op set.
//!
//! Both engines are placed behind an `Engine` trait so the rest of the
//! pipeline is engine-agnostic.
//!
//! See `PLAN.md` §2.3 (Inference engine) and §6 Phase 0 (bench).
//!
//! Status: scaffold only.
