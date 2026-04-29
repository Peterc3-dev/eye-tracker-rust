//! Phase 0 — measure `burn-wgpu` Vulkan inference latency.
//!
//! Goal: 1000 iterations of EllSeg @ 320×240 monochrome on the discovered
//! Vulkan adapter. Log mean / p50 / p99 ms.
//!
//! Pass criteria: ≤12 ms mean, ≤20 ms p99 (single eye).
//!
//! See `PLAN.md` §6 Phase 0.

fn main() {
    println!("bench-infer scaffold — see PLAN.md §6 Phase 0");
}
