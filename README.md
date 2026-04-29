# eye-tracker-rust

Realtime CNN-based binocular pupil/gaze tracker driving the Wayland cursor on Linux. Pure Rust + Vulkan via [`wgpu`](https://github.com/gfx-rs/wgpu) / [`burn`](https://github.com/tracel-ai/burn).

Camera input is from EyeTrackVR-style ESP32-S3 modules with IR illumination, streamed over Wi-Fi UDP. Inference runs on any Vulkan-capable GPU (developed against AMD Radeon 890M / gfx1150 on CachyOS, but the binary should run unchanged on RDNA/Ada/Intel/Apple).

**Status:** scaffold only. See [`PLAN.md`](./PLAN.md) for the full design and phased build plan.

## Why Vulkan and not ROCm?

Trade-off documented in [`PLAN.md` §4](./PLAN.md). Short version: at our model size (~1.3 M params, EllSeg-Gen at 320×240), the ~30-50% conv-perf delta between HIP and Vulkan is invisible inside a 33 ms / 30 FPS frame budget. In exchange we get a single-binary that runs on any GPU, no ROCm version coupling, no `HSA_OVERRIDE_GFX_VERSION` dance, and no self-built ONNX Runtime.

## Workspace layout

```
eye-tracker-rust/
├── Cargo.toml              # workspace root
├── PLAN.md                 # design + phased build plan
├── crates/
│   ├── etk-capture/        # UDP MJPEG receiver
│   ├── etk-preproc/        # JPEG decode, ROI crop, normalize, GPU upload
│   ├── etk-infer/          # burn-wgpu (primary) + wonnx (alt) behind Engine trait
│   ├── etk-geom/           # ellipse fit, glint, calibration math
│   ├── etk-output/         # libei + uinput cursor sinks
│   ├── etk-config/         # serde + notify live-reload
│   └── etk-gui/            # egui calibration & preview
├── apps/
│   ├── bench-infer/        # Phase 0: measure inference latency
│   ├── capture-preview/    # Phase 1: live camera feed
│   ├── track-preview/      # Phase 2-3: inference + overlay
│   └── eye-tracker/        # final daemon
└── models/                 # ONNX weights (Git LFS planned)
```

## Build

Standard Rust toolchain (stable). Vulkan loader required at runtime (RADV via mesa or AMDVLK).

```bash
cargo check --workspace          # verify the workspace
cargo build --release            # full build
cargo run --release -p bench-infer   # Phase 0 benchmark (once implemented)
```

System packages on Arch / CachyOS:
```bash
sudo pacman -S vulkan-icd-loader vulkan-radeon vulkan-headers libei
```

## Hardware

EyeTrackVR canonical BOM (~$50-70). See [`PLAN.md` §5](./PLAN.md). Long-lead items: 2× Seeed XIAO ESP32-S3 Sense + IR LED PCB v4 Gerbers from JLCPCB.

## Phased plan

| Phase | Goal | Effort |
|---|---|---|
| 0 | Vulkan inference bench (EllSeg @ 320×240 ≤ 12 ms) | 1 evening |
| 1 | UDP capture + live preview | 1 evening |
| 2 | Inference + mask/ellipse overlay | 1-2 evenings |
| 3 | Geometry + 2D screen mapping | 1 evening |
| 4 | Cursor injection via libei (uinput fallback) | ½ evening |
| 5 | 9-point calibration UX in egui | 1 evening |
| 6 | Binocular fusion | 1 evening |
| 7 | Polish: One-Euro filter, dwell-click, autostart | 1-2 evenings |

Full per-phase pass criteria, risks, and decision checkpoints in [`PLAN.md`](./PLAN.md).

## License

MIT OR Apache-2.0
