# `eye-tracker-rust` — Implementation Plan

> Target: realtime CNN-based binocular pupil/gaze tracking driving the Wayland cursor on AMD Ryzen AI 9 HX 370 (Radeon 890M / gfx1150) under CachyOS + KDE Plasma 6.6.4. Camera capture from EyeTrackVR-style ESP32-S3 modules over Wi-Fi UDP. **Inference layer: Rust + Vulkan via wgpu** (decided 2026-04-29 — chosen over ROCm for stability + portability; the ~30-50% conv perf delta is a non-issue at the model size we're running).

---

## 1. Architecture & Data Flow

```mermaid
flowchart LR
    subgraph HW["Wearable hardware (per eye)"]
      CAM[OV2640<br/>IR-pass filter]
      LED[850 nm IR LEDs<br/>≤5 mA, non-focused]
      MCU[XIAO ESP32-S3 Sense<br/>OpenIris firmware]
      CAM --> MCU
      LED -.illuminate.-> CAM
    end

    MCU -- UDP/MJPEG over 5 GHz Wi-Fi --> NET((Local SSID))

    subgraph DAEMON["eye-tracker-rust (Rust daemon)"]
      direction TB
      RX[capture::Receiver<br/>tokio UDP, per-eye thread]
      RING[(crossbeam ArrayQueue<br/>cap=4 frames/eye)]
      PRE[preproc::Stage<br/>JPEG decode → ROI crop → norm]
      INF[infer::Engine<br/>burn-wgpu / wonnx<br/>VULKAN compute]
      GEO[geom::EllipseFit + glint]
      MAP[mapper::Calibration<br/>9-pt homography]
      KAL[smooth::OneEuro + dwell]
      SINK[output::CursorSink]
      RX --> RING --> PRE --> INF --> GEO --> MAP --> KAL --> SINK
      CFG[config.toml<br/>notify watcher] -.live reload.-> PRE & MAP & KAL
    end

    SINK -- 1st: libei via ashpd<br/>RemoteDesktop portal --> KWIN[KWin / Plasma 6.6.4]
    SINK -- 2nd: uinput fallback --> KERN[/dev/uinput]
    SINK -- broadcast --> MQTT[(MQTT/UDP fan-out)]

    GUI[egui calibration UI<br/>winit + wgpu] <-->|shared frames| PRE
```

**33 ms frame budget @ 30 FPS:**

| Stage | Vulkan budget | Notes |
|---|---|---|
| Wi-Fi UDP RX + jitter buffer | 5 ms | One frame slack |
| JPEG decode | 2 ms | `image` crate or skip if raw greyscale |
| ROI crop + normalize + GPU upload | 2 ms | wgpu staging buffer |
| CNN inference (per eye, 320×240, **Vulkan**) | 10 ms | Both eyes batched ≈12 ms total |
| Ellipse fit + glint + screen map | 1 ms | CPU |
| Smoothing + cursor send | 1 ms | One-Euro + libei |
| **Slack** | **12 ms** | Drops, GC, recompile |

Vulkan compute is ~2× HIP for conv-heavy nets but EllSeg only needs ~1 GFLOP/frame; the 890M's Vulkan path delivers ≥4 TFLOPs sustained on RDNA 3.5 — plenty of headroom.

---

## 2. Crate Selection

### 2.1 Camera receive — `tokio` + custom UDP framing
OpenIris streams MJPEG over a tiny HTTP `multipart/x-mixed-replace` socket. Use `tokio` 1.52 + `bytes::BytesMut` + a JPEG SOI/EOI scanner. ~80 LoC. Skip `gstreamer-rs` (200 MB transitive deps for nothing).

### 2.2 Image / tensor ops — `image` 0.25 + `imageproc` 0.26 + `ndarray` 0.17
`image` for JPEG decode + resize, `imageproc` for ellipse-residual + CCL on the predicted mask, `ndarray` as the standard interchange tensor.

### 2.3 Inference engine — **Vulkan via wgpu** (primary decision)

| Engine | Vulkan story | Verdict |
|---|---|---|
| **`burn` 0.21 + `burn-wgpu`** | Mature WGPU backend (Vulkan on Linux via RADV or AMDVLK; cross-platform free). `burn-import` reads ONNX directly into a Burn graph. No FFI. Single Rust binary. | **Primary pick.** |
| **`wonnx` 0.5+** | Pure-Rust ONNX runtime on top of wgpu. Smaller scope than burn — *just* inference, no training. Op coverage is the unknown for EllSeg (does cover Conv2d, ConvTranspose2d, BN, ReLU, Sigmoid, Sub, Mul). | **Alternate** if burn-import struggles with EllSeg's exact op set. |
| `ort` + ROCm/MIGraphX EP | Faster (~30-50% on convs) but requires self-built `libonnxruntime.so` linked against ROCm 7.2.0 + custom MIOpen for gfx1150 + `HSA_OVERRIDE_GFX_VERSION=11.0.0`. **Not worth the toolchain wrestling for v1.** | Future Phase 8 if perf becomes a real problem (it won't). |
| `candle` | No Vulkan/wgpu support yet. ROCm is open issue. | Reject. |

**Decision: build on `burn-wgpu` from day one.** If a specific op fails to import, swap that branch to `wonnx` or write a custom Burn op. Both engines target the same Vulkan surface, so the GPU path is unchanged either way.

### 2.4 Cursor injection — `reis` 0.6.1 + `ashpd` 0.13.10 (primary), `evdev` 0.13 (fallback)
- `reis` is the maintained Rust libei impl (ids1024 / Pop!_OS), Apr 2026 active. Wire via the `org.freedesktop.portal.RemoteDesktop` portal that `ashpd` opens; KWin under Plasma 6.6.4 implements that portal and exposes libei over the resulting fd. **The future-proof Wayland-native path.**
- Fallback: `evdev` writing to `/dev/uinput`. User is already in `input` group. Cross-DE, works on X11 too. KWin treats it as a real mouse — focus follows the synthetic cursor.
- Avoid `wlroots-virtual-pointer-unstable-v1` — KWin doesn't implement it.

### 2.5 Config + hot reload
`serde` + `toml` + `notify` 9.0.0-rc.3. Watch `~/.config/eye-tracker-rust/config.toml`.

### 2.6 GUI for calibration & live preview — `egui` 0.34 (eframe + winit + wgpu)
Same wgpu instance for UI + camera preview texture + inference compute. Vulkan-on-Linux via RADV is rock-solid (used by OBS, Blender, Bevy).

### 2.7 Supporting
- `crossbeam-queue` SPSC ring buffers
- `tracing` + `tracing-flame` for perf bench
- `clap` 4.6 CLI
- One-Euro filter hand-rolled (~30 LoC, beats Kalman for cursor smoothing)

---

## 3. Model Selection

### Survey

| Model | Input | Params | Notes |
|---|---|---|---|
| **EllSeg-Gen** ([RSKothari/EllSeg](https://github.com/RSKothari/EllSeg)) | 320×240 | ~1.3 M | UNet + ellipse regression head. Best published accuracy. Easy ONNX export. |
| Custom 32-ch UNet | 192×192 | ~250 K | Train in 30 min on TEyeD. Two orders simpler to debug. ~2 ms Vulkan. |
| DeepVOG | 240×320 | ~700 K | Older Keras, but produces full 3D eyeball model. Heavy. |
| PupilNet (Fuhl et al.) | 24×24 | ~30 K | Pre-deep-learning, classifier-style. Useful baseline. |

### Pick

- **Primary: EllSeg-Gen → ONNX → burn-import → burn-wgpu.** Both mask + ellipse parameters in one pass.
- **Bring-up baseline: custom 32-ch UNet** for Phase 2 — strips one variable when debugging the inference path.

Both fit comfortably in 16 GB VRAM with batch=2 (binocular). Datasets: **TEyeD** (Tübingen, free), **OpenEDS-2020** (FB Reality Labs, registration required), **NVGAZE** (synthetic).

---

## 4. The Vulkan Path — Why This Wins

**Eliminated yak-shaving:**
- No ORT-from-source build (saves 90 min build + ABI breakage risk)
- No `HSA_OVERRIDE_GFX_VERSION=11.0.0` gfx1100/gfx1150 dance
- No custom MIOpen link path
- No ROCm-version-coupling (ROCm 7.2 vs 7.1 vs 7.0 all moot)
- No "is the EP supported on RDNA 3.5" verification

**What you trade:**
- ~30-50% slower convs on RDNA 3.5 (HIP/MIGraphX → Vulkan/wgpu). Real but invisible at our model size.
- Fewer specialized kernels (no MIGraphX graph fusion). Again, invisible at 1 GFLOP/frame.

**What you gain:**
- Same binary runs on AMD, NVIDIA, Intel, Apple (MoltenVK). The eye-tracker daemon is now a single-file artifact you can toss on any laptop.
- Rust-native end-to-end. No Python sidecar, no cmake archeology.
- Vulkan is *more stable* than ROCm on RDNA 3.5 right now — the gfx1150 ROCm path is still community-patched.
- `wgpu` is heavily maintained (Bevy, Firefox, Servo) and has aggressive validation layer support — easier to debug than HIP kernels.

**Vulkan stack on this machine:**
- Driver: RADV (mesa) is preferred. AMDVLK as fallback. Both on Strix Halo / gfx1150 are working in CachyOS as of April 2026.
- Validate at Phase 0: `vulkaninfo --summary` and `wgpu-info` should both report `AMD Radeon Graphics` with API ≥ 1.3.

---

## 5. Hardware Procurement

**Path A — EyeTrackVR canonical BOM** (recommended):

| Part | Qty | Source | US$ |
|---|---|---|---|
| Seeed XIAO ESP32-S3 Sense (OV2640 + antenna) | 2 | Seeed / DigiKey | $14 ea |
| 850 nm IR LED (Vishay TSAL6100, **non-focused**) | 4-6 | Mouser | $0.50 ea |
| ETVR IR LED PCB v4 (JLCPCB from Gerbers) | 2 | JLCPCB | $5 + ship |
| 3.7 V 100-200 mAh Li-Po | 1-2 | Adafruit | $5 ea |
| TP4056 charger module | 1 | AliExpress | $1 |
| Rocker switch + USB-C breakout | 1 | — | $3 |
| 3D-printed glasses-clip mount | 2 | print at home | ~$1 PLA |
| **Total** | | | **≈$50-70** |

**Long-lead item:** XIAOs from Seeed (10-day shipping). **Order today.** Print clips while waiting.

Skip Path B (HM01B0 + ESP32-S3 SuperMini) — saves $20 but doubles assembly time and OpenIris doesn't first-class support HM01B0.

---

## 6. Phased Implementation Plan

### Phase 0 — Vulkan inference bench (1 evening)
- New cargo workspace; one binary `bench-infer`.
- `cargo add burn burn-wgpu burn-import ndarray`
- Download EllSeg ONNX weights from `RSKothari/EllSeg` releases.
- Use `burn-import` to convert ONNX → Burn graph at compile time.
- Generate synthetic 320×240 grayscale tensor; run 1000 iterations on `burn-wgpu` Vulkan backend; log mean / p50 / p99 ms.
- **Pass criteria:** ≤12 ms mean, ≤20 ms p99 for single-eye 320×240 EllSeg. Binocular batch=2 should be ≤16 ms mean. If not, drop to the custom 32-ch UNet — if THAT fails, the Vulkan stack itself is broken and the Phase 0 deliverable becomes diagnosing wgpu/RADV.
- Bonus: try `wonnx` head-to-head as a sanity check — should produce identical output, slightly different perf.

### Phase 1 — Capture (1 evening)
- Flash both XIAOs with OpenIris via `EyeTrackVR/FirmwareFlashingTool`.
- Configure as `etvr-left` and `etvr-right`.
- Rust binary `capture-preview`: tokio UDP receiver per eye, JPEG SOI/EOI parser, decode via `image`, draw to egui texture. Side-by-side panes with FPS + latency overlay.
- **Pass criteria:** 60 fps preview, no tearing, ≤30 ms blink-to-pixel latency.

### Phase 2 — Inference (1-2 evenings)
- Combine Phase 0 + Phase 1 into `track-preview`. ROI-crop a 256×256 patch around the previous-frame's pupil centroid.
- Overlay predicted mask in red, fitted ellipse in green.
- **Pass criteria:** stable mask under blinks/head-movement/glasses-occlusion; ≥30 FPS sustained.

### Phase 3 — Geometry & 2D screen mapping (1 evening)
- `geom::EllipseFit` from EllSeg's regression head. Add LM refinement on mask boundary points.
- Glint detection (brightest CC within iris mask). Use glint-relative-to-pupil-centroid as the gaze feature (skip 3D eyeball model day 1).
- Implement 2D 9-point homography: `(gx_left, gy_left, gx_right, gy_right) → (screen_x, screen_y)`.
- **Pass criteria:** calibration math compiles + produces sane numbers on canned data.

### Phase 4 — Cursor injection (½ evening + Wayland yak-shaving)
- `output::cursor_sink_libei` using `reis` + `ashpd`. Open RemoteDesktop portal; send absolute pointer events.
- `output::cursor_sink_uinput` using `evdev` for `/dev/uinput`. Auto-detect: try libei, fall back to uinput.
- **Pass criteria:** cursor moves smoothly; KWin focus follows the synthetic cursor (proves it's a real cursor, not an overlay).

### Phase 5 — Calibration UX (1 evening)
- 9-point calibration in egui: fullscreen, dot moves through 3×3 grid, hold ≥1s on each. Compute homography per eye + binocular average.
- Save to `~/.config/eye-tracker-rust/calibration.toml`. Live-reload via `notify`.
- **Pass criteria:** ≤1° accuracy in central 60% of FOV (~40 px on 1080p at arm's length).

### Phase 6 — Binocular fusion (1 evening)
- Inference batched as `[2, 1, H, W]`.
- Confidence-weighted average. If one eye's confidence < 0.6, ignore (handles winks).
- **Pass criteria:** closing one eye doesn't jump the cursor.

### Phase 7 — Polish (1-2 evenings)
- One-Euro filter (`min_cutoff=1.0, beta=0.007`)
- Dwell-click: hold within 30 px circle for 0.7s → emit `BTN_LEFT`
- Hotkey pause via `evdev` global grab or libei keyboard listener
- Systemd user service for autostart
- MQTT/UDP broadcast sink for OBS overlays

**Total: 7-9 evenings of focused work** (~25-35 hours).

---

## 7. Risks & Escape Hatches

| # | Risk | Likelihood | Escape hatch |
|---|---|---|---|
| 1 | `burn-import` chokes on a specific EllSeg ONNX op | Medium | Swap that branch to `wonnx`, or train custom 32-ch UNet (Phase 2 fallback) — fewer ops, all common. |
| 2 | RADV driver bug on RDNA 3.5 trips wgpu validation layer | Low-Medium | Switch to AMDVLK (`AMD_VULKAN_ICD=AMDVLK`). Both ICDs ship on CachyOS. |
| 3 | KDE Plasma 6.6.4 RemoteDesktop portal gates libei behind a UI prompt that blocks autostart | Medium | uinput fallback. Note: `xdg-desktop-portal-kde` 6.4+ has "always allow" toggle. |
| 4 | OpenIris UDP latency > 30 ms on bad Wi-Fi | Medium-High | (a) dedicated 5 GHz SSID; (b) USB-tether one XIAO via CDC-ACM; (c) onboard inference on ESP32-S3 with TFLite Micro (emit only ellipse params, ~100 byte/frame). |
| 5 | Binocular fusion adds noise rather than reducing it | Low-Medium | Use monocular dominant-eye only — what Tobii Eye X did, works fine for cursor control. |
| 6 | IR LED current spec wrong → eye-safety violation | Low (if following EyeTrackVR docs) | Read EyeTrackVR safety docs. 5 mA/eye max, **non-focused** emitters. Verify with IR-camera selfie. |
| 7 | Phase 0 burn-wgpu perf below 12 ms target | Low (model is tiny) | First: validate `vulkaninfo` shows API 1.3 + correct device. Second: try `wonnx`. Third: drop to custom UNet. Fourth (last resort): Phase 8 = port to `ort+ROCm`. |

**Eliminated risks** (vs. the ROCm-first plan): ORT-from-source build failures, gfx1150 EP verification, MIOpen ABI breakage, Python sidecar IPC overhead, ROCm version-coupling. **All gone.**

---

## 8. Decision Checkpoints

### Checkpoint A — End of Phase 0
**Q:** Did `burn-wgpu` cleanly run EllSeg at ≤12 ms?
- Yes → continue Vulkan path confidently
- No, but `wonnx` works → switch to wonnx, plan one Phase 8 spike on `burn-wgpu` op coverage later
- Both fail → likely a wgpu/RADV/Vulkan stack issue. Run `vkcube`, `wgpu-info`. Try AMDVLK. If still broken, the Phase 0 deliverable becomes diagnosing the GPU stack.

### Checkpoint B — End of Phase 2
**Q:** Mask quality good? Ellipse fit ≤2 px noisy and stable through blinks?
- Yes → continue
- No → swap models. EllSeg-Gen → custom UNet on TEyeD. Or check BMVC 2025 for newer pupil-segmentation papers.
- Cursor jitter source ambiguous → add `tracing-flame` profiling **before** Phase 7 smoothing papers over a real bug.

### Checkpoint C — End of Phase 4
**Q:** Cursor responsive (≤50 ms blink-to-cursor) when used as actual mouse for 5 minutes?
- Yes → ship; rest is polish
- Lags → profile full pipeline. Usual culprit: hidden `tokio::sync::mutex` in hot path, or JPEG decode on the same thread as inference.
- Jumps unpredictably → calibration, not engineering. Tune Phase 5 before Phase 6.
- Works but feels alien → normal first 30 min of any eye-tracker. Use it for a day. If still alien at day 3, tune One-Euro filter, not architecture.

---

## 9. Repo Layout

```
eye-tracker-rust/
├── Cargo.toml                 # workspace
├── README.md
├── PLAN.md                    # this file
├── crates/
│   ├── etk-capture/           # tokio UDP + JPEG framing
│   ├── etk-preproc/           # crop / normalize / GPU upload
│   ├── etk-infer/             # burn-wgpu + wonnx behind Engine trait
│   ├── etk-geom/              # ellipse fit, glint, calibration math
│   ├── etk-output/            # libei + uinput sinks
│   ├── etk-config/            # serde + notify watcher
│   └── etk-gui/               # egui calibration + preview
├── apps/
│   ├── bench-infer/           # Phase 0 binary
│   ├── capture-preview/       # Phase 1 binary
│   ├── track-preview/         # Phase 2-3 binary
│   └── eye-tracker/           # final daemon
└── models/
    └── ellseg.onnx            # via Git LFS
```

No `sidecar/` directory — pure Rust end-to-end via Vulkan.

---

## 10. Phase-1 First Move

1. **Order hardware today.** XIAO ESP32-S3 Sense ×2 + IR LED PCB v4 Gerbers from JLCPCB — 7-day critical path.
2. **Phase 0 in parallel** — no waiting needed:
   - `vulkaninfo --summary` + `wgpu-info` → confirm AMD Radeon shows API 1.3
   - `cargo new --lib eye-tracker-rust && cd $_ && cargo add burn burn-wgpu burn-import`
   - Download EllSeg weights, write 60-line bench, log ms.
3. **Print glasses-clip mount** while shipping is in flight (designs in `EyeTrackVR/EyeTrackVR-Hardware/3d_Printed_Mounts`).
4. **Riskiest unknown** (now small): does `burn-import` cleanly load EllSeg-Gen's ONNX? Validate Phase 0 day 1 before writing any other Rust code. If it doesn't, the swap to `wonnx` is mechanical (same input/output tensors, different runtime).

---

## Critical Files & References

- `/home/raz/projects/eye-tracker-rust/PLAN.md` (this file)
- [`tracel-ai/burn`](https://github.com/tracel-ai/burn) — backend matrix, burn-wgpu docs
- [`webonnx/wonnx`](https://github.com/webonnx/wonnx) — fallback inference engine
- [`ids1024/reis`](https://github.com/ids1024/reis/tree/main/examples) — libei usage patterns
- [`RSKothari/EllSeg`](https://github.com/RSKothari/EllSeg) — model + ONNX export utility
- [`EyeTrackVR/OpenIris`](https://github.com/EyeTrackVR/OpenIris) — firmware + UDP protocol docs
- [`EyeTrackVR/EyeTrackVR-Hardware`](https://github.com/EyeTrackVR/EyeTrackVR-Hardware) — Gerbers + 3D-printed mounts
- `/home/raz/rust-vs-electron-gaze-tradeoffs.md` — companion trade-off analysis (Tauri default, Wayland-cursor-injection wins, etc.)

---

*Plan vetted 2026-04-29. Inference layer pivoted from ROCm to Vulkan after user signal — accepts ~30% conv perf delta in exchange for build-system simplicity, hardware portability, and elimination of all ROCm-version yak-shaving. Vulkan is the right v1 substrate; revisit ROCm in a hypothetical Phase 8 only if a real perf wall appears.*
