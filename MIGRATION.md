# Migration to tensorrt_cpp_api v7

This branch (`v7-migration`) ports YOLOv8-TensorRT-CPP from the v6 `tensorrt-cpp-api` to **v7**,
which is a clean break (new namespace `trtcpp`, no-throw `Status`/`Result`, name-keyed tensors,
PImpl headers with no OpenCV/TensorRT leakage). See the library's `docs/upgrading_from_v6.md`.

## Verification status

Built and run on an RTX 3080 Laptop GPU against a CUDA-12.6 OpenCV-CUDA build and the v7 library:
`detect_object_image` on `images/team.jpg` (FP16 YOLOv8n) runs end-to-end and detects the people in
the frame and writes the annotated image (~10 objects; the exact count shifts by one or two across
FP16 engine rebuilds, as TensorRT's tactic selection moves borderline detections across the
confidence threshold). The library, `opencv_interop`, the preproc sublib, and the YoloV8 detection
code compile and run. The video/CSI demo targets additionally need an OpenCV built with
`highgui`/`videoio`.

## Required after pulling: init the submodule

`libs/tensorrt-cpp-api` is a git submodule pinned to a v7 release; populate it before building:

```sh
git submodule update --init
```

## What changed

**Build (`CMakeLists.txt`)**
- C++17 → **C++20** (v7 requirement).
- Enable the library's OpenCV interop and preprocessing before `add_subdirectory`:
  `TRT_CPP_API_WITH_OPENCV=ON`, `TRT_CPP_API_BUILD_PREPROC=ON`.
- Link the namespaced v7 targets `tensorrt_cpp_api::tensorrt_cpp_api` + `tensorrt_cpp_api::preproc`
  (was `tensorrt_cpp_api`); dropped the `libs/.../src` include — v7 propagates its own include root.

**Inference layer (`src/yolov8.{h,cpp}`)**
- `Engine<float>` → `trtcpp::Engine` (non-templated; runtime `DType`). IO is now name-keyed; the
  class caches `m_inputName`/`m_outputNames`/`m_inputShape`/`m_outputShapes` and a reusable
  NCHW-float input `Tensor` plus a `Stream`.
- `Options` + `buildLoadNetwork(onnx, SUB, DIV, NORMALIZE)` → `BuildOptions` +
  `EngineBuilder::buildAndLoad(onnx, opts)`.
- Preprocessing: the v6 OpenCV `cvtColor` + `resizeKeepAspectRatioPadRightBottom` + in-engine
  HWC→NCHW/normalize is replaced by **one fused kernel**, `preproc::letterboxToTensor`, fed a
  zero-copy `trtcpp::opencv::viewOf(gpuImg)` device view (BGR→RGB via `swapRB`, letterbox pad
  right/bottom, `scale = 1/255`). Box-mapping ratio is unchanged.
- `runInference(GpuMat, nested-vectors)` + `Engine<float>::transformOutput` → `engine.infer(...)`
  returning name-keyed owning `Tensor`s; each output is read back with `toHost(stream)` (explicit
  D2H + sync) into a flat `std::vector<float>`. The detect/pose/seg **post-processing math is
  unchanged** — only how it obtains dims (`getOutputDims().d[i]` → cached `Shape[i]`) and the flat
  output buffers.
- Errors: v6 `bool`/exception checks → unwrap `Result`/`Status` (throwing a `std::runtime_error`
  with `.status().message()` to preserve this app's exception-based control flow).

**Precision (`src/cmd_line_util.h`)**
- `Precision::FP32/FP16/INT8` → `trtcpp::Precision::kFp32/kFp16/kInt8Qdq`.
- **INT8 caveat:** `kInt8Qdq` expects an explicit Q/DQ ONNX (no calibration data). The v6 flow
  (a calibration-image directory) maps to `kInt8CalibLegacy`, which is only available when the
  library is built against **TensorRT < 11** and requires constructing an `ICalibrator`
  (`tensorrt_cpp_api/calibrator.h`) and setting `BuildOptions.calibrator`. That wiring is **not**
  ported here — quantize to a QDQ ONNX, or restore a calibrator if you need legacy PTQ.

**Misc**
- Added `src/stopwatch.h` (a small `std::chrono` `preciseStopwatch`) to replace the timing utility
  v6 shipped in the engine library and v7 does not.
- The OpenCV modules the v6 `engine.h` pulled in transitively (`imgcodecs`, `videoio`, `highgui`)
  are now included explicitly where used.
