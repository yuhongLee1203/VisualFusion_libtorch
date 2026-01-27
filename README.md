# VisualFusion LibTorch

🔥 **Real-Time EO-IR Image Alignment and Fusion System with Deep Learning**

## 📋 Version Information

```
# PC / x86
pytorch=1.13.1
libtorch=1.13.1
cudnn=8
onnxruntime=1.18.0
tensorrt=8.4
cuda=11

# Jetson Orin NX
Pytorch = 2.5.0
CUDA = 12.6
cuDNN = 9.3
TensorRT = 10.3
Python = 3.10.12
```

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![LibTorch](https://img.shields.io/badge/LibTorch-1.13.1-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org/)

## 🚀 Overview

VisualFusion LibTorch is a high-performance computer vision system for **EO-IR (Electro-Optical/Infrared) image alignment and fusion**. It leverages deep learning models (SemLA) to detect corresponding feature points between EO-IR image pairs, computes robust homography matrices using RANSAC, and generates high-quality fused outputs with advanced edge-preserving algorithms.

### ✨ Key Features

- 🎯 **Deep Learning Feature Detection**: SemLA model for accurate keypoint detection and matching
- 🖼️ **EO-IR Image Fusion**: Seamless fusion with shadow enhancement and edge preservation
- 📐 **RANSAC Homography**: Robust estimation with outlier filtering
- 🎛️ **Homography Smoothing**: Temporal consistency with configurable smoothing parameters
- ⚙️ **Flexible Cropping**: Support for VideoCut and PictureCut parameters
- 📊 **Performance Timing**: Built-in profiling for each processing stage
- 🚄 **FP16 Support**: Half-precision inference for faster GPU performance

## 🏗️ Project Structure

```
VisualFusion/
├── tensorRT_nx/                 # TensorRT implementation for Jetson Orin NX (ARM64)
│   ├── main.cpp                 # Main processing pipeline
│   ├── CMakeLists.txt           # CMake build configuration
│   ├── gcc.sh                   # Build script
│   │
│   ├── config/                  # Configuration files
│   │   └── config.json          # Runtime configuration
│   │
│   ├── lib_image_fusion/        # Core computer vision libraries
│   │   ├── include/             # Header files
│   │   │   ├── app_config.h             # Configuration management
│   │   │   ├── app_utils.h              # Utility functions
│   │   │   ├── core_image_align_tensorrt.h   # TensorRT alignment
│   │   │   ├── core_image_fusion_trt.h       # TensorRT fusion
│   │   │   ├── homography_manager.h          # Homography smoothing
│   │   │   └── image_processor.h             # Main processor
│   │   └── src/                 # Implementation
│   │       ├── app_config.cpp              # Config loader
│   │       ├── app_utils.cpp               # Utilities implementation
│   │       ├── core_image_align_tensorrt.cpp   # TensorRT inference
│   │       ├── core_image_fusion_trt.cpp       # GPU fusion
│   │       ├── homography_manager.cpp          # Smoothing logic
│   │       └── image_processor.cpp             # Processing pipeline
│   │
│   ├── utils/                   # Utility modules
│   │   └── src/
│   │       └── util_timer.cpp   # Performance timing
│   │
│   ├── model/                   # Model files
│   │   └── NX/                  # Jetson NX specific models
│   │       ├── trt_Nx_fp16.engine     # FP16 alignment model
│   │       ├── trt_Nx_fp32.engine     # FP32 alignment model
│   │       ├── border_1_fusion.trt    # Fusion model (thin edge)
│   │       └── border_4_fusion.trt    # Fusion model (thick edge)
│   │
│   ├── nlohmann/                # JSON library
│   ├── build/                   # Build artifacts
│   ├── output/                  # Output directory
│   └── current_homo.json        # Cached homography
│
├── IR_Convert_v21_libtorch/    # LibTorch C++ implementation for PC (x86)
├── IR_Convert_v21_libtorch_nx/ # LibTorch C++ implementation for Jetson NX
├── Onnx/                        # ONNX Runtime implementation
├── tensorRT/                    # TensorRT implementation for PC (x86)
│
└── convert_to_libtorch/         # Model conversion utilities
    ├── export_to_jit_fp16.py    # PyTorch → LibTorch FP16
    ├── export_to_jit_fp32.py    # PyTorch → LibTorch FP32
    ├── export_to_onnx_fp16.py   # PyTorch → ONNX FP16
    ├── export_to_onnx_fp32.py   # PyTorch → ONNX FP32
    ├── export_to_tensorrt_fp16.py  # PyTorch → TensorRT FP16
    ├── export_to_tensorrt_fp32.py  # PyTorch → TensorRT FP32
    ├── pytorch2trt_fusion_v2.py    # Image Fusion Model → TensorRT
    ├── model_jit/               # SemLA model implementation
    └── reg.ckpt                 # Pretrained weights
```

## 🔧 Supported Inference Engines

| Engine | Status | Model Format | Precision | Device Support |
|--------|--------|--------------|-----------|----------------|
| **LibTorch** | ✅ Ready | `.zip` (TorchScript) | FP32/FP16 | CPU/CUDA |
| **ONNX Runtime** | ✅ Ready | `.onnx` | FP32/FP16 | CPU/CUDA |
| **TensorRT** | ✅ Ready | `.engine` | FP32/FP16 | CUDA |

## 📋 Requirements

### System Dependencies (PC / x86)
- **OS**: Ubuntu 20.04+ (tested on Ubuntu 20.04.6 LTS)
- **CPU**: Multi-core processor (x86 architecture)
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **GPU**: NVIDIA GPU with CUDA 11.x support

### System Dependencies (Jetson Orin NX)
- **OS**: NVIDIA JetPack
- **CPU**: ARM64 architecture
- **Memory**: 8GB+ shared memory
- **GPU**: Jetson Orin NX integrated GPU

### Software Dependencies

#### C++ Build Tools
- **GCC**: 9.0+
- **CMake**: 3.18+
- **OpenCV**: 4.5+

#### Python & Libraries (PC / x86)
- **Python**: 3.8+
- **PyTorch**: 1.13.1
- **ONNX**: 1.14+
- **onnxruntime**: 1.18.0
- **numpy**, **opencv-python**

#### GPU Libraries (PC / x86)
- **CUDA**: 11.x
- **cuDNN**: 8.x
- **TensorRT**: 8.4.x (for TensorRT backend)

#### Environment (Jetson Orin NX)
- **Python**: 3.10.12
- **PyTorch**: 2.5.0
- **CUDA**: 12.6
- **cuDNN**: 9.3
- **TensorRT**: 10.3

## 🛠️ Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd VisualFusion_libtorch
```

### 2. Install Python Dependencies

```bash
cd convert_to_libtorch
pip install torch==1.13.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
pip install onnx onnxruntime-gpu==1.18.0 opencv-python numpy
```

### 3. Build TensorRT NX Version

```bash
cd tensorRT_nx
bash gcc.sh && ./build/out
```

The build script will:
- Configure TensorRT libraries for Jetson Orin NX
- Compile C++ source files with CUDA support
- Link against TensorRT, CUDA, and OpenCV libraries
- Generate executable: `./build/out`

**Requirements**:
- Jetson Orin NX with JetPack installed
- TensorRT 10.3.x libraries
- CUDA 12.6
- cuDNN 9.3


## 📦 Model Conversion

The project supports multiple inference backends. Convert the pretrained model to your desired format:

### LibTorch (TorchScript)

#### FP16 Model (Recommended for GPU)
```bash
cd convert_to_libtorch
python export_to_jit_fp16.py
```
- **Input**: `reg.ckpt` (PyTorch checkpoint)
- **Output**: `../IR_Convert_v21_libtorch/model/SemLA_fp16.zip`
- **Format**: FP16 TorchScript
- **Use case**: GPU inference with Tensor Core acceleration
- **Pipeline**: PyTorch FP16 → LibTorch FP16 (direct export)

#### FP32 Model
```bash
cd convert_to_libtorch
python export_to_jit_fp32.py
```
- **Input**: `reg.ckpt`
- **Output**: `../IR_Convert_v21_libtorch/model/SemLA_fp32.zip`
- **Format**: FP32 TorchScript
- **Use case**: CPU inference or maximum precision

### ONNX Runtime

#### FP32 Model
```bash
cd convert_to_libtorch
python export_to_onnx_fp32.py
```
- **Output**: `../Onnx/model/SemLA_onnx_opset12_fp32.onnx`

#### FP16 Model
```bash
cd convert_to_libtorch
python export_to_onnx_fp16.py
```
- **Output**: `../Onnx/model/onnx_op12_fp16.onnx`

### TensorRT

#### FP32 Engine
```bash
cd convert_to_libtorch
python export_to_tensorrt_fp32.py
```
- **Pipeline**: PyTorch FP32 → ONNX FP32 → TensorRT FP32
- **Output**: `../tensorRT/model/GPU30s/trt_semla_fp32_op12.engine`

#### FP16 Engine
```bash
cd convert_to_libtorch
python export_to_tensorrt_fp16.py
```
- **Pipeline**: PyTorch FP32 → ONNX FP32 → TensorRT FP16 (using `trtexec --fp16`)
- **Output**: `../tensorRT/model/GPU30s/trt_semla_fp16_op12.engine`

**Requirements**: TensorRT conversion requires CUDA 11.x, cuDNN 8.x, and TensorRT 8.4.x libraries installed.

**Note**: TensorRT engines are GPU-specific and should be rebuilt when moving to different hardware.

### Fusion Model (TensorRT)

The image fusion algorithm is also exported as a TensorRT model for GPU acceleration. This replaces the CPU-based `core_image_fusion.cpp`.

#### Export Fusion Model

```bash
cd convert_to_libtorch
python pytorch2trt_fusion_v2.py export <edge_border> <height> <width>
```

**Parameters**:
- `edge_border`: Edge thickness (default: 4). **This value is fixed at export time and cannot be changed at runtime.**
- `height`: Image height (default: 240)
- `width`: Image width (default: 320)

**Examples**:
```bash
# Export with edge_border=4 (default, thicker edges)
python pytorch2trt_fusion_v2.py export 4

# Export with edge_border=1 (thinner edges)
python pytorch2trt_fusion_v2.py export 1

# Export with custom size
python pytorch2trt_fusion_v2.py export 4 240 320
```

**Output files** (in `../tensorRT_nx/model/NX/`):
- `border_<edge_border>_fusion.onnx` - ONNX intermediate file
- `border_<edge_border>_fusion.trt` - TensorRT engine (FP32)

**Configuration**: After exporting, update `config.json`:
```json
{
    "use_trt_fusion": true,
    "fusion_trt_engine": "./model/NX/border_4_fusion.trt"
}
```

**Note**: 
- The fusion model uses **fixed arithmetic operations** (Sobel convolution, shift, etc.), not learned weights
- FP32 precision is used (FP16 is unnecessary for fixed operations)
- To change edge thickness, re-export with a different `edge_border` value

## ⚙️ Configuration

All runtime parameters are configured via `config/config.json`:

### Basic Configuration

```json
{
    "input_dir": "/path/to/input",
    "output_dir": "/path/to/output",
    "output": true,
    
    "device": "cuda",
    "pred_mode": "fp16",
    "model_path": "/path/to/model/SemLA_fp16.zip"
}
```

#### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | string | Input directory containing EO-IR image pairs | `"./input"` |
| `output_dir` | string | Output directory for results | `"./output"` |
| `output` | boolean | Enable saving output images | `false` |
| `device` | string | Inference device: `"cuda"` or `"cpu"` | `"cpu"` |
| `pred_mode` | string | Precision mode: `"fp16"` or `"fp32"` | `"fp32"` |
| `model_path` | string | Path to model file (.zip for LibTorch) | `"./model/SemLA_fp32.zip"` |

**Important**: 
- When `pred_mode="fp16"`, use `SemLA_fp16.zip` model and `device="cuda"`
- When `pred_mode="fp32"`, use `SemLA_fp32.zip` model
- FP16 mode requires CUDA device
- Model is pre-converted to FP16/FP32 in Python, C++ loads it directly

### Image Processing

```json
{
    "pred_width": 320,
    "pred_height": 240,
    "output_width": 320,
    "output_height": 240,
    
    "VideoCut": true,
    "Vcut_x": 870,
    "Vcut_y": 235,
    "Vcut_w": 2020,
    "Vcut_h": 1680,
    
    "PictureCut": true,
    "Pcut_x": 220,
    "Pcut_y": 0,
    "Pcut_w": 1920,
    "Pcut_h": 1080
}
```

| Parameter | Description |
|-----------|-------------|
| `pred_width/height` | Input size for model inference (320x240 recommended) |
| `output_width/height` | Output image dimensions |
| `VideoCut` | Enable video frame cropping |
| `Vcut_x/y/w/h` | Video crop region (x, y, width, height) |
| `PictureCut` | Enable picture cropping before fusion |
| `Pcut_x/y/w/h` | Picture crop region |

### Fusion Settings

```json
{
    "fusion_shadow": true,
    "fusion_threshold_equalization": 128,
    "fusion_threshold_equalization_low": 72,
    "fusion_threshold_equalization_high": 192,
    "fusion_threshold_equalization_zero": 64,
    "use_trt_fusion": true,
    "fusion_trt_engine": "./model/NX/border_4_fusion.trt"
}
```

| Parameter | Description |
|-----------|-------------|
| `fusion_shadow` | Enable shadow enhancement (CPU fusion only) |
| `fusion_threshold_*` | Histogram equalization thresholds (CPU fusion only) |
| `use_trt_fusion` | Enable TensorRT GPU fusion model |
| `fusion_trt_engine` | Path to TensorRT fusion engine (`.trt`) |

**Important - Edge Border Configuration**:
- The `edge_border` parameter (edge thickness) is **fixed at model export time**
- It **cannot** be changed dynamically at runtime via config.json
- To change edge thickness, you must **re-export** the fusion model with a different `edge_border` value
- See [Fusion Model Conversion](#fusion-model-tensorrt) section for export instructions

### Homography & Alignment

```json
{
    "perspective_check": true,
    "perspective_distance": 6,
    "perspective_accuracy": 0.85,
    
    "align_distance_last": 15.0,
    "align_distance_line": 10.0,
    "align_angle_mean": 10.0,
    "align_angle_sort": 0.7,
    
    "smooth_max_translation_diff": 80.0,
    "smooth_max_rotation_diff": 0.05,
    "smooth_alpha": 0.05
}
```

| Parameter | Description |
|-----------|-------------|
| `perspective_check` | Enable perspective validation |
| `perspective_distance` | RANSAC inlier threshold (pixels, default: 6.0) |
| `perspective_accuracy` | Minimum inlier ratio (0.0-1.0) |
| `align_distance_*` | Feature alignment distance thresholds |
| `align_angle_*` | Angle-based filtering parameters |
| `smooth_*` | Temporal smoothing parameters |

## 🚀 Usage

### TensorRT NX Version

#### Prepare Input Data

Organize your EO-IR image pairs with `_EO` and `_IR` suffixes:

```
input/
├── scene_001_EO.jpg
├── scene_001_IR.jpg
├── scene_002_EO.jpg
├── scene_002_IR.jpg
...
```

Or video files:
```
input/
├── video_001_EO.mp4
├── video_001_IR.mp4
...
```

#### Run Inference

```bash
cd tensorRT_nx
./build/out 
```

#### Output

Results are saved to the configured `output_dir`:

```
output/
├── scene_001_EO.jpg    # Combined output (5 images: IR orig | EO orig | IR proc | EO warped | Fused)
├── scene_002_EO.jpg
...
```

CSV logs are also generated:
- `image_homo_errors.csv` - Homography errors for images
- `video_homo_errors.csv` - Homography errors for videos
- `itiming_log.csv` - Performance timing logs

## 🔍 Processing Pipeline

The system follows this processing flow:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Input Loading                                            │
│    - Read EO-IR image pairs from input directory            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 2. Video/Image Cropping (Optional)                          │
│    - Apply VideoCut if enabled                              │
│    - Apply PictureCut if enabled                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 3. Image Preprocessing                                      │
│    - Convert to grayscale                                   │
│    - Resize to pred_width × pred_height (320×240)           │
│    - Normalize to [0, 1]                                    │
│    - Convert to FP16 (if pred_mode="fp16")                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 4. Deep Learning Inference (SemLA Model)                    │
│    - Input: EO and IR grayscale images (320×240)            │
│    - Model: Pre-converted TorchScript (FP16/FP32)           │
│    - Output: Corresponding keypoint pairs (up to 1200)      │
│    - Precision: Matches pred_mode (FP16 or FP32)            │
│    - Device: CPU or CUDA                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 5. Homography Computation                                   │
│    - RANSAC with 6.0px threshold                            │
│    - Perspective validation                                 │
│    - Outlier filtering                                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 6. Homography Smoothing (Temporal Consistency)              │
│    - Check translation/rotation differences                 │
│    - Weighted average with previous frame                   │
│    - Fallback to previous on large jumps                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 7. Image Fusion (TensorRT GPU or CPU)                       │
│    - TRT Mode: Load fusion TRT engine, GPU processing       │
│    - CPU Mode: Sobel edge detection + blending              │
│    - Edge thickness fixed at model export (TRT)             │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│ 8. Output Generation                                        │
│    - Fused image                                            │
│    - Visualization with keypoints and matches               │
│    - Save to output directory                               │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Algorithm Details

### 1. SemLA Feature Matching

**Model**: Semantic Line Association (SemLA)
- **Input**: Pair of 320×240 grayscale images (EO, IR)
- **Output**: Corresponding keypoint coordinates (up to 1200 pairs)
- **Architecture**: CNN-based feature detector + matcher
- **Precision**: FP32 or FP16 (pre-converted in Python)

**Post-processing**:
- Filter out invalid points (0, 0)
- RANSAC outlier removal (6.0px threshold)
- Perspective validation (min 99% inliers)

### 2. Homography Computation

**Method**: RANSAC-based homography estimation
- **Algorithm**: `cv::findHomography()` with `RANSAC`
- **Threshold**: 6.0 pixels (configurable via `perspective_distance`)
- **Confidence**: 0.99 (99% confidence)
- **Min Inliers**: Configurable via `perspective_accuracy` (default 0.85)

**Validation**:
- Check inlier ratio
- Verify homography matrix validity
- Fallback to identity on failure

### 3. Homography Smoothing

**Purpose**: Temporal consistency across frames

**Algorithm**:
```cpp
// Extract translation
current_tx = H[0][2]
current_ty = H[1][2]

// Extract rotation (approximation)
current_rot = atan2(H[1][0], H[0][0])

// Check differences
if (|current_tx - prev_tx| > max_translation_diff ||
    |current_ty - prev_ty| > max_translation_diff ||
    |current_rot - prev_rot| > max_rotation_diff) {
    // Large jump detected, use previous homography
    H = prev_H
} else {
    // Smooth homography
    H = alpha * H + (1 - alpha) * prev_H
}
```

**Parameters**:
- `smooth_max_translation_diff`: Max allowed translation jump (pixels)
- `smooth_max_rotation_diff`: Max allowed rotation jump (radians)
- `smooth_alpha`: Smoothing factor (0.0 = fully smooth, 1.0 = no smoothing)

### 4. Image Fusion

The image fusion can be performed using either **CPU** or **TensorRT GPU** backend.

#### TensorRT GPU Fusion (Recommended)

When `use_trt_fusion: true` in config.json:

**Pipeline**:
1. **Load TensorRT Engine**: Pre-exported fusion model (`border_X_fusion.trt`)
2. **Input**: 
   - EO grayscale image [1, 1, H, W]
   - IR color image [1, 3, H, W]
3. **GPU Processing**:
   - Sobel edge detection with Gaussian blur
   - Shadow effect using shift operations (fixed `edge_border` from export)
   - Edge overlay on IR image
4. **Output**: Fused RGB image [1, 3, H, W]

**Advantages**:
- GPU-accelerated processing
- Consistent results across platforms
- No CPU-GPU data transfer overhead during fusion

**Limitations**:
- `edge_border` (edge thickness) is fixed at model export time
- Must re-export model to change edge thickness

#### CPU Fusion (Legacy)

When `use_trt_fusion: false` in config.json:

**Steps**:
1. **Warp IR image** using computed homography
2. **Edge detection** on EO image (Sobel-based)
3. **Shadow enhancement** (if `fusion_shadow: true`):
   - Histogram equalization with configurable thresholds
   - Adaptive contrast adjustment
4. **Blending**:
   - Edge-aware alpha composition
5. **Output**: Fused RGB image

**Note**: CPU fusion uses `core_image_fusion.cpp` and supports runtime `fusion_edge_border` configuration.


### Poor Alignment Quality

**Symptoms**: Misaligned or flickering fusion results

**Solutions**:
1. **Increase RANSAC threshold**:
   ```json
   {
     "perspective_distance": 15,
     "perspective_accuracy": 0.95
   }
   ```

2. **Adjust smoothing parameters**:
   ```json
   {
     "smooth_max_translation_diff": 50.0,
     "smooth_alpha": 0.1
   }
   ```

3. **Check input image quality**:
   - Ensure sufficient overlap between EO-IR pairs
   - Verify image focus and lighting

### FP16 vs FP32 Inconsistency

**Issue**: Different results between FP16 and FP32

**Cause**: Numerical precision differences

**Mitigation**:
- FP16/FP32 precision determined at model conversion (Python)
- Deterministic settings applied in both Python and C++
- TF32 disabled globally
- Expect small numerical differences (<0.1% typically)

## 🔬 Advanced Topics

### Custom Model Training

To train your own SemLA model:

1. Prepare dataset (EO-IR image pairs with ground truth)
2. Modify `convert_to_libtorch/model_jit/SemLA.py`
3. Train model
4. Export checkpoint: `reg.ckpt`
5. Convert to deployment format:
   ```bash
   python export_to_jit_fp16.py  # or fp32
   ```

### Real-time Video Processing

For live video streams:

1. Modify `main.cpp` to accept video stream input
2. Use `cv::VideoCapture`
3. Implement frame buffering
4. Consider FP16 mode for higher throughput

## 📖 API Reference

### C++ Core Classes

#### `core::AppConfig`

**Purpose**: Configuration management for loading and validating runtime parameters from JSON file.

**Constructor**:
```cpp
AppConfig()  // Default constructor
```

**Key Methods**:
```cpp
bool load(const std::string& config_path);
void show() const;
bool validate() const;
```

**Key Members**:
- **Input/Output**: `input_dir`, `output_dir`, `output_enabled`
- **Image Size**: `pred_width`, `pred_height`, `output_width`, `output_height`
- **Cropping**: `video_cut_enabled`, `vcut_x/y/w/h`, `picture_cut_enabled`, `pcut_x/y/w/h`
- **Model**: `device`, `pred_mode`, `model_path`
- **Fusion**: `use_trt_fusion`, `fusion_trt_engine`, `fusion_edge_border`
- **Alignment**: `align_distance_last`, `align_angle_mean`, `align_angle_sort`
- **Smoothing**: `smooth_max_translation_diff`, `smooth_max_rotation_diff`, `smooth_alpha`
- **Pipeline**: `align_start_frame`, `align_stop_frame`, `align_on_first_frame`

---

#### `core::ImageAlignTensorRT`

**Purpose**: TensorRT-based feature matching and homography estimation using SemLA model.

**Constructor**:
```cpp
static std::shared_ptr<ImageAlignTensorRT> create_instance(const Param& param);
```

**Param Structure**:
```cpp
struct Param {
    int pred_width = 320;
    int pred_height = 240;
    std::string engine_path;
    std::string pred_mode = "fp32";  // "fp16" or "fp32"
    
    Param& set_size(int pw, int ph, int ow, int oh);
    Param& set_engine(const std::string& path);
    Param& set_pred_mode(const std::string& mode);
};
```

**Key Methods**:
```cpp
void align(const cv::Mat& eo, const cv::Mat& ir,
           std::vector<cv::Point2i>& eo_pts,
           std::vector<cv::Point2i>& ir_pts,
           cv::Mat& H);
```
- **Input**: EO and IR grayscale images (320×240, CV_8UC1)
- **Output**: 
  - `eo_pts`: Keypoints in EO image
  - `ir_pts`: Corresponding keypoints in IR image
  - `H`: Homography matrix (3×3)
- **Features**: TensorRT inference, RANSAC homography computation

```cpp
void set_current_image_name(const std::string& image_name);
```
- Sets image name for logging purposes

---

#### `core::ImageFusionTRT`

**Purpose**: GPU-accelerated image fusion using TensorRT.

**Constructor**:
```cpp
ImageFusionTRT(Param param);
static ptr create_instance(Param param);
```

**Param Structure**:
```cpp
struct Param {
    std::string engine_path = "";
    int width = 320;
    int height = 240;
    
    Param& set_engine_path(const std::string& path);
    Param& set_size(int w, int h);
};
```

**Key Methods**:
```cpp
cv::Mat fusion(const cv::Mat& eo_gray, const cv::Mat& ir_color);
```
- **Input**: 
  - `eo_gray`: Grayscale EO image (CV_8UC1)
  - `ir_color`: Color IR image (CV_8UC3)
- **Output**: Fused color image (CV_8UC3)
- **Features**: Sobel edge detection, shadow enhancement, GPU processing

```cpp
cv::Mat edge(const cv::Mat& eo_gray);
```
- **Input**: Grayscale EO image
- **Output**: Edge map
- **Note**: Edge detection only (no fusion)

```cpp
bool is_initialized() const;
```
- Returns initialization status

---

#### `core::HomographyManager`

**Purpose**: Temporal smoothing and validation of homography matrices for video sequences.

**Constructor**:
```cpp
HomographyManager(double max_trans_diff = 30.0,
                  double max_rot_diff = 0.03,
                  double alpha = 0.05);
```

**Key Methods**:
```cpp
cv::Mat update(const cv::Mat& new_homo);
```
- **Input**: New homography matrix
- **Output**: Smoothed homography (weighted average with previous)
- **Algorithm**: 
  - Checks translation/rotation difference
  - Falls back to previous if jump too large
  - Otherwise applies: `H_smooth = alpha * H_new + (1-alpha) * H_prev`

```cpp
std::pair<double, double> calculate_difference(const cv::Mat& homo1, 
                                               const cv::Mat& homo2) const;
```
- **Input**: Two homography matrices
- **Output**: `<translation_diff, rotation_diff>`

```cpp
cv::Mat get_current() const;
void reset();
void set_parameters(double max_trans_diff, double max_rot_diff, double alpha);
```

---

#### `core::ImageProcessor`

**Purpose**: Main processing pipeline for image/video fusion.

**Constructor**:
```cpp
explicit ImageProcessor(const AppConfig& config);
```

**Key Methods**:
```cpp
bool initialize();
```
- Initializes TensorRT modules, homography manager, and timers
- **Returns**: `true` if successful

```cpp
bool process_image(const std::string& eo_path,
                   const std::string& ir_path,
                   const std::string& save_path);
```
- **Input**: Paths to EO/IR image pair
- **Output**: Saves combined result (5 images horizontal)
- **Pipeline**: Load → Crop → Resize → Align → Warp → Fuse → Save
- **Returns**: `true` if successful

```cpp
bool process_video(const std::string& eo_path,
                   const std::string& ir_path,
                   const std::string& save_path);
```
- **Input**: Paths to EO/IR video pair
- **Output**: Saves video with fusion results
- **Features**: Frame skipping, adaptive alignment, progress logging
- **Returns**: `true` if successful

```cpp
void show_timer_results();
```
- Displays performance statistics (resize, align, fusion times)

---

### Utility Functions (`core::utils`)

**File Operations**:
```cpp
bool is_file_exist(const std::string& path);
bool is_dir_exist(const std::string& path);
bool is_video(const std::string& path);
bool get_pair_paths(const std::string& path, 
                    std::string& eo_path, 
                    std::string& ir_path);
std::string extract_file_name(const std::string& path);
std::string extract_base_name(const std::string& path);
```

**Image Processing**:
```cpp
cv::Mat crop_image(const cv::Mat& src, int x, int y, int w, int h);
cv::Mat warp_with_homography(const cv::Mat& src, const cv::Mat& M, 
                              const cv::Size& size, int interp = cv::INTER_LINEAR);
cv::Mat combine_images_horizontal(const std::vector<cv::Mat>& images);
```

**Homography Operations**:
```cpp
cv::Mat refine_homography_with_ransac(std::vector<cv::Point2i>& eo_pts,
                                      std::vector<cv::Point2i>& ir_pts,
                                      const cv::Mat& initial_H,
                                      double ransac_threshold = 6.0);
```
- **Input/Output**: Keypoint vectors (modified in-place, outliers removed)
- **Returns**: Refined homography matrix

```cpp
bool save_homography_to_cache(const std::string& cache_file_path, const cv::Mat& H);
cv::Mat load_homography_from_cache(const std::string& cache_file_path);
bool is_homography_cache_exists(const std::string& cache_file_path);
```

**Error Calculation**:
```cpp
cv::Mat read_gt_homography(const std::string& gt_path, const std::string& img_name);
cv::Mat read_gt_homography_for_frame(const std::string& video_name,
                                     int frame_number,
                                     const std::string& gt_base_path);
double calc_feature_point_mse(const cv::Mat& homo_pred,
                              const cv::Mat& homo_gt,
                              const std::vector<cv::Point2i>& eo_pts);
void write_error_to_csv(const std::string& filename,
                        const std::string& name,
                        double error,
                        const std::vector<std::pair<std::string, std::string>>& extra_cols = {});
```

---

### Timer (`util::Timer`)

**Key Methods**:
```cpp
void start();
void stop();
void show() const;
```
- Tracks execution time with statistics (total, average, min, max, count)

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Complete TensorRT integration for more platforms
- Real-time video streaming support
- Performance optimizations (CUDA kernels, memory management)
- Additional fusion algorithms
- Documentation improvements
- Custom model training pipelines


## 🙏 Acknowledgments

- [OpenCV](https://opencv.org/) - Computer vision primitives
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [LibTorch](https://pytorch.org/cppdocs/) - C++ frontend for PyTorch
- [ONNX](https://onnx.ai/) - Model interoperability
- [TensorRT](https://developer.nvidia.com/tensorrt) - High-performance inference
- SemLA research team for feature matching algorithm

---

<div align="center">
  <sub>Built with ❤️ for computer vision research and applications</sub>
</div>

