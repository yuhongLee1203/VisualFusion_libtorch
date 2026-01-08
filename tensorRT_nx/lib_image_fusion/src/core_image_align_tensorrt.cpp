#include "../include/core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <experimental/filesystem>
#include <cuda_fp16.h>  // 添加 FP16 支援

// 支援 FP32, FP16, INT8 三種精度模式
// INT8 模式：輸入使用 FP32，量化在 TensorRT engine 內部進行

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override
  {
    // Suppress verbose logging
    if (severity <= Severity::kERROR) {
      std::cout << msg << std::endl;
    }
  }
};

namespace core {

// PIMPL Idiom: Implementation class
class ImageAlignTensorRTImpl : public ImageAlignTensorRT {
private:
  Param param_;
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;

  // CSV logging for inference time
  std::ofstream csv_file_;
  std::string current_image_name_ = "";  // 當前圖片名稱

  // Helper to load engine from file
  bool load_engine(const std::string& engine_path)
  {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "ERROR: Could not open engine file: " << engine_path << std::endl;
      return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
      std::cerr << "ERROR: Could not read engine file." << std::endl;
      return false;
    }

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
      std::cerr << "ERROR: Failed to create TensorRT runtime." << std::endl;
      return false;
    }

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size());
    if (!engine_) {
      std::cerr << "ERROR: Failed to deserialize CUDA engine." << std::endl;
      return false;
    }

    return true;
  }

public:
  // Constructor
  ImageAlignTensorRTImpl(const Param& param) : ImageAlignTensorRT(param), param_(param)
  {
    // Engine 建立時 NVIDIA_TF32_OVERRIDE=0，執行時也必須設置為 0
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);

    printf("Model initialization completed\n");

    // 初始化 CSV 文件（與 LibTorch 版本一致）
    std::string csv_filename = "./itiming_log.csv";
    bool file_exists = std::experimental::filesystem::exists(csv_filename);
    csv_file_.open(csv_filename, std::ios::app);

    if (!file_exists && csv_file_.is_open()) {
      // 寫入CSV標頭（與 LibTorch 格式一致）
      csv_file_ << "Filename,Operation,Time_s,Mode,keypoints\n";
    }

    if (!load_engine(param_.engine_path)) {
      throw std::runtime_error("Failed to load TensorRT engine.");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
      throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream.");
    }

    // 智能 warmup: 執行 warmup 來初始化 CUDA kernels，但不保留內部狀態
    printf("Performing smart warmup to initialize CUDA kernels...\n");
    smart_warmup_tensorrt();
  }

  // Destructor
  ~ImageAlignTensorRTImpl()
  {
    if (csv_file_.is_open()) {
      csv_file_.close();
    }
    if (stream_) cudaStreamDestroy(stream_);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
  }

  // Smart warmup for TensorRT: 初始化 CUDA kernels 但保持精度
  void smart_warmup_tensorrt()
  {
    printf("Smart warmup for CUDA kernel initialization...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

    const auto t0 = std::chrono::high_resolution_clock::now();

    // 只執行一次推理來初始化 CUDA kernels
    std::vector<cv::Point2i> dummy_eo_mkpts, dummy_ir_mkpts;
    pred(eo, ir, dummy_eo_mkpts, dummy_ir_mkpts);

    // 重新創建 execution context 以清除內部狀態，保持第一次推理的精度
    printf("Recreating TensorRT execution context to maintain first-inference precision...\n");
    if (context_) delete context_;
    context_ = engine_->createExecutionContext();
    if (!context_) {
      throw std::runtime_error("Failed to recreate TensorRT execution context");
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Smart warmup completed in %.2f s\n", period);
  }

  // 更新圖片名稱的方法
  void set_current_image_name(const std::string& image_name)
  {
    current_image_name_ = image_name;
  }

  // Main alignment function
  void align(const cv::Mat& eo, const cv::Mat& ir,
             std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts,
             cv::Mat& H) override
  {
    // predict keypoints
    pred(eo, ir, eo_pts, ir_pts);

    // CORRECTED: 與Python代碼完全一致的特徵點縮放條件檢查
    if (std::abs(param_.out_width_scale - 1.0) > 1e-6 ||
        std::abs(param_.out_height_scale - 1.0) > 1e-6 ||
        param_.bias_x > 0 || param_.bias_y > 0) {
      for (cv::Point2i& i : eo_pts) {
        i.x = i.x * param_.out_width_scale;
        i.y = i.y * param_.out_height_scale;
      }
      for (cv::Point2i& i : ir_pts) {
        i.x = i.x * param_.out_width_scale;
        i.y = i.y * param_.out_height_scale;
      }

      std::cout << "  - Feature point scaling applied: pred(" << param_.pred_width << "x" << param_.pred_height
                << ") -> out(" << param_.out_width_scale * param_.pred_width << "x" << param_.out_height_scale * param_.pred_height
                << "), scale=(" << param_.out_width_scale << ", " << param_.out_height_scale
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    } else {
      std::cout << "  - No feature point scaling needed: scale=(" << param_.out_width_scale << ", " << param_.out_height_scale
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    }

    // 輸出最終特徵點數量
    std::cout << "  - Final feature points after coordinate adjustment: " << eo_pts.size() << std::endl;
  }

  // Pre-processing and inference prediction
  void pred(const cv::Mat& eo, const cv::Mat& ir,
            std::vector<cv::Point2i>& eo_mkpts, std::vector<cv::Point2i>& ir_mkpts)
  {
    // 1. Pre-processing
    cv::Mat eo_resized, ir_resized;
    cv::resize(eo, eo_resized, cv::Size(param_.pred_width, param_.pred_height));
    cv::resize(ir, ir_resized, cv::Size(param_.pred_width, param_.pred_height));

    cv::Mat eo_gray, ir_gray;

    // 檢查圖像是否已經是灰度圖
    if (eo_resized.channels() == 3) {
      cv::cvtColor(eo_resized, eo_gray, cv::COLOR_BGR2GRAY);
    } else {
      eo_gray = eo_resized.clone();
    }

    if (ir_resized.channels() == 3) {
      cv::cvtColor(ir_resized, ir_gray, cv::COLOR_BGR2GRAY);
    } else {
      ir_gray = ir_resized.clone();
    }

    // CORRECTED: 完全按照Python代碼邏輯進行處理
    // Python: img0_tensor = torch.from_numpy(eo_pred)[None][None].to(device, dtype=fpMode) / 255.

    // 1. 確保圖像是uint8格式（對應Python numpy array）
    cv::Mat eo_uint8, ir_uint8;
    eo_gray.convertTo(eo_uint8, CV_8U);
    ir_gray.convertTo(ir_uint8, CV_8U);

    cv::Mat eo_float, ir_float;
    eo_gray.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
    ir_gray.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

    // 根據 pred_mode 決定使用 FP32、FP16 或 INT8 輸入
    // 注意：INT8 模式下，輸入仍使用 FP32，量化在 TensorRT engine 內部進行
    bool use_fp16 = (param_.pred_mode == "fp16");
    bool use_int8 = (param_.pred_mode == "int8");
    std::string mode_str = use_int8 ? "INT8 (FP32 input)" : (use_fp16 ? "FP16" : "FP32");
    std::cout << "debug: Using " << mode_str << " mode (pred_mode=" << param_.pred_mode << ")" << std::endl;

    int leng = 0;
    bool success = false;

    // 計時 - 模型推論開始

    // ===== FP32/INT8 模式：使用 FP32 輸入 =====
    // 注意：即使是 INT8 engine，輸入數據也使用 FP32
    // TensorRT 會在 engine 內部自動進行 INT8 量化
    std::vector<float> eo_data(param_.pred_width * param_.pred_height);
    std::vector<float> ir_data(param_.pred_width * param_.pred_height);

    // 將FP32圖像數據拷貝到向量中
    memcpy(eo_data.data(), eo_float.data, eo_data.size() * sizeof(float));
    memcpy(ir_data.data(), ir_float.data, ir_data.size() * sizeof(float));

    auto model_inference_start = std::chrono::high_resolution_clock::now();
    // run the model with FP32 input (適用於 FP32, FP16, INT8 所有 engine 類型)
    success = run_inference(eo_data, ir_data, eo_mkpts, ir_mkpts, leng);
    // 計時 - 模型推論結束
    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        model_inference_end - model_inference_start).count() / 1000000.0;  // 轉換為秒

    if (!success) {
      eo_mkpts.clear();
      ir_mkpts.clear();
      return;
    }

    // DEBUG: 確認 leng 值和實際特徵點數量
    printf("[DEBUG] leng from model: %d, eo_mkpts.size(): %lu\n", leng, eo_mkpts.size());

    // 寫入CSV檔案 - 只記錄模型推論時間
    write_timing_to_csv("Model_Inference", model_inference_time, leng, current_image_name_);

    printf("Model inference time: %.6f s\n", model_inference_time);

    // DEBUG: 輸出特徵點數量
    std::cout << "  - Model extracted " << eo_mkpts.size() << " feature point pairs" << std::endl;
  }

  // The core inference logic - 簡化版本：只讀取 mkpt0 和 mkpt1
  bool run_inference(const std::vector<float>& eo_data, const std::vector<float>& ir_data,
                     std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps,
                     int& leng1)
  {
    const int32_t num_io_tensors = engine_->getNbIOTensors();
    // 新模型只有 2 inputs + 2 outputs
    if (num_io_tensors != 4) {
      std::cerr << "ERROR: Expected 4 IO tensors (2 inputs + 2 outputs), but got " << num_io_tensors << std::endl;
      return false;
    }

    std::vector<void*> buffers(num_io_tensors);

    // Get tensor names and allocate GPU buffers
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      auto dims = engine_->getTensorShape(tensor_name);
      nvinfer1::DataType dtype = engine_->getTensorDataType(tensor_name);
      size_t element_size = 0;

      switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
          element_size = sizeof(float);
          break;
        case nvinfer1::DataType::kHALF:  // 支援 FP16 engine
          element_size = sizeof(__half);
          break;
        case nvinfer1::DataType::kINT32:
          element_size = sizeof(int32_t);
          break;
        default:
          std::cerr << "ERROR: Unsupported data type for tensor " << tensor_name << std::endl;
          for (int j = 0; j < i; ++j) {
            cudaFree(buffers[j]);
          }
          return false;
      }

      size_t size = 1;
      for (int32_t d = 0; d < dims.nbDims; ++d) {
        size *= dims.d[d];
      }
      size *= element_size;

      if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
        std::cerr << "ERROR: Failed to allocate GPU memory for tensor " << tensor_name << std::endl;
        for (int j = 0; j < i; ++j) {
          cudaFree(buffers[j]);
        }
        return false;
      }
    }

    // Find tensor indices
    int eo_img_idx = -1, ir_img_idx = -1, mkpt0_idx = -1, mkpt1_idx = -1;
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      if (std::string(tensor_name) == "vi_img") eo_img_idx = i;
      else if (std::string(tensor_name) == "ir_img") ir_img_idx = i;
      else if (std::string(tensor_name) == "mkpt0") mkpt0_idx = i;
      else if (std::string(tensor_name) == "mkpt1") mkpt1_idx = i;
    }

    if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0) {
      std::cerr << "ERROR: Could not find required tensors" << std::endl;
      for (void* buf : buffers) {
        cudaFree(buf);
      }
      return false;
    }

    // Set tensor addresses
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      context_->setTensorAddress(tensor_name, buffers[i]);
    }

    // Copy input data to GPU (支援 FP32 輸入到 FP16 engine 的自動轉換)
    const char* eo_tensor_name = engine_->getIOTensorName(eo_img_idx);
    nvinfer1::DataType input_dtype = engine_->getTensorDataType(eo_tensor_name);

    if (input_dtype == nvinfer1::DataType::kHALF) {
      // Engine 是 FP16，需要將 FP32 輸入轉換為 FP16
      std::cout << "debug: Converting FP32 input to FP16 for TRT FP16 engine" << std::endl;

      std::vector<__half> eo_data_fp16(eo_data.size());
      std::vector<__half> ir_data_fp16(ir_data.size());

      for (size_t i = 0; i < eo_data.size(); i++) {
        eo_data_fp16[i] = __float2half(eo_data[i]);
        ir_data_fp16[i] = __float2half(ir_data[i]);
      }

      cudaMemcpyAsync(buffers[eo_img_idx], eo_data_fp16.data(), eo_data_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(buffers[ir_img_idx], ir_data_fp16.data(), ir_data_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
    } else {
      // Engine 是 FP32，直接複製
      cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
    }

    // Execute model
    if (!context_->enqueueV3(stream_)) {
      std::cerr << "ERROR: Failed to execute TensorRT model" << std::endl;
      for (void* buf : buffers) {
        cudaFree(buf);
      }
      return false;
    }

    // Prepare output buffers
    const char* mkpt0_tensor_name = engine_->getIOTensorName(mkpt0_idx);
    auto mkpt0_dims = engine_->getTensorShape(mkpt0_tensor_name);
    size_t mkpt0_count = 1;
    for (int32_t d = 0; d < mkpt0_dims.nbDims; ++d) {
      mkpt0_count *= mkpt0_dims.d[d];
    }
    std::vector<int32_t> eo_kps_raw(mkpt0_count);

    const char* mkpt1_tensor_name = engine_->getIOTensorName(mkpt1_idx);
    auto mkpt1_dims = engine_->getTensorShape(mkpt1_tensor_name);
    size_t mkpt1_count = 1;
    for (int32_t d = 0; d < mkpt1_dims.nbDims; ++d) {
      mkpt1_count *= mkpt1_dims.d[d];
    }
    std::vector<int32_t> ir_kps_raw(mkpt1_count);

    // Copy outputs from GPU
    cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    // Parse output dimensions
    int num_keypoints = 0;
    int num_coords = 2;

    if (mkpt0_dims.nbDims == 3) {  // [1, 1200, 2]
      num_keypoints = mkpt0_dims.d[1];
      num_coords = mkpt0_dims.d[2];
    } else if (mkpt0_dims.nbDims == 2) {  // [1200, 2]
      num_keypoints = mkpt0_dims.d[0];
      num_coords = mkpt0_dims.d[1];
    } else {
      return false;
    }

    if (num_coords != 2) {
      return false;
    }

    eo_kps.clear();
    ir_kps.clear();

    // 讀取所有點，跳過座標為 (0, 0) 的點
    for (int i = 0; i < num_keypoints; ++i) {
      int x_eo = static_cast<int>(eo_kps_raw[i * num_coords + 0]);
      int y_eo = static_cast<int>(eo_kps_raw[i * num_coords + 1]);

      int x_ir = static_cast<int>(ir_kps_raw[i * num_coords + 0]);
      int y_ir = static_cast<int>(ir_kps_raw[i * num_coords + 1]);

      // 跳過原點座標（無效點）
      if (x_eo == 0 && y_eo == 0) {
        continue;
      }

      eo_kps.emplace_back(x_eo, y_eo);
      ir_kps.emplace_back(x_ir, y_ir);
    }

    leng1 = eo_kps.size();

    for (void* buf : buffers) {
      cudaFree(buf);
    }

    return true;
  }

  // FP16 inference function - 簡化版本：只讀取 mkpt0 和 mkpt1
  bool run_inference_fp16(const std::vector<__half>& eo_data, const std::vector<__half>& ir_data,
                          std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps,
                          int& leng1)
  {
    const int32_t num_io_tensors = engine_->getNbIOTensors();
    if (num_io_tensors != 4) {  // 2 inputs + 2 outputs
      std::cerr << "debug: ERROR: Expected 4 IO tensors for FP16, but got " << num_io_tensors << std::endl;
      return false;
    }

    std::vector<void*> buffers(num_io_tensors);

    // Get tensor names and allocate GPU buffers
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      auto dims = engine_->getTensorShape(tensor_name);
      nvinfer1::DataType dtype = engine_->getTensorDataType(tensor_name);
      size_t element_size = 0;
      std::string type_str = "UNKNOWN";

      // Determine element size based on data type
      switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
          element_size = sizeof(float);
          type_str = "FLOAT";
          break;
        case nvinfer1::DataType::kHALF:
          element_size = sizeof(__half);
          type_str = "HALF";
          break;
        case nvinfer1::DataType::kINT32:
          element_size = sizeof(int32_t);
          type_str = "INT32";
          break;
        default:
          std::cerr << "debug: ERROR: Unsupported data type for FP16 tensor " << tensor_name << " (Type: " << static_cast<int>(dtype) << ")" << std::endl;
          // Free already allocated buffers before returning
          for (int j = 0; j < i; ++j) cudaFree(buffers[j]);
          return false;
      }
      std::cout << "debug: [run_inference_fp16] Tensor: " << i << ", Name: " << tensor_name << ", Type: " << type_str << std::endl;

      size_t size = 1;
      for (int32_t d = 0; d < dims.nbDims; ++d) {
        size *= dims.d[d];
      }
      size *= element_size;

      if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
        std::cerr << "debug: ERROR: CUDA memory allocation failed for FP16 tensor " << i << " (" << tensor_name << ")" << std::endl;
        // Free already allocated buffers before returning
        for (int j = 0; j < i; ++j) cudaFree(buffers[j]);
        return false;
      }
    }

    // Find tensor indices - 只需要 2 inputs + 2 outputs
    int eo_img_idx = -1, ir_img_idx = -1, mkpt0_idx = -1, mkpt1_idx = -1;
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      if (std::string(tensor_name) == "vi_img") eo_img_idx = i;
      else if (std::string(tensor_name) == "ir_img") ir_img_idx = i;
      else if (std::string(tensor_name) == "mkpt0") mkpt0_idx = i;
      else if (std::string(tensor_name) == "mkpt1") mkpt1_idx = i;
    }

    if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0) {
      std::cerr << "debug: ERROR: Could not find required bindings for FP16." << std::endl;
      for (void* buf : buffers) cudaFree(buf);
      return false;
    }

    // Set tensor addresses
    for (int32_t i = 0; i < num_io_tensors; ++i) {
      const char* tensor_name = engine_->getIOTensorName(i);
      context_->setTensorAddress(tensor_name, buffers[i]);
    }

    // Copy FP16 input data to GPU
    cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);

    // Execute model
    std::cout << "debug: Executing FP16 model..." << std::endl;
    if (!context_->enqueueV3(stream_)) {
      std::cerr << "debug: ERROR: Failed to enqueue FP16 inference." << std::endl;
      return false;
    }

    // Prepare output buffers
    const char* mkpt0_tensor_name = engine_->getIOTensorName(mkpt0_idx);
    auto mkpt0_dims = engine_->getTensorShape(mkpt0_tensor_name);
    size_t mkpt0_count = 1;
    for (int32_t d = 0; d < mkpt0_dims.nbDims; ++d) {
      mkpt0_count *= mkpt0_dims.d[d];
    }
    std::vector<int32_t> eo_kps_raw(mkpt0_count);

    const char* mkpt1_tensor_name = engine_->getIOTensorName(mkpt1_idx);
    auto mkpt1_dims = engine_->getTensorShape(mkpt1_tensor_name);
    size_t mkpt1_count = 1;
    for (int32_t d = 0; d < mkpt1_dims.nbDims; ++d) {
      mkpt1_count *= mkpt1_dims.d[d];
    }
    std::vector<int32_t> ir_kps_raw(mkpt1_count);

    // Copy outputs from GPU
    cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);

    // Wait for all CUDA operations to complete
    cudaStreamSynchronize(stream_);
    std::cout << "debug: FP16 model execution and data copy complete." << std::endl;

    // Parse output dimensions
    int num_keypoints = 0;
    int num_coords = 2;  // Always 2 for (x, y)

    if (mkpt0_dims.nbDims == 3) {  // Expected case: [1, 1200, 2]
      num_keypoints = mkpt0_dims.d[1];
      num_coords = mkpt0_dims.d[2];
    } else if (mkpt0_dims.nbDims == 2) {  // Fallback for [1200, 2]
      num_keypoints = mkpt0_dims.d[0];
      num_coords = mkpt0_dims.d[1];
    } else {
      std::cerr << "debug: ERROR: Unexpected number of dimensions for FP16 keypoints: " << mkpt0_dims.nbDims << std::endl;
      return false;
    }

    if (num_coords != 2) {
      std::cerr << "debug: ERROR: Expected 2 coordinates per FP16 keypoint, but got " << num_coords << std::endl;
      return false;
    }

    eo_kps.clear();
    ir_kps.clear();
    std::cout << "debug: [run_inference_fp16] Parsing all " << num_keypoints << " keypoints, filtering (0,0)..." << std::endl;

    // 讀取所有點，跳過座標為 (0, 0) 的點
    for (int i = 0; i < num_keypoints; ++i) {
      int x_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 0]));
      int y_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 1]));

      int x_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 0]));
      int y_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 1]));

      // 跳過原點座標（無效點）
      if (x_eo == 0 && y_eo == 0) {
        continue;
      }

      eo_kps.emplace_back(x_eo, y_eo);
      ir_kps.emplace_back(x_ir, y_ir);
    }

    leng1 = eo_kps.size();

    std::cout << "debug: [run_inference_fp16] After filtering. Valid keypoints: " << eo_kps.size() << std::endl;

    for (int i = 0; i < std::min(5, (int)eo_kps.size()); ++i) {
      std::cout << "debug: [run_inference_fp16] Valid KP " << i << ": EO(" << eo_kps[i].x << "," << eo_kps[i].y
                << "), IR(" << ir_kps[i].x << "," << ir_kps[i].y << ")" << std::endl;
    }

    // Free GPU buffers
    for (void* buf : buffers) {
      cudaFree(buf);
    }

    return true;
  }

  // 寫入計時資料到CSV檔案
  void write_timing_to_csv(const std::string& operation, double time_s, int leng, const std::string& filename)
  {
    std::string csv_filename = "./itiming_log.csv";
    bool file_exists = std::experimental::filesystem::exists(csv_filename);

    std::ofstream csv_file(csv_filename, std::ios::app);

    if (!file_exists) {
      // 寫入CSV標頭
      csv_file << "Filename,Operation,Time_s,Mode,keypoints\n";
    }

    // 使用檔案名稱代替時間戳，如果沒有提供檔案名稱則使用時間戳
    std::string identifier;
    if (!filename.empty()) {
      identifier = filename;
    } else {
      std::cout << "warm up" << std::endl;
      return;
    }

    // 寫入資料
    csv_file << identifier << ","
             << operation << ","
             << std::fixed << std::setprecision(6) << time_s << ","
             << param_.pred_mode << ","
             << leng << "\n";

    csv_file.close();
  }
};

// Factory function to create an instance of the implementation
std::shared_ptr<ImageAlignTensorRT> ImageAlignTensorRT::create_instance(const Param& param)
{
  return std::make_shared<ImageAlignTensorRTImpl>(param);
}

// Dummy implementations for the base class to allow linkage
ImageAlignTensorRT::ImageAlignTensorRT(const Param& param) {}
ImageAlignTensorRT::~ImageAlignTensorRT() {}

}  // namespace core
