/*
 * core_image_fusion_trt.cpp
 *
 * TensorRT 版本的 Image Fusion 實現
 */

#include "core_image_fusion_trt.h"

#include <algorithm>
#include <fstream>
#include <iostream>

namespace core {

ImageFusionTRT::ImageFusionTRT(Param param) : param_(std::move(param))
{
  initialized_ = init_engine();
  if (!initialized_) {
    std::cerr << "[ImageFusionTRT] Failed to initialize TensorRT engine!"
              << std::endl;
  }
}

ImageFusionTRT::~ImageFusionTRT()
{
  release_resources();
}

bool ImageFusionTRT::init_engine()
{
  // 檢查引擎文件
  std::ifstream file(param_.engine_path, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[ImageFusionTRT] Engine file not found: " << param_.engine_path
              << std::endl;
    return false;
  }

  // 讀取引擎數據
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(size);
  file.read(engine_data.data(), size);
  file.close();

  // 創建 runtime 和 engine
  runtime_ = nvinfer1::createInferRuntime(logger_);
  if (!runtime_) {
    std::cerr << "[ImageFusionTRT] Failed to create runtime!" << std::endl;
    return false;
  }

  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
  if (!engine_) {
    std::cerr << "[ImageFusionTRT] Failed to deserialize engine!" << std::endl;
    return false;
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    std::cerr << "[ImageFusionTRT] Failed to create execution context!"
              << std::endl;
    return false;
  }

  // 創建 CUDA stream
  cudaStreamCreate(&stream_);

  // 計算 buffer 大小
  int h = param_.height;
  int w = param_.width;
  eo_gray_size_ = 1 * 1 * h * w * sizeof(float);
  ir_color_size_ = 1 * 3 * h * w * sizeof(float);
  output_size_ = 1 * 3 * h * w * sizeof(float);

  // 分配 GPU 記憶體
  cudaMalloc(&d_eo_gray_, eo_gray_size_);
  cudaMalloc(&d_ir_color_, ir_color_size_);
  cudaMalloc(&d_output_, output_size_);

  // 分配 Host 記憶體
  h_eo_gray_.resize(1 * 1 * h * w);
  h_ir_color_.resize(1 * 3 * h * w);
  h_output_.resize(1 * 3 * h * w);

  std::cout << "[ImageFusionTRT] Engine loaded successfully!" << std::endl;
  std::cout << "  Engine path: " << param_.engine_path << std::endl;
  std::cout << "  Input size: " << w << "x" << h << std::endl;

  return true;
}

void ImageFusionTRT::release_resources()
{
  if (d_eo_gray_) {
    cudaFree(d_eo_gray_);
    d_eo_gray_ = nullptr;
  }
  if (d_ir_color_) {
    cudaFree(d_ir_color_);
    d_ir_color_ = nullptr;
  }
  if (d_output_) {
    cudaFree(d_output_);
    d_output_ = nullptr;
  }

  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (context_) {
    delete context_;
    context_ = nullptr;
  }
  if (engine_) {
    delete engine_;
    engine_ = nullptr;
  }
  if (runtime_) {
    delete runtime_;
    runtime_ = nullptr;
  }
}

void ImageFusionTRT::preprocess(const cv::Mat& eo_gray, const cv::Mat& ir_color)
{
  int h = param_.height;
  int w = param_.width;

  // EO gray: [H, W] CV_8UC1 -> [1, 1, H, W] float normalized
  cv::Mat eo_resized;
  if (eo_gray.rows != h || eo_gray.cols != w) {
    cv::resize(eo_gray, eo_resized, cv::Size(w, h));
  }
  else {
    eo_resized = eo_gray;
  }

  // Normalize and copy to host buffer
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      h_eo_gray_[y * w + x] =
          static_cast<float>(eo_resized.at<uchar>(y, x)) / 255.0f;
    }
  }

  // IR color: [H, W, 3] CV_8UC3 -> [1, 3, H, W] float normalized
  cv::Mat ir_resized;
  if (ir_color.rows != h || ir_color.cols != w) {
    cv::resize(ir_color, ir_resized, cv::Size(w, h));
  }
  else {
    ir_resized = ir_color;
  }

  // Convert to CHW format and normalize
  int hw = h * w;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      cv::Vec3b pixel = ir_resized.at<cv::Vec3b>(y, x);
      // BGR format (OpenCV default)
      h_ir_color_[0 * hw + y * w + x] =
          static_cast<float>(pixel[0]) / 255.0f;  // B
      h_ir_color_[1 * hw + y * w + x] =
          static_cast<float>(pixel[1]) / 255.0f;  // G
      h_ir_color_[2 * hw + y * w + x] =
          static_cast<float>(pixel[2]) / 255.0f;  // R
    }
  }

  // Copy to GPU
  cudaMemcpyAsync(d_eo_gray_, h_eo_gray_.data(), eo_gray_size_,
                  cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(d_ir_color_, h_ir_color_.data(), ir_color_size_,
                  cudaMemcpyHostToDevice, stream_);
}

cv::Mat ImageFusionTRT::postprocess()
{
  int h = param_.height;
  int w = param_.width;

  // Copy from GPU
  cudaMemcpyAsync(h_output_.data(), d_output_, output_size_,
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  // Convert [1, 3, H, W] float -> [H, W, 3] CV_8UC3
  cv::Mat result(h, w, CV_8UC3);
  int hw = h * w;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float b = h_output_[0 * hw + y * w + x];
      float g = h_output_[1 * hw + y * w + x];
      float r = h_output_[2 * hw + y * w + x];

      // Clamp and convert to 8-bit
      result.at<cv::Vec3b>(y, x) = cv::Vec3b(
          static_cast<uchar>(std::clamp(b * 255.0f, 0.0f, 255.0f)),
          static_cast<uchar>(std::clamp(g * 255.0f, 0.0f, 255.0f)),
          static_cast<uchar>(std::clamp(r * 255.0f, 0.0f, 255.0f)));
    }
  }

  return result;
}

cv::Mat ImageFusionTRT::fusion(const cv::Mat& eo_gray, const cv::Mat& ir_color)
{
  if (!initialized_) {
    std::cerr << "[ImageFusionTRT] Engine not initialized!" << std::endl;
    return cv::Mat();
  }

  // 預處理
  preprocess(eo_gray, ir_color);

  // 設置輸入輸出 bindings
  // 假設輸入順序: eo_gray, ir_color; 輸出: fused
  void* bindings[3] = {d_eo_gray_, d_ir_color_, d_output_};

  // 獲取 tensor 名稱並設置地址 (TensorRT 8.x API)
  for (int i = 0; i < engine_->getNbIOTensors(); i++) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      if (std::string(name) == "eo_gray") {
        context_->setTensorAddress(name, d_eo_gray_);
      }
      else if (std::string(name) == "ir_color") {
        context_->setTensorAddress(name, d_ir_color_);
      }
    }
    else {
      context_->setTensorAddress(name, d_output_);
    }
  }

  // 執行推理
  bool success = context_->enqueueV3(stream_);
  if (!success) {
    std::cerr << "[ImageFusionTRT] Inference failed!" << std::endl;
    return cv::Mat();
  }

  // 後處理
  cv::Mat result = postprocess();

  return result;
}

cv::Mat ImageFusionTRT::edge(const cv::Mat& eo_gray)
{
  // TensorRT 版本直接返回融合結果，這裡提供一個空實現
  // 如果需要單獨的邊緣檢測，需要另外導出一個只輸出 edge 的模型
  std::cerr << "[ImageFusionTRT] edge() not implemented for TRT version. "
            << "Use fusion() instead." << std::endl;
  return cv::Mat();
}

}  // namespace core
