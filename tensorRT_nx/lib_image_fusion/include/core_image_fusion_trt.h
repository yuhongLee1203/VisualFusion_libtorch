/*
 * core_image_fusion_trt.h
 *
 * TensorRT 版本的 Image Fusion
 * 使用 GPU 加速進行邊緣檢測和融合
 */

#ifndef INCLUDE_CORE_IMAGE_FUSION_TRT_H_
#define INCLUDE_CORE_IMAGE_FUSION_TRT_H_

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

namespace core {

class ImageFusionTRT {
 public:
  using ptr = std::shared_ptr<ImageFusionTRT>;

  struct Param {
    std::string engine_path = "";
    int width = 320;
    int height = 240;

    Param& set_engine_path(const std::string& path)
    {
      engine_path = path;
      return *this;
    }
    Param& set_size(int w, int h)
    {
      width = w;
      height = h;
      return *this;
    }
  };

  static ptr create_instance(Param param)
  {
    return std::make_shared<ImageFusionTRT>(param);
  }

  ImageFusionTRT(Param param);
  ~ImageFusionTRT();

  // 主要接口 - 完整的融合流程
  // eo_gray: 灰度 EO 圖像 [H, W] CV_8UC1
  // ir_color: 彩色 IR 圖像 [H, W, 3] CV_8UC3
  // 返回: 融合結果 [H, W, 3] CV_8UC3
  cv::Mat fusion(const cv::Mat& eo_gray, const cv::Mat& ir_color);

  // 僅邊緣檢測
  cv::Mat edge(const cv::Mat& eo_gray);

  // 是否初始化成功
  bool is_initialized() const { return initialized_; }

 private:
  // 初始化 TensorRT 引擎
  bool init_engine();

  // 釋放資源
  void release_resources();

  // 預處理: OpenCV Mat -> GPU buffer (normalized float)
  void preprocess(const cv::Mat& eo_gray, const cv::Mat& ir_color);

  // 後處理: GPU buffer -> OpenCV Mat
  cv::Mat postprocess();

  // TensorRT Logger
  class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override
    {
      if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
      }
    }
  };

  Param param_;
  bool initialized_ = false;

  // TensorRT 相關
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;

  // CUDA 資源
  cudaStream_t stream_ = nullptr;

  // GPU buffers
  void* d_eo_gray_ = nullptr;   // [1, 1, H, W] float
  void* d_ir_color_ = nullptr;  // [1, 3, H, W] float
  void* d_output_ = nullptr;    // [1, 3, H, W] float

  // Host buffers for input/output
  std::vector<float> h_eo_gray_;
  std::vector<float> h_ir_color_;
  std::vector<float> h_output_;

  // Buffer sizes
  size_t eo_gray_size_ = 0;
  size_t ir_color_size_ = 0;
  size_t output_size_ = 0;
};

}  // namespace core

#endif  // INCLUDE_CORE_IMAGE_FUSION_TRT_H_
