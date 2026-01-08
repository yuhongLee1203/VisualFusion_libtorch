#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace core {

class ImageAlignTensorRT {
public:
  struct Param {
    // CORRECTED: 使用與LibTorch版本完全一致的參數名稱
    int pred_width = 320;    // 與LibTorch一致：pred_width
    int pred_height = 240;   // 與LibTorch一致：pred_height
    int output_w = 320;
    int output_h = 240;
    float out_width_scale = 1.0f;
    float out_height_scale = 1.0f;
    float bias_x = 0.0f;
    float bias_y = 0.0f;
    std::string engine_path;
    std::string pred_mode = "fp32";  // 添加 pred_mode 參數，預設為 fp32
    std::string image_name = "";     // 添加圖片名稱參數

    // CORRECTED: 與LibTorch版本完全一致的set_size邏輯
    Param& set_size(int pw, int ph, int ow, int oh)
    {
      pred_width = pw;
      pred_height = ph;
      output_w = ow;
      output_h = oh;
      // 與LibTorch版本保持一致的縮放計算：輸出尺寸/預測尺寸
      out_width_scale = ow / (float)pw;
      out_height_scale = oh / (float)ph;
      return *this;
    }

    Param& set_scale_and_bias(float scale_w, float scale_h, float bx, float by)
    {
      out_width_scale = scale_w;
      out_height_scale = scale_h;
      bias_x = bx;
      bias_y = by;
      return *this;
    }

    Param& set_engine(const std::string& path)
    {
      engine_path = path;
      return *this;
    }

    Param& set_pred_mode(const std::string& mode)
    {
      pred_mode = mode;
      return *this;
    }

    Param& set_image_name(const std::string& name)
    {
      image_name = name;
      return *this;
    }
  };

  static std::shared_ptr<ImageAlignTensorRT> create_instance(const Param& param);

  // The main alignment function. It takes two images, finds matching keypoints,
  // and computes the homography matrix.
  // The keypoint vectors (eo_pts, ir_pts) are cleared and then filled with the results.
  virtual void align(const cv::Mat& eo, const cv::Mat& ir,
                     std::vector<cv::Point2i>& eo_pts,
                     std::vector<cv::Point2i>& ir_pts, cv::Mat& H) = 0;

  // Set current image name for CSV logging
  virtual void set_current_image_name(const std::string& image_name) = 0;

  virtual ~ImageAlignTensorRT();

protected:  // Changed from private to protected to allow implementation class to inherit
  ImageAlignTensorRT(const Param& param);

private:
  // PIMPL (Pointer to implementation) pattern can be used here if we want to hide all private members
  // For now, we keep it simple. The implementation will be in the .cpp file.
};

}  // namespace core
