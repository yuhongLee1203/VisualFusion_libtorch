/*
 * core_super_glue.h
 *
 *  Created on: Feb 6, 2024
 *      Author: arthurho
 *
 *  Modified on: Feb 23, 2024
 *      Author: HongKai
 *
 *  Modified on: May 14, 2024
 *      Author: HongKai
 */

#ifndef INCLUDE_CORE_IMAGE_ALIGN_LIBTORCH_H_
#define INCLUDE_CORE_IMAGE_ALIGN_LIBTORCH_H_

#include <memory>
#include <string>
#include <experimental/filesystem>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace core
{

  class ImageAlign
  {
  public:
    using ptr = std::shared_ptr<ImageAlign>;

#define degree_to_rad(degree) ((degree) * M_PI / 180.0);

    struct Param
    {
      // 預測尺寸
      int pred_width = 0;
      int pred_height = 0;

      // 小視窗尺寸
      int small_window_width = 0;
      int small_window_height = 0;

      // 切割視窗尺寸
      int clip_window_width = 0;
      int clip_window_height = 0;

      // 輸出座標縮放尺寸
      float out_width_scale = 1.0;
      float out_height_scale = 1.0;

      int bias_x = 0;
      int bias_y = 0;

      // 模型
      std::string mode = "fp32";
      std::string device = "cpu";
      std::string model_path = "";

      // 前一幀距離、水平/垂直篩選距離、平均角度範圍、排序角度範圍
      float distance_last = 10.0;
      float distance_line = 10.0;
      float distance_mean = 20.0;
      float angle_mean = degree_to_rad(10.0);
      float angle_sort = 0.6;

      Param &set_size(int pw, int ph, int ow, int oh)
      {
        pred_width = pw;
        pred_height = ph;

        small_window_width = pw / 8;
        small_window_height = ph / 8;

        out_width_scale = ow / (float)pw;
        out_height_scale = oh / (float)ph;

        clip_window_width = small_window_width / 10;
        clip_window_height = small_window_height / 10;
        return *this;
      }

      Param &set_net(std::string device, std::string model_path, std::string mode = "fp32")
      {
        // 檢查模型
        if (!std::experimental::filesystem::exists(model_path))
          throw std::invalid_argument("Model file not found");
        else
          this->model_path = model_path;

        // 設定裝置
        if (device.compare("cpu") == 0 || device.compare("cuda") == 0)
          this->device = device;
        else
          throw std::invalid_argument("Device not supported");

        // 設定模型輸出模式
        if (mode.compare("fp32") == 0 || mode.compare("fp16") == 0)
          this->mode = mode;
        else
          throw std::invalid_argument("Model output mode not supported");

        return *this;
      }

      Param &set_distance(float line, float last, float mean)
      {
        distance_line = line;
        distance_last = last;
        distance_mean = mean;
        return *this;
      }

      Param &set_angle(float mean, float sort)
      {
        angle_mean = degree_to_rad(mean);
        angle_sort = sort;
        return *this;
      }

      Param &set_bias(int x, int y)
      {
        bias_x = x;
        bias_y = y;
        return *this;
      }
    };

    static ptr create_instance(const Param &param)
    {
      return std::make_shared<ImageAlign>(std::move(param));
    }

    explicit ImageAlign(Param param);

    void align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H);

  private:
    Param param_;

    // device
    torch::Device device{torch::kCPU};

    // net
    torch::jit::script::Module net;

    // warm up
    void warm_up();

    // prediction
    void pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts);

    // linear equation
    std::vector<float> line_equation(cv::Point2f &pt1, cv::Point2f &pt2);

    // using one linear equation and one point to judge the position
    int judge_h_line(std::vector<float> &line, cv::Point2i &pt);
    int judge_v_line(std::vector<float> &line, cv::Point2i &pt);

    // using two linear equations to judge the position
    int judge_quadrant(std::vector<float> &line_v, std::vector<float> &line_h, cv::Point2i &pt);

    // show polar image
    void show(std::vector<std::vector<int>> q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<float> v_line, std::vector<float> h_line);

    // class keypoints into 8 quadrants
    void class_quadrant(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    // combine all keypoints into 1 quadrants
    void combine_quadrant(std::vector<std::vector<int>> &q_idx);

    // check embalanced quadrants
    bool check_quadrant_imbalance(std::vector<std::vector<int>> &q_idx);

    // find separation line
    void find_line(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<float> &v_line, std::vector<float> &h_line);

    // apply keypoints
    void apply_keypoints(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<int> &filter_idx);

    // apply quadrants
    void apply_quadrants(std::vector<std::vector<int>> &q_idx, std::vector<int> &filter_idx);

    // distance of keypoint and line
    float distance_line(std::vector<float> &line, cv::Point2i &pt);

    // filter by diagonal quadrants
    std::vector<int> filter_diagonal(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    // filter by same quadrants
    std::vector<int> filter_same(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    // filter by distance of line
    std::vector<int> filter_distance(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    // filter by angle
    std::vector<int> filter_mean_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);
    std::vector<int> filter_sort_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    // filter by distance
    std::vector<int> filter_mean_distance(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    // filter last H
    std::vector<int> filter_last_H(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, cv::Mat &H);
  };

} /* namespace core */

#endif /* INCLUDE_CORE_IMAGE_ALIGN_H_ */
