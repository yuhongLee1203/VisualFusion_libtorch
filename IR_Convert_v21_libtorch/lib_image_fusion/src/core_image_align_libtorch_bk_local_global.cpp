/*
 * core_super_glue.cpp
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

#include <core_image_align_libtorch.h>
#include "util_timer.h"

namespace core
{
  ImageAlign::ImageAlign(Param param) : param_(std::move(param))
  {
    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    if (param_.device.compare("cuda") == 0 && torch::cuda::is_available())
    {
      torch::Device cuda(torch::kCUDA);
      device = cuda;
    }

    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);

    if (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0)
      net.to(torch::kHalf);

    // if (param_.device.compare("cuda") == 0)
    //   warm_up();
  }

  // warm up
  void ImageAlign::warm_up()
  {
    printf("Warm up...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; i++)
    {
      cv::Mat H;
      std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
      pred(eo, ir, eo_mkpts, ir_mkpts);
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Warm up done in %.2f s\n", period);
  }

  // prediction
  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    // resize input image to pred_width x pred_height
    cv::Mat eo_temp, ir_temp;
    cv::resize(eo, eo_temp, cv::Size(param_.pred_width, param_.pred_height));
    cv::resize(ir, ir_temp, cv::Size(param_.pred_width, param_.pred_height));

    // normalize eo and ir to 0-1, and convert from cv::Mat to torch::Tensor
    torch::Tensor eo_tensor, ir_tensor;
    if (param_.mode.compare("fp16") == 0)
    {
      eo_temp.convertTo(eo_temp, CV_16F, 1.0f / 255.0f);
      ir_temp.convertTo(ir_temp, CV_16F, 1.0f / 255.0f);
      eo_tensor = torch::from_blob(eo_temp.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat16).to(device);
      ir_tensor = torch::from_blob(ir_temp.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat16).to(device);
    }
    else
    {
      eo_temp.convertTo(eo_temp, CV_32F, 1.0f / 255.0f);
      ir_temp.convertTo(ir_temp, CV_32F, 1.0f / 255.0f);
      eo_tensor = torch::from_blob(eo_temp.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).to(device);
      ir_tensor = torch::from_blob(ir_temp.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).to(device);
    }

    // run the model
    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    // get mkpts from the model output
    torch::Tensor eo_mkpts = pred_[0].toTensor();
    torch::Tensor ir_mkpts = pred_[1].toTensor();

    // clean up eo_pts and ir_pts
    eo_pts.clear();
    ir_pts.clear();

    // convert mkpts to cv::Point2i
    for (int i = 0; i < eo_mkpts.size(0); i++)
    {
      eo_pts.push_back(cv::Point2i(static_cast<int>(eo_mkpts[i][0].item<long>()), static_cast<int>(eo_mkpts[i][1].item<long>())));
      ir_pts.push_back(cv::Point2i(static_cast<int>(ir_mkpts[i][0].item<long>()), static_cast<int>(ir_mkpts[i][1].item<long>())));
    }
  }

  // align with last H
  void ImageAlign::align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H)
  {
    // predict keypoints
    pred(eo, ir, eo_pts, ir_pts);

    // keypoints length
    int len = eo_pts.size();

    // get small window size
    int small_window_height = param_.small_window_height;
    int small_window_width = param_.small_window_width;

    // create query table (small window size) for index, mask, angle, and distance
    // v_ means vector; q_ means query
    std::vector<float> v_angle, v_distance;
    cv::Mat q_angle(small_window_height, small_window_width, CV_32F, cv::Scalar(0)), q_distance(small_window_height, small_window_width, CV_32F, cv::Scalar(0));
    std::vector<std::vector<int>> q_idx(small_window_height, std::vector<int>(small_window_width, 0)), q_mask(small_window_height, std::vector<int>(small_window_width, 0));

    // output keypoints
    std::vector<cv::Point2i> eo_mkpts, ir_mkpts;

    // calculate the angle and distance of each keypoints, and store in query table
    for (int i = 0; i < len; i++)
    {
      cv::Point2i diff = ir_pts[i] - eo_pts[i];
      cv::Point2i pt = eo_pts[i] / 8;

      // calculate angle and distance
      float angle = atan2(diff.y, diff.x);
      float distance = sqrt(diff.x * diff.x + diff.y * diff.y);

      // update angle and distance
      int x = pt.x, y = pt.y;
      if (x >= 0 && x < small_window_width && y >= 0 && y < small_window_height)
      {
        q_idx[y][x] = i;
        q_mask[y][x] = 1;
        v_angle.push_back(angle);
        v_distance.push_back(distance);
        q_angle.at<float>(y, x) = angle;
        q_distance.at<float>(y, x) = distance;
      }
    }

    // === local ===

    // 定義卷積核
    cv::Mat local_kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;

    // 計算角度和距離的均值
    cv::Mat local_angle_mean, local_distance_mean;
    cv::filter2D(q_angle, local_angle_mean, -1, local_kernel);
    cv::filter2D(q_distance, local_distance_mean, -1, local_kernel);

    // 計算角度和距離的差異
    cv::Mat local_angle_diff = cv::abs(q_angle - local_angle_mean);
    cv::Mat local_distance_diff = cv::abs(q_distance - local_distance_mean);

    // 計算差異的均值
    cv::Mat local_angle_threshold, local_distance_threshold;
    cv::filter2D(local_angle_diff, local_angle_threshold, -1, local_kernel);
    cv::filter2D(local_distance_diff, local_distance_threshold, -1, local_kernel);

    // 遍歷每個小窗口
    std::vector<int> remove_list;
    for (int y = 0; y < small_window_height; y++)
    {
      for (int x = 0; x < small_window_width; x++)
      {
        // 如果沒有關鍵點，跳過
        if (q_mask[y][x] == 0)
          continue;

        // 檢查中心點的差異是否在閾值內
        if (local_distance_threshold.at<float>(y, x) * 1.2 > local_distance_diff.at<float>(y, x) && local_angle_diff.at<float>(y, x) * 1.2 > local_angle_diff.at<float>(y, x))
        {
          q_mask[y][x] = 0;
          remove_list.push_back(q_idx[y][x]);
        }
      }
    }

    // 移除點、角度、距離
    std::reverse(remove_list.begin(), remove_list.end());
    for (int i = 0; i < remove_list.size(); i++)
    {
      int idx = remove_list[i];
      v_angle.erase(v_angle.begin() + idx);
      v_distance.erase(v_distance.begin() + idx);
    }

    // === global ===
    // sort angle
    float global_angle_mean = std::accumulate(v_angle.begin(), v_angle.end(), 0.0) / v_angle.size();

    std::vector<float> global_angle_diff;
    for (int i = 0; i < v_angle.size(); i++)
      global_angle_diff.push_back(std::fabs(v_angle[i] - global_angle_mean));
    std::sort(global_angle_diff.begin(), global_angle_diff.end());

    // select best angle
    int global_angle_threshold_idx = global_angle_diff.size() * param_.angle_sort;
    if (global_angle_threshold_idx >= global_angle_diff.size())
      global_angle_threshold_idx = global_angle_diff.size() - 1;
    float global_angle_threshold = global_angle_diff[global_angle_threshold_idx];

    // mean distance
    float global_distance_threshold = std::accumulate(v_distance.begin(), v_distance.end(), 0.0) / v_distance.size();

    // filter by angle and distance
    for (int y = 0; y < small_window_height; y++)
    {
      for (int x = 0; x < small_window_width; x++)
      {
        if (q_mask[y][x] == 0)
          continue;

        if (q_angle.at<float>(y, x) < global_angle_threshold)
        {
          int pt_idx = q_idx[y][x];
          eo_mkpts.push_back(eo_pts[pt_idx]);
          ir_mkpts.push_back(ir_pts[pt_idx]);
        }
      }
    }

    eo_pts = eo_mkpts;
    ir_pts = ir_mkpts;

    if (param_.out_width_scale - 1 > 1e-6 || param_.out_height_scale - 1 > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
    {
      for (cv::Point2i &i : eo_pts)
      {
        i.x = i.x * param_.out_width_scale + param_.bias_x;
        i.y = i.y * param_.out_height_scale + param_.bias_y;
      }
      for (cv::Point2i &i : ir_pts)
      {
        i.x = i.x * param_.out_width_scale + param_.bias_x;
        i.y = i.y * param_.out_height_scale + param_.bias_y;
      }
    }
  }
} /* namespace core */
