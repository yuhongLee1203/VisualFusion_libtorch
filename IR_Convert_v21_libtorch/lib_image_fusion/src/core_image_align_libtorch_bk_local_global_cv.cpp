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
    float v_angle[len], v_distance[len];
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
        v_angle[i] = angle;
        v_distance[i] = distance;
        q_angle.at<float>(y, x) = angle;
        q_distance.at<float>(y, x) = distance;
      }
    }

    // === global ===
    // average angle, distance
    float global_angle_mean = 0, global_distance_mean = 0;
    float *a = v_angle, *d = v_distance;
    for (int i = 0; i < len; i++, a++, d++)
    {
      global_angle_mean += *a;
      global_distance_mean += *d;
    }
    global_angle_mean /= len;
    global_distance_mean /= len;

    // get threshold distance
    float global_distance_threshold = param_.distance_mean;

    // calculate difference of angle, mask of distance
    std::vector<float> global_angle_diff;
    std::vector<bool> global_distance_mask;
    a = v_angle, d = v_distance;
    for (int i = 0; i < len; i++, a++, d++)
    {
      global_angle_diff.push_back(std::fabs(*a - global_angle_mean));
      global_distance_mask.push_back(std::fabs(*d - global_distance_mean) < global_distance_threshold);
    }

    // sort angle
    std::vector<float> global_angle_sort = global_angle_diff;
    std::sort(global_angle_sort.begin(), global_angle_sort.end());

    // get threshold angle
    int global_angle_threshold_idx = len * param_.angle_sort;
    if (global_angle_threshold_idx >= len)
      global_angle_threshold_idx = len - 1; // avoid out of range
    float global_angle_threshold = global_angle_sort[global_angle_threshold_idx];

    // get filtered index
    int global_filtered[len] = {0};
    for (int i = 0; i < len; i++)
      if (global_angle_diff[i] < global_angle_threshold && global_distance_mask[i]) // ad
                                                                                    // if (global_angle_diff[i] < global_angle_threshold) // a
                                                                                    // if (global_distance_mask[i]) // d
        global_filtered[i] = 1;

    // === local ===
    // define kernel
    cv::Mat local_kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;

    // calculate mean of angle, distance
    cv::Mat local_angle_mean, local_distance_mean;
    cv::filter2D(q_angle, local_angle_mean, -1, local_kernel);
    cv::filter2D(q_distance, local_distance_mean, -1, local_kernel);

    // calculate difference between original and mean
    cv::Mat local_angle_diff = cv::abs(q_angle - local_angle_mean);
    cv::Mat local_distance_diff = cv::abs(q_distance - local_distance_mean);

    // calculate mean of difference
    cv::Mat local_angle_threshold, local_distance_threshold;
    cv::filter2D(local_angle_diff, local_angle_threshold, -1, local_kernel);
    cv::filter2D(local_distance_diff, local_distance_threshold, -1, local_kernel);

    // get filtered index
    int local_filtered[len] = {0};
    for (int i = 0; i < len; i++)
    {
      cv::Point2i pt = eo_pts[i] / 8;
      int x = pt.x, y = pt.y;

      if (x >= 0 && x < small_window_width && y >= 0 && y < small_window_height)
        // if (local_distance_threshold.at<float>(y, x) * 1.2 > local_distance_diff.at<float>(y, x) && local_angle_diff.at<float>(y, x) * 1.2 > local_angle_diff.at<float>(y, x)) // ad
        // if (local_angle_diff.at<float>(y, x) < global_angle_threshold) // a
        if (local_distance_diff.at<float>(y, x) < global_distance_threshold) // d
          local_filtered[i] = 1;
    }

    // filtered index
    for (int i = 0; i < len; i++)
    {
      if (global_filtered[i] && local_filtered[i])
      {
        eo_mkpts.push_back(eo_pts[i]);
        ir_mkpts.push_back(ir_pts[i]);
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
