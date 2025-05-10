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

  // linear equation
  std::vector<float> ImageAlign::line_equation(cv::Point2f &pt1, cv::Point2f &pt2)
  {
    float m = (pt2.y - pt1.y) / (pt2.x - pt1.x);
    float b = pt1.y - m * pt1.x;
    return {m, b};
  }

  // using horizental linear equation and one point to judge the position
  int ImageAlign::judge_h_line(std::vector<float> &line, cv::Point2i &pt)
  {
    // 判斷水平線
    // -1 上方
    // 0  線上
    // 1  下方
    float y_line = line[0] * pt.x + line[1]; // y = mx + b

    if (pt.y < y_line)
      return -1;
    if (pt.y > y_line)
      return 1;
    return 0;
  }

  // using vertical linear equation and one point to judge the position
  int ImageAlign::judge_v_line(std::vector<float> &line, cv::Point2i &pt)
  {
    // 判斷垂直線
    // -1 左方
    // 0  線上
    // 1  右方
    float x_line = (pt.y - line[1]) / line[0]; // x = (y - b) / m

    if (pt.x < x_line)
      return -1;
    if (pt.x > x_line)
      return 1;
    return 0;
  }

  // using horizental and vertical linear equations to judge the position
  int ImageAlign::judge_quadrant(std::vector<float> &v_line, std::vector<float> &h_line, cv::Point2i &pt)
  {
    // 左上 => 0
    // 右上 => 1
    // 右下 => 2
    // 左下 => 3
    // 左　 => 4
    // 上　 => 5
    // 右　 => 6
    // 下　 => 7
    // 原點 => -1

    int res_h = judge_h_line(h_line, pt); // 上、下
    int res_v = judge_v_line(v_line, pt); // 左、右

    if (res_h == -1 && res_v == -1)
      return 0;
    if (res_h == -1 && res_v == 1)
      return 1;
    if (res_h == 1 && res_v == 1)
      return 2;
    if (res_h == 1 && res_v == -1)
      return 3;
    if (res_h == 0 && res_v == -1)
      return 4;
    if (res_h == -1 && res_v == 0)
      return 5;
    if (res_h == 0 && res_v == 1)
      return 6;
    if (res_h == 1 && res_v == 0)
      return 7;

    return -1;
  }

  // show polar image
  void ImageAlign::show(std::vector<std::vector<int>> q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<float> v_line, std::vector<float> h_line)
  {
    // empty image that has the same size as pred image
    cv::Mat eo_temp = cv::Mat::zeros(param_.pred_height, param_.pred_width, CV_8UC3);

    std::vector<cv::Scalar> colors = {cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), cv::Scalar(128, 128, 128), cv::Scalar(128, 128, 128), cv::Scalar(128, 128, 128), cv::Scalar(128, 128, 128)};

    for (int i = 0; i < 8; i++)
    {
      cv::Scalar color = colors[i];
      for (int j = 0; j < q_idx[i].size(); j++)
      {
        int idx = q_idx[i][j];
        cv::Point2i pt0 = eo_mkpts[idx];
        cv::Point2i pt1 = ir_mkpts[idx];
        cv::arrowedLine(eo_temp, pt0, pt1, color, 1);
      }
    }

    // draw line
    cv::Point2i pt0, pt1;
    pt0.x = 0;
    pt0.y = h_line[1];
    pt1.x = param_.pred_width;
    pt1.y = h_line[0] * pt1.x + h_line[1];
    // cv::line(eo_temp, pt0, pt1, cv::Scalar(255, 255, 0), 1);
    cv::line(eo_temp, pt0, pt1, cv::Scalar(255, 255, 255), 1);

    pt0.x = 0;
    pt0.y = v_line[1];
    pt1.x = param_.pred_width;
    pt1.y = v_line[0] * pt1.x + v_line[1];
    // cv::line(eo_temp, pt0, pt1, cv::Scalar(255, 0, 255), 1);
    cv::line(eo_temp, pt0, pt1, cv::Scalar(255, 255, 255), 1);

    cv::imshow("polar", eo_temp);
    // cv::imwrite("v19_polar.jpg", eo_temp);
    cv::waitKey(0);
  }

  // class keypoints into 4 quadrants and 4 directions
  void ImageAlign::class_quadrant(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)
  {
    // create empty vector for quadrant index and angle
    q_idx.clear();
    q_diff.clear();

    for (int i = 0; i < 9; i++)
    {
      q_idx.push_back(std::vector<int>());
    }

    int len = eo_mkpts.size();
    for (int i = 0; i < len; i++)
    {
      // get eo and ir keypoints
      cv::Point2i eo_pt = eo_mkpts[i];
      cv::Point2i ir_pt = ir_mkpts[i];

      // calculate the different of axis
      cv::Point2i diff = ir_pt - eo_pt;

      // quadrant
      int q = -1;

      if (diff.x < 0 && diff.y < 0)
        q = 0;
      else if (diff.x > 0 && diff.y < 0)
        q = 1;
      else if (diff.x > 0 && diff.y > 0)
        q = 2;
      else if (diff.x < 0 && diff.y > 0)
        q = 3;
      else if (diff.x < 0 && diff.y == 0)
        q = 4;
      else if (diff.x == 0 && diff.y < 0)
        q = 5;
      else if (diff.x > 0 && diff.y == 0)
        q = 6;
      else if (diff.x == 0 && diff.y > 0)
        q = 7;
      else
        q = 8;

      q_idx[q].push_back(i);
      q_diff.push_back(diff);
    }
  }

  // combine all keypoints into 1 quadrants
  void ImageAlign::combine_quadrant(std::vector<std::vector<int>> &q_idx)
  {
    std::vector<int> res;

    for (std::vector<int> &i : q_idx)
    {
      for (int &idx : i)
      {
        res.push_back(idx);
      }
    }

    q_idx.clear();
    q_idx.push_back(res);
    for (int i = 1; i < 9; i++)
    {
      q_idx.push_back(std::vector<int>());
    }
  }

  // check embalanced quadrants
  bool ImageAlign::check_quadrant_imbalance(std::vector<std::vector<int>> &q_idx)
  {
    int len0 = q_idx[0].size();
    int len1 = q_idx[1].size();
    int len2 = q_idx[2].size();
    int len3 = q_idx[3].size();

    int sum0 = len1 + len2 + len3;
    int sum1 = len0 + len2 + len3;
    int sum2 = len0 + len1 + len3;
    int sum3 = len0 + len1 + len2;

    if (len0 > sum0 || len1 > sum1 || len2 > sum2 || len3 > sum3)
      return true;
    return false;
  }

  // find separation line
  void ImageAlign::find_line(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<float> &v_line, std::vector<float> &h_line)
  {
    // grativity center in 4 quadrants
    std::vector<cv::Point2f> q_gravity;
    for (std::vector<int> &i : q_idx)
    {
      cv::Point2f gravity;
      for (int &idx : i)
      {
        cv::Point2i pt = eo_mkpts[idx];
        gravity.x += pt.x;
        gravity.y += pt.y;
      }

      int len = i.size();
      gravity.x /= len;
      gravity.y /= len;

      q_gravity.push_back(gravity);
    }

    // centers between 4 quadrants
    cv::Point2f h1_center = (q_gravity[0] + q_gravity[3]) / 2;
    cv::Point2f h2_center = (q_gravity[1] + q_gravity[2]) / 2;
    cv::Point2f v1_center = (q_gravity[0] + q_gravity[1]) / 2;
    cv::Point2f v2_center = (q_gravity[2] + q_gravity[3]) / 2;

    v_line = line_equation(v1_center, v2_center);
    h_line = line_equation(h1_center, h2_center);
  }

  // apply keypoints
  void ImageAlign::apply_keypoints(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<int> &filter_idx)
  {
    std::vector<cv::Point2i> eo_mkpts_temp, ir_mkpts_temp;
    for (int &i : filter_idx)
    {
      eo_mkpts_temp.push_back(eo_mkpts[i]);
      ir_mkpts_temp.push_back(ir_mkpts[i]);
    }

    eo_mkpts = eo_mkpts_temp;
    ir_mkpts = ir_mkpts_temp;
  }

  // apply filter result to quadrants
  void ImageAlign::apply_quadrants(std::vector<std::vector<int>> &q_idx, std::vector<int> &filter_idx)
  {
    std::vector<std::vector<int>> q_idx_temp;

    int count = 0;
    for (std::vector<int> &i : q_idx)
    {
      q_idx_temp.push_back(std::vector<int>());
      for (int &idx : i)
      {
        if (std::find(filter_idx.begin(), filter_idx.end(), idx) != filter_idx.end())
          q_idx_temp[count].push_back(idx);
      }
      count++;
    }

    q_idx = q_idx_temp;
  }

  // distance of keypoint and line
  float ImageAlign::distance_line(std::vector<float> &line, cv::Point2i &pt)
  {
    float m = line[0], b = line[1];
    float x = pt.x, y = pt.y;
    float d = abs(m * x - y + b) / sqrt(m * m + 1);
    return d;
  }

  // filter by diagonal quadrants
  std::vector<int> ImageAlign::filter_diagonal(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)
  {
    std::vector<int> res;

    int diagonals[4] = {2, 3, 0, 1}; // diagonal quadrant list

    int count = 0;
    for (std::vector<int> &i : q_idx)
    {
      // remove diagonal quadrants (0~3)
      if (count < 4)
      {
        int diagonal = diagonals[count]; // diagonals of this count
        for (int &idx : i)
        {
          cv::Point2i pt = eo_mkpts[idx];
          int pos = judge_quadrant(v_line, h_line, pt);
          if (pos != diagonal)
            res.push_back(idx);
        }
      }
      // remain all directions (4~8)
      else
      {
        for (int &idx : i)
        {
          res.push_back(idx);
        }
      }

      count++;
    }

    return res;
  }

  // filter by same quadrants
  std::vector<int> ImageAlign::filter_same(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)
  {
    std::vector<int> res;

    int count = 0;
    for (std::vector<int> &i : q_idx)
    {
      // remain same quadrants (0~3)
      if (count < 4)
      {
        for (int &idx : i)
        {
          cv::Point2i pt = eo_mkpts[idx];
          int pos = judge_quadrant(v_line, h_line, pt);
          if (pos == count)
            res.push_back(idx);
        }
      }
      // remain all directions (4~8)
      else
      {
        for (int &idx : i)
        {
          res.push_back(idx);
        }
      }

      count++;
    }

    return res;
  }

  // filter by distance of line
  std::vector<int> ImageAlign::filter_distance(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)
  {
    std::vector<int> res;

    // illustration
    // direction: 0 degree
    //   * judge position with v_line, and locate in left(-1) of line
    //   * distance with h_line
    // direction: 90 degree
    //   * judge position with h_line, and locate in up(-1) of line
    //   * distance with v_line
    // direction: 180 degree
    //   * judge position with v_line, and locate in right(1) of line
    //   * distance with h_line
    // direction: 270 degree
    //   * judge position with h_line, and locate in down(1) of line
    //   * distance with v_line
    // center point
    //  * distance with v_line and h_line

    int targets[4] = {-1, -1, 1, 1};
    std::vector<float> *dis_lines[4] = {&h_line, &v_line, &h_line, &v_line};
    std::vector<float> *judge_lines[4] = {&v_line, &h_line, &v_line, &h_line};

    int count = 0;
    for (std::vector<int> &i : q_idx)
    {
      // judge direction (4~8)
      if (count == 4 || count == 6)
      {
        int count_idx = count - 4;
        int target = targets[count_idx];
        std::vector<float> *dis_line = dis_lines[count_idx];
        std::vector<float> *judge_line = judge_lines[count_idx];

        for (int &idx : i)
        {
          cv::Point2i pt = eo_mkpts[idx];

          // check position located correct or not
          int pos = judge_v_line(*judge_line, pt);
          if (pos != target)
            continue;

          // check distance between line and keypoint
          float distance = distance_line(*dis_line, pt);
          if (distance < param_.distance_line)
            res.push_back(idx);
        }
      }
      else if (count == 5 || count == 7)
      {
        int count_idx = count - 4;
        int target = targets[count_idx];
        std::vector<float> *dis_line = dis_lines[count_idx];
        std::vector<float> *judge_line = judge_lines[count_idx];

        for (int &idx : i)
        {
          cv::Point2i pt = eo_mkpts[idx];

          // check position located correct or not
          int pos = judge_h_line(*judge_line, pt);
          if (pos != target)
            continue;

          // check distance between line and keypoint
          float distance = distance_line(*dis_line, pt);
          if (distance < param_.distance_line)
            res.push_back(idx);
        }
      }
      else if (count == 8)
      {
        for (int &idx : i)
        {
          cv::Point2i pt = eo_mkpts[idx];

          // check distance between line and keypoint
          float distance1 = distance_line(v_line, pt);
          float distance2 = distance_line(h_line, pt);

          if (distance1 < param_.distance_line || distance2 < param_.distance_line)
            res.push_back(idx);
        }
      }
      // remain all quandrant and center (0~3, 8)
      else
      {
        for (int &idx : i)
        {
          res.push_back(idx);
        }
      }

      count++;
    }

    return res;
  }

  // filter by angle
  std::vector<int> ImageAlign::filter_mean_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)
  {
    std::vector<int> res;

    int count = 0;

    for (std::vector<int> &i : q_idx)
    {
      // remain angle between 10 with mean angle (0~3)
      if (count < 4)
      {
        int len = i.size();
        cv::Point2i mean_diff(0, 0);

        if (len == 0)
          continue;

        // calculate mean of x and y, that xy means the different of axis
        for (int &idx : i)
        {
          mean_diff += q_diff[idx];
        }
        mean_diff /= len;
        float mean_ang = atan2(mean_diff.y, mean_diff.x);

        // calculate the angle with mean by using mean xy
        for (int &idx : i)
        {
          float ang = atan2(q_diff[idx].y, q_diff[idx].x);
          if (abs(ang - mean_ang) < param_.angle_mean)
            res.push_back(idx);
        }
      }
      // remain all directions (4~8)
      else
      {
        for (int &idx : i)
        {
          res.push_back(idx);
        }
      }

      count++;
    }

    return res;
  }

  std::vector<int> ImageAlign::filter_sort_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)
  {
    std::vector<int> res;

    int count = 0;

    for (std::vector<int> &i : q_idx)
    {
      // remain angle between 10 with mean angle (0~3)
      if (count < 4)
      {
        int len = i.size();
        cv::Point2i mean_diff(0, 0);

        if (len == 0)
          continue;

        // calculate mean of x and y, that xy means the different of axis
        for (int &idx : i)
        {
          mean_diff += q_diff[idx];
        }
        mean_diff /= len;
        float mean_angle = atan2(mean_diff.y, mean_diff.x);

        // calculate the angle difference with mean
        std::vector<float> angles;
        for (int &idx : i)
        {
          angles.push_back(std::fabs(atan2(q_diff[idx].y, q_diff[idx].x) - mean_angle));
        }

        // sort angles, and remain original variable "angle", that can be a table to query the index of q_idx
        std::vector<float> sort_angles = angles;
        std::sort(sort_angles.begin(), sort_angles.end());

        // get the index of the first "angle_sort"% of the total number of keypoints
        int sort_number = (float)len * param_.angle_sort;
        float sort_angle = sort_angles[sort_number];

        if (sort_number >= len)
          throw std::runtime_error("ImageAlign::filter_sort_angle: sort_number is out of range");
        float angle_threshold = sort_angles[sort_number];

        for (int j = 0; j < len; j++)
        {
          if (angles[j] < angle_threshold)
            res.push_back(i[j]);
        }
      }
      // remain all directions (4~8)
      else
      {
        for (int &idx : i)
        {
          res.push_back(idx);
        }
      }

      count++;
    }

    return res;
  }

  // filter by distance
  std::vector<int> ImageAlign::filter_mean_distance(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)
  {
    std::vector<int> res;

    for (std::vector<int> &i : q_idx)
    {
      int len = i.size();
      cv::Point2i mean_diff(0, 0);

      if (len == 0)
        continue;

      // calculate mean of x and y
      for (int &idx : i)
      {
        mean_diff += q_diff[idx];
      }
      mean_diff /= len;

      float mean_distance = sqrt(mean_diff.x * mean_diff.x + mean_diff.y * mean_diff.y);

      // calculate the distance with mean by using mean xy
      for (int &idx : i)
      {
        cv::Point2i diff = q_diff[idx];
        float distance = sqrt(diff.x * diff.x + diff.y * diff.y);
        if (abs(distance - mean_distance) < param_.distance_mean)
          res.push_back(idx);
      }
    }

    return res;
  }

  // filter last H
  std::vector<int> ImageAlign::filter_last_H(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, cv::Mat &H)
  {
    std::vector<int> res;

    double a = H.at<double>(0, 0), b = H.at<double>(0, 1), c = H.at<double>(0, 2);
    double d = H.at<double>(1, 0), e = H.at<double>(1, 1), f = H.at<double>(1, 2);
    double g = H.at<double>(2, 0), h = H.at<double>(2, 1);

    int count = 0;
    for (std::vector<int> &i : q_idx)
    {
      for (int &idx : i)
      {
        cv::Point2i src = cv::Point2i(eo_mkpts[idx].x * param_.out_width_scale + param_.bias_x, eo_mkpts[idx].y * param_.out_height_scale + param_.bias_y);
        cv::Point2i dst = cv::Point2i(ir_mkpts[idx].x * param_.out_width_scale + param_.bias_x, ir_mkpts[idx].y * param_.out_height_scale + param_.bias_y);

        int x = src.x, y = src.y;
        float z_base = g * x + h * y + 1;
        float x_dst = (a * x + b * y + c) / z_base;
        float y_dst = (d * x + e * y + f) / z_base;

        float diff_x = dst.x - x_dst;
        float diff_y = dst.y - y_dst;
        float diff = sqrt(diff_x * diff_x + diff_y * diff_y);

        if (diff < param_.distance_last)
          res.push_back(idx);
      }
    }

    return res;
  }

} /* namespace core */
