/*
 * core_image_fusuion.cpp
 *
 *  Created on: Feb 15, 2024
 *      Author: arthurho
 *
 *  Modified on: Feb 22, 2024
 *      Author: HongKai
 *
 * Modified on: Feb 29, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 8, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 17, 2024
 *      Author: HongKai
 */

#include <core_image_fusion.h>

namespace core
{

  ImageFusion::ImageFusion(Param param) : param_(std::move(param)) {}

  cv::Mat ImageFusion::equalization(cv::Mat &in)
  {
    cv::Mat out = in.clone();

    // 統計出現次數
    cv::Mat sum;
    int histSize = 256;
    float range[2] = {0, 256};
    const float *hisRange = {range};
    cv::calcHist(&in, 1, 0, cv::Mat(), sum, 1, &histSize, &hisRange);

    cv::Scalar mean = cv::mean(in);

    int th = param_.threshold_equalization;
    int th0 = param_.threshold_equalization_zero;
    int th1 = param_.threshold_equalization_low;
    int th2 = param_.threshold_equalization_high;

    if (mean[0] <= th)
    {
      cv::Mat table(1, 256, CV_8U);
      unsigned char *tb = table.data;

      int min = 0;
      while (sum.at<float>(min) == 0)
        min++;

      min = std::max(min, th0);
      th1 = std::max(th1, min);

      int range = th2 - th1;

      int pn = 0;
      for (int i = th1; i <= th2; i++)
        pn += sum.at<float>(i);

      float prob = 0.0;
      for (int i = 0; i < 256; i++)
      {
        if (i < min)
          tb[i] = 0;
        else if (th1 <= i && i < th2)
        {
          prob += sum.at<float>(i) / pn;
          tb[i] = prob * range + th1;
        }
        else
          tb[i] = i;
      }

      cv::LUT(in, table, out);
    }

    return out;
  }

  cv::Mat ImageFusion::edge(cv::Mat &in)
  {
    cv::Mat out;
    cv::blur(in, out, cv::Size(3, 3));
    cv::Canny(out, out, 100, 200, 3);
    return out;
  }

  cv::Mat ImageFusion::fusion(cv::Mat &eo, cv::Mat &ir)
  {
    cv::Mat boder, shadow;
    cv::Mat out = ir.clone();

    if (param_.edge_border > 1)
      cv::dilate(eo, boder, param_.bdStruct);
    else
      boder = eo;

    if (param_.do_shadow)
    {
      cv::dilate(boder, shadow, param_.sdStruct);
      cv::cvtColor(shadow, shadow, cv::COLOR_GRAY2BGR);
      cv::subtract(out, shadow, out);
    }

    cv::cvtColor(boder, boder, cv::COLOR_GRAY2BGR);
    cv::add(out, boder, out);

    return out;
  }
} /* namespace core */
