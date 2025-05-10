/*
 * core_image_resizer.cpp
 *
 *  Created on: Feb 15, 2024
 *      Author: arthurho
 *
 * Modified on: Feb 22, 2024
 *      Author: HongKai
 *
 * Modified on: Feb 29, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 12, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 17, 2024
 *      Author: HongKai
 *
 * Modified on: Jul 19, 2024
 *      Author: HongKai
 */

#include <core_image_resizer.h>

namespace core
{

  ImageResizer::ImageResizer(Param param) : param_(std::move(param))
  {
  }

  void ImageResizer::resize(cv::Mat &eo_in, cv::Mat &ir_in, cv::Mat &eo_out, cv::Mat &ir_out)
  {
    cv::resize(eo_in, eo_out, cv::Size(param_.eo_out_width, param_.eo_out_height));
    cv::resize(ir_in, ir_out, cv::Size(param_.ir_out_width, param_.ir_out_height));
  }

  cv::Mat ImageResizer::clip_eo(cv::Mat &in)
  {
    return in(param_.eo_rect);
  }

  cv::Mat ImageResizer::clip_ir(cv::Mat &in)
  {
    return in(param_.ir_rect);
  }
} /* namespace core */
