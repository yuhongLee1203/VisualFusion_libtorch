/*
 * core_image_to_gray.cpp
 *
 *  Created on: Feb 22, 2024
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

#include <core_image_to_gray.h>

namespace core
{

  ImageToGray::ImageToGray(Param param) : param_(std::move(param))
  {
  }

  cv::Mat ImageToGray::gray(cv::Mat &in)
  {
    cv::Mat out;
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    
    return out;
  }

} /* namespace core */
