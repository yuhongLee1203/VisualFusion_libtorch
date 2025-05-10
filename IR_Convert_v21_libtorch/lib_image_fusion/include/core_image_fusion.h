/*
 * core_image_fusuion.h
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
 * Modified on: Mar 17, 2024
 *      Author: HongKai
 */

#ifndef INCLUDE_CORE_IMAGE_FUSION_H_
#define INCLUDE_CORE_IMAGE_FUSION_H_

#include <memory>
#include <opencv2/opencv.hpp>

namespace core {

class ImageFusion {
public:
  using ptr = std::shared_ptr<ImageFusion>;

  struct Param {
    int edge_border = 1;
    bool do_shadow = false;
    
    int threshold_equalization = 128;
    int threshold_equalization_low  = 0;
    int threshold_equalization_high = 255;
    int threshold_equalization_zero = 0;

    cv::Mat bdStruct, sdStruct = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));

    Param &set_shadow(bool v) { do_shadow = v; return *this; }  // NOLINT
    Param &set_edge_border(int v) { 
      int bd = v - 1;
      int bd2 = bd * 2 + 1;
      edge_border = v;
      bdStruct = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(bd2, bd2), cv::Point(bd, bd));
      return *this;
    }  // NOLINT

    Param &set_threshold_equalization(int v) { threshold_equalization = v; return *this; }  // NOLINT
    Param &set_threshold_equalization_low(int v) { threshold_equalization_low = v; return *this; }  // NOLINT
    Param &set_threshold_equalization_high(int v) { threshold_equalization_high = v; return *this; }  // NOLINT
    Param &set_threshold_equalization_zero(int v) { threshold_equalization_zero = v; return *this; }  // NOLINT
  };

  static ptr create_instance(Param param)
  {
    return std::make_shared<ImageFusion>(param);
  }

  ImageFusion(Param param);

  cv::Mat edge(cv::Mat &in);
  cv::Mat equalization(cv::Mat &in);
  cv::Mat fusion(cv::Mat &eo, cv::Mat &ir);

private:
  Param param_;
  static unsigned char safeRange[1024];
};

} /* namespace core */

#endif /* INCLUDE_CORE_IMAGE_FUSION_H_ */
