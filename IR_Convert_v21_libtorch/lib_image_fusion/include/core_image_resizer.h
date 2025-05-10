/*
 * core_image_resizer.h
 *
 *  Created on: Feb 15, 2024
 *      Author: arthurho
 * 
 * Modified on: Feb 22, 2024
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

#ifndef INCLUDE_CORE_IMAGE_RESIZER_H_
#define INCLUDE_CORE_IMAGE_RESIZER_H_

#include <memory>
#include <opencv2/opencv.hpp>

namespace core {
  
class ImageResizer {
public:
  using ptr = std::shared_ptr<ImageResizer>;

  struct Param {
    int eo_out_width;
    int eo_out_height;
    cv::Rect eo_rect;

    int ir_out_width;
    int ir_out_height;
    cv::Rect ir_rect;

    Param &set_eo(int w, int h){
      eo_out_width = w;
      eo_out_height = h;
      return *this;
    }

    Param &set_eo(int w, int h, int clip_w, int clip_h){
      eo_out_width = w;
      eo_out_height = h;

      int x = (w - clip_w) / 2;
      int y = (h - clip_h) / 2;
      eo_rect = cv::Rect(x, y, clip_w, clip_h);
      return *this;
    }

    Param &set_ir(int w, int h){
      ir_out_width = w;
      ir_out_height = h;
      return *this;
    }

    Param &set_ir(int w, int h, int clip_w, int clip_h){
      ir_out_width = w;
      ir_out_height = h;

      int x = (w - clip_w) / 2;
      int y = (h - clip_h) / 2;
      ir_rect = cv::Rect(x, y, clip_w, clip_h);
      return *this;
    }
};

  static ptr create_instance(Param param)
  {
    return std::make_shared<ImageResizer>(std::move(param));
  }

  explicit ImageResizer(Param param);

  void resize(cv::Mat &eo_in, cv::Mat &ir_in, cv::Mat &eo_out, cv::Mat &ir_out);
  cv::Mat clip_eo(cv::Mat &in);
  cv::Mat clip_ir(cv::Mat &in);

  int get_eo_clip_x() { return param_.eo_rect.x; }
  int get_eo_clip_y() { return param_.eo_rect.y; }
  int get_ir_clip_x() { return param_.ir_rect.x; }
  int get_ir_clip_y() { return param_.ir_rect.y; }
  
private:
  Param param_;
};

} /* namespace core */

#endif /* INCLUDE_CORE_IMAGE_RESIZER_H_ */
