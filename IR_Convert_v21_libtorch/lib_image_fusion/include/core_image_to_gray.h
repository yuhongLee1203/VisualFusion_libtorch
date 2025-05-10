/*
 * core_image_to_gray.h
 *
 *  Created on: Feb 22, 2024
 *      Author: HongKai
 * 
 * Modified on: Mar 17, 2024
 *      Author: HongKai
 */

#ifndef INCLUDE_CORE_IMAGE_TO_GRAY_H_
#define INCLUDE_CORE_IMAGE_TO_GRAY_H_

#include <memory>
#include <opencv2/opencv.hpp>

namespace core
{
class ImageToGray{
  public:
    using ptr = std::shared_ptr<ImageToGray>;

    struct Param{};

    static ptr create_instance(Param param)
    {
      return std::make_shared<ImageToGray>(std::move(param));
    }

    explicit ImageToGray(Param param);

    cv::Mat gray(cv::Mat &in);

  private:
    Param param_;
  };

} /* namespace core */

#endif /* INCLUDE_CORE_IMAGE_RESIZER_H_ */
