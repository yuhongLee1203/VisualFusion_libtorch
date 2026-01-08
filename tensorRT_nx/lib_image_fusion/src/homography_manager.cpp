#include "homography_manager.h"

namespace core {

HomographyManager::HomographyManager(double max_trans_diff,
                                     double max_rot_diff,
                                     double alpha)
    : max_translation_diff_(max_trans_diff),
      max_rotation_diff_(max_rot_diff),
      smooth_alpha_(alpha),
      fallback_count_(0)
{
}

std::pair<double, double> HomographyManager::calculate_difference(
    const cv::Mat& homo1,
    const cv::Mat& homo2) const
{
  if (homo1.empty() || homo2.empty()) {
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }

  // 計算平移差異
  double translation_diff = sqrt(
      pow(homo1.at<double>(0, 2) - homo2.at<double>(0, 2), 2) +
      pow(homo1.at<double>(1, 2) - homo2.at<double>(1, 2), 2));

  // 計算旋轉差異
  double angle1 = atan2(homo1.at<double>(1, 0), homo1.at<double>(0, 0));
  double angle2 = atan2(homo2.at<double>(1, 0), homo2.at<double>(0, 0));
  double rotation_diff = abs(angle1 - angle2);

  // 處理角度循環問題
  if (rotation_diff > M_PI) {
    rotation_diff = 2 * M_PI - rotation_diff;
  }

  return {translation_diff, rotation_diff};
}

bool HomographyManager::should_update(const cv::Mat& new_homo) const
{
  if (previous_homo_.empty()) {
    return true;
  }

  auto [trans_diff, rot_diff] = calculate_difference(previous_homo_, new_homo);

  return (trans_diff <= max_translation_diff_ && rot_diff <= max_rotation_diff_);
}

cv::Mat HomographyManager::update(const cv::Mat& new_homo)
{
  if (new_homo.empty()) {
    return previous_homo_;
  }

  // 第一次更新
  if (previous_homo_.empty()) {
    previous_homo_ = new_homo.clone();
    return new_homo;
  }

  // 判斷是否應該更新
  if (should_update(new_homo)) {
    // 平滑混合
    cv::Mat smoothed =
        smooth_alpha_ * new_homo + (1 - smooth_alpha_) * previous_homo_;
    previous_homo_ = smoothed.clone();
    return smoothed;
  }
  else {
    // 差異太大，保持之前的
    return previous_homo_;
  }
}

cv::Mat HomographyManager::get_current() const
{
  return previous_homo_;
}

void HomographyManager::set_parameters(double max_trans_diff,
                                       double max_rot_diff,
                                       double alpha)
{
  max_translation_diff_ = max_trans_diff;
  max_rotation_diff_ = max_rot_diff;
  smooth_alpha_ = alpha;
}

void HomographyManager::reset()
{
  previous_homo_ = cv::Mat();
  fallback_count_ = 0;
}

}  // namespace core
