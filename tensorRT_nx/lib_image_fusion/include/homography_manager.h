#ifndef HOMOGRAPHY_MANAGER_H
#define HOMOGRAPHY_MANAGER_H

#include <opencv2/opencv.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace core {

/**
 * @brief 平滑 Homography 管理器
 * 負責 homography 的平滑更新和差異計算
 */
class HomographyManager {
public:
    /**
     * @brief 構造函數
     * @param max_trans_diff 最大平移差異閾值
     * @param max_rot_diff 最大旋轉差異閾值
     * @param alpha 平滑係數 (0-1)
     */
    HomographyManager(double max_trans_diff = 30.0, 
                      double max_rot_diff = 0.03, 
                      double alpha = 0.05);
    
    /**
     * @brief 計算兩個 homography 矩陣的差異
     * @param homo1 第一個 homography
     * @param homo2 第二個 homography
     * @return <平移差異, 旋轉差異>
     */
    std::pair<double, double> calculateDifference(const cv::Mat& homo1, 
                                                   const cv::Mat& homo2) const;
    
    /**
     * @brief 判斷是否應該更新 homography
     * @param new_homo 新的 homography
     * @return 是否應該更新
     */
    bool shouldUpdate(const cv::Mat& new_homo) const;
    
    /**
     * @brief 更新 homography (帶平滑)
     * @param new_homo 新的 homography
     * @return 平滑後的 homography
     */
    cv::Mat update(const cv::Mat& new_homo);
    
    /**
     * @brief 獲取當前 homography
     * @return 當前的 homography 矩陣
     */
    cv::Mat getCurrent() const;
    
    /**
     * @brief 設置參數
     */
    void setParameters(double max_trans_diff, double max_rot_diff, double alpha);
    
    /**
     * @brief 重置管理器
     */
    void reset();
    
    /**
     * @brief 獲取 fallback 計數
     */
    int getFallbackCount() const { return fallback_count_; }
    
    /**
     * @brief 增加 fallback 計數
     */
    void incrementFallback() { fallback_count_++; }
    
    /**
     * @brief 重置 fallback 計數
     */
    void resetFallback() { fallback_count_ = 0; }
    
private:
    double max_translation_diff_;
    double max_rotation_diff_;
    double smooth_alpha_;
    cv::Mat previous_homo_;
    int fallback_count_;
};

} // namespace core

#endif // HOMOGRAPHY_MANAGER_H
