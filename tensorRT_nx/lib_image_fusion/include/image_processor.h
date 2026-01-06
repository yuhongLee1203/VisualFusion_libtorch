#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "app_config.h"
#include "app_utils.h"
#include "homography_manager.h"
#include "core_image_fusion_trt.h"
#include "core_image_align_tensorrt.h"
#include "util_timer.h"

namespace core {

/**
 * @brief 影像處理器
 * 統一處理圖片和影片的融合流程
 */
class ImageProcessor {
public:
    /**
     * @brief 構造函數
     * @param config 配置物件
     */
    explicit ImageProcessor(const AppConfig& config);
    
    /**
     * @brief 初始化處理器
     * @return 是否成功初始化
     */
    bool initialize();
    
    /**
     * @brief 處理單張圖片
     * @param eo_path EO 圖片路徑
     * @param ir_path IR 圖片路徑
     * @param save_path 儲存路徑
     * @return 是否成功
     */
    bool processImage(const std::string& eo_path, 
                      const std::string& ir_path,
                      const std::string& save_path);
    
    /**
     * @brief 處理影片
     * @param eo_path EO 影片路徑
     * @param ir_path IR 影片路徑
     * @param save_path 儲存路徑
     * @return 是否成功
     */
    bool processVideo(const std::string& eo_path,
                      const std::string& ir_path,
                      const std::string& save_path);
    
    /**
     * @brief 顯示計時結果
     */
    void showTimerResults();
    
private:
    // 配置
    const AppConfig& config_;
    
    // 模組 - 只使用 TRT fusion
    std::unique_ptr<ImageFusionTRT> fusion_trt_;
    std::shared_ptr<ImageAlignTensorRT> image_align_;
    HomographyManager homo_manager_;
    
    // 計時器
    Timer timer_resize_;
    Timer timer_gray_;
    Timer timer_align_;
    Timer timer_homo_;
    Timer timer_fusion_;
    Timer timer_edge_;
    
    // 私有方法
    
    /**
     * @brief 執行融合
     * @param eo_gray EO 灰階影像
     * @param ir_color IR 彩色影像
     * @param M Homography 矩陣
     * @return 融合後的影像
     */
    cv::Mat performFusion(const cv::Mat& eo_gray, const cv::Mat& ir_color, const cv::Mat& M);
    
    /**
     * @brief 計算並更新 homography
     * @param eo EO 影像
     * @param ir IR 影像
     * @param eo_pts 輸出 EO 特徵點
     * @param ir_pts 輸出 IR 特徵點
     * @param frame_cnt 幀數
     * @return Homography 矩陣
     */
    cv::Mat computeHomography(const cv::Mat& eo, const cv::Mat& ir,
                              std::vector<cv::Point2i>& eo_pts,
                              std::vector<cv::Point2i>& ir_pts,
                              int frame_cnt);
    
    /**
     * @brief 組合輸出影像
     * @param ir_original 原始 IR
     * @param eo_original 原始 EO
     * @param ir_processed 處理後 IR
     * @param eo_warped Warp 後 EO
     * @param fused 融合影像
     * @return 組合後的影像
     */
    cv::Mat composeOutput(const cv::Mat& ir_original, const cv::Mat& eo_original,
                          const cv::Mat& ir_processed, const cv::Mat& eo_warped,
                          const cv::Mat& fused);
};

} // namespace core

#endif // IMAGE_PROCESSOR_H
