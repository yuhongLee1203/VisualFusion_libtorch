#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "nlohmann/json.hpp"

namespace core {

/**
 * @brief 應用程式配置管理類
 * 負責讀取和管理所有配置參數
 */
class AppConfig {
public:
    // 輸入輸出相關
    std::string input_dir;
    std::string output_dir;
    bool output_enabled;
    
    // 影像尺寸
    int output_width;
    int output_height;
    int pred_width;
    int pred_height;
    
    // 裁剪參數 - 影片
    bool video_cut_enabled;
    int vcut_x, vcut_y, vcut_w, vcut_h;
    
    // 裁剪參數 - 圖片
    bool picture_cut_enabled;
    int pcut_x, pcut_y, pcut_w, pcut_h;
    
    // 模型相關
    std::string device;
    std::string pred_mode;
    std::string model_path;
    
    // 融合參數
    bool fusion_shadow;
    int fusion_edge_border;
    int fusion_threshold_equalization;
    int fusion_threshold_equalization_low;
    int fusion_threshold_equalization_high;
    int fusion_threshold_equalization_zero;
    std::string fusion_trt_engine;
    bool use_trt_fusion;
    
    // 對齊參數
    float align_angle_mean;
    float align_angle_sort;
    float align_distance_last;
    float align_distance_line;
    
    // 平滑 homography 參數
    double smooth_max_translation_diff;
    double smooth_max_rotation_diff;
    double smooth_alpha;
    
    // 計算頻率
    int compute_per_frame;
    
    // GT 路徑
    std::string gt_homo_base_path;
    std::string gt_video_base_path;
    
    // skip frames
    nlohmann::json skip_frames_config;
    
    /**
     * @brief 從檔案載入配置
     * @param config_path 配置檔案路徑
     * @return 是否成功載入
     */
    bool load(const std::string& config_path);
    
    /**
     * @brief 顯示當前配置
     */
    void show() const;
    
    /**
     * @brief 驗證配置是否有效
     * @return 是否有效
     */
    bool validate() const;
    
private:
    /**
     * @brief 初始化預設配置值
     */
    void initDefaults(nlohmann::json& config);
    
    /**
     * @brief 從 JSON 解析配置
     */
    void parseFromJson(const nlohmann::json& config);
};

} // namespace core

#endif // APP_CONFIG_H
