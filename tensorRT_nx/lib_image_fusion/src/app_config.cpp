#include "app_config.h"

namespace core {

void AppConfig::initDefaults(nlohmann::json& config) {
    config.emplace("input_dir", "./input");
    config.emplace("output_dir", "./output");
    config.emplace("output", false);
    
    config.emplace("device", "cpu");
    config.emplace("pred_mode", "fp32");
    config.emplace("model_path", "./model/SemLA_jit_cpu.zip");
    
    config.emplace("VideoCut", false);
    config.emplace("Vcut_x", 0);
    config.emplace("Vcut_y", 0);
    config.emplace("Vcut_h", -1);
    config.emplace("Vcut_w", -1);
    
    config.emplace("PictureCut", false);
    config.emplace("Pcut_x", 0);
    config.emplace("Pcut_y", 0);
    config.emplace("Pcut_h", -1);
    config.emplace("Pcut_w", -1);
    
    config.emplace("compute_per_frame", 2);
    
    config.emplace("output_width", 480);
    config.emplace("output_height", 360);
    config.emplace("pred_width", 320);
    config.emplace("pred_height", 240);
    
    config.emplace("fusion_shadow", true);
    config.emplace("fusion_edge_border", 2);
    config.emplace("fusion_threshold_equalization", 128);
    config.emplace("fusion_threshold_equalization_low", 72);
    config.emplace("fusion_threshold_equalization_high", 192);
    config.emplace("fusion_threshold_equalization_zero", 64);
    config.emplace("fusion_trt_engine", "./model/NX/image_fusion_v2_320x240_fp32.trt");
    config.emplace("use_trt_fusion", true);
    
    config.emplace("align_angle_sort", 0.6);
    config.emplace("align_angle_mean", 10.0);
    config.emplace("align_distance_last", 10.0);
    config.emplace("align_distance_line", 10.0);
    
    config.emplace("smooth_max_translation_diff", 15.0);
    config.emplace("smooth_max_rotation_diff", 0.02);
    config.emplace("smooth_alpha", 0.03);
    
    config.emplace("skip_frames", nlohmann::json::object());
    
    config.emplace("gt_homo_base_path", "/circ330/HomoLabels320240/Version3");
    config.emplace("gt_video_base_path", "/circ330/HomoLabels2023_990");
    
    // Homography 快取模式 - 新增
    config.emplace("use_model_prediction", true);
    config.emplace("homo_cache_file", "./current_homo.json");
    
    // Pipeline 控制參數 - 新增
    config.emplace("align_start_frame", 15);   // 從第 15 幀開始執行 align
    config.emplace("align_stop_frame", -1);    // -1 表示永不停止，會一直執行
    config.emplace("align_on_first_frame", true);  // 第一幀強制執行 align
}

void AppConfig::parseFromJson(const nlohmann::json& config) {
    // 輸入輸出
    input_dir = config["input_dir"];
    output_dir = config["output_dir"];
    output_enabled = config["output"];
    
    // 影像尺寸
    output_width = config["output_width"];
    output_height = config["output_height"];
    pred_width = config["pred_width"];
    pred_height = config["pred_height"];
    
    // 裁剪參數 - 影片
    video_cut_enabled = config["VideoCut"];
    vcut_x = config["Vcut_x"];
    vcut_y = config["Vcut_y"];
    vcut_w = config["Vcut_w"];
    vcut_h = config["Vcut_h"];
    
    // 裁剪參數 - 圖片
    picture_cut_enabled = config["PictureCut"];
    pcut_x = config["Pcut_x"];
    pcut_y = config["Pcut_y"];
    pcut_w = config["Pcut_w"];
    pcut_h = config["Pcut_h"];
    
    // 模型相關
    device = config["device"];
    pred_mode = config["pred_mode"];
    model_path = config["model_path"];
    
    // 融合參數
    fusion_shadow = config["fusion_shadow"];
    fusion_edge_border = config["fusion_edge_border"];
    fusion_threshold_equalization = config["fusion_threshold_equalization"];
    fusion_threshold_equalization_low = config["fusion_threshold_equalization_low"];
    fusion_threshold_equalization_high = config["fusion_threshold_equalization_high"];
    fusion_threshold_equalization_zero = config["fusion_threshold_equalization_zero"];
    fusion_trt_engine = config["fusion_trt_engine"];
    use_trt_fusion = config["use_trt_fusion"];
    
    // 對齊參數
    align_angle_mean = config["align_angle_mean"];
    align_angle_sort = config["align_angle_sort"];
    align_distance_last = config["align_distance_last"];
    align_distance_line = config["align_distance_line"];
    
    // 平滑參數
    smooth_max_translation_diff = config["smooth_max_translation_diff"];
    smooth_max_rotation_diff = config["smooth_max_rotation_diff"];
    smooth_alpha = config["smooth_alpha"];
    
    // 計算頻率
    compute_per_frame = config["compute_per_frame"];
    
    // GT 路徑
    if (config.contains("gt_homo_base_path")) {
        gt_homo_base_path = config["gt_homo_base_path"];
    } else {
        gt_homo_base_path = "/circ330/HomoLabels320240/Version3";
    }
    if (config.contains("gt_video_base_path")) {
        gt_video_base_path = config["gt_video_base_path"];
    } else {
        gt_video_base_path = "/circ330/HomoLabels2023_990";
    }
    
    // skip frames
    skip_frames_config = config["skip_frames"];
    
    // Homography 快取模式 - 新增
    use_model_prediction = config["use_model_prediction"];
    homo_cache_file = config["homo_cache_file"];
    
    // Pipeline 控制參數 - 新增
    align_start_frame = config["align_start_frame"];
    align_stop_frame = config["align_stop_frame"];
    align_on_first_frame = config["align_on_first_frame"];
}

bool AppConfig::load(const std::string& config_path) {
    // 檢查檔案是否存在
    if (!std::filesystem::is_regular_file(config_path)) {
        std::cout << "\033[1;31m[ ERROR ]\033[0m Config file not found: " << config_path << std::endl;
        return false;
    }
    
    try {
        // 讀取 JSON 檔案
        std::ifstream file(config_path);
        nlohmann::json config;
        file >> config;
        
        // 初始化預設值
        initDefaults(config);
        
        // 解析配置
        parseFromJson(config);
        
        return true;
    } catch (const std::exception& e) {
        std::cout << "\033[1;31m[ ERROR ]\033[0m Failed to parse config: " << e.what() << std::endl;
        return false;
    }
}

void AppConfig::show() const {
    std::cout << "[ Config ]" << std::endl;
    std::cout << "\tOutput Size: " << output_width << " x " << output_height << std::endl;
    std::cout << "\tPredict Size: " << pred_width << " x " << pred_height << std::endl;
    std::cout << "\tModel Path: " << model_path << std::endl;
    std::cout << "\tDevice: " << device << std::endl;
    std::cout << "\tPred Mode: " << pred_mode << std::endl;
    std::cout << "\tFusion Shadow: " << fusion_shadow << std::endl;
    std::cout << "\tFusion Edge Border: " << fusion_edge_border << " (CPU only)" << std::endl;
    std::cout << "\tUse TRT Fusion: " << use_trt_fusion << std::endl;
    std::cout << "\tFusion TRT Engine: " << fusion_trt_engine << std::endl;
    std::cout << "\tSmooth Max Translation Diff: " << smooth_max_translation_diff << std::endl;
    std::cout << "\tSmooth Max Rotation Diff: " << smooth_max_rotation_diff << std::endl;
    std::cout << "\tSmooth Alpha: " << smooth_alpha << std::endl;
    std::cout << "\tCompute Per Frame: " << compute_per_frame << std::endl;
    std::cout << "\tUse Model Prediction: " << (use_model_prediction ? "true" : "false") << std::endl;
    std::cout << "\tHomo Cache File: " << homo_cache_file << std::endl;
    std::cout << "\tAlign Start Frame: " << align_start_frame << std::endl;
    std::cout << "\tAlign Stop Frame: " << align_stop_frame << " (-1 = never stop)" << std::endl;
    std::cout << "\tAlign on First Frame: " << (align_on_first_frame ? "true" : "false") << std::endl;
}

bool AppConfig::validate() const {
    // 檢查輸入目錄
    if (!std::filesystem::is_directory(input_dir)) {
        std::cout << "\033[1;31m[ ERROR ]\033[0m Input directory not found: " << input_dir << std::endl;
        return false;
    }
    
    // 檢查輸出目錄
    if (output_enabled && !std::filesystem::is_directory(output_dir)) {
        std::cout << "\033[1;31m[ ERROR ]\033[0m Output directory not found: " << output_dir << std::endl;
        return false;
    }
    
    std::cout << "[ Directories ]" << std::endl;
    std::cout << "\t Input: " << input_dir << std::endl;
    if (output_enabled) {
        std::cout << "\tOutput: " << output_dir << std::endl;
    }
    
    return true;
}

} // namespace core
