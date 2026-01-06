

#include <iostream>
#include <filesystem>
#include <string>

#include "utils/src/util_timer.cpp"

#include "lib_image_fusion/src/core_image_align_tensorrt.cpp"
#include "lib_image_fusion/src/core_image_fusion_trt.cpp"
#include "lib_image_fusion/src/app_config.cpp"
#include "lib_image_fusion/src/app_utils.cpp"
#include "lib_image_fusion/src/homography_manager.cpp"
#include "lib_image_fusion/src/image_processor.cpp"

using namespace std;
using namespace std::filesystem;

int main(int argc, char** argv) {
    // ===== 1. 載入配置 =====
    string config_path = "./config/config.json";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    core::AppConfig config;
    if (!config.load(config_path)) {
        return -1;
    }
    
    // 驗證配置
    if (!config.validate()) {
        return -1;
    }
    
    // 顯示配置
    config.show();
    
    // ===== 2. 初始化處理器 =====
    core::ImageProcessor processor(config);
    if (!processor.initialize()) {
        cerr << "Failed to initialize processor" << endl;
        return -1;
    }
    
    // ===== 3. 處理輸入目錄中的所有檔案 =====
    for (const auto& file : directory_iterator(config.input_dir)) {
        // 獲取配對路徑
        string eo_path, ir_path;
        if (!core::utils::getPairPaths(file.path().string(), eo_path, ir_path)) {
            continue;
        }
        
        // 構建儲存路徑
        string save_path = config.output_dir;
        if (save_path.back() != '/' && save_path.back() != '\\') {
            save_path += "/";
        }
        save_path += core::utils::extractFileName(eo_path);
        
        cout << "\n========================================" << endl;
        cout << "Processing: " << core::utils::extractFileName(eo_path) << endl;
        cout << "========================================" << endl;
        
        // 根據檔案類型處理
        if (core::utils::isVideo(eo_path)) {
            cout << "[MODE] Video Processing" << endl;
            processor.processVideo(eo_path, ir_path, save_path);
        } else {
            cout << "[MODE] Image Processing" << endl;
            processor.processImage(eo_path, ir_path, save_path);
        }
        
        // 顯示計時結果
        processor.showTimerResults();
    }
    
    cout << "\n========================================" << endl;
    cout << "All processing completed!" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
