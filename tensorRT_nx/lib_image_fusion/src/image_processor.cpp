#include "image_processor.h"
#include <iostream>

namespace core {

ImageProcessor::ImageProcessor(const AppConfig& config)
    : config_(config)
    , homo_manager_(config.smooth_max_translation_diff,
                    config.smooth_max_rotation_diff,
                    config.smooth_alpha)
    , timer_resize_("Resize")
    , timer_gray_("Gray")
    , timer_align_("Align")
    , timer_homo_("Homo")
    , timer_fusion_("Fusion")
    , timer_edge_("Edge") {
}

bool ImageProcessor::initialize() {
    // 初始化 TRT 融合模組
    fusion_trt_ = std::make_unique<ImageFusionTRT>(
        ImageFusionTRT::Param()
            .set_engine_path(config_.fusion_trt_engine)
            .set_size(config_.pred_width, config_.pred_height));
    
    if (!fusion_trt_->isInitialized()) {
        std::cerr << "\033[1;31m[ ERROR ]\033[0m TRT Fusion initialization failed!" << std::endl;
        std::cerr << "  Engine path: " << config_.fusion_trt_engine << std::endl;
        return false;
    }
    std::cout << "[INFO] Using TensorRT GPU fusion" << std::endl;
    
    // 初始化對齊模組
    image_align_ = ImageAlignTensorRT::create_instance(
        ImageAlignTensorRT::Param()
            .set_size(config_.pred_width, config_.pred_height, 
                      config_.output_width, config_.output_height)
            .set_engine(config_.model_path)
            .set_pred_mode(config_.pred_mode));
    
    return true;
}

cv::Mat ImageProcessor::performFusion(const cv::Mat& eo_gray, const cv::Mat& ir_color, 
                                       const cv::Mat& M) {
    int out_w = config_.output_width;
    int out_h = config_.output_height;
    int pred_w = config_.pred_width;
    int pred_h = config_.pred_height;
    
    // Warp EO gray
    cv::Mat eo_warped = utils::warpWithHomography(eo_gray, M, cv::Size(out_w, out_h));
    
    cv::Mat result;
    
    timer_fusion_.start();
    
    // 準備 TRT 輸入
    cv::Mat eo_for_trt, ir_for_trt;
    if (eo_warped.cols != pred_w || eo_warped.rows != pred_h) {
        cv::resize(eo_warped, eo_for_trt, cv::Size(pred_w, pred_h));
        cv::resize(ir_color, ir_for_trt, cv::Size(pred_w, pred_h));
    } else {
        eo_for_trt = eo_warped;
        ir_for_trt = ir_color;
    }
    
    cv::Mat fused = fusion_trt_->fusion(eo_for_trt, ir_for_trt);
    
    // Resize 回輸出尺寸
    if (fused.cols != out_w || fused.rows != out_h) {
        cv::resize(fused, result, cv::Size(out_w, out_h));
    } else {
        result = fused;
    }
    
    timer_fusion_.stop();
    
    return result;
}

cv::Mat ImageProcessor::computeHomography(const cv::Mat& eo, const cv::Mat& ir,
                                           std::vector<cv::Point2i>& eo_pts,
                                           std::vector<cv::Point2i>& ir_pts,
                                           int frame_cnt) {
    eo_pts.clear();
    ir_pts.clear();
    
    // 設定當前幀名稱
    std::string frame_name = "frame_" + std::to_string(frame_cnt);
    image_align_->set_current_image_name(frame_name);
    
    // 對齊
    timer_align_.start();
    cv::Mat M;
    image_align_->align(eo, ir, eo_pts, ir_pts, M);
    timer_align_.stop();
    
    std::cout << "  - Frame " << frame_cnt << ": Found " << eo_pts.size() 
              << " feature points" << std::endl;
    
    // RANSAC 優化
    timer_homo_.start();
    if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
        cv::Mat H = utils::refineHomographyWithRANSAC(eo_pts, ir_pts, M, 8.0);
        
        if (!H.empty() && cv::determinant(H) > 1e-6 && cv::determinant(H) < 1e6) {
            // 判斷是否接受新的 homography
            if (homo_manager_.getCurrent().empty()) {
                M = homo_manager_.update(H);
                homo_manager_.resetFallback();
                std::cout << "  - Frame " << frame_cnt << ": First homography computed" << std::endl;
            } else {
                auto [trans_diff, rot_diff] = homo_manager_.calculateDifference(
                    homo_manager_.getCurrent(), H);
                
                std::cout << "  - Frame " << frame_cnt << ": Trans diff=" << trans_diff 
                          << "px, Rot diff=" << rot_diff << "rad" << std::endl;
                
                if (trans_diff > config_.smooth_max_translation_diff || 
                    rot_diff > config_.smooth_max_rotation_diff) {
                    homo_manager_.incrementFallback();
                    std::cout << "    -> Difference too large, fallback=" 
                              << homo_manager_.getFallbackCount() << std::endl;
                    
                    if (homo_manager_.getFallbackCount() >= 3) {
                        std::cout << "    -> Force accept and reset!" << std::endl;
                        homo_manager_.reset();
                        homo_manager_.setParameters(config_.smooth_max_translation_diff,
                                                    config_.smooth_max_rotation_diff,
                                                    config_.smooth_alpha);
                        M = homo_manager_.update(H);
                    } else {
                        M = homo_manager_.getCurrent();
                    }
                } else {
                    std::cout << "    -> Smoothly updating (alpha=" 
                              << config_.smooth_alpha << ")" << std::endl;
                    M = homo_manager_.update(H);
                    homo_manager_.resetFallback();
                }
            }
        } else {
            M = homo_manager_.getCurrent();
            if (M.empty()) {
                M = cv::Mat::eye(3, 3, CV_64F);
            }
            std::cout << "  - Frame " << frame_cnt << ": Poor quality, using previous" << std::endl;
        }
    } else {
        M = homo_manager_.getCurrent();
        if (M.empty()) {
            M = cv::Mat::eye(3, 3, CV_64F);
        }
        std::cout << "  - Frame " << frame_cnt << ": Insufficient points, using previous" << std::endl;
    }
    timer_homo_.stop();
    
    return M;
}

cv::Mat ImageProcessor::composeOutput(const cv::Mat& ir_original, const cv::Mat& eo_original,
                                       const cv::Mat& ir_processed, const cv::Mat& eo_warped,
                                       const cv::Mat& fused) {
    int out_w = config_.output_width;
    int out_h = config_.output_height;
    
    // 創建輸出影像: IR原始 | EO原始 | IR處理 | EO變換 | 融合
    cv::Mat output = cv::Mat(out_h, out_w * 5, CV_8UC3);
    
    ir_original.copyTo(output(cv::Rect(0, 0, out_w, out_h)));
    eo_original.copyTo(output(cv::Rect(out_w, 0, out_w, out_h)));
    ir_processed.copyTo(output(cv::Rect(out_w * 2, 0, out_w, out_h)));
    eo_warped.copyTo(output(cv::Rect(out_w * 3, 0, out_w, out_h)));
    
    // 確保融合影像尺寸正確
    cv::Mat fused_resized = fused;
    if (fused.cols != out_w || fused.rows != out_h) {
        cv::resize(fused, fused_resized, cv::Size(out_w, out_h));
    }
    fused_resized.copyTo(output(cv::Rect(out_w * 4, 0, out_w, out_h)));
    
    return output;
}

bool ImageProcessor::processImage(const std::string& eo_path,
                                   const std::string& ir_path,
                                   const std::string& save_path) {
    int out_w = config_.output_width;
    int out_h = config_.output_height;
    
    // 讀取影像
    cv::Mat eo = cv::imread(eo_path);
    cv::Mat ir = cv::imread(ir_path);
    
    if (eo.empty() || ir.empty()) {
        std::cerr << "Failed to read images" << std::endl;
        return false;
    }
    
    // 裁剪
    if (config_.picture_cut_enabled) {
        eo = utils::cropImage(eo, config_.pcut_x, config_.pcut_y, 
                              config_.pcut_w, config_.pcut_h);
    }
    
    // Resize
    cv::Mat eo_resized, ir_resized;
    cv::resize(eo, eo_resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
    cv::resize(ir, ir_resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
    
    // 設定圖片名稱
    std::string img_name = utils::extractBaseName(eo_path);
    
    cv::Mat M;
    std::vector<cv::Point2i> eo_pts, ir_pts;
    
    if (config_.use_model_prediction) {
        // 模式一：使用 model 預測 homography
        std::cout << "  [MODE] Using model prediction" << std::endl;
        
        // 轉灰階
        cv::Mat gray_eo, gray_ir;
        cv::cvtColor(eo_resized, gray_eo, cv::COLOR_BGR2GRAY);
        cv::cvtColor(ir_resized, gray_ir, cv::COLOR_BGR2GRAY);
        
        image_align_->set_current_image_name(img_name);
        
        // 對齊
        image_align_->align(gray_eo, gray_ir, eo_pts, ir_pts, M);
        
        std::cout << "    Found " << eo_pts.size() << " feature points" << std::endl;
        
        // RANSAC 優化
        M = utils::refineHomographyWithRANSAC(eo_pts, ir_pts, M, 6.0);
        if (M.empty()) {
            M = cv::Mat::eye(3, 3, CV_64F);
        }
        
        // 儲存 homography 到快取 (覆蓋模式，單一檔案)
        utils::saveHomographyToCache(config_.homo_cache_file, M);
        
    } else {
        // 模式二：從快取載入 homography
        std::cout << "  [MODE] Loading homography from cache" << std::endl;
        
        M = utils::loadHomographyFromCache(config_.homo_cache_file);
        if (M.empty()) {
            std::cerr << "  [ERROR] Failed to load homography from cache, using identity matrix" << std::endl;
            M = cv::Mat::eye(3, 3, CV_64F);
        }
    }
    
    // Warp EO
    cv::Mat eo_warped = utils::warpWithHomography(eo_resized, M, cv::Size(out_w, out_h));
    
    // 融合
    cv::Mat gray_eo_warped;
    cv::cvtColor(eo_warped, gray_eo_warped, cv::COLOR_BGR2GRAY);
    cv::Mat fused = performFusion(gray_eo_warped, ir_resized, cv::Mat::eye(3, 3, CV_64F));
    
    // 組合輸出
    cv::Mat output = composeOutput(ir_resized, eo_resized, ir_resized, eo_warped, fused);
    
    // 儲存
    if (config_.output_enabled) {
        cv::imwrite(save_path + ".jpg", output);
        std::cout << "Saved to: " << save_path << ".jpg" << std::endl;
    }
    
    // 計算誤差 (如果有 GT 且使用 model 預測)
    if (config_.use_model_prediction) {
        cv::Mat gt_homo = utils::readGTHomography(config_.gt_homo_base_path, img_name);
        if (!gt_homo.empty() && !eo_pts.empty()) {
            double mse = utils::calcFeaturePointMSE(M, gt_homo, eo_pts);
            std::cout << "    MSE Error: " << mse << " px^2" << std::endl;
            
            utils::writeErrorToCSV("image_homo_errors.csv", img_name, mse);
        }
    }
    
    return true;
}

bool ImageProcessor::processVideo(const std::string& eo_path,
                                   const std::string& ir_path,
                                   const std::string& save_path) {
    int out_w = config_.output_width;
    int out_h = config_.output_height;
    
    // 開啟影片
    cv::VideoCapture eo_cap(eo_path);
    cv::VideoCapture ir_cap(ir_path);
    
    if (!eo_cap.isOpened() || !ir_cap.isOpened()) {
        std::cerr << "Failed to open videos" << std::endl;
        return false;
    }
    
    // 跳過幀數
    utils::skipFrames(eo_path, eo_cap, config_.skip_frames_config);
    utils::skipFrames(ir_path, ir_cap, config_.skip_frames_config);
    
    // 獲取影片資訊
    int fps_ir = static_cast<int>(ir_cap.get(cv::CAP_PROP_FPS));
    int fps_eo = static_cast<int>(eo_cap.get(cv::CAP_PROP_FPS));
    int frame_rate = fps_ir / fps_eo;
    
    std::cout << "  - IR: " << fps_ir << " fps" << std::endl;
    std::cout << "  - EO: " << fps_eo << " fps" << std::endl;
    std::cout << "  - Rate: " << frame_rate << std::endl;
    std::cout << "  - Use Model Prediction: " << (config_.use_model_prediction ? "true" : "false") << std::endl;
    
    // 創建輸出影片
    cv::VideoWriter writer;
    if (config_.output_enabled) {
        std::string mode_suffix = config_.use_model_prediction ? "_model" : "_cache";
        std::string output_filename = save_path + "_" + 
            std::to_string(config_.compute_per_frame) + mode_suffix + "_fusion.mp4";
        writer.open(output_filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                    fps_ir, cv::Size(out_w * 5, out_h));
    }
    
    // 重置 homography 管理器
    homo_manager_.reset();
    homo_manager_.setParameters(config_.smooth_max_translation_diff,
                                config_.smooth_max_rotation_diff,
                                config_.smooth_alpha);
    
    // 處理幀
    cv::Mat eo, ir;
    cv::Mat M = cv::Mat::eye(3, 3, CV_64F);
    std::vector<cv::Point2i> eo_pts, ir_pts;
    int cnt = 15;
    
    std::string video_name = utils::extractFileName(eo_path);
    
    while (true) {
        ir_cap.read(ir);
        eo_cap.read(eo);
        
        if (eo.empty() || ir.empty()) {
            break;
        }
        
        // 裁剪
        if (config_.video_cut_enabled) {
            eo = utils::cropImage(eo, config_.vcut_x, config_.vcut_y,
                                  config_.vcut_w, config_.vcut_h);
        }
        
        // Resize
        timer_resize_.start();
        cv::Mat img_eo, img_ir;
        cv::resize(eo, img_eo, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        cv::resize(ir, img_ir, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);
        timer_resize_.stop();
        
        // 轉灰階
        timer_gray_.start();
        cv::Mat gray_eo, gray_ir;
        cv::cvtColor(img_eo, gray_eo, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_ir, gray_ir, cv::COLOR_BGR2GRAY);
        timer_gray_.stop();
        
        // 計算 homography (根據頻率)
        if (cnt % config_.compute_per_frame == 0) {
            if (config_.use_model_prediction) {
                // 模式一：使用 model 預測 homography
                M = computeHomography(img_eo, img_ir, eo_pts, ir_pts, cnt);
                
                // 儲存 homography 到快取 (覆蓋模式，單一檔案)
                utils::saveHomographyToCache(config_.homo_cache_file, M);
            } else {
                // 模式二：從快取載入 homography
                cv::Mat cached_H = utils::loadHomographyFromCache(config_.homo_cache_file);
                if (!cached_H.empty()) {
                    M = homo_manager_.update(cached_H);
                } else {
                    std::cerr << "  [WARNING] Frame " << cnt << ": Cache not found, using previous" << std::endl;
                    M = homo_manager_.getCurrent();
                    if (M.empty()) {
                        M = cv::Mat::eye(3, 3, CV_64F);
                    }
                }
            }
        } else {
            M = homo_manager_.getCurrent();
            if (M.empty()) {
                M = cv::Mat::eye(3, 3, CV_64F);
            }
        }
        
        // Warp EO
        cv::Mat eo_warped = utils::warpWithHomography(img_eo, M, cv::Size(out_w, out_h));
        
        // 融合
        cv::Mat fused = performFusion(gray_eo, img_ir, M);
        
        // 組合輸出
        cv::Mat output = composeOutput(img_ir, img_eo, img_ir, eo_warped, fused);
        
        // 寫入影片
        if (config_.output_enabled && writer.isOpened()) {
            writer.write(output);
        }
        
        // 計算誤差 (如果有 GT 且有特徵點 且使用 model 預測)
        if (config_.use_model_prediction && !eo_pts.empty()) {
            cv::Mat gt_homo = utils::readGTHomographyForFrame(video_name, cnt, 
                                                              config_.gt_video_base_path);
            if (!gt_homo.empty()) {
                double mse = utils::calcFeaturePointMSE(M, gt_homo, eo_pts);
                std::cout << "    [Frame " << cnt << "] MSE: " << mse << " px^2" << std::endl;
                
                std::vector<std::pair<std::string, std::string>> extra_cols = {
                    {"Frame", std::to_string(cnt)},
                    {"ComputePerFrame", std::to_string(config_.compute_per_frame)}
                };
                utils::writeErrorToCSV("video_homo_errors.csv", video_name, mse, extra_cols);
            }
        }
        
        // 跳過 IR 幀
        for (int i = 0; i < frame_rate - 1; i++) {
            cv::Mat temp;
            ir_cap.read(temp);
        }
        
        cnt++;
    }
    
    // 釋放資源
    eo_cap.release();
    ir_cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    
    return true;
}

void ImageProcessor::showTimerResults() {
    timer_resize_.show();
    timer_gray_.show();
    timer_align_.show();
    timer_homo_.show();
    timer_edge_.show();
    timer_fusion_.show();
}

} // namespace core
