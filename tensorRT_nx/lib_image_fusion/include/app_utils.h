#ifndef APP_UTILS_H
#define APP_UTILS_H

#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "nlohmann/json.hpp"

namespace core {

/**
 * @brief 應用程式工具函數集
 */
namespace utils {

/**
 * @brief 顯示錯誤訊息
 * @param msg 錯誤訊息
 */
void alert(const std::string& msg);

/**
 * @brief 檢查檔案是否存在
 * @param path 檔案路徑
 * @return 是否存在
 */
bool isFileExist(const std::string& path);

/**
 * @brief 檢查目錄是否存在
 * @param path 目錄路徑
 * @return 是否存在
 */
bool isDirExist(const std::string& path);

/**
 * @brief 檢查檔案是否為影片
 * @param path 檔案路徑
 * @return 是否為影片
 */
bool isVideo(const std::string& path);

/**
 * @brief 獲取配對的 EO/IR 檔案路徑
 * @param path 輸入路徑
 * @param eo_path 輸出 EO 路徑
 * @param ir_path 輸出 IR 路徑
 * @return 是否成功找到配對
 */
bool getPairPaths(const std::string& path, std::string& eo_path, std::string& ir_path);

/**
 * @brief 從路徑提取檔案名稱（不含副檔名）
 * @param path 檔案路徑
 * @return 檔案名稱
 */
std::string extractFileName(const std::string& path);

/**
 * @brief 從路徑提取基礎名稱（移除 _EO/_IR 和副檔名）
 * @param path 檔案路徑
 * @return 基礎名稱
 */
std::string extractBaseName(const std::string& path);

/**
 * @brief 裁剪影像
 * @param src 來源影像
 * @param x X 座標
 * @param y Y 座標
 * @param w 寬度 (-1 表示到邊界)
 * @param h 高度 (-1 表示到邊界)
 * @return 裁剪後的影像
 */
cv::Mat cropImage(const cv::Mat& src, int x, int y, int w, int h);

/**
 * @brief 跳過影片幀數
 * @param path 影片路徑
 * @param cap VideoCapture 物件
 * @param skip_frames_config 跳過配置
 */
void skipFrames(const std::string& path, cv::VideoCapture& cap, 
                const nlohmann::json& skip_frames_config);

/**
 * @brief 讀取 GT homography (圖片模式)
 * @param gt_path GT 基礎路徑
 * @param img_name 圖片名稱
 * @return Homography 矩陣
 */
cv::Mat readGTHomography(const std::string& gt_path, const std::string& img_name);

/**
 * @brief 讀取 GT homography (影片模式，根據幀數)
 * @param video_name 影片名稱
 * @param frame_number 幀數
 * @param gt_base_path GT 基礎路徑
 * @return Homography 矩陣
 */
cv::Mat readGTHomographyForFrame(const std::string& video_name, int frame_number,
                                  const std::string& gt_base_path);

/**
 * @brief 計算特徵點 MSE 誤差
 * @param homo_pred 預測的 homography
 * @param homo_gt GT homography
 * @param eo_pts EO 特徵點
 * @return MSE 誤差值
 */
double calcFeaturePointMSE(const cv::Mat& homo_pred, const cv::Mat& homo_gt,
                           const std::vector<cv::Point2i>& eo_pts);

/**
 * @brief 使用 RANSAC 優化 homography 並過濾離群點
 * @param eo_pts EO 特徵點 (輸入/輸出)
 * @param ir_pts IR 特徵點 (輸入/輸出)
 * @param initial_H 初始 homography
 * @param ransac_threshold RANSAC 閾值
 * @return 優化後的 homography
 */
cv::Mat refineHomographyWithRANSAC(std::vector<cv::Point2i>& eo_pts,
                                    std::vector<cv::Point2i>& ir_pts,
                                    const cv::Mat& initial_H,
                                    double ransac_threshold = 6.0);

/**
 * @brief 對影像應用 homography 變換
 * @param src 來源影像
 * @param M Homography 矩陣
 * @param size 輸出尺寸
 * @param interp 插值方式
 * @return 變換後的影像
 */
cv::Mat warpWithHomography(const cv::Mat& src, const cv::Mat& M, 
                           const cv::Size& size, int interp = cv::INTER_LINEAR);

/**
 * @brief 組合多張影像成一行
 * @param images 影像列表
 * @return 組合後的影像
 */
cv::Mat combineImagesHorizontal(const std::vector<cv::Mat>& images);

/**
 * @brief 寫入誤差到 CSV
 * @param filename CSV 檔案名稱
 * @param name 名稱
 * @param error 誤差值
 * @param extra_cols 額外欄位 (可選)
 */
void writeErrorToCSV(const std::string& filename, const std::string& name,
                     double error, const std::vector<std::pair<std::string, std::string>>& extra_cols = {});

// ====== Homography 快取功能 (單一檔案模式) ======

/**
 * @brief 儲存 homography 矩陣到檔案 (覆蓋模式，永遠只有一個檔案)
 * @param cache_file_path 快取檔案完整路徑
 * @param H Homography 矩陣
 * @return 是否成功
 */
bool saveHomographyToCache(const std::string& cache_file_path, const cv::Mat& H);

/**
 * @brief 從快取載入 homography 矩陣
 * @param cache_file_path 快取檔案完整路徑
 * @return Homography 矩陣 (空矩陣表示失敗)
 */
cv::Mat loadHomographyFromCache(const std::string& cache_file_path);

/**
 * @brief 檢查快取是否存在
 * @param cache_file_path 快取檔案完整路徑
 * @return 是否存在
 */
bool isHomographyCacheExists(const std::string& cache_file_path);

} // namespace utils
} // namespace core

#endif // APP_UTILS_H
