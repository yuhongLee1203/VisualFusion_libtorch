#include "app_utils.h"

#include <iostream>

namespace core {
namespace utils {

void alert(const std::string& msg)
{
  std::cout << "\033[1;31m[ ERROR ]\033[0m " << msg << std::endl;
}

bool is_file_exist(const std::string& path)
{
  bool res = std::filesystem::is_regular_file(path);
  if (!res) {
    alert("File not found: " + path);
  }
  return res;
}

bool is_dir_exist(const std::string& path)
{
  bool res = std::filesystem::is_directory(path);
  if (!res) {
    alert("Directory not found: " + path);
  }
  return res;
}

bool is_video(const std::string& path)
{
  std::vector<std::string> video_ext = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"};
  for (const auto& ext : video_ext) {
    if (path.find(ext) != std::string::npos) {
      return true;
    }
  }
  return false;
}

bool get_pair_paths(const std::string& path,
                    std::string& eo_path,
                    std::string& ir_path)
{
  ir_path = path;
  eo_path = path;

  if (path.find("_EO") != std::string::npos) {
    ir_path.replace(ir_path.find("_EO"), 3, "_IR");
  }
  else {
    return false;
  }

  // 檢查檔案是否存在（不顯示錯誤訊息）
  if (!std::filesystem::is_regular_file(eo_path) ||
      !std::filesystem::is_regular_file(ir_path)) {
    return false;
  }

  return true;
}

std::string extract_file_name(const std::string& path)
{
  std::string file = path.substr(path.find_last_of("/\\") + 1);
  size_t dot_pos = file.find_last_of(".");
  if (dot_pos != std::string::npos) {
    file = file.substr(0, dot_pos);
  }
  return file;
}

std::string extract_base_name(const std::string& path)
{
  std::string name = extract_file_name(path);

  // 移除 _EO 或 _IR 後綴
  size_t eo_pos = name.find("_EO");
  size_t ir_pos = name.find("_IR");
  if (eo_pos != std::string::npos) {
    name = name.substr(0, eo_pos);
  }
  else if (ir_pos != std::string::npos) {
    name = name.substr(0, ir_pos);
  }

  return name;
}

cv::Mat crop_image(const cv::Mat& src, int x, int y, int w, int h)
{
  int crop_x = std::max(0, x);
  int crop_y = std::max(0, y);
  int crop_w = w;
  int crop_h = h;

  if (w < 0) {
    crop_w = src.cols - crop_x;
  }
  if (h < 0) {
    crop_h = src.rows - crop_y;
  }

  crop_w = std::min(crop_w, src.cols - crop_x);
  crop_h = std::min(crop_h, src.rows - crop_y);

  cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
  return src(roi).clone();
}

void skip_frames(const std::string& path,
                 cv::VideoCapture& cap,
                 const nlohmann::json& skip_frames_config)
{
  if (skip_frames_config.empty()) {
    return;
  }

  std::string name = extract_file_name(path);

  if (skip_frames_config.contains(name)) {
    int skip = skip_frames_config[name];
    if (skip > 0) {
      cap.set(cv::CAP_PROP_POS_FRAMES, skip);
    }
  }
}

cv::Mat read_gt_homography(const std::string& gt_path, const std::string& img_name)
{
  std::string json_file = gt_path + "/IR_" + img_name + ".json";

  if (!std::filesystem::exists(json_file)) {
    std::cout << "GT file not found: " << json_file << std::endl;
    return cv::Mat();
  }

  try {
    std::ifstream file(json_file);
    nlohmann::json j;
    file >> j;

    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    auto h_array = j["H"];
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        H.at<double>(i, k) = h_array[i][k];
      }
    }
    std::cout << "GT homography loaded from: " << json_file << std::endl;
    return H;
  }
  catch (const std::exception& e) {
    std::cout << "Error reading GT homography from " << json_file << ": "
              << e.what() << std::endl;
    return cv::Mat();
  }
}

cv::Mat read_gt_homography_for_frame(const std::string& video_name,
                                     int frame_number,
                                     const std::string& gt_base_path)
{
  std::string base_name = video_name;

  // 移除 _EO 或 _IR 後綴
  size_t eo_pos = base_name.find("_EO");
  size_t ir_pos = base_name.find("_IR");
  if (eo_pos != std::string::npos) {
    base_name = base_name.substr(0, eo_pos);
  }
  else if (ir_pos != std::string::npos) {
    base_name = base_name.substr(0, ir_pos);
  }

  // 移除副檔名
  size_t dot_pos = base_name.find_last_of(".");
  if (dot_pos != std::string::npos) {
    base_name = base_name.substr(0, dot_pos);
  }

  // 構建 GT 檔案路徑
  std::string gt_path = gt_base_path + "/" + base_name + "_IR";
  std::string json_file = gt_path + "/IR_" + std::to_string(frame_number) + ".json";

  if (!std::filesystem::exists(json_file)) {
    return cv::Mat();
  }

  try {
    std::ifstream file(json_file);
    nlohmann::json j;
    file >> j;

    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    auto h_array = j["H"];
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        H.at<double>(i, k) = h_array[i][k];
      }
    }
    std::cout << "GT homography loaded for frame " << frame_number << std::endl;
    return H;
  }
  catch (const std::exception& e) {
    std::cout << "Error reading GT: " << e.what() << std::endl;
    return cv::Mat();
  }
}

double calc_feature_point_mse(const cv::Mat& homo_pred,
                              const cv::Mat& homo_gt,
                              const std::vector<cv::Point2i>& eo_pts)
{
  if (homo_pred.empty() || homo_gt.empty() || eo_pts.empty()) {
    return -1.0;
  }

  // 將 EO 特徵點轉換為 float 格式
  std::vector<cv::Point2f> eo_pts_f;
  for (const auto& pt : eo_pts) {
    eo_pts_f.push_back(
        cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
  }

  // 預測點
  std::vector<cv::Point2f> kpts_pred;
  cv::perspectiveTransform(eo_pts_f, kpts_pred, homo_pred);

  // GT 點
  std::vector<cv::Point2f> kpts_gt;
  cv::perspectiveTransform(eo_pts_f, kpts_gt, homo_gt);

  // 計算 MSE
  double total_squared_error = 0.0;
  int valid_points = 0;

  for (size_t i = 0; i < kpts_pred.size() && i < kpts_gt.size(); ++i) {
    double dx = kpts_pred[i].x - kpts_gt[i].x;
    double dy = kpts_pred[i].y - kpts_gt[i].y;
    total_squared_error += dx * dx + dy * dy;
    valid_points++;
  }

  if (valid_points == 0) {
    return -1.0;
  }

  return total_squared_error / valid_points;
}

cv::Mat refine_homography_with_ransac(std::vector<cv::Point2i>& eo_pts,
                                      std::vector<cv::Point2i>& ir_pts,
                                      const cv::Mat& initial_H,
                                      double ransac_threshold)
{
  if (eo_pts.size() < 4 || ir_pts.size() < 4) {
    return initial_H;
  }

  // 轉換為 float
  std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
  for (const auto& pt : eo_pts) {
    eo_pts_f.push_back(
        cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
  }
  for (const auto& pt : ir_pts) {
    ir_pts_f.push_back(
        cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
  }

  cv::Mat mask;
  cv::Mat H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC,
                                 ransac_threshold, mask, 3000, 0.99);

  if (H.empty() || mask.empty()) {
    return initial_H;
  }

  int inliers = cv::countNonZero(mask);
  if (inliers < 4 || cv::determinant(H) < 1e-6 || cv::determinant(H) > 1e6) {
    return initial_H;
  }

  // 過濾 inlier 特徵點
  std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
  for (int i = 0; i < mask.rows; i++) {
    if (mask.at<uchar>(i, 0) > 0) {
      filtered_eo_pts.push_back(eo_pts[i]);
      filtered_ir_pts.push_back(ir_pts[i]);
    }
  }

  eo_pts = filtered_eo_pts;
  ir_pts = filtered_ir_pts;

  return H;
}

cv::Mat warp_with_homography(const cv::Mat& src,
                             const cv::Mat& M,
                             const cv::Size& size,
                             int interp)
{
  if (src.empty()) {
    return src;
  }

  if (M.empty() || cv::determinant(M) < 1e-6) {
    cv::Mat result;
    cv::resize(src, result, size, 0, 0, interp);
    return result;
  }

  cv::Mat warped;
  cv::warpPerspective(src, warped, M, size, interp);
  return warped;
}

cv::Mat combine_images_horizontal(const std::vector<cv::Mat>& images)
{
  if (images.empty()) {
    return cv::Mat();
  }

  int total_width = 0;
  int max_height = 0;

  for (const auto& img : images) {
    total_width += img.cols;
    max_height = std::max(max_height, img.rows);
  }

  cv::Mat result = cv::Mat::zeros(max_height, total_width, images[0].type());

  int x_offset = 0;
  for (const auto& img : images) {
    img.copyTo(result(cv::Rect(x_offset, 0, img.cols, img.rows)));
    x_offset += img.cols;
  }

  return result;
}

void write_error_to_csv(
    const std::string& filename,
    const std::string& name,
    double error,
    const std::vector<std::pair<std::string, std::string>>& extra_cols)
{
  std::ofstream csv_file;
  bool file_exists = std::filesystem::exists(filename);
  csv_file.open(filename, std::ios::app);

  if (!file_exists) {
    csv_file << "Name,Error";
    for (const auto& col : extra_cols) {
      csv_file << "," << col.first;
    }
    csv_file << "\n";
  }

  csv_file << name << "," << error;
  for (const auto& col : extra_cols) {
    csv_file << "," << col.second;
  }
  csv_file << "\n";

  csv_file.close();
}

// ====== Homography 快取功能實作 (單一檔案模式) ======

bool save_homography_to_cache(const std::string& cache_file_path, const cv::Mat& H)
{
  if (H.empty()) {
    return false;
  }

  // 確保快取目錄存在
  std::filesystem::path p(cache_file_path);
  if (p.has_parent_path() && !std::filesystem::exists(p.parent_path())) {
    std::filesystem::create_directories(p.parent_path());
  }

  try {
    nlohmann::json j;
    nlohmann::json h_array = nlohmann::json::array();

    for (int i = 0; i < 3; i++) {
      nlohmann::json row = nlohmann::json::array();
      for (int k = 0; k < 3; k++) {
        row.push_back(H.at<double>(i, k));
      }
      h_array.push_back(row);
    }

    j["H"] = h_array;

    std::ofstream file(cache_file_path);
    file << j.dump(4);
    file.close();

    std::cout << "  [CACHE] Updated homography: " << cache_file_path << std::endl;
    return true;
  }
  catch (const std::exception& e) {
    std::cerr << "  [CACHE ERROR] Failed to save homography: " << e.what()
              << std::endl;
    return false;
  }
}

cv::Mat load_homography_from_cache(const std::string& cache_file_path)
{
  if (!std::filesystem::exists(cache_file_path)) {
    std::cerr << "  [CACHE] Homography cache not found: " << cache_file_path
              << std::endl;
    return cv::Mat();
  }

  try {
    std::ifstream file(cache_file_path);
    nlohmann::json j;
    file >> j;

    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    auto h_array = j["H"];
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        H.at<double>(i, k) = h_array[i][k];
      }
    }

    std::cout << "  [CACHE] Loaded homography from: " << cache_file_path
              << std::endl;
    return H;
  }
  catch (const std::exception& e) {
    std::cerr << "  [CACHE ERROR] Failed to load homography: " << e.what()
              << std::endl;
    return cv::Mat();
  }
}

bool is_homography_cache_exists(const std::string& cache_file_path)
{
  return std::filesystem::exists(cache_file_path);
}

}  // namespace utils
}  // namespace core
