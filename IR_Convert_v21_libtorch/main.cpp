// 直接將一通道餵給三通道

#include <ratio>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "lib_image_fusion/include/core_image_to_gray.h"
#include "lib_image_fusion/src/core_image_to_gray.cpp"

#include "lib_image_fusion/include/core_image_resizer.h"
#include "lib_image_fusion/src/core_image_resizer.cpp"

#include "lib_image_fusion/include/core_image_fusion.h"
#include "lib_image_fusion/src/core_image_fusion.cpp"

#include "lib_image_fusion/include/core_image_perspective.h"
#include "lib_image_fusion/src/core_image_perspective.cpp"

// #include "lib_image_fusion/include/core_image_align_onnx_polar.h"
// #include "lib_image_fusion/src/core_image_align_onnx_polar.cpp"

#include "lib_image_fusion/include/core_image_align_libtorch.h"
#include "lib_image_fusion/src/core_image_align_libtorch.cpp"

#include "utils/include/util_timer.h"
#include "utils/src/util_timer.cpp"

#include "nlohmann/json.hpp"

using namespace cv;
using namespace std;
using namespace filesystem;
using json = nlohmann::json;

// show error message
inline void alert(const string &msg)
{
  std::cout << string("\033[1;31m[ ERROR ]\033[0m ") + msg << std::endl;
}

// check file exit
inline bool is_file_exit(const string &path)
{
  bool res = is_regular_file(path);
  if (!res)
    alert(string("File not found: ") + path);
  return res;
}

// check directory exit
inline bool is_dir_exit(const string &path)
{
  bool res = is_directory(path);
  if (!res)
    alert(string("File not found: ") + path);
  return res;
}

// init config
inline void init_config(nlohmann::json &config)
{
  config.emplace("input_dir", "./input");
  config.emplace("output_dir", "./output");
  config.emplace("output", false);

  config.emplace("device", "cpu");
  config.emplace("pred_mode", "fp32");
  config.emplace("model_path", "./model/SemLA_jit_cpu.zip");

  config.emplace("output_width", 640);
  config.emplace("output_height", 512);

  config.emplace("pred_width", 320);
  config.emplace("pred_height", 256);

  config.emplace("fusion_shadow", false);
  config.emplace("fusion_edge_border", 1);
  config.emplace("fusion_threshold_equalization", 128);
  config.emplace("fusion_threshold_equalization_low", 72);
  config.emplace("fusion_threshold_equalization_high", 192);
  config.emplace("fusion_threshold_equalization_zero", 64);

  config.emplace("perspective_check", true);
  config.emplace("perspective_distance", 10);
  config.emplace("perspective_accuracy", 0.85);

  config.emplace("align_angle_sort", 0.6);
  config.emplace("align_angle_mean", 10.0);
  config.emplace("align_distance_last", 10.0);
  config.emplace("align_distance_line", 10.0);

  config.emplace("skip_frames", nlohmann::json::object());
}

// get pair file
inline bool get_pair(const string &path, string &eo_path, string &ir_path)
{
  ir_path = path;
  eo_path = path;

  if (path.find("_EO") != string::npos)
    ir_path.replace(ir_path.find("_EO"), 3, "_IR");
  else
    return false;

  // 檢查檔案是否存在
  if (!is_file_exit(eo_path))
    return false;
  if (!is_file_exit(ir_path))
    return false;

  return true;
}

// check file is video or image
inline bool is_video(const string &path)
{
  std::vector<string> video_ext = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"};
  for (const string &ext : video_ext)
    if (path.find(ext) != string::npos)
      return true;
  return false;
}

// skip frames
inline void skip_frames(const string &path, cv::VideoCapture &cap, nlohmann::json &config)
{
  nlohmann::json skip_frames = config["skip_frames"];
  if (skip_frames.empty())
    return;

  string file = path.substr(path.find_last_of("/\\") + 1);
  string name = file.substr(0, file.find_last_of("."));

  int skip = 0;

  if (skip_frames.contains(name))
    skip = skip_frames[name];

  if (skip > 0)
    cap.set(cv::CAP_PROP_POS_FRAMES, skip);
}

int main(int argc, char **argv)
{
  // ----- Config -----
  json config;
  string config_path = "./config/config.json";
  {
    // check argument
    if (argc > 1)
      config_path = argv[1];

    // check config file
    if (!is_file_exit(config_path))
      return 0;

    // read config file
    ifstream temp(config_path);
    temp >> config;

    // init
    init_config(config);
  }

  // ----- Input / Output -----
  // input and output directory
  bool isOut = config["output"];
  string input_dir = config["input_dir"];
  string output_dir = config["output_dir"];
  {
    // show directories
    cout << "[ Directories ]" << endl;

    // check input directory
    if (!is_dir_exit(input_dir))
      return 0;
    cout << "\t Input: " << input_dir << endl;

    // check output directory
    if (isOut)
    {
      if (!is_dir_exit(output_dir))
        return 0;
      cout << "\tOutput: " << output_dir << endl;
    }
  }

  // ----- Get Config -----
  // get output and predict size
  int out_w = config["output_width"], out_h = config["output_height"];
  int pred_w = config["pred_width"], pred_h = config["pred_height"];

  // get model info
  string device = config["device"];
  string pred_mode = config["pred_mode"];
  string model_path = config["model_path"];

  // get fusion parameter
  bool fusion_shadow = config["fusion_shadow"];
  int fusion_edge_border = config["fusion_edge_border"];
  int fusion_threshold_equalization = config["fusion_threshold_equalization"];
  int fusion_threshold_equalization_low = config["fusion_threshold_equalization_low"];
  int fusion_threshold_equalization_high = config["fusion_threshold_equalization_high"];
  int fusion_threshold_equalization_zero = config["fusion_threshold_equalization_zero"];

  // get perspective parameter
  bool perspective_check = config["perspective_check"];
  float perspective_distance = config["perspective_distance"];
  float perspective_accuracy = config["perspective_accuracy"];

  // get align parameter
  float align_angle_mean = config["align_angle_mean"];
  float align_angle_sort = config["align_angle_sort"];
  float align_distance_last = config["align_distance_last"];
  float align_distance_line = config["align_distance_line"];

  // show config
  {
    cout << "[ Config ]" << endl;
    cout << "\tOutput Size: " << out_w << " x " << out_h << endl;
    cout << "\tPredict Size: " << pred_w << " x " << pred_h << endl;
    cout << "\tModel Path: " << model_path << endl;
    cout << "\tDevice: " << device << endl;
    cout << "\tPred Mode: " << pred_mode << endl;
    cout << "\tFusion Shadow: " << fusion_shadow << endl;
    cout << "\tFusion Edge Border: " << fusion_edge_border << endl;
    cout << "\tFusion Threshold Equalization: " << fusion_threshold_equalization << endl;
    cout << "\tFusion Threshold Equalization Low: " << fusion_threshold_equalization_low << endl;
    cout << "\tFusion Threshold Equalization High: " << fusion_threshold_equalization_high << endl;
    cout << "\tFusion Threshold Equalization Zero: " << fusion_threshold_equalization_zero << endl;
    cout << "\tPerspective Check: " << perspective_check << endl;
    cout << "\tPerspective Distance: " << perspective_distance << endl;
    cout << "\tPerspective Accuracy: " << perspective_accuracy << endl;
    cout << "\tAlign Angle Mean: " << align_angle_mean << endl;
    cout << "\tAlign Angle Sort: " << align_angle_sort << endl;
    cout << "\tAlign Distance Last: " << align_distance_last << endl;
    cout << "\tAlign Distance Line: " << align_distance_line << endl;
  }

  // ----- Start -----
  for (const auto &file : directory_iterator(input_dir))
  {
    // Get file path and name
    string eo_path, ir_path, save_path = output_dir;
    bool isPair = get_pair(file.path().string(), eo_path, ir_path);
    if (!isPair)
      continue;
    else
    {
      // save path
      string file = eo_path.substr(eo_path.find_last_of("/\\") + 1);
      string name = file.substr(0, file.find_last_of("."));
      if (save_path.back() != '/' && save_path.back() != '\\')
        save_path += "/";
      save_path += name;
    }

    // Check file is video
    bool isVideo = is_video(eo_path);

    // Get frame size, frame rate, and create capture/writer
    int eo_w, eo_h, ir_w, ir_h, frame_rate;
    VideoCapture eo_cap, ir_cap;
    VideoWriter writer;
    if (isVideo)
    {
      eo_cap.open(eo_path);
      ir_cap.open(ir_path);
      skip_frames(eo_path, eo_cap, config);
      skip_frames(ir_path, ir_cap, config);

      eo_w = (int)eo_cap.get(3), eo_h = (int)eo_cap.get(4);
      ir_w = (int)ir_cap.get(3), ir_h = (int)ir_cap.get(4);
      frame_rate = (int)ir_cap.get(5) / (int)eo_cap.get(5);
      if (isOut)
      {
        writer.open(save_path + ".mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(out_w * 3, out_h));
      }
    }
    else
    {
      Mat eo = imread(eo_path);
      Mat ir = imread(ir_path);
      eo_w = eo.cols, eo_h = eo.rows;
      ir_w = ir.cols, ir_h = ir.rows;
    }

    // Calcualte resized size
    int eo_new_w = eo_w * ((float)out_h / eo_h);

    // Create instance
    auto image_gray = core::ImageToGray::create_instance(core::ImageToGray::Param());

    auto image_resizer = core::ImageResizer::create_instance(
        core::ImageResizer::Param()
            .set_eo(eo_new_w, out_h, out_w, out_h)
            .set_ir(out_w, out_h));

    auto image_fusion = core::ImageFusion::create_instance(
        core::ImageFusion::Param()
            .set_shadow(fusion_shadow)
            .set_edge_border(fusion_edge_border)
            .set_threshold_equalization_high(fusion_threshold_equalization_high)
            .set_threshold_equalization_low(fusion_threshold_equalization_low)
            .set_threshold_equalization_zero(fusion_threshold_equalization_zero));

    auto image_perspective = core::ImagePerspective::create_instance(
        core::ImagePerspective::Param()
            .set_check(perspective_check, perspective_accuracy, perspective_distance));

    auto image_align = core::ImageAlign::create_instance(
        core::ImageAlign::Param()
            .set_size(pred_w, pred_h, out_w, out_h)
            .set_net(device, model_path, pred_mode)
            .set_distance(align_distance_line, align_distance_last, 20)
            .set_angle(align_angle_mean, align_angle_sort)
            .set_bias(image_resizer->get_eo_clip_x(), image_resizer->get_eo_clip_y()));

    // 開始計時
    auto timer_base = core::Timer("All");
    auto timer_resize = core::Timer("Resize");
    auto timer_gray = core::Timer("Gray");
    auto timer_clip = core::Timer("Clip");
    auto timer_equalization = core::Timer("Equalization");
    auto timer_perspective = core::Timer("Perspective");
    auto timer_find_homo = core::Timer("Homo");
    auto timer_fusion = core::Timer("Fusion");
    auto timer_edge = core::Timer("Edge");
    auto timer_align = core::Timer("Align");

    // 讀取影片
    Mat eo, ir;

    while (1)
    {
      // 讀取影像
      if (isVideo)
      {
        eo_cap.read(eo);
        for (int i = 0; i < frame_rate; i++)
          ir_cap.read(ir);
      }
      else
      {
        eo = cv::imread(eo_path);
        ir = cv::imread(ir_path);
      }

      // 退出迴圈條件
      if (eo.empty() || ir.empty())
        break;

      // 幀數計數
      timer_base.start();

      Mat out;
      Mat eo_edge;
      Mat eo_resize, ir_resize, eo_clip, eo_wrap;
      Mat eo_gray, ir_gray;

      {
        timer_resize.start();
        image_resizer->resize(eo, ir, eo_resize, ir_resize);
        timer_resize.stop();
      }

      {
        timer_gray.start();
        eo_gray = image_gray->gray(eo_resize);
        ir_gray = image_gray->gray(ir_resize);
        timer_gray.stop();
      }

      {
        timer_clip.start();
        eo_clip = image_resizer->clip_eo(eo_gray);
        timer_clip.stop();
      }

      // {
      //   timer_equalization.start();
      //   eo_gray = image_fusion->equalization(eo_gray);
      //   timer_equalization.stop();
      // }

      vector<cv::Point2i> eo_pts, ir_pts;
      {
        // cv::Mat H = image_perspective->get_perspective_matrix();
        cv::Mat H;
        timer_align.start();
        image_align->align(eo_clip, ir_gray, eo_pts, ir_pts, H);
        timer_align.stop();
      }

      {
        timer_find_homo.start();
        bool sta = image_perspective->find_perspective_matrix_msac(eo_pts, ir_pts);
        timer_find_homo.stop();
      }

      {
        timer_perspective.start();
        eo_wrap = image_perspective->wrap(eo_gray, eo_new_w, out_h);
        eo_wrap = image_resizer->clip_eo(eo_wrap);
        timer_perspective.stop();
      }

      {
        timer_fusion.start();
        eo_edge = image_fusion->edge(eo_wrap);
        timer_fusion.stop();
      }

      {
        timer_edge.start();
        out = image_fusion->fusion(eo_edge, ir_resize);
        timer_edge.stop();
      }

      timer_base.stop();

      Mat catArr[] = {eo_resize, ir_resize, out};
      Mat cat;
      cv::hconcat(catArr, 3, cat);

      int bias = image_resizer->get_eo_clip_x();
      for (int i = 0; i < eo_pts.size(); i++)
      {
        cv::Point2i eo_pt(eo_pts[i].x, eo_pts[i].y);
        cv::Point2i ir_pt(ir_pts[i].x + eo_new_w - bias, ir_pts[i].y);
        cv::circle(cat, eo_pt, 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(cat, ir_pt, 2, cv::Scalar(0, 255, 0), -1);
        cv::line(cat, eo_pt, ir_pt, cv::Scalar(255, 0, 0), 1);
      }

      imshow("out", cat);

      if (isVideo)
      {
        if (isOut)
          writer.write(cat);

        int key = waitKey(1);
        if (key == 27)
          return 0;
      }
      else
      {
        if (isOut)
          imwrite(save_path + ".jpg", cat);

        int key = waitKey(0);
        if (key == 27)
          return 0;

        break;
      }
    }

    timer_resize.show();
    timer_gray.show();
    timer_clip.show();
    timer_equalization.show();
    timer_find_homo.show();
    timer_edge.show();
    timer_perspective.show();
    timer_fusion.show();
    timer_align.show();

    eo_cap.release();
    ir_cap.release();
    if (isOut)
      writer.release();

    // return 0;
  }
}