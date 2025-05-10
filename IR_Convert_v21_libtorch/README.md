## Basic
* 資料夾結構
    - IR_Convert_v17_libtorch/
        - .vscode/
            - (Visual Studio Code 設定檔)
        - build/ 
            - (編譯檔)
        - config/
            - (設定檔)
            - config.json
        - input/
            - (預設輸入影像/影片)
        - lib_image_fusion/
            - include/
                - core_image_align_libtorch.h
                - core_image_fusion.h
                - core_image_perspective.h
                - core_image_resizer.h
                - core_image_to_gray.h
            - src/
                - core_image_align_libtorch.cpp
                - core_image_fusion.cpp
                - core_image_perspective.cpp
                - core_image_resizer.cpp
                - core_image_to_gray.cpp
        - model/
            - SemLA_jit_cpu.zip
            - SemLA_jit_cuda.zip
        - nlohman/
            - json_fwd.hpp
            - json.hpp
        - output/
            - (預設輸出影像/影片)
        - utils/
            - include/
                - util_timer.h
            - src/
                - util_timer.cpp
        - CMakeLists.txt
        - main.cpp
        - README.md
* 執行
    `./build/out <optional_config_file>`

## Config

* `input_dir`
    * 輸入資料夾
    * 型別: string
    * 預設: "./input/"
* `output_dir`
    * 輸出資料夾
    * 型別: string
    * 預設: "./output/"
* `output`
    * 是否輸出影像
    * 型別: bool
    * 預設: false
* `device`
    * 裝置
    * 型別: string
    * 預設: "cpu"
    * 選項: ["cpu", "cuda"]
* `pred_mode`
    * 模型模式
    * 型別: string
    * 預設: "fp32"
    * 選項: ["fp32", "fp16"]
    * 備註:
      *  只有在 device 為 cuda 時有效
      *  模型需要使用 fp16 版本
* `model_path`
    * 模型路徑
    * 型別: string
    * 預設: "./model/SemLA_jit_cuda.zip"
* `output_width`
    * 輸出寬度
    * 型別: int
    * 預設: 640
* `output_height`
    * 輸出高度
    * 型別: int
    * 預設: 512
* `pred_width`
    * 預測寬度
    * 型別: int
    * 預設: 320
* `pred_height`
    * 預測高度
    * 型別: int
    * 預設: 256
* `fusion_shadow`
    * [融合]是否啟用陰影
    * 型別: bool
    * 預設: false
* `fusion_edge_border`
    * [融合]邊緣粗細
    * 型別: int
    * 預設: 1
* `fusion_threshold_equalization`
    * [融合]均衡化的閾值
    * 型別: int
    * 預設: 128
* `fusion_threshold_equalization_low`
    * [融合]均衡化的低閾值
    * 型別: int
    * 預設: 72
* `fusion_threshold_equalization_high`
    * [融合]均衡化的高閾值
    * 型別: int
    * 預設: 192
* `fusion_threshold_equalization_zero`
    * [融合]均衡化的零閾值
    * 型別: int
    * 預設: 64
* `perspective_check`
    * [透視]是否檢查轉換矩陣的分數
    * 型別: bool
    * 預設: true
* `perspective_distance`
    * [透視]檢查轉換矩陣的距離
    * 型別: float
    * 預設: 10
* `perspective_accuracy`
    * [透視]檢查轉換矩陣的分數閾值
    * 型別: float
    * 預設: 0.85
* `align_angle_sort`
    * [對齊]與平均角度差排序前幾％的角度
    * 型別: float
    * 預設: 0.6
    * 備註: 0.6 代表選擇排序前 60% 的角度
* `align_angle_mean`
    * [對齊]與平均角度差的閾值
    * 型別: float
    * 預設: 10.0
    * 單位: 度
* `align_distance_line`
    * [對齊]水平/垂直篩選時的距離條件
    * 型別: float
    * 預設: 10.0
* `align_distance_last`
    * [對齊]使用上一次變化矩陣篩選時的距離條件
    * 型別: float
    * 預設: 15.0
* `skip_frames`
    * 跳過幀數
    * 型別: object
    * 預設: {}
    * 範例:
        ```json
        "skip_frames":{
            "video_file_name": <frame_number>,
        }
        ```
    * 備註: frame_number 可以是整數或浮點數

## ImageToGray
* `ptr core::ImageToGray::create_instance(core::ImageToGray::Param param)`
    * 建立 ImageToGray 實體
    * 參數: 
        * ImageToGray 參數
    * ImageToGray 實體

* `cv::Mat core::ImageToGray::resize(cv::Mat &in)`
    * 影像轉灰階
    * 參數: 
        * in: 影像輸入
    * 回傳: 
        * 灰階影像

## ImageResizer

* struct core::ImageResizer::Param
    * `set_eo(int w, int h)`
        * 設定 eo 輸出尺寸
        * 參數: 
            * w: 輸出寬度
            * h: 輸出高度
        * 回傳: 無
    * `set_eo(int w, int h, int clip_w, int clip_h)`
        * 設定 eo 輸出尺寸及裁切範圍
        * 參數: 
            * w: 輸出寬度
            * h: 輸出高度
            * clip_w: 裁切後寬度
            * clip_h: 裁切後高度
        * 回傳: 無
    * `set_ir(int w, int h)`
        * 設定 ir 輸出尺寸
        * 參數: 
            * w: 輸出寬度
            * h: 輸出高度
        * 回傳: 無
    * `set_ir(int w, int h, int clip_w, int clip_h)`
        * 設定 ir 輸出尺寸及裁切範圍
        * 參數: 
            * w: 輸出寬度
            * h: 輸出高度
            * clip_w: 裁切後寬度
            * clip_h: 裁切後高度
        * 回傳: 無

* `ptr core::ImageResizer::create_instance(core::ImageResizer::Param param)`
    * 建立 ImageResizer 實體
    * 參數: 
        * param: ImageResizer 參數
    * 回傳: ImageResizer 實體

* `void core::ImageResizer::resize(cv::Mat &eo_in, cv::Mat &ir_in, cv::Mat &eo_out, cv::Mat &ir_out)`
    * 影像縮放
    * 參數: 
        * eo_in: eo 影像輸入
        * ir_in: ir 影像輸入
        * eo_out: eo 影像輸出
        * ir_out: ir 影像輸出
    * 回傳: 無

* `cv::Mat core::ImageResizer::clip_eo(cv::Mat &in)`
    * 裁切 eo 影像
    * 參數: 
        * in: 影像輸入
    * 回傳: 裁剪後影像

* `cv::Mat core::ImageResizer::clip_ir(cv::Mat &in)`
    * 裁切 ir 影像
    * 參數: 
        * in: 影像輸入
    * 回傳: 裁剪後影像

* `void core::ImageResizer::get_eo_clip_x()`
    * 取得 eo 裁切 x 座標
    * 回傳: eo 裁切 x 座標

* `void core::ImageResizer::get_eo_clip_y()`
    * 取得 eo 裁切 y 座標
    * 回傳: eo 裁切 y 座標

* `void core::ImageResizer::get_ir_clip_x()`
    * 取得 ir 裁切 x 座標
    * 回傳: ir 裁切 x 座標
  
* `void core::ImageResizer::get_ir_clip_y()`
    * 取得 ir 裁切 y 座標
    * 回傳: ir 裁切 y 座標

## ImageAlign

* struct core::ImageAlign::Param
    * `set_size(int pw, int ph, int ow, int oh)`
        * 設定預測尺寸與輸出尺寸
        * 參數: 
            * pw: 預測寬度
            * ph: 預測高度
            * ow: 輸出寬度
            * oh: 輸出高度
        * 回傳: 無
    * `set_net(std::string device, std::string model_path, std::string mode = "fp32")`
        * 設定模型
        * 參數: 
            * device: 裝置
            * model_path: 模型路徑
            * mode: 模型模式
        * 回傳: 無
    * `set_distance(float line, float last)`
        * 設定距離參數
        * 參數: 
            * line: 水平/垂直篩選時的距離條件
            * last: 使用上一次變化矩陣篩選時的距離條件
        * 回傳: 無
    * `set_angle(float mean, float sort)`
        * 設定角度參數
        * 參數: 
            * mean: 使用平均角度篩選時的角度條件
            * sort: 使用排序角度篩選時的角度條件
        * 回傳: 無

* `ptr core::ImageAlign::create_instance(core::ImageAlign::Param param)`
    * 建立 ImageAlign 實體
    * 參數: 
        * ImageAlign 參數
    * 回傳: ImageAlign 實體

* `void core::ImageAlign::align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H)`
    * 模型預測、篩選
    * 參數: 
        * eo: eo 影像輸入
        * ir: ir 影像輸入
        * eo_pts: eo 特徵點輸出
        * ir_pts: ir 特徵點輸出
        * H: 篩選用變換矩陣，通常給空白 cv::Mat 或是前一幀的變換矩陣
    * 回傳: 無

* `void core::ImageAlign::warm_up()`
    * 模型預熱
    * 參數: 無
    * 回傳: 無

* `void core::ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts)`
    * 模型預測
    * 參數: 
        * eo: eo 影像輸入
        * ir: ir 影像輸入
        * eo_pts: eo 特徵點輸出
        * ir_pts: ir 特徵點輸出
    * 回傳: 無

* `std::vector<float> core::ImageAlign::line_equation(cv::Point2f &pt1, cv::Point2f &pt2)`
    * 計算直線方程式
    * 參數: 
        * pt1: 點 1
        * pt2: 點 2
    * 回傳: 
        * {斜率, 截距}

* `int core::ImageAlign::judge_h_line(std::vector<float> &line, cv::Point2i &pt)`
    * 判斷點與水平線的關係
    * 參數: 
        * line: 水平線方程式
        * pt: 點
    * 回傳: 
        * 1: 點在水平線下方
        * 0: 點在水平線上
        * -1: 點在水平線上方

* `int core::ImageAlign::judge_v_line(std::vector<float> &line, cv::Point2i &pt)`
    * 判斷點與垂直線的關係
    * 參數: 
        * line: 垂直線方程式
        * pt: 點
    * 回傳: 
        * 1: 點在垂直線右方
        * 0: 點在垂直線上
        * -1: 點在垂直線左方

* `int core::ImageAlign::judge_quadrant(std::vector<float> &line_v, std::vector<float> &line_h, cv::Point2i &pt)`
    * 判斷點與象限的關係
    * 參數: 
        * line_v: 垂直線方程式
        * line_h: 水平線方程式
        * pt: 點
      * 回傳: 
        * 0: 左上
        * 1: 右上
        * 2: 右下
        * 3: 左下
        * 4: 左線上
        * 5: 上線上
        * 6: 右線上
        * 7: 下線上
        * -1: 兩線交點

* `void show(std::vector<std::vector<int>> q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<float> v_line, std::vector<float> h_line)`
    * 顯示箭頭
    * 參數: 
        * q_idx: 象限索引
        * eo_mkpts: eo 特徵點
        * ir_mkpts: ir 特徵點
        * v_line: 垂直線方程式
        * h_line: 水平線方程式
    * 回傳: 無

* `void class_quadrant(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)`
    * 分類象限
        * 在索引的每一層會紀錄原先的 eo_mkpts 的索引
        * 象限索引會有9層
          * 0: 左上
          * 1: 右上
          * 2: 右下
          * 3: 左下
          * 4: 左
          * 5: 上
          * 6: 右
          * 7: 下
          * 8: 點
    * 參數: 
        * eo_mkpts: eo 特徵點
        * ir_mkpts: ir 特徵點
        * q_idx: 輸出象限索引
        * q_diff: 輸出 eo 和 ir 座標差異
    * 回傳: 無

* `void find_line(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<float> &v_line, std::vector<float> &h_line)`
  * 尋找切割線
  * 參數: 
      * q_idx: 象限索引
      * eo_mkpts: eo 特徵點
      * v_line: 輸出垂直線方程式
      * h_line: 輸出水平線方程式

* `void apply_keypoints(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<int> &filter_idx)`
    * 利用索引清單重新整理特徵點
    * 參數: 
        * eo_mkpts: 輸入輸出 eo 特徵點
        * ir_mkpts: 輸入輸出 ir 特徵點
        * filter_idx: 篩選索引
    * 回傳: 無

* `void apply_quadrants(std::vector<std::vector<int>> &q_idx, std::vector<int> &filter_idx)`
    * 利用象限索引清單重新整理特徵點
    * 參數: 
        * q_idx: 輸入輸出象限索引
        * filter_idx: 篩選索引
    * 回傳: 無

* `float distance_line(std::vector<float> &line, cv::Point2i &pt)`
    * 計算點到直線的距離
    * 參數: 
        * line: 直線方程式
        * pt: 點
    * 回傳: 距離

* `std::vector<int> filter_diagonal(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)`
    * 篩除在對角象限的點
    * 參數: 
        * v_line: 垂直線方程式
        * h_line: 水平線方程式
        * q_idx: 象限索引
        * eo_mkpts: eo 特徵點
    * 回傳: 篩選索引

* `std::vector<int> filter_same(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)`
    * 篩除不在同一象限的點
    * 參數: 
        * v_line: 垂直線方程式
        * h_line: 水平線方程式
        * q_idx: 象限索引
        * eo_mkpts: eo 特徵點
    * 回傳: 篩選索引

* `std::vector<int> filter_distance(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts)`
    * 篩除距離水平與垂直線過遠的點
    * 參數: 
        * v_line: 垂直線方程式
        * h_line: 水平線方程式
        * q_idx: 象限索引
        * eo_mkpts: eo 特徵點
    * 回傳: 篩選索引

* `std::vector<int> filter_mean_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)`
    * 篩除與平均角度差異過大的點
    * 參數: 
        * q_idx: 象限索引
        * q_diff: eo 和 ir 座標差異
    * 回傳: 篩選索引

* `std::vector<int> filter_sort_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff)`
    * 篩除與平均角度差異排序後過大的點
    * 參數: 
        * q_idx: 象限索引
        * q_diff: eo 和 ir 座標差異
    * 回傳: 篩選索引

* `std::vector<int> filter_last_H(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, cv::Mat &H)`
    * 篩除與上一次變換矩陣差異過大的點
    * 參數: 
        * q_idx: 象限索引
        * eo_mkpts: eo 特徵點
        * ir_mkpts: ir 特徵點
        * H: 變換矩陣
    * 回傳: 篩選索引

## ImageFusion

* struct core::ImageFusion::Param
    * `set_shadow(bool v)`
        * 是否啟用陰影
        * 參數: 
            * v: 是否啟用陰影
        * 回傳: 無
    * `set_edge_border(int v)`
        * 設定邊緣粗細
        * 參數: 
            * v: 邊緣粗細
        * 回傳: 無
    * `set_threshold_equalization(int v)`
        * 設定均衡化的閾值
        * 參數: 
            * v: 均衡化的閾值
        * 回傳: 無
    * `set_threshold_equalization_low(int v)`
        * 設定均衡化的低閾值
        * 參數: 
            * v: 均衡化的低閾值
        * 回傳: 無
    * `set_threshold_equalization_hign(int v)`
        * 設定均衡化的高閾值
        * 參數: 
            * v: 均衡化的高閾值
        * 回傳: 無
    * `set_threshold_equalization_zero(int v)`
        * 設定均衡化的零閾值
        * 參數: 
            * v: 均衡化的零閾值
        * 回傳: 無

* `ptr core::ImageFusion::create_instance(core::ImageFusion::Param param)`
    * 建立 ImageFusion 實體
    * 參數: 
        * param: ImageFusion 參數
    * 回傳: ImageFusion 實體

* `cv::Mat core::ImageFusion::edge(cv::Mat &in)`
    * 繪製邊緣
    * 參數: 
        * in: 影像輸入
    * 回傳: 影像輸出

* `cv::Mat core::ImageFusion::equalization(cv::Mat &in)`
    * 均衡化
    * 參數: 
        * in: 影像輸入
    * 回傳: 影像輸出

* `cv::Mat core::ImageFusion::fusion(cv::Mat &eo, cv::Mat &ir)`
    * 影像融合
    * 參數: 
        * eo: eo 影像輸入
        * ir: ir 影像輸入
    * 回傳: 融合影像

## ImagePerspective

* struct core::ImagePerspective::Param
    * `set_check(bool chk, float acc, float dis)`
        * 檢查分數、距離
        * 參數: 
            * chk: 是否檢查
            * acc: 分數閾值
            * dis: 距離閾值
        * 回傳: 無
    * `set_bias(int x, int y)`
        * 設定座標偏移
        * 參數: 
            * x: x 軸偏移
            * y: y 軸偏移
        * 回傳: 無

* `ptr core::ImagePerspective::create_instance(core::ImagePerspective::Param param)`
    * 建立 ImagePerspective 實體
    * 參數: 
        * param: ImagePerspective 參數
    * 回傳: ImagePerspective 實體

* `cv::Mat core::ImagePerspective::wrap(cv::Mat &in, int width, int height)`
    * 套用變換矩陣
    * 參數: 
        * in: 影像輸入
        * width: 輸出寬度
        * height: 輸出高度
    * 回傳: 影像輸出

* `bool core::ImagePerspective::find_perspective_matrix(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst)`
    * 尋找變換矩陣
    * 參數: 
        * src: 原始座標
        * dst: 目標座標
    * 回傳: 是否成功

* `int core::ImagePerspective::count_allow(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H)`
    * 使用指定變換矩陣計算符合條件的點數
    * 參數: 
        * src: 原始座標
        * dst: 目標座標
        * H: 變換矩陣
    * 回傳: 符合條件的點數

* `float core::ImagePerspective::calculate_mse(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H)`
    * 使用指定變換矩陣計算均方誤差
    * 參數: 
        * src: 原始座標
        * dst: 目標座標
        * H: 變換矩陣
    * 回傳: 均方誤差

* `cv::Mat core::ImagePerspective::get_perspective_matrix()`
    * 取得變換矩陣
    * 參數: 無
    * 回傳: 變換矩陣