# Cpp Style

## 說明:

此文件參考以下的文件: [google cpp style][], 再修改一些地方而制定.   
此文件只說明大綱, 細節請參考以上link.  
[google cpp style]: https://google.github.io/styleguide/cppguide.html  


## 編輯:
  
+   禁止使用tab, 使用space, 縮排使用2個space.

+   撰寫格式如下:

    ```cpp
    /*
     * Indentation
     */
    #include <math.h>
    
    class Point {
    public:
      Point(double x, double y) :
        x_(x), y_(y)
      {
      }
    
      double compute_distance(const Point& other) const;
      int compare_x(const Point& other) const;
      void add_positive_n(int n);
    
    private:
      double x_;
      double y_;
    };
    
    double Point::compute_distance(const Point& other) const
    {
      double dx = x_ - other.x;
      double dy = y_ - other.y;
    
      return sqrt(dx * dx + dy * dy);  // this is comment
    }
    
    int Point::compare_x(const Point& other) const
    {
      if (x_ < other.x) {
        return -1;
      } 
      else if (x_ > other.x) {
        return 1;
      } 
      else {
        return 0;
      }
    }
    
    void Point::add_positive_n(int n)
    {
      for (int i = 0; i < n; ++i) {
        x_ += 1;
        y_ += 1;
      }
    }
    
    struct Image {
      int width;
      int height;
      uint8_t *ptr;
    };
    
    namespace my_nm {
    int foo(int bar)
    {
      switch (bar) {
        case 0:
          ++bar;
          break;
        case 1:
          --bar;
        default: {
          bar += bar;
          break;
        }
      }
    }
    }  // namespace my_nm
    ```


## 一般性建議:

+   使用c++14的功能, 在ubuntu 20及以上的平台, 建議使用c++17的功能.

+   每一個project必須定義其name space, 所有與project相關的class, function必須在此name space之下. 
    如class具一般性, 可自行定義其name space, 不在project的name space之下, 以增加共用性.
    + 如
    
    ```cpp
    namespace agv {
      std::string to_string(const std::vector<uint8_t> &values);
    }
    ```
            
+   善用static_assert.
    + Static_assert()在編譯時檢查. 如果不通過, 則編譯錯誤. 如
    
    ```cpp
      static_assert(DEBUG == 0, "DEBUG should be 0");
    ```
    
+   善用c++的auto, 以避免冗長的type名稱.
    + 如
    
    ```cpp
    const auto current_time = std::chrono::system_clock::now();
    auto等於std::chrono::time_point<std::chrono::system_clock>.
    ```

+   善用lambda來取代短小的函數.
    + 如
    
    ```cpp
    auto print = [](const int& n) { std::cout << " " << n; };
    std::for_each(nums.begin(), nums.end(), print);
    std::for_each(nums.begin(), nums.end(), [](int &n){ n++; });
    ```

+   善用c++的STL (standard template library), 禁止自行撰寫類似的功能.

+   善用&&(rvalue reference)或const &(const reference)以減少copy.

+   local變數在使用時才宣告, 並同時初始化. 其存活時間越短越好.

+   class內的變數盡量放在private下.

+   struct的目的是儲存資料, class的目的是執行功能.

+   盡量使用c++的cast, 如 `static_cast<float>(double_value)`.

+   對iterator, 使用++it, 而非it++.
    + It++是先取出值再將原來的值加1, 所以需要暫存位置. ++it先加1再取值.
    
+   宣告變數時就加上const, 除非此變數之後需要變更.

+   不要假設int的大小. 如需固定大小的整數, 使用<cstdint>.

+   盡量不要使用preprocessor的功能(如#if, #ifdef等等).
    + preprocessor的行為在編譯時決定, runtime無法變更. 有些preprocessor的選項其實是相關連的.  
      有時更改了一個, 忘了更改另外一個. 另外, preprocessor無法檢查語法.  
      很多preprocessor的行為可以使用繼承與template來取代, 讓程式可以更強健.
    
    ```cpp
    //using value_type = float;
    using value_type = double;
  
    template <class T> struct opencv_depth_value;
    template <> struct opencv_depth_value<float>
    { const static int value = CV_32F; };
    template <> struct opencv_depth_value<double>
    { const static int value = CV_64F; };
   
    template <int depth_value> struct opencv_depth_type;
    template <> struct opencv_depth_type<CV_32F>
    { using type = float; };
    template <> struct opencv_depth_type<CV_64F>
    { using type = double; };
    template <> struct opencv_depth_type<CV_8U>
    { using type = uint8_t; };
    ```
    
+   使用繼承, 可以在runtime時指定程式行為. 例如, 可以使用command line option指定camera為真的camera或模擬的camera.
    
+   盡量不要使用typedef, 使用using.


## 一般性禁止: (除非有特殊理由, 否則不可使用)

+   禁止使用global scope的變數.

+   禁止使用非const, 或constexpr的file scope變數.

+   禁止使用macro來取代常數, 使用const變數, constexpr變數, 或enum class,

+   禁止使用macro來取代函數, 使用inline函數.

+   禁止使用new來allocate變數, 使用std::make_shared或srd::make_unique.
    + std::shared_ptr或std::unique_ptr可以自動釋放記憶體. 除此之外, `shared_ptr`及`unique_ptr`可以用來告知使用者所有權的問題.
    
+   禁止使用c++的native array, 使用`std::array`或`std::vector`.

+   禁止使用`NULL`或0來代表null pointer, 使用`nullptr`.

+   禁止使用enum, 使用enum class.

+   禁止使用c的header file(如stdlib.h), 使用c++的形式(cstdlib)

+   禁止使用非標準的c++功能.

+   禁止自行宣告external函數, 變數, class等等, 使用對應的header file.


## Function:

+   盡量使用return來回傳結果, 必要時可以使用output參數來回傳結果. 此參數需為指標.

+   基本型態的輸入參數可以使用pass by value. 其他型態的輸入參數則盡量使用pass by const reference.
    + 如
    
    ```cpp
    Func(int v1, double v2);
    Func(const std::vector<int> &values);
    ```

+   盡量不要使用同時具備輸入及輸出功能的參數.

+   輸入參數在輸出參數之前.

+   盡可能減少參數的數目.

+   函數內容盡可能簡短.

+   函數命名須要明確, 需說明其目的而非方法.

+   如函數功能簡單且短小, 可以考慮使用`inline`.
    + 如
    
    ```cpp
    inline bool is_connected() const { return socket_ >= 0; }
    ```


## Header file:

+   header file必須include所有其需要的檔案.

+   header file必須使用#define guard.
    + project_root/src/include/my_project/header.h -> PROJECT_ROOT_SRC_INCLUDE_MY_PROJECT_HEADER_H_ or INCLUDE_MY_PROJECT_HEADER_H_

+   Header file的宣告順序:
    + 在 dir/foo.cpp, header file的順序如下(同類別的檔案, 依字母排序):
    
    ```
    <dir/foo.h>

    <C system files>
    
    <C++ system files>
    
    <其他library的 .h files>

    <計畫的.h files>
    ```


## 命名:

+   盡量不要使用縮寫, 但一些習知的縮寫例外. 命名的目的是要讓閱讀者盡快了解其意義, 而非節省打字時間.

+   不要將自己定義的任何東西放在 `std` (或其他習知的name space)之下.

+   如欲與一些習知的命名方式相容, 可以不使用以下的命名規則. 如, STL中常用的 `value_type`.

+   檔案: `under_line.cpp`, `under_line.h`.

+   Name space: `my_space`.

+   函數: `this_is_a_function()`.

+   變數: `this_is_a_variable`.

+   常數: `this_is_constant_variable`.

+   Enum class member: `ENUM_MEMBER` or `enum_member`.

+   Macro: `MY_MACRO`.

+   Type(class, struct, enum class等等): `ThisIsType`.

+   Class member函數: `class_member_function()`.

+   Class member變數: `class_member_variable_`. (名稱結尾帶有under line)
    + 如
    
    ```cpp
    bool is_child_error() { return is_child_error_; }
    void set_event_processor(EventProcessor *event_processor)
    {
      event_processor_ = event_processor;
    }
    ```

+   Struct member變數: `struct_member_variable`. (名稱結尾沒有under line)

+   Ros:
    + Package: `under_scored`.
    + Topic/service: `under_scored`.
    + File: `under_scored.cpp, under_scored.h`.
    + Library: `libmy_great_thing`.
    + Msg/srv/action: `AddTwoInts.srv`

