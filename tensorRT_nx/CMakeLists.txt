cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(out)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# ============================================================
# 自動尋找依賴套件 (使用 find_package)
# ============================================================

# 1. OpenCV - 自動尋找
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui calib3d videoio dnn)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 2. CUDA - 自動尋找
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
else()
    message(FATAL_ERROR "CUDA not found!")
endif()

# 3. TensorRT - 使用 find_library 自動尋找
# 嘗試在標準路徑中尋找 TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
        /usr/include
        /usr/local/include
        $ENV{TENSORRT_ROOT}/include
    DOC "TensorRT include directory"
)

find_library(TENSORRT_NVINFER_LIB nvinfer
    PATHS
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        $ENV{TENSORRT_ROOT}/lib
    DOC "TensorRT nvinfer library"
)

find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser
    PATHS
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
        $ENV{TENSORRT_ROOT}/lib
    DOC "TensorRT nvonnxparser library"
)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB)
    message(STATUS "Found TensorRT:")
    message(STATUS "  Include dir: ${TENSORRT_INCLUDE_DIR}")
    message(STATUS "  nvinfer lib: ${TENSORRT_NVINFER_LIB}")
    message(STATUS "  nvonnxparser lib: ${TENSORRT_NVONNXPARSER_LIB}")
else()
    message(FATAL_ERROR "TensorRT not found! Please install TensorRT or set TENSORRT_ROOT environment variable.")
endif()

# 4. CUDA Runtime - 自動尋找
find_library(CUDART_LIB cudart
    PATHS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
    DOC "CUDA runtime library"
)

find_library(CUBLAS_LIB cublas
    PATHS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
        /usr/local/cuda/lib64
        /usr/local/cuda/lib
    DOC "CUDA BLAS library"
)

if(CUDART_LIB AND CUBLAS_LIB)
    message(STATUS "Found CUDA libraries:")
    message(STATUS "  cudart: ${CUDART_LIB}")
    message(STATUS "  cublas: ${CUBLAS_LIB}")
else()
    message(FATAL_ERROR "CUDA runtime libraries not found!")
endif()

# ============================================================
# Include 目錄設定
# ============================================================
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/nlohmann
    ${PROJECT_SOURCE_DIR}/utils/include
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/include
)

# ============================================================
# 編譯目標 (已移除未使用的 to_gray, resizer, perspective)
# ============================================================
add_executable(out main.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_align_tensorrt.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_fusion.cpp
    ${PROJECT_SOURCE_DIR}/utils/src/util_timer.cpp
)

# ============================================================
# 連結函式庫
# ============================================================
target_link_libraries(out
    ${OpenCV_LIBS}
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    ${CUDART_LIB}
    ${CUBLAS_LIB}
    ${CUDA_LIBRARIES}
    stdc++fs
)

# ============================================================
# 安裝設定 (可選)
# ============================================================
install(TARGETS out DESTINATION bin)

