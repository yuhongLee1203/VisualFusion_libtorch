cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(out)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(OpenCV_DIR "/usr/include/opencv4")
# set(CMAKE_PREFIX_PATH "/home/hkai/envs/libtorch_2.3.0_cu121")
set(CMAKE_PREFIX_PATH "/home/hkai/envs/libtorch_1.10.0_cu113")

# Avoid error when libtorch compiler with opencv4.x
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui calib3d videoio)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/include)

link_directories(${OpenCV_LIBS} /usr/local/lib)
link_directories(${PROJECT_SOURCE_DIR}/utils/src)
link_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/src)

add_executable(out main.cpp)
target_link_libraries(out ${OpenCV_LIBS} ${TORCH_LIBRARIES} -lstdc++fs)


set_property(TARGET out PROPERTY CXX_STANDARD 17)