cmake_minimum_required(VERSION 3.5)
project(GeneralDeepLearning_Infer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

# 设置编译模式
set(CMAKE_CONFIGURATION_TYPES "Release")

# 设置编译输出目录
set(RELEASE_DIR ${CMAKE_SOURCE_DIR}/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${RELEASE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${RELEASE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${RELEASE_DIR}/bin)

if(WIN32)
    message(STATUS "Configure for Windows")
    if(MSVC)
        set(AUTHOR_NAME "charles chao shu")
        set(PRODUCT_NAME "this is a test demo")
        set(LegalCopyright "Copyright (C) 2024 Charles Chao shu, All rights reserved.")
        set(VERSION_MAJOR 1)
        set(VERSION_MINOR 0)
        set(VERSION_PATCH 0)
        set(VERSION_REVISION 1)

        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/DemoVersionInfo.rc.in 
                       ${CMAKE_CURRENT_SOURCE_DIR}/DemoVersionInfo.rc
                       @ONLY)
        set(DEMO_VERSION  ${CMAKE_CURRENT_SOURCE_DIR}/DemoVersionInfo.rc)

    endif(MSVC)
endif(WIN32)

add_subdirectory("./Modules/GeneralOnnxRuntime_Infer")

file(GLOB DEMO_INCLUDE_DIR  "${CMAKE_SOURCE_DIR}/Demo/Header/*.h")
file(GLOB DEMO_SRC_DIR "${CMAKE_SOURCE_DIR}/Demo/Impl/*.cpp")
add_executable(Demo ${DEMO_INCLUDE_DIR} ${DEMO_SRC_DIR}
                    ${DEMO_VERSION})
add_dependencies(Demo GeneralOnnxRuntime_Infer)

# 配置GeneralOnnxRuntime_Infer的头文件和lib库
target_include_directories(Demo PRIVATE ${CMAKE_SOURCE_DIR}/Modules/GeneralOnnxRuntime_Infer/Header)
target_link_directories(Demo PRIVATE ${RELEASE_DIR}/lib)
target_link_libraries(Demo PRIVATE GeneralOnnxRuntime_Infer.lib)

# 配置opencv
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/opencv)
target_include_directories(Demo PRIVATE ${OPENCV_DIR}/include)
target_link_directories(Demo PRIVATE ${OPENCV_DIR}/lib)
target_link_libraries(Demo PRIVATE opencv_world460.lib)

# 配置onnxruntime
set(ORT_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/onnxruntime)
target_include_directories(Demo PRIVATE ${ORT_DIR}/include)
target_link_directories(Demo PRIVATE ${ORT_DIR}/lib)
target_link_libraries(Demo PRIVATE onnxruntime.lib
                      onnxruntime_providers_cuda.lib
                      onnxruntime_providers_shared.lib
                      onnxruntime_providers_tensorrt.lib
                      )

# 配置spdlog
set(LOG_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/spdlog)
target_include_directories(Demo PRIVATE ${LOG_DIR}/include)


