cmake_minimum_required(VERSION 3.5)
project(GeneralOnnxRuntime_Infer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CONFIGURATION_TYPES "Release")

set(Release_Dir ${CMAKE_SOURCE_DIR}/Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${Release_Dir}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${Release_Dir}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${Release_Dir}/bin)

if(WIN32)
    message(STATUS "Configure for Windows")
    if(MSVC)
        set(AUTHOR_NAME "charles chao shu")
        set(PRODUCT_NAME "GeneralOnnxRuntime_Infer")
        set(LegalCopyright "Copyright (C) 2024 Charles Chao shu, All rights reserved.")
        set(VERSION_MAJOR 1)
        set(VERSION_MINOR 0)
        set(VERSION_PATCH 0)
        set(VERSION_REVISION 1)

        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/OnnxInferVersionInfo.rc.in
            ${CMAKE_CURRENT_SOURCE_DIR}/OnnxInferVersionInfo.rc
            @ONLY)
        set(ONNX_VERSION ${CMAKE_CURRENT_SOURCE_DIR}/OnnxInferVersionInfo.rc)

    endif(MSVC)
endif(WIN32)

file(GLOB SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Header/*.h")
file(GLOB SRC_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/Impl/*.cpp")

add_library(GeneralOnnxRuntime_Infer SHARED ${SRC_DIR} ${SRC_INCLUDE} ${ONNX_VERSION})
target_compile_definitions(GeneralOnnxRuntime_Infer PRIVATE GENERAL_ONNXRUNTIME_INFER_EXPORTS)

target_include_directories(GeneralOnnxRuntime_Infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Header)

# 配置opencv
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/opencv)
target_include_directories(GeneralOnnxRuntime_Infer PRIVATE ${OPENCV_DIR}/include)
target_link_directories(GeneralOnnxRuntime_Infer PRIVATE ${OPENCV_DIR}/lib)
target_link_libraries(GeneralOnnxRuntime_Infer PRIVATE opencv_world460.lib)

# 配置onnxruntime
set(ORT_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/onnxruntime)
target_include_directories(GeneralOnnxRuntime_Infer PRIVATE ${ORT_DIR}/include)
target_link_directories(GeneralOnnxRuntime_Infer PRIVATE ${ORT_DIR}/lib)
target_link_libraries(GeneralOnnxRuntime_Infer PRIVATE onnxruntime.lib
                      onnxruntime_providers_cuda.lib onnxruntime_providers_shared.lib
                      onnxruntime_providers_tensorrt.lib)

# 配置spdlog
set(Log_Dir ${CMAKE_SOURCE_DIR}/ThirdParty/spdlog)
target_include_directories(GeneralOnnxRuntime_Infer PRIVATE ${Log_Dir}/include)


