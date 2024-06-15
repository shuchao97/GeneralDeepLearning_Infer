![](https://img.shields.io/badge/-C++-00599C?&logo=c++&logoColor=FFFFFF) ![](https://img.shields.io/badge/-C-A8B9CC&logo=c&logoColor=FFFFFF) ![](https://img.shields.io/badge/-ONNX-005CED&logo=onnx&logoColor=FFFFFF) ![](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=FFFFFF) ![](https://img.shields.io/badge/CMake-5DE34F) ![](https://img.shields.io/badge/SPDLOG-FF233F) ![](https://img.shields.io/badge/ONNXRuntime-FABEF09)  
![](https://img.shields.io/badge/Charles_shu-life_is_a_very_very_long_learning_process-1ABF223)

#### GeneralDeepLearing_Infer
本仓库是一个基于C++实现的深度学习推理库，现支持上述常见大部分Onnx模型推理，后续将继续增加更多常见模型的推理过程,包括对应的TensorRT和LibTorch的实现将在下一次更新计划中

![](https://img.shields.io/badge/YOLOV8-Detection-green) ![](https://img.shields.io/badge/YOLOV8-Segmentation-red) ![](https://img.shields.io/badge/YOLOV8-POSE-005DEF) 
![](https://img.shields.io/badge/YOLOV7-Detection-FFFF) ![](https://img.shields.io/badge/YOLOV7-Segmentation-A11B23) ![](https://img.shields.io/badge/YOLOV5-Detection-FFFF) ![](https://img.shields.io/badge/YOLOV5-Segmentation-F1111F) ![](https://img.shields.io/badge/UNet-1ABF223)

🚀🚀🚀目前大部分开源库实现了对一些常见模型的推理代码，然而大部分都是仅仅实现了OnnxRuntime或者TensorRT单个推理库的推理代码且部分代码只是一个demo，没有进行适当的代码封装，不利于快速地工程开发和落地应用，这是本人建立本仓库的初衷，仓库的名称为GeneralDeepLearning_Infer，起初的想法是想构建一个对于大多数模型"General"通用的推理库，但是就现阶段的代码而言还远远没有达到通用"General"一词。计算机视觉模型的输入数据不单单是灰度图像、彩色图像、有序点云图像(3D相机)，同时还包括激光雷达无序点云数据、毫米波雷达数据等，也有可能是这些数据的多种组合。如何真正做到General还需要进一步思考。

言归正传，下面简要介绍该库文件组织结构

GeneralDeepLearing_Infer/  
│  
├── Demo/            // 测试代码文件夹  
│   ├── Header/      // 测试代码头文件文件夹  
│   └── Impl/        // 测试代码实现文件文件夹  
│  
├── Example/         // 测试模型和数据文件夹  
│   ├── Onnx/        // onnx模型文件夹  
│   ├── TensorRT/    // tensorrt模型文件夹  
│   └── TorchScript/ // TorchScript  
|  
├── Modules/                        // 各种推理库文件夹  
│   ├── GeneralLibTorch_Infer/      // LibTorch推理库 待开发  
│   ├── GeneralOnnxRuntime_Infer/   // OnnxRuntime推理库
│   └── GeneralTensorRT_Infer/      // TensorRT推理库 待开发  
|  
├── Release/                        // 编译输出文件夹  
│   ├── bin/                        // 可执行文件输出文件夹  
│   └── lib/                        // 库文件输出文件夹  
│  
└── ThirdParty/                     // 第三方库文件夹  
    ├── onnxruntime/                // onnxruntime库  
    ├── opencv/                     // opencv库  
    └── spodlog/                    // spdlog库

推荐使用windows中Cmake-Gui构建工程，使用Visual Studio 2019打开工程编译即可,详细步骤如下  

1、首先下载仓库代码
```donwnload
git clone https://github.com/CharlesShu/GeneralDeepLearning_Infer.git
```
2、解压该文件包，在文件包内新建build文件夹， 打开Cmake-Gui，选择source code 文件夹选择GeneralDeepLearning_Infer文件夹，build binary文件夹选择build文件夹，点击Configure，选择Visual Studio 2019， x64， 点击Finish，点击Generate，点击Open Project，打开Visual Studio 2019，编译即可

3、编译中点击ALL_BUILD,右键选择重新生成，编译后的dll和测试exe存放在Release/bin目录中，动态库对应的lib文件存放在Release/lib中。推荐使用ALL_BUILD编译，一次性编译所有工程，如果自己想手动编译的话，Demo工程要依赖于GeneralOnnxRuntime_Infer，手动编译先编译后者，然后编译前者

4、单击Demo工程，右键设置为启动项，打开Demo.cpp源文件，在main函数中，修改如下：

```
//修改图片路径，注意R"()"中直接加原始路径就可以，不用再加引号和双斜杠，字符不会被转义
cv::Mat curr_image = cv::imread(R"(F:\CV_Treasure\GeneralOnnxRuntime_Infer\data\car.jpg)");
cv::Mat next_image = cv::imread(R"(F:\YOLO\infer-main\workspace\inference\gril.jpg)");
```
```
//修改模型路径 Demo.cpp 19 lines
modelParam.OnnxModelPath =
R"(F:\CV_Treasure\GeneralOnnxRuntime_Infer\data\yolov8n-pose.onnx)";
//修改模型任务类型和模型类型 Demo.cpp 26 27 lines
modelTypeParams.TaskType = TASK_POSE_ESTIMATION;   // 参考StructNameDefines.h中定义
modelTypeParams.ModelType = YOLO_V8;               // 参考StructNameDefines.h中定义
```
```
// 修改日志保存路径 检测结果也将存放于该文件夹下  Demo.cpp 32 33 lines
logParams.LogPath = "F:\\CV_Treasure\\GeneralOnnxRuntime_Infer\\log";
```
```
// 修改基类指针指向的派生类对象, 根据网络选择合适的派生类对象 Demo.cpp 61 lines
 std::unique_ptr<GeneralOnnxBase> netobj = std::make_unique<GeneralOnnxYOLO>(); //YOLO
 or netobj = std::make_unique<GeneralOnnxUNet>() //UNet
```
5、完成上述修改后，点击运行即可。

🔥🔥🔥注意当前版本代码仅支持windows使用，支持多batch推理，不支持动态高宽(个人觉得没啥用)，暂时不支持异步多线程调用

🚨🚨🚨 本仓库中包含的onnxruntime库版本为1.17，请检查电脑的cuda版本是否支持该版本onnxruntime库，如果cuda版本较低不支持该onnxruntime库，需要自行下载适配的onnxruntime版本https://github.com/microsoft/onnxruntime/releases, 按照原文件夹名字和文件组织结构替换即可，同时需要将onnxtuntime.dll复制到lib文件夹中的四个.dll复制到Relese\bin目录下，opencv和spdlog不会存在兼容性问题

🙈🙈🙈 最后希望各位大佬点个赞
#### 引用
[1] https://github.com/Shiftqueue/onnxruntime_use_cpp  
[2] https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp  
[3] https://github.com/ultralytics/ultralytics  
[4] https://github.com/microsoft/onnxruntime
