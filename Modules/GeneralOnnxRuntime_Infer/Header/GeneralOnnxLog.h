#pragma once
#include <spdlog/spdlog.h>

#include "StructNameDefines.h"

class GeneralOnnxLog {
private:

  _LogInitParams _LogConfig;

  std::shared_ptr<spdlog::logger> _LogPtr = nullptr;

public:

  enum LogMessageLevel {
	DEBUG	  = 0x01,
	INFO	  = 0x02,
	WARNING	  = 0x03,
	ERR		  = 0x04,
  };

  void InitLog(_LogInitParams config);
  void PrintLog(LogMessageLevel msgLevel, std::string msgData);
  void ReleaseLog();

  GeneralOnnxLog() = default;

};
