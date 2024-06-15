#include <io.h>

#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include "GeneralOnnxLog.h"

void GeneralOnnxLog::InitLog(_LogInitParams config) {
 
  _LogConfig = config;	// record current log config

  std::vector<spdlog::sink_ptr> sink_vec;
  if (_LogConfig.IsShowConsole) {
	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	sink_vec.push_back(console_sink);
  }

  if (_LogConfig.IsSaveFile && _access(_LogConfig.LogPath.c_str(), 0) == 0) {
	if (_LogConfig.LogFileSizeLimit == 0)  _LogConfig.LogFileSizeLimit = 1024 * 1024 * 3;
	if (_LogConfig.LogFileNumLimit == 0)  _LogConfig.LogFileNumLimit = 5;
	auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
	  _LogConfig.LogPath + "\\" + _LogConfig.LogName + ".log", 
	  _LogConfig.LogFileSizeLimit, _LogConfig.LogFileNumLimit, false);
	sink_vec.push_back(file_sink);
  }

  _LogPtr = std::make_shared<spdlog::logger>(_LogConfig.LogName, sink_vec.begin(), sink_vec.end());
  _LogPtr->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] [thread %t] %v");
  _LogPtr->set_level(spdlog::level::from_str("info"));
  _LogPtr->flush_on(spdlog::level::from_str("info"));

}

void GeneralOnnxLog::PrintLog(LogMessageLevel msgLevel, std::string msgData) {

  if (_LogPtr == nullptr) return;

  switch (msgLevel) {
  case DEBUG:
	_LogPtr->log(spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::debug, msgData);
	break;
  case INFO:
	_LogPtr->log(spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::info, msgData);
	break;
  case WARNING:
	_LogPtr->log(spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::warn, msgData);
	break;
  case ERR:
	_LogPtr->log(spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::critical, msgData);
	break;
  default:
	_LogPtr->log(spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::info, msgData);
	break;
  }
}

void GeneralOnnxLog::ReleaseLog() {
  _LogPtr.reset();
}