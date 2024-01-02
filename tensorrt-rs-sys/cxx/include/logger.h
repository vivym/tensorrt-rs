#pragma once

#include <memory>
#include <string>
#include <NvInferRuntime.h>
#include "rust/cxx.h"

namespace trt_rs::logger {

using nvinfer1::ILogger;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;

    void log(int32_t severity, rust::Str msg) noexcept;

    void set_level(int32_t severity) noexcept;
};

std::unique_ptr<Logger> create_logger();

} // namespace trt_rs::logger

