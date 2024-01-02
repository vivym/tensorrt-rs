#include <iostream>
#include "logger.h"

namespace trt_rs::logger {

void Logger::log(Severity severity, const char *msg) noexcept {
    // Would advise using a proper logging utility such as https://github.com/gabime/spdlog
    // For the sake of this tutorial, will just log to the console.

    // Only log Warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << "[LOG]: " << msg << std::endl;
    }
}

void Logger::log(int32_t severity, rust::Str msg) noexcept {
    const auto msg_str = std::string(msg);
    log(static_cast<Severity>(severity), msg_str.c_str());
}

std::unique_ptr<Logger> create_logger() {
    return std::make_unique<Logger>();
}

} // namespace trt_rs::logger

