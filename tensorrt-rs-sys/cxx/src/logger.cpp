#include <iostream>
#include "spdlog/spdlog.h"
#include "logger.h"

namespace trt_rs::logger {

void Logger::log(Severity severity, const char *msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            spdlog::critical(msg);
            break;
        case Severity::kERROR:
            spdlog::error(msg);
            break;
        case Severity::kWARNING:
            spdlog::warn(msg);
            break;
        case Severity::kINFO:
            spdlog::info(msg);
            break;
        case Severity::kVERBOSE:
            spdlog::debug(msg);
            break;
        default:
            spdlog::debug(msg);
            break;
    }
}

void Logger::log(int32_t severity, rust::Str msg) noexcept {
    const auto msg_str = std::string(msg);
    log(static_cast<Severity>(severity), msg_str.c_str());
}

void Logger::set_level(int32_t severity) noexcept {
    const auto level = static_cast<Severity>(severity);
    switch (level)
    {
    case Severity::kINTERNAL_ERROR:
        spdlog::set_level(spdlog::level::critical);
        break;
    case Severity::kERROR:
        spdlog::set_level(spdlog::level::err);
        break;
    case Severity::kWARNING:
        spdlog::set_level(spdlog::level::warn);
        break;
    case Severity::kINFO:
        spdlog::set_level(spdlog::level::info);
        break;
    case Severity::kVERBOSE:
        spdlog::set_level(spdlog::level::debug);
        break;
    default:
        break;
    }
}

std::unique_ptr<Logger> create_logger() {
    return std::make_unique<Logger>();
}

} // namespace trt_rs::logger

