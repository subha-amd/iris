// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <string>

namespace iris {
namespace logging {

inline std::shared_ptr<spdlog::logger>& get_logger() {
  static std::shared_ptr<spdlog::logger> logger = nullptr;
  return logger;
}

inline void init(int rank, spdlog::level::level_enum level = spdlog::level::info) {
  static bool initialized = false;
  if (initialized) {
    return;
  }

  // Create logger with rank-specific name
  std::string logger_name = "iris_rank_" + std::to_string(rank);
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  get_logger() = std::make_shared<spdlog::logger>(logger_name, console_sink);

  // Set pattern: [rank_X] [level] message
  get_logger()->set_pattern("[%n] [%^%l%$] %v");
  get_logger()->set_level(level);

  // Register with spdlog
  spdlog::register_logger(get_logger());

  initialized = true;
}

inline void init_from_env(int rank) {
  // Check environment variable for log level
  const char* env_level = std::getenv("IRIS_LOG_LEVEL");
  spdlog::level::level_enum level = spdlog::level::info;

  if (env_level != nullptr) {
    std::string level_str(env_level);
    if (level_str == "trace") {
      level = spdlog::level::trace;
    } else if (level_str == "debug") {
      level = spdlog::level::debug;
    } else if (level_str == "info") {
      level = spdlog::level::info;
    } else if (level_str == "warn" || level_str == "warning") {
      level = spdlog::level::warn;
    } else if (level_str == "error") {
      level = spdlog::level::err;
    } else if (level_str == "critical") {
      level = spdlog::level::critical;
    } else if (level_str == "off") {
      level = spdlog::level::off;
    }
  }

  init(rank, level);
}

} // namespace logging
} // namespace iris

// Convenience macros for logging (internal use only)
#define IRIS_LOG_TRACE(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->trace(__VA_ARGS__)
#define IRIS_LOG_DEBUG(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->debug(__VA_ARGS__)
#define IRIS_LOG_INFO(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->info(__VA_ARGS__)
#define IRIS_LOG_WARN(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->warn(__VA_ARGS__)
#define IRIS_LOG_ERROR(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->error(__VA_ARGS__)
#define IRIS_LOG_CRITICAL(...) if (::iris::logging::get_logger()) ::iris::logging::get_logger()->critical(__VA_ARGS__)

