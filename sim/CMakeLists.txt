cmake_minimum_required(VERSION 3.16)

# sim/*.cpp 自动编成同名可执行文件（与 tests/ 同一套机制）
include(${PROJECT_SOURCE_DIR}/cmake/auto_executables.cmake)

auto_add_executables(
  DIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEFAULT_LIBS ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim
)
