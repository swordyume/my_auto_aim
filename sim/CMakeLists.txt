cmake_minimum_required(VERSION 3.16)

# 这个文件只能作为仓库根工程的一部分加载。
# 如果被 CLion 单独当成顶层工程加载，${PROJECT_SOURCE_DIR} 会变成 sim/，
# 下面 include 不到仓库根目录的 cmake/auto_executables.cmake。
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR
    "sim/CMakeLists.txt 不能单独加载。\n"
    "请在 CLion 中加载仓库根目录的 CMakeLists.txt：\n"
    "  ${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt")
endif()

# sim/*.cpp 自动编成同名可执行文件（与 tests/ 同一套机制）
include(${PROJECT_SOURCE_DIR}/cmake/auto_executables.cmake)

auto_add_executables(
  DIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEFAULT_LIBS ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim
)
