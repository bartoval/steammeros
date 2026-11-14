cmake_minimum_required(VERSION 3.24)
project(stems VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Go links PIE executables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # build/compile_commands.json, so the editor sees the real flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(stems STATIC src/version.cpp)
target_include_directories(stems PUBLIC include)
target_compile_definitions(stems PRIVATE STEMS_VERSION="${PROJECT_VERSION}")
target_compile_options(stems PRIVATE -Wall -Wextra)

# pkg-config file consumed by cgo (`#cgo pkg-config: stems`).
file(GENERATE OUTPUT pkgconfig/stems.pc CONTENT "Name: stems
Description: In-process music source separation (static archive)
Version: ${PROJECT_VERSION}
Cflags: -I${CMAKE_CURRENT_SOURCE_DIR}/include
Libs: $<TARGET_FILE:stems> -lstdc++ -lm
")
