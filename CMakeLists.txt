# Static build: ONNX Runtime compiled from source (tools/build_ort.sh) and linked in, for a
# single binary that needs only system libraries. Linux. Not needed for the Go module's
# default build, which compiles the C++ itself and loads ONNX Runtime at run time.
#
# Produces, in the build folder:
#   libstems.a + pkgconfig/stems.pc                          C/C++ users: our code + ONNX Runtime
#   libonnxruntime-static.a + pkgconfig/onnxruntime-static.pc  Go with -tags stems_static
cmake_minimum_required(VERSION 3.24)

file(STRINGS stems/stems.h STEMS_VERSION_LINE REGEX "^#define STEMS_VERSION ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" STEMS_VERSION "${STEMS_VERSION_LINE}")
project(stems VERSION ${STEMS_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Go links PIE executables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # build/compile_commands.json, so the editor sees the real flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime static libraries, from tools/build_ort.sh.
set(ORT_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/build/Linux/Release)
if(NOT EXISTS ${ORT_BUILD}/libonnxruntime_session.a)
  message(FATAL_ERROR "ONNX Runtime not built yet: run tools/build_ort.sh first")
endif()
file(GLOB_RECURSE ORT_ARCHIVES ${ORT_BUILD}/*.a)
list(FILTER ORT_ARCHIVES EXCLUDE REGEX "/CMakeFiles/|libprotoc|libgtest|libgmock")

# Our code, in stems/. The ONNX Runtime and pocketfft headers are in stems/external/.
find_package(Threads REQUIRED)
add_library(stems_core STATIC
  stems/api.cpp stems/ort.cpp stems/resample.cpp stems/scnet.cpp stems/separator.cpp stems/spectrogram.cpp)
target_include_directories(stems_core PUBLIC stems)
target_include_directories(stems_core PRIVATE stems/external/onnxruntime stems/external/pocketfft)
target_compile_options(stems_core PRIVATE -Wall -Wextra)
target_link_libraries(stems_core PRIVATE Threads::Threads)

# Merges archives into one with ar's MRI script mode (plain `ar` would nest them).
function(merge_archives target output)
  set(mri "CREATE ${output}\n")
  foreach(archive IN LISTS ARGN)
    string(APPEND mri "ADDLIB ${archive}\n")
  endforeach()
  string(APPEND mri "SAVE\nEND\n")
  file(GENERATE OUTPUT ${target}.mri CONTENT "${mri}")
  add_custom_command(OUTPUT ${output}
    COMMAND ${CMAKE_COMMAND} -E rm -f ${output}
    COMMAND ${CMAKE_AR} -M < ${CMAKE_CURRENT_BINARY_DIR}/${target}.mri
    DEPENDS ${ARGN}
    COMMENT "Merging ${target}")
  add_custom_target(${target} ALL DEPENDS ${output})
endfunction()

set(STEMS_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libstems.a)
set(ORT_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime-static.a)
merge_archives(stems ${STEMS_ARCHIVE} $<TARGET_FILE:stems_core> ${ORT_ARCHIVES})
merge_archives(onnxruntime-static ${ORT_ARCHIVE} ${ORT_ARCHIVES})
add_dependencies(stems stems_core)

# pkg-config files (`#cgo pkg-config: ...`).
file(GENERATE OUTPUT pkgconfig/stems.pc CONTENT "Name: stems
Description: In-process music source separation, with ONNX Runtime linked in
Version: ${PROJECT_VERSION}
Cflags: -I${CMAKE_CURRENT_SOURCE_DIR}/stems
Libs: ${STEMS_ARCHIVE} -lstdc++ -lm -pthread
")
file(GENERATE OUTPUT pkgconfig/onnxruntime-static.pc CONTENT "Name: onnxruntime-static
Description: ONNX Runtime static libraries, for the Go module's stems_static tag
Version: ${PROJECT_VERSION}
Libs: ${ORT_ARCHIVE} -lstdc++ -lm -pthread
")
