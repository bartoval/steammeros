cmake_minimum_required(VERSION 3.24)
project(stems VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Go links PIE executables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # build/compile_commands.json, so the editor sees the real flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# pocketfft: header-only FFT (BSD-3), the same code NumPy uses.
FetchContent_Declare(pocketfft
  URL https://github.com/mreineck/pocketfft/archive/4d865cba24359b05b37da4e75aee24498c3b78e7.tar.gz
  URL_HASH SHA256=db236e44b1020dd0b415e5040ce10d8be604fefb7ce83ec0b32bc0f78975aa2f)
FetchContent_MakeAvailable(pocketfft)

# STEMS_CUDA=ON: run the model on NVIDIA GPUs. Links the official ONNX Runtime GPU build
# dynamically (tools/fetch_ort_gpu.sh); its CUDA provider is a shared library by design.
# OFF (default): ONNX Runtime built from source and linked statically (tools/build_ort.sh).
option(STEMS_CUDA "Build against the ONNX Runtime CUDA release" OFF)

set(ORT_REQUIREMENTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/requirements.lock)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ORT_REQUIREMENTS})
file(STRINGS ${ORT_REQUIREMENTS} ORT_VERSION_LINE REGEX "^onnxruntime==")
string(REPLACE "onnxruntime==" "" STEMS_ORT_VERSION "${ORT_VERSION_LINE}")
if(STEMS_CUDA)
  set(ORT_GPU_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-gpu)
  if(NOT EXISTS ${ORT_GPU_DIR}/lib/libonnxruntime.so)
    message(FATAL_ERROR "ONNX Runtime GPU not there yet: run tools/fetch_ort_gpu.sh first")
  endif()
  set(ORT_INCLUDE ${ORT_GPU_DIR}/include)
  set(ORT_ARCHIVES "")  # nothing to merge: ONNX Runtime stays a shared library
  set(ORT_LIBS "-L${ORT_GPU_DIR}/lib -Wl,-rpath,${ORT_GPU_DIR}/lib -lonnxruntime")
else()
  set(ORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)
  set(ORT_BUILD ${ORT_DIR}/build/Linux/Release)
  if(NOT EXISTS ${ORT_BUILD}/libonnxruntime_session.a)
    message(FATAL_ERROR "ONNX Runtime not built yet: run tools/build_ort.sh first")
  endif()
  set(ORT_INCLUDE ${ORT_DIR}/include/onnxruntime/core/session)
  file(GLOB_RECURSE ORT_ARCHIVES ${ORT_BUILD}/*.a)
  list(FILTER ORT_ARCHIVES EXCLUDE REGEX "/CMakeFiles/|libprotoc|libgtest|libgmock")
  set(ORT_LIBS "")
endif()

# Our code.
find_package(Threads REQUIRED)
add_library(stems_core STATIC
  src/api.cpp
  src/resample.cpp
  src/scnet.cpp
  src/separator.cpp
  src/spectrogram.cpp)
target_include_directories(stems_core PUBLIC include)
target_include_directories(stems_core PRIVATE
  ${pocketfft_SOURCE_DIR}
  ${ORT_INCLUDE})
target_compile_definitions(stems_core PRIVATE STEMS_VERSION="${PROJECT_VERSION}")
target_compile_definitions(stems_core PRIVATE STEMS_ORT_VERSION="${STEMS_ORT_VERSION}")
target_compile_options(stems_core PRIVATE -Wall -Wextra)
target_link_libraries(stems_core PRIVATE Threads::Threads)

# libstems.a = our code + all of ONNX Runtime in one archive, so users link a single file
# (with STEMS_CUDA, just our code).
# ar's MRI script mode can merge archives (plain `ar` would nest them).
set(STEMS_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/libstems.a)
set(MRI "CREATE ${STEMS_ARCHIVE}\nADDLIB $<TARGET_FILE:stems_core>\n")
foreach(archive IN LISTS ORT_ARCHIVES)
  string(APPEND MRI "ADDLIB ${archive}\n")
endforeach()
string(APPEND MRI "SAVE\nEND\n")
file(GENERATE OUTPUT merge.mri CONTENT "${MRI}")

add_custom_command(OUTPUT ${STEMS_ARCHIVE}
  COMMAND ${CMAKE_COMMAND} -E rm -f ${STEMS_ARCHIVE}
  COMMAND ${CMAKE_AR} -M < ${CMAKE_CURRENT_BINARY_DIR}/merge.mri
  DEPENDS stems_core ${ORT_ARCHIVES}
  COMMENT "Merging our code and ONNX Runtime into libstems.a")
add_custom_target(stems ALL DEPENDS ${STEMS_ARCHIVE})

# pkg-config file consumed by cgo (`#cgo pkg-config: stems`).
file(GENERATE OUTPUT pkgconfig/stems.pc CONTENT "Name: stems
Description: In-process music source separation
Version: ${PROJECT_VERSION}
Cflags: -I${CMAKE_CURRENT_SOURCE_DIR}/include
Libs: ${STEMS_ARCHIVE} ${ORT_LIBS} -lstdc++ -lm -pthread
")
