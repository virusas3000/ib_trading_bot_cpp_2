cmake_minimum_required(VERSION 3.18)
project(trading_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ── pybind11 ──────────────────────────────────────────────────────────────
# Locate pybind11 installed via pip
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────
pybind11_add_module(trading_engine
    src/indicators.cpp
    src/signals.cpp
    src/bindings.cpp
)

target_include_directories(trading_engine PRIVATE include)

# ── Optimisation flags ────────────────────────────────────────────────────
target_compile_options(trading_engine PRIVATE
    -O3 -march=native -ffast-math -Wall -Wextra
)

# ── Install: copy .so to project root ────────────────────────────────────
install(TARGETS trading_engine DESTINATION ${CMAKE_SOURCE_DIR}/..)
