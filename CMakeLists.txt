cmake_minimum_required(VERSION 3.16)

project(visual_odometry_project LANGUAGES CXX)

# -----------------------------
# C++ settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Dependencies
# -----------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# -----------------------------
# Repo Projective ICP code
# -----------------------------
add_library(repo_picp
    src/repo_picp/camera.cpp
    src/repo_picp/picp_solver.cpp
)

target_include_directories(repo_picp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/repo_picp
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(repo_picp
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

# -----------------------------
# visual odometry code
# -----------------------------
add_library(vo_core
    src/dataset_io.cpp
    src/epipolar_initializer.cpp
    src/triangulation.cpp
    src/vo_pipeline.cpp
    src/evaluation.cpp
)

target_include_directories(vo_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/repo_picp
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vo_core
    PUBLIC
        repo_picp
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

# -----------------------------
# Main executable
# -----------------------------
add_executable(vo_main
    app/vo_main.cpp
)

target_link_libraries(vo_main
    PRIVATE
        vo_core
)

# -----------------------------
# Initialization test executable
# -----------------------------
add_executable(test_initialization
    tests/test_initialization.cpp
)

target_link_libraries(test_initialization
    PRIVATE
        vo_core
)

# -----------------------------
# Dataset reader test executable
# -----------------------------
add_executable(test_dataset_reader
    tests/test_dataset_reader.cpp
)

target_link_libraries(test_dataset_reader
    PRIVATE
        vo_core
)