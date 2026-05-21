cmake_minimum_required(VERSION 3.16)

project(visual_odometry_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(repo_picp
    src/repo_picp/camera.cpp
    src/repo_picp/picp_solver.cpp
)

target_include_directories(repo_picp PUBLIC
    src/repo_picp
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(repo_picp PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_library(vo_core
    src/dataset_io.cpp
    src/epipolar_initializer.cpp
    src/triangulation.cpp
    src/vo_pipeline.cpp
    src/evaluation.cpp
)

target_include_directories(vo_core PUBLIC
    src
    src/repo_picp
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vo_core PUBLIC
    repo_picp
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

add_executable(vo_main
    app/vo_main.cpp
)

target_link_libraries(vo_main PRIVATE
    vo_core
)