/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the process of defining the basic objects that will be needed for epipolar geometry initialization.

The implementation is in epipolar_initializer.cpp.

The structure of the code is as follows,
  Declare the main epipolar initialization function.
  Declare the shorter call initialization function.
*/

#pragma once

#include "vo_types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

EpipolarInitResult initializeEpipolar(
    const Frame& frame0,
    const Frame& frame1,
    const Eigen::Matrix3f& K,
    float initial_translation_scale,
    float max_reprojection_error_px,
    float max_ray_gap);

EpipolarInitResult initializeEpipolar(
    const Frame& frame0,
    const Frame& frame1,
    const Eigen::Matrix3f& K);