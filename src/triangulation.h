/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project, 
  covers the process of defining the basic objects that will be needed for triangulation of the 3D points from two views.

The implementation is in triangulation.cpp.

The structure of the code is as follows,
  Define the output of the triangulation function, which includes whether the triangulation was successful, the 3D position of the landmark in the world frame, the depth of the landmark in both camera frames, and the ray gap.
  Declare the triangulation function. It takes the camera intrinsics K, the poses of the two cameras T_wc_a and T_wc_b, and the pixel locations of the landmark in both images uv_a and uv_b. It returns a TriangulationResult struct.
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

//Define the output of the triangulation function, which includes whether the triangulation was successful, the 3D position of the landmark in the world frame, the depth of the landmark in both camera frames, and the ray gap.
struct TriangulationResult {
  bool success = false;

  Eigen::Vector3f p_world = Eigen::Vector3f::Zero();

  //Depth from two cameras (s_a and s_b)
  float depth_a = 0.0f;
  float depth_b = 0.0f;

  //Gap between the two rays if noise present the intersection of camera rays will not be perfect, so we need the defiene the error "gap".
  float ray_gap = 0.0f;
};

//Declaring the triangulation function. It takes the camera intrinsics K, the poses of the two cameras T_wc_a and T_wc_b, and the pixel locations of the landmark in both images uv_a and uv_b. It returns a TriangulationResult struct.
TriangulationResult triangulateTwoViews(
    const Eigen::Matrix3f& K,
    const Eigen::Isometry3f& T_wc_a,
    const Eigen::Vector2f& uv_a,
    const Eigen::Isometry3f& T_wc_b,
    const Eigen::Vector2f& uv_b);