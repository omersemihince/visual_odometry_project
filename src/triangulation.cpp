/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the triangulation of a 3D landmark from two image observations.

Notice,
  the file heavily based on the Prof.Grisetti's (and his lab's) slide probabilistic_robotics_25_essentials_in_computer_vision.pdf and the related MATLAB code, projective_geometry.
  theoric summary can be found in the readme file, the code structure is same as the summary.

Structure of the code is as follows,
  1. Receive two camera poses T_wc_a and T_wc_b, the camera matrix K, and two pixel measurements uv_a and uv_b.
  2. Recover the bearing vectors by undoing the camera matrix and normalizing the vectors.
  3. Convert the camera poses from T_wc convention into; camera center in world coordinates "t" & ray direction in world coordinates "d"
  4. Solve the LS system, [ray_a  -ray_b] [s_a s_b]^T = center_b - center_a
  5. Check if inputs are invalid, rays are almost parallel, scales are not positive, the triangulated point is behind one of the cameras.
  6. Compute the closest point on both rays.
  7. Return the midpoint of the two closest ray points as p_world.
  8. Return the two positive scales and the ray gap as quality information.
*/

#include "triangulation.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include <cmath>
#include <limits>

namespace {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

constexpr float kMinPositiveScale = 1e-6f;
constexpr float kMinBearingNorm = 1e-9f;
constexpr float kMinRaySinAngle = 1e-6f;
constexpr float kMinKDeterminant = 1e-9f;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

bool isFinitePositive(float value) {
  return std::isfinite(value) && value > kMinPositiveScale;
}

bool isValidTransform(const Eigen::Isometry3f& T) {
  return T.matrix().allFinite();
}

// Recover normalized bearing vector from image coordinates.
bool bearingFromPixel(
    const Eigen::Matrix3f& K_inv,
    const Eigen::Vector2f& uv,
    Eigen::Vector3f& bearing) {
  bearing.setZero();

  if (!uv.allFinite()) {
    return false;
  }

  Eigen::Vector3f z;
  z << uv.x(), uv.y(), 1.0f;

  Eigen::Vector3f d = K_inv * z;

  if (!d.allFinite()) {
    return false;
  }

  const float n = d.norm();

  if (n <= kMinBearingNorm || !std::isfinite(n)) {
    return false;
  }

  d /= n;

  //In the pinhole camera model, the ray should point forward in the camera frame.
  if (d.z() <= kMinPositiveScale) {
    return false;
  }

  bearing = d;
  return true;
}


//Camera center in the world coordinates is: C_world = -R_wc^T * t_wc
Eigen::Vector3f cameraCenterWorld(const Eigen::Isometry3f& T_wc) {
  const Eigen::Matrix3f R_wc = T_wc.linear();
  const Eigen::Vector3f t_wc = T_wc.translation();

  return -R_wc.transpose() * t_wc;
}

// Convert a bearing vector from camera coordinates to world coordinates.
// Since T_wc maps world to camera, we need to invert the rotation to get the ray direction in world coordinates.
// R_cw = R_wc^T -> ray_world = R_cw * bearing_camera
bool rayDirectionWorld(
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& bearing_camera,
    Eigen::Vector3f& ray_world) {
  ray_world = T_wc.linear().transpose() * bearing_camera;

  if (!ray_world.allFinite()) {
    return false;
  }

  const float n = ray_world.norm();

  if (n <= kMinBearingNorm || !std::isfinite(n)) {
    return false;
  }

  ray_world /= n;
  return true;
}

// Check whether the final 3D point is in front of both cameras.
bool hasPositiveCameraDepth(
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world) {
  const Eigen::Vector3f p_camera = T_wc * p_world;

  if (!p_camera.allFinite()) {
    return false;
  }

  return p_camera.z() > kMinPositiveScale;
}

}  // namespace

// -----------------------------------------------------------------------------
// Triangulation function
// -----------------------------------------------------------------------------

TriangulationResult triangulateTwoViews(
    const Eigen::Matrix3f& K,
    const Eigen::Isometry3f& T_wc_a,
    const Eigen::Vector2f& uv_a,
    const Eigen::Isometry3f& T_wc_b,
    const Eigen::Vector2f& uv_b) {
  TriangulationResult result;

  // ---------------------------------------------------------------------------
  // Check if inputs valid
  // ---------------------------------------------------------------------------

  if (!K.allFinite()) {
    return result;
  }

  if (std::abs(K.determinant()) <= kMinKDeterminant) {
    return result;
  }

  if (!isValidTransform(T_wc_a) || !isValidTransform(T_wc_b)) {
    return result;
  }

  if (!uv_a.allFinite() || !uv_b.allFinite()) {
    return result;
  }

  const Eigen::Matrix3f K_inv = K.inverse();

  // ---------------------------------------------------------------------------
  // Recover bearing vectors in camera coordinates
  // ---------------------------------------------------------------------------

  Eigen::Vector3f bearing_a_camera = Eigen::Vector3f::Zero();
  Eigen::Vector3f bearing_b_camera = Eigen::Vector3f::Zero();

  if (!bearingFromPixel(K_inv, uv_a, bearing_a_camera)) {
    return result;
  }

  if (!bearingFromPixel(K_inv, uv_b, bearing_b_camera)) {
    return result;
  }

  // ---------------------------------------------------------------------------
  // Convert camera poses into world-frame centers and world-frame rays
  // ---------------------------------------------------------------------------

  const Eigen::Vector3f center_a_world = cameraCenterWorld(T_wc_a);
  const Eigen::Vector3f center_b_world = cameraCenterWorld(T_wc_b);

  if (!center_a_world.allFinite() || !center_b_world.allFinite()) {
    return result;
  }

  Eigen::Vector3f ray_a_world = Eigen::Vector3f::Zero();
  Eigen::Vector3f ray_b_world = Eigen::Vector3f::Zero();

  if (!rayDirectionWorld(T_wc_a, bearing_a_camera, ray_a_world)) {
    return result;
  }

  if (!rayDirectionWorld(T_wc_b, bearing_b_camera, ray_b_world)) {
    return result;
  }

  // If the rays are almost parallel, triangulation is numerically unstable.
  const float ray_sin_angle = ray_a_world.cross(ray_b_world).norm();

  if (!std::isfinite(ray_sin_angle) || ray_sin_angle <= kMinRaySinAngle) {
    return result;
  }

  // ---------------------------------------------------------------------------
  // Solve for sclaes,
  // [ray_a  -ray_b] [s_a s_b]^T = center_b - center_a
  // ---------------------------------------------------------------------------

  Eigen::Matrix<float, 3, 2> A;
  A.col(0) = ray_a_world;
  A.col(1) = -ray_b_world;

  const Eigen::Vector3f b = center_b_world - center_a_world;

  Eigen::ColPivHouseholderQR<Eigen::Matrix<float, 3, 2>> qr(A);

  if (qr.rank() < 2) {
    return result;
  }

  const Eigen::Vector2f scales = qr.solve(b);

  if (!scales.allFinite()) {
    return result;
  }

  const float s_a = scales.x();
  const float s_b = scales.y();

  // Accept only points in front of both cameras.
  if (!isFinitePositive(s_a) || !isFinitePositive(s_b)) {
    return result;
  }

  // ---------------------------------------------------------------------------
  // Compute closest points on the two rays
  // ---------------------------------------------------------------------------

  const Eigen::Vector3f point_on_ray_a = center_a_world + s_a * ray_a_world;
  const Eigen::Vector3f point_on_ray_b = center_b_world + s_b * ray_b_world;

  if (!point_on_ray_a.allFinite() || !point_on_ray_b.allFinite()) {
    return result;
  }

  const Eigen::Vector3f p_world = 0.5f * (point_on_ray_a + point_on_ray_b);

  if (!p_world.allFinite()) {
    return result;
  }

  // Extra depth check using the final midpoint.
  if (!hasPositiveCameraDepth(T_wc_a, p_world)) {
    return result;
  }

  if (!hasPositiveCameraDepth(T_wc_b, p_world)) {
    return result;
  }

  // ---------------------------------------------------------------------------
  // Results
  // ---------------------------------------------------------------------------

  result.success = true;
  result.p_world = p_world;

  // These are the positive ray scales s_a and s_b from the theory summary.
  result.depth_a = s_a;
  result.depth_b = s_b;

  // If the two rays do not intersect perfectly because of noise,
  // this is the distance between their closest points.
  result.ray_gap = (point_on_ray_a - point_on_ray_b).norm();

  if (!std::isfinite(result.ray_gap)) {
    result.success = false;
    result.p_world.setZero();
    result.depth_a = 0.0f;
    result.depth_b = 0.0f;
    result.ray_gap = 0.0f;
  }

  return result;
}