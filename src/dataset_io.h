/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the objects forr dataset file reader "dataset_io.cpp".

Dataset files:
  camera.dat
  meas-XXXX.dat
  world.dat       -> only for evaluation
  trajectory.dat  -> only for evaluation
  trajectoy.dat   

Note that,
  world.dat and trajectory.dat / trajectoy.dat must not be used by the VO pipeline.
  They are only for evaluation/debugging.
*/

#pragma once

#include "vo_types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

// -----------------------------------------------------------------------------
// Camera information from camera.dat
// -----------------------------------------------------------------------------

struct CameraData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

  // Pose of camera w.r.t. robot.
  // The supervisor README says to ignore this for VO.
  Eigen::Isometry3f cam_transform = Eigen::Isometry3f::Identity();
  bool has_cam_transform = false;

  float z_near = 0.0f;
  float z_far = 0.0f;

  int width = 0;
  int height = 0;
};

// -----------------------------------------------------------------------------
// Ground truth map entry from world.dat
// -----------------------------------------------------------------------------

struct WorldGroundTruthPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int landmark_id = -1;

  Eigen::Vector3f position = Eigen::Vector3f::Zero();

  Eigen::Matrix<float, 10, 1> appearance =
      Eigen::Matrix<float, 10, 1>::Zero();

  bool has_appearance = false;
};

// -----------------------------------------------------------------------------
// Ground truth trajectory entry from trajectory.dat / trajectoy.dat
// -----------------------------------------------------------------------------

struct TrajectoryGroundTruthPose {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int pose_id = -1;

  // Dataset poses are planar:
  // [x, y, theta]
  Eigen::Vector3f odometry_pose_2d = Eigen::Vector3f::Zero();
  Eigen::Vector3f groundtruth_pose_2d = Eigen::Vector3f::Zero();
};

// -----------------------------------------------------------------------------
// Complete dataset container
// -----------------------------------------------------------------------------

struct VisualOdometryDataset {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraData camera;

  std::vector<Frame> frames;

  std::unordered_map<int, WorldGroundTruthPoint> world_gt;

  std::vector<TrajectoryGroundTruthPose> trajectory_gt;
};

// -----------------------------------------------------------------------------
// File/path helpers
// -----------------------------------------------------------------------------

int extractMeasurementSequenceFromFilename(const std::filesystem::path& path);

std::vector<std::filesystem::path> findMeasurementFiles(
    const std::filesystem::path& dataset_dir);

// -----------------------------------------------------------------------------
// Individual readers
// -----------------------------------------------------------------------------

bool readCameraDat(
    const std::filesystem::path& camera_path,
    CameraData& camera,
    std::string* error_message = nullptr);

bool readMeasurementDat(
    const std::filesystem::path& measurement_path,
    Frame& frame,
    int sequence_override = -1,
    std::string* error_message = nullptr);

bool readAllMeasurementFrames(
    const std::filesystem::path& dataset_dir,
    std::vector<Frame>& frames,
    std::string* error_message = nullptr);

bool readWorldDat(
    const std::filesystem::path& world_path,
    std::unordered_map<int, WorldGroundTruthPoint>& world_gt,
    std::string* error_message = nullptr);

bool readTrajectoryDat(
    const std::filesystem::path& trajectory_path,
    std::vector<TrajectoryGroundTruthPose>& trajectory_gt,
    std::string* error_message = nullptr);

// -----------------------------------------------------------------------------
// Full dataset reader
// -----------------------------------------------------------------------------

bool readVisualOdometryDataset(
    const std::filesystem::path& dataset_dir,
    VisualOdometryDataset& dataset,
    bool load_ground_truth = true,
    std::string* error_message = nullptr);

// -----------------------------------------------------------------------------
// Utility for later evaluation
// -----------------------------------------------------------------------------

Eigen::Isometry3f planarPoseToIsometry3f(const Eigen::Vector3f& pose_2d);