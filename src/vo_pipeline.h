/*
version 0.2
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the visual odometry pipeline.
*/

#pragma once

#include "dataset_io.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Isometry3fVector =
    std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>>;

struct VOPipelineParams {
  float initial_translation_scale = 1.0f;
  float init_max_reprojection_error_px = 5.0f;
  float init_max_ray_gap = 0.25f;

  int picp_iterations = 10;
  int min_known_matches = 6;
  int min_inliers = 6;
  int min_projectable_matches = 8;

  float required_projectable_ratio = 0.50f;
  float acceptance_ratio = 0.80f;
  float loose_acceptance_ratio = 0.60f;

  float triangulation_max_reprojection_error_px = 8.0f;
  float triangulation_max_ray_gap = 0.25f;
  float triangulation_max_normalized_ray_gap = 0.05f;
  float triangulation_min_baseline = 0.005f;

  bool silence_picp_solver = true;
};

struct VOPipelineFrameLog {
  int seq = -1;

  bool pose_valid = false;
  bool map_updated = false;
  bool continuity_only = false;

  int image_landmarks = 0;
  int map_landmarks = 0;
  int map_projectable = 0;

  int known = 0;
  int projectable = 0;
  int required_projectable = 0;
  int inliers = 0;

  int new_candidates = 0;
  int tri_attempts = 0;
  int tri_success = 0;
  int tri_accepted = 0;

  int new_points = 0;

  float kernel = 0.0f;
  float mean_error_px = 0.0f;
  float median_error_px = 0.0f;
  float inlier_projectable_ratio = 0.0f;
  float inlier_known_ratio = 0.0f;

  std::string status;
};
struct VOPipelineResult {
  bool success = false;
  std::string message;

  Isometry3fVector poses;
  std::vector<int> pose_sequences;

  std::unordered_map<int, Landmark> landmarks;
  std::vector<VOPipelineFrameLog> frame_logs;
};

class VisualOdometryPipeline {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit VisualOdometryPipeline(const VOPipelineParams& params = VOPipelineParams());

  VOPipelineResult run(const VisualOdometryDataset& dataset);

private:
  using FrameObservation = std::pair<int, Eigen::Vector2f>;
  using ObservationHistory =
      std::vector<FrameObservation, Eigen::aligned_allocator<FrameObservation>>;
  using ObservationHistoryById = std::unordered_map<int, ObservationHistory>;

  void reset();

  bool initialize(
      const std::vector<Frame>& frames,
      VOPipelineResult& result);

  bool processFrame(
      const Frame& frame,
      VOPipelineFrameLog& log);

  void fillResult(VOPipelineResult& result) const;

  void addPose(
      int seq,
      const Eigen::Isometry3f& T_wc);

  bool poseAt(
      int seq,
      Eigen::Isometry3f& T_wc) const;

  void addObservations(
      const Frame& frame);

  int addNewPoints(
      const Frame& frame,
      const Eigen::Isometry3f& T_wc,
      VOPipelineFrameLog* log);

  Isometry3fVector makeInitialGuesses() const;

  VOPipelineParams params_;
  CameraData camera_;

  std::unordered_map<int, Landmark> landmarks_;
  ObservationHistoryById observation_history_;

  Isometry3fVector poses_;
  std::vector<int> pose_sequences_;
  std::unordered_map<int, std::size_t> pose_index_by_seq_;
};