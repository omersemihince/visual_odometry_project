/*
version 0.2
author: Ömer (Semih) İnce
*/

#pragma once

#include "dataset_io.h"
#include "vo_pipeline.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <filesystem>
#include <string>
#include <vector>

struct PoseEvaluationEntry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int seq0 = -1;
  int seq1 = -1;

  float rotation_trace_error = 0.0f;

  float estimated_translation_norm = 0.0f;
  float groundtruth_translation_norm = 0.0f;
  float translation_ratio_est_over_gt = 0.0f;
};

struct MapEvaluationEntry {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int landmark_id = -1;

  Eigen::Vector3f estimated = Eigen::Vector3f::Zero();
  Eigen::Vector3f estimated_scaled = Eigen::Vector3f::Zero();
  Eigen::Vector3f estimated_rigid_aligned = Eigen::Vector3f::Zero();
  Eigen::Vector3f groundtruth_in_first_frame = Eigen::Vector3f::Zero();

  float error = 0.0f;
  float error_rigid_aligned = 0.0f;
};

using PoseEvaluationEntryVector =
    std::vector<PoseEvaluationEntry, Eigen::aligned_allocator<PoseEvaluationEntry>>;

using MapEvaluationEntryVector =
    std::vector<MapEvaluationEntry, Eigen::aligned_allocator<MapEvaluationEntry>>;

struct EvaluationResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool success = false;
  std::string message;

  int matched_pose_pairs = 0;
  int matched_landmarks = 0;

  float mean_rotation_trace_error = 0.0f;
  float median_rotation_trace_error = 0.0f;

  float mean_translation_ratio_est_over_gt = 0.0f;
  float median_translation_ratio_est_over_gt = 0.0f;

  float point_scale_to_gt = 1.0f;

  float map_rmse = 0.0f;
  float map_mean_error = 0.0f;

  bool has_map_rigid_alignment = false;
  float map_rigid_aligned_rmse = 0.0f;
  float map_rigid_aligned_mean_error = 0.0f;

  Eigen::Matrix3f map_alignment_R = Eigen::Matrix3f::Identity();
  Eigen::Vector3f map_alignment_t = Eigen::Vector3f::Zero();

  PoseEvaluationEntryVector pose_errors;
  MapEvaluationEntryVector map_errors;
};

EvaluationResult evaluateVisualOdometry(
    const VisualOdometryDataset& dataset,
    const VOPipelineResult& vo_result);

bool writeEvaluationFiles(
    const std::filesystem::path& output_dir,
    const VisualOdometryDataset& dataset,
    const VOPipelineResult& vo_result,
    const EvaluationResult& evaluation,
    std::string* error_message = nullptr);