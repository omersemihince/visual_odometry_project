/*
version 0.2
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the evaluation implementation for the Visual Odometry project.

Note that ground truth is used only here.

The evaluator follows,
  1. Pose evaluation uses relative motion:
       rel_T = inv(T_0) * T_1
       rel_GT = inv(GT_0) * GT_1
       error_T = inv(rel_T) * rel_GT
  2. Rotation error:
       trace(I - error_R)
  3. Translation scale ratio:
       norm(rel_T.translation) / norm(rel_GT.translation)
  4. Map evaluation:
       use the median trajectory scale ratio to scale estimated landmarks
       compute raw RMSE against world.dat
       also compute rigid-aligned RMSE for map-shape accuracy.

Outputs are as follows,
  evaluation_summary.txt
  pose_errors.csv
  trajectory_estimate.csv
  trajectory_gt.csv
  trajectory_compare.csv
  map_estimate_scaled.csv
  map_errors.csv
  frame_logs.csv

New version solves, the inconsistency issue between the data world frame and choosen world frame. 
*/

#include "evaluation.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kMinTranslationNorm = 1e-6f;

bool fail(std::string* error_message, const std::string& message) {
  if (error_message) {
    *error_message = message;
  }

  return false;
}

std::unordered_map<int, TrajectoryGroundTruthPose> makeGroundTruthPoseMap(
    const std::vector<TrajectoryGroundTruthPose>& trajectory_gt) {
  std::unordered_map<int, TrajectoryGroundTruthPose> map;

  for (const TrajectoryGroundTruthPose& pose : trajectory_gt) {
    map[pose.pose_id] = pose;
  }

  return map;
}

bool findGroundTruthPose(
    const std::unordered_map<int, TrajectoryGroundTruthPose>& gt_by_seq,
    int seq,
    TrajectoryGroundTruthPose& pose) {
  const auto it = gt_by_seq.find(seq);

  if (it == gt_by_seq.end()) {
    return false;
  }

  pose = it->second;
  return true;
}

float median(std::vector<float> values) {
  if (values.empty()) {
    return 0.0f;
  }

  std::sort(values.begin(), values.end());

  const std::size_t mid = values.size() / 2;

  if (values.size() % 2 == 1) {
    return values[mid];
  }

  return 0.5f * (values[mid - 1] + values[mid]);
}

float mean(const std::vector<float>& values) {
  if (values.empty()) {
    return 0.0f;
  }

  double sum = 0.0;

  for (float value : values) {
    sum += static_cast<double>(value);
  }

  return static_cast<float>(sum / static_cast<double>(values.size()));
}

Eigen::Vector3f cameraCenterFromWorldInCameraPose(
    const Eigen::Isometry3f& T_wc) {
  return -T_wc.linear().transpose() * T_wc.translation();
}

void computeRigidAlignedMapError(EvaluationResult& evaluation) {
  /*
  Computes the best rigid alignment from the scaled estimated map to GT.
  This is useful because VO map coordinates are in the first camera frame,
  while world.dat is in the dataset world frame.
  */
  if (evaluation.map_errors.size() < 3) {
    return;
  }

  Eigen::Vector3f estimated_mean = Eigen::Vector3f::Zero();
  Eigen::Vector3f gt_mean = Eigen::Vector3f::Zero();

  for (const MapEvaluationEntry& entry : evaluation.map_errors) {
    estimated_mean += entry.estimated_scaled;
    gt_mean += entry.groundtruth_in_first_frame;
  }

  const float n =
      static_cast<float>(evaluation.map_errors.size());

  estimated_mean /= n;
  gt_mean /= n;

  Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

  for (const MapEvaluationEntry& entry : evaluation.map_errors) {
    const Eigen::Vector3f dx = entry.estimated_scaled - estimated_mean;
    const Eigen::Vector3f dy = entry.groundtruth_in_first_frame - gt_mean;

    H.noalias() += dx * dy.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      H,
      Eigen::ComputeFullU | Eigen::ComputeFullV);

  if (svd.matrixU().size() == 0 || svd.matrixV().size() == 0) {
    return;
  }

  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  if (!U.allFinite() || !V.allFinite()) {
    return;
  }

  Eigen::Matrix3f R = V * U.transpose();

  if (R.determinant() < 0.0f) {
    V.col(2) *= -1.0f;
    R = V * U.transpose();
  }

  if (!R.allFinite()) {
    return;
  }

  const Eigen::Vector3f t = gt_mean - R * estimated_mean;

  if (!t.allFinite()) {
    return;
  }

  double squared_error_sum = 0.0;
  double error_sum = 0.0;
  int valid_count = 0;

  for (MapEvaluationEntry& entry : evaluation.map_errors) {
    entry.estimated_rigid_aligned =
        R * entry.estimated_scaled + t;

    entry.error_rigid_aligned =
        (entry.estimated_rigid_aligned -
         entry.groundtruth_in_first_frame).norm();

    if (!std::isfinite(entry.error_rigid_aligned)) {
      continue;
    }

    squared_error_sum +=
        static_cast<double>(entry.error_rigid_aligned) *
        static_cast<double>(entry.error_rigid_aligned);

    error_sum += static_cast<double>(entry.error_rigid_aligned);

    ++valid_count;
  }

  if (valid_count == 0) {
    return;
  }

  evaluation.has_map_rigid_alignment = true;
  evaluation.map_alignment_R = R;
  evaluation.map_alignment_t = t;

  evaluation.map_rigid_aligned_rmse =
      static_cast<float>(
          std::sqrt(squared_error_sum /
                    static_cast<double>(valid_count)));

  evaluation.map_rigid_aligned_mean_error =
      static_cast<float>(
          error_sum / static_cast<double>(valid_count));
}

bool openOutputFile(
    std::ofstream& os,
    const fs::path& path,
    std::string* error_message) {
  os.open(path);

  if (!os) {
    return fail(error_message, "Could not write file: " + path.string());
  }

  os << std::fixed << std::setprecision(9);
  return true;
}

bool writePoseErrorsCsv(
    const fs::path& output_dir,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "pose_errors.csv", error_message)) {
    return false;
  }

  os << "seq0,seq1,rotation_trace_error,estimated_translation_norm,"
     << "groundtruth_translation_norm,translation_ratio_est_over_gt\n";

  for (const PoseEvaluationEntry& entry : evaluation.pose_errors) {
    os << entry.seq0 << ","
       << entry.seq1 << ","
       << entry.rotation_trace_error << ","
       << entry.estimated_translation_norm << ","
       << entry.groundtruth_translation_norm << ","
       << entry.translation_ratio_est_over_gt << "\n";
  }

  return true;
}

bool writeTrajectoryEstimateCsv(
    const fs::path& output_dir,
    const VOPipelineResult& vo_result,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "trajectory_estimate.csv", error_message)) {
    return false;
  }

  os << "seq,camera_x,camera_y,camera_z,scaled_camera_x,scaled_camera_y,"
     << "scaled_camera_z,qx,qy,qz,qw,t_wc_x,t_wc_y,t_wc_z\n";

  for (std::size_t i = 0; i < vo_result.poses.size(); ++i) {
    const Eigen::Isometry3f& T_wc = vo_result.poses[i];
    const Eigen::Vector3f center = cameraCenterFromWorldInCameraPose(T_wc);
    const Eigen::Vector3f center_scaled = evaluation.point_scale_to_gt * center;

    Eigen::Quaternionf q(T_wc.inverse().linear());
    q.normalize();

    os << vo_result.pose_sequences[i] << ","
       << center.x() << ","
       << center.y() << ","
       << center.z() << ","
       << center_scaled.x() << ","
       << center_scaled.y() << ","
       << center_scaled.z() << ","
       << q.x() << ","
       << q.y() << ","
       << q.z() << ","
       << q.w() << ","
       << T_wc.translation().x() << ","
       << T_wc.translation().y() << ","
       << T_wc.translation().z() << "\n";
  }

  return true;
}

bool writeTrajectoryGroundTruthCsv(
    const fs::path& output_dir,
    const VisualOdometryDataset& dataset,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "trajectory_gt.csv", error_message)) {
    return false;
  }

  os << "seq,gt_x,gt_y,gt_theta,odom_x,odom_y,odom_theta\n";

  for (const TrajectoryGroundTruthPose& pose : dataset.trajectory_gt) {
    os << pose.pose_id << ","
       << pose.groundtruth_pose_2d.x() << ","
       << pose.groundtruth_pose_2d.y() << ","
       << pose.groundtruth_pose_2d.z() << ","
       << pose.odometry_pose_2d.x() << ","
       << pose.odometry_pose_2d.y() << ","
       << pose.odometry_pose_2d.z() << "\n";
  }

  return true;
}

bool writeTrajectoryCompareCsv(
    const fs::path& output_dir,
    const VisualOdometryDataset& dataset,
    const VOPipelineResult& vo_result,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "trajectory_compare.csv", error_message)) {
    return false;
  }

  const std::unordered_map<int, TrajectoryGroundTruthPose> gt_by_seq =
      makeGroundTruthPoseMap(dataset.trajectory_gt);

  TrajectoryGroundTruthPose gt0;

  if (vo_result.pose_sequences.empty() ||
      !findGroundTruthPose(gt_by_seq, vo_result.pose_sequences.front(), gt0)) {
    return fail(error_message, "Could not find first ground-truth pose.");
  }

  const Eigen::Isometry3f T_gt0 = planarPoseToIsometry3f(gt0.groundtruth_pose_2d);

  os << "seq,est_x,est_y,est_z,est_scaled_x,est_scaled_y,est_scaled_z,"
     << "gt_initial_x,gt_initial_y,gt_initial_z,gt_x,gt_y,gt_theta\n";

  for (std::size_t i = 0; i < vo_result.poses.size(); ++i) {
    const int seq = vo_result.pose_sequences[i];

    TrajectoryGroundTruthPose gt_pose;

    if (!findGroundTruthPose(gt_by_seq, seq, gt_pose)) {
      continue;
    }

    const Eigen::Vector3f est_center =
        cameraCenterFromWorldInCameraPose(vo_result.poses[i]);

    const Eigen::Vector3f est_center_scaled =
        evaluation.point_scale_to_gt * est_center;

    const Eigen::Isometry3f T_gt =
        planarPoseToIsometry3f(gt_pose.groundtruth_pose_2d);

    const Eigen::Vector3f gt_center_initial =
        T_gt0.inverse() * T_gt.translation();

    os << seq << ","
       << est_center.x() << ","
       << est_center.y() << ","
       << est_center.z() << ","
       << est_center_scaled.x() << ","
       << est_center_scaled.y() << ","
       << est_center_scaled.z() << ","
       << gt_center_initial.x() << ","
       << gt_center_initial.y() << ","
       << gt_center_initial.z() << ","
       << gt_pose.groundtruth_pose_2d.x() << ","
       << gt_pose.groundtruth_pose_2d.y() << ","
       << gt_pose.groundtruth_pose_2d.z() << "\n";
  }

  return true;
}

bool writeMapEstimateCsv(
    const fs::path& output_dir,
    const VOPipelineResult& vo_result,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "map_estimate_scaled.csv", error_message)) {
    return false;
  }

  os << "landmark_id,est_x,est_y,est_z,scaled_x,scaled_y,scaled_z\n";

  std::vector<int> ids;
  ids.reserve(vo_result.landmarks.size());

  for (const auto& item : vo_result.landmarks) {
    ids.push_back(item.first);
  }

  std::sort(ids.begin(), ids.end());

  for (int id : ids) {
    const auto it = vo_result.landmarks.find(id);

    if (it == vo_result.landmarks.end()) {
      continue;
    }

    const Landmark& landmark = it->second;

    if (!landmark.initialized || !landmark.p_world.allFinite()) {
      continue;
    }

    const Eigen::Vector3f scaled =
        evaluation.point_scale_to_gt * landmark.p_world;

    os << id << ","
       << landmark.p_world.x() << ","
       << landmark.p_world.y() << ","
       << landmark.p_world.z() << ","
       << scaled.x() << ","
       << scaled.y() << ","
       << scaled.z() << "\n";
  }

  return true;
}

bool writeMapErrorsCsv(
    const fs::path& output_dir,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  /*
  Writes raw scaled map errors and rigid-aligned map errors.
  */
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "map_errors.csv", error_message)) {
    return false;
  }

  os << "landmark_id,est_x,est_y,est_z,scaled_x,scaled_y,scaled_z,"
     << "aligned_x,aligned_y,aligned_z,"
     << "gt_first_x,gt_first_y,gt_first_z,error,error_rigid_aligned\n";

  for (const MapEvaluationEntry& entry : evaluation.map_errors) {
    os << entry.landmark_id << ","
       << entry.estimated.x() << ","
       << entry.estimated.y() << ","
       << entry.estimated.z() << ","
       << entry.estimated_scaled.x() << ","
       << entry.estimated_scaled.y() << ","
       << entry.estimated_scaled.z() << ","
       << entry.estimated_rigid_aligned.x() << ","
       << entry.estimated_rigid_aligned.y() << ","
       << entry.estimated_rigid_aligned.z() << ","
       << entry.groundtruth_in_first_frame.x() << ","
       << entry.groundtruth_in_first_frame.y() << ","
       << entry.groundtruth_in_first_frame.z() << ","
       << entry.error << ","
       << entry.error_rigid_aligned << "\n";
  }

  return true;
}

bool writeFrameLogsCsv(
    const fs::path& output_dir,
    const VOPipelineResult& vo_result,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "frame_logs.csv", error_message)) {
    return false;
  }

  os << "seq,status,pose_valid,map_updated,continuity_only,img,known,map,"
     << "map_projectable,projectable,required_projectable,inliers,kernel,"
     << "mean_error_px,median_error_px,inlier_projectable_ratio,"
     << "inlier_known_ratio,new_candidates,tri_success,tri_attempts,"
     << "tri_accepted,new_points\n";

  for (const VOPipelineFrameLog& log : vo_result.frame_logs) {
    os << log.seq << ","
       << log.status << ","
       << log.pose_valid << ","
       << log.map_updated << ","
       << log.continuity_only << ","
       << log.image_landmarks << ","
       << log.known << ","
       << log.map_landmarks << ","
       << log.map_projectable << ","
       << log.projectable << ","
       << log.required_projectable << ","
       << log.inliers << ","
       << log.kernel << ","
       << log.mean_error_px << ","
       << log.median_error_px << ","
       << log.inlier_projectable_ratio << ","
       << log.inlier_known_ratio << ","
       << log.new_candidates << ","
       << log.tri_success << ","
       << log.tri_attempts << ","
       << log.tri_accepted << ","
       << log.new_points << "\n";
  }

  return true;
}

bool writeSummaryTxt(
    const fs::path& output_dir,
    const EvaluationResult& evaluation,
    const VOPipelineResult& vo_result,
    std::string* error_message) {
  std::ofstream os;

  if (!openOutputFile(os, output_dir / "evaluation_summary.txt", error_message)) {
    return false;
  }

  os << "VO result\n";
  os << "success: " << vo_result.success << "\n";
  os << "message: " << vo_result.message << "\n";
  os << "\n";

  os << "Pose evaluation\n";
  os << "matched_pose_pairs: " << evaluation.matched_pose_pairs << "\n";
  os << "mean_rotation_trace_error: "
     << evaluation.mean_rotation_trace_error << "\n";
  os << "median_rotation_trace_error: "
     << evaluation.median_rotation_trace_error << "\n";
  os << "mean_translation_ratio_est_over_gt: "
     << evaluation.mean_translation_ratio_est_over_gt << "\n";
  os << "median_translation_ratio_est_over_gt: "
     << evaluation.median_translation_ratio_est_over_gt << "\n";
  os << "point_scale_to_gt: " << evaluation.point_scale_to_gt << "\n";
  os << "\n";

  os << "Map evaluation\n";
  os << "matched_landmarks: " << evaluation.matched_landmarks << "\n";
  os << "map_rmse_raw: " << evaluation.map_rmse << "\n";
  os << "map_mean_error_raw: " << evaluation.map_mean_error << "\n";
  os << "map_rigid_aligned_rmse: "
     << evaluation.map_rigid_aligned_rmse << "\n";
  os << "map_rigid_aligned_mean_error: "
     << evaluation.map_rigid_aligned_mean_error << "\n";
  os << "has_map_rigid_alignment: "
     << evaluation.has_map_rigid_alignment << "\n";
  os << "\n";

  os << "Map rigid alignment\n";
  os << "R:\n";
  os << evaluation.map_alignment_R << "\n";
  os << "t: " << evaluation.map_alignment_t.transpose() << "\n";
  os << "\n";

  os << "status: " << evaluation.message << "\n";

  return true;
}

}  // namespace

EvaluationResult evaluateVisualOdometry(
    const VisualOdometryDataset& dataset,
    const VOPipelineResult& vo_result) {
  EvaluationResult evaluation;

  if (vo_result.poses.size() != vo_result.pose_sequences.size()) {
    evaluation.message = "Evaluation failed: pose vector and sequence vector have different sizes.";
    return evaluation;
  }

  if (vo_result.poses.size() < 2) {
    evaluation.message = "Evaluation failed: need at least two estimated poses.";
    return evaluation;
  }

  if (dataset.trajectory_gt.empty()) {
    evaluation.message = "Evaluation failed: trajectory ground truth is empty.";
    return evaluation;
  }

  const std::unordered_map<int, TrajectoryGroundTruthPose> gt_by_seq =
      makeGroundTruthPoseMap(dataset.trajectory_gt);

  std::vector<float> rotation_errors;
  std::vector<float> translation_ratios;

  for (std::size_t i = 1; i < vo_result.poses.size(); ++i) {
    const int seq0 = vo_result.pose_sequences[i - 1];
    const int seq1 = vo_result.pose_sequences[i];

    TrajectoryGroundTruthPose gt0;
    TrajectoryGroundTruthPose gt1;

    if (!findGroundTruthPose(gt_by_seq, seq0, gt0) ||
        !findGroundTruthPose(gt_by_seq, seq1, gt1)) {
      continue;
    }

    const Eigen::Isometry3f T_est0 = vo_result.poses[i - 1].inverse();
    const Eigen::Isometry3f T_est1 = vo_result.poses[i].inverse();

    const Eigen::Isometry3f rel_T = T_est0.inverse() * T_est1;

    const Eigen::Isometry3f T_gt0 =
        planarPoseToIsometry3f(gt0.groundtruth_pose_2d);

    const Eigen::Isometry3f T_gt1 =
        planarPoseToIsometry3f(gt1.groundtruth_pose_2d);

    const Eigen::Isometry3f rel_GT = T_gt0.inverse() * T_gt1;

    const Eigen::Isometry3f error_T = rel_T.inverse() * rel_GT;

    const float rotation_error =
        (Eigen::Matrix3f::Identity() - error_T.linear()).trace();

    const float est_norm = rel_T.translation().norm();
    const float gt_norm = rel_GT.translation().norm();

    float ratio = std::numeric_limits<float>::quiet_NaN();

    if (std::isfinite(est_norm) &&
        std::isfinite(gt_norm) &&
        est_norm > kMinTranslationNorm &&
        gt_norm > kMinTranslationNorm) {
      ratio = est_norm / gt_norm;
    }

    PoseEvaluationEntry entry;
    entry.seq0 = seq0;
    entry.seq1 = seq1;
    entry.rotation_trace_error = rotation_error;
    entry.estimated_translation_norm = est_norm;
    entry.groundtruth_translation_norm = gt_norm;
    entry.translation_ratio_est_over_gt = ratio;

    evaluation.pose_errors.push_back(entry);

    if (std::isfinite(rotation_error)) {
      rotation_errors.push_back(rotation_error);
    }

    if (std::isfinite(ratio)) {
      translation_ratios.push_back(ratio);
    }
  }

  evaluation.matched_pose_pairs =
      static_cast<int>(evaluation.pose_errors.size());

  if (evaluation.pose_errors.empty()) {
    evaluation.message = "Evaluation failed: no pose pairs matched ground truth.";
    return evaluation;
  }

  evaluation.mean_rotation_trace_error = mean(rotation_errors);
  evaluation.median_rotation_trace_error = median(rotation_errors);

  evaluation.mean_translation_ratio_est_over_gt = mean(translation_ratios);
  evaluation.median_translation_ratio_est_over_gt = median(translation_ratios);

  if (evaluation.median_translation_ratio_est_over_gt > kMinTranslationNorm &&
      std::isfinite(evaluation.median_translation_ratio_est_over_gt)) {
    evaluation.point_scale_to_gt =
        1.0f / evaluation.median_translation_ratio_est_over_gt;
  }

  if (!dataset.world_gt.empty() && !vo_result.pose_sequences.empty()) {
    TrajectoryGroundTruthPose gt_first;

    if (findGroundTruthPose(gt_by_seq, vo_result.pose_sequences.front(), gt_first)) {
      const Eigen::Isometry3f T_gt_first =
          planarPoseToIsometry3f(gt_first.groundtruth_pose_2d);

      double squared_error_sum = 0.0;
      double error_sum = 0.0;

      std::vector<int> ids;
      ids.reserve(vo_result.landmarks.size());

      for (const auto& item : vo_result.landmarks) {
        ids.push_back(item.first);
      }

      std::sort(ids.begin(), ids.end());

      for (int id : ids) {
        const auto landmark_it = vo_result.landmarks.find(id);
        const auto gt_it = dataset.world_gt.find(id);

        if (landmark_it == vo_result.landmarks.end() ||
            gt_it == dataset.world_gt.end()) {
          continue;
        }

        const Landmark& landmark = landmark_it->second;

        if (!landmark.initialized || !landmark.p_world.allFinite()) {
          continue;
        }

        MapEvaluationEntry entry;
        entry.landmark_id = id;
        entry.estimated = landmark.p_world;
        entry.estimated_scaled = evaluation.point_scale_to_gt * landmark.p_world;
        entry.groundtruth_in_first_frame =
            T_gt_first.inverse() * gt_it->second.position;
        entry.error =
            (entry.estimated_scaled - entry.groundtruth_in_first_frame).norm();

        if (!std::isfinite(entry.error)) {
          continue;
        }

        evaluation.map_errors.push_back(entry);

        squared_error_sum +=
            static_cast<double>(entry.error) * static_cast<double>(entry.error);

        error_sum += static_cast<double>(entry.error);
      }

      evaluation.matched_landmarks =
          static_cast<int>(evaluation.map_errors.size());

      if (!evaluation.map_errors.empty()) {
        evaluation.map_rmse = static_cast<float>(
            std::sqrt(squared_error_sum /
                      static_cast<double>(evaluation.map_errors.size())));

        evaluation.map_mean_error = static_cast<float>(
            error_sum / static_cast<double>(evaluation.map_errors.size()));

        computeRigidAlignedMapError(evaluation);
      }
    }
  }

  evaluation.success = true;
  evaluation.message = "Evaluation done.";

  return evaluation;
}

bool writeEvaluationFiles(
    const std::filesystem::path& output_dir,
    const VisualOdometryDataset& dataset,
    const VOPipelineResult& vo_result,
    const EvaluationResult& evaluation,
    std::string* error_message) {
  std::error_code ec;
  fs::create_directories(output_dir, ec);

  if (ec) {
    return fail(
        error_message,
        "Could not create output directory: " + output_dir.string());
  }

  if (!writeSummaryTxt(output_dir, evaluation, vo_result, error_message)) {
    return false;
  }

  if (!writePoseErrorsCsv(output_dir, evaluation, error_message)) {
    return false;
  }

  if (!writeTrajectoryEstimateCsv(output_dir, vo_result, evaluation, error_message)) {
    return false;
  }

  if (!writeTrajectoryGroundTruthCsv(output_dir, dataset, error_message)) {
    return false;
  }

  if (!writeTrajectoryCompareCsv(
          output_dir,
          dataset,
          vo_result,
          evaluation,
          error_message)) {
    return false;
  }

  if (!writeMapEstimateCsv(output_dir, vo_result, evaluation, error_message)) {
    return false;
  }

  if (!writeMapErrorsCsv(output_dir, evaluation, error_message)) {
    return false;
  }

  if (!writeFrameLogsCsv(output_dir, vo_result, error_message)) {
    return false;
  }

  return true;
}