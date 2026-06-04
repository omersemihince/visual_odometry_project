/*
version 0.2
author: Ömer (Semih) İnce

Minimal executable for the Visual Odometry pipeline.

Usage:
  ./vo_main data
  ./vo_main data output/evaluation

Ground truth is loaded for evaluation only.
The VO pipeline itself does not use ground truth.
*/

#include "../src/dataset_io.h"
#include "../src/evaluation.h"
#include "../src/vo_pipeline.h"

#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  /*
  Main entry point.
  Loads the dataset, runs VO, prints one clean line per frame,
  evaluates poses/map against GT, and writes evaluation CSV/TXT files.
  */
  std::filesystem::path dataset_dir = "data";
  std::filesystem::path output_dir = "output/evaluation";

  if (argc > 1) {
    dataset_dir = argv[1];
  }

  if (argc > 2) {
    output_dir = argv[2];
  }

  VisualOdometryDataset dataset;
  std::string error_message;

  const bool load_ground_truth = true;

  if (!readVisualOdometryDataset(
          dataset_dir,
          dataset,
          load_ground_truth,
          &error_message)) {
    std::cerr << "dataset error: " << error_message << std::endl;
    return 1;
  }

  std::cout << "frames=" << dataset.frames.size()
            << " width=" << dataset.camera.width
            << " height=" << dataset.camera.height
            << " gt_poses=" << dataset.trajectory_gt.size()
            << " gt_map=" << dataset.world_gt.size()
            << std::endl;

  VOPipelineParams params;

  VisualOdometryPipeline pipeline(params);
  const VOPipelineResult result = pipeline.run(dataset);

  for (const VOPipelineFrameLog& log : result.frame_logs) {
    std::cout << "seq " << log.seq
              << " | " << log.status
              << " | img=" << log.image_landmarks
              << " known=" << log.known
              << " map=" << log.map_landmarks
              << " map_proj=" << log.map_projectable
              << " projectable=" << log.projectable
              << " inliers=" << log.inliers
              << " kernel=" << log.kernel
              << " median=" << log.median_error_px
              << " new_cand=" << log.new_candidates
              << " tri=" << log.tri_success << "/" << log.tri_attempts
              << " map+=" << log.new_points
              << std::endl;
  }

  std::cout << result.message << std::endl;

  const EvaluationResult evaluation =
      evaluateVisualOdometry(dataset, result);

  if (!evaluation.success) {
    std::cerr << "evaluation error: " << evaluation.message << std::endl;

    if (!result.success) {
      return 2;
    }

    return 3;
  }

  std::string evaluation_write_error;

  if (!writeEvaluationFiles(
          output_dir,
          dataset,
          result,
          evaluation,
          &evaluation_write_error)) {
    std::cerr << "evaluation write error: "
              << evaluation_write_error
              << std::endl;

    if (!result.success) {
      return 2;
    }

    return 4;
  }

  std::cout << "evaluation pose_pairs=" << evaluation.matched_pose_pairs
            << " mean_rot=" << evaluation.mean_rotation_trace_error
            << " median_scale_est_over_gt="
            << evaluation.median_translation_ratio_est_over_gt
            << " point_scale_to_gt=" << evaluation.point_scale_to_gt
            << " map_matches=" << evaluation.matched_landmarks
            << " map_rmse=" << evaluation.map_rmse
            << std::endl;

  std::cout << "evaluation files: " << output_dir << std::endl;

  if (!result.success) {
    return 2;
  }

  return 0;
}