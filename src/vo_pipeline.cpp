/*
version 0.2
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the visual odometry pipeline.

Loads the dataset (takes measurements).
Uses epipolar initialization for frames 0 and 1. Creates the initial map.
Projective ICP,
  build trusted 3D-2D matches from initialized landmarks,
  generate multiple inital pose guesses (motion model, constant pose, etc.),
  run multiple kernels with PICP and pick the best result,
  refine the best result with inliers and PICP again (If PICP fails introduce emergency kernels for continuity of the trajectory).

New version solves, unaccaptable performance issues by introducing multiple initialization & muliple kernel strategies as explained above. 
*/

#include "vo_pipeline.h"

#include "epipolar_initializer.h"
#include "triangulation.h"

#include "repo_picp/camera.h"
#include "repo_picp/picp_solver.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <streambuf>
#include <unordered_set>

namespace {

constexpr float kTriangulationMaxNormalizedRayGap = 0.05f;
constexpr float kTriangulationMinBaseline = 0.001f;
constexpr float kTriangulationMinReprojectionGatePx = 32.0f;

struct TrackedMatch {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int landmark_id = -1;
  Eigen::Vector3f p_world = Eigen::Vector3f::Zero();
  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
};

struct PoseCheck {
  bool accepted = false;

  int known = 0;
  int projectable = 0;
  int required_projectable = 0;
  int inliers = 0;

  float mean_error_px = std::numeric_limits<float>::infinity();
  float median_error_px = std::numeric_limits<float>::infinity();
  float inlier_projectable_ratio = 0.0f;
  float inlier_known_ratio = 0.0f;

  std::vector<int> inlier_indices;
};

struct Candidate {
  bool valid = false;

  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  PoseCheck check;

  float kernel = 0.0f;
};

class NullBuffer : public std::streambuf {
public:
  int overflow(int c) override {
    return c == std::char_traits<char>::eof() ? 0 : c;
  }
};

class ScopedCerrSilencer {
public:
  explicit ScopedCerrSilencer(bool enabled) {
    if (enabled) {
      old_ = std::cerr.rdbuf(&buffer_);
    }
  }

  ~ScopedCerrSilencer() {
    if (old_) {
      std::cerr.rdbuf(old_);
    }
  }

private:
  NullBuffer buffer_;
  std::streambuf* old_ = nullptr;
};

bool isValidPose(const Eigen::Isometry3f& T) {
  if (!T.matrix().allFinite()) {
    return false;
  }

  const float det = T.linear().determinant();

  if (!std::isfinite(det)) {
    return false;
  }

  return std::abs(det - 1.0f) < 1e-2f;
}

bool projectPoint(
    const CameraData& camera,
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world,
    Eigen::Vector2f& uv) {
  const Eigen::Vector3f p_camera = T_wc * p_world;

  if (!p_camera.allFinite() || p_camera.z() <= 1e-6f) {
    return false;
  }

  const Eigen::Vector3f z = camera.K * p_camera;

  if (!z.allFinite() || std::abs(z.z()) <= 1e-9f) {
    return false;
  }

  uv.x() = z.x() / z.z();
  uv.y() = z.y() / z.z();

  if (!uv.allFinite()) {
    return false;
  }

  if (uv.x() < 0.0f || uv.x() > static_cast<float>(camera.width - 1)) {
    return false;
  }

  if (uv.y() < 0.0f || uv.y() > static_cast<float>(camera.height - 1)) {
    return false;
  }

  return true;
}
int countInitializedMapPoints(
    const std::unordered_map<int, Landmark>& landmarks) {
  int count = 0;

  for (const auto& item : landmarks) {
    const Landmark& landmark = item.second;

    if (landmark.initialized && landmark.p_world.allFinite()) {
      ++count;
    }
  }

  return count;
}

int countProjectableMapPoints(
    const std::unordered_map<int, Landmark>& landmarks,
    const CameraData& camera,
    const Eigen::Isometry3f& T_wc) {
  int count = 0;

  for (const auto& item : landmarks) {
    const Landmark& landmark = item.second;

    if (!landmark.initialized || !landmark.p_world.allFinite()) {
      continue;
    }

    Eigen::Vector2f uv;

    if (projectPoint(camera, T_wc, landmark.p_world, uv)) {
      ++count;
    }
  }

  return count;
}

float reprojectionErrorPx(
    const CameraData& camera,
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world,
    const Eigen::Vector2f& uv_measured) {
  Eigen::Vector2f uv_projected;

  if (!projectPoint(camera, T_wc, p_world, uv_projected)) {
    return std::numeric_limits<float>::infinity();
  }

  return (uv_projected - uv_measured).norm();
}

Eigen::Vector3f cameraCenterWorldFromPose(
    const Eigen::Isometry3f& T_wc) {
  return -T_wc.linear().transpose() * T_wc.translation();
}

float cameraBaselineWorld(
    const Eigen::Isometry3f& T_wc_a,
    const Eigen::Isometry3f& T_wc_b) {
  const Eigen::Vector3f c_a = cameraCenterWorldFromPose(T_wc_a);
  const Eigen::Vector3f c_b = cameraCenterWorldFromPose(T_wc_b);

  if (!c_a.allFinite() || !c_b.allFinite()) {
    return 0.0f;
  }

  return (c_a - c_b).norm();
}

bool triangulationGapAccepted(
    const TriangulationResult& tri,
    const VOPipelineParams& params) {
  if (!tri.success) {
    return false;
  }

  const float mean_depth = 0.5f * (tri.depth_a + tri.depth_b);

  if (!std::isfinite(mean_depth) || mean_depth <= 0.0f) {
    return false;
  }

  const float adaptive_gap =
      std::max(params.triangulation_max_ray_gap,
               kTriangulationMaxNormalizedRayGap * mean_depth);

  return tri.ray_gap <= adaptive_gap;
}

float medianOfSorted(std::vector<float>& values) {
  if (values.empty()) {
    return std::numeric_limits<float>::infinity();
  }

  std::sort(values.begin(), values.end());

  const std::size_t mid = values.size() / 2;

  if (values.size() % 2 == 1) {
    return values[mid];
  }

  return 0.5f * (values[mid - 1] + values[mid]);
}

PoseCheck checkPose(
    const std::vector<TrackedMatch>& matches,
    const CameraData& camera,
    const Eigen::Isometry3f& T_wc,
    float kernel,
    const VOPipelineParams& params,
    float acceptance_ratio) {
  PoseCheck check;

  check.known = static_cast<int>(matches.size());

  if (check.known == 0 || kernel <= 0.0f || !isValidPose(T_wc)) {
    return check;
  }

  const int by_ratio =
      static_cast<int>(std::ceil(params.required_projectable_ratio *
                                 static_cast<float>(check.known)));

  check.required_projectable =
      std::max(params.min_projectable_matches, by_ratio);

  std::vector<float> errors;
  errors.reserve(matches.size());

  for (std::size_t i = 0; i < matches.size(); ++i) {
    Eigen::Vector2f uv_projected;

    if (!projectPoint(camera, T_wc, matches[i].p_world, uv_projected)) {
      continue;
    }

    const float err = (uv_projected - matches[i].uv).norm();

    if (!std::isfinite(err)) {
      continue;
    }

    ++check.projectable;
    errors.push_back(err);

    if (err * err <= kernel) {
      ++check.inliers;
      check.inlier_indices.push_back(static_cast<int>(i));
    }
  }

  if (!errors.empty()) {
    float sum = 0.0f;

    for (float err : errors) {
      sum += err;
    }

    check.mean_error_px = sum / static_cast<float>(errors.size());
    check.median_error_px = medianOfSorted(errors);
  }

  if (check.projectable > 0) {
    check.inlier_projectable_ratio =
        static_cast<float>(check.inliers) /
        static_cast<float>(check.projectable);
  }

  if (check.known > 0) {
    check.inlier_known_ratio =
        static_cast<float>(check.inliers) /
        static_cast<float>(check.known);
  }

  check.accepted =
      check.projectable >= check.required_projectable &&
      check.inliers >= params.min_inliers &&
      check.inlier_projectable_ratio >= acceptance_ratio;

  return check;
}

std::vector<TrackedMatch> selectMatches(
    const std::vector<TrackedMatch>& matches,
    const std::vector<int>& indices) {
  std::vector<TrackedMatch> selected;
  selected.reserve(indices.size());

  for (int index : indices) {
    if (index >= 0 && index < static_cast<int>(matches.size())) {
      selected.push_back(matches[static_cast<std::size_t>(index)]);
    }
  }

  return selected;
}

Eigen::Isometry3f solvePICPPose(
    const std::vector<TrackedMatch>& matches,
    const CameraData& camera,
    const Eigen::Isometry3f& initial_guess,
    float kernel,
    const VOPipelineParams& params) {
  if (matches.empty()) {
    return initial_guess;
  }

  pr::Vector3fVector world_points;
  pr::Vector2fVector image_points;
  pr::IntPairVector correspondences;

  world_points.reserve(matches.size());
  image_points.reserve(matches.size());
  correspondences.reserve(matches.size());

  for (std::size_t i = 0; i < matches.size(); ++i) {
    world_points.push_back(matches[i].p_world);
    image_points.push_back(matches[i].uv);
    correspondences.push_back(pr::IntPair(static_cast<int>(i),
                                          static_cast<int>(i)));
  }

  pr::Camera pr_camera(
      camera.height,
      camera.width,
      camera.K,
      initial_guess);

  pr::PICPSolver solver;
  solver.init(pr_camera, world_points, image_points);
  solver.setKernelThreshold(kernel);

  ScopedCerrSilencer silencer(params.silence_picp_solver);

  for (int i = 0; i < params.picp_iterations; ++i) {
    if (!solver.oneRound(correspondences, true)) {
      break;
    }
  }

  const Eigen::Isometry3f result = solver.camera().worldInCameraPose();

  if (!isValidPose(result)) {
    return initial_guess;
  }

  return result;
}

bool betterCandidate(
    const Candidate& candidate,
    const Candidate& best) {
  if (!candidate.valid) {
    return false;
  }

  if (!best.valid) {
    return true;
  }

  if (candidate.check.accepted != best.check.accepted) {
    return candidate.check.accepted;
  }

  if (candidate.check.inliers != best.check.inliers) {
    return candidate.check.inliers > best.check.inliers;
  }

  if (candidate.check.inlier_projectable_ratio !=
      best.check.inlier_projectable_ratio) {
    return candidate.check.inlier_projectable_ratio >
           best.check.inlier_projectable_ratio;
  }

  if (candidate.check.inlier_known_ratio != best.check.inlier_known_ratio) {
    return candidate.check.inlier_known_ratio > best.check.inlier_known_ratio;
  }

  if (candidate.check.median_error_px != best.check.median_error_px) {
    return candidate.check.median_error_px < best.check.median_error_px;
  }

  return candidate.check.mean_error_px < best.check.mean_error_px;
}

Candidate runCandidate(
    const std::vector<TrackedMatch>& matches,
    const CameraData& camera,
    const Eigen::Isometry3f& guess,
    float kernel,
    const VOPipelineParams& params,
    float acceptance_ratio) {
  Candidate candidate;
  candidate.kernel = kernel;

  if (!isValidPose(guess)) {
    return candidate;
  }

  const Eigen::Isometry3f solved =
      solvePICPPose(matches, camera, guess, kernel, params);

  candidate.pose = solved;
  candidate.check =
      checkPose(matches, camera, solved, kernel, params, acceptance_ratio);
  candidate.valid = true;

  if (candidate.check.inlier_indices.size() >=
      static_cast<std::size_t>(params.min_inliers)) {
    const std::vector<TrackedMatch> inlier_matches =
        selectMatches(matches, candidate.check.inlier_indices);

    const Eigen::Isometry3f refined =
        solvePICPPose(inlier_matches, camera, solved, kernel, params);

    PoseCheck refined_check =
        checkPose(matches, camera, refined, kernel, params, acceptance_ratio);

    Candidate refined_candidate;
    refined_candidate.valid = isValidPose(refined);
    refined_candidate.pose = refined;
    refined_candidate.check = refined_check;
    refined_candidate.kernel = kernel;

    if (betterCandidate(refined_candidate, candidate)) {
      candidate = refined_candidate;
    }
  }

  return candidate;
}

Candidate runKernel(
    const std::vector<TrackedMatch>& matches,
    const Isometry3fVector& guesses,
    const CameraData& camera,
    float kernel,
    const VOPipelineParams& params,
    float acceptance_ratio) {
  Candidate best;

  for (const Eigen::Isometry3f& guess : guesses) {
    const Candidate candidate =
        runCandidate(matches, camera, guess, kernel, params, acceptance_ratio);

    if (betterCandidate(candidate, best)) {
      best = candidate;
    }
  }

  return best;
}

Isometry3fVector oneGuess(const Eigen::Isometry3f& T) {
  Isometry3fVector guesses;
  guesses.push_back(T);
  return guesses;
}

std::vector<TrackedMatch> buildTrustedMatches(
    const Frame& frame,
    const std::unordered_map<int, Landmark>& landmarks) {
  std::vector<TrackedMatch> matches;
  matches.reserve(frame.observations.size());

  std::unordered_set<int> used_ids;

  for (const Observation& obs : frame.observations) {
    if (obs.landmark_id < 0) {
      continue;
    }

    if (used_ids.count(obs.landmark_id)) {
      continue;
    }

    const auto it = landmarks.find(obs.landmark_id);

    if (it == landmarks.end()) {
      continue;
    }

    const Landmark& landmark = it->second;

    if (!landmark.initialized || !landmark.p_world.allFinite()) {
      continue;
    }

    TrackedMatch match;
    match.landmark_id = obs.landmark_id;
    match.p_world = landmark.p_world;
    match.uv = obs.uv;

    matches.push_back(match);
    used_ids.insert(obs.landmark_id);
  }

  return matches;
}

Eigen::Isometry3f scaledMotion(
    const Eigen::Isometry3f& motion,
    float scale,
    bool use_translation,
    bool use_rotation) {
  Eigen::Isometry3f result = Eigen::Isometry3f::Identity();

  if (use_translation) {
    result.translation() = scale * motion.translation();
  }

  if (use_rotation) {
    Eigen::Quaternionf q_motion(motion.linear());
    q_motion.normalize();

    const Eigen::Quaternionf q_identity = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf q_scaled = q_identity.slerp(scale, q_motion);
    q_scaled.normalize();

    result.linear() = q_scaled.toRotationMatrix();
  }

  return result;
}

void appendUniqueObservation(
    std::vector<std::pair<int, Eigen::Vector2f>>& observations,
    int seq,
    const Eigen::Vector2f& uv) {
  for (const auto& item : observations) {
    if (item.first == seq && (item.second - uv).squaredNorm() < 1e-8f) {
      return;
    }
  }

  observations.push_back({seq, uv});
}

} 

VisualOdometryPipeline::VisualOdometryPipeline(const VOPipelineParams& params)
    : params_(params) {}

VOPipelineResult VisualOdometryPipeline::run(
    const VisualOdometryDataset& dataset) {
  reset();

  VOPipelineResult result;

  camera_ = dataset.camera;

  if (dataset.frames.size() < 2) {
    result.success = false;
    result.message = "VO failed: need at least two frames.";
    return result;
  }

  if (!initialize(dataset.frames, result)) {
    fillResult(result);
    result.success = false;
    return result;
  }

  for (std::size_t i = 2; i < dataset.frames.size(); ++i) {
    VOPipelineFrameLog log;

    if (!processFrame(dataset.frames[i], log)) {
      result.frame_logs.push_back(log);
      fillResult(result);
      result.success = false;

      std::ostringstream oss;
      oss << "VO failed at frame " << dataset.frames[i].seq
          << ": " << log.status;

      result.message = oss.str();
      return result;
    }

    result.frame_logs.push_back(log);
  }

  fillResult(result);
  result.success = true;

  std::ostringstream oss;
  oss << "VO done. poses=" << result.poses.size()
      << " map=" << result.landmarks.size();

  result.message = oss.str();

  return result;
}

void VisualOdometryPipeline::reset() {
  camera_ = CameraData();

  landmarks_.clear();
  observation_history_.clear();

  poses_.clear();
  pose_sequences_.clear();
  pose_index_by_seq_.clear();
}

bool VisualOdometryPipeline::initialize(
    const std::vector<Frame>& frames,
    VOPipelineResult& result) {
  /*
  Initializes the first two frames with epipolar geometry.
  Frame 0 is the world frame.
  */
  const Frame& frame0 = frames[0];
  const Frame& frame1 = frames[1];

  const EpipolarInitResult init = initializeEpipolar(
      frame0,
      frame1,
      camera_.K,
      params_.initial_translation_scale,
      params_.init_max_reprojection_error_px,
      params_.init_max_ray_gap);

  if (!init.success) {
    result.message = init.message;
    return false;
  }

  landmarks_ = init.landmarks;

  addPose(frame0.seq, init.T_wc0);
  addPose(frame1.seq, init.T_wc1);

  addObservations(frame0);
  addObservations(frame1);

  VOPipelineFrameLog log0;
  log0.seq = frame0.seq;
  log0.image_landmarks = static_cast<int>(frame0.observations.size());
  log0.map_landmarks = countInitializedMapPoints(landmarks_);
  log0.map_projectable =
      countProjectableMapPoints(landmarks_, camera_, init.T_wc0);
  log0.pose_valid = true;
  log0.map_updated = true;
  log0.status = "init";

  VOPipelineFrameLog log1;
  log1.seq = frame1.seq;
  log1.image_landmarks = static_cast<int>(frame1.observations.size());
  log1.map_landmarks = countInitializedMapPoints(landmarks_);
  log1.map_projectable =
      countProjectableMapPoints(landmarks_, camera_, init.T_wc1);
  log1.pose_valid = true;
  log1.map_updated = true;
  log1.new_points = static_cast<int>(landmarks_.size());
  log1.status = "init";

  result.frame_logs.push_back(log0);
  result.frame_logs.push_back(log1);

  return true;
}

bool VisualOdometryPipeline::processFrame(
    const Frame& frame,
    VOPipelineFrameLog& log) {
  log.seq = frame.seq;
  log.image_landmarks = static_cast<int>(frame.observations.size());
  log.map_landmarks = countInitializedMapPoints(landmarks_);

  const std::vector<TrackedMatch> matches =
      buildTrustedMatches(frame, landmarks_);

  log.known = static_cast<int>(matches.size());

  const Isometry3fVector guesses = makeInitialGuesses();

  if (matches.size() < static_cast<std::size_t>(params_.min_known_matches)) {
    const Eigen::Isometry3f predicted =
        guesses.empty() ? Eigen::Isometry3f::Identity() : guesses.front();

    addPose(frame.seq, predicted);

    log.map_projectable =
        countProjectableMapPoints(landmarks_, camera_, predicted);

    addObservations(frame);

    log.map_landmarks = countInitializedMapPoints(landmarks_);
    log.pose_valid = true;
    log.continuity_only = true;
    log.status = "few-known";
    return true;
  }

  Candidate best;

  Candidate k1000 =
      runKernel(matches, guesses, camera_, 1000.0f,
                params_, params_.acceptance_ratio);

  if (k1000.valid && k1000.check.accepted) {
    best = k1000;

    Candidate k200 =
        runKernel(matches, oneGuess(best.pose), camera_, 200.0f,
                  params_, params_.acceptance_ratio);

    if (k200.valid && k200.check.accepted) {
      best = k200;

      Candidate k30 =
          runKernel(matches, oneGuess(best.pose), camera_, 30.0f,
                    params_, params_.acceptance_ratio);

      if (k30.valid && k30.check.accepted) {
        best = k30;

        Candidate k10 =
            runKernel(matches, oneGuess(best.pose), camera_, 10.0f,
                      params_, params_.acceptance_ratio);

        if (k10.valid && k10.check.accepted) {
          best = k10;
        }
      } else {
        Candidate k50 =
            runKernel(matches, oneGuess(best.pose), camera_, 50.0f,
                      params_, params_.acceptance_ratio);

        if (k50.valid && k50.check.accepted) {
          best = k50;
        }
      }
    } else {
      Candidate k500 =
          runKernel(matches, oneGuess(best.pose), camera_, 500.0f,
                    params_, params_.acceptance_ratio);

      if (k500.valid && k500.check.accepted) {
        best = k500;

        Candidate k350 =
            runKernel(matches, oneGuess(best.pose), camera_, 350.0f,
                      params_, params_.acceptance_ratio);

        if (k350.valid && k350.check.accepted) {
          best = k350;
        }
      } else {
        Candidate k800 =
            runKernel(matches, oneGuess(best.pose), camera_, 800.0f,
                      params_, params_.acceptance_ratio);

        if (k800.valid && k800.check.accepted) {
          best = k800;
        }
      }
    }
  } else {
    Candidate k2000 =
        runKernel(matches, guesses, camera_, 2000.0f,
                  params_, params_.acceptance_ratio);

    if (k2000.valid && k2000.check.accepted) {
      best = k2000;

      Candidate k1500 =
          runKernel(matches, oneGuess(best.pose), camera_, 1500.0f,
                    params_, params_.acceptance_ratio);

      if (k1500.valid && k1500.check.accepted) {
        best = k1500;

        Candidate k1250 =
            runKernel(matches, oneGuess(best.pose), camera_, 1250.0f,
                      params_, params_.acceptance_ratio);

        if (k1250.valid && k1250.check.accepted) {
          best = k1250;
        }
      }
    } else {
      const float high_kernels[] = {3000.0f, 10000.0f, 20000.0f, 40000.0f};

      for (float kernel : high_kernels) {
        Candidate high =
            runKernel(matches, guesses, camera_, kernel,
                      params_, params_.acceptance_ratio);

        if (high.valid && high.check.accepted) {
          best = high;
          break;
        }
      }
    }

    if (!best.valid || !best.check.accepted) {
      const float rescue_kernels[] = {2000.0f, 3000.0f, 10000.0f, 20000.0f, 40000.0f};

      for (float kernel : rescue_kernels) {
        Candidate rescue =
            runKernel(matches, guesses, camera_, kernel,
                      params_, params_.loose_acceptance_ratio);

        if (rescue.valid && rescue.check.accepted) {
          best = rescue;
          break;
        }
      }
    }

    if (best.valid && best.check.accepted && best.kernel > 1000.0f) {
      Candidate k1000_retry =
          runKernel(matches, oneGuess(best.pose), camera_, 1000.0f,
                    params_, params_.acceptance_ratio);

      if (k1000_retry.valid && k1000_retry.check.accepted) {
        best = k1000_retry;

        Candidate k800 =
            runKernel(matches, oneGuess(best.pose), camera_, 800.0f,
                      params_, params_.acceptance_ratio);

        if (k800.valid && k800.check.accepted) {
          best = k800;

          Candidate k500 =
              runKernel(matches, oneGuess(best.pose), camera_, 500.0f,
                        params_, params_.acceptance_ratio);

          if (k500.valid && k500.check.accepted) {
            best = k500;

            Candidate k350 =
                runKernel(matches, oneGuess(best.pose), camera_, 350.0f,
                          params_, params_.acceptance_ratio);

            if (k350.valid && k350.check.accepted) {
              best = k350;
            }
          }
        }
      }
    }
  }

  if (!best.valid || !best.check.accepted) {
    log.map_landmarks = countInitializedMapPoints(landmarks_);
    log.pose_valid = false;
    log.status = "picp-failed";
    return false;
  }

  addPose(frame.seq, best.pose);

  log.pose_valid = true;
  log.kernel = best.kernel;
  log.projectable = best.check.projectable;
  log.required_projectable = best.check.required_projectable;
  log.inliers = best.check.inliers;
  log.mean_error_px = best.check.mean_error_px;
  log.median_error_px = best.check.median_error_px;
  log.inlier_projectable_ratio = best.check.inlier_projectable_ratio;
  log.inlier_known_ratio = best.check.inlier_known_ratio;

  log.map_projectable =
      countProjectableMapPoints(landmarks_, camera_, best.pose);

  const bool can_update_map =
      best.check.accepted &&
      best.kernel <= 350.0f;

  if (can_update_map) {
    log.new_points = addNewPoints(frame, best.pose, &log);
    log.map_updated = true;
    log.status = "ok";
  } else {
    log.map_updated = false;
    log.continuity_only = true;
    log.status = "continuity";
  }

  addObservations(frame);

  log.map_landmarks = countInitializedMapPoints(landmarks_);

  return true;
}

void VisualOdometryPipeline::fillResult(
    VOPipelineResult& result) const {
  result.poses = poses_;
  result.pose_sequences = pose_sequences_;
  result.landmarks = landmarks_;
}

void VisualOdometryPipeline::addPose(
    int seq,
    const Eigen::Isometry3f& T_wc) {
  const auto it = pose_index_by_seq_.find(seq);

  if (it != pose_index_by_seq_.end()) {
    poses_[it->second] = T_wc;
    return;
  }

  pose_index_by_seq_[seq] = poses_.size();
  pose_sequences_.push_back(seq);
  poses_.push_back(T_wc);
}

bool VisualOdometryPipeline::poseAt(
    int seq,
    Eigen::Isometry3f& T_wc) const {
  const auto it = pose_index_by_seq_.find(seq);

  if (it == pose_index_by_seq_.end()) {
    return false;
  }

  T_wc = poses_[it->second];
  return true;
}

void VisualOdometryPipeline::addObservations(
    const Frame& frame) {
  /*
  Stores observation history for later triangulation.
  */
  for (const Observation& obs : frame.observations) {
    if (obs.landmark_id < 0) {
      continue;
    }

    observation_history_[obs.landmark_id].push_back({frame.seq, obs.uv});

    auto landmark_it = landmarks_.find(obs.landmark_id);

    if (landmark_it != landmarks_.end() && landmark_it->second.initialized) {
      appendUniqueObservation(
          landmark_it->second.observations,
          frame.seq,
          obs.uv);
    }
  }
}

int VisualOdometryPipeline::addNewPoints(
    const Frame& frame,
    const Eigen::Isometry3f& T_wc,
    VOPipelineFrameLog* log) {
  /*
  Grows the map after accepted PICP.
  */
  int added = 0;

  for (const Observation& obs : frame.observations) {
    if (obs.landmark_id < 0) {
      continue;
    }

    const auto existing = landmarks_.find(obs.landmark_id);

    if (existing != landmarks_.end() && existing->second.initialized) {
      continue;
    }

    const auto history_it = observation_history_.find(obs.landmark_id);

    if (history_it == observation_history_.end()) {
      continue;
    }

    if (history_it->second.empty()) {
      continue;
    }

    if (log) {
      ++log->new_candidates;
    }

    bool found = false;
    Landmark best_landmark;
    float best_score = std::numeric_limits<float>::infinity();

    for (auto hist_it = history_it->second.rbegin();
         hist_it != history_it->second.rend();
         ++hist_it) {
      const auto& old_obs = *hist_it;

      if (old_obs.first == frame.seq) {
        continue;
      }

      Eigen::Isometry3f old_pose;

      if (!poseAt(old_obs.first, old_pose)) {
        continue;
      }

      const float baseline = cameraBaselineWorld(old_pose, T_wc);

      if (!std::isfinite(baseline) || baseline < kTriangulationMinBaseline) {
        continue;
      }

      if (log) {
        ++log->tri_attempts;
      }

      const TriangulationResult tri =
          triangulateTwoViews(
              camera_.K,
              old_pose,
              old_obs.second,
              T_wc,
              obs.uv);

      if (tri.success && log) {
        ++log->tri_success;
      }

      if (!triangulationGapAccepted(tri, params_)) {
        continue;
      }

      const Eigen::Vector3f p_old_camera = old_pose * tri.p_world;
      const Eigen::Vector3f p_new_camera = T_wc * tri.p_world;

      if (!p_old_camera.allFinite() || !p_new_camera.allFinite()) {
        continue;
      }

      if (p_old_camera.z() <= 1e-6f || p_new_camera.z() <= 1e-6f) {
        continue;
      }

      const float err_old =
          reprojectionErrorPx(camera_, old_pose, tri.p_world, old_obs.second);

      const float err_new =
          reprojectionErrorPx(camera_, T_wc, tri.p_world, obs.uv);

      if (!std::isfinite(err_old) || !std::isfinite(err_new)) {
        continue;
      }

      const float mean_err = 0.5f * (err_old + err_new);

      const float max_reprojection_error =
          params_.triangulation_max_reprojection_error_px;

      if (mean_err > max_reprojection_error) {
        continue;
      }

      const float mean_depth = 0.5f * (tri.depth_a + tri.depth_b);

      if (!std::isfinite(mean_depth) || mean_depth <= 1e-6f) {
        continue;
      }

      const float normalized_gap = tri.ray_gap / mean_depth;
      const float score = mean_err + 10.0f * normalized_gap;

      if (score < best_score) {
        best_score = score;

        best_landmark = Landmark();
        best_landmark.id = obs.landmark_id;
        best_landmark.p_world = tri.p_world;
        best_landmark.initialized = true;

        appendUniqueObservation(
            best_landmark.observations,
            old_obs.first,
            old_obs.second);

        appendUniqueObservation(
            best_landmark.observations,
            frame.seq,
            obs.uv);

        found = true;
      }
    }

    if (!found) {
      continue;
    }

    landmarks_[obs.landmark_id] = best_landmark;
    ++added;

    if (log) {
      ++log->tri_accepted;
    }
  }

  return added;
}

Isometry3fVector VisualOdometryPipeline::makeInitialGuesses() const {
  /*
  Creates deterministic PICP guesses.
  The full constant motion guess is first.
  */
  Isometry3fVector guesses;

  if (poses_.empty()) {
    guesses.push_back(Eigen::Isometry3f::Identity());
    return guesses;
  }

  const Eigen::Isometry3f last = poses_.back();

  if (poses_.size() < 2) {
    guesses.push_back(last);
    return guesses;
  }

  const Eigen::Isometry3f prev = poses_[poses_.size() - 2];
  const Eigen::Isometry3f motion = last * prev.inverse();

  const Eigen::Isometry3f full = motion * last;
  const Eigen::Isometry3f translation_only =
      scaledMotion(motion, 1.0f, true, false) * last;
  const Eigen::Isometry3f rotation_only =
      scaledMotion(motion, 1.0f, false, true) * last;
  const Eigen::Isometry3f damped =
      scaledMotion(motion, 0.5f, true, true) * last;
  const Eigen::Isometry3f faster =
      scaledMotion(motion, 1.5f, true, true) * last;

  guesses.push_back(full);
  guesses.push_back(translation_only);
  guesses.push_back(rotation_only);
  guesses.push_back(last);
  guesses.push_back(damped);
  guesses.push_back(faster);

  return guesses;
}