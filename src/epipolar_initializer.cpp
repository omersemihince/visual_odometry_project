/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  covers the epipolar geometry initialization step. 
  utilizes form 8point algorithm. 

Notice,
  the file heavily based on the Prof.Grisetti's (and his lab's) slide probabilistic_robotics_25_essentials_in_computer_vision.pdf and the related MATLAB code, projective_geometry.
  theoric summary can be found in the readme file, the code structure is same as the summary.

Structure of the code is as follows,
  1. Recieve one image sequence from the dataset and extract feautures. 
  2. Find id correspondences between the two images. Give error if the number of correspondences is less than 8.
  3. For each corespondence, recover the bearing vector by undoing the camera matrix and normalizing the vector.
  4. Construct the A matrix for the 8-point algorithm, and relatedly construct the H matrix.
  5. Solve for the essential matrix E, by finding the eigenvector corresponding to the smallest eigenvalue of H.
  6. Find the possible rotation and translation pairs by decomposing the essential matrix E using SVD.
  7. Find the correct pair by using triangulation.cpp
  8. Set the scale of the translation from the choosen pair.
  9. Return the rotation, translation. And return the 3D points using triangulation.cpp.
*/

#include "epipolar_initializer.h"
#include "vo_types.h"
#include "triangulation.h" // we will call it to get the 3D point

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace {

// -----------------------------------------------------------------------------
// Parameter structures
// -----------------------------------------------------------------------------

struct InitMatch {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int landmark_id = -1; //Landmark with correspondence.

  Eigen::Vector2f uv0 = Eigen::Vector2f::Zero(); //Pixel coordinates in the first image.
  Eigen::Vector2f uv1 = Eigen::Vector2f::Zero(); //Pixel coordinates in the second image.

  Eigen::Vector3f bearing0 = Eigen::Vector3f::Zero(); //d recovered from uv0 undoing K.
  Eigen::Vector3f bearing1 = Eigen::Vector3f::Zero(); //d recovered from uv1 undoing K.
};

//for candidate pose evaluation and selection.
struct PoseCandidateScore { 
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  bool evaluated = false;//Whether this candidate was evaluated.

  Eigen::Isometry3f T_wc1 = Eigen::Isometry3f::Identity();

  //Correct R,t should put points in front of both cameras.
  int num_positive_depth = 0; //Number of triangulated landmarks with positive depth in both cameras.
  int num_accepted = 0;//Number of triangulated landmarks that passed the reprojection error and ray gap checks.

  //quality measures
  float mean_reprojection_error_px = std::numeric_limits<float>::infinity();
  float mean_ray_gap = std::numeric_limits<float>::infinity();

  //initial map created by the current candidate
  std::unordered_map<int, Landmark> landmarks;
};

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

//testing mechanism for validation of triangulated points through depth.
bool isFinitePositive(float value) {
  return std::isfinite(value) && value > 0.0f;
}

//computing mechanism for the bearing vector from pixel coordinates, by undoing the camera matrix and normalizing the vector.
Eigen::Vector3f bearingFromPixel(
    const Eigen::Matrix3f& K_inv,
    const Eigen::Vector2f& uv) {
  Eigen::Vector3f z;
  z << uv.x(), uv.y(), 1.0f;

  Eigen::Vector3f bearing = K_inv * z;

  const float n = bearing.norm();

  //safety check for zero or near-zero norm, which would indicate an invalid bearing vector.
  if (n <= 1e-9f || !std::isfinite(n)) {
    return Eigen::Vector3f::Zero();
  }

  bearing /= n; //normalization
  return bearing;
}

//computing mechanism for projecting a 3D point into the image plane.
bool projectPoint(
    const Eigen::Matrix3f& K,
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world,
    Eigen::Vector2f& uv_projected) {
  const Eigen::Vector3f p_camera = T_wc * p_world;

  if (!p_camera.allFinite()) {
    return false;
  }

  if (p_camera.z() <= 1e-6f) {
    return false;
  }

  const Eigen::Vector3f z = K * p_camera;

  if (!z.allFinite()) {
    return false;
  }

  //Avoid division by zero or near-zero.
  if (std::abs(z.z()) <= 1e-9f) {
    return false;
  }

  uv_projected.x() = z.x() / z.z();
  uv_projected.y() = z.y() / z.z();

  return uv_projected.allFinite();
}

//error = projected point - measured point
float reprojectionErrorPx(
    const Eigen::Matrix3f& K,
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world,
    const Eigen::Vector2f& uv_measured) {
  Eigen::Vector2f uv_projected;

  if (!projectPoint(K, T_wc, p_world, uv_projected)) {
    return std::numeric_limits<float>::infinity();
  }

  return (uv_projected - uv_measured).norm();
}

// -----------------------------------------------------------------------------
// Debug/Diagnosis helpers
// -----------------------------------------------------------------------------

constexpr bool kVerboseEpipolarDebug = true;

float computeEpipolarResidualRms(
    const std::vector<InitMatch>& matches,
    const Eigen::Matrix3f& E) {
  if (matches.empty()) {
    return std::numeric_limits<float>::infinity();
  }

  double sum_sq = 0.0;

  for (const InitMatch& match : matches) {
    const float r = match.bearing1.transpose() * E * match.bearing0;
    sum_sq += static_cast<double>(r) * static_cast<double>(r);
  }

  return static_cast<float>(
      std::sqrt(sum_sq / static_cast<double>(matches.size())));
}

float computeEpipolarResidualMaxAbs(
    const std::vector<InitMatch>& matches,
    const Eigen::Matrix3f& E) {
  float max_abs = 0.0f;

  for (const InitMatch& match : matches) {
    const float r = match.bearing1.transpose() * E * match.bearing0;
    max_abs = std::max(max_abs, std::abs(r));
  }

  return max_abs;
}

//mechanism for printing the candidate pose and related scores for diagnoses.
void printCandidateDebug(
    int candidate_index,
    const PoseCandidateScore& score) {
  std::cerr << "[epipolar debug] candidate " << candidate_index << "\n";
  std::cerr << "  det(R):              "
            << score.T_wc1.linear().determinant() << "\n";
  std::cerr << "  ||t||:               "
            << score.T_wc1.translation().norm() << "\n";
  std::cerr << "  t:                   "
            << score.T_wc1.translation().transpose() << "\n";
  std::cerr << "  positive depth:      "
            << score.num_positive_depth << "\n";
  std::cerr << "  accepted landmarks:  "
            << score.num_accepted << "\n";
  std::cerr << "  mean reprojection:   "
            << score.mean_reprojection_error_px << " px\n";
  std::cerr << "  mean ray gap:        "
            << score.mean_ray_gap << "\n";
}

//mechanism for comparing two pose candidates.
bool isBetterCandidate(
    const PoseCandidateScore& candidate,
    const PoseCandidateScore& best) {
  if (!candidate.evaluated) {
    return false;
  }

  if (!best.evaluated) {
    return true;
  }

  if (candidate.num_accepted != best.num_accepted) {
    return candidate.num_accepted > best.num_accepted;
  }

  if (candidate.num_positive_depth != best.num_positive_depth) {
    return candidate.num_positive_depth > best.num_positive_depth;
  }

  if (candidate.mean_reprojection_error_px != best.mean_reprojection_error_px) {
    return candidate.mean_reprojection_error_px < best.mean_reprojection_error_px;
  }

  return candidate.mean_ray_gap < best.mean_ray_gap;
}

// -----------------------------------------------------------------------------
// ID correspondences
// -----------------------------------------------------------------------------

std::vector<InitMatch> findIdCorrespondences(
    const Frame& frame0,
    const Frame& frame1,
    const Eigen::Matrix3f& K_inv) {
  std::unordered_map<int, Eigen::Vector2f> observations0_by_id;

  //Loop over image 0 observations  
  for (const Observation& obs0 : frame0.observations) {
    if (obs0.landmark_id < 0) {
      continue;
    }

    // If there is a duplicate ID in the same frame, keep the first one.
    observations0_by_id.emplace(obs0.landmark_id, obs0.uv);
  }

  //Allocate memory
  std::vector<InitMatch> matches; 
  matches.reserve(std::min(frame0.observations.size(), frame1.observations.size()));

  //Loop over image 1 observations, and find matches with image 0 based on landmark ID.
  for (const Observation& obs1 : frame1.observations) {
    if (obs1.landmark_id < 0) {
      continue;
    }

    const auto it = observations0_by_id.find(obs1.landmark_id);
    if (it == observations0_by_id.end()) {
      continue;
    }

    //Crearing one correspondence
    InitMatch match;
    match.landmark_id = obs1.landmark_id;
    match.uv0 = it->second;//Pixel coordinates in the zeroth image.
    match.uv1 = obs1.uv;//Pixel coordinates in the first image.

    match.bearing0 = bearingFromPixel(K_inv, match.uv0);
    match.bearing1 = bearingFromPixel(K_inv, match.uv1);

    //Safety checks for invalid bearing vectors.
    if (match.bearing0.squaredNorm() <= 1e-12f) {
      continue;
    }

    if (match.bearing1.squaredNorm() <= 1e-12f) {
      continue;
    }

    matches.push_back(match);//adding the valid match to the list of matches
  }

  return matches;
}

// -----------------------------------------------------------------------------
// 8-point algorithm
// -----------------------------------------------------------------------------

bool estimateEssentialMatrix(
    const std::vector<InitMatch>& matches,
    Eigen::Matrix3f& E,
    std::string& message) {
  if (matches.size() < 8) {
    std::ostringstream oss;
    oss << "Epipolar initialization failed: need at least 8 correspondences, got "
        << matches.size() << ".";
    message = oss.str();
    return false;
  }

  //Constructing the H matrix for the 8-point algorithm.
  Eigen::Matrix<float, 9, 9> H = Eigen::Matrix<float, 9, 9>::Zero();

  //Loop over all matches to fill the H matrix.
  for (const InitMatch& match : matches) {
    //d0
    const float x0 = match.bearing0.x();
    const float y0 = match.bearing0.y();
    const float z0 = match.bearing0.z();
    //d1
    const float x1 = match.bearing1.x();
    const float y1 = match.bearing1.y();
    const float z1 = match.bearing1.z();

    Eigen::Matrix<float, 9, 1> a;
    a << x1 * x0,
         x1 * y0,
         x1 * z0,
         y1 * x0,
         y1 * y0,
         y1 * z0,
         z1 * x0,
         z1 * y0,
         z1 * z0;

    H.noalias() += a * a.transpose();//noalias is used to tell Eigen that there is no aliasing in the operation, as little robustification.
  }

  //Solve for e, the vectorized form of E.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 9, 9>> eig(H);

  if (eig.info() != Eigen::Success) {
    message = "Epipolar initialization failed: eigen decomposition of H failed.";
    return false;
  }

  const Eigen::Matrix<float, 9, 1> e = eig.eigenvectors().col(0);

  //Reshape e into the 3x3 essential matrix E.
  E << e(0), e(1), e(2),
       e(3), e(4), e(5),
       e(6), e(7), e(8);

  if (!E.allFinite() || E.norm() <= 1e-9f) {
    message = "Epipolar initialization failed: estimated essential matrix is invalid.";
    return false;
  }

  E /= E.norm();

  return true;
}

//Improving E against noise making sigma3 \sim 0 -> sigma3 = 0 and sigma1 \sim sigma2 -> sigma1 = sigma2 using SVD.
bool enforceEssentialConstraint(
    const Eigen::Matrix3f& E_raw,
    Eigen::Matrix3f& E_corrected,
    std::string& message) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      E_raw,
      Eigen::ComputeFullU | Eigen::ComputeFullV);

  if (svd.matrixU().determinant() == 0.0f ||
      svd.matrixV().determinant() == 0.0f) {
    message = "Epipolar initialization failed: invalid SVD while correcting E.";
    return false;
  }

  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  if (U.determinant() < 0.0f) {
    U.col(2) *= -1.0f;
  }

  if (V.determinant() < 0.0f) {
    V.col(2) *= -1.0f;
  }

  const Eigen::Vector3f singular_values = svd.singularValues();//get sigma1, sigma2, sigma3
  const float sigma = 0.5f * (singular_values.x() + singular_values.y());

  //Enforce the essential matrix constraint by setting the singular values to (sigma, sigma, 0), taeking the avarage of sigma1 and sigma2.
  Eigen::Matrix3f S = Eigen::Matrix3f::Zero();
  S(0, 0) = sigma;
  S(1, 1) = sigma;
  S(2, 2) = 0.0f;

  E_corrected = U * S * V.transpose();//Rebuild with SVD

  if (!E_corrected.allFinite() || E_corrected.norm() <= 1e-9f) {
    message = "Epipolar initialization failed: corrected essential matrix is invalid.";
    return false;
  }

  E_corrected /= E_corrected.norm();

  return true;
}

// -----------------------------------------------------------------------------
// Dcompose E into possible R,t pairs
// -----------------------------------------------------------------------------

//Making the Rotation matrix valid.
Eigen::Matrix3f projectToSO3(const Eigen::Matrix3f& R_raw) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      R_raw,
      Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  Eigen::Matrix3f R = U * V.transpose();

  if (R.determinant() < 0.0f) {
    U.col(2) *= -1.0f;
    R = U * V.transpose();
  }

  return R;
}

//Recover 4 possible R,t pairs from E.
std::vector<Eigen::Isometry3f> decomposeEssentialMatrix(
    const Eigen::Matrix3f& E,
    float translation_scale) {
  std::vector<Eigen::Isometry3f> candidates;
  candidates.reserve(4);

  //Set the scale
  if (translation_scale <= 0.0f || !std::isfinite(translation_scale)) {
    translation_scale = 1.0f;
  }

  Eigen::JacobiSVD<Eigen::Matrix3f> svd(
      E,
      Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();

  if (U.determinant() < 0.0f) {
    U.col(2) *= -1.0f;
  }

  if (V.determinant() < 0.0f) {
    V.col(2) *= -1.0f;
  }

  //The W matrix used in the decomposition of E to get the possible rotations.
  Eigen::Matrix3f W;
  W << 0.0f, -1.0f, 0.0f,
       1.0f,  0.0f, 0.0f,
       0.0f,  0.0f, 1.0f;

  Eigen::Matrix3f R1 = projectToSO3(U * W * V.transpose());
  Eigen::Matrix3f R2 = projectToSO3(U * W.transpose() * V.transpose());

  Eigen::Vector3f t = U.col(2);//only the direction of t can be recovered, not the scale.
  const float t_norm = t.norm();

  if (t_norm <= 1e-9f || !std::isfinite(t_norm)) {
    return candidates;
  }

  t = translation_scale * t / t_norm;//Set the scale of the translation, and normalize it.

  const std::vector<Eigen::Matrix3f> rotations = {R1, R2};
  const std::vector<Eigen::Vector3f> translations = {t, -t};

  //Combine the rotations and translations to form the 4 possible T_wc1 candidates.
  for (const Eigen::Matrix3f& R : rotations) {
    for (const Eigen::Vector3f& tr : translations) {
      Eigen::Isometry3f T_wc1 = Eigen::Isometry3f::Identity();
      T_wc1.linear() = R;
      T_wc1.translation() = tr;

      if (T_wc1.matrix().allFinite()) {
        candidates.push_back(T_wc1);
      }
    }
  }

  return candidates;
}

// -----------------------------------------------------------------------------
// Evaluate a pose candidate through triangulation
// -----------------------------------------------------------------------------

PoseCandidateScore evaluatePoseCandidate(
    const Eigen::Matrix3f& K,
    const Frame& frame0,
    const Frame& frame1,
    const std::vector<InitMatch>& matches,
    const Eigen::Isometry3f& T_wc1,
    float max_reprojection_error_px,
    float max_ray_gap) {
  PoseCandidateScore score;
  score.evaluated = true;
  score.T_wc1 = T_wc1;

  const Eigen::Isometry3f T_wc0 = Eigen::Isometry3f::Identity();//Set fraem 0 as the world frame, so T_wc0 is identity.

  const int seq0 = frame0.seq >= 0 ? frame0.seq : 0;
  const int seq1 = frame1.seq >= 0 ? frame1.seq : 1;

  float reprojection_error_sum = 0.0f;
  float ray_gap_sum = 0.0f;

  int reprojection_error_count = 0;
  int ray_gap_count = 0;

  for (const InitMatch& match : matches) {
    const TriangulationResult tri = triangulateTwoViews(
        K,
        T_wc0,
        match.uv0,
        T_wc1,
        match.uv1);

    if (!tri.success) {
      continue;
    }

    if (!tri.p_world.allFinite()) {
      continue;
    }

    //Skip point behind the camera
    if (!isFinitePositive(tri.depth_a) || !isFinitePositive(tri.depth_b)) {
      continue;
    }

    score.num_positive_depth++;

    const float err0 = reprojectionErrorPx(K, T_wc0, tri.p_world, match.uv0);
    const float err1 = reprojectionErrorPx(K, T_wc1, tri.p_world, match.uv1);

    if (!std::isfinite(err0) || !std::isfinite(err1)) {
      continue;
    }

    const float mean_err = 0.5f * (err0 + err1);

    reprojection_error_sum += mean_err;
    reprojection_error_count++;

    if (std::isfinite(tri.ray_gap)) {
      ray_gap_sum += tri.ray_gap;
      ray_gap_count++;
    }

    const bool reprojection_ok = mean_err <= max_reprojection_error_px;
    const bool ray_gap_ok = std::isfinite(tri.ray_gap) && tri.ray_gap <= max_ray_gap;

    if (!reprojection_ok || !ray_gap_ok) {
      continue;
    }

    //Create a landmark for the successfully triangulated point, and add it to the map.
    Landmark landmark;
    landmark.id = match.landmark_id;
    landmark.p_world = tri.p_world;
    landmark.initialized = true;

    landmark.observations.push_back({seq0, match.uv0});
    landmark.observations.push_back({seq1, match.uv1});

    score.landmarks.emplace(match.landmark_id, landmark);
    score.num_accepted++;
  }

  if (reprojection_error_count > 0) {
    score.mean_reprojection_error_px =
        reprojection_error_sum / static_cast<float>(reprojection_error_count);
  }

  if (ray_gap_count > 0) {
    score.mean_ray_gap =
        ray_gap_sum / static_cast<float>(ray_gap_count);
  }

  return score;
}

}  // namespace

EpipolarInitResult initializeEpipolar(
    const Frame& frame0,
    const Frame& frame1,
    const Eigen::Matrix3f& K,
    float initial_translation_scale,
    float max_reprojection_error_px,
    float max_ray_gap) {
  EpipolarInitResult result;

  result.success = false;
  result.T_wc0 = Eigen::Isometry3f::Identity();
  result.T_wc1 = Eigen::Isometry3f::Identity();

  //Is K valid
  if (!K.allFinite()) {
    result.message = "Epipolar initialization failed: camera matrix K contains invalid values.";
    return result;
  }

  if (std::abs(K.determinant()) <= 1e-9f) {
    result.message = "Epipolar initialization failed: camera matrix K is singular.";
    return result;
  }
  //------- 

  if (initial_translation_scale <= 0.0f || !std::isfinite(initial_translation_scale)) {
    initial_translation_scale = 1.0f;
  }

  //theresholds
  if (max_reprojection_error_px <= 0.0f || !std::isfinite(max_reprojection_error_px)) {
    max_reprojection_error_px = 5.0f;
  }

  if (max_ray_gap <= 0.0f || !std::isfinite(max_ray_gap)) {
    max_ray_gap = 0.25f;
  }
  //-------
  const Eigen::Matrix3f K_inv = K.inverse();

  //landmark correspondences + normalized bearings.
  const std::vector<InitMatch> matches =
      findIdCorrespondences(frame0, frame1, K_inv);

  result.num_matches = static_cast<int>(matches.size());

  if (matches.size() < 8) {
    std::ostringstream oss;
    oss << "Epipolar initialization failed: need at least 8 ID correspondences, got "
        << matches.size() << ".";
    result.message = oss.str();
    return result;
  }

  //estimate E.
  Eigen::Matrix3f E_raw = Eigen::Matrix3f::Zero();
  std::string message;

  if (!estimateEssentialMatrix(matches, E_raw, message)) {
    result.message = message;
    return result;
  }

  //enforce the essential matrix constraint.
  Eigen::Matrix3f E = Eigen::Matrix3f::Zero();

  if (!enforceEssentialConstraint(E_raw, E, message)) {
    result.message = message;
    return result;
  }

  //Debug information for checking whether the essential matrix itself is good.
  if (kVerboseEpipolarDebug) {
    std::cerr << "\n[epipolar debug] Essential matrix diagnostics\n";
    std::cerr << "  raw E residual RMS:        "
              << computeEpipolarResidualRms(matches, E_raw) << "\n";
    std::cerr << "  raw E residual max abs:    "
              << computeEpipolarResidualMaxAbs(matches, E_raw) << "\n";
    std::cerr << "  corrected E residual RMS:  "
              << computeEpipolarResidualRms(matches, E) << "\n";
    std::cerr << "  corrected E residual max:  "
              << computeEpipolarResidualMaxAbs(matches, E) << "\n";
    std::cerr << "  E corrected:\n" << E << "\n";
  }

  //decompose E into four possible transformation matrix
  const std::vector<Eigen::Isometry3f> pose_candidates =
      decomposeEssentialMatrix(E, initial_translation_scale);

  if (pose_candidates.empty()) {
    result.message = "Epipolar initialization failed: essential matrix decomposition produced no valid pose candidates.";
    return result;
  }

  //choose best candidate by triangulation.
  PoseCandidateScore best_score;

  int candidate_index = 0;

  for (const Eigen::Isometry3f& T_wc1_candidate : pose_candidates) {
    const PoseCandidateScore candidate_score = evaluatePoseCandidate(
        K,
        frame0,
        frame1,
        matches,
        T_wc1_candidate,
        max_reprojection_error_px,
        max_ray_gap);

    if (kVerboseEpipolarDebug) {
      printCandidateDebug(candidate_index, candidate_score);
    }

    if (isBetterCandidate(candidate_score, best_score)) {
      best_score = candidate_score;
    }

    ++candidate_index;
  }

  if (!best_score.evaluated) {
    result.message = "Epipolar initialization failed: no pose candidate could be evaluated.";
    return result;
  }

  //final results from the best candidate
  result.T_wc0 = Eigen::Isometry3f::Identity();
  result.T_wc1 = best_score.T_wc1;
  result.landmarks = best_score.landmarks;
  result.num_valid_triangulated = best_score.num_accepted;

  if (result.num_valid_triangulated < 8) {
    std::ostringstream oss;
    oss << "Epipolar initialization failed: only "
        << result.num_valid_triangulated
        << " valid triangulated landmarks after cheirality/reprojection checks. "
        << "Matches: " << result.num_matches
        << ", positive-depth triangulations: " << best_score.num_positive_depth
        << ", mean reprojection error: " << best_score.mean_reprojection_error_px
        << " px, mean ray gap: " << best_score.mean_ray_gap << ".";
    result.message = oss.str();
    return result;
  }

  result.success = true;

  std::ostringstream oss;
  oss << "Epipolar initialization succeeded. "
      << "Matches: " << result.num_matches
      << ", valid triangulated landmarks: " << result.num_valid_triangulated
      << ", positive-depth triangulations: " << best_score.num_positive_depth
      << ", mean reprojection error: " << best_score.mean_reprojection_error_px
      << " px, mean ray gap: " << best_score.mean_ray_gap
      << ", imposed translation scale: " << initial_translation_scale << ".";

  result.message = oss.str();

  return result;
}

//Creating shorter calls in the form (frame0, frame1, K) with default parameters for the thresholds and scale, which will call the main function with the default values.
EpipolarInitResult initializeEpipolar(
    const Frame& frame0,
    const Frame& frame1,
    const Eigen::Matrix3f& K) {
  constexpr float kInitialTranslationScale = 1.0f;
  constexpr float kMaxReprojectionErrorPx = 5.0f;
  constexpr float kMaxRayGap = 0.25f;

  return initializeEpipolar(
      frame0,
      frame1,
      K,
      kInitialTranslationScale,
      kMaxReprojectionErrorPx,
      kMaxRayGap);
}