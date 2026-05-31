/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
  implements reading of the provided dataset files.

Structure:
  1. Read camera.dat:
       K, optional cam_transform, z_near, z_far, width, height.
  2. Read meas-XXXX.dat:
       sequence number, optional gt/odom planar poses, point observations.
  3. Read world.dat:
       landmark ground truth --for evaluation only--.
  4. Read trajectory.dat / trajectoy.dat:
       trajectory ground truth --for evaluation only--.

At the end the reader will give the necessary output for,

-CameraDatacamera,

    Camera matrix "camera.K"

    Camera visibility limits: A reconstructed landmark is acceptable only if its depth is inside the camera's valid sensing range. 
    So after triangulation or projection, a point should satisfy → $z_{near}<s<z_{far}$ where s is the depth.
         "camera.z_near"
         "camera.z_far"

    The valid image domain, A projected point must lie inside the image so valid coordinates must satisfy → $0 \leq u < width$ & $0\leq v < height$
         "camera.width"
         "camera.height"

    Frame  "std::vector<Frame>frames"

    Observation -- One measured 2D feature in one image

    Actual landmark ID  "frames[0].observations[i].landmark_id"

    Name / Order of the landmark in the current file "frames[0].observations[i].local_id"

    Image coordinate "frames[0].observations[i].uv"  So, image [col, row]

    Appearance vector "frames[0].observations[i].appearance"  So, 10D appearance vector

- For evaluation later,
    "dataset.world_gt"
    "dataset.trajectory_gt"
*/

#include "dataset_io.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Constants
constexpr int kAppearanceSize = 10;
constexpr float kMinCameraMatrixDeterminant = 1e-9f;


// Error helper
bool fail(std::string* error_message, const std::string& message) {
  if (error_message) {
    *error_message = message;
  }
  return false;
}

std::string trim(const std::string& input) {
  const std::size_t begin = input.find_first_not_of(" \t\r\n");

  if (begin == std::string::npos) {
    return "";
  }

  const std::size_t end = input.find_last_not_of(" \t\r\n");
  return input.substr(begin, end - begin + 1);
}

std::string removeComment(const std::string& line) {
  std::size_t cut = std::string::npos;

  const std::size_t hash_pos = line.find('#');
  const std::size_t percent_pos = line.find('%');

  if (hash_pos != std::string::npos) {
    cut = hash_pos;
  }

  if (percent_pos != std::string::npos) {
    cut = std::min(cut, percent_pos);
  }

  if (cut == std::string::npos) {
    return line;
  }

  return line.substr(0, cut);
}

std::string toLower(std::string text) {
  for (char& c : text) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  return text;
}

std::string firstToken(const std::string& line) {
  std::istringstream iss(line);

  std::string token;
  iss >> token;

  while (!token.empty() &&
         (token.back() == ':' ||
          token.back() == ',' ||
          token.back() == ';')) {
    token.pop_back();
  }

  return token;
}

// Number extraction
std::vector<double> extractNumbersFromText(const std::string& text) {
  std::vector<double> values;

  static const std::regex number_regex(
      R"([-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?)");

  auto begin = std::sregex_iterator(text.begin(), text.end(), number_regex);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    try {
      values.push_back(std::stod(it->str()));
    } catch (...) {
      // Ignore malformed numeric text.
    }
  }

  return values;
}

bool readWholeFile(const fs::path& path, std::string& text) {
  std::ifstream is(path);

  if (!is) {
    return false;
  }

  std::ostringstream ss;
  ss << is.rdbuf();

  text = ss.str();
  return true;
}

int roundToInt(double value) {
  return static_cast<int>(std::llround(value));
}

bool isFiniteFloat(float value) {
  return std::isfinite(value);
}

Eigen::Matrix<float, kAppearanceSize, 1> readAppearanceVector(
    const std::vector<double>& values,
    std::size_t start_index,
    bool& has_appearance) {
  Eigen::Matrix<float, kAppearanceSize, 1> appearance;
  appearance.setZero();

  has_appearance = false;

  if (values.size() < start_index + kAppearanceSize) {
    return appearance;
  }

  for (int i = 0; i < kAppearanceSize; ++i) {
    appearance(i) = static_cast<float>(values[start_index + i]);

    if (!isFiniteFloat(appearance(i))) {
      appearance.setZero();
      has_appearance = false;
      return appearance;
    }
  }

  has_appearance = true;
  return appearance;
}

bool tokenLooksLikeSequence(const std::string& token) {
  return token == "seq" ||
         token == "sequence" ||
         token == "frame" ||
         token == "frame_id" ||
         token == "pose_id";
}

bool tokenLooksLikeOdometry(const std::string& token) {
  return token == "odom" ||
         token == "odometry" ||
         token == "odometry_pose" ||
         token == "odom_pose";
}

bool tokenLooksLikeGroundTruth(const std::string& token) {
  return token == "gt" ||
         token == "groundtruth" ||
         token == "ground_truth" ||
         token == "groundtruth_pose" ||
         token == "ground_truth_pose" ||
         token == "gt_pose";
}

bool parsePlanarPoseFromNumbers(
    const std::vector<double>& values,
    Eigen::Vector3f& pose_2d) {
  if (values.size() < 3) {
    return false;
  }

  pose_2d << static_cast<float>(values[0]),
             static_cast<float>(values[1]),
             static_cast<float>(values[2]);

  return pose_2d.allFinite();
}

bool parsePointObservationLine(
    const std::string& line,
    Observation& obs) {
  const std::vector<double> values = extractNumbersFromText(line);

  if (values.size() < 4) {
    return false;
  }

  obs = Observation();

  obs.local_id = roundToInt(values[0]);
  obs.landmark_id = roundToInt(values[1]);

  obs.uv << static_cast<float>(values[2]),
            static_cast<float>(values[3]);

  bool has_appearance = false;
  obs.appearance = readAppearanceVector(values, 4, has_appearance);
  obs.has_appearance = has_appearance;

  if (obs.landmark_id < 0) {
    return false;
  }

  if (!obs.uv.allFinite()) {
    return false;
  }

  return true;
}

bool isMeasurementFilename(const fs::path& path) {
  const std::string name = path.filename().string();

  static const std::regex meas_regex(R"(meas-\d+\.dat)");

  return std::regex_match(name, meas_regex);
}

fs::path findTrajectoryPath(const fs::path& dataset_dir) {
  const fs::path correct_path = dataset_dir / "trajectory.dat";

  if (fs::exists(correct_path)) {
    return correct_path;
  }

  // The user folder listing contains this typo.
  const fs::path typo_path = dataset_dir / "trajectoy.dat";

  if (fs::exists(typo_path)) {
    return typo_path;
  }

  return correct_path;
}

}  // namespace

int extractMeasurementSequenceFromFilename(const fs::path& path) {
  const std::string name = path.filename().string();

  static const std::regex seq_regex(R"(meas-(\d+)\.dat)");

  std::smatch match;

  if (!std::regex_match(name, match, seq_regex)) {
    return -1;
  }

  if (match.size() < 2) {
    return -1;
  }

  try {
    return std::stoi(match[1].str());
  } catch (...) {
    return -1;
  }
}

std::vector<fs::path> findMeasurementFiles(const fs::path& dataset_dir) {
  std::vector<fs::path> files;

  if (!fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
    return files;
  }

  for (const fs::directory_entry& entry : fs::directory_iterator(dataset_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    if (isMeasurementFilename(entry.path())) {
      files.push_back(entry.path());
    }
  }

  std::sort(
      files.begin(),
      files.end(),
      [](const fs::path& a, const fs::path& b) {
        const int seq_a = extractMeasurementSequenceFromFilename(a);
        const int seq_b = extractMeasurementSequenceFromFilename(b);

        if (seq_a >= 0 && seq_b >= 0) {
          return seq_a < seq_b;
        }

        return a.filename().string() < b.filename().string();
      });

  return files;
}

// -----------------------------------------------------------------------------
// camera.dat reader
// -----------------------------------------------------------------------------

bool readCameraDat(
    const fs::path& camera_path,
    CameraData& camera,
    std::string* error_message) {
  camera = CameraData();

  std::string text;

  if (!readWholeFile(camera_path, text)) {
    return fail(
        error_message,
        "Could not open camera file: " + camera_path.string());
  }

  const std::vector<double> values = extractNumbersFromText(text);

  // Minimum:
  // K has 9 numbers.
  // Then we need z_near, z_far, width, height = 4 numbers.
  if (values.size() < 13) {
    std::ostringstream oss;
    oss << "camera.dat should contain at least 13 numeric values: "
        << "9 for K, plus z_near, z_far, width, height. "
        << "Found " << values.size() << " numeric values in "
        << camera_path.string();

    return fail(error_message, oss.str());
  }

  camera.K << static_cast<float>(values[0]),
              static_cast<float>(values[1]),
              static_cast<float>(values[2]),
              static_cast<float>(values[3]),
              static_cast<float>(values[4]),
              static_cast<float>(values[5]),
              static_cast<float>(values[6]),
              static_cast<float>(values[7]),
              static_cast<float>(values[8]);

  if (!camera.K.allFinite() ||
      std::abs(camera.K.determinant()) <= kMinCameraMatrixDeterminant) {
    std::ostringstream oss;
    oss << "Invalid camera matrix K in " << camera_path.string();

    return fail(error_message, oss.str());
  }

  // If the file has enough numbers for a 4x4 cam_transform after K, read it.
  // We ignore it for VO, but keeping it is useful for debugging.
  if (values.size() >= 29) {
    Eigen::Matrix4f cam_transform_matrix = Eigen::Matrix4f::Identity();

    int index = 9;

    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        cam_transform_matrix(r, c) = static_cast<float>(values[index]);
        ++index;
      }
    }

    if (cam_transform_matrix.allFinite()) {
      camera.cam_transform = Eigen::Isometry3f(cam_transform_matrix);
      camera.has_cam_transform = true;
    }
  }

  const std::size_t n = values.size();

  camera.z_near = static_cast<float>(values[n - 4]);
  camera.z_far = static_cast<float>(values[n - 3]);
  camera.width = roundToInt(values[n - 2]);
  camera.height = roundToInt(values[n - 1]);

  if (!std::isfinite(camera.z_near) ||
      !std::isfinite(camera.z_far) ||
      camera.z_near < 0.0f ||
      camera.z_far <= camera.z_near) {
    std::ostringstream oss;
    oss << "Invalid z_near/z_far in " << camera_path.string()
        << ". z_near=" << camera.z_near
        << ", z_far=" << camera.z_far;

    return fail(error_message, oss.str());
  }

  if (camera.width <= 0 || camera.height <= 0) {
    std::ostringstream oss;
    oss << "Invalid image width/height in " << camera_path.string()
        << ". width=" << camera.width
        << ", height=" << camera.height;

    return fail(error_message, oss.str());
  }

  return true;
}

// -----------------------------------------------------------------------------
// meas-XXXX.dat reader
// -----------------------------------------------------------------------------

bool readMeasurementDat(
    const fs::path& measurement_path,
    Frame& frame,
    int sequence_override,
    std::string* error_message) {
  frame = Frame();

  if (sequence_override >= 0) {
    frame.seq = sequence_override;
  } else {
    frame.seq = extractMeasurementSequenceFromFilename(measurement_path);
  }

  frame.T_wc = Eigen::Isometry3f::Identity();
  frame.pose_valid = false;

  std::ifstream is(measurement_path);

  if (!is) {
    return fail(
        error_message,
        "Could not open measurement file: " + measurement_path.string());
  }

  std::string line;

  int line_number = 0;
  int valid_points = 0;
  int malformed_point_lines = 0;

  int anonymous_header_line_index = 0;

  while (std::getline(is, line)) {
    ++line_number;

    const std::string clean_line = trim(removeComment(line));

    if (clean_line.empty()) {
      continue;
    }

    const std::string token = toLower(firstToken(clean_line));

    if (token == "point") {
      Observation obs;

      if (!parsePointObservationLine(clean_line, obs)) {
        ++malformed_point_lines;
        continue;
      }

      frame.observations.push_back(obs);
      ++valid_points;
      continue;
    }

    const std::vector<double> values = extractNumbersFromText(clean_line);

    if (values.empty()) {
      continue;
    }

    if (tokenLooksLikeSequence(token)) {
      frame.seq = roundToInt(values[0]);
      continue;
    }

    if (tokenLooksLikeGroundTruth(token)) {
      if (parsePlanarPoseFromNumbers(values, frame.groundtruth_pose_2d)) {
        frame.has_groundtruth_pose = true;
      }
      continue;
    }

    if (tokenLooksLikeOdometry(token)) {
      if (parsePlanarPoseFromNumbers(values, frame.odometry_pose_2d)) {
        frame.has_odometry_pose = true;
      }
      continue;
    }

    if (anonymous_header_line_index == 0 && values.size() == 1) {
      frame.seq = roundToInt(values[0]);
      ++anonymous_header_line_index;
      continue;
    }

    if (anonymous_header_line_index == 1 && values.size() >= 3) {
      if (parsePlanarPoseFromNumbers(values, frame.groundtruth_pose_2d)) {
        frame.has_groundtruth_pose = true;
      }

      ++anonymous_header_line_index;
      continue;
    }

    if (anonymous_header_line_index == 2 && values.size() >= 3) {
      if (parsePlanarPoseFromNumbers(values, frame.odometry_pose_2d)) {
        frame.has_odometry_pose = true;
      }

      ++anonymous_header_line_index;
      continue;
    }
  }

  if (frame.seq < 0) {
    std::ostringstream oss;
    oss << "Could not determine sequence number for "
        << measurement_path.string();

    return fail(error_message, oss.str());
  }

  if (frame.observations.empty()) {
    std::ostringstream oss;
    oss << "No valid point observations were read from "
        << measurement_path.string()
        << ". Malformed point lines: " << malformed_point_lines;

    return fail(error_message, oss.str());
  }

  return true;
}

bool readAllMeasurementFrames(
    const fs::path& dataset_dir,
    std::vector<Frame>& frames,
    std::string* error_message) {
  frames.clear();

  const std::vector<fs::path> measurement_files =
      findMeasurementFiles(dataset_dir);

  if (measurement_files.empty()) {
    return fail(
        error_message,
        "No meas-XXXX.dat files found in dataset directory: " +
            dataset_dir.string());
  }

  frames.reserve(measurement_files.size());

  for (const fs::path& measurement_path : measurement_files) {
    const int seq = extractMeasurementSequenceFromFilename(measurement_path);

    Frame frame;

    std::string local_error;

    if (!readMeasurementDat(measurement_path, frame, seq, &local_error)) {
      std::ostringstream oss;
      oss << "Failed while reading measurement file "
          << measurement_path.string()
          << ". Reason: " << local_error;

      return fail(error_message, oss.str());
    }

    frames.push_back(frame);
  }

  return true;
}

// -----------------------------------------------------------------------------
// world.dat reader
// -----------------------------------------------------------------------------

bool readWorldDat(
    const fs::path& world_path,
    std::unordered_map<int, WorldGroundTruthPoint>& world_gt,
    std::string* error_message) {
  world_gt.clear();

  std::ifstream is(world_path);

  if (!is) {
    return fail(
        error_message,
        "Could not open world file: " + world_path.string());
  }

  std::string line;
  int line_number = 0;
  int valid_points = 0;

  while (std::getline(is, line)) {
    ++line_number;

    const std::string clean_line = trim(removeComment(line));

    if (clean_line.empty()) {
      continue;
    }

    const std::vector<double> values = extractNumbersFromText(clean_line);

    if (values.size() < 4) {
      continue;
    }

    WorldGroundTruthPoint point;

    point.landmark_id = roundToInt(values[0]);

    point.position << static_cast<float>(values[1]),
                      static_cast<float>(values[2]),
                      static_cast<float>(values[3]);

    bool has_appearance = false;
    point.appearance = readAppearanceVector(values, 4, has_appearance);
    point.has_appearance = has_appearance;

    if (point.landmark_id < 0 || !point.position.allFinite()) {
      continue;
    }

    world_gt[point.landmark_id] = point;
    ++valid_points;
  }

  if (world_gt.empty()) {
    std::ostringstream oss;
    oss << "No valid ground-truth landmarks were read from "
        << world_path.string();

    return fail(error_message, oss.str());
  }

  return true;
}

// -----------------------------------------------------------------------------
// trajectory.dat / trajectoy.dat reader
// -----------------------------------------------------------------------------

bool readTrajectoryDat(
    const fs::path& trajectory_path,
    std::vector<TrajectoryGroundTruthPose>& trajectory_gt,
    std::string* error_message) {
  trajectory_gt.clear();

  std::ifstream is(trajectory_path);

  if (!is) {
    return fail(
        error_message,
        "Could not open trajectory file: " + trajectory_path.string());
  }

  std::string line;
  int line_number = 0;

  while (std::getline(is, line)) {
    ++line_number;

    const std::string clean_line = trim(removeComment(line));

    if (clean_line.empty()) {
      continue;
    }

    const std::vector<double> values = extractNumbersFromText(clean_line);

    // Expected:
    // POSE_ID odom_x odom_y odom_theta gt_x gt_y gt_theta
    if (values.size() < 7) {
      continue;
    }

    TrajectoryGroundTruthPose pose;

    pose.pose_id = roundToInt(values[0]);

    pose.odometry_pose_2d << static_cast<float>(values[1]),
                             static_cast<float>(values[2]),
                             static_cast<float>(values[3]);

    pose.groundtruth_pose_2d << static_cast<float>(values[4]),
                                static_cast<float>(values[5]),
                                static_cast<float>(values[6]);

    if (pose.pose_id < 0) {
      continue;
    }

    if (!pose.odometry_pose_2d.allFinite() ||
        !pose.groundtruth_pose_2d.allFinite()) {
      continue;
    }

    trajectory_gt.push_back(pose);
  }

  if (trajectory_gt.empty()) {
    std::ostringstream oss;
    oss << "No valid ground-truth trajectory poses were read from "
        << trajectory_path.string();

    return fail(error_message, oss.str());
  }

  std::sort(
      trajectory_gt.begin(),
      trajectory_gt.end(),
      [](const TrajectoryGroundTruthPose& a,
         const TrajectoryGroundTruthPose& b) {
        return a.pose_id < b.pose_id;
      });

  return true;
}

// -----------------------------------------------------------------------------
// Full dataset reader
// -----------------------------------------------------------------------------

bool readVisualOdometryDataset(
    const fs::path& dataset_dir,
    VisualOdometryDataset& dataset,
    bool load_ground_truth,
    std::string* error_message) {
  dataset = VisualOdometryDataset();

  if (!fs::exists(dataset_dir)) {
    return fail(
        error_message,
        "Dataset directory does not exist: " + dataset_dir.string());
  }

  if (!fs::is_directory(dataset_dir)) {
    return fail(
        error_message,
        "Dataset path is not a directory: " + dataset_dir.string());
  }

  const fs::path camera_path = dataset_dir / "camera.dat";

  std::string local_error;

  if (!readCameraDat(camera_path, dataset.camera, &local_error)) {
    return fail(error_message, local_error);
  }

  if (!readAllMeasurementFrames(dataset_dir, dataset.frames, &local_error)) {
    return fail(error_message, local_error);
  }

  if (!load_ground_truth) {
    return true;
  }

  const fs::path world_path = dataset_dir / "world.dat";

  if (!readWorldDat(world_path, dataset.world_gt, &local_error)) {
    return fail(error_message, local_error);
  }

  const fs::path trajectory_path = findTrajectoryPath(dataset_dir);

  if (!readTrajectoryDat(trajectory_path, dataset.trajectory_gt, &local_error)) {
    return fail(error_message, local_error);
  }

  return true;
}

// -----------------------------------------------------------------------------
// Planar pose conversion for later evaluation
// -----------------------------------------------------------------------------

Eigen::Isometry3f planarPoseToIsometry3f(const Eigen::Vector3f& pose_2d) {
  const float x = pose_2d.x();
  const float y = pose_2d.y();
  const float theta = pose_2d.z();

  const float c = std::cos(theta);
  const float s = std::sin(theta);

  Eigen::Isometry3f T = Eigen::Isometry3f::Identity();

  T.linear() << c, -s, 0.0f,
                s,  c, 0.0f,
              0.0f, 0.0f, 1.0f;

  T.translation() << x, y, 0.0f;

  return T;
}