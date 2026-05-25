/*
version 0.1
author: Ömer (Semih) İnce

This file is part of the Visual Odometry project,
covers the test of the epipolar initialization step.

Structure of the code is as follows,
1. Load two frames.
   - Either create a small synthetic example.
   - Or read the first two measurement files from the dataset.
2. Count common landmark IDs between the two frames.
3. Call initializeEpipolar().
4. Print the estimated pose of camera 1 with respect to camera 0.
5. Print the first triangulated 3D landmarks.

To run:
  ./test_initialization --synthetic

Or with dataset:
  ./test_initialization --dataset /path/to/dataset

In WSL, a Windows folder can be used like:
  ./test_initialization --dataset "/mnt/c/Users/YOUR_WINDOWS_USER/Downloads/vo_dataset"
*/

#include "epipolar_initializer.h"
#include "vo_types.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// -----------------------------------------------------------------------------
// Printing helpers
// -----------------------------------------------------------------------------

void printUsage(const char* executable_name) {
  std::cout
      << "Usage:\n"
      << "  " << executable_name << " --synthetic\n"
      << "  " << executable_name << " --dataset /path/to/dataset\n\n"
      << "Examples:\n"
      << "  " << executable_name << " --synthetic\n"
      << "  " << executable_name << " --dataset /home/ince/datasets/vo_dataset\n"
      << "  " << executable_name << " --dataset \"/mnt/c/Users/YOUR_WINDOWS_USER/Downloads/vo_dataset\"\n";
}

void printMatrix3(const Eigen::Matrix3f& M, const std::string& name) {
  std::cout << name << " =\n" << M << "\n";
}

void printMatrix4(const Eigen::Isometry3f& T, const std::string& name) {
  std::cout << name << " =\n" << T.matrix() << "\n";
}

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

float rotationAngleDeg(const Eigen::Matrix3f& R_est, const Eigen::Matrix3f& R_gt) {
  const Eigen::Matrix3f R_error = R_est * R_gt.transpose();

  float c = 0.5f * (R_error.trace() - 1.0f);
  c = std::max(-1.0f, std::min(1.0f, c));

  return std::acos(c) * 180.0f / kPi;
}

float directionAngleDeg(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
  if (a.norm() <= 1e-9f || b.norm() <= 1e-9f) {
    return std::numeric_limits<float>::infinity();
  }

  float c = a.normalized().dot(b.normalized());
  c = std::max(-1.0f, std::min(1.0f, c));

  return std::acos(c) * 180.0f / kPi;
}

int countCommonIds(const Frame& frame0, const Frame& frame1) {
  std::unordered_set<int> ids0;

  for (const Observation& obs : frame0.observations) {
    if (obs.landmark_id >= 0) {
      ids0.insert(obs.landmark_id);
    }
  }

  int common = 0;

  for (const Observation& obs : frame1.observations) {
    if (obs.landmark_id >= 0 && ids0.count(obs.landmark_id) > 0) {
      ++common;
    }
  }

  return common;
}

// -----------------------------------------------------------------------------
// Reading camera.dat
// -----------------------------------------------------------------------------

bool readWholeFile(const fs::path& path, std::string& text) {
  std::ifstream is(path);

  if (!is) {
    std::cerr << "Could not open file: " << path << "\n";
    return false;
  }

  std::ostringstream ss;
  ss << is.rdbuf();
  text = ss.str();

  return true;
}

std::vector<float> extractNumbersFromText(const std::string& text) {
  std::vector<float> values;

  const std::regex number_regex(
      R"([-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?)");

  auto begin = std::sregex_iterator(text.begin(), text.end(), number_regex);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    values.push_back(std::stof(it->str()));
  }

  return values;
}

bool loadCameraMatrix(const fs::path& camera_path, Eigen::Matrix3f& K) {
  std::string text;

  if (!readWholeFile(camera_path, text)) {
    return false;
  }

  const std::vector<float> values = extractNumbersFromText(text);

  if (values.size() < 9) {
    std::cerr << "camera.dat should contain at least 9 numbers for K.\n";
    std::cerr << "Found only " << values.size() << " numbers.\n";
    return false;
  }

  K << values[0], values[1], values[2],
       values[3], values[4], values[5],
       values[6], values[7], values[8];

  if (!K.allFinite() || std::abs(K.determinant()) <= 1e-9f) {
    std::cerr << "Invalid camera matrix read from: " << camera_path << "\n";
    printMatrix3(K, "K");
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
// Reading meas-XXXX.dat
// -----------------------------------------------------------------------------

bool lineStartsWithPoint(const std::string& line) {
  std::istringstream iss(line);

  std::string first_word;
  iss >> first_word;

  return first_word == "point";
}

bool loadMeasurementFrame(const fs::path& measurement_path, int seq, Frame& frame) {
  std::ifstream is(measurement_path);

  if (!is) {
    std::cerr << "Could not open measurement file: " << measurement_path << "\n";
    return false;
  }

  frame = Frame();
  frame.seq = seq;
  frame.pose_valid = false;
  frame.T_wc = Eigen::Isometry3f::Identity();

  std::string line;
  int point_lines = 0;

  while (std::getline(is, line)) {
    if (!lineStartsWithPoint(line)) {
      continue;
    }

    ++point_lines;

    std::istringstream iss(line);

    std::string tag;
    int local_id = -1;
    int actual_id = -1;
    float u = 0.0f;
    float v = 0.0f;

    iss >> tag >> local_id >> actual_id >> u >> v;

    if (!iss) {
      std::cerr << "Could not parse this point line:\n";
      std::cerr << line << "\n";
      continue;
    }

    (void)local_id;

    Observation obs;
    obs.landmark_id = actual_id;
    obs.uv = Eigen::Vector2f(u, v);

    if (obs.landmark_id < 0 || !obs.uv.allFinite()) {
      continue;
    }

    frame.observations.push_back(obs);
  }

  if (frame.observations.empty()) {
    std::cerr << "No valid observations were read from: " << measurement_path << "\n";
    std::cerr << "Point lines found: " << point_lines << "\n";
    return false;
  }

  std::cout << "Loaded " << frame.observations.size()
            << " observations from " << measurement_path.filename().string() << "\n";

  return true;
}

std::vector<fs::path> findMeasurementFiles(const fs::path& dataset_path) {
  std::vector<fs::path> files;

  if (!fs::exists(dataset_path)) {
    std::cerr << "Dataset path does not exist: " << dataset_path << "\n";
    return files;
  }

  if (!fs::is_directory(dataset_path)) {
    std::cerr << "Dataset path is not a directory: " << dataset_path << "\n";
    return files;
  }

  for (const fs::directory_entry& entry : fs::directory_iterator(dataset_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    const std::string name = entry.path().filename().string();

    if (name.rfind("meas-", 0) == 0 && entry.path().extension() == ".dat") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());

  return files;
}

// -----------------------------------------------------------------------------
// Synthetic data
// -----------------------------------------------------------------------------

bool projectPoint(
    const Eigen::Matrix3f& K,
    const Eigen::Isometry3f& T_wc,
    const Eigen::Vector3f& p_world,
    Eigen::Vector2f& uv) {
  const Eigen::Vector3f p_camera = T_wc * p_world;

  if (!p_camera.allFinite() || p_camera.z() <= 1e-6f) {
    return false;
  }

  const Eigen::Vector3f z = K * p_camera;

  if (!z.allFinite() || std::abs(z.z()) <= 1e-9f) {
    return false;
  }

  uv.x() = z.x() / z.z();
  uv.y() = z.y() / z.z();

  return uv.allFinite();
}

struct SyntheticData {
  Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
  Frame frame0;
  Frame frame1;
  Eigen::Isometry3f T_wc1_gt = Eigen::Isometry3f::Identity();
};

SyntheticData createSyntheticData() {
  SyntheticData data;

  data.K << 500.0f, 0.0f,   320.0f,
            0.0f,   500.0f, 240.0f,
            0.0f,   0.0f,   1.0f;

  data.frame0.seq = 0;
  data.frame1.seq = 1;

  data.frame0.T_wc = Eigen::Isometry3f::Identity();
  data.frame1.T_wc = Eigen::Isometry3f::Identity();

  data.frame0.pose_valid = true;
  data.frame1.pose_valid = true;

  Eigen::AngleAxisf rot_y(5.0f * kPi / 180.0f, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_x(-2.0f * kPi / 180.0f, Eigen::Vector3f::UnitX());

  data.T_wc1_gt = Eigen::Isometry3f::Identity();
  data.T_wc1_gt.linear() = (rot_y * rot_x).toRotationMatrix();

  // T_wc convention:
  // p_camera = R_wc * p_world + t_wc
  //
  // The initializer can recover only the direction of t.
  data.T_wc1_gt.translation() = Eigen::Vector3f(0.40f, 0.02f, 0.10f);

  int id = 0;

  for (int ix = -4; ix <= 4; ++ix) {
    for (int iy = -3; iy <= 3; ++iy) {
      const float x = static_cast<float>(ix);
      const float y = static_cast<float>(iy);

      Eigen::Vector3f p_world;
      p_world.x() = 0.35f * x;
      p_world.y() = 0.30f * y;

      // Non-planar depth. This avoids a planar degeneracy.
      p_world.z() = 4.0f
                    + 0.07f * x * x
                    + 0.05f * y * y
                    + 0.03f * x * y
                    + 0.11f * static_cast<float>((ix + 2 * iy) % 3);

      Eigen::Vector2f uv0;
      Eigen::Vector2f uv1;

      if (!projectPoint(data.K, Eigen::Isometry3f::Identity(), p_world, uv0)) {
        continue;
      }

      if (!projectPoint(data.K, data.T_wc1_gt, p_world, uv1)) {
        continue;
      }

      Observation obs0;
      obs0.landmark_id = id;
      obs0.uv = uv0;

      Observation obs1;
      obs1.landmark_id = id;
      obs1.uv = uv1;

      data.frame0.observations.push_back(obs0);
      data.frame1.observations.push_back(obs1);

      ++id;
    }
  }

  return data;
}

// -----------------------------------------------------------------------------
// Main test
// -----------------------------------------------------------------------------

bool runInitializationTest(
    const Eigen::Matrix3f& K,
    const Frame& frame0,
    const Frame& frame1,
    bool has_gt,
    const Eigen::Isometry3f& T_wc1_gt) {
  std::cout << "\n================ Initialization Test ================\n";

  printMatrix3(K, "K");

  std::cout << "Frame 0 observations: " << frame0.observations.size() << "\n";
  std::cout << "Frame 1 observations: " << frame1.observations.size() << "\n";

  const int common_ids = countCommonIds(frame0, frame1);
  std::cout << "Common landmark IDs:  " << common_ids << "\n";

  if (common_ids < 8) {
    std::cerr << "Initialization needs at least 8 common landmarks.\n";
    return false;
  }

  const float initial_translation_scale = 1.0f;
  const float max_reprojection_error_px = 5.0f;
  const float max_ray_gap = 0.25f;

  const EpipolarInitResult result = initializeEpipolar(
      frame0,
      frame1,
      K,
      initial_translation_scale,
      max_reprojection_error_px,
      max_ray_gap);

  std::cout << "\nMessage:\n";
  std::cout << result.message << "\n";

  if (!result.success) {
    std::cerr << "\nINITIALIZATION FAILED.\n";
    return false;
  }

  std::cout << "\nINITIALIZATION SUCCEEDED.\n";

  printMatrix4(result.T_wc0, "Estimated T_wc0");
  printMatrix4(result.T_wc1, "Estimated T_wc1");

  const Eigen::Matrix3f R = result.T_wc1.linear();
  const Eigen::Vector3f t = result.T_wc1.translation();

  std::cout << "\nBasic checks:\n";
  std::cout << "  det(R):              " << R.determinant() << "\n";
  std::cout << "  ||R^T R - I||:       "
            << (R.transpose() * R - Eigen::Matrix3f::Identity()).norm() << "\n";
  std::cout << "  ||t||:               " << t.norm() << "\n";
  std::cout << "  matches:             " << result.num_matches << "\n";
  std::cout << "  triangulated points: " << result.num_valid_triangulated << "\n";

  if (has_gt) {
    std::cout << "\nSynthetic comparison:\n";

    const float rot_error =
        rotationAngleDeg(result.T_wc1.linear(), T_wc1_gt.linear());

    const float t_dir_error =
        directionAngleDeg(result.T_wc1.translation(), T_wc1_gt.translation());

    std::cout << "  rotation error [deg]:              " << rot_error << "\n";
    std::cout << "  translation direction error [deg]: " << t_dir_error << "\n";
    std::cout << "  Translation length is not compared, because monocular VO has arbitrary scale.\n";
  }

  std::cout << "\nFirst triangulated points:\n";

  int printed = 0;

  for (const auto& item : result.landmarks) {
    const Landmark& lm = item.second;

    std::cout << "  id " << std::setw(4) << lm.id
              << "  p_world = ["
              << std::setw(10) << lm.p_world.x() << ", "
              << std::setw(10) << lm.p_world.y() << ", "
              << std::setw(10) << lm.p_world.z() << "]\n";

    ++printed;

    if (printed >= 10) {
      break;
    }
  }

  std::cout << "=====================================================\n";

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::fixed << std::setprecision(6);

  bool use_synthetic = false;
  fs::path dataset_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--synthetic") {
      use_synthetic = true;
    } else if (arg == "--dataset") {
      if (i + 1 >= argc) {
        std::cerr << "--dataset needs a path.\n";
        printUsage(argv[0]);
        return 1;
      }

      dataset_path = fs::path(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  if (use_synthetic) {
    std::cout << "Running synthetic test.\n";

    const SyntheticData data = createSyntheticData();

    const bool ok = runInitializationTest(
        data.K,
        data.frame0,
        data.frame1,
        true,
        data.T_wc1_gt);

    return ok ? 0 : 1;
  }

  if (dataset_path.empty()) {
    std::cerr << "No dataset path was given.\n\n";
    printUsage(argv[0]);
    return 1;
  }

  std::cout << "Running dataset test.\n";
  std::cout << "Dataset path: " << dataset_path << "\n";

  Eigen::Matrix3f K = Eigen::Matrix3f::Identity();

  const fs::path camera_path = dataset_path / "camera.dat";

  if (!loadCameraMatrix(camera_path, K)) {
    return 1;
  }

  const std::vector<fs::path> measurement_files = findMeasurementFiles(dataset_path);

  if (measurement_files.size() < 2) {
    std::cerr << "Need at least two meas-XXXX.dat files.\n";
    std::cerr << "Found: " << measurement_files.size() << "\n";
    return 1;
  }

  std::cout << "Using:\n";
  std::cout << "  " << measurement_files[0] << "\n";
  std::cout << "  " << measurement_files[1] << "\n";

  Frame frame0;
  Frame frame1;

  if (!loadMeasurementFrame(measurement_files[0], 0, frame0)) {
    return 1;
  }

  if (!loadMeasurementFrame(measurement_files[1], 1, frame1)) {
    return 1;
  }

  const bool ok = runInitializationTest(
      K,
      frame0,
      frame1,
      false,
      Eigen::Isometry3f::Identity());

  return ok ? 0 : 1;
}