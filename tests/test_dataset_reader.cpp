/*
version 0.1
author: Ömer (Semih) İnce

Small test for checking whether the dataset reader works.
Note that, This does not run visual odometry. It only loads the dataset and prints basic information.
*/

#include "dataset_io.h"

#include <Eigen/Core>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;

namespace {

void printUsage(const char* exe_name) {
  std::cout << "Usage:\n";
  std::cout << "  " << exe_name << " --dataset /path/to/data\n\n";
  std::cout << "Example from build folder:\n";
  std::cout << "  " << exe_name << " --dataset ../data\n";
}

int countCommonIds(const Frame& a, const Frame& b) {
  std::unordered_set<int> ids_a;

  for (const Observation& obs : a.observations) {
    if (obs.landmark_id >= 0) {
      ids_a.insert(obs.landmark_id);
    }
  }

  int common = 0;

  for (const Observation& obs : b.observations) {
    if (obs.landmark_id >= 0 && ids_a.count(obs.landmark_id) > 0) {
      ++common;
    }
  }

  return common;
}

void printFirstObservation(const Frame& frame) {
  if (frame.observations.empty()) {
    std::cout << "  no observations\n";
    return;
  }

  const Observation& obs = frame.observations.front();

  std::cout << "  first observation:\n";
  std::cout << "    local id:    " << obs.local_id << "\n";
  std::cout << "    actual id:   " << obs.landmark_id << "\n";
  std::cout << "    image point: " << obs.uv.transpose() << "\n";
  std::cout << "    appearance:  ";

  if (!obs.has_appearance) {
    std::cout << "not available\n";
    return;
  }

  std::cout << obs.appearance.transpose() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::cout << std::fixed << std::setprecision(6);

  fs::path dataset_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg == "--dataset") {
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

  if (dataset_path.empty()) {
    std::cerr << "No dataset path was given.\n";
    printUsage(argv[0]);
    return 1;
  }

  VisualOdometryDataset dataset;
  std::string error_message;

  const bool ok = readVisualOdometryDataset(
      dataset_path,
      dataset,
      true,
      &error_message);

  if (!ok) {
    std::cerr << "Dataset loading failed.\n";
    std::cerr << error_message << "\n";
    return 1;
  }

  std::cout << "\nDataset loaded correctly.\n\n";

  std::cout << "Camera:\n";
  std::cout << "  K:\n" << dataset.camera.K << "\n";
  std::cout << "  z_near: " << dataset.camera.z_near << "\n";
  std::cout << "  z_far:  " << dataset.camera.z_far << "\n";
  std::cout << "  width:  " << dataset.camera.width << "\n";
  std::cout << "  height: " << dataset.camera.height << "\n\n";

  std::cout << "Measurements:\n";
  std::cout << "  number of frames: " << dataset.frames.size() << "\n";

  if (!dataset.frames.empty()) {
    const Frame& first = dataset.frames.front();
    const Frame& last = dataset.frames.back();

    std::cout << "  first frame seq: " << first.seq << "\n";
    std::cout << "  last frame seq:  " << last.seq << "\n";
    std::cout << "  observations in first frame: "
              << first.observations.size() << "\n";
    std::cout << "  observations in last frame:  "
              << last.observations.size() << "\n";

    printFirstObservation(first);
  }

  if (dataset.frames.size() >= 2) {
    const int common = countCommonIds(dataset.frames[0], dataset.frames[1]);

    std::cout << "\nFirst two frames:\n";
    std::cout << "  common actual landmark ids: " << common << "\n";

    if (common < 8) {
      std::cout << "  warning: epipolar initialization needs at least 8 common ids\n";
    }
  }

  std::cout << "\nGround truth files:\n";
  std::cout << "  world gt landmarks: " << dataset.world_gt.size() << "\n";
  std::cout << "  trajectory gt poses: " << dataset.trajectory_gt.size() << "\n";

  if (!dataset.trajectory_gt.empty()) {
    const TrajectoryGroundTruthPose& pose = dataset.trajectory_gt.front();

    std::cout << "  first trajectory pose id: " << pose.pose_id << "\n";
    std::cout << "  first odometry pose:      "
              << pose.odometry_pose_2d.transpose() << "\n";
    std::cout << "  first groundtruth pose:   "
              << pose.groundtruth_pose_2d.transpose() << "\n";
  }

  std::cout << "\nReader test finished.\n";

  return 0;
}