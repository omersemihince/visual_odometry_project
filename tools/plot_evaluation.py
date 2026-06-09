"""
version 0.1
author: Ömer (Semih) İnce

Plotting utility for the Visual Odometry project.

Reads the evaluation CSV files and creates report figures:
  1. Trajectory top view: estimate vs GT
  2. World top view: GT landmarks + estimated landmarks + GT/estimated poses
  3. World 3D view: GT landmarks + estimated landmarks + GT/estimated trajectories
  4. Rotation error plot
  5. Translation scale-ratio plot

Expected input files:
  output/evaluation/trajectory_estimate.csv
  output/evaluation/trajectory_compare.csv
  output/evaluation/pose_errors.csv
  output/evaluation/map_errors.csv
"""

import csv
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    """
    Reads one CSV file into a list of dictionaries.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    """
    Converts one CSV field to float.
    """
    return float(row[key])


def to_int(row, key):
    """
    Converts one CSV field to int.
    """
    return int(row[key])


def ensure_dir(path):
    """
    Creates the output directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def compute_rigid_alignment(source_points, target_points):
    """
    Computes the rigid alignment R,t such that:
      target ~= R * source + t
    Uses the Kabsch method.
    """
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)

    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = target_mean - R @ source_mean
    return R, t


def apply_rigid_alignment(points, R, t):
    """
    Applies a rigid alignment to a set of 3D points.
    """
    return (R @ points.T).T + t


def quaternion_to_yaw(qx, qy, qz, qw):
    """
    Extracts yaw from quaternion.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def load_map_data(map_errors_path):
    """
    Loads map points and computes the rigid alignment from scaled estimate to GT.
    """
    rows = read_csv(map_errors_path)

    ids = []
    est_scaled = []
    gt = []

    for row in rows:
        ids.append(to_int(row, "landmark_id"))
        est_scaled.append([
            to_float(row, "scaled_x"),
            to_float(row, "scaled_y"),
            to_float(row, "scaled_z"),
        ])
        gt.append([
            to_float(row, "gt_first_x"),
            to_float(row, "gt_first_y"),
            to_float(row, "gt_first_z"),
        ])

    est_scaled = np.asarray(est_scaled, dtype=float)
    gt = np.asarray(gt, dtype=float)

    R, t = compute_rigid_alignment(est_scaled, gt)
    est_aligned = apply_rigid_alignment(est_scaled, R, t)

    return {
        "ids": np.asarray(ids, dtype=int),
        "gt": gt,
        "est_scaled": est_scaled,
        "est_aligned": est_aligned,
        "R": R,
        "t": t,
    }


def load_trajectory_data(trajectory_estimate_path, trajectory_compare_path, R, t):
    """
    Loads trajectory data and aligns the estimated scaled trajectory to the GT frame.
    """
    est_rows = read_csv(trajectory_estimate_path)
    cmp_rows = read_csv(trajectory_compare_path)

    est_by_seq = {}
    for row in est_rows:
        seq = to_int(row, "seq")
        est_by_seq[seq] = row

    seqs = []
    gt_positions = []
    est_positions_scaled = []
    est_positions_aligned = []
    gt_yaws = []
    est_yaws = []

    for row in cmp_rows:
        seq = to_int(row, "seq")
        if seq not in est_by_seq:
            continue

        est_row = est_by_seq[seq]

        est_scaled = np.array([
            to_float(row, "est_scaled_x"),
            to_float(row, "est_scaled_y"),
            to_float(row, "est_scaled_z"),
        ], dtype=float)

        est_aligned = R @ est_scaled + t

        gt_pos = np.array([
            to_float(row, "gt_initial_x"),
            to_float(row, "gt_initial_y"),
            to_float(row, "gt_initial_z"),
        ], dtype=float)

        qx = to_float(est_row, "qx")
        qy = to_float(est_row, "qy")
        qz = to_float(est_row, "qz")
        qw = to_float(est_row, "qw")

        gt_theta = to_float(row, "gt_theta")
        est_theta = quaternion_to_yaw(qx, qy, qz, qw)

        seqs.append(seq)
        gt_positions.append(gt_pos)
        est_positions_scaled.append(est_scaled)
        est_positions_aligned.append(est_aligned)
        gt_yaws.append(gt_theta)
        est_yaws.append(est_theta)

    return {
        "seqs": np.asarray(seqs, dtype=int),
        "gt_positions": np.asarray(gt_positions, dtype=float),
        "est_positions_scaled": np.asarray(est_positions_scaled, dtype=float),
        "est_positions_aligned": np.asarray(est_positions_aligned, dtype=float),
        "gt_yaws": np.asarray(gt_yaws, dtype=float),
        "est_yaws": np.asarray(est_yaws, dtype=float),
    }


def load_pose_error_data(pose_errors_path):
    """
    Loads pose error data.
    """
    rows = read_csv(pose_errors_path)

    seq0 = []
    seq1 = []
    rotation_error = []
    translation_ratio = []

    for row in rows:
        seq0.append(to_int(row, "seq0"))
        seq1.append(to_int(row, "seq1"))
        rotation_error.append(to_float(row, "rotation_trace_error"))
        translation_ratio.append(to_float(row, "translation_ratio_est_over_gt"))

    return {
        "seq0": np.asarray(seq0, dtype=int),
        "seq1": np.asarray(seq1, dtype=int),
        "rotation_error": np.asarray(rotation_error, dtype=float),
        "translation_ratio": np.asarray(translation_ratio, dtype=float),
    }


def plot_trajectory_top_view(trajectory_data, output_path):
    """
    Creates the trajectory top-view plot.
    """
    gt = trajectory_data["gt_positions"]
    est = trajectory_data["est_positions_aligned"]

    plt.figure(figsize=(9, 7))
    plt.plot(gt[:, 0], gt[:, 1], "-o", markersize=2, linewidth=1.5, label="GT trajectory")
    plt.plot(est[:, 0], est[:, 1], "-o", markersize=2, linewidth=1.5, label="Estimated trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory top view: estimated vs GT")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_world_top_view(map_data, trajectory_data, output_path, pose_step=5, arrow_scale=0.15):
    """
    Creates the top-view world figure with landmarks and poses frame by frame.
    """
    gt_points = map_data["gt"]
    est_points = map_data["est_aligned"]

    gt_traj = trajectory_data["gt_positions"]
    est_traj = trajectory_data["est_positions_aligned"]
    gt_yaws = trajectory_data["gt_yaws"]
    est_yaws = trajectory_data["est_yaws"]
    seqs = trajectory_data["seqs"]

    plt.figure(figsize=(11, 9))

    plt.scatter(gt_points[:, 0], gt_points[:, 1], s=10, label="GT landmarks")
    plt.scatter(est_points[:, 0], est_points[:, 1], s=10, marker="x", label="Estimated landmarks")

    plt.plot(gt_traj[:, 0], gt_traj[:, 1], linewidth=2.0, label="GT trajectory")
    plt.plot(est_traj[:, 0], est_traj[:, 1], linewidth=2.0, label="Estimated trajectory")

    for i in range(0, len(seqs), pose_step):
        gt_x = gt_traj[i, 0]
        gt_y = gt_traj[i, 1]
        gt_dx = math.cos(gt_yaws[i]) * arrow_scale
        gt_dy = math.sin(gt_yaws[i]) * arrow_scale

        est_x = est_traj[i, 0]
        est_y = est_traj[i, 1]
        est_dx = math.cos(est_yaws[i]) * arrow_scale
        est_dy = math.sin(est_yaws[i]) * arrow_scale

        plt.arrow(gt_x, gt_y, gt_dx, gt_dy, head_width=0.04, length_includes_head=True)
        plt.arrow(est_x, est_y, est_dx, est_dy, head_width=0.04, length_includes_head=True)

        plt.text(gt_x, gt_y, str(seqs[i]), fontsize=7)
        plt.text(est_x, est_y, str(seqs[i]), fontsize=7)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("World top view: landmarks and poses")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_world_3d(map_data, trajectory_data, output_path):
    """
    Creates the 3D world/map figure.
    """
    gt_points = map_data["gt"]
    est_points = map_data["est_aligned"]

    gt_traj = trajectory_data["gt_positions"]
    est_traj = trajectory_data["est_positions_aligned"]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], s=8, label="GT landmarks")
    ax.scatter(est_points[:, 0], est_points[:, 1], est_points[:, 2], s=8, marker="x", label="Estimated landmarks")

    ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], linewidth=2.0, label="GT trajectory")
    ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], linewidth=2.0, label="Estimated trajectory")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D world/map: landmarks and trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_rotation_error(error_data, output_path):
    """
    Creates the rotation error plot.
    """
    seq1 = error_data["seq1"]
    rotation_error = error_data["rotation_error"]

    plt.figure(figsize=(10, 5))
    plt.plot(seq1, rotation_error, linewidth=1.5)
    plt.xlabel("frame")
    plt.ylabel("trace(I - R_error)")
    plt.title("Rotation error per relative step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def plot_translation_ratio(error_data, output_path):
    """
    Creates the translation scale-ratio plot.
    """
    seq1 = error_data["seq1"]
    translation_ratio = error_data["translation_ratio"]

    plt.figure(figsize=(10, 5))
    plt.plot(seq1, translation_ratio, linewidth=1.5)
    plt.xlabel("frame")
    plt.ylabel("||t_est|| / ||t_gt||")
    plt.title("Translation scale-ratio per relative step")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def main():
    """
    Main plotting function.
    """
    evaluation_dir = "output/evaluation"
    figures_dir = "output/evaluation/figures"

    if len(sys.argv) > 1:
        evaluation_dir = sys.argv[1]

    if len(sys.argv) > 2:
        figures_dir = sys.argv[2]

    ensure_dir(figures_dir)

    map_errors_path = os.path.join(evaluation_dir, "map_errors.csv")
    trajectory_estimate_path = os.path.join(evaluation_dir, "trajectory_estimate.csv")
    trajectory_compare_path = os.path.join(evaluation_dir, "trajectory_compare.csv")
    pose_errors_path = os.path.join(evaluation_dir, "pose_errors.csv")

    map_data = load_map_data(map_errors_path)
    trajectory_data = load_trajectory_data(
        trajectory_estimate_path,
        trajectory_compare_path,
        map_data["R"],
        map_data["t"],
    )
    error_data = load_pose_error_data(pose_errors_path)

    plot_trajectory_top_view(
        trajectory_data,
        os.path.join(figures_dir, "01_trajectory_top_view.png"),
    )

    plot_world_top_view(
        map_data,
        trajectory_data,
        os.path.join(figures_dir, "02_world_top_view.png"),
        pose_step=5,
        arrow_scale=0.15,
    )

    plot_world_3d(
        map_data,
        trajectory_data,
        os.path.join(figures_dir, "03_world_3d.png"),
    )

    plot_rotation_error(
        error_data,
        os.path.join(figures_dir, "04_rotation_error.png"),
    )

    plot_translation_ratio(
        error_data,
        os.path.join(figures_dir, "05_translation_ratio.png"),
    )

    print("Figures written to:", figures_dir)


if __name__ == "__main__":
    main()