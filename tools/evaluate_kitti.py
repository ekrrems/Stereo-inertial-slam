"""Evaluate a RigVis trajectory against KITTI ground truth.

Ground truth comes either from
  * the raw drive's OXTS (GPS/INS) files + calibration  (--drive, default in main.cpp), or
  * a KITTI Odometry poses file, e.g. poses/00.txt        (--gt).
Drive 2011_10_03_drive_0027 is KITTI Odometry sequence 00 (frames 0-4540).

Trajectory format (both files): KITTI odometry, one line per frame, the 12
numbers of the 3x4 camera-to-world matrix [R | t], row-major, left rectified
camera, first frame = identity.

Metrics
  ATE  absolute trajectory error: RMSE of positions after the best rigid
       alignment (rotation + translation, NO scale since stereo is metric).
       Measures the whole shape; one early rotation error dominates it.
  t_rel / r_rel  the official KITTI odometry metric: for every start frame
       (step 10) and segment length 100..800 m, compare the relative motion of
       estimate vs truth. Reported as % of distance and deg per 100 m.
       This is the number papers report (ORB-SLAM2 stereo, seq 00: ~0.70 %,
       ~0.25 deg/100 m, WITH local BA and loop closure).
  RPE  per-frame relative error (1-frame motion): shows WHERE it goes wrong.

Usage
  python3 tools/evaluate_kitti.py --est trajectory_kitti.txt --drive <..._sync folder>
  python3 tools/evaluate_kitti.py --est trajectory_kitti.txt --gt poses/00.txt
Options
  --first-frame N   estimate line 0 corresponds to ground-truth frame N
  --plot out.png    top view + error plots
  --results-csv f   append one summary row per run (default: next to --est)
"""
import argparse
import csv
import datetime
import math
import os
import sys

import numpy as np

KITTI_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]  # metres, devkit values
KITTI_STEP = 10                                             # start-frame step, devkit value


# --------------------------------------------------------------- loading

def load_kitti_poses(path):
    """KITTI odometry format -> list of 4x4 camera-to-world matrices."""
    poses = []
    with open(path) as f:
        for line_number, line in enumerate(f, 1):
            values = line.split()
            if not values:
                continue
            if len(values) != 12:
                sys.exit(f"{path}:{line_number}: expected 12 numbers, got {len(values)}")
            pose = np.eye(4)
            pose[:3, :4] = np.array(values, dtype=float).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)


def read_calib_file(path):
    values = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            try:
                values[key.strip()] = np.array(rest.split(), dtype=float)
            except ValueError:
                pass  # e.g. calib_time
    return values


def rigid(rotation_flat, translation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation_flat).reshape(3, 3)
    transform[:3, 3] = translation
    return transform


def ground_truth_from_oxts(drive_path):
    """KITTI raw devkit convertOxtsToPose + calibration -> left rectified camera poses.

    OXTS gives the IMU pose in a local Mercator frame. The camera pose is
        T_world_cam = T_world_imu * T_imu_cam,  T_cam_imu = R_rect_00 * T_cam0_velo * T_velo_imu
    and everything is expressed relative to the first camera frame.
    """
    drive_path = os.path.abspath(drive_path)
    day_path = os.path.dirname(drive_path)
    oxts_dir = os.path.join(drive_path, "oxts", "data")
    files = sorted(f for f in os.listdir(oxts_dir) if f.endswith(".txt"))
    if not files:
        sys.exit(f"No OXTS files in {oxts_dir}")

    imu_to_velo = read_calib_file(os.path.join(day_path, "calib_imu_to_velo.txt"))
    velo_to_cam = read_calib_file(os.path.join(day_path, "calib_velo_to_cam.txt"))
    cam_to_cam = read_calib_file(os.path.join(day_path, "calib_cam_to_cam.txt"))

    T_velo_imu = rigid(imu_to_velo["R"], imu_to_velo["T"])
    T_cam0_velo = rigid(velo_to_cam["R"], velo_to_cam["T"])
    R_rect = rigid(cam_to_cam["R_rect_00"], np.zeros(3))
    T_cam_imu = R_rect @ T_cam0_velo @ T_velo_imu
    T_imu_cam = np.linalg.inv(T_cam_imu)

    earth_radius = 6378137.0
    scale = None
    camera_poses = []
    for name in files:
        lat, lon, alt, roll, pitch, yaw = np.loadtxt(os.path.join(oxts_dir, name))[:6]
        if scale is None:
            scale = math.cos(math.radians(lat))
        x = scale * math.radians(lon) * earth_radius
        y = scale * earth_radius * math.log(math.tan(math.radians(90.0 + lat) / 2.0))

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        T_world_imu = np.eye(4)
        T_world_imu[:3, :3] = Rz @ Ry @ Rx
        T_world_imu[:3, 3] = [x, y, alt]
        camera_poses.append(T_world_imu @ T_imu_cam)

    first_inverse = np.linalg.inv(camera_poses[0])
    return np.array([first_inverse @ pose for pose in camera_poses])


# --------------------------------------------------------------- metrics

def rotation_angle(rotation):
    """Angle of a rotation matrix in radians (robust to rounding)."""
    return math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))


def align_rigid(estimated_positions, true_positions):
    """Umeyama/Kabsch without scale: R, t minimising |R*est + t - true|."""
    mean_est = estimated_positions.mean(axis=0)
    mean_true = true_positions.mean(axis=0)
    covariance = (true_positions - mean_true).T @ (estimated_positions - mean_est)
    U, _, Vt = np.linalg.svd(covariance)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))  # no reflections
    R = U @ D @ Vt
    t = mean_true - R @ mean_est
    return R, t


def absolute_trajectory_error(estimated, truth):
    est_positions = estimated[:, :3, 3]
    true_positions = truth[:, :3, 3]
    R, t = align_rigid(est_positions, true_positions)
    aligned = (R @ est_positions.T).T + t
    errors = np.linalg.norm(aligned - true_positions, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors, aligned


def trajectory_distances(poses):
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def kitti_relative_errors(estimated, truth):
    """Official KITTI odometry metric (devkit evaluate_odometry.cpp)."""
    distances = trajectory_distances(truth)
    segments = []  # (length, translation error fraction, rotation error rad/m)
    for first in range(0, len(truth), KITTI_STEP):
        for length in KITTI_LENGTHS:
            reached = np.nonzero(distances >= distances[first] + length)[0]
            if reached.size == 0:
                continue
            last = int(reached[0])
            delta_true = np.linalg.inv(truth[first]) @ truth[last]
            delta_est = np.linalg.inv(estimated[first]) @ estimated[last]
            error = np.linalg.inv(delta_est) @ delta_true
            segments.append((length,
                             np.linalg.norm(error[:3, 3]) / length,
                             rotation_angle(error[:3, :3]) / length))
    if not segments:
        return None, None, segments
    t_rel = 100.0 * np.mean([s[1] for s in segments])                    # %
    r_rel = 100.0 * math.degrees(1.0) * np.mean([s[2] for s in segments])  # deg / 100 m
    return t_rel, r_rel, segments


def per_frame_errors(estimated, truth):
    """1-frame relative pose error: translation [m] and rotation [deg]."""
    translation, rotation = [], []
    for i in range(1, len(truth)):
        delta_true = np.linalg.inv(truth[i - 1]) @ truth[i]
        delta_est = np.linalg.inv(estimated[i - 1]) @ estimated[i]
        error = np.linalg.inv(delta_est) @ delta_true
        translation.append(np.linalg.norm(error[:3, 3]))
        rotation.append(math.degrees(rotation_angle(error[:3, :3])))
    return np.array(translation), np.array(rotation)


# --------------------------------------------------------------- output

def make_plot(path, truth, aligned, ate_errors, rpe_t, rpe_r, first_frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(len(truth)) + first_frame
    fig = plt.figure(figsize=(14, 8))
    top = fig.add_subplot(1, 2, 1)
    # Camera frame: x right, z forward -> top view is (x, z).
    top.plot(truth[:, 0, 3], truth[:, 2, 3], label="ground truth", linewidth=1.5, color="#444444")
    top.plot(aligned[:, 0], aligned[:, 2], label="RigVis (aligned)", linewidth=1.2, color="#e07020")
    top.scatter(truth[0, 0, 3], truth[0, 2, 3], color="green", zorder=3, label="start")
    top.set_aspect("equal")
    top.set_xlabel("x [m]")
    top.set_ylabel("z [m]")
    top.set_title("Top view")
    top.legend()
    top.grid(alpha=0.3)

    ate_axis = fig.add_subplot(3, 2, 2)
    ate_axis.plot(frames, ate_errors, color="#e07020", linewidth=1)
    ate_axis.set_ylabel("ATE [m]")
    ate_axis.grid(alpha=0.3)

    rpe_t_axis = fig.add_subplot(3, 2, 4, sharex=ate_axis)
    rpe_t_axis.plot(frames[1:], rpe_t, color="#2070c0", linewidth=0.8)
    rpe_t_axis.set_ylabel("RPE trans [m/frame]")
    rpe_t_axis.grid(alpha=0.3)

    rpe_r_axis = fig.add_subplot(3, 2, 6, sharex=ate_axis)
    rpe_r_axis.plot(frames[1:], rpe_r, color="#a03090", linewidth=0.8)
    rpe_r_axis.set_ylabel("RPE rot [deg/frame]")
    rpe_r_axis.set_xlabel("frame")
    rpe_r_axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Plot saved to {path}")


def append_results(path, row):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"Summary appended to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--est", required=True, help="RigVis trajectory (KITTI format)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--drive", help="raw KITTI *_sync drive folder (uses OXTS + calib)")
    source.add_argument("--gt", help="KITTI odometry poses file, e.g. poses/00.txt")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--label", default="", help="free text stored in the results CSV")
    args = parser.parse_args()

    estimated = load_kitti_poses(args.est)
    truth_all = ground_truth_from_oxts(args.drive) if args.drive else load_kitti_poses(args.gt)

    # Pair estimate line i with ground-truth frame first_frame + i. A run stopped
    # early simply covers fewer frames.
    count = min(len(estimated), len(truth_all) - args.first_frame)
    if count < 2:
        sys.exit("Not enough overlapping frames to evaluate.")
    estimated = estimated[:count]
    truth = truth_all[args.first_frame:args.first_frame + count]
    # Express ground truth relative to its own first paired frame, like the estimate.
    truth = np.array([np.linalg.inv(truth[0]) @ pose for pose in truth])

    ate, ate_errors, aligned = absolute_trajectory_error(estimated, truth)
    t_rel, r_rel, segments = kitti_relative_errors(estimated, truth)
    rpe_t, rpe_r = per_frame_errors(estimated, truth)
    true_length = trajectory_distances(truth)[-1]
    est_length = trajectory_distances(estimated)[-1]
    worst = int(np.argmax(rpe_t)) + 1

    print()
    print("================ RigVis vs KITTI ground truth ================")
    print(f"Frames evaluated      : {count}  (gt frames {args.first_frame}-{args.first_frame + count - 1})")
    print(f"Path length           : truth {true_length:8.1f} m | estimate {est_length:8.1f} m "
          f"({100.0 * (est_length / true_length - 1.0):+.1f} %)")
    print(f"ATE (RMSE, aligned)   : {ate:8.3f} m   (max {ate_errors.max():.2f} m)")
    if t_rel is None:
        print(f"KITTI t_rel / r_rel   : n/a (needs >= {KITTI_LENGTHS[0]} m of driving)")
    else:
        used = sorted({s[0] for s in segments})
        print(f"KITTI t_rel           : {t_rel:8.3f} %       over {len(segments)} segments, lengths {used} m")
        print(f"KITTI r_rel           : {r_rel:8.3f} deg/100m")
    print(f"RPE per frame (mean)  : {rpe_t.mean():.3f} m, {rpe_r.mean():.3f} deg")
    print(f"Worst frame           : {args.first_frame + worst}  "
          f"({rpe_t[worst - 1]:.3f} m, {rpe_r[worst - 1]:.3f} deg)")
    print("Reference (seq 00)    : ORB-SLAM2 stereo ~0.70 %, ~0.25 deg/100m (with BA + loop closure)")
    print("=============================================================")

    if args.plot:
        make_plot(args.plot, truth, aligned, ate_errors, rpe_t, rpe_r, args.first_frame)

    results_path = args.results_csv or os.path.join(os.path.dirname(os.path.abspath(args.est)),
                                                    "evaluation_results.csv")
    append_results(results_path, {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "estimate": os.path.abspath(args.est),
        "frames": count,
        "first_frame": args.first_frame,
        "length_m": round(true_length, 1),
        "ate_m": round(ate, 3),
        "t_rel_pct": "" if t_rel is None else round(t_rel, 3),
        "r_rel_deg_per_100m": "" if r_rel is None else round(r_rel, 3),
        "rpe_t_mean_m": round(float(rpe_t.mean()), 4),
        "rpe_r_mean_deg": round(float(rpe_r.mean()), 4),
    })


if __name__ == "__main__":
    main()
