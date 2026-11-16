"""Plot RigVis per-frame tracking stats against yaw rate.

Usage:
    python3 tools/plot_frame_stats.py path/to/frame_stats.csv [--turn-dps 10]

Shaded bands = turns (|yaw rate| above --turn-dps).
Red ticks = frames that did not track on the local map (fallback or fail).
Read it left to right before each failure: the first curve that drops is the cause.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--turn-dps", type=float, default=10.0)
    parser.add_argument("--out", default=None, help="save to PNG instead of showing")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    x = df["frame"]
    turning = df["yaw_rate_dps"].abs() > args.turn_dps
    not_local = ~df["path"].isin(["local", "init"])

    panels = [
        ("Yaw rate [deg/s]", [("yaw_rate_dps", "yaw rate")]),
        ("Detection / stereo", [("features", "features"), ("stereo_matches", "stereo matches")]),
        ("Local map", [("local_map_points", "points in window"),
                       ("local_map_visible", "visible at prev pose")]),
        ("Local PnP", [("local_pairs", "3D-2D pairs"), ("local_inliers", "inliers"),
                       ("tracked_map_points", "tracked")]),
        ("Quality [0-1]", [("local_inlier_ratio", "inlier ratio w"),
                           ("grid_coverage", "grid coverage"),
                           ("hull_area", "hull area")]),
    ]

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(14, 2.3 * len(panels)))
    for ax, (title, series) in zip(axes, panels):
        for column, label in series:
            ax.plot(x, df[column], label=label, linewidth=1)
        # Shade turns.
        ax.fill_between(x, 0, 1, where=turning, transform=ax.get_xaxis_transform(),
                        color="orange", alpha=0.15, linewidth=0)
        # Mark non-local-map frames.
        for f in x[not_local]:
            ax.axvline(f, color="red", alpha=0.25, linewidth=0.6)
        ax.set_ylabel(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].axhline(0.4, color="gray", linestyle="--", linewidth=0.8)  # RANSAC danger zone
    axes[-1].set_xlabel("frame")

    kf_frames = x[df["keyframe"] == 1]
    axes[0].plot(kf_frames, [0] * len(kf_frames), "|", color="black", label="keyframe")
    axes[0].legend(loc="upper right", fontsize=8)

    path_counts = df["path"].value_counts().to_dict()
    fig.suptitle(f"RigVis frame stats  |  paths: {path_counts}", fontsize=10)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
