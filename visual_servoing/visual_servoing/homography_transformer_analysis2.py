#!/usr/bin/env python3
"""Analysis script for comparing ground truth coordinates vs simulated transformer output.

Data options
- manual ground truth and sim values
- optional ROS bag extraction (if rosbag2 libraries are available)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point2D:
    x: float
    y: float


def parse_list_of_pairs(values: List[str]) -> List[Point2D]:
    pairs = []
    for token in values:
        token = token.strip().replace("(", "").replace(")", "")
        if not token:
            continue
        # Expect x,y
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected pair x,y but got: {token}")
        pairs.append(Point2D(float(parts[0].strip()), float(parts[1].strip())))
    return pairs


def load_from_rosbag2(bag_path: str, topic: str = "/relative_cone") -> List[Point2D]:
    try:
        from rosbag2_py import SequentialReader
        from rosbag2_py import StorageOptions, ConverterOptions
    except ImportError as e:
        raise ImportError(
            "rosbag2_py not found."
        ) from e

    points: List[Point2D] = []
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()

    while reader.has_next():
        (topic_name, data_buffer) = reader.read_next()
        if topic_name != topic:
            continue
        try:
            # decode to ConeLocation
            from vs_msgs.msg import ConeLocation

            msg = ConeLocation()
            msg.deserialize(data_buffer)
            points.append(Point2D(msg.x_pos, msg.y_pos))
        except Exception:
            pass
    return points


def compute_error_metrics(
    ground_truths: List[Point2D], sim_values: List[Point2D]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = min(len(ground_truths), len(sim_values))
    if n == 0:
        raise ValueError("No paired points available for error metrics.")

    gt = np.array([[p.x, p.y] for p in ground_truths[:n]], dtype=float)
    sim = np.array([[p.x, p.y] for p in sim_values[:n]], dtype=float)
    diff = sim - gt
    dx = diff[:, 0]
    dy = diff[:, 1]
    d2 = np.sum(diff**2, axis=1)
    dist = np.sqrt(d2)
    gt_norm = np.sqrt(np.sum(gt**2, axis=1))
    relative_error = np.zeros_like(dist)
    nonzero = gt_norm > 1e-12
    relative_error[nonzero] = dist[nonzero] / gt_norm[nonzero]
    pct_error = relative_error * 100.0

    metrics = {
        "count": int(n),
        "mean_dx": float(np.mean(dx)),
        "mean_dy": float(np.mean(dy)),
        "mae_x": float(np.mean(np.abs(dx))),
        "mae_y": float(np.mean(np.abs(dy))),
        "mae_dist": float(np.mean(dist)),
        "rmse_x": float(np.sqrt(np.mean(dx**2))),
        "rmse_y": float(np.sqrt(np.mean(dy**2))),
        "rmse_dist": float(np.sqrt(np.mean(dist**2))),
        "mean_pct_error": float(np.mean(pct_error)),
        "std_pct_error": float(np.std(pct_error, ddof=0)),
        "std_dx": float(np.std(dx, ddof=0)),
        "std_dy": float(np.std(dy, ddof=0)),
        "std_dist": float(np.std(dist, ddof=0)),
        "max_dist": float(np.max(dist)),
    }

    return dx, dy, dist, metrics


def plot_comparison(
    ground_truths: List[Point2D],
    sim_values: List[Point2D],
    dx: np.ndarray,
    dy: np.ndarray,
    dist: np.ndarray,
    out_path: str = "analysis_plot.png",
) -> None:
    n = min(len(ground_truths), len(sim_values))
    gt = np.array([[p.x, p.y] for p in ground_truths[:n]])
    sim = np.array([[p.x, p.y] for p in sim_values[:n]])
    indices = np.arange(n)

    plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(gt[:, 0], gt[:, 1], c="tab:blue", marker="o", label="Ground Truth")
    ax1.scatter(sim[:, 0], sim[:, 1], c="tab:orange", marker="x", label="Sim Values")
    for i in range(n):
        ax1.plot([gt[i, 0], sim[i, 0]], [gt[i, 1], sim[i, 1]], c="gray", linewidth=0.8)
    ax1.set_title("Ground Truth vs Sim Coordinates")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(indices, dx, label="dx", marker="o")
    ax2.plot(indices, dy, label="dy", marker="x")
    ax2.set_title("Per-sample coordinate errors")
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Error (m)")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(indices, dist, label="Euclidean error", color="tab:red", marker="o")
    ax3.set_title("Per-sample Euclidean error")
    ax3.set_xlabel("Sample")
    ax3.set_ylabel("Distance error (m)")
    ax3.grid(True)

    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(dist, bins=min(20, max(5, n)), color="tab:green", alpha=0.75)
    ax4.set_title("Distance error distribution")
    ax4.set_xlabel("Distance error (m)")
    ax4.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")
    plt.show()


def print_metrics(metrics: dict) -> None:
    print("Error metrics:")
    print(f"  Paired samples: {metrics['count']}")
    print(f"  mean dx: {metrics['mean_dx']:.6f} m")
    print(f"  mean dy: {metrics['mean_dy']:.6f} m")
    print(f"  MAE x: {metrics['mae_x']:.6f} m")
    print(f"  MAE y: {metrics['mae_y']:.6f} m")
    print(f"  MAE distance: {metrics['mae_dist']:.6f} m")
    print(f"  RMSE x: {metrics['rmse_x']:.6f} m")
    print(f"  RMSE y: {metrics['rmse_y']:.6f} m")
    print(f"  RMSE distance: {metrics['rmse_dist']:.6f} m")
    print(f"  mean % error: {metrics['mean_pct_error']:.3f}%")
    print(f"  std % error: {metrics['std_pct_error']:.3f}%")
    print(f"  std dist: {metrics['std_dist']:.6f} m")
    print(f"  max dist: {metrics['max_dist']:.6f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ground truth coordinates vs simulation values")
    parser.add_argument("--ground_truths", type=str, default="",
                        help="Manual ground truth coordinates as x,y pairs separated by semicolon: '1.0,0.2;1.1,0.3'")
    parser.add_argument("--sim_values", type=str, default="",
                        help="Manual sim coordinates as x,y pairs separated by semicolon")
    parser.add_argument("--plot_out", type=str, default="",
                        help="Optional output path to save plot image instead of showing")
    parser.add_argument("--rosbag_path", type=str, default="",
                        help="Optional rosbag2 path to load sim values from relative_cone topic")
    parser.add_argument("--rosbag_topic", type=str, default="/relative_cone",
                        help="Topic name in rosbag for sim values")

    args = parser.parse_args()

    ground_truths: List[Point2D] = [
        Point2D(0.3683, 0.2794),
        Point2D(0.8001, 0.2794),
        Point2D(0.8001, -0.0508),
        Point2D(0.6477, -0.5588),
        Point2D(15 * 0.0254, -5 * 0.0254),
        Point2D(17 * 0.0254, -5 * 0.0254),
        Point2D(20 * 0.0254, -9 * 0.0254),
        Point2D(31 * 0.0254, -1 * 0.0254),
    ]
    sim_values: List[Point2D] = [
        Point2D(0.362, 0.294),
        Point2D(1.1, 0.444),
        Point2D(1.047, 0.019),
        Point2D(0.893, -0.562),
        Point2D(0.492, -0.163),
        Point2D(0.765, -0.326),
        Point2D(0.896, -0.199),
        Point2D(1.307, 0.107),
    ]

    if args.ground_truths:
        ground_truths = parse_list_of_pairs(args.ground_truths.split(";"))

    if args.sim_values:
        sim_values = parse_list_of_pairs(args.sim_values.split(";"))

    if args.rosbag_path:
        print(f"Loading sim values from rosbag: {args.rosbag_path} topic: {args.rosbag_topic}")
        try:
            ros_values = load_from_rosbag2(args.rosbag_path, args.rosbag_topic)
            if len(ros_values) == 0:
                print("No points loaded from rosbag; check path and topic.")
            else:
                sim_values = ros_values
        except Exception as exc:
            print(f"Could not load rosbag data: {exc}")
            if not sim_values:
                print("No valid sim values available. Exiting.")
                return 1

    if len(ground_truths) == 0:
        print("No ground truth points were provided. Define --ground_truths or --ground_truth_csv.")
        return 1
    if len(sim_values) == 0:
        print("No sim points were provided. Define --sim_values, --sim_csv, or --rosbag_path.")
        return 1

    n = min(len(ground_truths), len(sim_values))
    if len(ground_truths) != len(sim_values):
        print(f"Warning: lengths differ. Using first {n} pairs for analysis.")

    dx, dy, dist, metrics = compute_error_metrics(ground_truths, sim_values)
    print_metrics(metrics)

    plot_out_path = args.plot_out if args.plot_out else "analysis_plot.png"
    plot_comparison(ground_truths, sim_values, dx, dy, dist, out_path=plot_out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
