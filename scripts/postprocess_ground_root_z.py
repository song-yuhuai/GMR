import argparse
import pickle
from pathlib import Path

import numpy as np


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process GMR PKL by detrending root_pos.z only. "
            "This removes slow vertical drift while preserving local up/down dance motion."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input motion .pkl from GMR")
    parser.add_argument("--output", type=str, required=True, help="Output corrected .pkl path")
    # Kept for backward-compatible CLI calls; not used by root-z-only logic.
    parser.add_argument("--robot", type=str, default=None, help="Unused in root-z-only mode")
    parser.add_argument(
        "--baseline_window",
        type=int,
        default=30,
        help="Use median(trend[:N]) as reference level. Default: 30",
    )
    parser.add_argument(
        "--trend_window",
        type=int,
        default=51,
        help="Centered moving-average window used to estimate slow root-z trend. Default: 51",
    )
    parser.add_argument(
        "--max_abs_delta",
        type=float,
        default=0.08,
        help="Clamp correction delta to +/- this value in meters. Default: 0.08",
    )
    parser.add_argument(
        "--use_whole_clip_baseline",
        action="store_true",
        help="Use median(trend over whole clip) instead of first N frames",
    )
    parser.add_argument("--debug_print", action="store_true", help="Print extra debug details")
    parser.add_argument("--debug_plot", action="store_true", help="Plot root-z/trend/delta curves")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        motion_data = pickle.load(f)

    if "root_pos" not in motion_data:
        raise KeyError("Input PKL missing required key: root_pos")

    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] < 3:
        raise ValueError("root_pos must have shape [T, 3]")

    num_frames = root_pos.shape[0]
    if num_frames == 0:
        raise ValueError("Input motion has zero frames")

    root_z = root_pos[:, 2].copy()

    trend_window = max(1, int(args.trend_window))
    trend = moving_average(root_z, trend_window)

    if args.use_whole_clip_baseline:
        baseline_slice = trend
        baseline_mode = "whole_clip"
    else:
        n = int(np.clip(args.baseline_window, 1, num_frames))
        baseline_slice = trend[:n]
        baseline_mode = f"first_{n}_frames"

    z_ref = float(np.median(baseline_slice))

    raw_delta = z_ref - trend
    max_abs_delta = float(max(0.0, args.max_abs_delta))
    clipped_delta = np.clip(raw_delta, -max_abs_delta, max_abs_delta)

    new_root_z = root_z + clipped_delta

    corrected_root_pos = root_pos.copy()
    corrected_root_pos[:, 2] = new_root_z

    print(f"[root-z-detrend] baseline mode: {baseline_mode}")
    print(f"[root-z-detrend] z_ref: {z_ref:.6f}")
    print(f"[root-z-detrend] original root_z range: [{root_z.min():.6f}, {root_z.max():.6f}]")
    print(f"[root-z-detrend] trend range: [{trend.min():.6f}, {trend.max():.6f}]")
    print(f"[root-z-detrend] raw delta range: [{raw_delta.min():.6f}, {raw_delta.max():.6f}]")
    print(
        f"[root-z-detrend] clipped delta range: "
        f"[{clipped_delta.min():.6f}, {clipped_delta.max():.6f}] (max_abs_delta={max_abs_delta:.6f})"
    )

    sample_ids = [0, min(1, num_frames - 1), min(2, num_frames - 1), num_frames // 2, num_frames - 1]
    sample_ids = sorted(set(sample_ids))
    print("[root-z-detrend] sample frames (root_z old -> new):")
    for i in sample_ids:
        print(
            f"  frame {i:4d}: "
            f"{root_z[i]: .6f} -> {new_root_z[i]: .6f}, "
            f"delta={clipped_delta[i]: .6f}"
        )

    if args.debug_print:
        print("[root-z-detrend][debug] first 10 root_z:", np.array2string(root_z[:10], precision=6))
        print("[root-z-detrend][debug] first 10 trend:", np.array2string(trend[:10], precision=6))
        print("[root-z-detrend][debug] first 10 raw_delta:", np.array2string(raw_delta[:10], precision=6))
        print(
            "[root-z-detrend][debug] first 10 clipped_delta:",
            np.array2string(clipped_delta[:10], precision=6),
        )

    out_data = dict(motion_data)
    out_data["root_pos"] = corrected_root_pos.astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(out_data, f)

    print(f"[root-z-detrend] saved corrected motion to: {output_path}")

    if args.debug_plot:
        try:
            import matplotlib.pyplot as plt

            x = np.arange(num_frames)
            plt.figure(figsize=(12, 7))
            plt.plot(x, root_z, label="root_z (original)", linewidth=1.5)
            plt.plot(x, trend, label="trend", linewidth=2.0)
            plt.plot(x, raw_delta, label="delta_raw = z_ref - trend", linestyle="--")
            plt.plot(x, clipped_delta, label="delta_clipped", linewidth=2.0)
            plt.plot(x, new_root_z, label="root_z (corrected)", linewidth=1.8)
            plt.axhline(y=z_ref, color="k", linestyle=":", label="z_ref")
            plt.xlabel("Frame")
            plt.ylabel("Height / Correction (m)")
            plt.title("Root-z detrend / drift stabilization")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("[root-z-detrend] --debug_plot requested, but matplotlib is not installed.")


if __name__ == "__main__":
    main()
