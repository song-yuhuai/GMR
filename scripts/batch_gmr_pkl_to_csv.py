import argparse
import pickle
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR pickle files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, required=True, help="Path to the folder containing pickle files from GMR",
    )
    # NEW: z offset to shift the whole robot up/down via root_pos.z
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.0,
        help="Additive offset applied to root_pos.z (meters). Negative moves robot down. Default: 0.0",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    files = os.listdir(args.folder)
    pkl_files = [f for f in files if f.endswith(".pkl")]

    for i, file in enumerate(pkl_files):
        with open(os.path.join(args.folder, file), "rb") as f:
            motion_data = pickle.load(f)

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]

        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)

        # Root pos + NEW z_offset
        root_pos = np.array(motion_data["root_pos"], dtype=np.float32, copy=True)
        root_pos[:, 2] += np.float32(args.z_offset)  # apply z shift
        motion[:, :3] = root_pos

        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos

        if frame_rate > 30:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")

        out_path = os.path.join(out_folder, file.replace(".pkl", ".csv"))
        np.savetxt(out_path, motion, delimiter=",")
        print(f"({i+1}/{len(pkl_files)}) Saved to {out_path}")