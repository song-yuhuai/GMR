import argparse
import pathlib
import pickle

import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT

# Edit here if you want fixed custom start/end poses in this script.
# Format: [root_pos(3), root_quat_wxyz(4), dof_pos(N)]
# Keep None to disable custom pose.
#
# agibot_x2/x2_ultra.xml defaults:
# qpos0 = [root(7), 31 joint dofs], where all joint dofs are 0 by default.
CUSTOM_ENTRY_POSE = [0.0, 0.0, 0.68, 1.0, 0.0, 0.0, 0.0] + [-0.235, 0.0, 0.0, 0.5, -0.265, 0.0,
                 -0.235, 0.0, 0.0, 0.5, -0.265, 0.0,
                 0.0, 0.0] + [0.0] * 9
CUSTOM_EXIT_POSE = [0.0, 0.0, 0.68, 1.0, 0.0, 0.0, 0.0] + [-0.235, 0.0, 0.0, 0.5, -0.265, 0.0,
                 -0.235, 0.0, 0.0, 0.5, -0.265, 0.0,
                 0.0, 0.0] +[0.0] * 9
ENTRY_POSE_SOURCE = "custom_entry"
EXIT_POSE_SOURCE = "custom_exit"


def _blend_qpos(a, b, alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = a.copy()
    out[:3] = a[:3]

    qa = a[3:7]
    qb = b[3:7]
    if np.dot(qa, qb) < 0.0:
        qb = -qb
    q = (1.0 - alpha) * qa + alpha * qb
    qn = np.linalg.norm(q)
    out[3:7] = q / qn if qn > 1e-8 else qa

    out[7:] = (1.0 - alpha) * a[7:] + alpha * b[7:]
    return out


def _build_transition(start_q, end_q, transition_frames):
    if transition_frames <= 0:
        return []
    return [
        _blend_qpos(start_q, end_q, alpha=(i + 1) / float(transition_frames + 1))
        for i in range(transition_frames)
    ]


def _add_entry_exit_pose(qpos_seq, start_pose, end_pose, hold_frames, transition_frames):
    if len(qpos_seq) == 0:
        return qpos_seq

    parts = []
    if hold_frames > 0:
        parts.append(np.repeat(start_pose[None, :], hold_frames, axis=0))

    trans_in = _build_transition(start_pose, qpos_seq[0], transition_frames)
    if trans_in:
        parts.append(np.asarray(trans_in))

    parts.append(qpos_seq)

    trans_out = _build_transition(qpos_seq[-1], end_pose, transition_frames)
    if trans_out:
        parts.append(np.asarray(trans_out))

    if hold_frames > 0:
        parts.append(np.repeat(end_pose[None, :], hold_frames, axis=0))

    return np.concatenate(parts, axis=0)


def _load_robot_qpos0(robot):
    if robot not in ROBOT_XML_DICT:
        raise ValueError(f"Unknown robot '{robot}'.")
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    return model.qpos0.copy()


def _resolve_custom_pose(custom_pose, qpos_dim, name):
    if custom_pose is None:
        raise ValueError(
            f"{name} is None. Please edit CUSTOM_ENTRY_POSE/CUSTOM_EXIT_POSE in this file."
        )
    pose = np.asarray(custom_pose, dtype=float).reshape(-1)
    if pose.shape[0] == qpos_dim:
        return pose

    # Auto-adapt qpos length mismatch:
    # keep root 7 DoFs and resize joint DoFs to match the motion file.
    if pose.shape[0] >= 7 and qpos_dim >= 7:
        out = np.zeros(qpos_dim, dtype=float)
        out[:7] = pose[:7]
        copy_joint = min(pose.shape[0] - 7, qpos_dim - 7)
        if copy_joint > 0:
            out[7:7 + copy_joint] = pose[7:7 + copy_joint]
        print(
            f"[warn] {name} length mismatch: expected {qpos_dim}, got {pose.shape[0]}. "
            f"Auto-adapted by keeping root and resizing joint DoFs."
        )
        return out

    raise ValueError(
        f"{name} length mismatch: expected {qpos_dim}, got {pose.shape[0]}."
    )


def _resolve_pose(source, qpos_seq, robot):
    if source == "action_first":
        return qpos_seq[0].copy()
    if source == "action_last":
        return qpos_seq[-1].copy()
    if source == "custom_entry":
        return _resolve_custom_pose(CUSTOM_ENTRY_POSE, qpos_seq.shape[1], "CUSTOM_ENTRY_POSE")
    if source == "custom_exit":
        return _resolve_custom_pose(CUSTOM_EXIT_POSE, qpos_seq.shape[1], "CUSTOM_EXIT_POSE")
    if source == "robot_qpos0":
        if not robot:
            raise ValueError("Pose source 'robot_qpos0' requires --robot.")
        return _load_robot_qpos0(robot)
    raise ValueError(f"Unsupported pose source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True, type=str, help="Input retargeted pkl path.")
    parser.add_argument("--output_pkl", required=True, type=str, help="Output pkl path.")
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOT_XML_DICT.keys()),
        default=None,
        help="Only required when ENTRY_POSE_SOURCE/EXIT_POSE_SOURCE uses robot_qpos0.",
    )
    parser.add_argument("--pose_hold_sec", type=float, default=2.0)
    parser.add_argument("--pose_transition_sec", type=float, default=0.4)
    parser.add_argument(
        "--root_rot_format",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion format stored in pkl root_rot.",
    )
    args = parser.parse_args()

    with open(args.input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos = np.asarray(motion_data["root_pos"])
    root_rot = np.asarray(motion_data["root_rot"])
    dof_pos = np.asarray(motion_data["dof_pos"])
    fps = float(motion_data["fps"])

    if not (len(root_pos) == len(root_rot) == len(dof_pos)):
        raise ValueError("root_pos/root_rot/dof_pos frame counts do not match.")
    if len(root_pos) == 0:
        raise ValueError("Input motion has zero frames.")

    if args.root_rot_format == "xyzw":
        root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]
    else:
        root_rot_wxyz = root_rot

    qpos_seq = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)

    entry_pose = _resolve_pose(ENTRY_POSE_SOURCE, qpos_seq, args.robot)
    exit_pose = _resolve_pose(EXIT_POSE_SOURCE, qpos_seq, args.robot)

    entry_pose[:3] = qpos_seq[0, :3]
    exit_pose[:3] = qpos_seq[-1, :3]
    hold_frames = max(0, int(round(float(args.pose_hold_sec) * fps)))
    transition_frames = max(0, int(round(float(args.pose_transition_sec) * fps)))
    qpos_out = _add_entry_exit_pose(
        qpos_seq=qpos_seq,
        start_pose=entry_pose,
        end_pose=exit_pose,
        hold_frames=hold_frames,
        transition_frames=transition_frames,
    )

    motion_out = dict(motion_data)
    motion_out["root_pos"] = qpos_out[:, :3]
    root_rot_out_wxyz = qpos_out[:, 3:7]
    if args.root_rot_format == "xyzw":
        motion_out["root_rot"] = root_rot_out_wxyz[:, [1, 2, 3, 0]]
    else:
        motion_out["root_rot"] = root_rot_out_wxyz
    motion_out["dof_pos"] = qpos_out[:, 7:]
    motion_out["fps"] = fps

    output_path = pathlib.Path(args.output_pkl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(motion_out, f)

    print(
        f"Saved {output_path} | in_frames={len(qpos_seq)} out_frames={len(qpos_out)} "
        f"hold_frames={hold_frames} transition_frames={transition_frames}"
    )


if __name__ == "__main__":
    main()
