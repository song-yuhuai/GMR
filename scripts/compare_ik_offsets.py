import argparse
import json
import pathlib
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data
from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
from general_motion_retargeting import GeneralMotionRetargeting


def load_static_model(model_file: pathlib.Path):
    import smplx
    import torch

    model_file = model_file.resolve()
    base = model_file.parent if (model_file.parent / "smplx").exists() else model_file.parent.parent
    body_model = smplx.create(
        model_path=str(base),
        model_type="smplx",
        gender="neutral",
        ext=model_file.suffix.lstrip("."),
        use_pca=False,
    )
    num_frames = 1
    zeros = lambda d: torch.zeros((num_frames, d), dtype=torch.float32)
    betas = torch.zeros((1, getattr(body_model, "num_betas", 10)))
    output = body_model(
        betas=betas,
        global_orient=zeros(3),
        body_pose=zeros(body_model.NUM_BODY_JOINTS * 3),
        transl=zeros(3),
        left_hand_pose=zeros(45),
        right_hand_pose=zeros(45),
        jaw_pose=zeros(3),
        leye_pose=zeros(3),
        reye_pose=zeros(3),
        expression=zeros(10),
        return_full_pose=True,
    )
    smplx_data = {"betas": betas.numpy(), "gender": np.array("neutral")}
    return smplx_data, body_model, output


def robot_body_pose(name, model_obj, data_obj):
    body_id = mj.mj_name2id(model_obj, mj.mjtObj.mjOBJ_BODY, name)
    pos = data_obj.xpos[body_id]
    mat = data_obj.xmat[body_id].reshape(3, 3)
    return pos, R.from_matrix(mat)


def load_motion_frames(motion_path: pathlib.Path, start: int, num_frames: int, frame_step: int):
    if motion_path.suffix.lower() == ".bvh":
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames_all, _ = load_bvh_file(str(motion_path), format="lafan1")
        n = len(frames_all)
        idxs = list(range(max(0, start), n, max(1, frame_step)))
        if num_frames > 0:
            idxs = idxs[:num_frames]
        frames = [frames_all[i] for i in idxs]
        return frames, idxs, "bvh_lafan1"

    if motion_path.suffix.lower() == ".pkl":
        smplx_data, body_model, smplx_output = load_static_model(motion_path)
    else:
        smplx_data, body_model, smplx_output, _ = load_smplx_file(str(motion_path), ROOT / "assets" / "body_models")
    n = int(smplx_output.vertices.shape[0])
    idxs = list(range(max(0, start), n, max(1, frame_step)))
    if num_frames > 0:
        idxs = idxs[:num_frames]
    frames = [get_smplx_data(smplx_data, body_model, smplx_output, i) for i in idxs]
    return frames, idxs, "smplx"


def collect_points_and_print(ik, model, data, human, target_override=None, print_prefix=""):
    points = []
    residual_norms = []
    rot_errs_deg = []
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = ik.get(table_name, {})
        if not table:
            continue
        print(f"\n{print_prefix}[{table_name}]")
        for frame_name, entry in table.items():
            human_body, pos_weight, rot_weight, pos_off, rot_off = entry

            if target_override is not None:
                if human_body not in target_override:
                    print(f"  {frame_name:25s} -> {human_body:20s} (missing in retarget targets)")
                    continue
                target_pos, target_rot = target_override[human_body]
            else:
                if human_body not in human:
                    print(f"  {frame_name:25s} -> {human_body:20s} (missing in motion data)")
                    continue
                human_pos, human_quat = human[human_body]
                target_rot = R.from_quat(np.array(human_quat, dtype=float), scalar_first=True)
                target_pos = np.array(human_pos, dtype=float)

            robot_pos, robot_rot = robot_body_pose(frame_name, model, data)

            diff = robot_pos - target_pos
            rot_err = target_rot.inv() * robot_rot
            rot_err_deg = float(np.degrees(rot_err.magnitude()))
            rot_err_euler_deg = rot_err.as_euler("xyz", degrees=True)
            print(f"  {frame_name:25s} ↔ {human_body:20s}  target={target_pos} robot={robot_pos} residual={diff}")
            if float(rot_weight) > 0:
                print(
                    f"    orientation residual: angle={rot_err_deg:.3f} deg, "
                    f"euler_xyz_deg={np.array2string(rot_err_euler_deg, precision=3)}"
                )
            else:
                print(
                    f"    orientation residual: angle={rot_err_deg:.3f} deg "
                    f"(ignored, rot_weight=0)"
                )
            res_norm = float(np.linalg.norm(diff))
            print(f"    residual norm: {res_norm:.6f}")
            residual_norms.append(res_norm)
            rot_errs_deg.append(rot_err_deg)
            points.append(
                (
                    frame_name,
                    robot_pos.copy(),
                    robot_rot,
                    target_pos.copy(),
                    target_rot,
                    diff.copy(),
                    rot_err_deg,
                    rot_err_euler_deg.copy(),
                )
            )
    return points, residual_norms, rot_errs_deg


def visualize_points(points, vis_path: pathlib.Path, show: bool, show_frames: bool):
    if not points:
        print("No points to visualize.")
        return
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        print(f"Visualization failed: matplotlib not available ({exc}).")
        return

    robot_pts = np.array([p[1] for p in points])
    robot_rots = [p[2] for p in points]
    human_pts = np.array([p[3] for p in points])
    human_rots = [p[4] for p in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Robot vs Human Alignment")
    ax.scatter(robot_pts[:, 0], robot_pts[:, 1], robot_pts[:, 2], c="tab:blue", label="Robot", s=40)
    ax.scatter(human_pts[:, 0], human_pts[:, 1], human_pts[:, 2], c="tab:orange", label="Human", s=40, marker="^")
    for (_, r, _, h, _, _, _, _) in points:
        ax.plot([r[0], h[0]], [r[1], h[1]], [r[2], h[2]], c="gray", linewidth=0.5)
    if show_frames:
        axis_len = 0.1
        axis_colors = ["red", "green", "blue"]
        for r_pos, r_rot in zip(robot_pts, robot_rots):
            mat = r_rot.as_matrix()
            for axis in range(3):
                ax.plot(
                    [r_pos[0], r_pos[0] + axis_len * mat[0, axis]],
                    [r_pos[1], r_pos[1] + axis_len * mat[1, axis]],
                    [r_pos[2], r_pos[2] + axis_len * mat[2, axis]],
                    color=axis_colors[axis],
                )
        for h_pos, h_rot in zip(human_pts, human_rots):
            mat = h_rot.as_matrix()
            for axis in range(3):
                ax.plot(
                    [h_pos[0], h_pos[0] + axis_len * mat[0, axis]],
                    [h_pos[1], h_pos[1] + axis_len * mat[1, axis]],
                    [h_pos[2], h_pos[2] + axis_len * mat[2, axis]],
                    color=axis_colors[axis],
                    linestyle="dashed",
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.view_init(elev=20, azim=135)
    all_pts = np.vstack((robot_pts, human_pts))
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    half = max_range / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    plt.tight_layout()

    if vis_path:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(vis_path, dpi=200)
        print(f"Saved figure to {vis_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="IK json, e.g. smplx_to_g1.json")
    parser.add_argument("--robot", required=True, help="Robot key, e.g. unitree_g1")
    parser.add_argument("--motion_file", required=True, help="Stage II SMPL-X npz/pkl or BVH file (lafan1)")
    parser.add_argument("--frame", type=int, default=0, help="Start frame / single frame index")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames for sequential evaluation")
    parser.add_argument("--frame_step", type=int, default=1, help="Frame stride for sequential evaluation")
    parser.add_argument("--trajectory_eval", action="store_true", help="Evaluate sequential retargeting consistency")
    parser.add_argument("--vis", action="store_true", help="Show 3D scatter of robot vs human points")
    parser.add_argument("--vis_path", type=pathlib.Path, default=None, help="Optional path to save figure")
    parser.add_argument("--show_frames", action="store_true", help="Overlay frame axes for robot and human")
    parser.add_argument("--after_ik", action="store_true", help="Run IK solve and compare robot vs retarget targets")
    parser.add_argument("--use_velocity_limit", action="store_true", help="Pass through to GMR for sequence mode")
    parser.add_argument("--joint_blend", type=float, default=0.0, help="Pass through to GMR for sequence mode")
    parser.add_argument("--joint_step_limit", type=float, default=0.0, help="Pass through to GMR for sequence mode")
    args = parser.parse_args()

    # load IK config
    with open(args.config) as f:
        ik = json.load(f)

    # load robot MJCF
    xml = ROBOT_XML_DICT[args.robot]
    model = mj.MjModel.from_xml_path(str(xml))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    motion_path = pathlib.Path(args.motion_file)
    frames, frame_indices, src_human = load_motion_frames(motion_path, args.frame, args.num_frames, args.frame_step)
    if len(frames) == 0:
        raise RuntimeError("No frames selected. Check --frame/--num_frames/--frame_step.")

    vis_path = args.vis_path.resolve() if args.vis_path else None
    eval_sequence = args.trajectory_eval or args.num_frames > 1

    if eval_sequence:
        # Sequential mode: use the same forward retarget dynamics as smplx_to_robot.
        IK_CONFIG_DICT[src_human][args.robot] = str(pathlib.Path(args.config).resolve())
        retarget = GeneralMotionRetargeting(
            src_human=src_human,
            tgt_robot=args.robot,
            verbose=False,
            use_velocity_limit=args.use_velocity_limit,
            joint_blend=args.joint_blend,
            joint_step_limit=args.joint_step_limit,
        )
        cfg = retarget.configuration
        cfg.data.qpos[:] = retarget.model.qpos0
        cfg.data.qvel[:] = 0
        mj.mj_forward(retarget.model, cfg.data)

        all_res = []
        all_rot = []
        prev_q = None
        max_joint_step = 0.0
        last_points = []
        for i, human in enumerate(frames):
            retarget.retarget(human)
            points, res, rot = collect_points_and_print(
                ik,
                retarget.model,
                retarget.configuration.data,
                human,
                target_override={
                    body_name: (
                        np.array(pos, dtype=float),
                        R.from_quat(np.array(quat, dtype=float), scalar_first=True),
                    )
                    for body_name, (pos, quat) in retarget.scaled_human_data.items()
                },
                print_prefix=f"[frame {frame_indices[i]}] ",
            )
            all_res.extend(res)
            all_rot.extend(rot)
            last_points = points
            q = retarget.configuration.data.qpos.copy()
            if prev_q is not None:
                step = float(np.max(np.abs(q[7:] - prev_q[7:])))
                max_joint_step = max(max_joint_step, step)
            prev_q = q

        if all_res:
            print("\n[sequence summary]")
            print(f"  frames: {len(frames)}")
            print(f"  residual norm mean/max: {np.mean(all_res):.6f}/{np.max(all_res):.6f}")
            print(f"  orientation err deg mean/max: {np.mean(all_rot):.3f}/{np.max(all_rot):.3f}")
            print(f"  max joint-step (abs delta q[7:]): {max_joint_step:.6f}")
        if args.vis or vis_path:
            visualize_points(last_points, vis_path, args.vis, args.show_frames)
        sys.exit(0)

    # Single-frame mode (original behavior)
    human = frames[0]
    target_override = None
    if args.after_ik:
        IK_CONFIG_DICT[src_human][args.robot] = str(pathlib.Path(args.config).resolve())
        retarget = GeneralMotionRetargeting(
            src_human=src_human,
            tgt_robot=args.robot,
            verbose=False,
            use_velocity_limit=args.use_velocity_limit,
            joint_blend=args.joint_blend,
            joint_step_limit=args.joint_step_limit,
        )
        cfg = retarget.configuration
        cfg.data.qpos[:] = retarget.model.qpos0
        cfg.data.qvel[:] = 0
        mj.mj_forward(retarget.model, cfg.data)
        retarget.retarget(human)
        model = retarget.model
        data = retarget.configuration.data
        target_override = {
            body_name: (
                np.array(pos, dtype=float),
                R.from_quat(np.array(quat, dtype=float), scalar_first=True),
            )
            for body_name, (pos, quat) in retarget.scaled_human_data.items()
        }

    points, _, _ = collect_points_and_print(
        ik,
        model,
        data,
        human,
        target_override=target_override,
    )
    if args.vis or vis_path:
        visualize_points(points, vis_path, args.vis, args.show_frames)
