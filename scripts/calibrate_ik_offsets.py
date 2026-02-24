import argparse
import copy
import importlib.util
import json
import pathlib
import pickle
import sys
from typing import Dict, Iterable, Optional, Tuple, List

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "gmr_params", ROOT / "general_motion_retargeting" / "params.py"
)
params = importlib.util.module_from_spec(spec)
spec.loader.exec_module(params)
ROBOT_XML_DICT = params.ROBOT_XML_DICT

smpl_spec = importlib.util.spec_from_file_location(
    "gmr_smpl_utils", ROOT / "general_motion_retargeting" / "utils" / "smpl.py"
)
smpl_module = importlib.util.module_from_spec(smpl_spec)
smpl_spec.loader.exec_module(smpl_module)
get_smplx_data = smpl_module.get_smplx_data

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
import smplx
import torch
from general_motion_retargeting.ik_config_utils import parse_rot_offset, format_rot_offset_like


def scale_human_data(
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    human_root_name: str,
    human_scale_table: Dict[str, float],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Replicate the scaling logic used during retargeting."""
    root_pos, root_quat = human_data[human_root_name]
    scaled_root_pos = human_scale_table.get(human_root_name, 1.0) * root_pos

    scaled = {human_root_name: (scaled_root_pos, root_quat)}
    for body_name, (pos, quat) in human_data.items():
        if body_name == human_root_name:
            continue
        scale = human_scale_table.get(body_name, None)
        if scale is None:
            scaled[body_name] = (pos, quat)
            continue
        local = (pos - root_pos) * scale
        scaled[body_name] = (local + scaled_root_pos, quat)
    for body_name, value in human_data.items():
        scaled.setdefault(body_name, value)
    return scaled


def prepare_betas(
    body_model: smplx.SMPLXLayer,
    betas: Optional[Iterable[float]],
    model_file: Optional[pathlib.Path] = None,
) -> torch.Tensor:
    """Return a 1xN tensor of betas pulled from CLI, model file, or zeros."""
    num_betas = int(getattr(body_model, "num_betas", 10))
    if betas is not None:
        betas_list = list(betas)
        if len(betas_list) < num_betas:
            betas_list = betas_list + [0.0] * (num_betas - len(betas_list))
        betas_arr = np.asarray(betas_list[:num_betas], dtype=np.float32)
        return torch.from_numpy(betas_arr).view(1, -1)

    if model_file is not None and model_file.suffix in {".pkl", ".p"}:
        try:
            with model_file.open("rb") as f:
                model_data = pickle.load(f, encoding="latin1")
            for key in ("betas", "betas_mean", "shape_mean"):
                if key in model_data:
                    betas_arr = np.asarray(model_data[key]).reshape(-1)
                    if betas_arr.size < num_betas:
                        betas_arr = np.pad(
                            betas_arr,
                            (0, num_betas - betas_arr.size),
                            mode="constant",
                        )
                    return torch.from_numpy(betas_arr[:num_betas]).float().view(1, -1)
        except Exception:
            pass

    return torch.zeros((1, num_betas), dtype=torch.float32)


def load_static_smplx_pose(
    model_file: pathlib.Path,
    gender: str = "neutral",
    betas: Optional[Iterable[float]] = None,
) -> Tuple[dict, smplx.SMPLXLayer, torch.Tensor, float]:
    """Create a single-frame SMPL-X output using a static body model file."""
    model_file = model_file.resolve()
    if not model_file.exists():
        raise FileNotFoundError(f"SMPL-X model file not found: {model_file}")

    base_path = model_file.parent
    if not (base_path / "smplx").exists():
        parent_candidate = base_path.parent
        if (parent_candidate / "smplx").exists():
            base_path = parent_candidate

    ext = model_file.suffix.lstrip(".")
    body_model = smplx.create(
        model_path=str(base_path),
        model_type="smplx",
        gender=gender,
        ext=ext,
        use_pca=False,
    )

    betas_tensor = prepare_betas(body_model, betas, model_file=model_file)
    num_frames = 1

    def zeros(dim: int) -> torch.Tensor:
        return torch.zeros((num_frames, dim), dtype=torch.float32)

    smplx_output = body_model(
        betas=betas_tensor,
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

    smplx_data = {
        "betas": betas_tensor.detach().cpu().numpy(),
        "gender": np.array(gender),
    }

    betas_np = smplx_data["betas"]
    human_height = 1.66 + 0.1 * betas_np.reshape(-1)[0] if betas_np.size else 1.66

    return smplx_data, body_model, smplx_output, float(human_height)


def get_robot_body_pose(
    model: mj.MjModel, data: mj.MjData, body_name: str
) -> Tuple[np.ndarray, R]:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in robot model.")
    pos = data.xpos[body_id].copy()
    rot_mat = data.xmat[body_id].reshape(3, 3)
    rot = R.from_matrix(rot_mat)
    return pos, rot


def compute_local_offset(
    human_pos: np.ndarray,
    robot_pos: np.ndarray,
    robot_rot: R,
) -> np.ndarray:
    """Return local (body-frame) offset so that transformed human matches robot."""
    global_delta = robot_pos - human_pos
    # Apply inverse rotation of target so we store offset in body frame
    return robot_rot.inv().apply(global_delta)


def compute_rot_offset(human_rot: R, robot_rot: R) -> R:
    """Rotation multiplier applied on the right to align human orientation to robot."""
    return human_rot.inv() * robot_rot


def parse_target_identifier(target: str) -> Tuple[str, int]:
    """Parse identifiers like 'frame_name' or 'frame_name@2' into (frame, table_index)."""
    if "@2" in target:
        frame, _ = target.split("@", 1)
        return frame, 2
    if "@1" in target:
        frame, _ = target.split("@", 1)
        return frame, 1
    return target, 1


def get_table(ik_config: dict, table_index: int) -> Dict[str, list]:
    key = "ik_match_table1" if table_index == 1 else "ik_match_table2"
    return ik_config.setdefault(key, {})


def apply_manual_edits(
    ik_config: dict,
    targets: List[str],
    pos_offset: Optional[Iterable[float]],
    rot_quat: Optional[Iterable[float]],
    rot_euler: Optional[Iterable[float]],
    euler_in_degrees: bool,
    copy_from: Optional[str],
) -> List[str]:
    """Update specified entries with manual offsets. Returns list of modified keys."""
    if not targets and not copy_from:
        return []

    if copy_from and not targets:
        raise ValueError("Specify at least one --edit target when using --copy_from.")

    updated_entries = []

    source_offsets = None
    if copy_from:
        src_frame, src_table_idx = parse_target_identifier(copy_from)
        src_table = get_table(ik_config, src_table_idx)
        if src_frame not in src_table:
            raise KeyError(f"Source frame '{copy_from}' not found in IK config.")
        src_entry = src_table[src_frame]
        source_offsets = (list(src_entry[3]), copy.deepcopy(src_entry[4]))

    if rot_quat is not None and rot_euler is not None:
        raise ValueError("Specify either --set_quat or --set_euler, not both.")

    quat_from_euler = None
    if rot_euler is not None:
        quat_from_euler = R.from_euler(
            "xyz", rot_euler, degrees=euler_in_degrees
        ).as_quat(scalar_first=True)

    for target in targets:
        frame_name, table_idx = parse_target_identifier(target)
        table = get_table(ik_config, table_idx)
        if frame_name not in table:
            print(f"[manual] Skip '{target}': frame not found.")
            continue
        entry = table[frame_name]

        if source_offsets is not None and (targets or (copy_from and not targets)):
            entry[3] = list(source_offsets[0])
            entry[4] = copy.deepcopy(source_offsets[1])

        if pos_offset is not None:
            entry[3] = [float(x) for x in pos_offset]

        if rot_quat is not None:
            entry[4] = [float(x) for x in rot_quat]
        elif quat_from_euler is not None:
            # Keep manual CLI behavior stable: store as quaternion.
            entry[4] = [float(x) for x in quat_from_euler]

        updated_entries.append(f"{frame_name}@{table_idx}")

    return updated_entries


def update_table(
    table: Dict[str, list],
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    robot_model: mj.MjModel,
    robot_data: mj.MjData,
    human_missing: set,
    robot_missing: set,
    debug_points: Optional[list] = None,
) -> None:
    for frame_name, entry in table.items():
        human_body = entry[0]
        if human_body not in human_data:
            human_missing.add(human_body)
            continue
        try:
            robot_pos, robot_rot = get_robot_body_pose(robot_model, robot_data, frame_name)
        except ValueError:
            robot_missing.add(frame_name)
            continue

        human_pos, human_quat = human_data[human_body]
        human_rot = R.from_quat(human_quat, scalar_first=True)

        rot_offset = compute_rot_offset(human_rot, robot_rot)
        local_offset = compute_local_offset(human_pos, robot_pos, robot_rot)

        entry[3] = [float(x) for x in local_offset.tolist()]
        entry[4] = format_rot_offset_like(entry[4], rot_offset)
        if debug_points is not None:
            debug_points.append(
                {
                    "frame": frame_name,
                    "human_body": human_body,
                    "robot_pos": robot_pos.copy(),
                    "human_pos": human_pos.copy(),
                }
            )


def visualize_alignment(debug_points: list, vis_path: Optional[pathlib.Path], show: bool) -> None:
    if not debug_points:
        print("[calibrate] No points available for visualization.")
        return
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        print(f"[calibrate] Visualization skipped: matplotlib not available ({exc}).")
        return

    robot_points = np.array([item["robot_pos"] for item in debug_points])
    human_points = np.array([item["human_pos"] for item in debug_points])
    labels = [item["frame"] for item in debug_points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Robot vs Human Alignment (Rest Pose)")
    ax.scatter(
        robot_points[:, 0],
        robot_points[:, 1],
        robot_points[:, 2],
        c="tab:blue",
        label="Robot",
        s=40,
    )
    ax.scatter(
        human_points[:, 0],
        human_points[:, 1],
        human_points[:, 2],
        c="tab:orange",
        label="Human",
        s=40,
        marker="^",
    )

    for idx, label in enumerate(labels):
        ax.plot(
            [robot_points[idx, 0], human_points[idx, 0]],
            [robot_points[idx, 1], human_points[idx, 1]],
            [robot_points[idx, 2], human_points[idx, 2]],
            c="gray",
            linewidth=0.5,
        )
        ax.text(robot_points[idx, 0], robot_points[idx, 1], robot_points[idx, 2], label, fontsize=6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.view_init(elev=20, azim=135)
    plt.tight_layout()

    if vis_path:
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(vis_path, dpi=200)
        print(f"[calibrate] Visualization saved to {vis_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def collect_alignment_points(
    ik_config: dict,
    human_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    robot_model: mj.MjModel,
    robot_data: mj.MjData,
    apply_offsets: bool = True,
) -> list:
    points = []
    for table_idx, table_key in enumerate(["ik_match_table1", "ik_match_table2"], start=1):
        table = ik_config.get(table_key, {})
        for frame_name, entry in table.items():
            human_body = entry[0]
            if human_body not in human_data:
                continue
            try:
                robot_pos, _ = get_robot_body_pose(robot_model, robot_data, frame_name)
            except ValueError:
                continue
            human_pos, human_quat = human_data[human_body]
            human_rot = R.from_quat(human_quat, scalar_first=True)
            if apply_offsets:
                rot_offset = parse_rot_offset(entry[4])
                updated_rot = human_rot * rot_offset
                pos_offset = np.array(entry[3])
                human_plot_pos = human_pos + updated_rot.apply(pos_offset)
            else:
                human_plot_pos = human_pos
            points.append(
                {
                    "frame": f"{frame_name}@{table_idx}",
                    "robot_pos": robot_pos.copy(),
                    "human_pos": human_plot_pos.copy(),
                }
            )
    return points


def print_table_entries(ik_config: dict, targets: Optional[List[str]] = None) -> None:
    target_set = set()
    if targets:
        for target in targets:
            frame_name, table_idx = parse_target_identifier(target)
            target_set.add((frame_name, table_idx))

    for table_idx, table_key in enumerate(["ik_match_table1", "ik_match_table2"], start=1):
        table = ik_config.get(table_key, {})
        if not table:
            continue
        print(f"[{table_key}]")
        for frame_name, entry in table.items():
            if target_set and (frame_name, table_idx) not in target_set:
                continue
            pos_str = ", ".join(f"{v:.5f}" for v in entry[3])
            if isinstance(entry[4], dict):
                rot_str = json.dumps(entry[4])
            else:
                rot_str = ", ".join(f"{v:.5f}" for v in entry[4])
            print(
                f"  {frame_name}: human={entry[0]}, pos=[{pos_str}], rot=[{rot_str}], "
                f"weights=({entry[1]}, {entry[2]})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-calibrate IK config offsets from a reference SMPL-X pose."
    )
    parser.add_argument(
        "--template",
        type=pathlib.Path,
        required=True,
        help="Path to the IK config JSON template containing mapping tables.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="Robot key present in general_motion_retargeting.params.ROBOT_XML_DICT.",
    )
    parser.add_argument(
        "--smplx_model",
        type=pathlib.Path,
        default=None,
        help="Static SMPL-X body model file (npz/pkl) used to synthesize a T-pose when pose data is required.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["male", "female", "neutral"],
        help="Gender flag for static SMPL-X body model loading.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=None,
        help="Optional shape coefficients when using a static SMPL-X body model.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run automatic offset calibration based on the reference pose.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print IK table entries and exit (after applying edits if any).",
    )
    parser.add_argument(
        "--edit",
        action="append",
        default=[],
        help="Frame identifier to edit (e.g., pelvis or pelvis@2 for table2). Repeatable.",
    )
    parser.add_argument(
        "--set_pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Assign position offset (local frame) to selected entries.",
    )
    parser.add_argument(
        "--set_quat",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Assign rotation offset quaternion (wxyz) to selected entries.",
    )
    parser.add_argument(
        "--set_euler",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        help="Assign rotation offset using XYZ Euler angles.",
    )
    parser.add_argument(
        "--euler_deg",
        action="store_true",
        help="Interpret --set_euler values in degrees (default radians).",
    )
    parser.add_argument(
        "--copy_from",
        type=str,
        default=None,
        help="Copy offsets from another frame (format frame or frame@2). Requires --edit.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Display a 3D scatter plot of human vs robot rest-pose alignment.",
    )
    parser.add_argument(
        "--vis_path",
        type=pathlib.Path,
        default=None,
        help="Optional path to save the alignment visualization as an image.",
    )
    parser.add_argument(
        "--vis_raw",
        action="store_true",
        help="Visualize raw SMPL joints (ignore configured offsets).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional output JSON path. If omitted, overwrite the input config.",
    )
    args = parser.parse_args()

    template_path = args.template.resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    robot_xml_path = ROBOT_XML_DICT.get(args.robot)
    if robot_xml_path is None:
        raise KeyError(f"Robot '{args.robot}' not defined in ROBOT_XML_DICT.")

    robot_xml_path = robot_xml_path.resolve()
    if not robot_xml_path.exists():
        raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")

    with template_path.open("r") as f:
        ik_config = json.load(f)

    need_pose = args.auto or args.vis or args.vis_path
    human_data = None
    human_height = ik_config.get("human_height_assumption", 1.8)
    if need_pose:
        if args.smplx_model is None:
            raise ValueError("Please provide --smplx_model when using --auto/--vis options.")
        if not args.smplx_model.exists():
            raise FileNotFoundError(f"SMPL-X body model file not found: {args.smplx_model}")
        smplx_data, body_model, smplx_output, human_height = load_static_smplx_pose(
            args.smplx_model, gender=args.gender, betas=args.betas
        )
        frame_idx = 0
        human_data = get_smplx_data(smplx_data, body_model, smplx_output, frame_idx)
        human_data = scale_human_data(
            human_data,
            ik_config["human_root_name"],
            ik_config["human_scale_table"],
        )

    model = mj.MjModel.from_xml_path(str(robot_xml_path))
    model_data = mj.MjData(model)
    mj.mj_forward(model, model_data)

    missing_human = set()
    missing_robot = set()
    config_changed = False

    auto_debug_points = []
    if args.auto:
        debug_points = [] if (args.vis or args.vis_path is not None) else None
        update_table(
            ik_config["ik_match_table1"],
            human_data,
            model,
            model_data,
            missing_human,
            missing_robot,
            debug_points,
        )
        update_table(
            ik_config["ik_match_table2"],
            human_data,
            model,
            model_data,
            missing_human,
            missing_robot,
            debug_points,
        )
        config_changed = True
        if debug_points is not None:
            auto_debug_points = debug_points

    manual_updates = []
    if (args.set_pos is not None or args.set_quat is not None or args.set_euler is not None or args.copy_from) and not args.edit:
        print("[manual] Please specify --edit targets when using --set_pos/--set_quat/--set_euler/--copy_from.")
        sys.exit(1)
    try:
        manual_updates = apply_manual_edits(
            ik_config,
            args.edit,
            args.set_pos,
            args.set_quat,
            args.set_euler,
            args.euler_deg,
            args.copy_from,
        )
    except (ValueError, KeyError) as exc:
        print(f"[manual] {exc}")
        sys.exit(1)

    if manual_updates:
        config_changed = True
        print(f"[manual] Updated entries: {', '.join(manual_updates)}")

    if args.auto:
        ik_config["human_height_assumption"] = float(human_height)

    if args.list:
        target_print = args.edit if args.edit else None
        print_table_entries(ik_config, target_print)
        if not (config_changed or args.auto or args.output or args.vis or args.vis_path):
            sys.exit(0)

    if args.vis or args.vis_path:
        vis_points = collect_alignment_points(
            ik_config,
            human_data,
            model,
            model_data,
            apply_offsets=not args.vis_raw,
        )
        if not vis_points and auto_debug_points:
            vis_points = auto_debug_points
        vis_path = args.vis_path.resolve() if args.vis_path else None
        visualize_alignment(vis_points, vis_path, args.vis)

    if config_changed or args.output:
        if args.output:
            output_path = args.output.resolve()
        else:
            default_name = template_path.stem + "_manual.json"
            output_path = (template_path.parent / default_name).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(ik_config, f, indent=4)
        print(f"[calibrate] Updated offsets written to {output_path}")

    if args.auto:
        if missing_human:
            print(f"[calibrate] Warning: missing human bodies: {sorted(missing_human)}")
        if missing_robot:
            print(f"[calibrate] Warning: missing robot frames: {sorted(missing_robot)}")
