import argparse
import pathlib
import pickle

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

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


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


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


class ExitIKSolver:
    """Small MuJoCo Jacobian-based solver for exit transition foot locking.

    We reuse MuJoCo's built-in Jacobians (`mj_jacBody`) and `mj_integratePos`
    to avoid introducing an external IK stack for this postprocess script.
    """

    def __init__(
        self,
        robot,
        link_names,
        iters,
        damping,
        foot_weight,
        pose_weight,
        step_weight,
        tol,
        lock_root_xyz=True,
        preserve_foot_orientation=False,
    ):
        if robot not in ROBOT_XML_DICT:
            raise ValueError(f"Unknown robot '{robot}'.")
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
        self.data = mj.MjData(self.model)
        self.nv = self.model.nv
        self.nq = self.model.nq
        self.link_names = list(link_names)
        self.link_ids = []
        for name in self.link_names:
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in robot model: {ROBOT_XML_DICT[robot]}")
            self.link_ids.append(bid)

        self.iters = int(iters)
        self.damping = float(damping)
        self.foot_weight = float(foot_weight)
        self.pose_weight = float(pose_weight)
        self.step_weight = float(step_weight)
        self.tol = float(tol)
        self.lock_root_xyz = bool(lock_root_xyz)
        self.preserve_foot_orientation = bool(preserve_foot_orientation)

        # Build posture task map for 1-DoF non-root joints.
        self.pose_qpos_ids = []
        self.pose_dof_ids = []
        for j in range(1, self.model.njnt):
            jtype = int(self.model.jnt_type[j])
            qadr = int(self.model.jnt_qposadr[j])
            dadr = int(self.model.jnt_dofadr[j])
            if jtype in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
                self.pose_qpos_ids.append(qadr)
                self.pose_dof_ids.append(dadr)
        self.pose_qpos_ids = np.asarray(self.pose_qpos_ids, dtype=np.int32)
        self.pose_dof_ids = np.asarray(self.pose_dof_ids, dtype=np.int32)

        self.dv_mask = np.ones(self.nv, dtype=float)
        if self.lock_root_xyz:
            self.dv_mask[:3] = 0.0

    def _forward(self, qpos):
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)

    def _foot_positions(self):
        return np.vstack([self.data.xpos[bid].copy() for bid in self.link_ids])

    def _foot_rotations(self):
        return [R.from_matrix(self.data.xmat[bid].reshape(3, 3)) for bid in self.link_ids]

    def solve(self, q_init, q_pose_target, foot_pos_targets, foot_rot_targets=None, frame_idx=-1):
        q = q_init.copy()
        q_prev = q_init.copy()
        success = False
        final_pos_err_norm = None

        for it in range(self.iters):
            self._forward(q)

            # Foot position residuals/Jacobians.
            foot_res = []
            foot_jacs = []
            for k, bid in enumerate(self.link_ids):
                cur = self.data.xpos[bid].copy()
                err = cur - foot_pos_targets[k]
                jacp = np.zeros((3, self.nv), dtype=float)
                jacr = np.zeros((3, self.nv), dtype=float)
                mj.mj_jacBody(self.model, self.data, jacp, jacr, bid)
                foot_res.append(err)
                foot_jacs.append(jacp)

            # Optional foot orientation residuals.
            ori_res = []
            ori_jacs = []
            if self.preserve_foot_orientation and foot_rot_targets is not None:
                for k, bid in enumerate(self.link_ids):
                    r_cur = R.from_matrix(self.data.xmat[bid].reshape(3, 3))
                    r_err = foot_rot_targets[k].inv() * r_cur
                    ori_res.append(r_err.as_rotvec())
                    jacp = np.zeros((3, self.nv), dtype=float)
                    jacr = np.zeros((3, self.nv), dtype=float)
                    mj.mj_jacBody(self.model, self.data, jacp, jacr, bid)
                    ori_jacs.append(jacr)

            # Posture residual/Jacobian on non-root 1-DoF joints.
            pose_res = q[self.pose_qpos_ids] - q_pose_target[self.pose_qpos_ids]
            J_pose = np.zeros((len(self.pose_dof_ids), self.nv), dtype=float)
            if len(self.pose_dof_ids) > 0:
                J_pose[np.arange(len(self.pose_dof_ids)), self.pose_dof_ids] = 1.0

            rows = []
            rhs = []

            wf = np.sqrt(self.foot_weight)
            for jacp, err in zip(foot_jacs, foot_res):
                rows.append(wf * jacp)
                rhs.append(wf * err)

            if len(ori_res) > 0:
                wfo = np.sqrt(self.foot_weight)
                for jacr, err in zip(ori_jacs, ori_res):
                    rows.append(wfo * jacr)
                    rhs.append(wfo * err)

            if len(self.pose_dof_ids) > 0 and self.pose_weight > 0:
                wp = np.sqrt(self.pose_weight)
                rows.append(wp * J_pose)
                rhs.append(wp * pose_res)

            # Small-step regularization on dv.
            if self.step_weight > 0:
                ws = np.sqrt(self.step_weight)
                rows.append(ws * np.eye(self.nv, dtype=float))
                rhs.append(np.zeros(self.nv, dtype=float))

            A = np.vstack(rows)
            b = np.concatenate(rhs)

            # Mask out locked DoFs (root xyz when requested).
            A = A * self.dv_mask[None, :]

            ATA = A.T @ A
            g = A.T @ b
            ATA = ATA + (self.damping ** 2) * np.eye(self.nv, dtype=float)

            try:
                dv = -np.linalg.solve(ATA, g)
            except np.linalg.LinAlgError:
                dv = -np.linalg.pinv(ATA) @ g
            dv = dv * self.dv_mask

            q_new = q.copy()
            mj.mj_integratePos(self.model, q_new, dv, 1.0)
            if self.lock_root_xyz:
                q_new[:3] = q_init[:3]
            q_prev = q
            q = q_new

            pos_err_norm = float(np.linalg.norm(np.concatenate(foot_res)))
            step_norm = float(np.linalg.norm(dv))
            final_pos_err_norm = pos_err_norm
            print(
                f"[exit_ik] frame={frame_idx} iter={it + 1}/{self.iters} "
                f"foot_pos_err={pos_err_norm:.6e} step_norm={step_norm:.6e}"
            )
            if pos_err_norm < self.tol:
                success = True
                break

            if step_norm < 1e-10 and pos_err_norm < 10.0 * self.tol:
                success = True
                break

        self._forward(q)
        return q, success, final_pos_err_norm


def _build_exit_transition_with_ik(start_q, end_q, transition_frames, ik_solver):
    if transition_frames <= 0:
        return [], {}

    ik_solver._forward(start_q)
    foot_pos_target = ik_solver._foot_positions()
    foot_rot_target = ik_solver._foot_rotations() if ik_solver.preserve_foot_orientation else None
    print("[exit_ik] Last original-frame foot anchors (world xyz):")
    for name, p in zip(ik_solver.link_names, foot_pos_target):
        print(f"  - {name}: [{p[0]: .6f}, {p[1]: .6f}, {p[2]: .6f}]")

    out = []
    q_prev = start_q.copy()
    drift_norms = []
    converged_count = 0

    for i in range(transition_frames):
        alpha = (i + 1) / float(transition_frames + 1)
        desired_q = _blend_qpos(start_q, end_q, alpha=alpha)
        q_solved, success, _ = ik_solver.solve(
            q_init=q_prev,
            q_pose_target=desired_q,
            foot_pos_targets=foot_pos_target,
            foot_rot_targets=foot_rot_target,
            frame_idx=i,
        )
        if success:
            converged_count += 1
        else:
            print(f"[exit_ik][warn] frame={i}: solver did not fully converge to tol={ik_solver.tol}")

        ik_solver._forward(q_solved)
        cur_foot_pos = ik_solver._foot_positions()
        errs = np.linalg.norm(cur_foot_pos - foot_pos_target, axis=1)
        drift_norms.extend(errs.tolist())
        print(
            f"[exit_ik] frame={i}: foot_xyz_error="
            + ", ".join(f"{name}={err:.6e}m" for name, err in zip(ik_solver.link_names, errs))
        )

        out.append(q_solved.copy())
        q_prev = q_solved

    stats = {
        "converged_frames": converged_count,
        "total_frames": transition_frames,
        "max_drift_m": float(np.max(drift_norms)) if drift_norms else 0.0,
        "mean_drift_m": float(np.mean(drift_norms)) if drift_norms else 0.0,
    }
    print(
        f"[exit_ik] transition summary: converged={stats['converged_frames']}/{stats['total_frames']} "
        f"max_drift={stats['max_drift_m']:.6e}m mean_drift={stats['mean_drift_m']:.6e}m"
    )
    return out, stats


def _add_entry_exit_pose(
    qpos_seq,
    start_pose,
    end_pose,
    hold_frames,
    transition_frames,
    exit_ik_solver=None,
):
    if len(qpos_seq) == 0:
        return qpos_seq

    parts = []
    if hold_frames > 0:
        parts.append(np.repeat(start_pose[None, :], hold_frames, axis=0))

    trans_in = _build_transition(start_pose, qpos_seq[0], transition_frames)
    if trans_in:
        parts.append(np.asarray(trans_in))

    parts.append(qpos_seq)

    if exit_ik_solver is None:
        trans_out = _build_transition(qpos_seq[-1], end_pose, transition_frames)
    else:
        trans_out, _ = _build_exit_transition_with_ik(
            start_q=qpos_seq[-1],
            end_q=end_pose,
            transition_frames=transition_frames,
            ik_solver=exit_ik_solver,
        )
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
    parser.add_argument("--exit_ik_enable", type=_str2bool, default=False)
    parser.add_argument(
        "--exit_ik_link_names",
        nargs="+",
        default=["left_ankle_roll_link", "right_ankle_roll_link"],
        help="Body names used as planted-foot anchors for exit transition.",
    )
    parser.add_argument("--exit_ik_iters", type=int, default=40)
    parser.add_argument("--exit_ik_damping", type=float, default=1e-3)
    parser.add_argument("--exit_ik_foot_weight", type=float, default=1000.0)
    parser.add_argument("--exit_ik_pose_weight", type=float, default=5.0)
    parser.add_argument("--exit_ik_step_weight", type=float, default=1e-2)
    parser.add_argument("--exit_ik_tol", type=float, default=1e-4)
    parser.add_argument("--exit_ik_lock_root_xyz", type=_str2bool, default=True)
    parser.add_argument("--exit_ik_preserve_foot_orientation", type=_str2bool, default=False)
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
    exit_ik_solver = None
    if args.exit_ik_enable:
        if not args.robot:
            raise ValueError("--exit_ik_enable requires --robot so MuJoCo model can be loaded.")
        exit_ik_solver = ExitIKSolver(
            robot=args.robot,
            link_names=args.exit_ik_link_names,
            iters=args.exit_ik_iters,
            damping=args.exit_ik_damping,
            foot_weight=args.exit_ik_foot_weight,
            pose_weight=args.exit_ik_pose_weight,
            step_weight=args.exit_ik_step_weight,
            tol=args.exit_ik_tol,
            lock_root_xyz=args.exit_ik_lock_root_xyz,
            preserve_foot_orientation=args.exit_ik_preserve_foot_orientation,
        )
        print(
            f"[exit_ik] enabled with links={args.exit_ik_link_names}, iters={args.exit_ik_iters}, "
            f"damping={args.exit_ik_damping}, w_foot={args.exit_ik_foot_weight}, "
            f"w_pose={args.exit_ik_pose_weight}, w_step={args.exit_ik_step_weight}, "
            f"tol={args.exit_ik_tol}, lock_root_xyz={args.exit_ik_lock_root_xyz}, "
            f"preserve_foot_orientation={args.exit_ik_preserve_foot_orientation}"
        )

    qpos_out = _add_entry_exit_pose(
        qpos_seq=qpos_seq,
        start_pose=entry_pose,
        end_pose=exit_pose,
        hold_frames=hold_frames,
        transition_frames=transition_frames,
        exit_ik_solver=exit_ik_solver,
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
