import argparse
import pathlib
import os
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

from rich import print


def _run_framewise(retarget, frames):
    qpos_seq = []
    for frame in frames:
        qpos_seq.append(retarget.retarget(frame))
    return np.asarray(qpos_seq)


def _clip_joint_velocity(qpos_seq, fps, max_joint_vel):
    if max_joint_vel <= 0 or len(qpos_seq) <= 1:
        return qpos_seq
    q = qpos_seq.copy()
    max_step = float(max_joint_vel) / float(fps)
    for i in range(1, len(q)):
        dq = q[i, 7:] - q[i - 1, 7:]
        q[i, 7:] = q[i - 1, 7:] + np.clip(dq, -max_step, max_step)
    return q


def _smooth_dofs(qpos_seq, window, polyorder):
    if len(qpos_seq) < 5:
        return qpos_seq
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return qpos_seq

    q = qpos_seq.copy()
    win = int(window)
    if win % 2 == 0:
        win += 1
    win = min(win, len(q) if len(q) % 2 == 1 else len(q) - 1)
    if win < 5:
        return q

    poly = min(int(polyorder), win - 1)
    q[:, 7:] = savgol_filter(q[:, 7:], window_length=win, polyorder=poly, axis=0)
    return q


def _run_protomotions_like(
    retarget_ctor,
    frames,
    actual_human_height,
    src_human,
    tgt_robot,
    ik_config_path,
    use_velocity_limit,
    joint_blend,
    joint_step_limit,
    fps,
    max_joint_vel,
    smooth_window,
    smooth_polyorder,
):
    # Forward pass (current GMR behavior).
    retarget_fwd = retarget_ctor(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=tgt_robot,
        ik_config_path=ik_config_path,
        use_velocity_limit=use_velocity_limit,
        joint_blend=joint_blend,
        joint_step_limit=joint_step_limit,
        verbose=False,
    )
    q_fwd = _run_framewise(retarget_fwd, frames)

    # Backward pass to reduce one-way warm-start bias.
    retarget_bwd = retarget_ctor(
        actual_human_height=actual_human_height,
        src_human=src_human,
        tgt_robot=tgt_robot,
        ik_config_path=ik_config_path,
        use_velocity_limit=use_velocity_limit,
        joint_blend=joint_blend,
        joint_step_limit=joint_step_limit,
        verbose=False,
    )
    q_bwd = _run_framewise(retarget_bwd, list(reversed(frames)))[::-1]

    # Blend only joint DoFs.
    # Do not blend root quaternion directly, or it can create invalid rotations/spins.
    q = q_fwd.copy()
    q[:, 7:] = 0.5 * q_fwd[:, 7:] + 0.5 * q_bwd[:, 7:]
    q = _smooth_dofs(q, smooth_window, smooth_polyorder)
    q = _clip_joint_velocity(q, fps=fps, max_joint_vel=max_joint_vel)
    return q


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        # required=True,
        default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1General_c3d/General_A1_-_Stand_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsKicks_c3d/G8_-__roundhouse_left_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/AMASS/KIT_572_dance_chacha11_stageii.npz"
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male2MartialArtsPunches_c3d/E1_-__Jab_left_stageii.npz",
        # default="/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/Male1Running_c3d/Run_C24_-_quick_side_step_left_stageii.npz",
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "gp2_v2", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung", "agibot_x2"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional IK config json path. If not provided, use default from params.py.",
    )
    
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion.",
    )

    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )

    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit the rate of the retargeted robot motion to keep the same as the human motion.",
    )
    parser.add_argument(
        "--retarget_mode",
        choices=["frame_ik", "trajectory"],
        default="frame_ik",
        help="frame_ik: original per-frame IK. trajectory: ProtoMotions-like two-pass + smoothing.",
    )
    parser.add_argument(
        "--max_joint_vel",
        type=float,
        default=20.0,
        help="Joint velocity clamp in rad/s used in trajectory mode.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=11,
        help="Savitzky-Golay window length for dof smoothing in trajectory mode.",
    )
    parser.add_argument(
        "--smooth_polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polyorder for dof smoothing in trajectory mode.",
    )
    parser.add_argument(
        "--use_velocity_limit",
        action="store_true",
        help="Enable IK joint velocity limits inside GMR solver for stability.",
    )
    parser.add_argument(
        "--joint_blend",
        type=float,
        default=0.0,
        help="Blend current joint solution with previous frame in [0,1]. Helps avoid IK branch jumps.",
    )
    parser.add_argument(
        "--joint_step_limit",
        type=float,
        default=0.0,
        help="Maximum per-frame joint delta (rad) for joints. 0 disables.",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.0,
        help="Vertical offset added to base_z when exporting (meters). Use -0.15 to move robot down 15 cm.",
    )

    args = parser.parse_args()


    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"
    
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
   
    # Initialize retargeting.
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        ik_config_path=args.config,
        use_velocity_limit=args.use_velocity_limit,
        joint_blend=args.joint_blend,
        joint_step_limit=args.joint_step_limit,
    )
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=aligned_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=f"videos/{args.robot}_{args.smplx_file.split('/')[-1].split('.')[0]}.mp4",)
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    if args.retarget_mode == "trajectory":
        print("[GMR] Running ProtoMotions-like trajectory retargeting...")
        qpos_seq = _run_protomotions_like(
            retarget_ctor=GMR,
            frames=smplx_data_frames,
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=args.robot,
            ik_config_path=args.config,
            use_velocity_limit=args.use_velocity_limit,
            joint_blend=args.joint_blend,
            joint_step_limit=args.joint_step_limit,
            fps=aligned_fps,
            max_joint_vel=args.max_joint_vel,
            smooth_window=args.smooth_window,
            smooth_polyorder=args.smooth_polyorder,
        )
    else:
        print("[GMR] Running frame-by-frame IK retargeting...")
        qpos_seq = _run_framewise(retarget, smplx_data_frames)

    # Playback (and optional loop) from precomputed sequence.
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    qpos_list = []
    i = -1
    while True:
        if args.loop:
            i = (i + 1) % len(qpos_seq)
        else:
            i += 1
            if i >= len(qpos_seq):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        qpos = qpos_seq[i]
        human_frame = smplx_data_frames[i if i < len(smplx_data_frames) else -1]
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=human_frame,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)
            
    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        if args.z_offset != 0.0:
            root_pos[:, 2] += args.z_offset
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        body_names = None
        
        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos,
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")
            
      
    
    robot_motion_viewer.close()
