import argparse
import copy
import gc
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.ik_config_utils import parse_rot_offset, format_rot_offset_like
from general_motion_retargeting.params import IK_CONFIG_DICT
from general_motion_retargeting.utils.smpl import get_smplx_data, load_smplx_file


def _build_retarget(src_human: str, robot: str):
    return GeneralMotionRetargeting(src_human=src_human, tgt_robot=robot, verbose=False)


def _load_frames(motion_file: pathlib.Path, max_frames: int, frame_step: int):
    if motion_file.suffix.lower() == ".bvh":
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, _ = load_bvh_file(str(motion_file), format="lafan1")
        return frames[::frame_step][:max_frames], "bvh_lafan1"

    smplx_data, body_model, smplx_output, _ = load_smplx_file(
        str(motion_file), ROOT / "assets" / "body_models"
    )
    frame_count = smplx_output.vertices.shape[0]
    idxs = list(range(0, frame_count, frame_step))[:max_frames]
    frames = [get_smplx_data(smplx_data, body_model, smplx_output, i) for i in idxs]
    return frames, "smplx"


def _robot_body_pose(model_obj, data_obj, name: str):
    body_id = mj.mj_name2id(model_obj, mj.mjtObj.mjOBJ_BODY, name)
    pos = data_obj.xpos[body_id]
    mat = data_obj.xmat[body_id].reshape(3, 3)
    return pos, R.from_matrix(mat)


def _build_rot_grid(rot_range_deg: float, rot_step_deg: float) -> List[np.ndarray]:
    vals = np.arange(-rot_range_deg, rot_range_deg + 1e-9, rot_step_deg)
    rots = []
    for rx in vals:
        for ry in vals:
            for rz in vals:
                rots.append(np.array([rx, ry, rz], dtype=float))
    return rots


def _build_pos_grid(pos_range: float, pos_step: float) -> List[np.ndarray]:
    vals = np.arange(-pos_range, pos_range + 1e-9, pos_step)
    poss = []
    for x in vals:
        for y in vals:
            for z in vals:
                poss.append(np.array([x, y, z], dtype=float))
    return poss


def _get_link_base_entries(cfg: dict, link: str):
    out = {}
    for table_name in ("ik_match_table1", "ik_match_table2"):
        tab = cfg.get(table_name, {})
        if link not in tab:
            continue
        body_name, _, _, pos_off, rot_off = tab[link]
        out[table_name] = {
            "body": body_name,
            "pos": np.array(pos_off, dtype=float),
            "rot": parse_rot_offset(rot_off),
            "rot_raw": rot_off,
        }
    return out


def _apply_link_delta(retarget: GeneralMotionRetargeting, base_entries: dict, pos_delta, rot_delta_xyz_deg):
    rot_delta = R.from_euler("xyz", rot_delta_xyz_deg, degrees=True)

    if "ik_match_table1" in base_entries:
        e = base_entries["ik_match_table1"]
        body = e["body"]
        if body in retarget.pos_offsets1 and body in retarget.rot_offsets1:
            retarget.pos_offsets1[body] = e["pos"] + pos_delta - retarget.ground
            base_r = e["rot"]
            retarget.rot_offsets1[body] = rot_delta * base_r

    if "ik_match_table2" in base_entries:
        e = base_entries["ik_match_table2"]
        body = e["body"]
        if body in retarget.pos_offsets2 and body in retarget.rot_offsets2:
            retarget.pos_offsets2[body] = e["pos"] + pos_delta - retarget.ground
            base_r = e["rot"]
            retarget.rot_offsets2[body] = rot_delta * base_r


def _evaluate_candidate(
    retarget: GeneralMotionRetargeting,
    frames,
    link: str,
    base_entries: dict,
) -> Tuple[float, float]:
    # Reset robot state for fair candidate comparison.
    retarget.configuration.data.qpos[:] = retarget.model.qpos0
    retarget.configuration.data.qvel[:] = 0
    mj.mj_forward(retarget.model, retarget.configuration.data)

    pos_err = 0.0
    ori_err = 0.0
    count = 0

    body_name = base_entries["ik_match_table1"]["body"] if "ik_match_table1" in base_entries else base_entries["ik_match_table2"]["body"]

    for human in frames:
        retarget.retarget(human)
        if body_name not in retarget.scaled_human_data:
            continue

        target_pos, target_quat = retarget.scaled_human_data[body_name]
        target_rot = R.from_quat(np.array(target_quat, dtype=float), scalar_first=True)
        robot_pos, robot_rot = _robot_body_pose(retarget.model, retarget.configuration.data, link)

        pos_err += float(np.linalg.norm(robot_pos - target_pos))
        ori_err += float(np.degrees((target_rot.inv() * robot_rot).magnitude()))
        count += 1

    if count == 0:
        return np.inf, np.inf
    return pos_err / count, ori_err / count


def _worker_eval_single_candidate(args):
    cfg_path = pathlib.Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    motion_path = pathlib.Path(args.motion_file).resolve()
    frames, src_human = _load_frames(motion_path, args.max_frames, args.frame_step)
    IK_CONFIG_DICT[src_human][args.robot] = str(cfg_path)

    link = args.worker_link
    if link not in cfg.get("ik_match_table1", {}) and link not in cfg.get("ik_match_table2", {}):
        payload = {"ok": False, "reason": f"link '{link}' not found in IK tables"}
    else:
        base_entries = _get_link_base_entries(cfg, link)
        retarget = _build_retarget(src_human, args.robot)
        pos_delta = np.array(args.worker_pos_delta, dtype=float)
        rot_delta = np.array(args.worker_rot_delta, dtype=float)
        _apply_link_delta(retarget, base_entries, pos_delta, rot_delta)
        pos_err, ori_err = _evaluate_candidate(retarget, frames, link, base_entries)
        payload = {"ok": True, "pos_err": float(pos_err), "ori_err": float(ori_err)}

    out_path = pathlib.Path(args.worker_output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))


def _worker_eval_candidate_batch(args):
    cfg_path = pathlib.Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())
    motion_path = pathlib.Path(args.motion_file).resolve()
    frames, src_human = _load_frames(motion_path, args.max_frames, args.frame_step)
    IK_CONFIG_DICT[src_human][args.robot] = str(cfg_path)

    link = args.worker_link
    batch_file = pathlib.Path(args.worker_batch_file).resolve()
    batch = json.loads(batch_file.read_text())

    payload = {"ok": True, "results": []}
    if link not in cfg.get("ik_match_table1", {}) and link not in cfg.get("ik_match_table2", {}):
        payload = {"ok": False, "reason": f"link '{link}' not found in IK tables", "results": []}
    else:
        base_entries = _get_link_base_entries(cfg, link)
        retarget = _build_retarget(src_human, args.robot)
        for cand in batch:
            pos_delta = np.array(cand["pos"], dtype=float)
            rot_delta = np.array(cand["rot"], dtype=float)
            _apply_link_delta(retarget, base_entries, pos_delta, rot_delta)
            pos_err, ori_err = _evaluate_candidate(retarget, frames, link, base_entries)
            payload["results"].append({"pos_err": float(pos_err), "ori_err": float(ori_err)})

    out_path = pathlib.Path(args.worker_output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))


def _save_config_with_updates(cfg: dict, best_updates: dict, out_path: pathlib.Path):
    out_cfg = copy.deepcopy(cfg)
    for link, rec in best_updates.items():
        if rec is None:
            continue
        _, _, _, pos_delta, rot_delta = rec
        rot_d = R.from_euler("xyz", rot_delta, degrees=True)
        for table_name in ("ik_match_table1", "ik_match_table2"):
            tab = out_cfg.get(table_name, {})
            if link not in tab:
                continue
            base_pos = np.array(cfg[table_name][link][3], dtype=float)
            base_rot_raw = cfg[table_name][link][4]
            new_rot = rot_d * parse_rot_offset(base_rot_raw)
            tab[link][3] = (base_pos + pos_delta).tolist()
            tab[link][4] = format_rot_offset_like(base_rot_raw, new_rot)

    out_path = pathlib.Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_cfg, indent=4) + "\n")
    return out_path


def _run_isolated_links(args, links):
    cfg_in = pathlib.Path(args.config).resolve()
    with tempfile.TemporaryDirectory(prefix="gmr_search_") as td:
        tmp_dir = pathlib.Path(td)
        for i, link in enumerate(links):
            is_last = i == (len(links) - 1)
            cfg_out = pathlib.Path(args.save_config).resolve() if (is_last and args.save_config) else (tmp_dir / f"partial_{i}_{link}.json")

            cmd = [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--config",
                str(cfg_in),
                "--robot",
                args.robot,
                "--motion_file",
                args.motion_file,
                "--links",
                link,
                "--max_frames",
                str(args.max_frames),
                "--frame_step",
                str(args.frame_step),
                "--pos_range",
                str(args.pos_range),
                "--pos_step",
                str(args.pos_step),
                "--rot_range_deg",
                str(args.rot_range_deg),
                "--rot_step_deg",
                str(args.rot_step_deg),
                "--objective_ori_weight",
                str(args.objective_ori_weight),
                "--topk",
                str(args.topk),
                "--progress_every",
                str(args.progress_every),
                "--reset_every",
                str(args.reset_every),
                "--save_config",
                str(cfg_out),
            ]
            if args.isolate_candidates:
                cmd.append("--isolate_candidates")
            print(f"\n[isolate_links] running link '{link}' in subprocess...")
            rc = subprocess.call(cmd)
            if rc != 0:
                # Native crashes typically surface as negative exit codes (signal-terminated).
                # Fallback: retry this link with candidate-level isolation.
                if rc < 0 and (not args.isolate_candidates):
                    print(
                        f"[isolate_links] link '{link}' crashed (exit {rc}); "
                        "retrying with --isolate_candidates ..."
                    )
                    cmd_retry = cmd + ["--isolate_candidates"]
                    rc_retry = subprocess.call(cmd_retry)
                    if rc_retry != 0:
                        raise RuntimeError(
                            f"Subprocess retry failed for link '{link}' with exit code {rc_retry}."
                        )
                else:
                    raise RuntimeError(f"Subprocess failed for link '{link}' with exit code {rc}.")
            cfg_in = cfg_out
            if args.checkpoint_config:
                ckpt = pathlib.Path(args.checkpoint_config).resolve()
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(cfg_in), str(ckpt))
                print(f"[isolate_links] checkpoint saved: {ckpt}")

        if not args.save_config:
            print(f"\n[isolate_links] final merged config: {cfg_in}")


def main():
    parser = argparse.ArgumentParser(description="Grid search IK pos/rot offsets for selected links.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument(
        "--links",
        default="left_elbow_link,left_wrist_roll_link,right_elbow_link,right_wrist_roll_link",
        help="Comma-separated robot links in IK table.",
    )
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--frame_step", type=int, default=2)
    parser.add_argument("--pos_range", type=float, default=0.02)
    parser.add_argument("--pos_step", type=float, default=0.01)
    parser.add_argument("--rot_range_deg", type=float, default=60.0)
    parser.add_argument("--rot_step_deg", type=float, default=30.0)
    parser.add_argument("--objective_ori_weight", type=float, default=0.002, help="Objective = pos_err + w * ori_err_deg")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--progress_every", type=int, default=250, help="Print progress every N candidates per link.")
    parser.add_argument(
        "--reset_every",
        type=int,
        default=200,
        help="Recreate IK solver every N candidates per link (0 disables). Useful to avoid long-run native crashes.",
    )
    parser.add_argument(
        "--isolate_links",
        action="store_true",
        help="Run each link in a separate subprocess and chain partial configs. Useful for native segfault mitigation.",
    )
    parser.add_argument(
        "--isolate_candidates",
        action="store_true",
        help="Evaluate each candidate in a subprocess. Slow but robust against candidate-specific native crashes.",
    )
    parser.add_argument(
        "--isolate_batch_size",
        type=int,
        default=1,
        help="Batch size per subprocess when --isolate_candidates is enabled. Larger is faster, smaller is safer.",
    )
    parser.add_argument("--worker_eval_single_candidate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_eval_candidate_batch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_link", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_pos_delta", type=float, nargs=3, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_rot_delta", type=float, nargs=3, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_batch_file", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker_output", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save_config", default=None, help="Optional output json path with best offsets applied.")
    parser.add_argument(
        "--checkpoint_config",
        default=None,
        help="Optional path to save incremental progress after each completed link.",
    )
    args = parser.parse_args()

    if args.worker_eval_single_candidate:
        _worker_eval_single_candidate(args)
        return
    if args.worker_eval_candidate_batch:
        _worker_eval_candidate_batch(args)
        return

    links = [x.strip() for x in args.links.split(",") if x.strip()]
    if args.isolate_links and len(links) > 1:
        _run_isolated_links(args, links)
        return

    cfg_path = pathlib.Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text())

    frames, src_human = _load_frames(pathlib.Path(args.motion_file).resolve(), args.max_frames, args.frame_step)
    IK_CONFIG_DICT[src_human][args.robot] = str(cfg_path)

    pos_grid = _build_pos_grid(args.pos_range, args.pos_step)
    rot_grid = _build_rot_grid(args.rot_range_deg, args.rot_step_deg)

    print(f"Frames used: {len(frames)}")
    print(f"Candidates per link: {len(pos_grid) * len(rot_grid)}")

    best_updates = {}
    results = {}

    for link in links:
        if link not in cfg.get("ik_match_table1", {}) and link not in cfg.get("ik_match_table2", {}):
            print(f"[skip] {link} not found in IK tables")
            continue

        base_entries = _get_link_base_entries(cfg, link)
        retarget = _build_retarget(src_human, args.robot)

        best = []
        best_score = np.inf
        best_candidate = None

        total = len(pos_grid) * len(rot_grid)
        done = 0
        crashed = 0
        t0 = time.time()
        candidates = [(p.copy(), r.copy()) for p in pos_grid for r in rot_grid]
        batch_size = max(1, int(args.isolate_batch_size))
        for base_idx in range(0, len(candidates), batch_size):
            batch = candidates[base_idx : base_idx + batch_size]
            if args.isolate_candidates:
                with tempfile.TemporaryDirectory(prefix="gmr_cand_") as td:
                    td_path = pathlib.Path(td)
                    out_file = td_path / "result.json"
                    batch_file = td_path / "batch.json"
                    batch_file.write_text(
                        json.dumps(
                            [
                                {"pos": [float(x) for x in pos_delta], "rot": [float(x) for x in rot_delta]}
                                for (pos_delta, rot_delta) in batch
                            ]
                        )
                    )
                    cmd = [
                        sys.executable,
                        str(pathlib.Path(__file__).resolve()),
                        "--worker_eval_candidate_batch",
                        "--config",
                        str(cfg_path),
                        "--robot",
                        args.robot,
                        "--motion_file",
                        str(pathlib.Path(args.motion_file).resolve()),
                        "--max_frames",
                        str(args.max_frames),
                        "--frame_step",
                        str(args.frame_step),
                        "--worker_link",
                        link,
                        "--worker_batch_file",
                        str(batch_file),
                        "--worker_output",
                        str(out_file),
                    ]
                    rc = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if rc != 0 or (not out_file.exists()):
                        done += len(batch)
                        crashed += len(batch)
                        if done % args.progress_every == 0 or done == total:
                            elapsed = time.time() - t0
                            rate = done / max(elapsed, 1e-6)
                            remain = (total - done) / max(rate, 1e-6)
                            print(
                                f"[{link}] {done}/{total} ({100.0*done/total:.1f}%) "
                                f"best_score={best_score:.6f} crashes={crashed} eta={remain/60.0:.1f} min"
                            )
                        continue
                    payload = json.loads(out_file.read_text())
                if not payload.get("ok", False):
                    done += len(batch)
                    continue
                if len(payload.get("results", [])) != len(batch):
                    done += len(batch)
                    crashed += len(batch)
                    continue
                for (pos_delta, rot_delta), cand_res in zip(batch, payload["results"]):
                    pos_err = float(cand_res["pos_err"])
                    ori_err = float(cand_res["ori_err"])
                    score = pos_err + args.objective_ori_weight * ori_err
                    rec = (score, pos_err, ori_err, pos_delta.copy(), rot_delta.copy())
                    done += 1

                    if len(best) < args.topk:
                        best.append(rec)
                        best.sort(key=lambda x: x[0])
                    elif score < best[-1][0]:
                        best[-1] = rec
                        best.sort(key=lambda x: x[0])

                    if score < best_score:
                        best_score = score
                        best_candidate = rec

                    if done % args.progress_every == 0 or done == total:
                        elapsed = time.time() - t0
                        rate = done / max(elapsed, 1e-6)
                        remain = (total - done) / max(rate, 1e-6)
                        print(
                            f"[{link}] {done}/{total} ({100.0*done/total:.1f}%) "
                            f"best_score={best_score:.6f} crashes={crashed} eta={remain/60.0:.1f} min"
                        )
                continue

            for pos_delta, rot_delta in batch:
                if args.isolate_candidates:
                    # handled by batch worker path above
                    continue
                if args.reset_every > 0 and done > 0 and (done % args.reset_every == 0):
                    del retarget
                    gc.collect()
                    retarget = _build_retarget(src_human, args.robot)
                _apply_link_delta(retarget, base_entries, pos_delta, rot_delta)
                pos_err, ori_err = _evaluate_candidate(retarget, frames, link, base_entries)
                score = pos_err + args.objective_ori_weight * ori_err
                rec = (score, pos_err, ori_err, pos_delta.copy(), rot_delta.copy())
                done += 1

                if len(best) < args.topk:
                    best.append(rec)
                    best.sort(key=lambda x: x[0])
                elif score < best[-1][0]:
                    best[-1] = rec
                    best.sort(key=lambda x: x[0])

                if score < best_score:
                    best_score = score
                    best_candidate = rec

                if done % args.progress_every == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    remain = (total - done) / max(rate, 1e-6)
                    print(
                        f"[{link}] {done}/{total} ({100.0*done/total:.1f}%) "
                        f"best_score={best_score:.6f} crashes={crashed} eta={remain/60.0:.1f} min"
                    )

        results[link] = best
        best_updates[link] = best_candidate
        print(f"\n[{link}] best score={best_candidate[0]:.6f}, pos_err={best_candidate[1]:.6f}, ori_err_deg={best_candidate[2]:.3f}")
        print(f"  pos_delta={best_candidate[3]}, rot_delta_xyz_deg={best_candidate[4]}")
        if args.checkpoint_config:
            ckpt_path = _save_config_with_updates(cfg, best_updates, pathlib.Path(args.checkpoint_config))
            print(f"[checkpoint] saved: {ckpt_path}")

    if args.save_config:
        out_path = _save_config_with_updates(cfg, best_updates, pathlib.Path(args.save_config))
        print(f"\nSaved best config to {out_path}")

    print("\nTop candidates per link:")
    for link, best in results.items():
        print(f"\n  {link}")
        for i, rec in enumerate(best, 1):
            score, pos_err, ori_err, pos_delta, rot_delta = rec
            print(
                f"    {i}. score={score:.6f} pos={pos_err:.6f} ori_deg={ori_err:.3f} "
                f"pos_delta={np.array2string(pos_delta, precision=4)} "
                f"rot_delta={np.array2string(rot_delta, precision=2)}"
            )


if __name__ == "__main__":
    main()
