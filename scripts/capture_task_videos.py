#!/usr/bin/env python3
"""Render the 14 Tac-Bench website task videos from success-only ManiFeel data.

Run this script with the virtual environment from tactile_benchmark_manifeel:

  /path/to/tactile_benchmark_manifeel/.venv/bin/python \
    scripts/capture_task_videos.py \
    --benchmark-root /path/to/tactile_benchmark_manifeel

The supervisor starts one worker process per task because Isaac Gym is not
safe to initialise repeatedly in one Python process.  Workers replay an
episode from the current success-only Zarr store, require the simulator's real
success predicate, and write only validated H.264 videos to a staging folder.
The supervisor replaces website assets only after every task succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any


WEBSITE_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = WEBSITE_ROOT / "source" / "tasks"
FRAME_WIDTH = 1536
FRAME_HEIGHT = 1024


@dataclass(frozen=True)
class TaskSpec:
    website_id: str
    store_id: str
    config_name: str
    output_name: str
    cabinet_subtask: str | None = None
    wrist_primary: bool = False


# This is deliberately the paper's 14-task presentation set.  It excludes the
# executable-but-not-paper variants Blocks and Cabinet Slide Up.
TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("pih", "pih", "isaacgym_config", "pih_front_trajectory.mp4"),
    TaskSpec("usb", "usb", "isaacgym_config_usb", "usb_front_trajectory.mp4"),
    TaskSpec("plug", "plug", "isaacgym_config_power_plug", "plug_front_trajectory.mp4"),
    TaskSpec("gear", "gear", "isaacgym_config_gear", "gear_front_trajectory.mp4"),
    TaskSpec("nutbolt", "nutbolt", "isaacgym_config_nut", "nutbolt_front_trajectory.mp4"),
    TaskSpec("bulb", "bulb", "isaacgym_config_bulb", "bulb_front_trajectory.mp4"),
    TaskSpec("explore", "object_search", "isaacgym_config_object_search", "explore_front_trajectory.mp4", wrist_primary=True),
    TaskSpec("blind_insert", "peg_reorientation", "isaacgym_config_peg_reorientation", "blind_insert_front_trajectory.mp4", wrist_primary=True),
    TaskSpec("sorting", "sorting", "isaacgym_config_ball_sorting", "sorting_front_trajectory.mp4"),
    TaskSpec("test_tube", "test_tube", "isaacgym_config_test_tube", "test_tube_front_trajectory.mp4"),
    TaskSpec("stack_cube", "stack_cube", "isaacgym_config_stack_cube", "stack_cube_front_trajectory.mp4"),
    TaskSpec("cabinet_hinge", "hinge_door", "isaacgym_config_cabinet", "cabinet_hinge_front_trajectory.mp4", cabinet_subtask="hinge_door"),
    TaskSpec("cabinet_drawer", "drawer", "isaacgym_config_cabinet", "cabinet_drawer_front_trajectory.mp4", cabinet_subtask="drawer"),
    TaskSpec("cabinet_sliding", "sliding_door", "isaacgym_config_cabinet", "cabinet_sliding_front_trajectory.mp4", cabinet_subtask="sliding_door"),
)


def _spec_for_id(task_id: str) -> TaskSpec:
    for spec in TASKS:
        if spec.website_id == task_id:
            return spec
    raise ValueError(f"Unknown website task: {task_id}")


def _scalar(value: Any) -> Any:
    """Return a Zarr scalar as a regular Python value."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_ffmpeg(output: pathlib.Path, fps: int) -> subprocess.Popen:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _finish_ffmpeg(process: subprocess.Popen) -> None:
    assert process.stdin is not None
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed with exit code {code}: {stderr.strip()}")


def _write_poster(video: pathlib.Path, poster: pathlib.Path) -> None:
    """Extract a stable preview frame without loading the full video in a browser."""
    poster.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(video),
        "-frames:v", "1", "-q:v", "3", str(poster),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"poster generation failed for {video}: {completed.stderr.strip()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument(
        "--allow-store-fallback", action="store_true",
        help="Render a success-only store trajectory if simulator replay cannot reproduce it.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=[spec.website_id for spec in TASKS])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _configure_env(cfg: Any, spec: TaskSpec, device: str) -> Any:
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    gpu_id = int(device.rsplit(":", 1)[-1]) if ":" in device else 0
    cfg.sim_device = device
    cfg.rl_device = device
    cfg.graphics_device_id = gpu_id
    cfg.headless = True
    cfg.capture_video = False
    cfg.force_render = True
    # TacSL camera/tactile rendering requires two environments on the supported
    # Isaac Gym runtime. Only environment zero is rendered below.
    cfg.num_envs = 2
    if spec.cabinet_subtask:
        OmegaConf.update(cfg, "task.env.cabinet_subtask", spec.cabinet_subtask, merge=True)

    for key, value in {
        "task.env.use_camera": True,
        "task.env.use_camera_obs": True,
        "task.env.enableCameraSensors": True,
        "task.env.use_shear_force": False,
        "task.env.use_tactile_field_obs": False,
    }.items():
        if OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, value, merge=True)

    cameras = OmegaConf.select(cfg, "task.env.camera_configs") or []
    for camera in cameras:
        if camera.get("name") not in {"front", "wrist"}:
            continue
        camera.image_size = [1024, 1024]
        obs_dims = OmegaConf.select(cfg, f"task.env.obsDims.{camera.name}")
        if obs_dims is not None:
            OmegaConf.update(cfg, f"task.env.obsDims.{camera.name}", [1024, 1024, 3], merge=True)
    return cfg


def _frame_from_obs(obs: dict[str, Any], key: str, np: Any) -> Any:
    if key not in obs:
        available = sorted(
            name for name, value in obs.items()
            if hasattr(value, "shape") and len(value.shape) == 4
        )
        raise KeyError(f"Camera {key!r} is unavailable; got {available}")
    value = obs[key][0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.dtype != np.uint8:
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[..., :3])


def _square_contain(image: Any, size: int, cv2: Any, np: Any) -> Any:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _label(image: Any, text: str, cv2: Any) -> Any:
    frame = image.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _compose_frame(obs: dict[str, Any], spec: TaskSpec, success: bool, cv2: Any, np: Any) -> Any:
    front = _frame_from_obs(obs, "front", np)
    wrist = _frame_from_obs(obs, "wrist", np)
    tactile = _frame_from_obs(obs, "right_tactile_camera_taxim", np)
    primary, primary_name, secondary, secondary_name = (
        (wrist, "Wrist", front, "Front") if spec.wrist_primary
        else (front, "Front", wrist, "Wrist")
    )
    primary = _label(cv2.resize(primary, (1024, 1024), interpolation=cv2.INTER_AREA), primary_name, cv2)
    secondary = _label(_square_contain(secondary, 512, cv2, np), secondary_name, cv2)
    tactile = _label(_square_contain(tactile, 512, cv2, np), "Right tactile", cv2)
    right = np.concatenate((secondary, tactile), axis=0)
    frame = np.concatenate((primary, right), axis=1)
    if success:
        cv2.putText(frame, "SUCCESS", (18, 990), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, "SUCCESS", (18, 990), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 230, 70), 2, cv2.LINE_AA)
    return frame


def _write_result(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _render_store_trajectory(
    store: Any, episode_ends: Any, episode_seeds: Any, spec: TaskSpec,
    staging_dir: pathlib.Path, fps: int, cv2: Any, np: Any,
) -> dict[str, Any]:
    """Encode a trajectory from an explicitly success-only source store."""
    start, end = 0, int(episode_ends[0])
    required_views = ("front", "wrist", "right_tactile_camera_taxim")
    missing = [key for key in required_views if f"data/{key}" not in store]
    if missing:
        raise RuntimeError(f"{spec.store_id}: fallback views unavailable: {missing}")
    partial = staging_dir / f".{spec.output_name}.store-fallback.partial.mp4"
    writer = _run_ffmpeg(partial, fps)
    frames = 0
    try:
        for index in range(start, end):
            obs = {key: store[f"data/{key}"][index:index + 1] for key in required_views}
            frame = _compose_frame(obs, spec, index == end - 1, cv2, np)
            assert writer.stdin is not None
            writer.stdin.write(frame.tobytes())
            frames += 1
        for _ in range(fps):
            writer.stdin.write(frame.tobytes())
            frames += 1
    finally:
        _finish_ffmpeg(writer)
    final_path = staging_dir / spec.output_name
    os.replace(partial, final_path)
    return {
        "episode": 0,
        "seed": int(episode_seeds[0]),
        "success_step": int(store["meta/success_steps"][0]),
        "frames": frames,
        "source_start": start,
        "source_end_exclusive": end,
        "success_evidence": "success-only-store",
    }


def _worker(args: argparse.Namespace) -> int:
    if not args.task or not args.result:
        raise ValueError("Worker mode requires --task and --result")
    benchmark_root = args.benchmark_root.resolve()
    spec = _spec_for_id(args.task)
    print(f"[capture] {spec.website_id}: preparing success-only replay", flush=True)
    if not (benchmark_root / "manifeel" / "config").is_dir():
        raise FileNotFoundError(f"Not a ManiFeel benchmark root: {benchmark_root}")

    # Isaac Gym must load before torch, directly or indirectly through ManiFeel.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ["DISPLAY"] = ""
    sys.path.insert(0, str(benchmark_root))
    import isaacgym  # noqa: F401
    import cv2
    import numpy as np
    import torch
    import zarr
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from manifeel.envs.env_wrapper import IsaacEnvWrapper

    store_path = benchmark_root / "data" / f"{spec.store_id}.zarr"
    if not store_path.is_dir():
        raise FileNotFoundError(f"Missing success-only store: {store_path}")
    store = zarr.open(str(store_path), mode="r")
    if "meta/success_only" not in store or _scalar(store["meta/success_only"][()]) is not True:
        raise RuntimeError(f"{store_path} is not marked success_only=true")
    required = ("data/action", "meta/episode_ends", "meta/episode_seeds")
    missing = [key for key in required if key not in store]
    if missing:
        raise RuntimeError(f"{store_path} missing required arrays: {', '.join(missing)}")
    episode_ends = store["meta/episode_ends"][:]
    episode_seeds = store["meta/episode_seeds"][:]
    if len(episode_ends) != len(episode_seeds) or not len(episode_ends):
        raise RuntimeError(f"{store_path} has inconsistent episode metadata")

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(benchmark_root / "manifeel" / "config")):
        cfg = _configure_env(compose(config_name=spec.config_name), spec, args.device)
        env = IsaacEnvWrapper(cfg)

    attempts = min(args.max_episodes, len(episode_ends))
    staging_dir = args.output_dir.resolve()
    staging_dir.mkdir(parents=True, exist_ok=True)
    final_path = staging_dir / spec.output_name
    capture: dict[str, Any] | None = None
    try:
        for episode_index in range(attempts):
            start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
            end = int(episode_ends[episode_index])
            seed = int(episode_seeds[episode_index])
            print(f"[capture] {spec.website_id}: episode={episode_index} seed={seed}", flush=True)
            actions = store["data/action"][start:end]
            if not len(actions):
                continue
            env.seed(seed)
            obs = env.reset()
            partial = staging_dir / f".{spec.output_name}.episode{episode_index}.partial.mp4"
            writer = _run_ffmpeg(partial, args.fps)
            successful = False
            success_step: int | None = None
            frames = 0
            try:
                for step, action in enumerate(actions, start=1):
                    # IsaacEnvWrapper.step() always adds a leading singleton
                    # dimension.  That works for some 6-D tasks but produces
                    # an invalid (1, 2) gripper target for 7-D tasks when the
                    # simulator's mandatory two environments are active.
                    # Feed both environments the same recorded action and
                    # consume only verified environment 0 below.
                    action_tensor = torch.from_numpy(
                        np.asarray(action, dtype=np.float32)
                    ).to(device=args.device).unsqueeze(0).repeat(env.num_envs, 1)
                    raw_obs, _reward, reset, _info = env.envs.step(action_tensor)
                    obs = raw_obs["obs"]
                    successful = bool(env.envs._check_success()[0].item())
                    frame = _compose_frame(obs, spec, successful, cv2, np)
                    assert writer.stdin is not None
                    writer.stdin.write(frame.tobytes())
                    frames += 1
                    if successful:
                        success_step = step
                        for _ in range(args.fps):
                            writer.stdin.write(frame.tobytes())
                            frames += 1
                        break
                    if bool(reset[0].item()):
                        break
            finally:
                _finish_ffmpeg(writer)
            if successful:
                os.replace(partial, final_path)
                capture = {
                    "episode": episode_index,
                    "seed": seed,
                    "success_step": success_step,
                    "frames": frames,
                    "source_start": start,
                    "source_end_exclusive": end,
                }
                print(f"[capture] {spec.website_id}: success at step {success_step}", flush=True)
                break
    finally:
        env.close()

    if capture is None and args.allow_store_fallback:
        capture = _render_store_trajectory(
            store, episode_ends, episode_seeds, spec, staging_dir, args.fps, cv2, np,
        )
        print(f"[capture] {spec.website_id}: using verified success-only store fallback", flush=True)

    if capture is None:
        _write_result(args.result, {"task": spec.website_id, "success": False, "attempts": attempts})
        return 3
    _write_result(args.result, {
        "task": spec.website_id,
        "success": True,
        "output": str(final_path),
        "store": str(store_path.resolve()),
        "capture": capture,
    })
    return 0


def _supervisor(args: argparse.Namespace) -> int:
    benchmark_root = args.benchmark_root.resolve()
    if not (benchmark_root / ".venv" / "bin" / "python").is_file():
        raise FileNotFoundError(f"Expected benchmark virtual environment at {benchmark_root / '.venv/bin/python'}")
    selected = ([_spec_for_id(args.task)] if args.task else list(TASKS))
    staging = pathlib.Path(tempfile.mkdtemp(prefix="tacbench-task-video-", dir="/tmp"))
    results: list[dict[str, Any]] = []
    for spec in selected:
        print(f"[capture] starting {spec.website_id}", flush=True)
        result_path = staging / f"{spec.website_id}.json"
        command = [
            sys.executable, str(pathlib.Path(__file__).resolve()),
            "--worker", "--task", spec.website_id,
            "--benchmark-root", str(benchmark_root),
            "--output-dir", str(staging), "--result", str(result_path),
            "--fps", str(args.fps), "--max-episodes", str(args.max_episodes), "--device", args.device,
        ]
        if args.allow_store_fallback:
            command.append("--allow-store-fallback")
        completed = subprocess.run(command, check=False)
        if not result_path.is_file():
            raise RuntimeError(f"{spec.website_id}: worker exited {completed.returncode} without a result file")
        result = json.loads(result_path.read_text())
        if completed.returncode or result.get("success") is not True:
            raise RuntimeError(f"{spec.website_id}: capture failed: {result}")
        results.append(result)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for spec, result in zip(selected, results):
        staged_video = pathlib.Path(result["output"])
        target = output_dir / spec.output_name
        target_tmp = output_dir / f".{spec.output_name}.new"
        shutil.copy2(staged_video, target_tmp)
        os.replace(target_tmp, target)
        poster = output_dir / "posters" / f"{spec.website_id}.jpg"
        _write_poster(target, poster)
        manifest_entries.append({
            "website_task": spec.website_id,
            "website_video": spec.output_name,
            "poster": str(poster.relative_to(output_dir)),
            "store_task": spec.store_id,
            "config": spec.config_name,
            "cabinet_subtask": spec.cabinet_subtask,
            "layout": "wrist-primary" if spec.wrist_primary else "front-primary",
            "views": ["front", "wrist", "right_tactile_camera_taxim"],
            "source_store": result["store"],
            "capture": result["capture"],
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": args.fps,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "sha256": _sha256(target),
        })
    manifest = {
        "format_version": 1,
        "task_count": len(manifest_entries),
        "capture_tool": "scripts/capture_task_videos.py",
        "entries": manifest_entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = _parse_args()
    return _worker(args) if args.worker else _supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
