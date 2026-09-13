#!/usr/bin/env python3
"""Generate scripted pick-and-place demonstrations as a LeRobot dataset.

Every episode randomizes the cube, bowl, clutter, lighting and the static
camera, then runs a scripted expert (the same IK pick-and-place as
`teleop.py --selftest`, with jittered timing and waypoints) at 30 Hz and
records what a policy would see.  Episodes where the cube does not end up
in the bowl are discarded.

    # macOS: default GL works.  Linux: MUJOCO_GL=egl for headless rendering.
    python collect_demos.py --episodes 200 --root data/so101_pick_cube

The result is a standard LeRobotDataset (videos + parquet) that
`lerobot-train` consumes directly, and that `eval_policy.py` reads the
normalization stats from.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from sim_env import CAMERAS, FPS, JOINT_NAMES, TASK, SO101PickEnv

DEFAULT_REPO_ID = "local/so101_sim_pick_cube"


# ---------------------------------------------------------------------------
# Scripted expert
# ---------------------------------------------------------------------------

def _wrap_pm45(a: float) -> float:
    """Fold an angle into (-45, 45] deg: a cube looks the same every 90 deg."""
    q = np.pi / 2
    return (a + q / 2) % q - q / 2


class ScriptedExpert:
    """Time-parameterised pick-and-place through the teleop IK controller.

    Each phase linearly interpolates the operator "handles" (target, pitch,
    roll, jaw) over its duration, exactly like `teleop._drive`, but advanced
    one control tick at a time so the commands can be recorded.
    """

    def __init__(self, env: SO101PickEnv, rng: np.random.Generator):
        self.env, self.ctl, self.rng = env, env.ctl, rng
        d = env.data
        start = env.cube_pos()
        # cube yaw from its quaternion, folded to the 90-degree symmetry
        a = env._free_qpos["cube_red"]
        w, x, y, z = d.qpos[a + 3:a + 7]
        cube_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        azimuth = np.arctan2(start[1], start[0])

        # Approach geometry, tuned by sweeping the randomized scene (see the
        # README): a steep-but-not-vertical pitch, the wrist rolled so the
        # jaws straddle the cube, and the grasp point pushed 4 mm sideways
        # (perpendicular to the azimuth) so the fixed jaw clears the cube's
        # face instead of landing on top of it.
        down = np.deg2rad(rng.uniform(72.0, 78.0))
        side = np.deg2rad(-90.0) + _wrap_pm45(cube_yaw - azimuth)
        lateral = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0]) * 0.004
        grasp = start + lateral + np.array([*rng.normal(0, 0.0015, 2), 0.0])
        hover = grasp + np.array([0, 0, rng.uniform(0.08, 0.10)])
        lift = grasp + np.array([0, 0, rng.uniform(0.12, 0.15)])
        # release from ~9 cm: low enough not to bounce out, high enough that a
        # cube hanging low in the jaw does not catch the rim on the way in
        over_bowl = env.bowl_pos() + np.array([*rng.normal(0, 0.003, 2),
                                               rng.uniform(0.085, 0.10)])
        retreat = over_bowl + np.array([0, 0, rng.uniform(0.07, 0.09)])
        j = lambda s: s * rng.uniform(0.85, 1.15)

        # (duration_s, handles at the END of the phase)
        self.phases = [
            (j(1.8), dict(target=hover, pitch=down, roll=side, grip=rng.uniform(0.55, 0.70))),
            (j(1.5), dict(target=grasp)),
            (j(1.0), dict(grip=0.10)),
            (j(1.3), dict(target=lift)),
            (j(2.4), dict(target=over_bowl)),
            (j(0.8), dict(grip=rng.uniform(0.60, 0.75))),
            (j(1.4), dict(target=retreat)),
            (0.7, dict()),                                   # hold still at the end
        ]
        self._phase = -1
        self._next_phase()

    def _next_phase(self) -> None:
        self._phase += 1
        if self._phase >= len(self.phases):
            return
        dur, goal = self.phases[self._phase]
        self._steps = max(1, int(round(dur * FPS)))
        self._i = 0
        c = self.ctl
        self._from = dict(target=c.target.copy(), pitch=c.pitch, roll=c.roll, grip=c.grip)
        self._goal = goal

    @property
    def done(self) -> bool:
        return self._phase >= len(self.phases)

    def act(self) -> np.ndarray:
        """Advance one tick and return the 6 joint targets to record."""
        if self.done:
            return self.env.current_action()
        self._i += 1
        a = self._i / self._steps
        c, f, g = self.ctl, self._from, self._goal
        if "target" in g:
            c.move_target(f["target"] + a * (g["target"] - f["target"]) - c.target)
        if "pitch" in g:
            c.rotate_tool(d_pitch=(f["pitch"] + a * (g["pitch"] - f["pitch"])) - c.pitch)
        if "roll" in g:
            c.rotate_tool(d_roll=(f["roll"] + a * (g["roll"] - f["roll"])) - c.roll)
        if "grip" in g:
            c.set_gripper(f["grip"] + a * (g["grip"] - f["grip"]))
        if self._i >= self._steps:
            self._next_phase()
        return self.env.current_action()


def run_episode(env: SO101PickEnv, rng: np.random.Generator, record: bool = True):
    """Roll out one scripted episode.  Returns (frames, success).

    With `record=False` nothing is rendered and `frames` is just a count.
    """
    obs = env.reset(randomize=True) if record else env.reset_noobs(randomize=True)
    expert = ScriptedExpert(env, rng)
    frames = [] if record else 0
    while not expert.done:
        action = expert.act()
        if record:
            frames.append((obs, action))
            obs = env.step(action)
        else:
            frames += 1
            env.step(action, render=False)
    return frames, env.success()


# ---------------------------------------------------------------------------
# Dataset writing
# ---------------------------------------------------------------------------

def dataset_features(img_size: int) -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),),
                              "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),),
                   "names": list(JOINT_NAMES)},
    }
    for key in CAMERAS:
        feats[f"observation.images.{key}"] = {
            "dtype": "video", "shape": (img_size, img_size, 3),
            "names": ["height", "width", "channels"]}
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200, help="successful episodes to keep")
    ap.add_argument("--root", default="data/so101_pick_cube", help="dataset directory")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help="LeRobot repo id (also the name used if you push to the Hub)")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="delete an existing --root first")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the expert and report success rate without writing anything")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = SO101PickEnv(img_size=args.img_size, seed=args.seed + 1)

    if args.dry_run:
        ok = 0
        for i in range(args.episodes):
            n, success = run_episode(env, rng, record=False)
            ok += success
            print(f"episode {i:4d}: {n:3d} frames  {'OK' if success else 'FAIL'}", flush=True)
        print(f"success rate {ok}/{args.episodes}")
        return 0

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root).resolve()
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite or choose another --root")
        shutil.rmtree(root)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, root=root, robot_type="so101_sim",
        features=dataset_features(args.img_size), use_videos=True,
        image_writer_threads=4,
    )

    kept = attempts = 0
    t0 = time.time()
    while kept < args.episodes:
        attempts += 1
        frames, success = run_episode(env, rng)
        if not success:
            print(f"  attempt {attempts}: expert failed, skipping")
            continue
        for obs, action in frames:
            frame = {"observation.state": obs["state"], "action": action, "task": TASK}
            for key in CAMERAS:
                frame[f"observation.images.{key}"] = obs[key]
            ds.add_frame(frame)
        ds.save_episode()
        kept += 1
        el = time.time() - t0
        print(f"episode {kept}/{args.episodes}  ({len(frames)} frames, "
              f"{attempts - kept} failed so far, {el / kept:.1f}s/ep, "
              f"eta {el / kept * (args.episodes - kept) / 60:.1f} min)")

    ds.finalize()
    env.close()
    print(f"\nwrote {kept} episodes to {root}")
    print(f"expert success rate: {kept}/{attempts}")
    print("visualise:  lerobot-dataset-viz --repo-id", args.repo_id, "--root", root, "--episode-index 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
