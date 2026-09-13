#!/usr/bin/env python3
"""Headless SO-101 pick-and-place environment for VLA data collection and eval.

Wraps the tabletop scene + IK controller from `../teleop.py` in a fixed-rate
control loop with episode-level domain randomization, and renders the camera
views a policy sees.  Used by `collect_demos.py` (scripted expert -> LeRobot
dataset) and `eval_policy.py` (run a fine-tuned pi0 in the loop).

Conventions (match a real LeRobot SO-101 follower so a policy trained here
has the same input/output layout as on hardware):

    observation.state  : 6 joint positions (rad)  [5 arm joints + gripper]
    action             : 6 joint position targets (rad), i.e. `data.ctrl`
    observation.images : "front" (static) and "wrist" (gripper camera), HxWx3 uint8
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # ../teleop.py
import teleop  # noqa: E402
from teleop import ARM_JOINTS, GRIPPER_JOINT, ArmController, build_model  # noqa: E402

FPS = 30
SUBSTEPS = 20                                   # physics steps per control step
JOINT_NAMES = (*ARM_JOINTS, GRIPPER_JOINT)      # order of state / action vectors
CAMERAS = {"front": "front", "wrist": "wrist_cam"}   # dataset key -> MJCF camera
TASK = "Pick up the red cube and put it in the bowl."

# Randomization ranges (metres, world frame).  The cube spawns in a polar
# sector in front of the arm on the +y side; the bowl stays on the -y side,
# so the carry is always a swing across the base.
#
# The cube radius is capped at 0.27 m: further out the 5-DOF wrist cannot
# get a steep enough approach for the scripted grasp to be reliable.  The
# cube's yaw is sampled relative to its azimuth (mod 90 deg, the cube is
# symmetric) inside the window the jaw can straddle without one finger
# landing on the table or on top of the cube.  Widen CUBE_YAW_OFFSET only if
# `collect_demos.py --dry-run` still passes.
CUBE_R = (0.19, 0.27)
CUBE_AZ = (np.deg2rad(8.0), np.deg2rad(40.0))
CUBE_YAW_OFFSET = (np.deg2rad(-30.0), np.deg2rad(10.0))
BOWL_X, BOWL_Y = (0.26, 0.34), (-0.21, -0.11)
DISTRACTOR_X, DISTRACTOR_Y = (0.14, 0.42), (-0.30, 0.28)
CAN_X = (0.37, 0.44)                            # the tall can stays out of the carry swing
PARK = np.array([-0.45, 0.55, 0.05])            # off-table spot for hidden distractors
DISTRACTORS = ("cube_green", "cube_blue", "cube_yellow", "ball", "can")
BOWL_RADIUS = 0.055
CONTACT_SOLREF = (0.01, 1.0)                    # time constant, damping ratio


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class SO101PickEnv:
    def __init__(self, img_size: int = 256, seed: int | None = None):
        self.model = build_model()
        self.model.opt.timestep = 1.0 / (FPS * SUBSTEPS)   # 1/600 s -> exact 30 Hz control
        # Stiffer contacts.  With the default solref a cube resting against the
        # bowl's thin staves stores enough penetration energy to spontaneously
        # launch itself out of the bowl a second after being dropped in.
        self.model.geom_solref[:] = CONTACT_SOLREF
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.img_size = img_size
        self._renderer = None

        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qpos_ids = np.array([m.jnt_qposadr[jid(n)] for n in JOINT_NAMES])
        self.act_ids = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
        self.cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
        self.bowl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        self.cam_ids = {k: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, v)
                        for k, v in CAMERAS.items()}
        self._free_qpos = {n: m.jnt_qposadr[jid(f"{n}_free")]
                           for n in ("cube_red", "bowl", *DISTRACTORS)}

        # Nominal values we perturb per episode, so randomization never drifts.
        self._light_diffuse0 = m.light_diffuse.copy()
        self._light_pos0 = m.light_pos.copy()
        self._cam_pos0 = m.cam_pos.copy()
        self.ctl = ArmController(self.model, self.data)
        self._last_cmd = self.data.ctrl[self.act_ids].copy()

    # -- episode setup ------------------------------------------------------

    def _place(self, name: str, xy: np.ndarray, z: float, yaw: float = 0.0) -> None:
        a = self._free_qpos[name]
        self.data.qpos[a:a + 3] = (*xy, z)
        self.data.qpos[a + 3:a + 7] = _yaw_quat(yaw)
        # freejoint velocities live at the joint's dof address
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        d = self.model.jnt_dofadr[jid]
        self.data.qvel[d:d + 6] = 0.0

    def reset(self, randomize: bool = True) -> dict:
        m, d, rng = self.model, self.data, self.rng
        mujoco.mj_resetDataKeyframe(m, d, 0)
        m.light_diffuse[:] = self._light_diffuse0
        m.light_pos[:] = self._light_pos0
        m.cam_pos[:] = self._cam_pos0

        if randomize:
            # the bowl is a free body too (the arm can nudge it), so it is
            # placed through qpos like everything else
            bowl_xy = np.array([rng.uniform(*BOWL_X), rng.uniform(*BOWL_Y)])
            self._place("bowl", bowl_xy, 0.0, rng.uniform(-np.pi, np.pi))

            r, az = rng.uniform(*CUBE_R), rng.uniform(*CUBE_AZ)
            cube_xy = np.array([r * np.cos(az), r * np.sin(az)])
            yaw = az + rng.uniform(*CUBE_YAW_OFFSET) + rng.integers(0, 4) * np.pi / 2
            self._place("cube_red", cube_xy, 0.0145, yaw)

            # distractors: shuffle, keep clear of the cube, bowl and each other;
            # a third of them are hidden off-table so the policy does not learn
            # to count on the exact set of clutter
            taken = [cube_xy, bowl_xy]
            for name in DISTRACTORS:
                if rng.random() < 0.33:
                    self._place(name, PARK[:2] + rng.uniform(-0.05, 0.05, 2), PARK[2])
                    continue
                for _ in range(40):
                    xy = np.array([rng.uniform(*(CAN_X if name == "can" else DISTRACTOR_X)),
                                   rng.uniform(*DISTRACTOR_Y)])
                    gap = 0.11 if name == "can" else 0.075
                    if all(np.linalg.norm(xy - t) > (0.10 if t is bowl_xy else gap) for t in taken):
                        break
                else:
                    xy = PARK[:2]
                z = {"ball": 0.018, "can": 0.048, "cube_yellow": 0.0125}.get(name, 0.0145)
                self._place(name, xy, z, rng.uniform(-np.pi, np.pi))
                taken.append(xy)

            # lighting and static camera jitter
            m.light_diffuse[:] = self._light_diffuse0 * rng.uniform(0.7, 1.3, (m.nlight, 1))
            m.light_pos[0] = self._light_pos0[0] + rng.normal(0, 0.06, 3)
            fc = self.cam_ids["front"]
            m.cam_pos[fc] = self._cam_pos0[fc] + rng.normal(0, 0.012, 3)

        mujoco.mj_forward(m, d)
        for _ in range(SUBSTEPS * 5):                # let props settle
            mujoco.mj_step(m, d)
        self.ctl.sync_from_state()
        self._last_cmd = self.data.ctrl[self.act_ids].copy()
        return self.observe()

    def reset_noobs(self, randomize: bool = True) -> None:
        """`reset` without rendering the first observation."""
        obs_fn, self.observe = self.observe, lambda: None
        try:
            self.reset(randomize)
        finally:
            self.observe = obs_fn

    # -- stepping -----------------------------------------------------------

    def observe(self) -> dict:
        obs = {"state": self.data.qpos[self.qpos_ids].astype(np.float32)}
        for key, cam in CAMERAS.items():
            obs[key] = self.render(cam)
        return obs

    def render(self, camera: str = "front") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.img_size, self.img_size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def step(self, action: np.ndarray, render: bool = True) -> dict | None:
        """Apply 6 joint position targets and advance one control period.

        `render=False` skips the camera renders (the expensive part) for
        dry runs that only care about the outcome.
        """
        # First-order hold: ramp the targets from the previous command to the
        # new one across the substeps, like a servo's motion profile.  A
        # staircase at 30 Hz jitters the arm enough to shake a cube out of
        # the torque-limited jaw.
        # (The IK controller writes straight into data.ctrl, so the ramp has
        # to start from the previous *tick's* command, not from data.ctrl.)
        target = np.asarray(action, dtype=float)
        start = self._last_cmd
        for i in range(1, SUBSTEPS + 1):
            self.data.ctrl[self.act_ids] = start + (i / SUBSTEPS) * (target - start)
            mujoco.mj_step(self.model, self.data)
        self._last_cmd = target.copy()
        return self.observe() if render else None

    def current_action(self) -> np.ndarray:
        """The joint targets currently on the actuators (what the expert commands)."""
        return self.data.ctrl[self.act_ids].astype(np.float32)

    # -- task ---------------------------------------------------------------

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube].copy()

    def bowl_pos(self) -> np.ndarray:
        return self.data.xpos[self.bowl].copy()

    def success(self) -> bool:
        c, b = self.cube_pos(), self.bowl_pos()
        return (np.linalg.norm(c[:2] - b[:2]) < BOWL_RADIUS
                and 0.005 < c[2] < 0.06)         # resting in the bowl, not held above it

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
