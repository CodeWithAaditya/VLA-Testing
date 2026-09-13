#!/usr/bin/env python3
"""Keyboard + trackpad teleoperation of the SO-101 arm in MuJoCo.

Drives `tabletop_scene.xml` (SO-101 with the wrist camera, on a table with a
bowl, cubes, a ball and a can).  You move a Cartesian target for the gripper;
a damped-least-squares IK solver turns that into joint commands for the six
position actuators.

    python teleop.py                 # interactive window
    python teleop.py --selftest      # headless: scripted pick-and-place, no window

Controls are listed in the on-screen help (press F1) and in README_TELEOP.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE / "tabletop_scene.xml"

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
TCP_SITE = "gripperframe"

# Where the grasp actually happens, in the `gripperframe` site frame.  The site
# sits at the finger tips; the jaws close along the site's +z at z=0 and the
# pads are centred near x=-0.02.  Offsetting up by ~15 mm puts the control point
# in the middle of a 3 cm object held between the pads.
TCP_OFFSET = np.array([-0.020, 0.0, 0.015])

GRIPPER_CLOSED = -0.15
GRIPPER_OPEN = 1.20

# The model ships the jaw actuator with the same 3.35 Nm limit as the big
# joints.  At that torque, clamping down on a rigid cube stores enough energy
# in the contact to fling it across the table when you open again.  Limiting
# the jaw the way you would limit a real servo makes grasping forgiving: still
# ample for anything on this table, but it stalls instead of exploding.
JAW_FORCE = 0.8

# ---------------------------------------------------------------------------
# Teleop tuning
# ---------------------------------------------------------------------------

LINEAR_RATE = 0.16          # m/s of target motion while a key is held
ANGULAR_RATE = 1.5          # rad/s for gripper pitch / roll
GRIPPER_RATE = 2.5          # rad/s for the jaw
FINE_SCALE = 0.25           # shift = fine mode

DRAG_GAIN = 0.0022          # metres of target motion per pixel, per metre of view distance
SCROLL_GAIN = 0.012         # metres of target motion per scroll click
ROT_DRAG_GAIN = 0.006       # radians of pitch/roll per pixel

IK_ITERS = 14
IK_DAMPING = 0.05
IK_MAX_DQ = 0.20            # rad per iteration
IK_W_ROT = 0.12             # approach direction is a soft preference; position wins
IK_REJECT = 0.006           # if the solver misses by more than this, refuse the target

# The gripper is asked to point along a direction, not to hold a full
# orientation: the SO-101 has one joint too few for that, and the wrist roll is
# better driven straight through as the hardware joint it is.  Pitch is
# therefore a *preference* -- near the edge of the workspace the solver keeps
# the position and lets the gripper tilt, which is what you want when
# teleoperating.
PITCH_LIMITS = (np.deg2rad(-25.0), np.deg2rad(95.0))

# Workspace box the target is confined to (metres, world frame).  Keeps the
# target on/above the table and inside the arm's reach.
TARGET_BOUNDS = np.array([[-0.12, 0.46], [-0.34, 0.34], [0.004, 0.42]])


def approach_dir(yaw: float, pitch: float) -> np.ndarray:
    """Unit vector the gripper should point along.

    The gripper's approach axis is the `gripperframe` site's +x.  pitch=0 points
    it horizontally outward, pitch=+90 deg points it straight down at the table.
    """
    return np.array([np.cos(yaw) * np.cos(pitch),
                     np.sin(yaw) * np.cos(pitch),
                     -np.sin(pitch)])


# ---------------------------------------------------------------------------
# Model loading (adds the wrist camera, which MJCF alone cannot bolt onto an
# included body)
# ---------------------------------------------------------------------------

def build_model() -> mujoco.MjModel:
    """Compile the scene and attach a camera to the wrist camera module.

    The camera pose is derived from the model rather than hard-coded: it is
    placed just in front of the lens housing and aimed at the grasp point, with
    image-up along the gripper's +z.
    """
    probe = mujoco.MjModel.from_xml_path(str(SCENE))
    pd = mujoco.MjData(probe)
    pd.qpos[:6] = 0.0
    mujoco.mj_forward(probe, pd)

    sid = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    bid = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera")
    site_R = pd.site_xmat[sid].reshape(3, 3)
    body_R = pd.xmat[bid].reshape(3, 3)
    approach, up = site_R[:, 0], site_R[:, 2]

    cam_w = pd.xpos[bid] + approach * 0.042          # clear of the lens housing
    look_at = pd.site_xpos[sid] + site_R @ TCP_OFFSET

    z_cam = -(look_at - cam_w)                        # MuJoCo cameras look down -z
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (body_R.T @ np.column_stack([x_cam, y_cam, z_cam])).flatten())

    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.actuator(GRIPPER_JOINT).forcerange = [-JAW_FORCE, JAW_FORCE]
    spec.body("wrist_camera").add_camera(
        name="wrist_cam",
        pos=(body_R.T @ (cam_w - pd.xpos[bid])).tolist(),
        quat=quat.tolist(),
        fovy=75,
    )
    return spec.compile()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ArmController:
    """Cartesian target -> joint commands, via damped least squares."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.ik_data = mujoco.MjData(model)

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.gripper_body = model.site_bodyid[self.site_id]
        self.arm_qpos = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ARM_JOINTS])
        self.arm_dofs = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ARM_JOINTS])
        # IK drives everything up to the wrist flex; wrist roll is commanded
        # straight from the operator's roll handle.
        self.ik_dofs = self.arm_dofs[:4]
        self.arm_range = np.array(
            [model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in ARM_JOINTS])
        self.grip_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)
        self.arm_act = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS])

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

        # teleop state
        self.q_cmd = data.qpos[self.arm_qpos].copy()
        self.target = np.zeros(3)
        self.pitch = 0.0
        self.roll = 0.0
        self.grip = float(data.ctrl[self.grip_act])
        self.ik_error = 0.0
        self.blocked = False

        self.sync_from_state()

    # -- kinematics ---------------------------------------------------------

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm_qpos] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        R = d.site_xmat[self.site_id].reshape(3, 3).copy()
        return d.site_xpos[self.site_id] + R @ TCP_OFFSET, R

    def sync_from_state(self) -> None:
        """Adopt the arm's current pose as the teleop target."""
        self.q_cmd = self.data.qpos[self.arm_qpos].copy()
        pos, R = self.tcp_pose(self.q_cmd)
        self.target = np.clip(pos, TARGET_BOUNDS[:, 0], TARGET_BOUNDS[:, 1])
        # recover the operator's handles from the pose the arm is actually in
        self.pitch = float(np.clip(-np.arcsin(np.clip(R[2, 0], -1, 1)), *PITCH_LIMITS))
        self.roll = float(self.q_cmd[4])
        self.grip = float(self.data.qpos[
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]])
        self.push_ctrl()

    # -- IK -----------------------------------------------------------------

    def desired_approach(self, target: np.ndarray) -> np.ndarray:
        # Yaw is slaved to the target's azimuth: that is exactly what the base
        # pan joint does, so the task stays consistent with the arm's structure.
        return approach_dir(float(np.arctan2(target[1], target[0])), self.pitch)

    def solve(self, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Damped least squares on [position, approach direction].

        Warm-started from `seed`, so successive teleop steps track smoothly and
        the arm does not jump between IK branches.
        """
        want = self.desired_approach(target)
        q = seed.copy()
        q[4] = self.roll
        err_norm = np.inf
        for _ in range(IK_ITERS):
            pos, R = self.tcp_pose(q)
            e_pos = target - pos

            # Rotation that swings the current approach axis onto the desired
            # one.  It has no component about the approach axis itself, so it
            # never fights the roll command.
            axis = np.cross(R[:, 0], want)
            sin_a = np.linalg.norm(axis)
            e_rot = (axis / sin_a * np.arctan2(sin_a, float(np.dot(R[:, 0], want)))
                     if sin_a > 1e-9 else np.zeros(3))

            err_norm = float(np.linalg.norm(e_pos))
            if err_norm < 2e-4 and np.linalg.norm(e_rot) < 3e-3:
                break

            mujoco.mj_jac(self.model, self.ik_data, self._jacp, self._jacr,
                          pos, self.gripper_body)
            J = np.vstack([self._jacp[:, self.ik_dofs],
                           IK_W_ROT * self._jacr[:, self.ik_dofs]])
            err = np.concatenate([e_pos, IK_W_ROT * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
            scale = np.linalg.norm(dq)
            if scale > IK_MAX_DQ:
                dq *= IK_MAX_DQ / scale
            q[:4] = np.clip(q[:4] + dq, self.arm_range[:4, 0], self.arm_range[:4, 1])
        return q, err_norm

    def move_target(self, delta: np.ndarray) -> None:
        """Move the target, keeping it on the arm's reachable set.

        If the solver cannot get there, the best-effort joint solution is still
        used but the target is snapped onto the pose actually achieved.  That
        way the target never runs away from the arm: at the edge of the
        workspace the gripper simply stops advancing.
        """
        proposed = np.clip(self.target + delta, TARGET_BOUNDS[:, 0], TARGET_BOUNDS[:, 1])
        q, err = self.solve(proposed, self.q_cmd)
        self.q_cmd, self.ik_error = q, err
        self.blocked = err > IK_REJECT
        self.target = self.tcp_pose(q)[0] if self.blocked else proposed
        self.push_ctrl()

    def rotate_tool(self, d_pitch: float = 0.0, d_roll: float = 0.0) -> None:
        """Re-aim the gripper. Pitch that the arm cannot honour is rolled back."""
        prev_pitch = self.pitch
        self.pitch = float(np.clip(self.pitch + d_pitch, *PITCH_LIMITS))
        self.roll = float(np.clip(self.roll + d_roll,
                                  self.arm_range[4, 0], self.arm_range[4, 1]))
        q, err = self.solve(self.target, self.q_cmd)
        if err > IK_REJECT and d_pitch:
            self.pitch = prev_pitch
            q, err = self.solve(self.target, self.q_cmd)
        self.blocked = err > IK_REJECT
        self.q_cmd, self.ik_error = q, err
        self.push_ctrl()

    def move_gripper(self, delta: float) -> None:
        self.grip = float(np.clip(self.grip + delta, GRIPPER_CLOSED, GRIPPER_OPEN))
        self.push_ctrl()

    def set_gripper(self, value: float) -> None:
        self.grip = float(np.clip(value, GRIPPER_CLOSED, GRIPPER_OPEN))
        self.push_ctrl()

    def nudge_joint(self, index: int, delta: float) -> None:
        """Direct joint jog, bypassing IK (mode 'J')."""
        if index < len(ARM_JOINTS):
            self.q_cmd[index] = float(np.clip(
                self.q_cmd[index] + delta,
                self.arm_range[index, 0], self.arm_range[index, 1]))
            self.push_ctrl()
            pos, R = self.tcp_pose(self.q_cmd)
            self.target = pos
            self.pitch = float(np.clip(-np.arcsin(np.clip(R[2, 0], -1, 1)), *PITCH_LIMITS))
            self.roll = float(self.q_cmd[4])
        else:
            self.move_gripper(delta)

    def push_ctrl(self) -> None:
        self.data.ctrl[self.arm_act] = self.q_cmd
        self.data.ctrl[self.grip_act] = self.grip


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

# (keys, what they do) -- rendered as the two columns of the help overlay
BINDINGS = [
    ("W / S", "move target away from / toward the base (x)"),
    ("A / D", "move target left / right (y)"),
    ("Q / E", "raise / lower target (z)"),
    ("left / right", "swing target around the base"),
    ("R / F", "pitch gripper up / down"),
    ("Z / X", "roll the wrist"),
    ("G / B", "open / close the jaw"),
    ("Enter", "toggle the jaw open or shut"),
    ("Shift", "hold for fine motion"),
    ("", ""),
    ("left-drag", "drag the target in the view plane"),
    ("Shift + left-drag", "drag the target across the table"),
    ("Alt + left-drag", "aim the gripper (pitch and roll)"),
    ("scroll", "push / pull the target away from you"),
    ("right-drag", "orbit the camera"),
    ("Ctrl + drag", "pan the camera"),
    ("Ctrl + scroll", "zoom the camera"),
    ("Tab", "swap what left-drag and right-drag do"),
    ("", ""),
    ("J", "Cartesian <-> joint-jog mode"),
    ("1 ... 6", "pick a joint to jog"),
    ("up / down", "jog the picked joint"),
    ("", ""),
    ("[ / ]", "cycle cameras"),
    ("C", "wrist camera picture-in-picture"),
    ("Home", "return to the home pose"),
    ("Backspace", "reset the whole scene"),
    ("Space", "pause physics"),
    ("F1", "hide this help"),
    ("Esc", "quit"),
]
HELP_KEYS = "\n".join(k for k, _ in BINDINGS)
HELP_DESC = "\n".join(v for _, v in BINDINGS)


class TeleopViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, ctl: ArmController):
        import glfw  # imported here so --selftest works without a display

        self.glfw = glfw
        self.model, self.data, self.ctl = model, data, ctl

        if not glfw.init():
            raise RuntimeError("failed to initialise GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        self.window = glfw.create_window(1440, 900, "SO-101 teleoperation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("failed to create a window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.scene = mujoco.MjvScene(model, maxgeom=20000)
        self.wrist_scene = mujoco.MjvScene(model, maxgeom=20000)

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.cam.lookat[:] = (0.20, 0.0, 0.08)
        self.cam.distance = 1.05
        self.cam.azimuth = 135.0
        self.cam.elevation = -24.0

        self.named_cams = [-1] + [
            i for i in range(model.ncam)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) != "wrist_cam"
        ]
        self.cam_index = 0

        self.wrist_cam = mujoco.MjvCamera()
        self.wrist_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.wrist_cam.fixedcamid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        self.target_mocap = model.body_mocapid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")]

        self.show_help = True
        self.show_wrist = True
        self.paused = False
        self.jog_mode = False
        self.jog_joint = 0
        self.swap_drag = False
        self.status = "ready"

        self.last_x = self.last_y = 0.0
        self.button_left = self.button_right = self.button_middle = False

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # -- input --------------------------------------------------------------

    def _mods(self):
        g, w = self.glfw, self.window
        down = lambda k: g.get_key(w, k) == g.PRESS
        return (down(g.KEY_LEFT_SHIFT) or down(g.KEY_RIGHT_SHIFT),
                down(g.KEY_LEFT_CONTROL) or down(g.KEY_RIGHT_CONTROL),
                down(g.KEY_LEFT_ALT) or down(g.KEY_RIGHT_ALT))

    def _on_key(self, window, key, scancode, act, mods):
        g = self.glfw
        if act not in (g.PRESS, g.REPEAT):
            return
        if key == g.KEY_ESCAPE:
            g.set_window_should_close(window, True)
        elif key == g.KEY_F1:
            self.show_help = not self.show_help
        elif key == g.KEY_SPACE:
            self.paused = not self.paused
            self.status = "paused" if self.paused else "running"
        elif key == g.KEY_ENTER:
            closed = self.ctl.grip < 0.5 * (GRIPPER_CLOSED + GRIPPER_OPEN)
            self.ctl.set_gripper(GRIPPER_OPEN if closed else GRIPPER_CLOSED)
            self.status = "jaw opening" if closed else "jaw closing"
        elif key == g.KEY_TAB:
            self.swap_drag = not self.swap_drag
            self.status = f"left-drag = {'camera' if self.swap_drag else 'robot'}"
        elif key == g.KEY_J:
            self.jog_mode = not self.jog_mode
            self.status = f"{'joint-jog' if self.jog_mode else 'Cartesian'} mode"
        elif key == g.KEY_C:
            self.show_wrist = not self.show_wrist
        elif key in (g.KEY_LEFT_BRACKET, g.KEY_RIGHT_BRACKET):
            step = 1 if key == g.KEY_RIGHT_BRACKET else -1
            self.cam_index = (self.cam_index + step) % len(self.named_cams)
            cid = self.named_cams[self.cam_index]
            if cid < 0:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.status = "camera: free"
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.cam.fixedcamid = cid
                self.status = "camera: " + mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        elif key == g.KEY_HOME:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.ctl.sync_from_state()
            self.status = "home pose"
        elif key == g.KEY_BACKSPACE:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)
            self.ctl.sync_from_state()
            self.status = "scene reset"
        elif self.jog_mode and g.KEY_1 <= key <= g.KEY_6:
            self.jog_joint = key - g.KEY_1
            names = list(ARM_JOINTS) + [GRIPPER_JOINT]
            self.status = f"jogging {names[self.jog_joint]}"

    def _on_button(self, window, button, act, mods):
        g = self.glfw
        pressed = act == g.PRESS
        if button == g.MOUSE_BUTTON_LEFT:
            self.button_left = pressed
        elif button == g.MOUSE_BUTTON_RIGHT:
            self.button_right = pressed
        elif button == g.MOUSE_BUTTON_MIDDLE:
            self.button_middle = pressed
        self.last_x, self.last_y = g.get_cursor_pos(window)

    def _view_axes(self):
        """Camera right / up / forward unit vectors, and the view distance."""
        gl = self.scene.camera[0]
        fwd = np.array(gl.forward, dtype=float)
        up = np.array(gl.up, dtype=float)
        right = np.cross(fwd, up)
        n = np.linalg.norm(right)
        right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        dist = max(0.25, float(np.linalg.norm(self.ctl.target - np.array(gl.pos))))
        return right, up, fwd, dist

    def _on_cursor(self, window, xpos, ypos):
        g = self.glfw
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos
        if not (self.button_left or self.button_right or self.button_middle):
            return

        shift, ctrl, alt = self._mods()
        _, height = g.get_window_size(window)
        robot_btn = self.button_right if self.swap_drag else self.button_left
        cam_btn = self.button_left if self.swap_drag else self.button_right

        if robot_btn and not ctrl:
            if alt:
                self.ctl.rotate_tool(d_pitch=dy * ROT_DRAG_GAIN,
                                     d_roll=dx * ROT_DRAG_GAIN)
                return
            right, up, fwd, dist = self._view_axes()
            # Scale with view distance so a drag moves the same number of
            # pixels on screen whether you are zoomed in or out.
            gain = DRAG_GAIN * dist
            if shift:
                # slide along the table: screen-right stays screen-right, but
                # vertical drag pushes into the scene instead of lifting.
                flat_r = np.array([right[0], right[1], 0.0])
                flat_f = np.array([fwd[0], fwd[1], 0.0])
                for v in (flat_r, flat_f):
                    n = np.linalg.norm(v)
                    if n > 1e-9:
                        v /= n
                delta = flat_r * dx * gain - flat_f * dy * gain
            else:
                delta = right * dx * gain - up * dy * gain
            self.ctl.move_target(delta)
        elif cam_btn or (robot_btn and ctrl) or self.button_middle:
            action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if (ctrl or self.button_middle)
                      else mujoco.mjtMouse.mjMOUSE_ROTATE_H)
            mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height,
                                  self.scene, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        shift, ctrl, _ = self._mods()
        if ctrl:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                  0.0, -0.05 * yoffset, self.scene, self.cam)
            return
        right, up, fwd, dist = self._view_axes()
        gain = SCROLL_GAIN * (dist / 1.0) * (FINE_SCALE if shift else 1.0)
        self.ctl.move_target(fwd * yoffset * gain + right * xoffset * gain)

    def _apply_held_keys(self, dt: float) -> None:
        g, w = self.glfw, self.window
        down = lambda k: g.get_key(w, k) == g.PRESS
        shift, _, _ = self._mods()
        scale = FINE_SCALE if shift else 1.0

        if self.jog_mode:
            step = 0.0
            if down(g.KEY_UP):
                step += ANGULAR_RATE * dt * scale
            if down(g.KEY_DOWN):
                step -= ANGULAR_RATE * dt * scale
            if step:
                self.ctl.nudge_joint(self.jog_joint, step)
            return

        lin = LINEAR_RATE * dt * scale
        delta = np.zeros(3)
        if down(g.KEY_W):
            delta[0] += lin
        if down(g.KEY_S):
            delta[0] -= lin
        if down(g.KEY_A):
            delta[1] += lin
        if down(g.KEY_D):
            delta[1] -= lin
        if down(g.KEY_Q):
            delta[2] += lin
        if down(g.KEY_E):
            delta[2] -= lin
        if delta.any():
            self.ctl.move_target(delta)

        ang = ANGULAR_RATE * dt * scale
        d_pitch = (ang if down(g.KEY_R) else 0.0) - (ang if down(g.KEY_F) else 0.0)
        d_roll = (ang if down(g.KEY_Z) else 0.0) - (ang if down(g.KEY_X) else 0.0)
        if d_pitch or d_roll:
            self.ctl.rotate_tool(d_pitch=d_pitch, d_roll=d_roll)

        grip = (GRIPPER_RATE * dt if down(g.KEY_G) else 0.0) - \
               (GRIPPER_RATE * dt if down(g.KEY_B) else 0.0)
        if grip:
            self.ctl.move_gripper(grip)

        if down(g.KEY_LEFT) or down(g.KEY_RIGHT):
            yaw = ANGULAR_RATE * dt * scale * (1 if down(g.KEY_LEFT) else -1)
            r = np.linalg.norm(self.ctl.target[:2])
            a = np.arctan2(self.ctl.target[1], self.ctl.target[0]) + yaw
            self.ctl.move_target(np.array([r * np.cos(a), r * np.sin(a),
                                           self.ctl.target[2]]) - self.ctl.target)

    # -- rendering ----------------------------------------------------------

    def _overlay(self, viewport):
        c = self.ctl
        names = list(ARM_JOINTS) + [GRIPPER_JOINT]
        mode = f"joint-jog [{names[self.jog_joint]}]" if self.jog_mode else "Cartesian"
        left = (f"mode\ntarget xyz\npitch / roll\njaw\nIK residual\n"
                f"left-drag\nsim time\nstatus")
        right = (f"{mode}\n"
                 f"{c.target[0]:+.3f}  {c.target[1]:+.3f}  {c.target[2]:+.3f} m\n"
                 f"{np.rad2deg(c.pitch):+.0f}° / {np.rad2deg(c.roll):+.0f}°\n"
                 f"{c.grip:+.2f} rad{'  (blocked)' if c.blocked else ''}\n"
                 f"{c.ik_error * 1000:.1f} mm{'  OUT OF REACH' if c.blocked else ''}\n"
                 f"{'camera' if self.swap_drag else 'robot'}\n"
                 f"{self.data.time:6.1f} s{'  [PAUSED]' if self.paused else ''}\n"
                 f"{self.status}")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           left, right, self.ctx)
        if self.show_help:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW,
                               mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport,
                               HELP_KEYS, HELP_DESC, self.ctx)
        else:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW,
                               mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, viewport,
                               "F1", "controls", self.ctx)

    def _render_wrist(self, full):
        w = max(200, int(full.width * 0.24))
        h = int(w * 0.75)
        margin = 14
        frame = mujoco.MjrRect(full.width - w - margin - 2, margin - 2, w + 4, h + 4)
        inner = mujoco.MjrRect(full.width - w - margin, margin, w, h)
        mujoco.mjr_rectangle(frame, 0.05, 0.05, 0.06, 1.0)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, self.pert,
                               self.wrist_cam, mujoco.mjtCatBit.mjCAT_ALL,
                               self.wrist_scene)
        mujoco.mjr_render(inner, self.wrist_scene, self.ctx)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, inner,
                           "wrist cam", "", self.ctx)

    # -- main loop ----------------------------------------------------------

    def run(self):
        g = self.glfw
        model, data = self.model, self.data
        frame_budget = 1.0 / 60.0
        wall_prev = time.perf_counter()

        while not g.window_should_close(self.window):
            now = time.perf_counter()
            frame_dt = min(0.05, now - wall_prev)
            wall_prev = now

            self._apply_held_keys(frame_dt)

            if not self.paused:
                sim_end = data.time + frame_dt
                while data.time < sim_end:
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            if self.target_mocap >= 0:
                data.mocap_pos[self.target_mocap] = self.ctl.target

            width, height = g.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(model, data, self.opt, self.pert, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            if self.show_wrist:
                self._render_wrist(viewport)
            self._overlay(viewport)

            g.swap_buffers(self.window)
            g.poll_events()

            slack = frame_budget - (time.perf_counter() - now)
            if slack > 0:
                time.sleep(slack)

        g.terminate()


# ---------------------------------------------------------------------------
# Headless self-test: a scripted pick-and-place through the same controller
# ---------------------------------------------------------------------------

def _drive(model, data, ctl, seconds, target=None, pitch=None, roll=None, grip=None):
    steps = max(1, int(seconds / model.opt.timestep))
    t0, p0, r0, g0 = ctl.target.copy(), ctl.pitch, ctl.roll, ctl.grip
    tgt = None if target is None else np.asarray(target, dtype=float)
    for i in range(steps):
        a = (i + 1) / steps
        if tgt is not None:
            ctl.move_target(t0 + a * (tgt - t0) - ctl.target)
        if pitch is not None:
            ctl.rotate_tool(d_pitch=(p0 + a * (pitch - p0)) - ctl.pitch)
        if roll is not None:
            ctl.rotate_tool(d_roll=(r0 + a * (roll - r0)) - ctl.roll)
        if grip is not None:
            ctl.set_gripper(g0 + a * (grip - g0))
        mujoco.mj_step(model, data)


def selftest(shots_dir: str | None = None) -> int:
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctl = ArmController(model, data)

    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
    bowl = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    start = data.xpos[cube].copy()
    # A steep-but-not-vertical approach with the jaws rolled horizontal: the
    # jaws close along the site's z axis, so they have to straddle the cube
    # sideways, and a fully vertical approach is outside the wrist-flex range
    # over much of the table.
    down, side = np.deg2rad(75.0), np.deg2rad(-90.0)

    print(f"cube_red starts at {np.round(start, 4)}")
    print(f"bowl at          {np.round(data.xpos[bowl], 4)}")

    approach = start + np.array([0.0, 0.0, 0.09])
    _drive(model, data, ctl, 1.8, target=approach, pitch=down, roll=side, grip=0.60)
    print(f"  above cube: target err {np.linalg.norm(ctl.target - approach) * 1000:.1f} mm, "
          f"IK residual {ctl.ik_error * 1000:.2f} mm")

    _drive(model, data, ctl, 1.6, target=start)
    _drive(model, data, ctl, 1.2, grip=0.10)
    _drive(model, data, ctl, 1.4, target=start + np.array([0.0, 0.0, 0.14]))

    lifted = data.xpos[cube].copy()
    grasped = lifted[2] > start[2] + 0.06
    print(f"  after lift: cube at {np.round(lifted, 4)}  -> "
          f"{'GRASPED' if grasped else 'DROPPED'}")

    # dip the cube inside the rim before letting go, or it bounces back out
    over_bowl = data.xpos[bowl].copy() + np.array([0.0, 0.0, 0.075])
    _drive(model, data, ctl, 2.4, target=over_bowl)
    _drive(model, data, ctl, 0.9, grip=0.70)
    _drive(model, data, ctl, 1.5, target=over_bowl + np.array([0.0, 0.0, 0.07]))
    for _ in range(1200):
        mujoco.mj_step(model, data)

    final = data.xpos[cube].copy()
    bowl_xy = data.xpos[bowl][:2]
    in_bowl = np.linalg.norm(final[:2] - bowl_xy) < 0.055 and final[2] > 0.005
    print(f"  final cube  {np.round(final, 4)}  -> "
          f"{'IN THE BOWL' if in_bowl else 'NOT in the bowl'}")

    if shots_dir:
        import struct
        import zlib

        renderer = mujoco.Renderer(model, 720, 1080)

        def write_png(path, rgb):
            h, w, _ = rgb.shape
            raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
            chunk = lambda t, d: (struct.pack(">I", len(d)) + t + d
                                  + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF))
            open(path, "wb").write(
                b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))

        for cam in ("overview", "front", "wrist_cam"):
            renderer.update_scene(data, camera=cam)
            write_png(f"{shots_dir}/selftest_{cam}.png", renderer.render())
        print(f"  wrote renders to {shots_dir}")

    ok = grasped and in_bowl
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run a scripted pick-and-place headless and report the result")
    ap.add_argument("--shots", metavar="DIR",
                    help="with --selftest, write renders to DIR")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.shots)

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctl = ArmController(model, data)

    print(__doc__.split("Controls")[0].strip())
    print("\nPress F1 in the window for the control list.")
    TeleopViewer(model, data, ctl).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
