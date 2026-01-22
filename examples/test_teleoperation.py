#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import select
import sys
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import gym_hil  # noqa: F401
try:
    from mujoco.glfw import glfw
except ImportError:
    print("GLFW not found. Please `pip install glfw`")
    exit(1)

SO101_SCENE_PATH = Path(__file__).resolve().parents[1] / "models" / "SO-ARM100" / "Simulation" / "SO101" / "scene.xml"
SO101_ENV_ID = "gym_hil/SO101KeyboardBase-v0"


def register_so101_env() -> str:
    from gymnasium.envs.registration import register, registry

    if SO101_ENV_ID not in registry:
        register(
            id=SO101_ENV_ID,
            entry_point=SO101GymEnv,
            max_episode_steps=5000,
        )
    return SO101_ENV_ID


def _joint_ranges_and_names(model) -> tuple[list[tuple[float, float]], list[str]]:
    ranges = []
    names = []
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id][0])
        low, high = model.jnt_range[joint_id]
        ranges.append((float(low), float(high)))
        names.append(model.joint(joint_id).name)
    return ranges, names


def _extract_joint_positions(data, actuator_joint_ids) -> np.ndarray:
    return data.qpos[actuator_joint_ids].astype(np.float32)


def _clamp_action(action: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    clamped = []
    for value, (low, high) in zip(action.tolist(), ranges):
        clamped.append(min(max(value, low), high))
    return np.asarray(clamped, dtype=np.float32)


def _apply_joint_delta(
    action: np.ndarray, joint_index: int, delta: float, ranges: list[tuple[float, float]]
) -> np.ndarray:
    updated = action.copy()
    if 0 <= joint_index < updated.shape[0]:
        updated[joint_index] += delta
    return _clamp_action(updated, ranges)


class SO101KeyboardController:
    def __init__(self, window, step_size: float, joint_count: int):
        self.window = window
        self.step_size = step_size
        self.joint_index = 0
        self.move_positive = False
        self.move_negative = False
        self.quit_requested = False
        self._keymap = {
            getattr(glfw, f"KEY_{i}"): i - 1 for i in range(1, min(joint_count, 9) + 1)
        }

    def start(self):
        glfw.set_key_callback(self.window, self._key_callback)

    def stop(self):
        if self.window:
            glfw.set_key_callback(self.window, None)

    def _key_callback(self, window, key, scancode, act, mods):
        is_press = act in (glfw.PRESS, glfw.REPEAT)
        if key == glfw.KEY_ESCAPE and is_press:
            self.quit_requested = True
            glfw.set_window_should_close(window, 1)
            return
        if key == glfw.KEY_UP:
            self.move_positive = is_press
        elif key == glfw.KEY_DOWN:
            self.move_negative = is_press
        elif is_press and key == glfw.KEY_LEFT_BRACKET:
            self.step_size = max(self.step_size * 0.5, 0.001)
        elif is_press and key == glfw.KEY_RIGHT_BRACKET:
            self.step_size = min(self.step_size * 2.0, 0.5)
        elif is_press and key in self._keymap:
            self.joint_index = self._keymap[key]

    def delta(self) -> float:
        if self.move_positive and not self.move_negative:
            return self.step_size
        if self.move_negative and not self.move_positive:
            return -self.step_size
        return 0.0

    def reset(self):
        self.move_positive = False
        self.move_negative = False
        self.quit_requested = False


def _print_keyboard_help(joint_names: list[str], step_size: float) -> None:
    names = ", ".join(f"{i + 1}:{name}" for i, name in enumerate(joint_names))
    key_limit = min(len(joint_names), 9)
    print(
        "\nControls (Viewer window must be focused)\n"
        f"  1-{key_limit}: select joint ({names})\n"
        "  ↑ ↓: joint +/-\n"
        "  [ and ]: step size down/up\n"
        "  Esc: quit\n"
        f"Current step size: {step_size}\n"
    )


def _parse_stdin_joint_command(line: str, joint_count: int) -> np.ndarray | None:
    parts = line.strip().split()
    if not parts:
        return None
    if parts[0].upper() == "J":
        parts = parts[1:]
    if len(parts) == joint_count + 1:
        parts = parts[1:]
    if len(parts) != joint_count:
        return None
    try:
        values = np.asarray([float(v) for v in parts], dtype=np.float32)
    except ValueError:
        return None
    if values.shape[0] != joint_count:
        return None
    return values


class SO101GymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        xml_path: str,
        seed: int = 0,
        control_dt: float = 0.05,
        physics_dt: float = 0.002,
    ):
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = max(1, int(round(control_dt / physics_dt)))
        self._random = np.random.RandomState(seed)

        self._actuator_joint_ids = [int(joint_id) for joint_id in self._model.actuator_trnid[:, 0]]
        ctrl_range = self._model.actuator_ctrlrange.copy()
        low = ctrl_range[:, 0].astype(np.float32)
        high = ctrl_range[:, 1].astype(np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        obs_dim = len(self._actuator_joint_ids)
        self.observation_space = gym.spaces.Dict(
            {"agent_pos": gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)}
        )

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:] = self._model.qpos0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        return self._compute_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] != self.action_space.shape[0]:
            action = action[: self.action_space.shape[0]]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._data.ctrl[:] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)
        return self._compute_observation(), 0.0, False, False, {}

    def _compute_observation(self):
        qpos = _extract_joint_positions(self._data, self._actuator_joint_ids)
        return {"agent_pos": qpos}

def main():
    parser = argparse.ArgumentParser(description="Control SO-ARM101 robotic arm interactively")
    parser.add_argument("--step-size", type=float, default=0.01, help="Step size for joint movement")
    parser.add_argument(
        "--render-mode", type=str, default="human", choices=["human", "rgb_array"], help="Rendering mode"
    )
    parser.add_argument("--use-keyboard", action="store_true", help="Use keyboard control")
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=2.0,
        help="Delay in seconds when resetting the environment (0.0 means no delay)",
    )
    parser.add_argument(
        "--controller-config", type=str, default=None, help="Path to controller configuration JSON file"
    )
    parser.add_argument(
        "--stdin-control",
        action="store_true",
        help="Accept absolute joint positions (radians) from stdin lines",
    )
    args = parser.parse_args()

    # Mode selection based on --use-keyboard flag
    if args.use_keyboard:
        # MANUAL KEYBOARD CONTROL (runs with `python`)
        print("Running in MANUAL KEYBOARD mode. This can be run with the standard `python` interpreter.")
        
        # Create the BASE environment for SO-ARM101 without any rendering wrappers
        env_id = register_so101_env()
        env = gym.make(env_id, xml_path=str(SO101_SCENE_PATH), max_episode_steps=5000)
        model = env.unwrapped.model
        data = env.unwrapped.data
        obs, _ = env.reset()
        joint_ranges, joint_names = _joint_ranges_and_names(model)
        actuator_joint_ids = env.unwrapped._actuator_joint_ids
        action_dim = int(env.action_space.shape[0])
        if len(joint_ranges) < action_dim:
            joint_ranges = list(joint_ranges) + [(-1.0, 1.0)] * (action_dim - len(joint_ranges))
            joint_names = list(joint_names) + [
                f"joint_{i}" for i in range(len(joint_names), action_dim)
            ]
        else:
            joint_ranges = joint_ranges[:action_dim]
            joint_names = joint_names[:action_dim]

        # Initialize the viewer and controller manually
        glfw.init()
        window = glfw.create_window(1280, 720, "HIL Teleoperation (Manual Keyboard Mode)", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        
        controller = SO101KeyboardController(
            window,
            step_size=args.step_size,
            joint_count=action_dim,
        )

        # Setup MuJoCo rendering
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 90
        cam.elevation = -30
        cam.distance = 0.8
        cam.lookat = np.array([0.0, 0.0, 0.15])
        _print_keyboard_help(joint_names, args.step_size)
        if args.stdin_control:
            print(
                "STDIN control enabled. Send lines like:\n"
                "  J <t_ms> <q1> <q2> ... <qN>\n"
                "or:\n"
                "  J <q1> <q2> ... <qN>\n"
                "Values are absolute joint positions in radians."
            )

        try:
            controller.start()
            last_action = None
            while not glfw.window_should_close(window) and not controller.quit_requested:
                if last_action is None:
                    last_action = _extract_joint_positions(data, actuator_joint_ids)
                action = np.asarray(last_action, dtype=np.float32)
                if action.shape[0] < action_dim:
                    action = np.pad(action, (0, action_dim - action.shape[0]))

                if args.stdin_control:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if ready:
                        line = sys.stdin.readline()
                        stdin_action = _parse_stdin_joint_command(line, action_dim)
                        if stdin_action is not None:
                            action = stdin_action
                            last_action = action
                            action = _clamp_action(action, joint_ranges)
                        else:
                            action = _clamp_action(action, joint_ranges)
                    else:
                        action = _clamp_action(action, joint_ranges)
                
                if not args.stdin_control:
                    delta = controller.delta()
                    if delta != 0.0:
                        action = _apply_joint_delta(action, controller.joint_index, delta, joint_ranges)
                    else:
                        action = _clamp_action(action, joint_ranges)

                obs, reward, terminated, truncated, info = env.step(action.tolist())
                last_action = action

                viewport_width, viewport_height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
                mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                mujoco.mjr_render(viewport, scene, context)
                glfw.swap_buffers(window)
                glfw.poll_events()
                
                if terminated or truncated:
                    print("Episode finished – resetting…")
                    time.sleep(args.reset_delay)
                    obs, _ = env.reset()
                    controller.reset()
                    last_action = None
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            controller.stop()
            env.close()
            glfw.terminate()
            print("Session ended")

    else:
        # DEFAULT GAMEPAD CONTROL (requires `mjpython` on macOS)
        print("Running in DEFAULT GAMEPAD mode.")
        print("WARNING: This mode requires the `mjpython` interpreter on macOS due to its use of `mujoco.viewer.launch_passive`.")
        
         # Create Franka environment - Use base environment first to debug
        env = gym.make(
            "gym_hil/PandaPickCubeBase-v0",  # Use the base environment for debugging
            render_mode=args.render_mode,
            image_obs=True,
        )

        # Print observation space for debugging
        print("Observation space:", env.observation_space)

        # Reset and check observation structure
        obs, _ = env.reset()
        print("Observation keys:", list(obs.keys()))
        if "pixels" in obs:
            print("Pixels keys:", list(obs["pixels"].keys()))

        # Now try with the wrapped version
        print("\nTrying wrapped environment...")
        env = gym.make(
            "gym_hil/PandaPickCubeGamepad-v0",
            render_mode=args.render_mode,
            image_obs=True,
            use_gamepad=True,
            max_episode_steps=1000,  # 100 seconds * 10Hz
        )

        # Print observation space for the wrapped environment
        print("Wrapped observation space:", env.observation_space)

        # Reset and check wrapped observation structure
        obs, _ = env.reset()
        print("Wrapped observation keys:", list(obs.keys()))

        # Reset environment
        obs, _ = env.reset()
        dummy_action = np.zeros(4, dtype=np.float32)
        # This ensures the "stay gripper" action is set when the intervention button is not pressed
        dummy_action[-1] = 1

        try:
            while True:
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(dummy_action)

                # Print some feedback
                if info.get("succeed", False):
                    print("\nSuccess! Block has been picked up.")

                # If auto-reset is disabled, manually reset when episode ends
                if terminated or truncated:
                    print("Episode ended, resetting environment")
                    obs, _ = env.reset()

                # Add a small delay to control update rate
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            env.close()
            print("Session ended")


if __name__ == "__main__":
    main()
