import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.tool_library import ToolLibrary
from sim.toolpath import compute_step_result
from sim.workpiece import Workpiece


class PocketMachiningEnv(gym.Env):
    """Gymnasium environment for CNC pocket machining optimization.

    The agent selects (tool, toolpath) pairs to cut a rectangular pocket
    from an aluminum block, minimizing total machining time.
    """

    metadata = {"render_modes": ["human"]}

    # Constants
    TOOL_CHANGE_TIME = 0.5        # minutes (30 seconds)
    MAX_STEPS = 50
    MAX_TIME = 30.0               # minutes, for normalization
    VOLUME_THRESHOLD = 0.98       # fraction of pocket that must be removed
    QUALITY_THRESHOLD = 0.7       # minimum surface quality
    FINISHING_VOLUME_LIMIT = 0.15 # remaining fraction must be below this to allow finishing

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.library = ToolLibrary()
        self.workpiece = Workpiece(length=100.0, width=60.0, depth=20.0)

        # Action space: one action per toolpath in the library
        self.action_space = spaces.Discrete(self.library.num_actions)

        # Observation: [remaining_fraction, surface_quality, current_tool_norm, time_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        self._current_tool_id: int = -1
        self._elapsed_time: float = 0.0
        self._step_count: int = 0
        self._total_energy: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.workpiece.reset()
        self._current_tool_id = -1
        self._elapsed_time = 0.0
        self._step_count = 0
        self._total_energy = 0.0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        toolpath = self.library.get_toolpath(action)
        tool = self.library.get_tool(toolpath.tool_id)

        step_time = 0.0

        # Tool change penalty
        if tool.id != self._current_tool_id:
            step_time += self.TOOL_CHANGE_TIME
            self._current_tool_id = tool.id

        # Check if action is valid
        is_finishing = tool.tool_type == "finishing"
        action_invalid = is_finishing and self.workpiece.remaining_fraction > self.FINISHING_VOLUME_LIMIT

        if action_invalid:
            # Penalize invalid action: waste time, remove nothing
            step_time += 0.5
            reward = -step_time
        else:
            # Compute material removal
            remaining_vol = self.workpiece.remaining_fraction * self.workpiece.total_volume
            result = compute_step_result(tool, toolpath, remaining_vol)

            self.workpiece.remove_material(result["volume_removed"], is_finishing)
            step_time += result["time_taken"]
            self._total_energy += result["energy_used"]

            reward = -step_time

        self._elapsed_time += step_time
        self._step_count += 1

        terminated = self.workpiece.is_complete(
            self.VOLUME_THRESHOLD, self.QUALITY_THRESHOLD
        )
        truncated = self._step_count >= self.MAX_STEPS

        # Bonus for completing the pocket
        if terminated:
            reward += 5.0

        # Penalty if truncated without completion
        if truncated and not terminated:
            reward -= 10.0

        if self.render_mode == "human":
            self._render_step(action, tool, toolpath, step_time, action_invalid)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        ws = self.workpiece.get_state()  # [remaining_fraction, surface_quality]
        num_tools = len(self.library.tools)
        tool_norm = (self._current_tool_id + 1) / num_tools if self._current_tool_id >= 0 else 0.0
        time_norm = min(self._elapsed_time / self.MAX_TIME, 1.0)
        return np.array([ws[0], ws[1], tool_norm, time_norm], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "elapsed_time": self._elapsed_time,
            "total_energy": self._total_energy,
            "remaining_fraction": self.workpiece.remaining_fraction,
            "surface_quality": self.workpiece.surface_quality,
            "step_count": self._step_count,
        }

    def _render_step(self, action, tool, toolpath, step_time, invalid):
        status = "INVALID" if invalid else "OK"
        print(
            f"Step {self._step_count:2d} | "
            f"Action {action} ({toolpath.name}) | "
            f"Tool: {tool.name} | "
            f"Time: {step_time:.2f}min | "
            f"Remaining: {self.workpiece.remaining_fraction:.1%} | "
            f"Quality: {self.workpiece.surface_quality:.2f} | "
            f"{status}"
        )
