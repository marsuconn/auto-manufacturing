"""Toolpath volume removal logic used by the environment simulation."""

from sim.tool_library import Tool, Toolpath


# Duration of a single simulation step in minutes
STEP_DURATION_MIN = 1.0


def compute_step_result(tool: Tool, toolpath: Toolpath,
                        remaining_volume: float) -> dict:
    """Compute the volume removed and time taken for one simulation step.

    Args:
        tool: The tool being used.
        toolpath: The toolpath strategy being executed.
        remaining_volume: Volume (mm³) of material left in the pocket.

    Returns:
        dict with keys:
            volume_removed (float): mm³ removed this step
            time_taken (float): minutes consumed this step
            energy_used (float): Watt-minutes of energy consumed
    """
    # Volume that can be removed in one step at the toolpath's rate
    max_removal = toolpath.volume_removal_rate * STEP_DURATION_MIN

    # Can't remove more than what's left
    volume_removed = min(max_removal, remaining_volume)

    # Time is proportional to how much of the step was actually used
    if max_removal > 0:
        time_taken = (volume_removed / max_removal) * STEP_DURATION_MIN
    else:
        time_taken = STEP_DURATION_MIN

    energy_used = toolpath.energy_consumption * time_taken

    return {
        "volume_removed": volume_removed,
        "time_taken": time_taken,
        "energy_used": energy_used,
    }
