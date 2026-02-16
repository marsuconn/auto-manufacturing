import numpy as np


class Workpiece:
    """Tracks workpiece state: remaining material volume and surface finish quality."""

    def __init__(self, length: float = 100.0, width: float = 60.0, depth: float = 20.0):
        self.length = length    # mm
        self.width = width      # mm
        self.depth = depth      # mm
        self.total_volume = length * width * depth  # mm³
        self.reset()

    def reset(self):
        self.volume_removed = 0.0
        self.surface_quality = 0.0  # 0 = raw stock, 1 = perfect finish

    @property
    def remaining_fraction(self) -> float:
        """Fraction of pocket volume still to be removed (1 = nothing cut, 0 = fully cut)."""
        return max(0.0, 1.0 - self.volume_removed / self.total_volume)

    def remove_material(self, volume: float, is_finishing: bool):
        """Remove material and update surface quality."""
        self.volume_removed = min(self.volume_removed + volume, self.total_volume)

        if is_finishing:
            # Finishing passes improve surface quality toward 1.0
            self.surface_quality = min(1.0, self.surface_quality + 0.25)
        else:
            # Roughing leaves poor finish, degrades quality slightly if already finished
            self.surface_quality = max(0.0, self.surface_quality - 0.05)

    def is_complete(self, volume_threshold: float = 0.98,
                    quality_threshold: float = 0.7) -> bool:
        """Pocket is complete when enough volume removed and surface finish is acceptable."""
        removed_fraction = self.volume_removed / self.total_volume
        return removed_fraction >= volume_threshold and self.surface_quality >= quality_threshold

    def get_state(self) -> np.ndarray:
        """Return observation components: [remaining_fraction, surface_quality]."""
        return np.array([self.remaining_fraction, self.surface_quality], dtype=np.float32)
