from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    id: int
    name: str
    diameter: float          # mm
    flute_count: int
    max_depth_of_cut: float  # mm
    max_feed_rate: float     # mm/min
    spindle_speed: float     # rpm
    tool_type: str           # "roughing" or "finishing"


@dataclass(frozen=True)
class Toolpath:
    id: int
    name: str
    tool_id: int
    volume_removal_rate: float   # mm³/min
    energy_consumption: float    # W
    step_over_ratio: float       # fraction of tool diameter
    step_down: float             # mm


class ToolLibrary:
    """Holds a predefined set of tools and their compatible toolpaths."""

    def __init__(self):
        self.tools: list[Tool] = [
            Tool(id=0, name="20mm Roughing Endmill", diameter=20.0,
                 flute_count=4, max_depth_of_cut=10.0, max_feed_rate=2000.0,
                 spindle_speed=8000, tool_type="roughing"),
            Tool(id=1, name="12mm Roughing Endmill", diameter=12.0,
                 flute_count=3, max_depth_of_cut=6.0, max_feed_rate=1500.0,
                 spindle_speed=10000, tool_type="roughing"),
            Tool(id=2, name="8mm Finishing Endmill", diameter=8.0,
                 flute_count=2, max_depth_of_cut=4.0, max_feed_rate=800.0,
                 spindle_speed=12000, tool_type="finishing"),
            Tool(id=3, name="50mm Face Mill", diameter=50.0,
                 flute_count=6, max_depth_of_cut=3.0, max_feed_rate=3000.0,
                 spindle_speed=5000, tool_type="roughing"),
        ]

        self.toolpaths: list[Toolpath] = [
            # 20mm Roughing Endmill toolpaths
            Toolpath(id=0, name="Adaptive clearing (20mm)", tool_id=0,
                     volume_removal_rate=12000.0, energy_consumption=1800.0,
                     step_over_ratio=0.4, step_down=8.0),
            Toolpath(id=1, name="Pocket roughing (20mm)", tool_id=0,
                     volume_removal_rate=9000.0, energy_consumption=1500.0,
                     step_over_ratio=0.6, step_down=5.0),
            # 12mm Roughing Endmill toolpaths
            Toolpath(id=2, name="Adaptive clearing (12mm)", tool_id=1,
                     volume_removal_rate=6000.0, energy_consumption=1200.0,
                     step_over_ratio=0.4, step_down=5.0),
            Toolpath(id=3, name="Pocket roughing (12mm)", tool_id=1,
                     volume_removal_rate=4500.0, energy_consumption=1000.0,
                     step_over_ratio=0.5, step_down=4.0),
            # 8mm Finishing Endmill toolpaths
            Toolpath(id=4, name="Contour finishing (8mm)", tool_id=2,
                     volume_removal_rate=800.0, energy_consumption=400.0,
                     step_over_ratio=0.1, step_down=3.0),
            Toolpath(id=5, name="Parallel finishing (8mm)", tool_id=2,
                     volume_removal_rate=600.0, energy_consumption=350.0,
                     step_over_ratio=0.08, step_down=2.0),
            # 50mm Face Mill toolpaths
            Toolpath(id=6, name="Face milling pass (50mm)", tool_id=3,
                     volume_removal_rate=15000.0, energy_consumption=2500.0,
                     step_over_ratio=0.7, step_down=2.0),
            Toolpath(id=7, name="Face milling light (50mm)", tool_id=3,
                     volume_removal_rate=10000.0, energy_consumption=1800.0,
                     step_over_ratio=0.5, step_down=1.5),
        ]

        self._tool_map = {t.id: t for t in self.tools}
        self._toolpath_map = {tp.id: tp for tp in self.toolpaths}

    @property
    def num_actions(self) -> int:
        return len(self.toolpaths)

    def get_tool(self, tool_id: int) -> Tool:
        return self._tool_map[tool_id]

    def get_toolpath(self, toolpath_id: int) -> Toolpath:
        return self._toolpath_map[toolpath_id]

    def get_tool_for_toolpath(self, toolpath_id: int) -> Tool:
        tp = self._toolpath_map[toolpath_id]
        return self._tool_map[tp.tool_id]
