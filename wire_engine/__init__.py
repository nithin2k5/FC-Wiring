"""Core wire inspection logic: color detection, connector location and order checks.

This package has no GUI dependencies, so it can be used from scripts, tests or
another frontend.
"""
from .detector import (
    InspectionResult,
    RegionResult,
    WireDetection,
    detect_wires,
    draw_result,
    inspect,
    load_config,
    locate_regions,
    make_anchor,
    save_config,
    scale_box,
    search_area,
)

__all__ = [
    "InspectionResult",
    "RegionResult",
    "WireDetection",
    "detect_wires",
    "draw_result",
    "inspect",
    "load_config",
    "locate_regions",
    "make_anchor",
    "save_config",
    "scale_box",
    "search_area",
]
