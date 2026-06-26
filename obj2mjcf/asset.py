from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from obj2mjcf.material import Material


@dataclass
class SubMesh:
    """A single per-material mesh produced from the source OBJ."""

    obj_path: Path  # exported per-material .obj (referenced by the MJCF emitter)
    geometry: Any  # in-memory trimesh.Trimesh (baked by the USD emitter)
    material_name: Optional[str]


@dataclass
class ProcessedAsset:
    """The format-agnostic result of processing one composite OBJ.

    Both the MJCF and USD emitters read this; it is the single source of truth so the
    pipeline does the OBJ splitting / material parsing / convex decomposition exactly once.
    """

    name: str  # OBJ stem
    work_dir: Path
    submeshes: List[SubMesh]
    materials: List[Material]
    collision_parts: List[Path] = field(default_factory=list)  # CoACD parts, or []
    decomp_success: bool = False
    meters_per_unit: float = 1.0
    up_axis: str = "Z"  # "Z" (MuJoCo convention) or "Y"

    @property
    def has_materials(self) -> bool:
        return len(self.materials) > 0


@dataclass
class EmitOpts:
    """Per-run options consumed by emitters."""

    add_free_joint: bool = False
    usd_physics: bool = False
    density: Optional[float] = None
    usd_binary: bool = True  # .usdc when True, .usda when False
