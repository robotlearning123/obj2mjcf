import math
from dataclasses import dataclass, fields
from typing import Optional, Sequence, Tuple

from obj2mjcf import constants

Color = Tuple[float, float, float]


def _parse_color(value: Optional[str], default: Color) -> Color:
    """Parse an MTL ``r g b`` triple, falling back to ``default``."""
    if value is None:
        return default
    parts = [float(x) for x in value.split()]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    if len(parts) == 1:  # grayscale shorthand.
        return (parts[0], parts[0], parts[0])
    return default


@dataclass
class UsdSurface:
    """Resolved UsdPreviewSurface inputs for a single material."""

    diffuse_color: Color
    opacity: float
    metallic: float
    roughness: float
    emissive_color: Color
    use_specular_workflow: bool
    specular_color: Color
    # Maps a UsdPreviewSurface input name to a local texture filename.
    textures: dict


@dataclass
class Material:
    """A convenience class for constructing materials from MTL files.

    Holds the classic MuJoCo Phong fields plus the common PBR extension fields
    (``Pr``/``Pm``/``Ke`` and their texture maps) so the same parsed material can drive
    both the MuJoCo and USD emitters.
    """

    name: str
    Ka: Optional[str] = None
    Kd: Optional[str] = None
    Ks: Optional[str] = None
    Ke: Optional[str] = None
    d: Optional[str] = None
    Tr: Optional[str] = None
    Ns: Optional[str] = None
    Pr: Optional[str] = None
    Pm: Optional[str] = None
    map_Kd: Optional[str] = None
    map_Ks: Optional[str] = None
    map_Ke: Optional[str] = None
    map_Pr: Optional[str] = None
    map_Pm: Optional[str] = None
    map_norm: Optional[str] = None
    map_d: Optional[str] = None

    @staticmethod
    def from_string(lines: Sequence[str]) -> "Material":
        """Construct a Material from a block of MTL lines (first line is ``newmtl``)."""
        valid = {f.name for f in fields(Material)}
        attrs = {"name": lines[0].split()[1]}
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            attr = constants.MTL_KEYWORD_TO_ATTR.get(parts[0])
            if attr is not None and attr in valid:
                attrs[attr] = " ".join(parts[1:]).strip()
        return Material(**attrs)

    # ------------------------------------------------------------------ MuJoCo
    def mjcf_rgba(self) -> str:
        Kd = self.Kd or "1.0 1.0 1.0"
        if self.d is not None:  # alpha
            alpha = self.d
        elif self.Tr is not None:  # 1 - alpha
            alpha = str(1.0 - float(self.Tr))
        else:
            alpha = "1.0"
        return f"{Kd} {alpha}"

    def mjcf_shininess(self) -> str:
        if self.Ns is not None:
            # Normalize Ns value to [0, 1]. Ns values normally range from 0 to 1000.
            Ns = float(self.Ns) / 1_000
        else:
            Ns = 0.5
        return f"{Ns}"

    def mjcf_specular(self) -> str:
        if self.Ks is not None:
            # Take the average of the specular RGB values.
            Ks = sum(list(map(float, self.Ks.split(" ")))) / 3
        else:
            Ks = 0.5
        return f"{Ks}"

    def mjcf_metallic(self) -> Optional[str]:
        """Metallic factor for MuJoCo's ``<material>`` (only when explicitly provided)."""
        return None if self.Pm is None else f"{float(self.Pm)}"

    def mjcf_roughness(self) -> Optional[str]:
        """Roughness factor for MuJoCo's ``<material>`` (only when explicitly provided)."""
        return None if self.Pr is None else f"{float(self.Pr)}"

    # --------------------------------------------------------------------- USD
    def usd_roughness(self) -> float:
        if self.Pr is not None:
            return max(0.0, min(1.0, float(self.Pr)))
        if self.Ns is not None:
            # Standard Blinn-Phong-exponent -> GGX-roughness approximation.
            return max(0.0, min(1.0, math.sqrt(2.0 / (float(self.Ns) + 2.0))))
        return constants.USD_DEFAULT_ROUGHNESS

    def usd_opacity(self) -> float:
        if self.d is not None:
            return float(self.d)
        if self.Tr is not None:
            return 1.0 - float(self.Tr)
        return 1.0

    def usd_preview_surface(self) -> UsdSurface:
        metallic = (
            float(self.Pm) if self.Pm is not None else constants.USD_DEFAULT_METALLIC
        )
        # Use the specular workflow only when there is no metallic input at all (neither a
        # scalar Pm nor a metallic map) and a specular color is available; otherwise use the
        # (default) metallic workflow.
        use_spec = self.Pm is None and self.map_Pm is None and self.Ks is not None
        textures = {}
        if self.map_Kd is not None:
            textures["diffuseColor"] = self.map_Kd
        if self.map_norm is not None:
            textures["normal"] = self.map_norm
        if self.map_Pr is not None:
            textures["roughness"] = self.map_Pr
        if self.map_Pm is not None:
            textures["metallic"] = self.map_Pm
        if self.map_Ke is not None:
            textures["emissiveColor"] = self.map_Ke
        if self.map_d is not None:
            textures["opacity"] = self.map_d
        return UsdSurface(
            diffuse_color=_parse_color(self.Kd, constants.USD_DEFAULT_DIFFUSE),
            opacity=self.usd_opacity(),
            metallic=metallic,
            roughness=self.usd_roughness(),
            emissive_color=_parse_color(self.Ke, (0.0, 0.0, 0.0)),
            use_specular_workflow=use_spec,
            specular_color=_parse_color(self.Ks, (0.0, 0.0, 0.0)),
            textures=textures,
        )

    def texture_attrs(self) -> dict:
        """All present texture maps as ``{attr_name: filename}`` (for copying)."""
        out = {}
        for attr in (
            "map_Kd",
            "map_Ks",
            "map_Ke",
            "map_Pr",
            "map_Pm",
            "map_norm",
            "map_d",
        ):
            val = getattr(self, attr)
            if val is not None:
                out[attr] = val
        return out
