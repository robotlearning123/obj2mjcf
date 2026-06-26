import logging
from pathlib import Path
from typing import List, Optional

import mujoco
from lxml import etree
from termcolor import cprint

from obj2mjcf import constants
from obj2mjcf.asset import EmitOpts, ProcessedAsset
from obj2mjcf.emitters import collision_color


class MJCFBuilder:
    """Builds a MuJoCo XML model from a :class:`ProcessedAsset`."""

    format = "mjcf"

    def __init__(self, asset: ProcessedAsset, opts: Optional[EmitOpts] = None):
        self.asset = asset
        self.opts = opts or EmitOpts()
        self.tree: Optional[etree._ElementTree] = None

    # ------------------------------------------------------------------ build
    def _add_default_classes(self, root: etree.Element) -> None:
        default_elem = etree.SubElement(root, "default")
        visual_default = etree.SubElement(default_elem, "default")
        visual_default.attrib["class"] = "visual"
        etree.SubElement(
            visual_default,
            "geom",
            group="2",
            type="mesh",
            contype="0",
            conaffinity="0",
        )
        collision_default = etree.SubElement(default_elem, "default")
        collision_default.attrib["class"] = "collision"
        etree.SubElement(collision_default, "geom", group="3", type="mesh")

    def _add_assets(self, root: etree.Element) -> etree.Element:
        asset_elem = etree.SubElement(root, "asset")
        for material in self.asset.materials:
            if material.map_Kd is not None:
                texture = Path(material.map_Kd)
                etree.SubElement(
                    asset_elem,
                    "texture",
                    type="2d",
                    name=texture.stem,
                    file=texture.name,
                )
                mat = etree.SubElement(
                    asset_elem,
                    "material",
                    name=material.name,
                    texture=texture.stem,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                )
            else:
                mat = etree.SubElement(
                    asset_elem,
                    "material",
                    name=material.name,
                    specular=material.mjcf_specular(),
                    shininess=material.mjcf_shininess(),
                    rgba=material.mjcf_rgba(),
                )
            metallic = material.mjcf_metallic()
            if metallic is not None:
                mat.attrib["metallic"] = metallic
            roughness = material.mjcf_roughness()
            if roughness is not None:
                mat.attrib["roughness"] = roughness
        return asset_elem

    def _add_visual_geometries(
        self, body: etree.Element, asset_elem: etree.Element
    ) -> None:
        process_mtl = self.asset.has_materials
        for sm in self.asset.submeshes:
            meshname = sm.obj_path.name
            etree.SubElement(asset_elem, "mesh", file=Path(meshname).as_posix())
            geom = etree.SubElement(body, "geom", mesh=Path(meshname).stem)
            if process_mtl and sm.material_name is not None:
                geom.attrib["material"] = sm.material_name
            geom.attrib["class"] = "visual"

    def _add_collision_geometries(
        self, body: etree.Element, asset_elem: etree.Element
    ) -> None:
        if self.asset.decomp_success and self.asset.collision_parts:
            collisions = sorted(
                self.asset.collision_parts,
                key=lambda x: int(x.stem.split("_")[-1]),
            )
            for i, collision in enumerate(collisions):
                etree.SubElement(asset_elem, "mesh", file=collision.name)
                r, g, b = collision_color(i)
                geom = etree.SubElement(
                    body,
                    "geom",
                    mesh=collision.stem,
                    rgba=f"{r} {g} {b} 1",
                )
                geom.attrib["class"] = "collision"
        else:
            for sm in self.asset.submeshes:
                geom = etree.SubElement(body, "geom", mesh=sm.obj_path.stem)
                geom.attrib["class"] = "collision"

    def build(self) -> "MJCFBuilder":
        root = etree.Element("mujoco", model=self.asset.name)
        self._add_default_classes(root)
        asset_elem = self._add_assets(root)
        worldbody = etree.SubElement(root, "worldbody")
        body = etree.SubElement(worldbody, "body", name=self.asset.name)
        if self.opts.add_free_joint:
            etree.SubElement(body, "freejoint")
        self._add_visual_geometries(body, asset_elem)
        self._add_collision_geometries(body, asset_elem)
        tree = etree.ElementTree(root)
        etree.indent(tree, space=constants.XML_INDENTATION, level=0)
        self.tree = tree
        return self

    # -------------------------------------------------------------- save/check
    def save(self) -> List[Path]:
        if self.tree is None:
            self.build()
        xml_path = self.asset.work_dir / f"{self.asset.name}.xml"
        assert self.tree is not None
        self.tree.write(xml_path.as_posix(), encoding="utf-8")
        logging.info(f"Saved MJCF to {xml_path}")
        return [xml_path]

    def validate(self) -> None:
        """Compile and step the model; raises on any MuJoCo error."""
        if self.tree is None:
            self.build()
        assert self.tree is not None
        tmp_path = self.asset.work_dir / "tmp.xml"
        try:
            self.tree.write(tmp_path, encoding="utf-8")
            model = mujoco.MjModel.from_xml_path(tmp_path.as_posix())
            data = mujoco.MjData(model)
            mujoco.mj_step(model, data)
            cprint(f"{self.asset.name} (mjcf) compiled successfully!", "green")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
