import pathlib

import numpy as np
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from obj2mjcf import convert
from obj2mjcf.asset import EmitOpts, ProcessedAsset, SubMesh
from obj2mjcf.material import Material
from obj2mjcf.usd_builder import USDBuilder

_THIS_DIR = pathlib.Path(__file__).parent.absolute()


def _groups_obj(tmp_path: pathlib.Path) -> pathlib.Path:
    obj = tmp_path / "groups.obj"
    obj.write_bytes((_THIS_DIR / "groups.obj").read_bytes())
    return obj


def _write_png(path: pathlib.Path) -> None:
    Image.fromarray((np.random.rand(4, 4, 3) * 255).astype("uint8")).save(path)


def _open(path: pathlib.Path) -> Usd.Stage:
    stage = Usd.Stage.Open(path.as_posix())
    assert stage is not None
    return stage


# ----------------------------------------------------------------- geometry
def test_usd_stage_opens_with_valid_meshes(tmp_path) -> None:
    out = convert(_groups_obj(tmp_path), export=("usd",), usd_binary=False)
    stage = _open(out["usd"][0])
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert meshes
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        assert len(pts) > 0
        assert max(idx) < len(pts)
        assert sum(counts) == len(idx)


def test_usd_geometry_parity(tmp_path) -> None:
    obj = _groups_obj(tmp_path)
    out = convert(obj, export=("usd",), usd_binary=False)
    stage = _open(out["usd"][0])
    src = trimesh.load(obj.as_posix(), force="mesh")
    total_pts = sum(
        len(UsdGeom.Mesh(p).GetPointsAttr().Get())
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh)
    )
    assert total_pts == len(src.vertices)


def test_usd_stage_metadata(tmp_path) -> None:
    out = convert(
        _groups_obj(tmp_path),
        export=("usd",),
        usd_binary=False,
        meters_per_unit=0.01,
        up_axis="Y",
    )
    stage = _open(out["usd"][0])
    assert UsdGeom.GetStageUpAxis(stage) == "Y"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.01
    assert stage.GetDefaultPrim().IsValid()


# ----------------------------------------------------------------- compliance
def test_usd_passes_compliance(tmp_path) -> None:
    # convert(validate=True) raises if the stage is non-compliant.
    convert(_groups_obj(tmp_path), export=("usd",), usd_binary=False, validate=True)


# ------------------------------------------------------------------- physics
def test_usd_physics_dynamic_body(tmp_path) -> None:
    out = convert(
        _groups_obj(tmp_path),
        export=("usd",),
        usd_binary=False,
        usd_physics=True,
        add_free_joint=True,
        density=250.0,
    )
    stage = _open(out["usd"][0])
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetDensityAttr().Get() == 250.0
    assert any(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())


def test_usd_physics_static_when_no_free_joint(tmp_path) -> None:
    out = convert(
        _groups_obj(tmp_path),
        export=("usd",),
        usd_binary=False,
        usd_physics=True,
        add_free_joint=False,
    )
    stage = _open(out["usd"][0])
    root = stage.GetDefaultPrim()
    assert not root.HasAPI(UsdPhysics.RigidBodyAPI)  # static collider
    assert any(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse())


def test_usd_collision_parts_from_decomposition(tmp_path) -> None:
    # Build an asset with pre-decomposed convex parts (no CoACD run needed).
    box = trimesh.creation.box()
    box.export((tmp_path / "box.obj").as_posix())
    parts = []
    for i in range(2):
        p = tmp_path / f"box_collision_{i}.obj"
        trimesh.creation.box().export(p.as_posix())
        parts.append(p)
    asset = ProcessedAsset(
        name="box",
        work_dir=tmp_path,
        submeshes=[SubMesh(tmp_path / "box.obj", box, None)],
        materials=[],
        collision_parts=parts,
        decomp_success=True,
    )
    builder = USDBuilder(
        asset, EmitOpts(usd_physics=True, add_free_joint=True, usd_binary=False)
    )
    builder.build()
    paths = builder.save()
    builder.validate()
    stage = _open(paths[0])
    coll = [
        p
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh)
        and str(UsdGeom.Imageable(p).GetPurposeAttr().Get()) == "guide"
    ]
    assert len(coll) == 2
    for prim in coll:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert (
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            == "convexHull"
        )


# ------------------------------------------------------------------ materials
def test_usd_pbr_material_and_textures(tmp_path) -> None:
    box = trimesh.creation.box()
    _write_png(tmp_path / "albedo.png")
    _write_png(tmp_path / "normal.png")
    mat = Material(
        name="steel",
        Kd="0.2 0.2 0.22",
        Pr="0.3",
        Pm="1.0",
        map_Kd="albedo.png",
        map_norm="normal.png",
    )
    asset = ProcessedAsset(
        name="box",
        work_dir=tmp_path,
        submeshes=[SubMesh(tmp_path / "box.obj", box, "steel")],
        materials=[mat],
    )
    builder = USDBuilder(asset, EmitOpts(usd_binary=False))
    builder.build()
    paths = builder.save()
    builder.validate()
    stage = _open(paths[0])

    shader = UsdShade.Shader(stage.GetPrimAtPath("/box/Looks/steel/Shader"))
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    # Scalar with no texture is set directly.
    assert shader.GetInput("metallic").Get() == 1.0
    assert abs(shader.GetInput("roughness").Get() - 0.3) < 1e-6
    # diffuseColor + normal are textured -> connected, not set.
    assert shader.GetInput("diffuseColor").HasConnectedSource()
    assert shader.GetInput("normal").HasConnectedSource()
    assert stage.GetPrimAtPath("/box/Looks/steel/diffuseColor_tex").IsValid()
    assert stage.GetPrimAtPath("/box/Looks/steel/normal_tex").IsValid()
    # Mesh bound to the material.
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/box/Geom/box"))
    assert binding.GetDirectBindingRel().GetTargets()
