import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import trimesh
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from obj2mjcf.asset import EmitOpts, ProcessedAsset
from obj2mjcf.emitters import collision_color
from obj2mjcf.material import Material, UsdSurface

# UsdPreviewSurface inputs that carry color (rgb) vs. a single scalar channel (r).
_COLOR_INPUTS = {"diffuseColor", "emissiveColor", "specularColor"}
_NORMAL_INPUTS = {"normal"}


def _valid(name: str) -> str:
    """Sanitize an arbitrary string into a valid USD prim identifier."""
    return Tf.MakeValidIdentifier(name)


def _unique(name: str, used: set) -> str:
    """Disambiguate names that collide after sanitization (e.g. 'm 1' and 'm-1')."""
    candidate, i = name, 1
    while candidate in used:
        candidate = f"{name}_{i}"
        i += 1
    used.add(candidate)
    return candidate


class USDBuilder:
    """Builds a self-contained USD stage from a :class:`ProcessedAsset`.

    Emits geometry baked from the in-memory trimesh (USD cannot reference ``.obj``
    natively), ``UsdPreviewSurface`` materials, and optional ``UsdPhysics`` schemas.
    """

    format = "usd"

    def __init__(self, asset: ProcessedAsset, opts: Optional[EmitOpts] = None):
        self.asset = asset
        self.opts = opts or EmitOpts()
        self.stage: Optional[Usd.Stage] = None
        self._paths: List[Path] = []
        self._visual_meshes: List[Sdf.Path] = []

    # ------------------------------------------------------------------ build
    def build(self) -> "USDBuilder":
        stage = Usd.Stage.CreateInMemory()
        up = UsdGeom.Tokens.z if self.asset.up_axis.upper() == "Z" else UsdGeom.Tokens.y
        UsdGeom.SetStageUpAxis(stage, up)
        UsdGeom.SetStageMetersPerUnit(stage, self.asset.meters_per_unit)

        root_name = _valid(self.asset.name)
        root = UsdGeom.Xform.Define(stage, f"/{root_name}")
        stage.SetDefaultPrim(root.GetPrim())

        mat_prims: Dict[str, UsdShade.Material] = {}
        if self.asset.materials:
            UsdGeom.Scope.Define(stage, f"/{root_name}/Looks")
            used_names: set = set()
            for material in self.asset.materials:
                unique = _unique(_valid(material.name), used_names)
                mat_prims[material.name] = self._build_material(
                    stage, root_name, material, unique
                )

        UsdGeom.Scope.Define(stage, f"/{root_name}/Geom")
        for sm in self.asset.submeshes:
            self._build_mesh(stage, root_name, sm, mat_prims)

        if self.opts.usd_physics:
            self._apply_physics(stage, root_name, root)

        self.stage = stage
        return self

    # --------------------------------------------------------------- geometry
    def _build_mesh(self, stage, root_name, submesh, mat_prims) -> None:
        geom = submesh.geometry
        prim_name = _valid(submesh.obj_path.stem)
        path = f"/{root_name}/Geom/{prim_name}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        self._set_geometry(mesh, geom)
        UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.default_)
        self._visual_meshes.append(mesh.GetPath())

        name = submesh.material_name
        if name is not None and name in mat_prims:
            binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
            binding.Bind(mat_prims[name])

    def _set_geometry(self, mesh: UsdGeom.Mesh, geom: trimesh.Trimesh) -> None:
        verts = geom.vertices
        faces = geom.faces
        mesh.CreatePointsAttr(
            [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in verts]
        )
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
        bounds = geom.bounds  # (2, 3): min, max
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(*[float(v) for v in bounds[0]]),
                Gf.Vec3f(*[float(v) for v in bounds[1]]),
            ]
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        normals = getattr(geom, "vertex_normals", None)
        if normals is not None and len(normals) == len(verts):
            mesh.CreateNormalsAttr(
                [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in normals]
            )
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        uv = getattr(getattr(geom, "visual", None), "uv", None)
        if uv is not None and len(uv) == len(verts):
            primvars = UsdGeom.PrimvarsAPI(mesh)
            st = primvars.CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            st.Set([Gf.Vec2f(float(u), float(v)) for u, v in uv])

    # --------------------------------------------------------------- material
    def _build_material(
        self, stage, root_name, material: Material, prim_name: str
    ) -> UsdShade.Material:
        surf = material.usd_preview_surface()
        mat_path = f"/{root_name}/Looks/{prim_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        self._set_scalar_and_color_inputs(shader, surf)
        if surf.textures:
            self._wire_textures(stage, mat_path, shader, surf)

        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    def _set_scalar_and_color_inputs(self, shader, surf: UsdSurface) -> None:
        if "diffuseColor" not in surf.textures:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*surf.diffuse_color)
            )
        if "roughness" not in surf.textures:
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
                surf.roughness
            )
        if "metallic" not in surf.textures:
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(surf.metallic)
        if "opacity" not in surf.textures:
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(surf.opacity)
        if "emissiveColor" not in surf.textures and surf.emissive_color != (
            0.0,
            0.0,
            0.0,
        ):
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*surf.emissive_color)
            )
        if surf.use_specular_workflow:
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
            shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*surf.specular_color)
            )

    def _wire_textures(self, stage, mat_path, shader, surf: UsdSurface) -> None:
        st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        for input_name, filename in surf.textures.items():
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/{_valid(input_name)}_tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Path(filename).name)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            if input_name in _COLOR_INPUTS:
                out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                in_type = Sdf.ValueTypeNames.Color3f
            elif input_name in _NORMAL_INPUTS:
                # Tangent-space normal maps must be read raw and remapped [0,1] -> [-1,1].
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                    Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
                )
                tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                    Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
                )
                out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                in_type = Sdf.ValueTypeNames.Normal3f
            else:  # scalar channels: roughness, metallic, opacity
                # Non-color data must be read raw; the default "auto" marks 8-bit
                # 3/4-channel textures as sRGB and would gamma-decode linear values.
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                out = tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
                in_type = Sdf.ValueTypeNames.Float
            shader.CreateInput(input_name, in_type).ConnectToSource(out)

    # ---------------------------------------------------------------- physics
    def _apply_physics(self, stage, root_name, root) -> None:
        if self.opts.add_free_joint:  # dynamic rigid body
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            if self.opts.density is not None:
                mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
                mass_api.CreateDensityAttr(float(self.opts.density))

        if self.asset.decomp_success and self.asset.collision_parts:
            UsdGeom.Scope.Define(stage, f"/{root_name}/Collision")
            parts = sorted(
                self.asset.collision_parts, key=lambda x: int(x.stem.split("_")[-1])
            )
            for i, part in enumerate(parts):
                self._build_collision_part(stage, root_name, part, i)
        else:
            for mesh_path in self._visual_meshes:
                prim = stage.GetPrimAtPath(mesh_path)
                UsdPhysics.CollisionAPI.Apply(prim)
                mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mc.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)

    def _build_collision_part(self, stage, root_name, part: Path, index: int) -> None:
        geom = trimesh.load(part.as_posix(), force="mesh")
        assert isinstance(geom, trimesh.Trimesh)
        path = f"/{root_name}/Collision/{_valid(part.stem)}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        self._set_geometry(mesh, geom)
        UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.guide)
        r, g, b = collision_color(index)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(r, g, b)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    # -------------------------------------------------------------- save/check
    def save(self) -> List[Path]:
        if self.stage is None:
            self.build()
        assert self.stage is not None
        ext = "usdc" if self.opts.usd_binary else "usda"
        path = self.asset.work_dir / f"{self.asset.name}.{ext}"
        self.stage.Export(path.as_posix())
        logging.info(f"Saved USD to {path}")
        self._paths = [path]
        return [path]

    def validate(self) -> None:
        """Open the stage and run USD compliance checks; raises on failure."""
        if not self._paths:
            self.save()
        path = self._paths[0]
        stage = Usd.Stage.Open(path.as_posix())
        if stage is None:
            raise RuntimeError(f"USD stage failed to open: {path}")
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"USD stage has no meshes: {path}")
        for prim in meshes:
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts:
                raise RuntimeError(f"Mesh {prim.GetPath()} has no points")
            if idx and max(idx) >= len(pts):
                raise RuntimeError(f"Mesh {prim.GetPath()} has out-of-range face index")
        if hasattr(UsdUtils, "ComplianceChecker"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                checker = UsdUtils.ComplianceChecker()
                checker.CheckCompliance(path.as_posix())
            failed = list(checker.GetFailedChecks()) + list(checker.GetErrors())
        else:
            # OpenUSD 26.08 replaced ComplianceChecker with UsdValidation.
            from pxr import UsdValidation

            metadata = [
                item
                for item in UsdValidation.ValidationRegistry().GetAllValidatorMetadata()
                if item.name != "usdUtilsValidators:RootPackageValidator"
            ]
            errors = UsdValidation.ValidationContext(metadata=metadata).Validate(stage)
            failed = [
                error.GetErrorAsString()
                for error in errors
                if error.GetType() == UsdValidation.ValidationErrorType.Error
            ]
        if failed:
            raise RuntimeError(f"USD compliance failed for {path}: {failed}")
        from termcolor import cprint

        cprint(f"{self.asset.name} (usd) passed compliance!", "green")
