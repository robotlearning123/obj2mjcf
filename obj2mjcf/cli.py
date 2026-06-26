"""A CLI for processing composite Wavefront OBJ files into MuJoCo and/or USD assets."""

import logging
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import trimesh
import tyro
from PIL import Image
from termcolor import cprint

from obj2mjcf import constants
from obj2mjcf.asset import EmitOpts, ProcessedAsset, SubMesh
from obj2mjcf.emitters import AVAILABLE_FORMATS, select_emitters
from obj2mjcf.material import Material


@dataclass(frozen=True)
class CoacdArgs:
    """Arguments to pass to CoACD.

    Defaults and descriptions are copied from: https://github.com/SarahWeiii/CoACD
    """

    preprocess_resolution: int = 50
    """resolution for manifold preprocess (20~100), default = 50"""
    threshold: float = 0.05
    """concavity threshold for terminating the decomposition (0.01~1), default = 0.05"""
    max_convex_hull: int = -1
    """max # convex hulls in the result, -1 for no maximum limitation"""
    mcts_iterations: int = 100
    """number of search iterations in MCTS (60~2000), default = 100"""
    mcts_max_depth: int = 3
    """max search depth in MCTS (2~7), default = 3"""
    mcts_nodes: int = 20
    """max number of child nodes in MCTS (10~40), default = 20"""
    resolution: int = 2000
    """sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000"""
    pca: bool = False
    """enable PCA pre-processing, default = false"""
    seed: int = 0
    """random seed used for sampling, default = 0"""


@dataclass(frozen=True)
class Args:
    obj_dir: str
    """path to a directory containing obj files. All obj files in the directory will be
    converted"""
    obj_filter: Optional[str] = None
    """only convert obj files matching this regex"""
    export: Optional[str] = None
    """comma-separated output formats to write: mjcf,usd (default: none unless --save-mjcf)"""
    save_mjcf: bool = False
    """save an example XML (MJCF) file (shorthand for adding mjcf to --export)"""
    compile_model: bool = False
    """validate each emitted format (MJCF compiles in MuJoCo; USD runs compliance checks)"""
    verbose: bool = False
    """print verbose output"""
    decompose: bool = False
    """approximate mesh decomposition using CoACD"""
    coacd_args: CoacdArgs = field(default_factory=CoacdArgs)
    """arguments to pass to CoACD"""
    texture_resize_percent: float = 1.0
    """resize the texture to this percentage of the original size"""
    overwrite: bool = False
    """overwrite previous run output"""
    add_free_joint: bool = False
    """add a free joint to the root body (USD: makes the body a dynamic rigid body)"""
    usd_physics: bool = False
    """emit UsdPhysics rigid-body/collision/mass schemas on the USD output"""
    density: Optional[float] = None
    """mass density for the USD rigid body (UsdPhysics.MassAPI)"""
    meters_per_unit: float = 1.0
    """USD stage metersPerUnit metadata"""
    up_axis: str = "Z"
    """USD stage up axis: Z (MuJoCo convention) or Y"""
    usd_ascii: bool = False
    """write USD as ascii .usda instead of binary .usdc"""


def resize_texture(filename: Path, resize_percent) -> None:
    """Resize a texture to a percentage of its original size."""
    if resize_percent == 1.0:
        return
    image = Image.open(filename)
    new_width = int(image.size[0] * resize_percent)
    new_height = int(image.size[1] * resize_percent)
    logging.info(f"Resizing {filename} to {new_width}x{new_height}")
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized.save(filename)


def decompose_convex(
    filename: Path, work_dir: Path, coacd_args: CoacdArgs
) -> List[Path]:
    cprint(f"Decomposing {filename}", "yellow")

    import coacd  # noqa: F401

    obj_file = filename.resolve()
    logging.info(f"Decomposing {obj_file}")

    mesh = trimesh.load(obj_file, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)  # type: ignore

    parts = coacd.run_coacd(mesh=mesh, **asdict(coacd_args))

    out: List[Path] = []
    for i, (vs, fs) in enumerate(parts):
        submesh_name = work_dir / f"{obj_file.stem}_collision_{i}.obj"
        trimesh.Trimesh(vs, fs).export(submesh_name.as_posix())
        out.append(submesh_name)
    return out


def parse_mtl_name(lines: Iterable[str]) -> Optional[str]:
    mtl_regex = re.compile(r"^mtllib\s+(.+?\.mtl)(?:\s*#.*)?\s*\n?$")
    for line in lines:
        match = mtl_regex.match(line)
        if match is not None:
            return match.group(1)
    return None


def copy_textures(
    material: Material, src_dir: Path, work_dir: Path, resize_percent: float
) -> None:
    """Copy all of a material's texture maps into ``work_dir`` (flat) and rewrite paths.

    The diffuse map is converted to PNG (MuJoCo only supports PNG textures) and resized;
    other PBR maps keep their original format for the USD emitter.
    """
    for attr, rel in material.texture_attrs().items():
        texture_path = Path(rel)
        src = src_dir / texture_path
        if not src.exists():
            raise RuntimeError(
                f"The texture file {src} referenced in the MTL file "
                f"{material.name} does not exist"
            )
        dst = work_dir / texture_path.name
        shutil.copy(src, dst)
        if attr == "map_Kd" and texture_path.suffix.lower() in (".jpg", ".jpeg"):
            image = Image.open(dst)
            os.remove(dst)
            dst = (work_dir / texture_path.stem).with_suffix(".png")
            image.save(dst)
            resize_texture(dst, resize_percent)
        elif attr == "map_Kd":
            resize_texture(dst, resize_percent)
        setattr(material, attr, dst.name)


def _parse_mtls(filename: Path) -> List[Material]:
    name = parse_mtl_name(filename.read_text().splitlines(keepends=True))
    if name is None:
        return []
    mtl_filename = filename.parent / name
    if not mtl_filename.exists():
        raise RuntimeError(
            f"The MTL file {mtl_filename.resolve()} referenced in the OBJ file "
            f"{filename} does not exist"
        )
    logging.info(f"Found MTL file: {mtl_filename}")
    lines = [
        line.strip()
        for line in mtl_filename.read_text().splitlines()
        if line.strip() and not line.startswith(constants.MTL_COMMENT_CHAR)
    ]
    sub_mtls: List[List[str]] = []
    for line in lines:
        if line.startswith("newmtl"):
            sub_mtls.append([])
        if sub_mtls:
            sub_mtls[-1].append(line)
    return [Material.from_string(sub_mtl) for sub_mtl in sub_mtls]


def _build_asset(
    filename: Path,
    work_dir: Path,
    decompose: bool,
    coacd_args: CoacdArgs,
    texture_resize_percent: float,
    meters_per_unit: float,
    up_axis: str,
) -> ProcessedAsset:
    collision_parts: List[Path] = []
    if decompose:
        collision_parts = decompose_convex(filename, work_dir, coacd_args)

    mtls = _parse_mtls(filename)
    for mtl in mtls:
        logging.info(f"Found material: {mtl.name}")
        copy_textures(mtl, filename.parent, work_dir, texture_resize_percent)

    logging.info("Processing OBJ file with trimesh")
    mesh = trimesh.load(
        filename,
        split_object=True,
        group_material=True,
        process=False,
        maintain_order=False,
    )

    submeshes: List[SubMesh] = []
    if isinstance(mesh, trimesh.base.Trimesh):
        savename = work_dir / f"{filename.stem}.obj"
        logging.info(f"Saving mesh {savename}")
        mesh.export(savename.as_posix(), include_texture=True, header=None)
        submeshes.append(SubMesh(savename, mesh, mtls[0].name if mtls else None))
        # Edge case: the MTL declares many materials but the OBJ uses only one.
        if len(mtls) > 1:
            used = _referenced_material(filename)
            mtls = [m for m in mtls if m.name == used] or mtls[:1]
            submeshes[0].material_name = mtls[0].name
    else:
        logging.info("Grouping and saving submeshes by material")
        for i, (name, geom) in enumerate(mesh.geometry.items()):  # type: ignore[attr-defined]
            savename = work_dir / f"{filename.stem}_{i}.obj"
            logging.info(f"Saving submesh {savename}")
            geom.export(savename.as_posix(), include_texture=True, header=None)
            submeshes.append(SubMesh(savename, geom, name))

    mtls = list({m.name: m for m in mtls}.values())

    # Remove MTL/texture artifacts trimesh may have written during export.
    for f in work_dir.glob("**/*"):
        if f.is_file() and "material_0" in f.name and not f.name.endswith(".png"):
            f.unlink()

    return ProcessedAsset(
        name=filename.stem,
        work_dir=work_dir,
        submeshes=submeshes,
        materials=mtls,
        collision_parts=collision_parts,
        decomp_success=bool(collision_parts),
        meters_per_unit=meters_per_unit,
        up_axis=up_axis,
    )


def _referenced_material(filename: Path) -> Optional[str]:
    for line in filename.read_text().splitlines():
        if line.startswith("usemtl"):
            return line.split()[1]
    return None


def _prepare_work_dir(filename: Path, overwrite: bool) -> Optional[Path]:
    work_dir = filename.parent / filename.stem
    if work_dir.exists():
        if not overwrite:
            proceed = input(
                f"{work_dir.resolve()} already exists, maybe from a previous run? "
                "Proceeding will overwrite it.\nDo you wish to continue [y/n]: "
            )
            if proceed.lower() != "y":
                return None
        shutil.rmtree(work_dir)
    work_dir.mkdir(exist_ok=True)
    logging.info(f"Saving processed meshes to {work_dir}")
    return work_dir


def _emit(
    asset: ProcessedAsset,
    write: Sequence[str],
    validate: bool,
    emit_opts: EmitOpts,
) -> Dict[str, List[Path]]:
    write_set = set(write)
    emitters = select_emitters([f for f in AVAILABLE_FORMATS if f in write_set])
    produced: Dict[str, List[Path]] = {}
    for fmt, cls in emitters.items():
        builder = cls(asset, emit_opts)
        builder.build()
        if validate:
            builder.validate()
        produced[fmt] = builder.save()
    return produced


def _parse_formats(export: Optional[str]) -> List[str]:
    if not export:
        return []
    formats = [f.strip() for f in export.split(",") if f.strip()]
    unknown = [f for f in formats if f not in AVAILABLE_FORMATS]
    if unknown:
        raise ValueError(
            f"Unknown export format(s): {unknown}. Available: {list(AVAILABLE_FORMATS)}"
        )
    return formats


def process_obj(filename: Path, args: Args) -> Dict[str, List[Path]]:
    work_dir = _prepare_work_dir(filename, args.overwrite)
    if work_dir is None:
        return {}

    asset = _build_asset(
        filename,
        work_dir,
        args.decompose,
        args.coacd_args,
        args.texture_resize_percent,
        args.meters_per_unit,
        args.up_axis,
    )

    write = _parse_formats(args.export)
    if args.save_mjcf and "mjcf" not in write:
        write.append("mjcf")

    emit_opts = EmitOpts(
        add_free_joint=args.add_free_joint,
        usd_physics=args.usd_physics,
        density=args.density,
        usd_binary=not args.usd_ascii,
    )
    produced = _emit(asset, write, args.compile_model, emit_opts)
    # Back-compat: --compile-model with nothing written compiles the MJCF in memory.
    if args.compile_model and not write:
        compile_only = select_emitters(["mjcf"])["mjcf"](asset, emit_opts)
        compile_only.build()
        compile_only.validate()
    return produced


def convert(
    obj_path,
    *,
    export: Sequence[str] = ("mjcf",),
    decompose: bool = False,
    coacd_args: Optional[CoacdArgs] = None,
    texture_resize_percent: float = 1.0,
    add_free_joint: bool = False,
    usd_physics: bool = False,
    density: Optional[float] = None,
    meters_per_unit: float = 1.0,
    up_axis: str = "Z",
    usd_binary: bool = True,
    overwrite: bool = True,
    validate: bool = False,
) -> Dict[str, List[Path]]:
    """Convert a single composite OBJ to the requested formats.

    Returns a mapping of format name to the list of files written.
    """
    filename = Path(obj_path)
    work_dir = _prepare_work_dir(filename, overwrite)
    if work_dir is None:
        return {}
    asset = _build_asset(
        filename,
        work_dir,
        decompose,
        coacd_args or CoacdArgs(),
        texture_resize_percent,
        meters_per_unit,
        up_axis,
    )
    emit_opts = EmitOpts(
        add_free_joint=add_free_joint,
        usd_physics=usd_physics,
        density=density,
        usd_binary=usd_binary,
    )
    return _emit(asset, list(export), validate, emit_opts)


def main() -> None:
    args = tyro.cli(Args, description=__doc__)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    obj_files = list(Path(args.obj_dir).glob("*.obj"))
    if args.obj_filter is not None:
        obj_files = [
            x for x in obj_files if re.search(args.obj_filter, x.name) is not None
        ]

    for obj_file in obj_files:
        cprint(f"Processing {obj_file}", "yellow")
        process_obj(obj_file, args)
