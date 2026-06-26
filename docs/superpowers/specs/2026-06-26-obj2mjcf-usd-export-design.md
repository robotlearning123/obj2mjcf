# Design: obj2mjcf multi-format export — physics-ready USD + PBR materials

Date: 2026-06-26
Status: Draft (awaiting review)
Repo: `robotlearning123/obj2mjcf` (fork of `kevinzakka/obj2mjcf`), branch `feat/usd-export-pbr`

## 1. Context & motivation

`obj2mjcf` turns a composite Wavefront OBJ into a MuJoCo asset. Its pipeline
(`cli.py::process_obj`) already does the hard, reusable work — splitting the OBJ by
material into per-material sub-meshes, parsing the MTL into `Material` objects, copying
and converting textures, and optionally running CoACD convex decomposition — but then
hardcodes a single emitter, `MJCFBuilder`, producing only MuJoCo XML.

Downstream consumers (the OceanScale underwater-robotics stack: Isaac Sim 6, Isaac Lab 3,
Newton 1.3, OpenUSD as the scene format) need the *same* processed asset as **USD**, ideally
physics-ready so it drops into `newton.ModelBuilder.add_usd()` / Isaac directly. This spec
adds a second emitter (USD) and upgrades the material model to PBR, behind a small builder
abstraction so the MJCF path is preserved and future emitters (glTF/URDF) become trivial.

## 2. Goals / non-goals

**Goals**
- G1. Add a USD emitter producing a self-contained `.usdc`/`.usda` stage: geometry +
  `UsdPreviewSurface` materials + textures, with `metersPerUnit`/`upAxis` metadata.
- G2. Optional physics on the USD (`UsdPhysics` RigidBody + Collision + Mass) so it loads in
  Newton/Isaac.
- G3. Upgrade the material model to PBR (normal/metallic/roughness/emissive maps + scalars),
  enriching both USD and MJCF.
- G4. Introduce a `ProcessedAsset` intermediate + emitter registry + a library `convert()`
  API, and make output deterministic.
- G5. Preserve 100% backward compatibility: existing CLI flags and MJCF output unchanged;
  both existing tests stay green.

**Non-goals (YAGNI / out of scope)**
- N1. Articulation / joints / kinematic trees. `obj2mjcf` is a single-asset OBJ→sim tool;
  articulated assets are owned by the `cad-to-sim` / `photo-to-sim` pipelines.
- N2. glTF / URDF emitters (the seam makes them easy later; not built now).
- N3. Non-convex / primitive-fit collision (box/sphere/capsule fitting) beyond CoACD convex.
- N4. Multiple MTL files per OBJ (pre-existing limitation, left as-is).

## 3. Verified facts (provenance)

All checked by execution on 2026-06-26 in the OceanScale `.venv` (`/home/robot/workspace/46-marine/.venv`):

| Fact | Result | How verified |
|------|--------|--------------|
| `pxr` USD available | USD `(0, 26, 3)`; `UsdGeom`/`UsdShade`/`UsdPhysics`/`UsdUtils` all import | `python -c "from pxr import ..."` |
| Portable compliance check | `UsdUtils.ComplianceChecker` present (pip `usd-core` ships no `usdchecker` CLI on PATH) | import + attr access |
| Physics token | `UsdPhysics.Tokens.convexDecomposition == "convexDecomposition"` | attr read |
| MuJoCo PBR material | MuJoCo `3.8.1` `<material metallic= roughness=>` compiles | `MjModel.from_xml_string(...)` |
| Newton USD load | Newton `1.3.0` has `ModelBuilder.add_usd` | `hasattr(newton.ModelBuilder, "add_usd")` |
| Tool baseline | fork == upstream `kevinzakka/obj2mjcf` (0 fork-specific commits); `pytest` → 2 passed; `groups.obj` converts + MuJoCo compiles | clone + `pytest`, real run |

Implication: USD geometry must be **baked** (USD cannot natively reference `.obj`); the
compliance oracle uses the Python `ComplianceChecker` API; Newton `add_usd()` is a real,
gated physics oracle.

## 4. Architecture — the builder seam

### 4.1 `ProcessedAsset` (new, `obj2mjcf/asset.py`)
The explicit intermediate both emitters consume:

```python
@dataclass(frozen=True)
class ProcessedAsset:
    name: str                      # OBJ stem
    work_dir: Path
    submeshes: list[SubMesh]       # one per material group (geometry + material ref)
    materials: list[Material]
    collision_parts: list[Path]    # CoACD .obj parts, or []
    decomp_success: bool
    meters_per_unit: float = 1.0
    up_axis: str = "Z"             # "Z" (MuJoCo convention) or "Y"

@dataclass(frozen=True)
class SubMesh:
    obj_path: Path                 # exported per-material .obj (MJCF references this)
    geometry: trimesh.Trimesh      # in-memory, for baking into USD
    material_name: str | None
```

`process_obj` builds `ProcessedAsset` from the existing per-material export loop (no new
mesh IO — the `trimesh` geometry is already in memory), then dispatches to emitters.

### 4.2 Emitter interface + registry
```python
class AssetEmitter(Protocol):
    format: str
    def build(self, asset: ProcessedAsset, opts: EmitOpts) -> None: ...
    def save(self) -> list[Path]: ...
    def validate(self) -> None: ...          # raises on failure; no-op allowed

EMITTERS: dict[str, type[AssetEmitter]] = {"mjcf": MJCFBuilder, "usd": USDBuilder}
```
`MJCFBuilder` is refactored to consume `ProcessedAsset` (behavior-preserving) and to expose
`validate()` wrapping its existing MuJoCo compile+step. `USDBuilder` is new.

### 4.3 Library API (new, `obj2mjcf/__init__.py` re-export)
```python
def convert(
    obj_path: str | Path,
    *,
    export: Sequence[str] = ("mjcf",),
    decompose: bool = False,
    coacd_args: CoacdArgs | None = None,
    texture_resize_percent: float = 1.0,
    add_free_joint: bool = False,
    usd_physics: bool = False,
    density: float | None = None,
    meters_per_unit: float = 1.0,
    up_axis: str = "Z",
    usd_binary: bool = True,         # .usdc; False -> .usda
    overwrite: bool = True,
    validate: bool = False,
) -> dict[str, list[Path]]:          # {"mjcf": [...], "usd": [...]}
```
This is the clean entry OceanScale calls — no more constructing the `tyro` `Args` dataclass.

### 4.4 CLI (back-compat)
- New: `--export mjcf,usd` (comma list, **default `mjcf`**), `--usd-physics`, `--density FLOAT`,
  `--meters-per-unit FLOAT`, `--up-axis {Z,Y}`, `--usd-ascii` (write `.usda`).
- Preserved: `--save-mjcf` → ensures `"mjcf"` in the export set; `--compile-model` →
  calls each selected emitter's `validate()` (MJCF compiles+steps; USD runs ComplianceChecker).
  All existing flags keep their meaning.

### 4.5 Determinism (fixes a real bug)
`mjcf_builder.add_collision_geometries` colors collision geoms with `np.random.rand(3)` →
non-reproducible output across runs. Replace with a deterministic palette
`hue = (index * 0.61803398875) % 1.0` → RGB via `colorsys.hsv_to_rgb(hue, 0.6, 0.9)`. Same
palette used for USD collision display color. Output becomes byte-stable.

## 5. Material model — PBR

### 5.1 Parser extension (`material.py`, `constants.py`)
Extend `MTL_FIELDS` and `Material` with the common PBR/`map_*` extensions:
`Pr` (roughness), `Pm` (metallic), `Ke` (emissive), `Ns`, plus maps `map_Pr`, `map_Pm`,
`map_Ke`, `norm`/`map_Bump` (normal), `map_d` (opacity). Unknown fields ignored (as today).
Existing `mjcf_rgba/mjcf_specular/mjcf_shininess` are **unchanged**.

### 5.2 Consumer accessors
- `Material.mjcf_*` — unchanged. Additionally expose `mjcf_metallic()`/`mjcf_roughness()`
  (wired into `<material>` only when present; MuJoCo 3.8.1 supports them — verified).
- `Material.usd_preview_surface_inputs() -> dict` for `UsdPreviewSurface`:

| USD input | Source | Fallback |
|-----------|--------|----------|
| `diffuseColor` (color3f) | `Kd` | `(0.8,0.8,0.8)` |
| `opacity` (float) | `d`, else `1 - Tr` | `1.0` |
| `metallic` (float) | `Pm` | `0.0` |
| `roughness` (float) | `Pr`, else from `Ns` (§5.3) | `0.5` |
| `emissiveColor` (color3f) | `Ke` | `(0,0,0)` |
| `useSpecularWorkflow` + `specularColor` | when no `Pm`: workflow=1, `specularColor=Ks` | workflow=0 |
| texture inputs | `map_Kd`→diffuse, `norm`→normal, `map_Pr`→roughness, `map_Pm`→metallic, `map_Ke`→emissive | omitted if absent |

### 5.3 Phong→PBR roughness (documented heuristic)
When only Phong shininess `Ns` is present:
`roughness = clamp(sqrt(2.0 / (Ns + 2.0)), 0.0, 1.0)` — the standard Blinn-Phong-exponent →
GGX-roughness approximation. Labeled in code as a heuristic with this citation.

## 6. USDBuilder (`obj2mjcf/usd_builder.py`)

### 6.1 Stage layout
```
/<Asset>                       Xform (defaultPrim)        [+ RigidBodyAPI if physics]
  /Looks
    /<mat>                     Material
      /Shader                  Shader  (UsdPreviewSurface)
      /diffuseTex …            Shader  (UsdUVTexture)      [if map_Kd]
      /stReader                Shader  (UsdPrimvarReader_float2)
  /Geom
    /<stem>_<i>                Mesh (visual)  -> material:binding, purpose=render
  /Collision                   [if physics]
    /<stem>_collision_<j>      Mesh  CollisionAPI+MeshCollisionAPI(convexHull), purpose=guide
```
Stage metadata: `UsdGeom.SetStageUpAxis(stage, up_axis)`,
`UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)`, `defaultPrim = /<Asset>`.

### 6.2 Geometry (baked from in-memory trimesh)
Per `SubMesh.geometry`: `points = vertices`; `faceVertexCounts = [3]*len(faces)` (trimesh is
triangulated); `faceVertexIndices = faces.flatten()`; `extent = [bbox_min, bbox_max]`. If
`geometry.visual.uv` present → `primvars:st` (`TexCoord2fArray`, interpolation `vertex`) via
`UsdGeomPrimvarsAPI`. If vertex normals present → `normals` attr, interpolation `vertex`
(else omitted; renderer computes). Missing UV ⇒ skip texture readers gracefully.

### 6.3 Materials & textures
One `UsdShade.Material`+`UsdPreviewSurface` per `Material`, bound to the matching mesh via
`UsdShade.MaterialBindingAPI`. Texture files copied into `work_dir` (reuse existing copy
logic; USD keeps original format — no MuJoCo PNG-only constraint). `UsdUVTexture` fed by a
shared `UsdPrimvarReader_float2` on `st`.

### 6.4 Physics (`--usd-physics`)
Mirrors the MJCF builder's collision logic exactly. Body semantics match MJCF (a jointless
MJCF body is welded to the world = static; a `freejoint` body is dynamic):
- **Dynamic** when `add_free_joint`: root `<Asset>` gets `UsdPhysics.RigidBodyAPI`
  (+ `UsdPhysics.MassAPI` with `density` when given; else the engine derives mass from
  geometry).
- **Static** otherwise: no `RigidBodyAPI` — colliders only (a fixed obstacle).
- Collision geometry (both cases):
  - If `decomp_success`: emit each CoACD part as a collision-only `Mesh` under `/Collision`
    with `CollisionAPI` + `MeshCollisionAPI(approximation="convexHull")`, `purpose=guide`.
  - Else: apply `CollisionAPI` + `MeshCollisionAPI(approximation="convexDecomposition")` to
    the visual mesh(es) (engine decomposes at load). Token verified to exist.

### 6.5 Output
Default `.usdc` (crate binary — what Isaac/Omniverse expect, compact, deterministic).
`--usd-ascii` / `convert(usd_binary=False)` → `.usda` (diff-able; used by tests).

## 7. Testing & quality gates — real oracles only

### 7.1 Fork unit/integration suite (`pytest tests/`, runs in CI with `usd-core` installed)
| Test | Oracle (checkable quantity) |
|------|------------------------------|
| `test_runs_without_error` (existing) | MJCF compiles — **kept green** |
| `test_parse_mtl_line` (existing) | regex parse — **kept green** |
| `test_mjcf_backcompat_golden` | MJCF for `groups.obj` byte-identical to committed golden (after determinism fix) |
| `test_usd_stage_opens` | `Usd.Stage.Open` ok; ≥1 `UsdGeom.Mesh`; every mesh `points` non-empty; `max(faceVertexIndices) < len(points)` |
| `test_usd_compliance` | `UsdUtils.ComplianceChecker().CheckCompliance(stage)`; `GetFailedChecks() == []` |
| `test_usd_geometry_parity` | USD total vert/face counts == source trimesh; bbox within `1e-6` |
| `test_usd_material_binding` | every Mesh resolves a `UsdShade` material; `diffuseColor` == `Kd` |
| `test_pbr_mtl_mapping` | synthetic MTL `Pr/Pm/Ke/norm` → correct shader inputs/texture prims |
| `test_phong_to_pbr_math` | `Ns=k` → `roughness == sqrt(2/(k+2))` |
| `test_determinism` | two runs → byte-identical `.usda` and `.xml` |
| `test_usd_physics_apis` | with `usd_physics`: `RigidBodyAPI` on root, `CollisionAPI` on collision prims, `MassAPI.density` set |

### 7.2 Gated physics oracle (OceanScale env; mark `@pytest.mark.slow`/`integration`)
`test_newton_loads_usd`: `newton.ModelBuilder().add_usd(<physics.usda>)` succeeds and yields
`>0` bodies and `>0` collision shapes. The real "does it actually load in the target engine"
oracle (Newton 1.3.0 `add_usd` verified).

### 7.3 Visual gate (project rule — both strong models)
Render the USD (Isaac Sim 6 RTX / `usdrecord` if available) and the MJCF (`mujoco`) of the
same asset; **cx (GPT-5.5) and Opus 4.8 each open the frames** and confirm geometry/material
parity (tolerance documented — UsdPreviewSurface ≠ MuJoCo shading exactly). Receipt written to
`artifacts/verify/obj2mjcf-usd-<date>.md` with both verdicts.

### 7.4 Static + CI
`ruff check`, `ruff format --check`, `mypy obj2mjcf` clean. Add `.github/workflows/test.yml`:
matrix py3.10/3.11/3.12, `pip install -e .[test,usd]`, run ruff+mypy+pytest.

### 7.5 Cross-model code review
cx (GPT-5.5) reviews the full diff (writer ≠ reviewer); Opus 4.8 independent second; both
verdicts recorded before any PR to the fork. Per `~/.claude/rules/cross-model-review.md`.

## 8. File-by-file change plan
| File | Change |
|------|--------|
| `obj2mjcf/asset.py` | **new** — `ProcessedAsset`, `SubMesh`, `EmitOpts` |
| `obj2mjcf/emitters.py` | **new** — `AssetEmitter` protocol, `EMITTERS` registry, `select_emitters` |
| `obj2mjcf/usd_builder.py` | **new** — `USDBuilder` (geometry/materials/physics/validate) |
| `obj2mjcf/material.py` | PBR fields + `usd_preview_surface_inputs()` + `mjcf_metallic/roughness` |
| `obj2mjcf/constants.py` | extend `MTL_FIELDS`; add PBR/USD constants |
| `obj2mjcf/mjcf_builder.py` | consume `ProcessedAsset`; deterministic colors; `validate()`; optional PBR attrs |
| `obj2mjcf/cli.py` | build `ProcessedAsset`; `--export`/USD flags; emitter dispatch; keep back-compat |
| `obj2mjcf/__init__.py` | export `convert`; bump `__version__` |
| `setup.py` | add `[usd]` extra (`usd-core>=24.0`); add to `testing`/`dev` |
| `tests/test_usd.py` | **new** — §7.1 USD/PBR tests |
| `tests/test_determinism.py` | **new** — byte-identical runs |
| `tests/integration/test_newton_load.py` | **new**, gated — §7.2 |
| `tests/data/groups_golden.xml` | **new** — MJCF golden |
| `.github/workflows/test.yml` | **new** — CI matrix |
| `README.md` | document `--export usd`, physics flags, `convert()` API |

## 9. Phasing & PR plan (small, reviewable, each independently green)
- **PR 1 — Builder seam (no behavior change):** `ProcessedAsset`/emitters/`convert()` API,
  `MJCFBuilder` refactor, deterministic colors, MJCF golden test. Gate: existing tests + golden pass.
- **PR 2 — Visual USD (Phase 1):** `USDBuilder` geometry+materials+units, `--export usd`,
  `[usd]` extra, CI. Gate: stage-opens / compliance / geometry-parity / material-binding tests.
- **PR 3 — PBR materials (Phase 2):** MTL PBR parse, `usd_preview_surface_inputs`, MJCF
  metallic/roughness. Gate: PBR mapping + Phong→PBR math tests.
- **PR 4 — Physics USD (Phase 3):** `UsdPhysics` rigid/collision/mass, `--usd-physics`,
  gated Newton load test + visual gate + receipt. Gate: physics-APIs test + Newton `add_usd`.

Each PR: cx + Opus review before merge; one logical change per PR.

## 10. Risks & mitigations
- **R1 `usd-core` vs full USD:** ComplianceChecker present in `usd-core` (verified); heavy
  Isaac/Newton checks are gated, not in fork CI. Mitigated.
- **R2 Shading fidelity:** UsdPreviewSurface ≠ MuJoCo exactly → visual parity is "close", with
  a documented tolerance; not a byte-render oracle.
- **R3 trimesh UV/normal gaps:** handle missing UV/normals gracefully (skip readers / let USD
  compute); covered by a no-UV test asset.
- **R4 Upstream divergence:** fork tracks `kevinzakka/obj2mjcf`; keep the seam minimal and
  rebase-friendly so upstream changes merge cleanly.
