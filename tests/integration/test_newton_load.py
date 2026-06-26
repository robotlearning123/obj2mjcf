"""Gated integration oracle: the physics USD must load in NVIDIA Newton.

Skipped unless ``newton`` is importable (it is not part of the fork's lightweight CI;
it runs in an environment that also has Newton/Warp, e.g. the OceanScale venv).
"""

import pathlib

import pytest

newton = pytest.importorskip("newton")

from obj2mjcf import convert  # noqa: E402

_TESTS_DIR = pathlib.Path(__file__).parent.parent.absolute()


@pytest.mark.slow
def test_newton_loads_physics_usd(tmp_path) -> None:
    obj = tmp_path / "groups.obj"
    obj.write_bytes((_TESTS_DIR / "groups.obj").read_bytes())
    out = convert(
        obj,
        export=("usd",),
        usd_physics=True,
        add_free_joint=True,
        density=400.0,
        usd_binary=False,
    )
    builder = newton.ModelBuilder()
    builder.add_usd(out["usd"][0].as_posix())
    model = builder.finalize()
    assert model.body_count >= 1
    assert model.shape_count >= 1
