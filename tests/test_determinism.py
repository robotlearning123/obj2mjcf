import pathlib

from obj2mjcf import convert
from obj2mjcf.emitters import collision_color

_THIS_DIR = pathlib.Path(__file__).parent.absolute()


def _convert_groups(root: pathlib.Path):
    root.mkdir(parents=True, exist_ok=True)
    obj = root / "groups.obj"
    obj.write_bytes((_THIS_DIR / "groups.obj").read_bytes())
    return convert(obj, export=("mjcf", "usd"), usd_binary=False)


def test_collision_color_is_deterministic() -> None:
    assert collision_color(3) == collision_color(3)
    assert collision_color(0) != collision_color(1)


def test_outputs_are_byte_identical_across_runs(tmp_path) -> None:
    out1 = _convert_groups(tmp_path / "run1")
    out2 = _convert_groups(tmp_path / "run2")
    for fmt in ("mjcf", "usd"):
        assert out1[fmt][0].read_bytes() == out2[fmt][0].read_bytes(), fmt


def test_mjcf_backcompat_golden(tmp_path) -> None:
    obj = tmp_path / "groups.obj"
    obj.write_bytes((_THIS_DIR / "groups.obj").read_bytes())
    out = convert(obj, export=("mjcf",))
    produced = out["mjcf"][0].read_text()
    golden = (_THIS_DIR / "data" / "groups_golden.xml").read_text()
    assert produced == golden
