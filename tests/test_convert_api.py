import pathlib

import pytest

from obj2mjcf import convert

_THIS_DIR = pathlib.Path(__file__).parent.absolute()


def _groups(tmp_path: pathlib.Path) -> pathlib.Path:
    obj = tmp_path / "groups.obj"
    obj.write_bytes((_THIS_DIR / "groups.obj").read_bytes())
    return obj


def test_export_string_is_treated_as_single_format(tmp_path) -> None:
    # A bare string must NOT iterate into characters and silently produce nothing.
    out = convert(_groups(tmp_path), export="usd", usd_binary=False)
    assert list(out.keys()) == ["usd"]
    assert out["usd"][0].exists()


def test_unknown_export_format_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown export format"):
        convert(_groups(tmp_path), export=("bogus",))


def test_unknown_export_in_string_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unknown export format"):
        convert(_groups(tmp_path), export="mjcf,usd")  # not split by convert()


def test_invalid_up_axis_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="up_axis"):
        convert(_groups(tmp_path), export=("usd",), up_axis="garbage")
