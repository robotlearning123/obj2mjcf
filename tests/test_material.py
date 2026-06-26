import math

from obj2mjcf.material import Material


def test_from_string_parses_phong_and_pbr() -> None:
    lines = [
        "newmtl steel",
        "Kd 0.20 0.20 0.22",
        "Ks 0.9 0.9 0.9",
        "Ns 250",
        "Pr 0.35",
        "Pm 1.0",
        "Ke 0.0 0.0 0.0",
        "d 0.8",
        "map_Kd albedo.png",
        "norm normal.png",
        "map_Pr rough.png",
    ]
    m = Material.from_string(lines)
    assert m.name == "steel"
    assert m.Kd == "0.20 0.20 0.22"
    assert m.Pr == "0.35"
    assert m.Pm == "1.0"
    assert m.map_Kd == "albedo.png"
    assert m.map_norm == "normal.png"  # 'norm' keyword aliases to map_norm
    assert m.map_Pr == "rough.png"


def test_from_string_exact_keyword_match_not_prefix() -> None:
    # 'disp' must NOT be captured by the 'd' (dissolve) field.
    m = Material.from_string(["newmtl m", "disp displacement.png", "d 0.5"])
    assert m.d == "0.5"


def test_phong_to_pbr_roughness() -> None:
    for ns in (10.0, 100.0, 500.0):
        m = Material(name="m", Ns=str(ns))
        assert math.isclose(m.usd_roughness(), math.sqrt(2.0 / (ns + 2.0)))


def test_pr_overrides_ns_for_roughness() -> None:
    m = Material(name="m", Ns="250", Pr="0.4")
    assert math.isclose(m.usd_roughness(), 0.4)


def test_usd_preview_surface_metallic_workflow() -> None:
    m = Material(name="m", Kd="0.1 0.2 0.3", Pm="1.0", Pr="0.25", d="0.5")
    surf = m.usd_preview_surface()
    assert surf.diffuse_color == (0.1, 0.2, 0.3)
    assert surf.metallic == 1.0
    assert math.isclose(surf.roughness, 0.25)
    assert math.isclose(surf.opacity, 0.5)
    assert surf.use_specular_workflow is False  # Pm given -> metallic workflow


def test_usd_preview_surface_specular_workflow() -> None:
    m = Material(name="m", Ks="0.8 0.8 0.8")  # no Pm -> specular workflow
    surf = m.usd_preview_surface()
    assert surf.use_specular_workflow is True
    assert surf.specular_color == (0.8, 0.8, 0.8)


def test_metallic_map_disables_specular_workflow() -> None:
    # A metallic texture (without a scalar Pm) still means metallic workflow.
    m = Material(name="m", Ks="0.8 0.8 0.8", map_Pm="metal.png")
    assert m.usd_preview_surface().use_specular_workflow is False


def test_opacity_from_tr() -> None:
    m = Material(name="m", Tr="0.25")
    assert math.isclose(m.usd_opacity(), 0.75)


def test_mjcf_pbr_attrs_only_when_present() -> None:
    assert Material(name="m").mjcf_metallic() is None
    assert Material(name="m").mjcf_roughness() is None
    assert Material(name="m", Pm="0.5").mjcf_metallic() == "0.5"
    assert Material(name="m", Pr="0.3").mjcf_roughness() == "0.3"
