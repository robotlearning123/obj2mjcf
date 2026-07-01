import colorsys
from pathlib import Path
from typing import Dict, List, Protocol, Type, cast

from obj2mjcf.asset import EmitOpts, ProcessedAsset

# Golden-ratio hue stepping: a fixed, well-spread palette that is fully deterministic
# (replaces the previous np.random collision colors so output is reproducible).
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def collision_color(index: int) -> tuple:
    """Deterministic RGB color for the ``index``-th collision geometry."""
    hue = (index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.6, 0.9)


class AssetEmitter(Protocol):
    """Emitters consume a :class:`ProcessedAsset` and write one output format."""

    format: str

    def __init__(self, asset: ProcessedAsset, opts: EmitOpts) -> None: ...

    def build(self) -> None: ...

    def save(self) -> List[Path]: ...

    def validate(self) -> None:
        """Raise if the produced output is invalid (no-op allowed)."""
        ...


AVAILABLE_FORMATS = ("mjcf", "usd")


def _load_emitter(fmt: str) -> Type[AssetEmitter]:
    """Import an emitter lazily so the MJCF path never requires the USD dependency."""
    if fmt == "mjcf":
        from obj2mjcf.mjcf_builder import MJCFBuilder

        return cast(Type[AssetEmitter], MJCFBuilder)
    if fmt == "usd":
        try:
            from obj2mjcf.usd_builder import USDBuilder
        except ImportError as e:  # pragma: no cover - exercised only without usd-core
            raise ImportError(
                "USD export requires the [usd] extra: pip install 'obj2mjcf[usd]'"
            ) from e

        return cast(Type[AssetEmitter], USDBuilder)
    raise ValueError(
        f"Unknown export format: {fmt!r}. Available: {list(AVAILABLE_FORMATS)}"
    )


def select_emitters(formats) -> Dict[str, Type[AssetEmitter]]:
    return {fmt: _load_emitter(fmt) for fmt in formats}
