"""Auto-Research Synthesizer package."""

from typing import Any


__app_name__ = "Auto-Research Synthesizer"
__version__ = "0.1.0"

__all__ = [
    "__app_name__",
    "__version__",
    "build_graph",
    "run",
]


def __getattr__(name: str) -> Any:
    """Lazily expose graph helpers without importing LLM dependencies upfront."""
    if name in {"build_graph", "run"}:
        from ars.core import graph

        return getattr(graph, name)

    raise AttributeError(f"module 'ars' has no attribute {name!r}")
