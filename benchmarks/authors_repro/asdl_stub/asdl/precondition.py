"""The three names ``train.py`` imports from ``asdl.precondition``. See ``asdl/__init__.py``."""

from __future__ import annotations

from typing import Any

_MESSAGE = (
    "asdl is not installed: benchmarks/authors_repro/asdl_stub only satisfies train.py's "
    "module-level import so that the AdaFisher and Adam arms can run. Reaching this code means a "
    "Shampoo or K-FAC arm was launched, which needs the real "
    "git+https://github.com/kazukiosawa/asdl@011a942 from the authors' requirements.txt."
)


class PreconditioningConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_MESSAGE)


class ShampooGradientMaker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_MESSAGE)


class KfacGradientMaker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_MESSAGE)
