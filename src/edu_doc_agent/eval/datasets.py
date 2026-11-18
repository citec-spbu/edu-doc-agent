"""Пути локальных эталонов реальных документов; сами файлы не входят в Git."""

import os
from pathlib import Path


def local_gold(name: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return Path(os.environ.get("DATA_DIR") or root / "data") / "evaluation" / name
