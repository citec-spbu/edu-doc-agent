"""Реальные РПД/РПП из data/ для тестов: docx превращаются в Markdown текущим парсером.

Папка data/ в git не входит. Если docx нет, `REAL_MD_DIR` пуст и тесты на реальных данных
пропускаются. Markdown собирается один раз за сессию во временной папке, поэтому тесты всегда
видят вывод актуальной версии `parsing/`, а не устаревшие md.
"""

import os
import tempfile
import unicodedata
from pathlib import Path

from edu_doc_agent.parsing import parse_document

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
DOCX_DIR = DATA_DIR / "РПД и РПП"


def _convert() -> list[Path]:
    docx = (
        sorted(p for p in DOCX_DIR.rglob("*.docx") if not p.name.startswith("~$"))
        if DOCX_DIR.exists()
        else []
    )
    if not docx:
        return []
    out = Path(tempfile.mkdtemp(prefix="edu_doc_agent_md_"))
    result = []
    for path in docx:
        target = out / (unicodedata.normalize("NFC", path.stem) + ".md")
        target.write_text(parse_document(path).to_markdown(), encoding="utf-8")
        result.append(target)
    return result


REAL_MD: list[Path] = _convert()


def find_md(fragment: str) -> Path | None:
    """Первый md по подстроке имени (имена в NFC: на macOS файловая система отдаёт NFD)."""
    return next((p for p in REAL_MD if fragment in p.name), None)
