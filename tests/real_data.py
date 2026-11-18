"""Реальные РПД/РПП из data/ для тестов: docx превращаются в Markdown текущим парсером.

Папка data/ в git не входит. Если docx нет, `REAL_MD_DIR` пуст и тесты на реальных данных
пропускаются. Markdown собирается один раз за сессию во временной папке, поэтому тесты всегда
видят вывод актуальной версии `parsing/`, а не устаревшие md.
"""

import atexit
import os
import shutil
import tempfile
import unicodedata
import zipfile
from pathlib import Path

from edu_doc_agent.parsing import parse_document

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR") or REPO_ROOT / "data")
DOCX_DIR = DATA_DIR / "РПД и РПП"
REAL_DOCX_BY_MD: dict[Path, Path] = {}


def _convert() -> list[Path]:
    docx = (
        sorted(p for p in DOCX_DIR.rglob("*.docx") if not p.name.startswith("~$"))
        if DOCX_DIR.exists()
        else []
    )
    if not docx:
        return []
    out = Path(tempfile.mkdtemp(prefix="edu_doc_agent_md_"))
    atexit.register(shutil.rmtree, out, ignore_errors=True)
    result = []
    for path in docx:
        target = out / (unicodedata.normalize("NFC", path.stem) + ".md")
        target.write_text(parse_document(path).to_markdown(), encoding="utf-8")
        result.append(target)
        REAL_DOCX_BY_MD[target] = path
    return result


REAL_MD: list[Path] = _convert()


def find_md(fragment: str) -> Path | None:
    """Первый md по подстроке имени (имена в NFC: на macOS файловая система отдаёт NFD)."""
    return next((p for p in REAL_MD if fragment in p.name), None)


def _zip_docx_markdown(archive: Path, name_suffix: str) -> str | None:
    """Markdown docx из архива в `data/`; None, если архива нет."""
    if not archive.exists():
        return None
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        name = next(
            n for n in z.namelist() if unicodedata.normalize("NFC", n).endswith(name_suffix)
        )
        path = Path(tmp) / "doc.docx"
        path.write_bytes(z.read(name))
        return parse_document(path).to_markdown()


def real_plan_markdown() -> str | None:
    """Учебный план 26/5003/1 из `data/Учебный план.zip`."""
    return _zip_docx_markdown(DATA_DIR / "Учебный план.zip", "Report_26_5003_1.docx")


def real_ckmv_markdown() -> str | None:
    """ЦКМВ из `data/ЦКМВ.zip`."""
    return _zip_docx_markdown(DATA_DIR / "ЦКМВ.zip", "ЦКМВ.docx")


def real_characteristic_markdown(corrected: bool = True) -> str | None:
    """Общая характеристика из `data/Характеристика.zip`: версия `_corr` или исходная."""
    suffix = "(ТОП)_corr.docx" if corrected else "(ТОП).docx"
    return _zip_docx_markdown(DATA_DIR / "Характеристика.zip", suffix)


def real_matrix_xlsx() -> bytes | None:
    """Матрица компетенций (xlsx) из `data/Матрица компетенций.zip`; None, если архива нет."""
    archive = DATA_DIR / "Матрица компетенций.zip"
    if not archive.exists():
        return None
    with zipfile.ZipFile(archive) as z:
        name = next(n for n in z.namelist() if unicodedata.normalize("NFC", n).endswith(".xlsx"))
        return z.read(name)
