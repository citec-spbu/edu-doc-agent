"""Сборка корпуса по манифесту: источники, overrides, YAML-снимки и отчёт."""

from pathlib import Path

import pytest
import yaml

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus import build as build_module
from edu_doc_agent.retrieval.corpus.build import build, main, write

HEAD = (
    "Приказ Министерства науки и высшего образования РФ от 23 августа 2017 г. N {n} "
    '"Об утверждении ФГОС ВО - бакалавриат по направлению подготовки {code} {name}"'
)


def _fgos(code: str, n: str, name: str) -> ParsedDocument:
    blocks = [
        Block(BlockKind.PARAGRAPH, HEAD.format(n=n, code=code, name=name)),
        Block(BlockKind.HEADING, "I. Общие положения", level=1),
        Block(BlockKind.PARAGRAPH, "1.1. Настоящий ФГОС представляет собой требования."),
        Block(BlockKind.HEADING, "III. Требования к результатам освоения", level=1),
        Block(BlockKind.PARAGRAPH, "3.1. Программа устанавливает компетенции:"),
        Block(BlockKind.PARAGRAPH, "УК-1. Способен к поиску информации."),
        Block(BlockKind.PARAGRAPH, "ОПК-1. Способен применять знания."),
    ]
    return ParsedDocument(source=code, blocks=blocks)


DOCS = {
    "020302.pdf": _fgos("02.03.02", "808", "Фундаментальная информатика"),
    "010302.pdf": _fgos("01.03.02", "9", "Прикладная математика и информатика"),
    "broken.pdf": ParsedDocument(source="broken", blocks=[]),
}


@pytest.fixture
def regs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "fgos").mkdir()
    for name in ("020302.pdf", "010302.pdf"):
        (tmp_path / "fgos" / name).write_bytes(b"")
    (tmp_path / "broken.pdf").write_bytes(b"")
    monkeypatch.setattr(build_module, "parse_document", lambda path: DOCS[path.name])
    manifest = {
        "sources": [
            {"files": "fgos/*.pdf", "type": "fgos"},
            {"file": "fgos/020302.pdf", "type": "fgos", "edition": "808 ред. от 27.02.2023"},
            {"file": "broken.pdf", "type": "spbu_standard"},
        ]
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    overrides = {"fgos-01.03.02": {"drop": ["fgos-01.03.02:1.1"]}}
    (tmp_path / "overrides.yaml").write_text(
        yaml.safe_dump(overrides, allow_unicode=True), encoding="utf-8"
    )
    return tmp_path


def test_build_sources_meta_and_overrides(regs: Path) -> None:
    documents, failures = build(regs / "manifest.yaml", regs / "overrides.yaml")
    by_id = {d.document.doc_id: d for d in documents}
    assert set(by_id) == {"fgos-01.03.02", "fgos-02.03.02"}
    assert by_id["fgos-02.03.02"].document.edition == "808 ред. от 27.02.2023"  # из манифеста
    assert by_id["fgos-01.03.02"].document.edition == "9 от 23.08.2017"  # найдено в тексте
    assert "fgos-01.03.02:1.1" not in {f.id for f in by_id["fgos-01.03.02"].fragments}
    assert len(failures) == 1 and failures[0].startswith("broken.pdf")


def test_write_snapshots_and_report(regs: Path) -> None:
    documents, failures = build(regs / "manifest.yaml")
    out = regs / "corpus"
    errors = write(documents, out, failures)
    assert errors == failures
    snapshot = yaml.safe_load((out / "fgos-02.03.02.yaml").read_text(encoding="utf-8"))
    assert snapshot["document"]["doc_id"] == "fgos-02.03.02"
    assert [f["id"] for f in snapshot["fragments"]] == [
        "fgos-02.03.02:1.1",
        "fgos-02.03.02:3.1",
        "fgos-02.03.02:3.1:УК-1",
        "fgos-02.03.02:3.1:ОПК-1",
    ]
    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "| fgos-02.03.02 |" in report
    assert "## Ошибки" in report and "broken.pdf" in report


def test_main_exit_code(regs: Path) -> None:
    out = regs / "corpus"
    assert main([str(regs / "manifest.yaml"), "--out", str(out)]) == 1  # broken.pdf
    assert (out / "REPORT.md").exists()


ACT = ParsedDocument(
    source="act",
    blocks=[
        Block(BlockKind.HEADING, 'Федеральный закон от 1 января 2000 г. N 1-ФЗ "О синтетике"'),
        Block(BlockKind.HEADING, "Глава 1. Общие положения"),
        Block(BlockKind.PARAGRAPH, "Статья 1. Предмет"),
        Block(BlockKind.PARAGRAPH, "1. Закон регулирует синтетику."),
    ],
)
CARD = """Профстандарт: 99.001
Синтетик
от 20.07.2022 № 999н
III. Характеристика обобщенных трудовых функций
3.1. Обобщенная трудовая функция
Наименование
Подготовка данных
Код
A
Уровень квалификации
6
"""


def test_federal_acts_and_prof_standards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "laws").mkdir()
    (tmp_path / "prof_standards").mkdir()
    for name in ("law.pdf", "nolayout.pdf"):
        (tmp_path / "laws" / name).write_bytes(b"")
    (tmp_path / "prof_standards" / "99.001.txt").write_text(CARD, encoding="utf-8")
    monkeypatch.setattr(build_module, "parse_document", lambda path: ACT)
    manifest = {
        "sources": [
            {"file": "laws/law.pdf", "type": "federal_act", "layout": "law", "doc_id": "1-fz"},
            {"file": "laws/nolayout.pdf", "type": "federal_act", "doc_id": "x"},
            {"files": "prof_standards/*.txt", "type": "prof_standard"},
        ]
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    documents, failures = build(tmp_path / "manifest.yaml")
    by_id = {d.document.doc_id: d for d in documents}
    assert set(by_id) == {"1-fz", "ps-99.001"}
    law = by_id["1-fz"]
    assert law.document.title == 'Федеральный закон от 1 января 2000 г. N 1-ФЗ "О синтетике"'
    assert law.document.edition == "1-ФЗ от 01.01.2000"  # редакция по шапке
    assert [f.id for f in law.fragments] == ["1-fz:ст.1:ч.1"]
    assert [f.id for f in by_id["ps-99.001"].fragments] == ["ps-99.001:general", "ps-99.001:A"]
    assert len(failures) == 1 and "layout" in failures[0]
    write(documents, tmp_path / "corpus", failures)
    report = (tmp_path / "corpus" / "REPORT.md").read_text(encoding="utf-8")
    assert "| ps-99.001 |" in report and "| 1 | 0 |" in report  # одна трудовая функция
