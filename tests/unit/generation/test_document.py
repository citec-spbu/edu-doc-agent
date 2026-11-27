"""Перенос правок изменения в исходный DOCX."""

import io

from docx import Document

from edu_doc_agent.extraction.models import Competence, RPDExtraction, WorkloadRow
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.generation.document import Patch, patch_docx, source_for


def source_docx() -> bytes:
    doc = Document()
    doc.add_paragraph("Трудоемкость в зачетных единицах: 6")
    p = doc.add_paragraph()
    p.add_run("УК-12 ").bold = True
    p.add_run("способен осуществлять поиск информации")
    doc.add_paragraph("ОПК-1 способен применять фундаментальные знания")
    table = doc.add_table(rows=2, cols=4)
    for row, values in zip(
        table.rows, (["Семестр 3", "30", "2", "6"], ["ИТОГО", "30", "2", "6"]), strict=True
    ):
        for cell, value in zip(row.cells, values, strict=True):
            cell.text = value
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def texts(content: bytes) -> tuple[list[str], list[list[str]]]:
    doc = Document(io.BytesIO(content))
    return [p.text for p in doc.paragraphs], [[c.text for c in r.cells] for r in doc.tables[0].rows]


def test_patches_paragraph_table_row_and_removal() -> None:
    patches = [
        Patch("credits", "set", 6.0, 8, "Трудоемкость в зачетных единицах: 6"),
        Patch("competences/0/code", "set", "УК-12", "УК-1", "| УК-12 способен осуществлять поиск"),
        Patch("workload_table/1/credits", "set", 6.0, 8, "| ИТОГО | 30 | 2 | 6 |"),
        Patch("competences/1", "remove", None, None, "| ОПК-1 способен применять"),
        Patch("competences", "append", None, {"code": "ОПК-2"}, None),
    ]

    result = patch_docx(source_docx(), patches)

    paragraphs, rows = texts(result.content)
    assert paragraphs[0] == "Трудоемкость в зачетных единицах: 8"
    assert paragraphs[1] == "УК-1 способен осуществлять поиск информации"
    assert Document(io.BytesIO(result.content)).paragraphs[1].runs[0].bold  # оформление на месте
    assert "ОПК-1 способен применять фундаментальные знания" not in paragraphs
    assert rows == [["Семестр 3", "30", "2", "6"], ["ИТОГО", "30", "2", "8"]]
    assert result.applied == [
        "credits",
        "competences/0/code",
        "workload_table/1/credits",
        "competences/1",
    ]
    assert result.manual == [
        "competences: новый элемент добавляется вручную: место в документе не определено"
    ]


def test_unclear_place_is_left_for_manual_edit() -> None:
    patches = [
        Patch("workload_table/1/lectures", "set", 2, 4, "| ИТОГО | 30 | 2 | 6 |"),
        Patch("goals", "set", "старые цели", "новые цели", "Такого абзаца нет"),
    ]
    patches.append(Patch("workload_table/1/self_study", "set", 30, 31, "| ИТОГО | 30 | 2 | 6 |"))

    result = patch_docx(source_docx(), patches)

    assert result.applied == ["workload_table/1/lectures", "workload_table/1/self_study"]
    assert result.manual == ["goals: место значения в документе не найдено"]
    _, rows = texts(result.content)
    assert rows[1] == ["ИТОГО", "31", "4", "6"]


def test_source_for_path() -> None:
    src = SourceRef(file="a.docx", line=1, quote="| ИТОГО | 30 |")
    title = SourceRef(file="a.docx", line=2, quote="Трудоемкость в зачетных единицах: 6")
    comp = SourceRef(file="a.docx", line=3, quote="| УК-1 способен")
    data = RPDExtraction(
        file="a.docx",
        credits=6,
        sources={"credits": title},
        competences=[Competence(code="УК-1", source=comp)],
        workload_table=[WorkloadRow(is_total=True, credits=6, source=src)],
    )

    assert source_for(data, "credits") == title
    assert source_for(data, "workload_table/0/credits") == src
    assert source_for(data, "competences/0/code") == comp


def test_value_inside_table_cell_text() -> None:
    doc = Document()
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "УК-12 способен осуществлять поиск"
    table.rows[0].cells[1].text = "УК-12.1 Анализирует задачу"
    out = io.BytesIO()
    doc.save(out)
    quote = "| УК-12 способен осуществлять поиск | УК-12.1 Анализирует задачу |"
    patch = Patch("competences/0/code", "set", "УК-12", "УК-1", quote)

    result = patch_docx(out.getvalue(), [patch])

    cells = [c.text for c in Document(io.BytesIO(result.content)).tables[0].rows[0].cells]
    # код правится там, где он целым словом; «УК-12.1» индикатора не трогается
    assert result.applied == ["competences/0/code"]
    assert cells == ["УК-1 способен осуществлять поиск", "УК-12.1 Анализирует задачу"]


def test_multiline_text_and_already_applied_value() -> None:
    doc = Document()
    doc.add_paragraph("1.1. Цели и задачи")
    doc.add_paragraph("Цель — научить готовить борщ.")
    doc.add_paragraph("Задачи — варить и солить.")
    doc.add_paragraph("1.2. Пререквизиты")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "УК-12 способен"
    table.rows[0].cells[1].text = "знает: поиск"
    out = io.BytesIO()
    doc.save(out)
    old_goals = "Цель — научить готовить борщ.\nЗадачи — варить и солить."
    row = "| УК-12 способен | знает: поиск |"
    patches = [
        Patch("goals", "set", old_goals, "Цель — изучить машинное обучение.", "Цель"),
        Patch("competences/0/code", "set", "УК-12", "УК-1", row),
        # результат обучения ссылается на тот же код в той же строке: уже внесено
        Patch("learning_outcomes/0/competence_codes/0", "set", "УК-12", "УК-1", row),
    ]

    result = patch_docx(out.getvalue(), patches)

    fixed = Document(io.BytesIO(result.content))
    assert [p.text for p in fixed.paragraphs] == [
        "1.1. Цели и задачи",
        "Цель — изучить машинное обучение.",
        "1.2. Пререквизиты",
    ]
    assert fixed.tables[0].rows[0].cells[0].text == "УК-1 способен"
    assert result.manual == []
