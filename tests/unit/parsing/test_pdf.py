"""Тесты парсера PDF на синтетическом документе (вёрстка как у выгрузок нормативов)

Фикстура `standard_normal_01.pdf` собирается скриптом tests/fixtures/make_standard_pdf.py.
"""

from pathlib import Path

import pytest

from edu_doc_agent.parsing import (
    BlockKind,
    CellSpan,
    NoTextLayerError,
    parse_document,
    supported_suffixes,
)
from edu_doc_agent.parsing.pdf_parser import _is_bold, _join

FIXTURES = Path(__file__).parents[2] / "fixtures"
STANDARD = FIXTURES / "standard_normal_01.pdf"


def test_paragraphs_headings_and_cleanup() -> None:
    blocks = parse_document(STANDARD).blocks
    texts = [(b.kind, b.text) for b in blocks if b.kind != BlockKind.TABLE]
    assert texts[0] == (BlockKind.HEADING, "I. Общие положения")  # жирный по центру
    assert texts[1][1].startswith("1.1. Настоящий синтетический стандарт содержит")
    assert texts[1][1].endswith("каждой из страниц документа, включая титульную.")  # 6 строк
    assert texts[2:] == [
        (BlockKind.PARAGRAPH, "1.2. Срок получения образования по программе:"),
        (BlockKind.PARAGRAPH, "в очной форме обучения составляет 4 года;"),  # отступ — абзац
        (BlockKind.PARAGRAPH, "в очно-заочной форме обучения составляет 5 лет."),
        (  # перенос «научно-» на строку: дефис сохранён, пробела нет
            BlockKind.PARAGRAPH,
            "Программа включает обязательную научно-исследовательскую работу и практику.",
        ),
        (  # абзац через границу страниц
            BlockKind.PARAGRAPH,
            "1.3. Пункт, который начинается внизу страницы и продолжается на следующей "
            "странице без отступа первой строки.",
        ),
        (BlockKind.PARAGRAPH, "2.1. Структура программы приведена в таблице ниже:"),
        (
            BlockKind.PARAGRAPH,
            "Описание структуры дано для синтетического примера и не связано с каким-либо "
            "реальным направлением подготовки, оно нужно для того, чтобы проверить сборку "
            "абзацев из нескольких строк.",
        ),
        (BlockKind.PARAGRAPH, "3.1. Последний пункт синтетического стандарта*(1)."),
        (
            BlockKind.PARAGRAPH,
            "Этот абзац нужен, чтобы основной шрифт и левое поле документа определялись по "
            "большинству строк, а не по первым строкам абзацев, и чтобы сноски ниже были "
            "мельче основного текста документа.",
        ),
    ]  # колонтитулы, номера страниц, черта и обе сноски убраны
    markdown = parse_document(STANDARD).to_markdown()
    for noise in ("служебный колонтитул", "Страница", "Сноска", "____"):
        assert noise not in markdown


def test_table_across_pages() -> None:
    blocks = parse_document(STANDARD).blocks
    (table,) = [b for b in blocks if b.kind == BlockKind.TABLE]
    assert blocks.index(table) == 9  # на своём месте между абзацами
    assert table.rows == [
        ["Структура", "", "Объем, з.е."],  # объединение по горизонтали
        ["Блок 1", "Дисциплины", "160"],
        ["Блок 2", "Практика (в том числе научно-исследовательская работа)", "20"],
        ["Блок 3", "Государственная итоговая аттестация", "9"],  # строка, разорванная страницей
        ["Итого", "", "240"],  # продолжение на следующей странице приклеено
    ]
    assert table.spans == [CellSpan(row=0, col=0, col_span=2), CellSpan(row=4, col=0, col_span=2)]


def test_scan_without_text_layer(tmp_path: Path) -> None:
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    with pytest.raises(NoTextLayerError):
        parse_document(scan)


def test_bold_font_names() -> None:
    assert _is_bold("ABCDEF+TimesNewRomanPS-BoldMT")
    assert not _is_bold("ABCDEF+TimesNewRomanPSMT")
    # ГАРАНТ: имя шрифта — байты cp1251 «Times New Roman Полужирный» в виде строки
    assert _is_bold("b'FPYNUE+Times New Roman \\xcf\\xee\\xeb\\xf3\\xe6\\xe8\\xf0\\xed\\xfb\\xe9'")
    assert not _is_bold("b'FPOJSZ+Times New Roman \\xca\\xf3\\xf0\\xf1\\xe8\\xe2'")  # «Курсив»


def test_join_lines() -> None:
    assert _join(["организационно-", "педагогических условий"]) == (
        "организационно-педагогических условий"
    )
    assert _join(["Информационно-коммуник", "ационные"], narrow=True) == (
        "Информационно-коммуникационные"
    )
    assert _join(["Информационно-коммуник", "ационные"]) == "Информационно-коммуник ационные"


def test_pdf_is_supported() -> None:
    assert ".pdf" in supported_suffixes()
