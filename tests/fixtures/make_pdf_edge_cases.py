"""Генератор маленьких синтетических PDF для пограничных случаев парсера PDF (ревью #41).

- `standard_two-tables_01.pdf` — две самостоятельные таблицы с одинаковым числом столбцов:
  первая кончается в верхней части страницы 1, вторая открывает страницу 2;
- `standard_long-table_01.pdf` — таблица на две страницы, ячейки продолжения стоят на тех же
  координатах, что и ячейки первой страницы (объединений нет);
- `standard_narrow-cells_01.pdf` — перенос строк в ячейках: целые слова и составное слово,
  оборванное без дефиса, потому что шире ячейки;
- `standard_vector-footnote_01.pdf` — черта перед сносками нарисована линией, а не «_____»;
- `standard_footnote-marks_01.pdf` — номера сносок в тексте: мелкие цифры выше строки, в том
  числе перенесённые в начало следующей строки вместе с точкой (#50);
- `standard_wide-table_01.pdf` — таблица шире страницы: правая граница за краем листа (#50).

Файлы уже лежат в репозитории; перегенерировать (нужны reportlab и шрифты DejaVu):
    uv run --with reportlab python tests/fixtures/make_pdf_edge_cases.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONTS = Path("/usr/share/fonts/truetype/dejavu")
HERE = Path(__file__).parent
W, H = A4
LEFT, INDENT, STEP = 50, 85, 14
BODY = [
    "Этот абзац нужен, чтобы основной шрифт и левое поле документа определялись",
    "по большинству строк синтетического документа, а не по первым строкам",
    "абзацев; текст не связан с каким-либо реальным нормативным документом.",
]


class Doc:
    def __init__(self, name: str) -> None:
        self.c = canvas.Canvas(str(HERE / name), pagesize=A4, invariant=True)
        self.c.setTitle("Синтетический документ")

    def text(self, x: float, top: float, s: str, size: float = 12) -> None:
        self.c.setFont("Serif", size)
        self.c.drawString(x, H - top - size, s)

    def para(self, top: float, lines: list[str]) -> float:
        for i, s in enumerate(lines):
            self.text(INDENT if i == 0 else LEFT, top + i * STEP, s)
        return top + len(lines) * STEP

    def row(self, top: float, widths: list[float], cells: list[list[str]], h: float = 20) -> float:
        x = LEFT
        for w, lines in zip(widths, cells, strict=True):
            self.c.rect(x, H - top - h, w, h)
            for i, s in enumerate(lines):
                self.text(x + 4, top + 3 + i * STEP, s, size=11)
            x += w
        return top + h

    def page(self) -> None:
        self.c.showPage()

    def save(self) -> None:
        self.c.save()


def two_tables() -> None:
    d = Doc("standard_two-tables_01.pdf")
    y = d.para(60, BODY)
    y = d.row(y + 10, [200, 100], [["Дисциплина"], ["Часы"]])
    y = d.row(y, [200, 100], [["Алгебра"], ["144"]])
    d.page()
    y = d.row(60, [200, 100], [["Компетенция"], ["Индикатор"]])
    y = d.row(y, [200, 100], [["УК-1"], ["УК-1.1"]])
    d.para(y + 20, BODY)
    d.save()


def long_table() -> None:
    d = Doc("standard_long-table_01.pdf")
    # страница 1 целиком — таблица: шапка и строки A1–A35 до низа страницы
    y = d.row(60, [200, 100], [["Код"], ["Значение"]])
    for n in range(1, 36):
        y = d.row(y, [200, 100], [[f"A{n}"], [f"Value {n}"]])
    d.page()
    # страница 2: продолжение с того же верхнего края — ячейки на тех же координатах
    y = 60
    for n in range(36, 41):
        y = d.row(y, [200, 100], [[f"A{n}"], [f"Value {n}"]])
    d.para(y + 20, BODY)
    d.save()


def narrow_cells() -> None:
    d = Doc("standard_narrow-cells_01.pdf")
    y = d.para(60, BODY)
    # узкий столбец: «Информационно-коммуник» занимает его почти целиком (слово шире ячейки)
    cut = "Информационно-коммуник"
    widths = [pdfmetrics.stringWidth(cut, "Serif", 11) + 10, 100]
    y = d.row(y + 10, widths, [["Форма обучения"], ["Срок"]])
    y = d.row(y, widths, [["в очно-заочной", "форме обучения"], ["5 лет"]], h=34)
    d.row(y, widths, [[cut, "ационные технологии"], ["4 года"]], h=34)
    d.save()


def vector_footnote() -> None:
    d = Doc("standard_vector-footnote_01.pdf")
    first = ["1.1. Первый пункт со ссылкой на первую сноску1.", "2.1. Второй пункт и сноска2."]
    for n in (1, 2):
        y = d.para(120, [first[n - 1], *BODY])  # ниже зоны колонтитулов
        d.para(y, BODY)
        d.c.setLineWidth(0.5)
        d.c.line(LEFT, H - 700, LEFT + 150, H - 700)  # черта сносок — графика
        d.text(LEFT, 708, f"{n} Сноска номер {n} мелким шрифтом под графической чертой.", size=8)
        d.page()
    d.save()


def footnote_marks() -> None:
    d = Doc("standard_footnote-marks_01.pdf")
    y = d.para(60, BODY)

    def mark(x: float, top: float, number: str) -> None:
        d.c.setFont("Serif", 7)
        d.c.drawString(x, H - top - 12 + 5, number)  # мельче и выше базовой линии

    first = "1. Программы разрабатываются организацией"
    d.text(INDENT, y, first)
    mark(INDENT + pdfmetrics.stringWidth(first, "Serif", 12), y, "10")
    d.text(INDENT + pdfmetrics.stringWidth(first, "Serif", 12) + 8, y, ".")
    y += STEP
    d.text(LEFT, y, "Программы утверждаются организацией")
    y += STEP
    mark(LEFT, y, "4")  # номер перенёсся на новую строку вместе с точкой
    d.text(LEFT + 4, y, ".")
    y += STEP * 2
    d.text(INDENT, y, "2. Примечание мелким шрифтом на своей строке не трогается:")
    d.text(LEFT, y + STEP, "объем 144 часа", size=7)
    d.save()


def wide_table() -> None:
    d = Doc("standard_wide-table_01.pdf")
    y = d.para(60, BODY)
    # правая граница таблицы — за краем страницы (ширина A4 — 595 pt)
    y = d.row(y + 10, [200, 420], [["Код"], ["Наименование"]])
    d.row(y, [200, 420], [["A/01.3"], ["Разработка кода"]])
    d.save()


def main() -> None:
    pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "DejaVuSerif.ttf")))
    two_tables()
    long_table()
    narrow_cells()
    vector_footnote()
    footnote_marks()
    wide_table()


if __name__ == "__main__":
    main()
