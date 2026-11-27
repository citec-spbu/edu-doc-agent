"""Генератор синтетического PDF `standard_normal_01.pdf` для тестов парсера PDF.

Вёрстка как у выгрузок нормативов (ГАРАНТ, КонсультантПлюс): абзацы с отступом первой строки,
колонтитулы на каждой странице, абзац через границу страниц, таблица с рамками на две страницы
(объединённые ячейки, строка, разорванная страницей), сноски после черты.

Файл уже лежит в репозитории; перегенерировать (нужны reportlab и шрифты DejaVu):
    uv run --with reportlab python tests/fixtures/make_standard_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONTS = Path("/usr/share/fonts/truetype/dejavu")
OUT = Path(__file__).with_name("standard_normal_01.pdf")
W, H = A4
LEFT, INDENT, STEP = 50, 85, 14


def main() -> None:
    pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "DejaVuSerif.ttf")))
    pdfmetrics.registerFont(TTFont("Serif-Bold", str(FONTS / "DejaVuSerif-Bold.ttf")))
    c = canvas.Canvas(str(OUT), pagesize=A4, invariant=True)
    c.setTitle("Синтетический стандарт")

    def text(x: float, top: float, s: str, font: str = "Serif", size: float = 12) -> None:
        c.setFont(font, size)
        c.drawString(x, H - top - size, s)

    def page_frame(n: int) -> None:
        text(LEFT, 20, "Синтетический стандарт № 7 — служебный колонтитул", size=9)
        text(LEFT, 805, f"Страница {n} из 3", size=9)

    def para(top: float, lines: list[str]) -> float:
        for i, s in enumerate(lines):
            text(INDENT if i == 0 else LEFT, top + i * STEP, s)
        return top + len(lines) * STEP

    def cell(x: float, top: float, w: float, h: float, lines: list[str]) -> None:
        c.rect(x, H - top - h, w, h)
        for i, s in enumerate(lines):
            text(x + 4, top + 3 + i * STEP, s, size=11)

    # страница 1
    page_frame(1)
    c.setFont("Serif-Bold", 12)
    c.drawCentredString(W / 2, H - 80 - 12, "I. Общие положения")
    y = para(
        110,
        [
            "1.1. Настоящий синтетический стандарт содержит обязательные требования",
            "к структуре программы и результатам ее освоения и предназначен только",
            "для проверки парсера документов в формате PDF с текстовым слоем, вёрстка",
            "которого повторяет выгрузки нормативных документов из правовых систем",
            "с отступом первой строки абзаца и колонтитулами на каждой из страниц",
            "документа, включая титульную.",
        ],
    )
    y = para(y, ["1.2. Срок получения образования по программе:"])
    y = para(y, ["в очной форме обучения составляет 4 года;"])
    y = para(y, ["в очно-заочной форме обучения составляет 5 лет."])
    y = para(y, ["Программа включает обязательную научно-", "исследовательскую работу и практику."])
    para(780 - STEP, ["1.3. Пункт, который начинается внизу страницы и"])
    c.showPage()

    # страница 2
    page_frame(2)
    y = 60
    text(LEFT, y, "продолжается на следующей странице без отступа первой строки.")
    y = para(y + STEP, ["2.1. Структура программы приведена в таблице ниже:"])
    y = para(
        y,
        [
            "Описание структуры дано для синтетического примера и не связано",
            "с каким-либо реальным направлением подготовки, оно нужно для того,",
            "чтобы проверить сборку абзацев из нескольких строк.",
        ],
    )
    x0, x1, x2, x3 = LEFT, 130, 400, 545
    top = 640
    cell(x0, top, x2 - x0, 20, ["Структура"])  # объединение по горизонтали
    cell(x2, top, x3 - x2, 20, ["Объем, з.е."])
    top += 20
    for a, b, v in [("Блок 1", ["Дисциплины"], "160")]:
        cell(x0, top, x1 - x0, 20, [a])
        cell(x1, top, x2 - x1, 20, b)
        cell(x2, top, x3 - x2, 20, [v])
        top += 20
    cell(x0, top, x1 - x0, 34, ["Блок 2"])
    cell(x1, top, x2 - x1, 34, ["Практика (в том числе", "научно-исследовательская работа)"])
    cell(x2, top, x3 - x2, 34, ["20"])
    top += 34
    cell(x0, top, x1 - x0, 20, ["Блок 3"])  # строка, разорванная страницей
    cell(x1, top, x2 - x1, 20, ["Государственная итоговая"])
    cell(x2, top, x3 - x2, 20, ["9"])
    c.showPage()

    # страница 3
    page_frame(3)
    top = 60
    cell(x0, top, x1 - x0, 20, [""])
    cell(x1, top, x2 - x1, 20, ["аттестация"])
    cell(x2, top, x3 - x2, 20, [""])
    top += 20
    cell(x0, top, x2 - x0, 20, ["Итого"])
    cell(x2, top, x3 - x2, 20, ["240"])
    y = para(top + 40, ["3.1. Последний пункт синтетического стандарта*(1)."])
    y = para(
        y,
        [
            "Этот абзац нужен, чтобы основной шрифт и левое поле документа",
            "определялись по большинству строк, а не по первым строкам абзацев,",
            "и чтобы сноски ниже были мельче основного текста документа.",
        ],
    )
    text(LEFT, y + STEP, "______________________")
    text(LEFT, y + 2 * STEP, "1 Сноска мелким шрифтом под чертой.", size=9)
    text(INDENT, y + 3 * STEP, "*(1) Сноска в конце документа обычным шрифтом.")
    c.showPage()
    c.save()


if __name__ == "__main__":
    main()
