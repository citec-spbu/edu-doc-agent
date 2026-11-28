"""Генератор синтетического PDF профстандарта `profstandard_normal_01.pdf` (#50).

Вёрстка повторяет PDF приказа Минтруда с текстовым слоем: оглавление с отточиями (одна
строка оглавления перенесена без отточия), шапки функций без рамок в три колонки
(«Наименование … Код … Уровень квалификации»), таблицы полей в рамках, подпись «Трудовые
действия» — объединённая ячейка на несколько строк. Пограничные случаи:

- название обобщённой функции в заголовке «3.1. …» перенесено на вторую строку;
- строка таблицы «Необходимые знания» разорвана страницей: продолжение «проверки программ»
  открывает страницу 2, следом — новое знание с заглавной буквы;
- шапка трудовой функции 3.1.2 начинается внизу страницы 2 и продолжается на странице 3;
- код функции набран кириллической «А».

Профстандарт «99.001 Синтетик» вымышлен. Файл уже лежит в репозитории; перегенерировать
(нужны reportlab и шрифты DejaVu):
    uv run --with reportlab python tests/fixtures/make_prof_standard_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONTS = Path("/usr/share/fonts/truetype/dejavu")
HERE = Path(__file__).parent
W, H = A4
LEFT, STEP, SIZE = 38, 13, 10
LABEL_W, VALUE_W = 150, 400


class Doc:
    def __init__(self, name: str) -> None:
        self.c = canvas.Canvas(str(HERE / name), pagesize=A4, invariant=True)
        self.c.setTitle("Синтетический профстандарт")

    def text(self, x: float, top: float, s: str, size: float = SIZE) -> None:
        self.c.setFont("Serif", size)
        self.c.drawString(x, H - top - size, s)

    def lines(self, top: float, lines: list[str], x: float = LEFT) -> float:
        for i, s in enumerate(lines):
            self.text(x, top + i * STEP, s)
        return top + len(lines) * STEP

    def cell(self, x: float, top: float, w: float, h: float, lines: list[str]) -> None:
        self.c.rect(x, H - top - h, w, h)
        for i, s in enumerate(lines):
            self.text(x + 3, top + 3 + i * STEP, s)

    def fields(self, top: float, rows: list[tuple[str, list[str]]]) -> float:
        """Таблица «подпись | значения»: подпись — объединённая ячейка на все свои значения,
        перенос строк в подписи — «\n»."""
        for label, values in rows:
            label_lines = label.split("\n") if label else []
            h = max(18, (len(label_lines) * STEP + 6) / len(values))
            self.cell(LEFT, top, LABEL_W, h * len(values), label_lines)
            for i, value in enumerate(values):
                self.cell(LEFT + LABEL_W, top + i * h, VALUE_W, h, [value])
            top += h * len(values)
        return top

    def head(self, top: float, name: list[str], code: str, level: str, label: list[str]) -> float:
        """Шапка функции без рамок: колонки строк перемешаны, как в настоящих PDF."""
        self.text(LEFT, top + STEP, "Наименование")
        for i, s in enumerate(name):
            self.text(123, top + i * STEP, s)
        self.text(351, top + STEP, "Код")
        self.text(395, top + STEP, code)
        for i, s in enumerate(label):
            self.text(453, top + i * STEP, s)
        self.text(556, top + STEP, level)
        return top + max(len(name), len(label), 2) * STEP + 6

    def origin(self, top: float) -> float:
        """«Происхождение …» слева без рамки, справа — таблица «Оригинал X»."""
        self.text(LEFT, top + 3, "Происхождение функции")
        self.cell(185, top, 100, 18, ["Оригинал"])
        self.cell(285, top, 40, 18, ["X"])
        self.cell(325, top, 263, 18, ["Заимствовано из оригинала"])
        return top + 30

    def page(self) -> None:
        self.c.showPage()

    def save(self) -> None:
        self.c.save()


def normal() -> None:
    d = Doc("profstandard_normal_01.pdf")
    y = d.lines(40, ["УТВЕРЖДЕН", "приказом Министерства", "от 1 марта 2030 № 999н"], x=400)
    y = d.lines(y + 10, ["ПРОФЕССИОНАЛЬНЫЙ СТАНДАРТ", "Синтетик", "Содержание"], x=200)
    y = d.lines(
        y,
        [
            "I. Общие сведения" + "." * 60 + " 1",
            "3.1. Обобщенная трудовая функция «Подготовка синтетических",
            "данных»" + "." * 70 + " 1",
        ],
    )
    y = d.lines(y + 10, ["I. Общие сведения"])
    d.text(LEFT, y, "Создание синтетических данных")
    d.text(517, y, "99.001")
    y = d.lines(y + STEP, ["(наименование вида профессиональной деятельности)"])
    y = d.lines(y, ["Основная цель вида профессиональной деятельности:"])
    y = d.lines(y, ["Подготовка данных для проверки программ"])
    y = d.lines(y + 10, ["III. Характеристика обобщенных трудовых функций"])
    y = d.lines(y + 6, ["3.1. Обобщенная трудовая функция «Подготовка синтетических", "данных»"])
    y = d.head(y + 6, ["Подготовка синтетических", "данных"], "А", "6", ["Уровень", "квалификации"])
    y = d.origin(y)
    y = d.fields(
        y,
        [
            ("Возможные наименования\nдолжностей, профессий", ["Инженер данных"]),
            ("Требования к образованию\nи обучению", ["Высшее образование – бакалавриат"]),
            ("Другие характеристики", ["–"]),
        ],
    )
    y = d.lines(y + 6, ["Дополнительные характеристики"])
    for row in (("ОКЗ", "2512", "Разработчики программного обеспечения"),
                ("ОКСО", "2.09.03.04", "Программная инженерия")):  # fmt: skip
        d.cell(LEFT, y, 100, 18, [row[0]])
        d.cell(LEFT + 100, y, 100, 18, [row[1]])
        d.cell(LEFT + 200, y, 350, 18, [row[2]])
        y += 18
    y = d.lines(y + 10, ["3.1.1. Трудовая функция"])
    y = d.head(
        y + 6,
        ["Сбор синтетических данных"],
        "А/01.6",
        "6",
        ["Уровень", "(подуровень)", "квалификации"],
    )
    y = d.origin(y)
    y = d.fields(
        y,
        [
            ("Трудовые действия", ["Сбор исходных данных", "Проверка полноты данных"]),
            ("Необходимые умения", ["Применять генераторы данных"]),
        ],
    )
    # строка «Необходимые знания» доходит до низа страницы и продолжается на следующей
    d.fields(max(y, H - 60), [("Необходимые знания", ["Методы генерации данных для"])])
    d.page()
    y = d.fields(38, [("", ["проверки программ"]), ("", ["Форматы хранения данных"])])
    y = d.fields(y, [("Другие характеристики", ["-"])])
    # трудовая функция 3.1.2: шапка начинается внизу страницы 2
    y = d.lines(H - 90, ["3.1.2. Трудовая функция"])
    d.text(LEFT, y + 6, "Наименование")
    d.text(123, y + 6, "Проверка синтетических")
    d.text(351, y + 6, "Код")
    d.text(395, y + 6, "А/02.6")
    d.text(466, y + 6, "Уровень")
    d.text(556, y + 6, "6")
    d.page()
    d.text(123, 40, "данных на полноту")
    d.text(453, 40, "квалификации")
    y = d.origin(60)
    y = d.fields(
        y,
        [
            ("Трудовые действия", ["Сверка данных с образцом"]),
            ("Необходимые умения", ["Сравнивать наборы данных"]),
            ("Необходимые знания", ["Форматы наборов данных"]),
        ],
    )
    y = d.lines(
        y + 10, ["IV. Сведения об организациях – разработчиках профессионального стандарта"]
    )
    d.lines(y, ["Синтетический институт"])
    d.save()


def main() -> None:
    pdfmetrics.registerFont(TTFont("Serif", str(FONTS / "DejaVuSerif.ttf")))
    normal()


if __name__ == "__main__":
    main()
