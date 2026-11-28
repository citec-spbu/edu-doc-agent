"""Федеральные акты из ГАРАНТа: закон по частям статей, приказ по пунктам (синтетические блоки)."""

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus import garant
from edu_doc_agent.retrieval.corpus.acts import chunk_act
from edu_doc_agent.retrieval.corpus.models import DocumentInfo

P, H, T = BlockKind.PARAGRAPH, BlockKind.HEADING, BlockKind.TABLE


def para(text: str, kind: BlockKind = P) -> Block:
    return Block(kind, text)


def frame(*lines: str) -> Block:
    """Рамка ГАРАНТа: таблица в одну колонку (вторая пустая)."""
    return Block(T, "", rows=[[line, ""] for line in lines])


def note(text: str, act: str) -> Block:
    full = f"Информация об изменениях: {text} - {act}"
    return frame(full, full, act, "См. предыдущую редакцию")


EDITIONS = Block(
    T,
    "",
    rows=[
        ["С изменениями и дополнениями от:"],
        ["С из2м мена… перемешанный слой"],
        ["7 мая 2013 г., 2 декабря 2019 г., 4 августа 2026 г."],
    ],
)

LAW = ParsedDocument(
    source="law.pdf",
    blocks=[
        para('Федеральный закон от 1 января 2000 г. N 1-ФЗ "О синтетике"', H),
        para("(с изменениями и дополнениями)", H),
        para("С изменениями и дополнениями от:", H),
        EDITIONS,
        para("Принят Государственной Думой 1 декабря 1999 года", H),
        frame("ГАРАНТ: См. Сравнительный анализ", "ГАРАНТ: См. Сравнительный анализ"),
        para("Глава 1. Общие положения", H),
        para("Статья 1. Основные понятия ГАРАНТ: См. комментарии к статье 1"),
        para("Для целей настоящего закона применяются следующие понятия:"),
        para("1) программа - комплекс характеристик;"),
        note("Пункт 2 изменен с 1 июля 2020 г.", "Федеральный закон от 2 декабря 2019 г. № 9-ФЗ"),
        para("См. предыдущую редакцию"),
        para("2) практика - вид учебной деятельности;"),
        para("3) утратил силу с 5 мая 2020 г. - Федеральный закон от 24 апреля 2020 г. № 7-ФЗ"),
        para("Статья 2. Общие требования к реализации образовательных"),
        para("программ"),
        para("1. Программы реализуются самостоятельно и в сетевой форме."),
        para("2. Используются различные технологии. 3. Практика может быть организована:"),
        para("1) в организации;"),
        note(
            "Пункт 2 изменен с 28 февраля 2023 г.", "Федеральный закон от 17 февраля 2023 г. № 2-ФЗ"
        ),
        para("2) в профильной организации."),
        note(
            "Часть 4 изменена с 1 сентября 2021 г.",
            "Федеральный закон от 30 декабря 2020 г. № 5-ФЗ",
        ),
        para("4. Количество зачетных единиц устанавливается стандартом."),
        para("5. Утратила силу с 1 мая 2024 г. - Федеральный закон от 14 апреля 2023 г. № 4-ФЗ"),
        para("Статья 3. Утратила силу."),
        para("1. Текст утратившей силу статьи."),
        para("──────────────────────────────"),
        para("1 Сноска в конце документа."),
    ],
)


def info(doc_id: str, edition: str = "1") -> DocumentInfo:
    return DocumentInfo(
        doc_id=doc_id, doc_type="federal_act", title=doc_id, edition=edition, source_file="x.pdf"
    )


def test_law_is_cut_into_article_parts_and_items() -> None:
    doc = chunk_act(LAW, info("1-fz"), "law")
    by_id = {f.id: f for f in doc.fragments}
    assert list(by_id) == [
        "1-fz:ст.1",
        "1-fz:ст.1:п.1",
        "1-fz:ст.1:п.2",
        "1-fz:ст.2:ч.1",
        "1-fz:ст.2:ч.2",
        "1-fz:ст.2:ч.3",  # приклеенная к ч. 2 часть отделена
        "1-fz:ст.2:ч.4",
    ]  # утратившие силу пункт 3, часть 5 и статья 3 не берутся, сноски тоже
    assert doc.warnings == []
    item = by_id["1-fz:ст.1:п.2"]
    assert item.text == "2) практика - вид учебной деятельности;"
    assert item.context == "Для целей настоящего закона применяются следующие понятия:"
    assert item.path == "Глава 1. Общие положения › Статья 1. Основные понятия › п. 2"
    part = by_id["1-fz:ст.2:ч.3"]
    assert part.number == "ст.2:ч.3"
    assert part.path.endswith(
        "Статья 2. Общие требования к реализации образовательных программ › ч. 3"
    )
    assert (
        part.text
        == "3. Практика может быть организована:\n1) в организации;\n2) в профильной организации."
    )
    assert all("ГАРАНТ" not in f.text and "См." not in f.text for f in doc.fragments)


def test_law_change_notes_bind_to_parts_and_items() -> None:
    by_id = {f.id: f for f in chunk_act(LAW, info("1-fz"), "law").fragments}
    assert [n.act for n in by_id["1-fz:ст.1:п.2"].changed_by] == [
        "Федеральный закон от 2 декабря 2019 г. № 9-ФЗ"
    ]
    # «Пункт 2 изменен» внутри части 3 — изменение части
    assert [n.since for n in by_id["1-fz:ст.2:ч.3"].changed_by] == ["с 28 февраля 2023 г."]
    assert [n.act for n in by_id["1-fz:ст.2:ч.4"].changed_by] == [
        "Федеральный закон от 30 декабря 2020 г. № 5-ФЗ"
    ]
    assert by_id["1-fz:ст.2:ч.1"].changed_by == []


def test_act_requisites_from_garant_head() -> None:
    found = garant.act_requisites(LAW)
    assert found["title"] == 'Федеральный закон от 1 января 2000 г. N 1-ФЗ "О синтетике"'
    assert found["edition"] == "1-ФЗ ред. от 04.08.2026"  # последняя дата изменений
    plain = ParsedDocument(source="x", blocks=[para('Приказ от 5 мая 2020 г. N 7 "О порядке"')])
    assert garant.act_requisites(plain)["edition"] == "7 от 05.05.2020"


ORDER = ParsedDocument(
    source="order.pdf",
    blocks=[
        para('Приказ Министерства от 6 апреля 2021 г. N 245 "Об утверждении Порядка"', H),
        para("С изменениями и дополнениями от:", H),
        EDITIONS,
        para("1.Утвердить прилагаемый Порядок."),
        para("2. Признать утратившим силу приказ от 5 апреля 2017 г. N 301."),
        para("Министр И.О. Фамилия"),
        para("Приложение", H),
        para("УТВЕРЖДЕН", H),
        para("приказом Министерства"),
        para("от 6 апреля 2021 г. N 245", H),
        para("Порядок организации синтетической деятельности", H),
        para("С изменениями и дополнениями от:", H),
        EDITIONS,
        para("I. Общие положения", H),
        para("1. Настоящий порядок является обязательным для организаций."),
        para("2. Особенности устанавливаются стандартами."),
        para("II. Организация деятельности", H),
        para("3. Программы реализуются по направлениям подготовки:"),
        para("по направлению подготовки одну программу;"),
        note(
            "Приложение дополнено пунктом 3(1) с 7 апреля 2023 г.",
            "Приказ Минобрнауки России от 2 марта 2023 г. № 244",
        ),
        para("3(1). Обучающиеся вправе выбирать дисциплины."),
        note(
            "Приложение дополнено сноской 11.1",
            "Приказ Минобрнауки России от 2 марта 2023 г. № 244",
        ),
        para("4. Утратил силу."),
        para("5. Величина зачетной единицы устанавливается организацией."),
        para("──────────────────────────────"),
        para("1 Часть 6 статьи 79 Федерального закона."),
        para("Приложение N 2", H),
        para("Примерная форма договора", H),
        para("1. Предмет договора"),
    ],
)


def test_order_is_cut_into_points_of_the_approved_document() -> None:
    doc = chunk_act(ORDER, info("order-245"), "order")
    by_id = {f.id: f for f in doc.fragments}
    assert list(by_id) == [
        "order-245:1",
        "order-245:2",
        "order-245:3",
        "order-245:3(1)",
        "order-245:5",
    ]  # текст приказа, утративший силу пункт 4, сноски и форма договора — не берутся
    assert (
        by_id["order-245:1"].text == "1. Настоящий порядок является обязательным для организаций."
    )
    assert by_id["order-245:3"].text.endswith("по направлению подготовки одну программу;")
    assert by_id["order-245:3"].path == (
        "Порядок организации синтетической деятельности › II. Организация деятельности › п. 3"
    )
    assert [n.since for n in by_id["order-245:3(1)"].changed_by] == ["с 7 апреля 2023 г."]
    assert by_id["order-245:5"].changed_by == []  # отметка о сноске ни к чему не относится


def test_order_without_approved_document_is_an_error() -> None:
    doc = chunk_act(ParsedDocument(source="x", blocks=[para("1. Утвердить.")]), info("x"), "order")
    assert doc.fragments == [] and any(w.startswith("ОШИБКА") for w in doc.warnings)
