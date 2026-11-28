"""Чанкер нормативов на синтетических блоках (вёрстка как у выгрузок ФГОС и ОС СПбГУ)."""

import pytest

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus import (
    DocumentInfo,
    Overrides,
    chunk_document,
    detect_info,
)
from edu_doc_agent.retrieval.corpus.chunker import MAX_PART_CHARS

H, P, T = BlockKind.HEADING, BlockKind.PARAGRAPH, BlockKind.TABLE

FGOS_INFO = DocumentInfo(
    doc_id="fgos-02.03.02",
    doc_type="fgos",
    title="ФГОС ВО — бакалавриат по направлению 02.03.02",
    edition="808 ред. от 08.02.2021",
    source_file="020302.pdf",
)


def p(text: str) -> Block:
    return Block(P, text)


def h(text: str) -> Block:
    return Block(H, text, level=1)


def t(*rows: list[str]) -> Block:
    return Block(T, rows=[list(r) for r in rows])


def fgos(*blocks: Block) -> ParsedDocument:
    preamble = [
        p("Приказ Министерства науки и высшего образования РФ от 23 августа 2017 г. N 808"),
        p("УТВЕРЖДЕН приказом … 1. Утвердить прилагаемый ФГОС."),  # номер в преамбуле — не пункт
    ]
    return ParsedDocument(source="fgos.pdf", blocks=[*preamble, *blocks])


def chunk(*blocks: Block, overrides: Overrides | None = None):
    return chunk_document(fgos(*blocks), FGOS_INFO, overrides)


def by_id(doc) -> dict:
    return {f.id: f for f in doc.fragments if f.part == 0}


def test_clauses_sections_and_preamble() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Настоящий ФГОС представляет собой"),
        p("совокупность обязательных требований."),  # продолжение пункта
        p("1.2. Получение образования допускается"),
        h("II. Требования к структуре программы"),
        h("бакалавриата"),  # перенос длинного заголовка раздела
        p("2.1.Структура программы включает блоки:"),  # без пробела после номера
        t(["Структура", "Объем"], ["Блок 1", "не менее 160"]),
        p("2.1.1 Блок 1 «Дисциплины (модули)»;"),  # без точки после номера
    )
    frags = by_id(doc)
    assert list(frags) == [
        "fgos-02.03.02:1.1",
        "fgos-02.03.02:1.2",
        "fgos-02.03.02:2.1",
        "fgos-02.03.02:2.1.1",
    ]
    assert frags["fgos-02.03.02:1.1"].text == (
        "1.1. Настоящий ФГОС представляет собой\nсовокупность обязательных требований."
    )
    assert frags["fgos-02.03.02:1.1"].path == "I. Общие положения › 1.1"
    clause = frags["fgos-02.03.02:2.1"]
    assert clause.path == "II. Требования к структуре программы бакалавриата › 2.1"
    assert clause.text.endswith("Структура | Объем\nБлок 1 | не менее 160")
    assert all(f.kind == "clause" and f.edition == FGOS_INFO.edition for f in doc.fragments)
    assert doc.warnings == ["не найдены УК", "не найдены ОПК"]  # в примере нет компетенций


def test_competency_table_rows_become_fragments() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Настоящий ФГОС представляет собой совокупность требований."),
        h("III. Требования к результатам освоения"),
        p("3.1. В результате освоения программы должны быть сформированы компетенции."),
        p("3.2. Программа бакалавриата должна устанавливать следующие универсальные компетенции:"),
        t(
            ["Наименование категории (группы) универсальных компетенций", "Код и наименование УК"],
            ["Системное и критическое мышление", "УК-1. Способен осуществлять поиск"],
            ["Разработка и реализация проектов", "УК-2. Способен определять круг задач"],
        ),
        # продолжение таблицы на следующей странице — без шапки
        t(["Командная работа и лидерство", "УК‑3. Способен осуществлять взаимодействие"]),
        p("3.3. Программа бакалавриата должна устанавливать общепрофессиональные компетенции:"),
        p("ОПК-1. Способен применять фундаментальные знания;"),  # компетенции абзацами
        p("ОПК-2. Способен разрабатывать алгоритмы."),
    )
    frags = by_id(doc)
    uk3 = frags["fgos-02.03.02:3.2:УК-3"]
    assert uk3.kind == "competency"
    assert uk3.parent == "fgos-02.03.02:3.2"
    assert uk3.category == "Командная работа и лидерство"
    assert uk3.codes == ["УК-3"]
    assert uk3.path == "III. Требования к результатам освоения › 3.2 › УК-3"
    assert frags["fgos-02.03.02:3.2:УК-1"].category == "Системное и критическое мышление"
    intro = frags["fgos-02.03.02:3.2"]
    assert intro.text.endswith("универсальные компетенции:")  # таблица не дублируется в пункте
    assert frags["fgos-02.03.02:3.3:ОПК-2"].text == "ОПК-2. Способен разрабатывать алгоритмы."
    assert frags["fgos-02.03.02:3.3:ОПК-1"].category is None
    ids = [f.id for f in doc.fragments]
    assert ids.index("fgos-02.03.02:3.2") < ids.index("fgos-02.03.02:3.2:УК-1")
    assert doc.warnings == []


def test_competency_gap_is_reported() -> None:
    doc = chunk(
        h("III. Требования к результатам освоения"),
        p("3.1. Программа устанавливает компетенции:"),
        p("УК-1. Способен к поиску."),
        p("УК-3. Способен к работе в команде."),
        p("ОПК-1. Способен применять знания."),
    )
    assert "УК не подряд: УК-1, УК-3" in doc.warnings


def test_change_notes_before_and_after_clause() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Текст пункта."),
        p("Информация об изменениях:"),
        p(
            "Пункт 1.2 изменен с 1 сентября 2021 г. - Приказ Минобрнауки России "
            "от 26 ноября 2020 г. N 1456 См. предыдущую редакцию"
        ),
        p("1.2. Новая редакция пункта."),
        p("1.3. Утратил силу."),
        p("Информация об изменениях:"),  # заголовок без текста отметки — не съедает пункт
        p("1.4. Следующий пункт."),
        p("Информация об изменениях:"),
        p(
            "Пункт 1.4 дополнен с 26 марта 2021 г. - "
            "Приказ Минобрнауки России от 8 февраля 2021 г. N 83"
        ),
    )
    frags = by_id(doc)
    assert list(frags) == [f"fgos-02.03.02:1.{i}" for i in range(1, 5)]
    note = frags["fgos-02.03.02:1.2"].changed_by[0]
    assert note.since == "с 1 сентября 2021 г."
    assert note.act == "Приказ Минобрнауки России от 26 ноября 2020 г. N 1456"
    assert frags["fgos-02.03.02:1.4"].changed_by[0].act.endswith("N 83")  # отметка после пункта
    assert frags["fgos-02.03.02:1.1"].changed_by == []
    assert "Информация" not in frags["fgos-02.03.02:1.3"].text


def test_appendix_prof_standards() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Текст пункта."),
        h("Приложение к ФГОС"),
        p("Перечень профессиональных стандартов,"),
        p("соответствующих профессиональной деятельности выпускников"),
        t(
            ["N п/п", "Код профессионального стандарта", "Наименование"],
            ["", "06 Связь, информационные и коммуникационные технологии", ""],
            ["1.", "06.001", 'Профессиональный стандарт "Программист"'],
            ["", "40 Сквозные виды деятельности", ""],
            ["2.", "40.011", 'Профессиональный стандарт "Специалист по НИОКР"'],
        ),
    )
    frags = by_id(doc)
    ps = frags["fgos-02.03.02:app:06.001"]
    assert ps.kind == "prof_standard_ref"
    assert ps.text == 'Профессиональный стандарт "Программист"'
    assert ps.codes == ["06.001"]
    assert ps.category == "06 Связь, информационные и коммуникационные технологии"
    assert ps.path == (
        "Приложение › Перечень профессиональных стандартов, соответствующих "
        "профессиональной деятельности выпускников › 06.001"
    )
    assert frags["fgos-02.03.02:app:40.011"].category == "40 Сквозные виды деятельности"
    assert frags["fgos-02.03.02:1.1"].text == "1.1. Текст пункта."


def test_out_of_order_number_stays_in_clause() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Первый пункт."),
        p("1.2. Второй пункт:"),
        p("1.1.1. опечатка нумерации в самом документе;"),
    )
    frags = by_id(doc)
    assert frags["fgos-02.03.02:1.2"].text.endswith("опечатка нумерации в самом документе;")
    assert "номер 1.1.1 не по порядку — оставлен в тексте" in doc.warnings


def test_long_clause_split_losslessly() -> None:
    sentence = "Организация обеспечивает условия реализации программы бакалавриата. "
    text = "4.1. " + sentence * 60 + "Хвост."
    doc = chunk(h("IV. Требования к условиям"), p(text.strip()))
    parts = [f for f in doc.fragments if f.id == "fgos-02.03.02:4.1"]
    assert [f.part for f in parts] == list(range(len(parts)))
    assert len(parts) > 1
    assert "".join(f.text for f in parts) == text.strip()
    assert all(len(f.text) <= MAX_PART_CHARS for f in parts[:-1])
    assert len(parts[-1].text) > len("Хвост.")  # короткий хвост присоединён к предыдущей части


def test_overrides_replace_add_drop() -> None:
    blocks = (
        h("III. Требования к результатам освоения"),
        p("3.1. Программа устанавливает компетенции: Самоорганизация мусор"),
        p("УК-1. Способен к поиску."),
        p("УК-3. Способен к работе в команде."),
        p("3.2. Лишний пункт."),
    )
    overrides = Overrides(
        replace={
            "fgos-02.03.02:3.1": "3.1. Программа устанавливает компетенции:",
            "fgos-02.03.02:3.1:УК-3": {"category": "Командная работа", "kind": "clause"},
        },
        add=[
            {
                "id": "fgos-02.03.02:3.1:УК-2",
                "kind": "competency",
                "number": "3.1",
                "path": "III › 3.1 › УК-2",
                "parent": "fgos-02.03.02:3.1",
                "text": "УК-2. Способен определять круг задач.",
            }
        ],
        drop=["fgos-02.03.02:3.2", "fgos-02.03.02:9.9"],
    )
    doc = chunk(*blocks, overrides=overrides)
    ids = [f.id for f in doc.fragments]
    assert ids == [
        "fgos-02.03.02:3.1",
        "fgos-02.03.02:3.1:УК-1",
        "fgos-02.03.02:3.1:УК-2",  # вставлен по порядку кодов
        "fgos-02.03.02:3.1:УК-3",
    ]
    frags = by_id(doc)
    assert frags["fgos-02.03.02:3.1"].text == "3.1. Программа устанавливает компетенции:"
    assert frags["fgos-02.03.02:3.1:УК-2"].codes == ["УК-2"]
    assert frags["fgos-02.03.02:3.1:УК-2"].edition == FGOS_INFO.edition
    assert frags["fgos-02.03.02:3.1:УК-3"].category == "Командная работа"
    assert frags["fgos-02.03.02:3.1:УК-3"].kind == "competency"  # kind менять нельзя
    assert "overrides: фрагмента fgos-02.03.02:9.9 нет" in doc.warnings
    assert "overrides: у fgos-02.03.02:3.1:УК-3 нельзя менять kind" in doc.warnings
    assert not any(w.startswith("УК не подряд") for w in doc.warnings)


def test_spbu_standard_profile_skips_appendix() -> None:
    info = DocumentInfo(
        doc_id="spbu-os",
        doc_type="spbu_standard",
        title="Образовательный стандарт СПбГУ",
        edition="11144/1 от 24.08.2026",
        source_file="os.pdf",
    )
    parsed = ParsedDocument(
        source="os.pdf",
        blocks=[
            p("ПРИКАЗ 1. Утвердить Образовательный стандарт."),
            h("1. Общие положения"),
            p("1.1. Настоящий образовательный стандарт устанавливает требования."),
            h("2. Требования к структуре образовательной программы"),
            p("2.1. Образовательная программа включает:"),
            p("2.1.1. общую характеристику;"),
            p("Приложение к Образовательному стандарту СПбГУ,"),
            p("УКБ-1: Способен участвовать в разработке"),
        ],
    )
    doc = chunk_document(parsed, info)
    assert [f.id for f in doc.fragments] == ["spbu-os:1.1", "spbu-os:2.1", "spbu-os:2.1.1"]
    assert doc.fragments[1].path == "2. Требования к структуре образовательной программы › 2.1"
    assert doc.warnings == []


@pytest.mark.parametrize(
    ("head", "edition"),
    [
        (  # выгрузка ГАРАНТ
            'Приказ Министерства науки и высшего образования РФ от 23 августа 2017 г. N 808 "Об '
            "утверждении федерального государственного образовательного стандарта высшего "
            "образования - бакалавриат по направлению подготовки 02.03.02 Фундаментальная "
            'информатика и информационные технологии" С изменениями и дополнениями от: '
            "26 ноября 2020 г., 8 февраля 2021 г.",
            "808 ред. от 08.02.2021",
        ),
        (  # выгрузка КонсультантПлюс
            'Приказ Минобрнауки России от 23.08.2017 N 808 (ред. от 08.02.2021) "Об утверждении '
            "федерального государственного образовательного стандарта высшего образования - "
            "бакалавриат по направлению подготовки 02.03.02 Фундаментальная информатика и "
            'информационные технологии"',
            "808 ред. от 08.02.2021",
        ),
    ],
)
def test_detect_info_fgos(head: str, edition: str) -> None:
    info = detect_info(ParsedDocument(source="x.pdf", blocks=[p(head)]), "fgos", "x.pdf")
    assert info.doc_id == "fgos-02.03.02"
    assert info.level == "бакалавриат"
    assert info.direction_name == "Фундаментальная информатика и информационные технологии"
    assert info.approval_act == "Приказ Минобрнауки России от 23.08.2017 N 808"
    assert info.edition == edition


def test_detect_info_manifest_wins_and_required_fields() -> None:
    parsed = ParsedDocument(source="x.pdf", blocks=[p("Образовательный стандарт СПбГУ")])
    with pytest.raises(ValueError, match="doc_id, title, edition"):
        detect_info(parsed, "spbu_standard", "x.pdf")
    info = detect_info(
        parsed, "spbu_standard", "x.pdf", doc_id="spbu-os", title="ОС СПбГУ", edition="1 от 2026"
    )
    assert (info.doc_id, info.title, info.edition) == ("spbu-os", "ОС СПбГУ", "1 от 2026")


def test_context_from_short_parent_clauses_and_category() -> None:
    doc = chunk(
        h("I. Общие положения"),
        p("1.1. Каждая программа представлена комплектом документации, включающей:"),
        p("1.1.1. рабочие программы дисциплин, содержащие:"),
        p("1.1.1.1. цели и задачи дисциплины;"),
        p("1.2. " + "Длинный самостоятельный пункт. " * 20),
        p("1.2.1. подпункт длинного пункта."),
        h("III. Требования к результатам освоения"),
        p("3.2. Программа устанавливает универсальные компетенции:"),
        t(
            ["Наименование категории (группы) универсальных компетенций", "Код и наименование"],
            ["Системное и критическое мышление", "УК-1. Способен к поиску"],
        ),
    )
    frags = by_id(doc)
    assert frags["fgos-02.03.02:1.1.1.1"].context == (
        "1.1. Каждая программа представлена комплектом документации, включающей:\n"
        "1.1.1. рабочие программы дисциплин, содержащие:"
    )
    assert frags["fgos-02.03.02:1.1.1.1"].text == "1.1.1.1. цели и задачи дисциплины;"  # дословно
    assert frags["fgos-02.03.02:1.2.1"].context == ""  # длинный родитель не контекст
    assert frags["fgos-02.03.02:1.1"].context == ""
    assert frags["fgos-02.03.02:3.2:УК-1"].context == (
        "3.2. Программа устанавливает универсальные компетенции:\nСистемное и критическое мышление"
    )
