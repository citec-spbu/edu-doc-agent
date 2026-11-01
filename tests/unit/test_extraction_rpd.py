"""Тесты извлечения из РПД: синтетическая фикстура и реальные РПД из data/ (docx → md парсером)."""

import re
from pathlib import Path

import pytest

from edu_doc_agent.extraction import RPDExtraction, extract_rpd, extract_rpd_file
from edu_doc_agent.extraction.md_utils import clean_text, parse_tables
from edu_doc_agent.extraction.models import LearningOutcome, SourceRef
from edu_doc_agent.extraction.patterns import COMPETENCE_RE, INDICATOR_RE, OUTCOME_RE, code_key
from tests.real_data import REAL_MD, find_md

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "rpd_synthetic.md"


def _is_rpd(path: Path) -> bool:
    head = path.read_text(encoding="utf-8")[:1500]
    return "".join(head.split()).upper().find("РАБОЧАЯПРОГРАММА") >= 0


REAL_RPD = [p for p in REAL_MD if _is_rpd(p)]

needs_real_data = pytest.mark.skipif(not REAL_RPD, reason="нет реальных docx РПД в data/")


# --- коды ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["УК-1", "ОПК-2", "ПКА-1", "УНК-3", "УКБ-2", "ПКП-ППК-ОС3", "ПКП-УНК-01", "ПКП-16-ППК-ОС3"],
)
def test_competence_code_full_match(text: str) -> None:
    m = COMPETENCE_RE.search(text)
    assert m and m.group(0) == text


def test_codes_are_not_normalized() -> None:
    assert COMPETENCE_RE.search("ПКП-ППК-ОС3").group(0) != COMPETENCE_RE.search(
        "ПКП-16-ППК-ОС3"
    ).group(0)


def test_indicator_and_outcome_codes() -> None:
    assert INDICATOR_RE.search("ПКП-УНК-03.01. З-1").group(0) == "ПКП-УНК-03.01"
    assert OUTCOME_RE.search("ПКП-ППК-ОС3.1. З-2. Знает").group(0) == "ПКП-ППК-ОС3.1. З-2"
    assert OUTCOME_RE.search("ПКП-ППК-Р2.1. 3-1. Знает")  # опечатка: цифра 3 вместо З
    assert not INDICATOR_RE.search("ПОДУК-1.2")  # код внутри слова не считаем


def test_clean_text_normalizes_markup_and_dashes() -> None:
    assert clean_text("ПКП–ППК—Р1 \u00a0x") == "ПКП-ППК-Р1 x"


def test_parse_tables_keeps_line_numbers_and_sections() -> None:
    md = "# A\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\ntext\n\n## B\n| c |\n"
    t1, t2 = parse_tables(md)
    assert t1.rows == [["a", "b"], ["1", "2"]] and t1.row_lines == [3, 5] and t1.section == "A"
    assert t2.rows == [["c"]] and t2.section == "B"


# --- синтетический РПД -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic() -> RPDExtraction:
    return extract_rpd_file(FIXTURE)


def test_synthetic_title_block(synthetic: RPDExtraction) -> None:
    assert synthetic.doc_kind == "discipline"
    assert synthetic.title == "Синтетическая дисциплина"
    assert synthetic.title_en == "Synthetic Discipline"
    assert synthetic.registration_number == "000123"
    assert synthetic.credits == 3.0
    assert synthetic.warnings == []


def test_synthetic_competences(synthetic: RPDExtraction) -> None:
    got = {c.code: c.name for c in synthetic.competences}
    assert got == {
        "ПКП-ППК-ТС1": "Способен проектировать синтетические системы",
        "ОПК-9": "Способен применять тестовые методы",
        "ПКП-16-ППК-ТС2": "способен оценивать качество",
    }


def test_synthetic_indicators(synthetic: RPDExtraction) -> None:
    by_code = {i.code: i for i in synthetic.indicators}
    assert list(by_code) == [
        "ПКП-ППК-ТС1.1",
        "ПКП-ППК-ТС1.2",
        "ОПК-9.1",
        "ОПК-9.2",
        "ПКП-ППК-ТС2.1",
    ]
    assert by_code["ПКП-ППК-ТС1.1"].level == "Средний"
    assert by_code["ПКП-ППК-ТС1.1"].text == "Строит модели систем"
    # в строке без столбца компетенции компетенция берётся из предыдущей строки
    assert by_code["ОПК-9.2"].competence_code == "ОПК-9"
    assert by_code["ПКП-ППК-ТС1.2"].competence_code == "ПКП-ППК-ТС1"


def test_synthetic_outcomes_all_styles(synthetic: RPDExtraction) -> None:
    outs = synthetic.learning_outcomes
    # префиксный код + маркеры `•`
    first = [o for o in outs if o.indicator_code == "ПКП-ППК-ТС1.1"]
    assert [(o.kind, o.code) for o in first] == [
        ("знает", "ПКП-ППК-ТС1.1. З-1"),
        ("знает", "ПКП-ППК-ТС1.1. З-2"),
        ("умеет", "ПКП-ППК-ТС1.1. У-1"),
    ]
    assert first[0].text == "устройство синтетических систем."
    # суффиксный код в скобках, маркеры `•`
    second = [o for o in outs if o.indicator_code == "ПКП-ППК-ТС1.2"]
    assert [o.text for o in second] == ["основы тестирования", "виды проверок", "писать тесты"]
    # без кодов вообще: вид берётся из заголовка «знает:/умеет:/владеет:»
    plain = [o for o in outs if o.competence_codes == ["ОПК-9"]]
    assert all(o.code is None for o in plain)
    assert [o.kind for o in plain] == ["знает", "умеет", "знает", "владеет"]
    # код без маркеров списка и опечатка «3-1»
    third = [o for o in outs if o.competence_codes == ["ПКП-16-ППК-ТС2"]]
    assert [(o.kind, o.code) for o in third] == [
        ("знает", "ПКП-16-ППК-ТС2.1. 3-1"),
        ("умеет", "ПКП-16-ППК-ТС2.1. У-1"),
    ]


def test_synthetic_assessments(synthetic: RPDExtraction) -> None:
    got = {a.indicator_code: a.materials for a in synthetic.assessments}
    assert got == {
        "ПКП-ППК-ТС1.1": ["Практические занятия 1-2", "тест", "вопросы к зачёту"],
        "ПКП-ППК-ТС1.2": ["Семинар 3"],
        "ОПК-9.1": ["Вопросы к экзамену"],
    }


def test_synthetic_control_forms(synthetic: RPDExtraction) -> None:
    got = [
        (f.semester, f.current, [(a.kind, a.details) for a in f.intermediate])
        for f in synthetic.control_forms
    ]
    assert got == [
        (3, ["практические работы", "тесты"], [("зачёт с оценкой", [])]),
        (4, [], [("экзамен", [])]),
    ]
    assert synthetic.control_forms[0].current_raw == "практические работы, тесты"


def test_source_points_to_real_line(synthetic: RPDExtraction) -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    for ind in synthetic.indicators:
        assert ind.code in lines[ind.source.line - 1]
        assert ind.source.section.startswith("1.3.")


def test_missing_tables_give_warnings() -> None:
    result = extract_rpd("# Пусто\n\nтекст", file="x.md")
    assert len(result.warnings) == 4 and not result.competences


# --- регрессии: случаи, найденные на реальном корпусе ---------------------------------------

HEAD = (
    "| Код и наименование компетенции | Планируемые результаты обучения | Код индикатора |\n"
    "| --- | --- | --- |\n"
)


def _title_doc(title: str) -> str:
    return (
        f"### Р А Б О Ч А Я П Р О Г Р А М М А\n\n### учебной дисциплины\n\n{title}\n\nEng\n\n"
        "### Язык(и) обучения\n\nрусский\n"
    )


def test_title_starting_with_word_yazyk_is_not_mistaken_for_language_line() -> None:
    assert (
        extract_rpd(_title_doc("Языки программирования"), "x.md").title == "Языки программирования"
    )


def test_outcomes_table_row_without_outcomes_column() -> None:
    md = HEAD + "| ОПК-1 Способен применять знания |  | ОПК-1.2 Использует знания |\n"
    r = extract_rpd(md, "x.md")
    assert [(c.code, c.name) for c in r.competences] == [("ОПК-1", "Способен применять знания")]
    assert [i.code for i in r.indicators] == ["ОПК-1.2"] and r.learning_outcomes == []


def test_outcomes_table_with_number_and_group_columns() -> None:
    md = (
        "| № | Группа | Код и наименование компетенции "
        "| Планируемые результаты обучения | Индикатор |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 1 | ОПК | ОПК-1: Способен применять знания "
        "| Знать основы; Уметь считать; | ОПК-1.1. Умеет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert r.competences[0].code == "ОПК-1" and r.competences[0].name == "Способен применять знания"
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "основы"),
        ("умеет", "считать"),
    ]


def test_competence_name_before_code_and_kind_headings_without_colon() -> None:
    md = (
        HEAD
        + "| способен проектировать (ПКП-МО-1) "
        + "| знать<br>• основы ML<br>уметь<br>• обучать модели "
        + "| ПКП-МО-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert (r.competences[0].code, r.competences[0].name) == ("ПКП-МО-1", "способен проектировать")
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "основы ML"),
        ("умеет", "обучать модели"),
    ]


def test_indicators_sharing_one_text() -> None:
    md = HEAD + "| ОПК-1. Способен | • Знает основы. | ОПК-1.1; ОПК-1.2: применяет знания. |\n"
    r = extract_rpd(md, "x.md")
    assert [(i.code, i.text) for i in r.indicators] == [
        ("ОПК-1.1", "применяет знания"),
        ("ОПК-1.2", "применяет знания"),
    ]


ASSESS_HEAD = (
    "| № | Код индикатора | Контрольно-измерительные материалы (КИМ) |\n| --- | --- | --- |\n"
)


def test_assessment_rows_continuation_multi_indicator_and_empty_kim() -> None:
    md = (
        ASSESS_HEAD
        + "| 1 | ОПК-1.1 Первый | Тест; Экзамен |\n"
        + "|  | ОПК-1.2 Второй | Тест |\n"
        + "| 2 | УК-1.1 Один  УК-1.2 Два | Контрольная |\n"
        + "| 3 | ОПК-2.1 Без КИМ |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(a.indicator_code, a.materials) for a in r.assessments] == [
        ("ОПК-1.1", ["Тест", "Экзамен"]),
        ("ОПК-1.2", ["Тест"]),
        ("УК-1.1", ["Контрольная"]),
        ("УК-1.2", ["Контрольная"]),
        ("ОПК-2.1", []),
    ]


# --- реальные документы (пропускаются, если в data/ нет docx)  ---


@needs_real_data
@pytest.mark.parametrize("path", REAL_RPD, ids=lambda p: p.stem)
def test_real_rpd_parser_invariants(path: Path) -> None:
    """Что должен найти парсер в любом РПД. Полнота оценочных средств и названий компетенций —
    свойство документа, а не парсера, поэтому здесь не проверяется."""
    r = extract_rpd_file(path)
    assert r.doc_kind in ("discipline", "practice") and r.title
    assert r.registration_number and re.fullmatch(r"\d{5,6}", r.registration_number)
    assert r.credits and r.credits > 0
    assert r.competences and r.indicators and r.learning_outcomes
    if r.doc_kind == "discipline":
        assert r.control_forms
        # в части РПД в 3.1.4 нет таблицы «индикатор → КИМ» (только перечень вопросов):
        # тогда об этом должно быть предупреждение
        assert r.assessments or any("оценочных средств" in w for w in r.warnings)
    assert all(INDICATOR_RE.fullmatch(i.code) for i in r.indicators)
    assert all(o.competence_codes and o.kind for o in r.learning_outcomes)
    assert all(f.intermediate for f in r.control_forms)
    # разметка парсера не должна просачиваться в значения
    texts = [i.text for i in r.indicators] + [o.text for o in r.learning_outcomes]
    assert not any("<br>" in t or "\n" in t for t in texts)


@needs_real_data
def test_real_corpus_has_no_unclassified_outcomes() -> None:
    for path in REAL_RPD:
        r = extract_rpd_file(path)
        assert r.competences, path.name
        assert all(o.kind for o in r.learning_outcomes), path.name


@needs_real_data
def test_real_math_analysis_exact_values() -> None:
    path = find_md("Математический анализ I")
    if path is None:
        pytest.skip("нет «Математический анализ I» в data/")
    r = extract_rpd_file(path)
    assert (r.doc_kind, r.title, r.title_en) == (
        "discipline",
        "Математический анализ I",
        "Mathematical Analysis I",
    )
    assert (r.registration_number, r.credits) == ("001856", 6.0)
    assert [c.code for c in r.competences] == ["УК-1", "ПКП-УНК-03", "ПКА-1", "ОПК-1"]
    assert len(r.indicators) == 7 and len(r.assessments) == 7
    assert {i.code: i.level for i in r.indicators}["ПКП-УНК-03.01"] == "Базовый"
    assert r.assessments[0].materials == ["Контрольные работы", "Вопросы к экзамену"]
    assert [(f.semester, [a.kind for a in f.intermediate]) for f in r.control_forms] == [
        (1, ["зачёт", "экзамен"])
    ]
    uk1 = [o for o in r.learning_outcomes if o.competence_codes == ["УК-1"]]
    assert [(o.kind, o.text) for o in uk1[:2]] == [
        ("умеет", "аргументировано, логически верно строить устную речь"),
        (
            "умеет",
            "использовать математический язык и символику при построении математических моделей",
        ),
    ]


@needs_real_data
def test_real_practice_exact_values() -> None:
    path = find_md("С2-5_ТОП__Учебная практика")
    if path is None:
        pytest.skip("нет РПП в data/")
    r = extract_rpd_file(path)
    assert (r.doc_kind, r.registration_number, r.credits) == ("practice", "064682", 18.0)
    assert r.title == "Учебная практика (проектно-технологическая)"
    first = next(o for o in r.learning_outcomes if o.code == "ПКП-ППК-Р1.1. 3-1")
    assert first.kind == "знает" and first.indicator_code == "ПКП-ППК-Р1.1"
    assert first.text == "алгоритмы решения типовых задач, области и способы их применения"


# --- формат вывода парсера: заголовки «###» в титуле, переносы строк в ячейках как <br> ---

_NEW_FORMAT = """### Санкт-Петербургский государственный университет

### Р А Б О Ч А Я П Р О Г Р А М М А

### учебной дисциплины

Тестовая дисциплина

Test Discipline

### Язык(и) обучения

русский

Трудоёмкость (границы трудоёмкости) в зачетных единицах: 3

Регистрационный номер рабочей программы: 000001

## 1.3. Перечень результатов обучения (learning outcomes)

| Код и наименование компетенции | Планируемые результаты обучения | Код индикатора и индикатор |
| --- | --- | --- |
|УК-1 а<br>б|знает:<br>ааа;<br>умеет:<br>ббб;<br>владеет:<br>ввв.|УК-1.1 Т1<br>УК-1.2 Т2 (Базовый)|

### 2.1. Организация

| Период | лекции | практические занятия | Трудоёмкость |
| --- | --- | --- | --- |
| Семестр 2 | 48 | 60 | 3 |

### 3.1.4. Методические материалы

| № | Код индикатора и индикатор | Контрольно-измерительные материалы |
| --- | --- | --- |
| 1 | УК-1.1 Анализирует задачу | Контрольные работы<br>Вопросы к экзамену |

| Период | Формы текущего контроля успеваемости | Виды промежуточной аттестации |
| --- | --- | --- |
| Семестр 2 | Контрольные работы<br>Коллоквиум | зачет, экзамен |
"""


def test_new_format_title_and_cells() -> None:
    r = extract_rpd(_NEW_FORMAT, file="new.md")
    assert (r.doc_kind, r.title, r.title_en) == (
        "discipline",
        "Тестовая дисциплина",
        "Test Discipline",
    )
    assert (r.registration_number, r.credits) == ("000001", 3.0)
    assert [(c.code, c.name) for c in r.competences] == [("УК-1", "а б")]
    assert [(i.code, i.text, i.level) for i in r.indicators] == [
        ("УК-1.1", "Т1", None),
        ("УК-1.2", "Т2", "Базовый"),
    ]
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "ааа"),
        ("умеет", "ббб"),
        ("владеет", "ввв."),
    ]
    assert r.assessments[0].materials == ["Контрольные работы", "Вопросы к экзамену"]
    (form,) = r.control_forms
    assert form.semester == 2 and form.current == ["Контрольные работы", "Коллоквиум"]
    assert [a.kind for a in form.intermediate] == ["зачёт", "экзамен"]
    assert r.warnings == []


def test_clean_text_turns_br_into_newline() -> None:
    assert clean_text("a<br>b <br/> c") == "a\nb\nc"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("экзамен", [("экзамен", [])]),
        ("Зачет", [("зачёт", [])]),
        ("зачет, экзамен", [("зачёт", []), ("экзамен", [])]),
        ("зачёт, устно, традиционная форма", [("зачёт", ["устно", "традиционная форма"])]),
        (
            "зачёт, устно, по результатам работы в семестре, экзамен, устно, традиционная форма",
            [
                ("зачёт", ["устно", "по результатам работы в семестре"]),
                ("экзамен", ["устно", "традиционная форма"]),
            ],
        ),
        ("зачёт (защита проекта)", [("зачёт", ["защита проекта"])]),
        ("дифференцированный зачёт", [("дифференцированный зачёт", [])]),
    ],
)
def test_parse_attestations(raw: str, expected: list) -> None:
    from edu_doc_agent.extraction.rpd import _parse_attestations

    assert [(a.kind, a.details) for a in _parse_attestations(raw)] == expected


def test_split_items_ignores_commas_in_parentheses_and_drops_repeats() -> None:
    from edu_doc_agent.extraction.rpd import _split_items

    assert _split_items("а (б, в), г; г") == ["а (б, в)", "г"]
    assert _split_items(None) == []


# --- регрессии корпуса docx -> md (75 документов) ------------------------------------------------


def test_kind_headings_without_colon_on_own_line() -> None:
    """«Знать<br>…<br>Уметь<br>…» — вид задан словом в отдельной строке, без двоеточия."""
    md = (
        HEAD
        + "| ОПК-1 Способен | Знать<br>основы<br>принципы; методы<br>Уметь<br>считать "
        + "| ОПК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "основы"),
        ("знает", "принципы; методы"),
        ("умеет", "считать"),
    ]


def test_kind_word_inside_sentence_is_not_a_heading() -> None:
    md = HEAD + "| ОПК-1 Способен | знает:<br>как студент умеет учиться | ОПК-1.1 Применяет |\n"
    r = extract_rpd(md, "x.md")
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "как студент умеет учиться")
    ]


def test_table_header_with_br_inside_phrases() -> None:
    """Заголовок «Планируемые<br>результаты<br>обучения» и столбец «№» не мешают найти таблицу."""
    md = (
        "| № | Код и<br>наименование<br>компетенции | Планируемые<br>результаты<br>обучения "
        "| Код индикатора |\n| --- | --- | --- | --- |\n"
        "|  | 2 | 3 | 4 |\n"
        "| 1 | УК-1 Способен | знать<br>основы | УК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [c.code for c in r.competences] == ["УК-1"]
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [("знает", "основы")]
    assert [i.code for i in r.indicators] == ["УК-1.1"]
    assert not any("1.3" in w for w in r.warnings)


def test_vertically_merged_outcomes_cell_is_taken_once() -> None:
    """Парсер повторяет объединённую по вертикали ячейку в каждой строке: результаты берём один
    раз, а компетенции всех строк группы собираем в `competence_codes`."""
    same = "знает:<br>основы<br>умеет:<br>считать"
    md = (
        HEAD
        + f"| УКБ-3 Способен | {same} | УКБ-3.1 Один |\n"
        + f"| ОПК-1 Способен | {same} | ОПК-1.2 Два |\n"
        + f"| ОПК-2 Способен | {same} | ОПК-2.2 Три |\n"
        + "| ПКП-ППК-ОС2 Способен | знает:<br>другое | ПКП-ППК-ОС2.1 Четыре |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [c.code for c in r.competences] == ["УКБ-3", "ОПК-1", "ОПК-2", "ПКП-ППК-ОС2"]
    assert [(o.competence_codes, o.text) for o in r.learning_outcomes] == [
        (["УКБ-3", "ОПК-1", "ОПК-2"], "основы"),
        (["УКБ-3", "ОПК-1", "ОПК-2"], "считать"),
        (["ПКП-ППК-ОС2"], "другое"),
    ]
    assert [i.code for i in r.indicators] == ["УКБ-3.1", "ОПК-1.2", "ОПК-2.2", "ПКП-ППК-ОС2.1"]


# --- траектории и формы обучения в таблице форм контроля -----------------------------------------

_FORMS_HEAD = (
    "| Период обучения (модуль) | Формы текущего контроля успеваемости "
    "| Виды промежуточной аттестации |\n| --- | --- | --- |\n"
)


def test_control_forms_keep_trajectory_and_study_form() -> None:
    md = (
        _FORMS_HEAD
        + "| ОСНОВНАЯ ТРАЕКТОРИЯ<br>Профиль ТОП-ИТ ПРО |  |  |\n"
        + "| очная форма обучения |  |  |\n"
        + "| Семестр 3 |  | экзамен |\n"
        + "| ТРАЕКТОРИЯ «Профиль Стандарт» |  |  |\n"
        + "| Форма обучения: очная |  |  |\n"
        + "| Семестр 4 | тесты | зачёт |\n"
    )
    a, b = extract_rpd(md, "x.md").control_forms
    assert (a.trajectory, a.study_form, a.semester) == (
        "ОСНОВНАЯ ТРАЕКТОРИЯ Профиль ТОП-ИТ ПРО",
        "очная",
        3,
    )
    assert (b.trajectory, b.study_form, b.semester) == ("ТРАЕКТОРИЯ «Профиль Стандарт»", "очная", 4)


def test_same_semester_in_different_trajectories_is_not_collapsed() -> None:
    md = (
        _FORMS_HEAD
        + "| ТРАЕКТОРИЯ «Профиль ТОП» |  |  |\n| очная форма обучения |  |  |\n"
        + "| Семестр 3 |  | экзамен |\n"
        + "| ТРАЕКТОРИЯ «Профиль Стандарт» |  |  |\n| очная форма обучения |  |  |\n"
        + "| Семестр 3 |  | экзамен |\n"
    )
    forms = extract_rpd(md, "x.md").control_forms
    assert [(f.trajectory, f.semester) for f in forms] == [
        ("ТРАЕКТОРИЯ «Профиль ТОП»", 3),
        ("ТРАЕКТОРИЯ «Профиль Стандарт»", 3),
    ]


def test_identical_rows_within_one_trajectory_are_still_deduplicated() -> None:
    md = (
        _FORMS_HEAD
        + "| ОСНОВНАЯ ТРАЕКТОРИЯ |  |  |\n| очная форма обучения |  |  |\n"
        + "| Семестр 1 |  | экзамен |\n| Семестр 1 |  | экзамен |\n"
    )
    assert len(extract_rpd(md, "x.md").control_forms) == 1


def test_study_forms_are_told_apart() -> None:
    md = (
        _FORMS_HEAD
        + "| ОСНОВНАЯ ТРАЕКТОРИЯ |  |  |\n"
        + "| очная форма обучения |  |  |\n| Семестр 1 |  | экзамен |\n"
        + "| заочная форма обучения |  |  |\n| Семестр 1 |  | экзамен |\n"
    )
    forms = extract_rpd(md, "x.md").control_forms
    assert [f.study_form for f in forms] == ["очная", "заочная"]


# --- практика без таблиц форм контроля и КИМ ------------------------------------------------------


def test_practice_without_control_tables_gives_warnings_not_garbage() -> None:
    md = (
        "### Р А Б О Ч А Я П Р О Г Р А М М А\n\n### П Р А К Т И К И\n\nТестовая практика\n\n"
        "Test Practice\n\n### Язык(и) обучения\n\nрусский\n\n"
        "Трудоемкость в зачетных единицах: 6\n\nРегистрационный номер рабочей программы: 000777\n\n"
        "### 3.1.1. Виды и формы текущего контроля\n\n- Семестр 2: 3 з.е. (108 часов)\n"
    )
    r = extract_rpd(md, "practice.md")
    assert (r.doc_kind, r.title, r.credits) == ("practice", "Тестовая практика", 6.0)
    assert r.control_forms == [] and r.assessments == []
    assert any("формы контроля" in w for w in r.warnings)
    assert any("оценочных средств" in w for w in r.warnings)


@needs_real_data
def test_real_practice_has_outcomes_but_no_control_forms() -> None:
    path = find_md("С2-5_ТОП__Учебная практика")
    if path is None:
        pytest.skip("нет РПП в data/")
    r = extract_rpd_file(path)
    assert r.doc_kind == "practice" and r.learning_outcomes
    assert r.control_forms == []
    assert any("формы контроля" in w for w in r.warnings)


# --- контракт: схема результата ------------------------------------------------------------------


def test_result_roundtrips_through_json(synthetic: RPDExtraction) -> None:
    again = RPDExtraction.model_validate_json(synthetic.model_dump_json())
    assert again == synthetic


def test_result_schema_fields_are_stable() -> None:
    """Потребители читают эти поля; изменение схемы должно быть осознанным."""
    schema = RPDExtraction.model_json_schema()
    assert set(schema["properties"]) == {
        "file",
        "doc_kind",
        "title",
        "title_en",
        "registration_number",
        "credits",
        "competences",
        "indicators",
        "learning_outcomes",
        "assessments",
        "control_forms",
        "prerequisites",
        "control_mentions",
        "workload",
        "workload_table",
        "semesters",
        "discipline_code",
        "language",
        "goals",
        "active_forms",
        "topic_hours",
        "grade_scales",
        "tickets",
        "control_materials",
        "developers",
        "program",
        "practice",
        "prof_standards",
        "standard_links",
        "llm",
        "warnings",
    }
    assert set(schema["$defs"]["ControlForm"]["properties"]) == {
        "semester",
        "trajectory",
        "study_form",
        "current",
        "intermediate",
        "current_raw",
        "intermediate_raw",
        "source",
    }
    outcome_fields = set(schema["$defs"]["LearningOutcome"]["properties"])
    assert "competence_codes" in outcome_fields and "competence_code" not in outcome_fields
    assert schema["additionalProperties"] is False


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        RPDExtraction.model_validate({"file": "x.md", "surprise": 1})


def test_merged_outcomes_group_without_indicators_and_repeated_competence() -> None:
    """Повторная строка группы без индикатора тоже добавляет компетенцию; повторов кода нет."""
    same = "знает:<br>основы"
    md = (
        HEAD
        + f"| УК-1 Способен | {same} | УК-1.1 Один |\n"
        + f"| УК-2 Способен | {same} |  |\n"
        + f"|  | {same} |  |\n"
        + "| УК-3 Способен | умеет:<br>считать | УК-3.1 Три |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [c.code for c in r.competences] == ["УК-1", "УК-2", "УК-3"]
    assert [(o.competence_codes, o.text) for o in r.learning_outcomes] == [
        (["УК-1", "УК-2"], "основы"),
        (["УК-3"], "считать"),
    ]


def test_same_text_in_non_adjacent_rows_is_not_a_merge() -> None:
    md = (
        HEAD
        + "| УК-1 Способен | знает:<br>основы | УК-1.1 Один |\n"
        + "| УК-2 Способен | знает:<br>другое | УК-2.1 Два |\n"
        + "| УК-3 Способен | знает:<br>основы | УК-3.1 Три |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(o.competence_codes, o.text) for o in r.learning_outcomes] == [
        (["УК-1"], "основы"),
        (["УК-2"], "другое"),
        (["УК-3"], "основы"),
    ]


# --- обратная совместимость: старое поле competence_code ------------------------------------------


def _outcome(**kw) -> LearningOutcome:
    return LearningOutcome(text="основы", source=SourceRef(file="x.md"), **kw)


def test_competence_code_is_first_of_competence_codes() -> None:
    assert _outcome(competence_codes=["УКБ-3", "ОПК-1"]).competence_code == "УКБ-3"
    assert _outcome().competence_code is None


def test_competence_code_is_kept_in_json_output() -> None:
    dumped = _outcome(competence_codes=["УК-1", "УК-2"]).model_dump()
    assert dumped["competence_code"] == "УК-1" and dumped["competence_codes"] == ["УК-1", "УК-2"]


def test_legacy_competence_code_input_is_accepted() -> None:
    assert _outcome(competence_code="УК-1").competence_codes == ["УК-1"]
    assert _outcome(competence_code=None).competence_codes == []
    # если пришли оба поля, список главнее
    both = _outcome(competence_code="УК-1", competence_codes=["ОПК-1", "ОПК-2"])
    assert both.competence_codes == ["ОПК-1", "ОПК-2"]


def test_legacy_json_without_competence_codes_is_readable() -> None:
    legacy = (
        '{"file": "x.md", "learning_outcomes": [{"kind": "знает", "text": "основы", '
        '"competence_code": "УК-1", "source": {"file": "x.md"}}]}'
    )
    (o,) = RPDExtraction.model_validate_json(legacy).learning_outcomes
    assert o.competence_codes == ["УК-1"] and o.competence_code == "УК-1"


# --- регрессии аудита регулярок (корпус из 75 документов) ------------------------------


def test_indicator_in_format_text_then_code_in_brackets() -> None:
    md = (
        HEAD
        + "| способен проектировать (ПКП-МО-1) | ПКП-МО-1.2. З-1. Знает основы. "
        + "| применяет сети (ПКП-МО-1.2)<br>обучает модели (Средний) (ПКП-МО-1.3) |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(i.code, i.text, i.level) for i in r.indicators] == [
        ("ПКП-МО-1.2", "применяет сети", None),
        ("ПКП-МО-1.3", "обучает модели", "Средний"),
    ]
    assert [i.competence_code for i in r.indicators] == ["ПКП-МО-1", "ПКП-МО-1"]


def test_prefix_indicator_with_code_in_brackets_is_not_read_as_suffix_format() -> None:
    md = HEAD + "| ОПК-1 Способен | знает:<br>основы | ОПК-1.1 Применяет методы (ОПК-1.2) |\n"
    first = extract_rpd(md, "x.md").indicators[0]
    assert (first.code, first.text) == ("ОПК-1.1", "Применяет методы (")  # а не наоборот


def test_code_key_ignores_leading_zeros() -> None:
    assert code_key("ПКП-УНК-03.01") == code_key("ПКП-УНК-3.1") == "ПКП-УНК-3.1"
    assert code_key("ПКП-УНК-03") == code_key("ПКП-УНК-3")
    assert code_key("ПКП-ППК-ОС3") == "ПКП-ППК-ОС3"
    assert code_key("ОПК-10.10") == "ОПК-10.10"  # нули внутри числа не трогаем
    assert code_key("ПКП-ППК-ОС3") != code_key("ПКП-16-ППК-ОС3")


def test_assessment_row_with_competence_code_instead_of_indicator() -> None:
    md = (
        HEAD
        + "| ПКП-УНК-3. Способен планировать | знает:<br>основы | ПКП-УНК-03.02. Планирует |\n\n"
        + ASSESS_HEAD
        + "| 1 | ПКП-УНК-03 Способен планировать | Отчёт; экзамен |\n"
        + "| 2 | ПКП-УНК-3 | План |\n"
        + "| 3 | ПКП-УНК-03.02 Планирует | Тест |\n"
    )
    r = extract_rpd(md, "x.md")
    got = [(a.indicator_code, a.competence_code, a.materials) for a in r.assessments]
    assert got == [
        (None, "ПКП-УНК-3", ["Отчёт", "экзамен"]),  # «03» и «3» — одна компетенция из 1.3
        (None, "ПКП-УНК-3", ["План"]),
        ("ПКП-УНК-03.02", None, ["Тест"]),
    ]


def test_trailing_comma_and_broken_bracket_before_code_are_cleaned() -> None:
    md = (
        HEAD
        + "| ОПК-1 Способен | знает:<br>основы,<br>методы<br>умеет:<br>"
        + "методы и приёмы отладки (, ОПК-1.1. У-1) | ОПК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(o.kind, o.code, o.text) for o in r.learning_outcomes] == [
        ("знает", None, "основы"),
        ("знает", None, "методы"),
        ("умеет", "ОПК-1.1. У-1", "методы и приёмы отладки"),
    ]


def test_competence_without_name_gives_warning() -> None:
    md = HEAD + "| ПКП-УНК-03 | умеет:<br>считать | ПКП-УНК-03.01 Демонстрирует |\n"
    r = extract_rpd(md, "x.md")
    assert r.competences[0].name is None
    assert any("ПКП-УНК-03" in w and "название" in w for w in r.warnings)


def test_kind_verb_is_dropped_from_text_only_when_it_matches_kind() -> None:
    md = (
        HEAD
        + "| ОПК-1 Способен | ОПК-1.1. З-1. Знает основы.<br>ОПК-1.1. У-1. Умеет считать.<br>"
        + "ОПК-1.1. В-1. Знает как владеть. | ОПК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(o.kind, o.text) for o in r.learning_outcomes] == [
        ("знает", "основы."),
        ("умеет", "считать."),
        ("владеет", "Знает как владеть."),  # глагол другого вида не трогаем
    ]


def test_verb_is_kept_when_nothing_meaningful_remains() -> None:
    md = HEAD + "| ОПК-1 Способен | ОПК-1.1. З-1. Знает да | ОПК-1.1 Применяет |\n"
    (o,) = extract_rpd(md, "x.md").learning_outcomes
    assert (o.kind, o.text) == ("знает", "Знает да")


def test_wrapped_item_after_comma_is_joined_when_line_endings_differ() -> None:
    md = (
        HEAD
        + "| УК-1 Способен | владеет:<br>методами формализации задач,<br>анализа данных;<br>"
        + "методологией моделирования. | УК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [o.text for o in r.learning_outcomes] == [
        "методами формализации задач, анализа данных",
        "методологией моделирования.",
    ]


def test_comma_separated_lines_stay_separate_when_all_end_with_comma() -> None:
    md = (
        HEAD
        + "| УК-1 Способен | умеет:<br>пользоваться средствами мониторинга,<br>"
        + "настраивать протоколы,<br>строить сети. | УК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [o.text for o in r.learning_outcomes] == [
        "пользоваться средствами мониторинга",
        "настраивать протоколы",
        "строить сети.",
    ]


def test_isolated_comma_line_is_joined_but_comma_run_is_a_list() -> None:
    """Реальный случай «Компьютерные сети»: две строки с запятой подряд, затем строки с «;»."""
    md = (
        HEAD
        + "| УК-1 Способен | умеет:<br>пользоваться средствами,<br>настраивать протоколы,<br>"
        + "анализировать работу сетей;<br>проектировать сети. | УК-1.1 Применяет |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [o.text for o in r.learning_outcomes] == [
        "пользоваться средствами",
        "настраивать протоколы",
        "анализировать работу сетей",
        "проектировать сети.",
    ]


def test_one_item_with_several_codes_in_brackets_gives_record_per_code() -> None:
    md = (
        HEAD
        + "| ОПК-1 Способен | знает:<br>методы отладки (ОПК-1.1. З-1, ОПК-1.2. З-2)<br>"
        + "основы (ОПК-1.1. З-3) | ОПК-1.1 Один<br>ОПК-1.2 Два |\n"
    )
    r = extract_rpd(md, "x.md")
    assert [(o.code, o.indicator_code, o.text) for o in r.learning_outcomes] == [
        ("ОПК-1.1. З-1", "ОПК-1.1", "методы отладки"),
        ("ОПК-1.2. З-2", "ОПК-1.2", "методы отладки"),
        ("ОПК-1.1. З-3", "ОПК-1.1", "основы"),
    ]
