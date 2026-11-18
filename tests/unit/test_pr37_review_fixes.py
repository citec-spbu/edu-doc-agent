"""Контрпримеры ревью #37 и #30: только синтетические документы и ответы API."""

# ruff: noqa: E501 (дословные синтетические тексты)

import copy
import io
import json

import httpx
import pytest
from docx import Document
from openpyxl import Workbook

from edu_doc_agent.eval.characteristic_llm_gold import score_characteristic_llm
from edu_doc_agent.eval.llm_gold import Report, score_case
from edu_doc_agent.eval.matrix_gold import score_matrix
from edu_doc_agent.eval.plan_gold import score_plan
from edu_doc_agent.extraction.characteristic import extract_characteristic
from edu_doc_agent.extraction.characteristic_llm import extract_characteristic_llm
from edu_doc_agent.extraction.ckmv import extract_ckmv
from edu_doc_agent.extraction.codes import check_competence_codes
from edu_doc_agent.extraction.crossdoc import compare_matrix_with_plan
from edu_doc_agent.extraction.curriculum import (
    CurriculumPlan,
    PlanCompetence,
    PlanEntry,
    extract_plan,
)
from edu_doc_agent.extraction.llm_extract import _locate
from edu_doc_agent.extraction.matrix import CompetenceMatrix, MatrixRow, extract_matrix
from edu_doc_agent.extraction.md_utils import split_sections
from edu_doc_agent.extraction.models import ActiveForm, Competence, RPDExtraction, WorkloadRow
from edu_doc_agent.extraction.rpd import _check_active_forms, extract_rpd, extract_rpd_file
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.extraction.workload_table import check_workload
from edu_doc_agent.llm import FakeLLMClient, LLMError, OpenAICompatibleClient
from edu_doc_agent.parsing import parse_document
from tests.unit.test_characteristic_llm_gold import entity, gold, perfect
from tests.unit.test_plan_gold import GOLD, _gold_plan

REF = SourceRef(file="synthetic.docx", line=3)


@pytest.mark.parametrize("reverse", [False, True])
def test_same_name_kinds_match_independently_of_order(reverse):
    expected = gold()
    expected["annotation"]["roles"] *= 2
    expected["annotation"]["roles"][1] = dict(
        expected["annotation"]["roles"][1], kind="опережающая"
    )
    doc = perfect()
    doc.roles.append(entity(doc.roles[0].name, "опережающая"))
    if reverse:
        doc.roles.reverse()
    assert score_characteristic_llm(doc, expected).passed()
    doc.roles[0].kind = doc.roles[1].kind
    assert not score_characteristic_llm(doc, expected).passed()


def test_incomplete_position_fails_acceptance():
    expected = gold()
    expected["annotation"]["positions"] = [{"name": "директор колледжа"}]
    doc = perfect()
    doc.positions = [entity("колледжа")]
    report = score_characteristic_llm(doc, expected)
    assert (report.scores["positions"].tp, report.scores["positions"].fp) == (0, 1)
    assert not report.passed()
    doc.positions = [entity("директоров колледжей")]
    assert score_characteristic_llm(doc, expected).passed()


def test_title_code_cannot_consume_next_label():
    doc = extract_characteristic(
        "Код, наименование направления подготовки/ специальности\n\nУровень бакалавриат\n"
        "Код, наименование укрупненной группы\nШифр программы ТЕСТ.9999\n",
        "synthetic.md",
    )
    assert doc.direction_code is None and doc.group_code is None
    assert sum("некорректный код" in w for w in doc.warnings) == 2


def test_competence_punctuation_and_unparsed_rows():
    doc = extract_characteristic(
        "## 2.1. Перечень универсальных компетенций, предусмотренных ФГОС\n"
        "УК-1. Проверяет маршрут судна.\nУК-ошибка Неполный код.\nПКП-ошибка Неполный код.\n",
        "synthetic.md",
    )
    assert [(c.code, c.name) for c in doc.competences] == [("УК-1", "Проверяет маршрут судна")]
    assert any("строка 3: не разобрана компетенция" in w for w in doc.warnings)
    assert any("строка 4: не разобрана компетенция" in w for w in doc.warnings)


@pytest.mark.parametrize(
    "body", [b"{}", b'{"choices": []}', b"not-json", b'{"choices":[{"message":{"content":null}}]}']
)
def test_malformed_api_envelope_is_sanitized_llm_error(body):
    client = OpenAICompatibleClient(
        base_url="https://synthetic.example",
        api_key="test",
        model="test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)),
    )
    with pytest.raises(LLMError, match="некорректная структура ответа LLM"):
        client.complete_json(system="", user="", schema={})


def test_malformed_annotation_does_not_cancel_other_tasks():
    replies = [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"principles": [{"text": "Открытость", "quote": "Открытость"}]}
                            )
                        }
                    }
                ]
            },
        ),
    ]
    client = OpenAICompatibleClient(
        base_url="https://synthetic.example",
        api_key="test",
        model="test",
        transport=httpx.MockTransport(lambda _: replies.pop(0)),
    )
    doc = extract_characteristic_llm(
        "### Аннотация\nПрограмма.\n### Миссия\nОткрытость\n",
        "synthetic.md",
        client,
        ["annotation", "mission"],
    )
    assert doc.principles[0].name == "Открытость"
    assert doc.warnings == ["annotation: сбой вызова LLM, задача пропущена"]


def test_ckmv_docx_manual_line_break_is_continuation(tmp_path):
    doc = Document()
    table = doc.add_table(rows=2, cols=4)
    for c, value in zip(
        table.rows[0].cells, ["Роль", "Компетенции", "Индикаторы", "Результаты"], strict=True
    ):
        c.text = value
    values = [
        "Навигатор",
        "УК-1. Проверяет маршрут",
        "УК-1.1. Ищет данные (Базовый)",
        "УК-1.1 З-1. Знает карты\nи ограничения источников.\nУК-1.1 У-1. Умеет искать.",
    ]
    for c, value in zip(table.rows[1].cells, values, strict=True):
        c.text = value
    path = tmp_path / "ckmv.docx"
    doc.save(path)
    result = extract_ckmv(parse_document(path).to_markdown(), path.name)
    assert [o.text for o in result.outcomes] == [
        "Знает карты и ограничения источников.",
        "Умеет искать.",
    ]
    assert result.warnings == []


def test_grouped_rpd_warnings_keep_all_locations():
    plan = CurriculumPlan(
        file="plan", competences=[PlanCompetence(code="УК-1", name="Ищет карты", source=REF)]
    )
    rpds = [
        RPDExtraction(
            file=f"rpd{i}",
            competences=[
                Competence(
                    code="УК-1",
                    name="Читает карты",
                    source=SourceRef(file=f"rpd{i}", section="1.3", line=i + 40),
                )
            ],
        )
        for i in range(2)
    ]
    warnings = check_competence_codes(plan, rpd=rpds)
    assert len(warnings) == 1
    assert all(f"rpd{i}, 1.3, строка {i + 40}" in warnings[0] for i in range(2))


def _comparison(entries, competences=None, profile="база"):
    row = MatrixRow(
        number=1,
        semester=1,
        profile=profile,
        discipline="Анализ",
        source=REF,
        competences=competences or ["УК-2"],
    )
    return compare_matrix_with_plan(
        CompetenceMatrix(file="matrix", rows=[row]), CurriculumPlan(file="plan", entries=entries)
    )


def test_pairing_does_not_use_competences_or_fallback_to_wrong_profile():
    exact = PlanEntry(name="Анализ", semester=1, competences=["УК-1"], source=REF)
    prefix = PlanEntry(name="Анализ данных", semester=1, competences=["УК-2"], source=REF)
    result = _comparison([exact, prefix])
    assert result.matched == 1 and result.equal == 0
    assert result.different[0].plan_entry == exact
    assert _comparison([exact, prefix], ["УК-1"]).equal == 1
    exact.profile = "Профиль СТАНДАРТ"
    assert _comparison([exact]).matched == 0
    exact.profile = None
    assert _comparison([exact, copy.deepcopy(exact)]).unmatched_reasons


def test_plan_multiline_names_gia_and_missing_hours():
    from tests.unit.test_extraction_matrix import PLAN_MD

    text = PLAN_MD.replace(
        "Алгебра<br>Algebra", "Методы анализа<br>и обработки данных<br>Data<br>Analysis"
    )
    result = extract_plan(text)
    assert (result.entries[0].name, result.entries[0].name_en) == (
        "Методы анализа и обработки данных",
        "Data Analysis",
    )
    assert any("не найдены часы" in w for w in result.warnings)
    gia = extract_plan(
        "| Код блока | Трудоёмкость | Форма итоговой аттестации | Коды компетенций |\n"
        "| --- | --- | --- | --- |\n| Блок.3.гиа | 9 | Защита макета | |\n"
    )
    assert gia.final_attestation[0].name == "Защита макета"
    assert any("не указаны компетенции" in w for w in gia.warnings)


def test_matrix_dashes_and_incomplete_rows_stay_visible():
    wb = Workbook()
    wb.active.append(["№ строки", "Дисциплина", "УК–1", "УК-2"])
    wb.active.append([1, None, 0, 0])
    wb.active.append([2, "Навигация", 1, 1])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    matrix = extract_matrix(out)
    assert [c.code for c in matrix.columns] == ["УК-1", "УК-2"]
    assert matrix.columns[0].source.quote == "УК–1"
    assert len(matrix.rows) == 2 and matrix.rows[0].discipline == ""
    assert any("строка 2: не указано название" in w for w in matrix.warnings)
    expected = {
        "rows": [r.model_dump() for r in matrix.rows],
        "columns": ["УК-1", "УК-2"],
        "totals": {},
    }
    assert score_matrix(matrix, expected).total["cells"] == 4
    matrix.rows.pop(0)
    report = score_matrix(matrix, expected)
    assert report.total["cells"] == 4 and report.correct["cells"] == 2
    assert not report.passed()


@pytest.mark.parametrize("field,key", [("contact_hours", "contact"), ("self_study_hours", "self")])
def test_hours_labels_are_part_of_quality(field, key):
    plan = _gold_plan()
    hours = getattr(plan.entries[0], field)
    first, second, *rest = hours
    setattr(
        plan.entries[0],
        field,
        {second: hours[first], first: hours[second], **{k: hours[k] for k in rest}},
    )
    report = score_plan(plan, GOLD)
    assert report.accuracy(field) < 1 and report.hours_cell_accuracy < 1
    assert not report.passed(1)


@pytest.mark.parametrize("coded", [True, False])
def test_rpd_docx_vertical_merges_with_multiple_competences(tmp_path, coded):
    doc = Document()
    table = doc.add_table(rows=3, cols=3)
    for c, value in zip(
        table.rows[0].cells,
        ["Код компетенции", "Планируемые результаты обучения", "Индикаторы"],
        strict=True,
    ):
        c.text = value
    table.cell(1, 0).merge(table.cell(2, 0)).text = "УК-1. Ищет карты\nУК-2. Выбирает маршрут"
    table.cell(1, 1).merge(table.cell(2, 1)).text = (
        "УК-1.1 З-1. Знает способы поиска карт.\nУК-2.1 З-1. Знает критерии выбора маршрута."
        if coded
        else "знает:\nосновы поиска и оценки карт"
    )
    table.cell(1, 2).text = "УК-1.1. Ищет карты"
    table.cell(2, 2).text = "УК-2.1. Выбирает маршрут"
    path = tmp_path / "merged.docx"
    doc.save(path)
    parsed = parse_document(path)
    for source in (parsed, parsed.to_markdown()):
        result = extract_rpd(source, path.name)
        if coded:
            assert [o.competence_codes for o in result.learning_outcomes] == [["УК-1"], ["УК-2"]]
        else:
            assert [o.competence_codes for o in result.learning_outcomes] == [["УК-1", "УК-2"]]


def test_equal_unmerged_cells_are_not_a_merge(tmp_path):
    doc = Document()
    table = doc.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Код компетенции"
    table.cell(0, 1).text = "Планируемые результаты обучения"
    for i in (1, 2):
        table.cell(i, 0).text = f"УК-{i}. Ищет карты"
        table.cell(i, 1).text = "знает: основы поиска карт"
    path = tmp_path / "unmerged.docx"
    doc.save(path)
    result = extract_rpd_file(path)
    assert [o.competence_codes for o in result.learning_outcomes] == [["УК-1"], ["УК-2"]]


@pytest.mark.parametrize(
    "quote,line",
    [
        ("Совпадающее длинное начало цитаты заканчивается иначе", 5),
        ("Совпадающее длинное начало цитаты продолжается дальше", 3),
    ],
)
def test_full_quote_locates_duplicate_prefix_and_line_break(quote, line):
    text = "### Аннотация\n\nСовпадающее длинное начало цитаты\nпродолжается дальше\nСовпадающее длинное начало цитаты заканчивается иначе\n"
    assert _locate(quote, split_sections(text), "synthetic", False).line == line
    reply = {
        "roles": [],
        "positions": [
            {
                "name": "Должность",
                "quote": quote.replace("цитаты продолжается", "цитаты\nпродолжается"),
            }
        ],
        "tracks": [],
        "projects": [],
        "legal_acts": [],
    }
    result = extract_characteristic_llm(
        text, "synthetic", FakeLLMClient(replies=[json.dumps(reply)]), ["annotation"]
    )
    assert result.positions[0].source.line == line


def test_full_prerequisite_names_and_fixed_conjunction():
    text = "## 1.2. Требования к подготовленности\nнавыки разработки простых программ на распространённых языках программирования (С, С++, Java); навыки работы в одной и более операционных системах на уровне пользователя."
    names = [p.name for p in extract_rpd(text, "synthetic").prerequisites]
    assert names == [
        "разработки простых программ на распространённых языках программирования (С, С++, Java)",
        "работы в одной и более операционных системах на уровне пользователя",
    ]


def test_negated_grading_and_hallucinated_quote_fragment_fail():
    rule = {
        "condition": "не менее 60 баллов",
        "effect": "допускается к экзамену",
        "quote": "Набрал не менее 60 баллов",
    }
    bad = dict(rule, condition="менее 60 баллов", effect="не допускается к экзамену")
    report = Report()
    report.by_task["grading_rules"] = score_case(
        "grading_rules", {"rules": [bad]}, {"rules": [rule]}
    )
    assert report.by_task["grading_rules"].attribute_accuracy == 0 and not report.passed()
    extra = {
        "condition": "выполнены все работы",
        "effect": "автоматически отлично",
        "quote": "60 баллов",
    }
    assert score_case("grading_rules", {"rules": [rule, extra]}, {"rules": [rule]}).fp == 1


@pytest.mark.parametrize(
    "hours,trajectories,stated,warn",
    [([18, 18], [None, None], 36, False), ([14, 20], ["А", "Б"], 34, True)],
)
def test_active_hours_sum_semesters_separately_from_trajectories(hours, trajectories, stated, warn):
    result = RPDExtraction(
        file="synthetic",
        active_forms=[ActiveForm(text="всего", hours=stated, is_total=True, source=REF)],
        workload_table=[
            WorkloadRow(semester=i + 1, trajectory=t, active_hours=h, source=REF)
            for i, (t, h) in enumerate(zip(trajectories, hours, strict=True))
        ],
    )
    _check_active_forms(result)
    assert bool(result.warnings) == warn


def test_successful_empty_llm_prerequisites_are_compared():
    result = extract_rpd(
        "## 1.2. Требования к подготовленности\nНужно знание по дисциплине «Алгебра».",
        "synthetic",
        FakeLLMClient(replies=['{"items": []}']),
    )
    assert result.llm.completed_tasks == ["prerequisites"]
    assert any("пререквизиты расходятся" in w for w in result.warnings)


def test_self_study_hour_totals_do_not_mix_with_contact():
    result = RPDExtraction(
        file="synthetic",
        workload_table=[
            WorkloadRow(
                semester=1,
                contact_hours={"текущий контроль": 10},
                self_study_hours={"текущий контроль": 64},
                source=REF,
            ),
            WorkloadRow(
                is_total=True,
                contact_hours={"текущий контроль": 10},
                self_study_hours={"текущий контроль": 28},
                source=REF,
            ),
        ],
    )
    check_workload(result)
    assert len(result.warnings) == 1 and "самостоятельная работа" in result.warnings[0]
    assert "64 ч" in result.warnings[0] and "28 ч" in result.warnings[0]


def test_rpd_scalar_sources_cover_title_language_and_goals():
    text = "### Рабочая программа\nдисциплины\nМорские системы\nMarine Systems\nЯзык обучения\nрусский\nРегистрационный номер: 009999\nТрудоёмкость: 3 зачётные единицы\n## 1.1. Цели\nИзучить карты.\n"
    result = extract_rpd(text, "synthetic")
    for name in (
        "doc_kind",
        "title",
        "title_en",
        "language",
        "goals",
        "credits",
        "registration_number",
        "discipline_code",
    ):
        assert getattr(result, name) is not None, name
        ref = result.sources[name]
        assert ref.file == "synthetic" and ref.quote and ref.line
        assert ref.quote in text


def test_multiline_title_number_provenance():
    text = "Регистрационный номер:\n009999\nТрудоёмкость:\n3\n"
    result = extract_rpd(text, "synthetic")
    assert result.registration_number == "009999" and result.credits == 3
    assert result.sources["registration_number"].line == 1
    assert result.sources["registration_number"].quote == "Регистрационный номер:\n009999"
    assert result.sources["credits"].line == 3
    assert result.sources["credits"].quote == "Трудоёмкость:\n3"


def test_zero_credits_and_explicit_zero_hours_are_distinct_from_missing_hours():
    from edu_doc_agent.extraction.curriculum import check_plan

    plan = CurriculumPlan(
        file="synthetic",
        entries=[
            PlanEntry(name="Вне зачётных единиц", credits=0, source=REF),
            PlanEntry(name="Нулевые часы", credits=1, contact_hours={"лекции": 0}, source=REF),
        ],
    )
    check_plan(plan)
    assert len(plan.warnings) == 1
    assert "0 ч не равно" in plan.warnings[0]
    assert "не найдены часы" not in plan.warnings[0]
