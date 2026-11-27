"""Проверки состава документов и компетенций программы: нормы СПбГУ, перечни и формулировки
компетенций ФГОС, покрытие, матрица, повторы и дубли документов."""

import pytest
from rpd_samples import rpd

from edu_doc_agent.extraction.characteristic import CharCompetence, ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvCompetence, CkmvDocument
from edu_doc_agent.extraction.curriculum import (
    CurriculumPlan,
    FinalAttestation,
    PlanCompetence,
    PlanEntry,
)
from edu_doc_agent.extraction.matrix import CompetenceMatrix, MatrixRow
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.validation.checks.competences import (
    competence_wording_matches,
    program_codes_present,
)
from edu_doc_agent.validation.checks.coverage import (
    competences_covered,
    matrix_matches_plan,
    no_duplicate_codes,
    single_document_per_kind,
)
from edu_doc_agent.validation.checks.structure import (
    characteristic_fields_present,
    plan_blocks_present,
    sections_present,
    study_form_allowed,
)
from edu_doc_agent.validation.units import CourseDoc, CourseDocuments, ProgramDocuments

SRC = SourceRef(file="plan.md", line=1)
OPK2 = (
    "Способен применять компьютерные/суперкомпьютерные методы, современное программное "
    "обеспечение, в том числе отечественного происхождения, для решения задач"
)


def entry(name: str, competences: list[str], **kw) -> PlanEntry:
    return PlanEntry(
        name=name, competences=competences, block=kw.pop("block", "Блок.1.дисц"), source=SRC, **kw
    )


def plan(*entries: PlanEntry, competences: dict[str, str] | None = None, **kw) -> CurriculumPlan:
    return CurriculumPlan(
        file="plan.md",
        entries=list(entries),
        competences=[
            PlanCompetence(code=c, name=n, source=SRC) for c, n in (competences or {}).items()
        ],
        **kw,
    )


def characteristic(competences: dict[str, str] | None = None, **kw) -> ProgramCharacteristic:
    items = [
        CharCompetence(code=c, name=n, group="", source=SRC) for c, n in (competences or {}).items()
    ]
    return ProgramCharacteristic(file="char.md", competences=items, **kw)


# --- нормы СПбГУ: разделы, сведения характеристики, форма обучения, блоки -------------------

RPD_TEXT = """\
## 1.1. Цели и задачи учебных занятий
## 1.3. Перечень результатов обучения
## 2.2. Структура и содержание учебных занятий
## 3.1. Методическое обеспечение
### 3.1.4 Методические материалы (оценочные средства)
## 3.2. Кадровое обеспечение
"""


def test_sections_present() -> None:
    unit = CourseDocuments(rpd(), None, (), RPD_TEXT)
    sections = {
        "цели и задачи": "цел.*задач",
        "оценочные средства": "оценочн",
        "материально-техническое обеспечение": "материально-техническ",
    }

    result = sections_present(unit, sections=sections)

    assert result.status == "violation"
    assert result.message == "нет разделов: материально-техническое обеспечение"
    assert sections_present(CourseDocuments(rpd(), None, ()), sections=sections).status == (
        "not_checked"
    )


def test_characteristic_fields_present() -> None:
    fields = {"languages": "язык обучения", "duration": "срок обучения"}
    program = ProgramDocuments(characteristic=characteristic(languages="русский"))

    result = characteristic_fields_present(program, fields=fields)

    assert result.message == "в общей характеристике не указано: срок обучения"
    assert characteristic_fields_present(ProgramDocuments(), fields=fields).status == (
        "not_checked"
    )


@pytest.mark.parametrize(
    ("form", "status"),
    [("очная", "passed"), ("Очная, очно-заочная", "passed"), ("заочная", "violation")],
)
def test_study_form_allowed(form: str, status: str) -> None:
    program = ProgramDocuments(characteristic=characteristic(study_form=form))

    assert study_form_allowed(program, forms=["очная", "очно-заочная"]).status == status


def test_plan_blocks_present() -> None:
    gia = FinalAttestation(block="Блок.3.гиа", credits=9, name="ВКР", source=SRC)
    blocks = {"Блок 1": "Блок.1", "Блок 2": "Блок.2", "Блок 3": "Блок.3"}
    program = ProgramDocuments(plan=plan(entry("Физика", []), final_attestation=[gia]))

    result = plan_blocks_present(program, blocks=blocks)

    assert result.message == "в учебном плане нет блоков: Блок 2"  # ГИА — из раздела 3


# --- компетенции программы против ФГОС и стандарта СПбГУ ------------------------------------


def test_program_codes_present() -> None:
    program = ProgramDocuments(
        plan=plan(competences={"УК-1": "", "ОПК-1": "", "ПКП-МО-1": ""}),
        characteristic=characteristic({"УК-1": "", "УК-2": ""}),
    )

    result = program_codes_present(program, codes=["УК-1", "УК-2"], prefixes=["ПКП"])

    assert result.message == ("учебный план: нет УК-2; характеристика: нет компетенций групп ПКП")


def test_competence_wording_matches() -> None:
    program = ProgramDocuments(
        plan=plan(competences={"ОПК-2": OPK2.replace("в том числе", "в т. ч.")}),
        characteristic=characteristic({"ОПК-2": OPK2.replace("происхождения", "производства")}),
    )

    result = competence_wording_matches(program, wordings={"ОПК-2": OPK2})

    # «в т. ч.» — то же, что «в том числе»; отличие показано словами, а не обрезанным текстом
    assert result.message == "- характеристика, ОПК-2: «происхождения» → «производства»"


# --- покрытие, матрица, повторы кодов, документы одного вида ---------------------------------


def test_competences_covered() -> None:
    gia = FinalAttestation(block="Блок.3", credits=9, name="ВКР", competences=["УК-2"], source=SRC)
    curriculum = plan(
        entry("Физика", ["УК-1"]),
        competences={"УК-1": "", "УК-2": "", "ОПК-4": ""},
        final_attestation=[gia],
    )

    result = competences_covered(ProgramDocuments(plan=curriculum))

    assert result.message == "не формируются ни одной строкой учебного плана: ОПК-4"


def test_matrix_matches_plan_in_canonical_codes() -> None:
    curriculum = plan(
        entry("Физика", ["УК-1", "ПКП-УНК-03"], semester=3, profile="Профиль СТАНДАРТ"),
        entry("Машинное обучение (осн курс)", ["ОПК-2", "ПКП-МО-1", "ПКП-ППК-Р2"], semester=5),
    )

    def matrix_row(n: int, name: str, semester: int, competences: list[str]) -> MatrixRow:
        return MatrixRow(
            number=n,
            discipline=name,
            semester=semester,
            profile="стандарт",
            competences=competences,
            source=SRC,
        )

    matrix = CompetenceMatrix(
        file="m.xlsx",
        rows=[
            matrix_row(1, "Физика", 3, ["УК-1", "УНК-03"]),  # тот же код в нумерованной форме
            matrix_row(2, "Машинное обучение", 5, ["ОПК-2", "ПКП-19-МО-1"]),
            matrix_row(3, "Химия", 1, ["УК-1"]),  # нет в плане под этим названием: не сверена
            matrix_row(4, "Государственная итоговая аттестация (защита ВКР)", 8, ["УК-1", "УК-2"]),
        ],
    )
    curriculum.final_attestation.append(
        FinalAttestation(block="Блок.3", credits=9, name="ВКР", competences=["УК-1"], source=SRC)
    )

    result = matrix_matches_plan(ProgramDocuments(plan=curriculum, matrix=matrix))

    assert result.status == "violation"
    assert result.message == (
        "- «Машинное обучение» (семестр 5): только в плане: ПКП-ППК-Р2\n"
        "- «Государственная итоговая аттестация (защита ВКР)» (семестр 8): только в матрице: УК-2\n"
        "не сверены строки матрицы, которых нет в учебном плане под тем же названием (1 из 4): "
        "«Химия» (семестр 1)"
    )

    # без расхождений, но не все строки сверены: правило не проверено, а не выполнено
    matrix.rows = [matrix.rows[0], matrix.rows[2]]
    partial = matrix_matches_plan(ProgramDocuments(plan=curriculum, matrix=matrix))
    assert partial.status == "not_checked"
    assert "(1 из 2): «Химия» (семестр 1)" in partial.message


def test_no_duplicate_codes() -> None:
    ckmv = CkmvDocument(
        file="ckmv.docx",
        competences=[
            CkmvCompetence(code=c, name="", source=SRC)
            for c in ("ПКП-9-ППК-ДА2", "ПКП-ППК-ДА2", "ПКП-МО-1")
        ],
    )
    program = ProgramDocuments(
        ckmv=ckmv, courses=(CourseDoc("d1", rpd(competences=["УК-1", "УК-1"]), "a.md"),)
    )

    result = no_duplicate_codes(program)

    assert result.message == "- ckmv.docx: повторяются ПКП-ППК-ДА2\n- a.md: повторяются УК-1"


def test_single_document_per_kind() -> None:
    program = ProgramDocuments(
        files_by_kind={"plan": ("plan.docx",), "characteristic": ("char.docx", "char_corr.docx")}
    )

    result = single_document_per_kind(program)

    assert result.message == (
        "загружено 2 общих характеристики: char.docx, char_corr.docx; "
        "проверяется последний — char_corr.docx"
    )
