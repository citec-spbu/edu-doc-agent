"""Сверка документов образовательной программы между собой (опорный документ — учебный план)."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from edu_doc_agent.extraction.characteristic import ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.codes import canonical_code, check_competence_codes
from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import CompetenceMatrix, MatrixRow
from edu_doc_agent.extraction.models import RPDExtraction


def check_program_consistency(plan: CurriculumPlan, char: ProgramCharacteristic) -> list[str]:
    """Реквизиты программы в плане и в характеристике; расхождения — повод для проверки."""
    pairs = {
        "направление": (
            plan.direction,
            f"{char.direction_code or ''} {char.direction or ''}".strip() or None,
        ),
        "уровень": (plan.level, char.level),
        "форма обучения": (plan.study_form, char.study_form),
        "языки обучения": (plan.languages, char.languages),
        "срок обучения": (plan.duration, char.duration),
        "название программы": (
            plan.program,
            f"{char.program_name or ''} {char.program_name_en or ''}".strip() or None,
        ),
    }
    warnings = []
    for label, (in_plan, in_char) in pairs.items():
        if in_plan is None or in_char is None:
            continue
        if normalize(in_plan) != normalize(in_char):
            warnings.append(f"{label}: в плане «{in_plan}», в характеристике «{in_char}»")
    return warnings


# --- матрица компетенций против плана ---------------------------------------------------------

_PLAN_PROFILE = {"стандарт": "стандарт", "топ": "топ"}  # профиль матрицы → слово в профиле плана


@dataclass
class RowDifference:
    row: MatrixRow
    only_matrix: list[str]
    only_plan: list[str]
    plan_entry: PlanEntry


@dataclass
class MatrixPlanComparison:
    """Итог сопоставления строк матрицы с элементами учебного плана."""

    matched: int = 0
    equal: int = 0
    different: list[RowDifference] = field(default_factory=list)
    unmatched: list[MatrixRow] = field(default_factory=list)
    unmatched_reasons: dict[int, str] = field(default_factory=dict)
    columns_not_in_plan: list[str] = field(default_factory=list)
    plan_competences_not_in_columns: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.columns_not_in_plan:
            out.append(
                f"матрица: компетенции, которых нет в плане: {', '.join(self.columns_not_in_plan)}"
            )
        if self.plan_competences_not_in_columns:
            out.append(
                "план: компетенции без столбца в матрице: "
                + ", ".join(self.plan_competences_not_in_columns)
            )
        for row in self.unmatched:
            out.append(
                f"матрица, строка {row.source.line} («{row.discipline}», сем. {row.semester}, "
                f"{row.profile}): "
                + self.unmatched_reasons.get(row.source.line, "нет соответствия в плане")
            )
        for d in self.different:
            parts = []
            if d.only_matrix:
                parts.append(f"только в матрице: {', '.join(d.only_matrix)}")
            if d.only_plan:
                parts.append(f"только в плане: {', '.join(d.only_plan)}")
            row = d.row
            out.append(
                f"матрица, строка {row.source.line} («{row.discipline}», сем. {row.semester}, "
                f"{row.profile}): компетенции различаются ({'; '.join(parts)})"
            )
        return out


def _profile_matches(matrix_profile: str | None, plan_profile: str | None) -> bool:
    if matrix_profile in (None, "база"):
        return plan_profile is None
    word = _PLAN_PROFILE.get(matrix_profile, matrix_profile)
    return plan_profile is not None and word in plan_profile.lower()


def _names_match(matrix_name: str, plan_name: str) -> bool:
    a, b = normalize(matrix_name), normalize(plan_name)
    return bool(a and b) and (a.startswith(b) or b.startswith(a))


def compare_matrix_with_plan(
    matrix: CompetenceMatrix, plan: CurriculumPlan
) -> MatrixPlanComparison:
    """Строки матрицы против элементов плана: по семестру, названию (с точностью до усечения) и
    профилю; у найденной строки сравниваются наборы компетенций в каноническом виде."""
    result = MatrixPlanComparison()
    matrix_codes = {canonical_code(c.code) for c in matrix.columns}
    plan_codes = {canonical_code(c.code) for c in plan.competences}
    result.columns_not_in_plan = sorted(matrix_codes - plan_codes)
    result.plan_competences_not_in_columns = sorted(plan_codes - matrix_codes)

    for row in matrix.rows:
        mset = {canonical_code(c) for c in row.competences}
        candidates = [
            e
            for e in plan.entries
            if e.semester == row.semester and _names_match(row.discipline, e.name)
        ]
        exact = [e for e in candidates if normalize(row.discipline) == normalize(e.name)]
        candidates = exact or candidates
        pool = [e for e in candidates if _profile_matches(row.profile, e.profile)]
        if row.part:
            pool = [e for e in pool if e.part and normalize(e.part).startswith(normalize(row.part))]
        if not pool:
            result.unmatched.append(row)
            if candidates:
                locations = "; ".join(
                    f"{e.source.file}:{e.source.line} (профиль {e.profile}, часть {e.part})"
                    for e in candidates
                )
                result.unmatched_reasons[row.source.line] = (
                    f"несовместимые профиль/часть плана: {locations}"
                )
            continue
        if len(pool) > 1:
            result.unmatched.append(row)
            locations = ", ".join(f"{e.source.file}:{e.source.line}" for e in pool)
            result.unmatched_reasons[row.source.line] = (
                f"неоднозначное соответствие в плане ({locations})"
            )
            continue
        best = pool[0]
        pset = {canonical_code(c) for c in best.competences}
        result.matched += 1
        if pset == mset:
            result.equal += 1
        else:
            result.different.append(
                RowDifference(row, sorted(mset - pset), sorted(pset - mset), best)
            )
    return result


# --- все документы ----------------------------------------------------------------------------


def check_documents(
    plan: CurriculumPlan,
    *,
    characteristic: ProgramCharacteristic | None = None,
    ckmv: CkmvDocument | None = None,
    matrix: CompetenceMatrix | None = None,
    rpd: Iterable[RPDExtraction] = (),
) -> list[str]:
    """Сквозная сверка документов ОП относительно учебного плана (опорный документ)."""
    warnings = check_competence_codes(plan, ckmv, rpd, characteristic)
    if characteristic is not None:
        warnings += check_program_consistency(plan, characteristic)
    if matrix is not None:
        warnings += compare_matrix_with_plan(matrix, plan).warnings
    return warnings
