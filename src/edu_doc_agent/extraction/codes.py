"""Сопоставление кодов компетенций, индикаторов и результатов между документами ОП.

Один и тот же код пишут в двух «диалектах»:

- канонический (учебный план, характеристика, РПД): `ПКП-ППК-Р2`, `ПКП-МО-1`, `ПКП-УНК-03`;
- нумерованный (ЦКМВ, матрица компетенций): `ПКП-2-ППК-Р2`, `ПКП-19-МО-1`, `УНК-03`, где число
  после `ПКП-` — сквозной номер, а у универсальных надпрофессиональных нет префикса `ПКП-`.

К тому же в РПД встречаются нули (`ПКП-УНК-3` и `ПКП-УНК-03`). `canonical_code` приводит все
написания к одному виду; сами извлечённые коды при этом хранятся как в тексте (ADR-0002), а
`CodeIndex` показывает, как один код записан в каждом документе.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# Каноническая форма кода — общая для извлечения и нормативов (ADR-0005), без тяжёлых импортов
from edu_doc_agent.codes import canonical_code as canonical_code
from edu_doc_agent.codes import clean_dashes as clean_dashes
from edu_doc_agent.codes import code_dialect as code_dialect
from edu_doc_agent.extraction.characteristic import ProgramCharacteristic
from edu_doc_agent.extraction.ckmv import CkmvDocument
from edu_doc_agent.extraction.curriculum import CurriculumPlan
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.models import RPDExtraction
from edu_doc_agent.extraction.source import SourceRef

# --- индекс кодов по документам ---------------------------------------------------------------


@dataclass
class CodeEntry:
    canonical: str
    spellings: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    names: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    references: dict[tuple[str, str | None], list[SourceRef]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def sources(self) -> set[str]:
        return set(self.spellings)


@dataclass
class CodeIndex:
    """Какие документы содержат код и как он в них записан; названия — для сверки текстов."""

    entries: dict[str, CodeEntry] = field(default_factory=dict)

    def add(
        self, source: str, code: str, name: str | None = None, ref: SourceRef | None = None
    ) -> None:
        key = canonical_code(code)
        if not key:
            return
        entry = self.entries.setdefault(key, CodeEntry(key))
        if code not in entry.spellings[source]:
            entry.spellings[source].append(code)
        if name and name not in entry.names[source]:
            entry.names[source].append(name)
        if ref is not None and ref not in entry.references[source, name]:
            entry.references[source, name].append(ref)

    def codes_of(self, source: str) -> set[str]:
        return {k for k, e in self.entries.items() if source in e.sources}

    def unknown_in(self, source: str, reference: str) -> list[str]:
        """Коды документа `source`, которых нет в `reference` (например, ЦКМВ против плана)."""
        return sorted(self.codes_of(source) - self.codes_of(reference))

    def name_differences(self, reference: str) -> list["NameDifference"]:
        """Названия других документов, не совпавшие ни с одним названием из `reference`.

        Сравнение с точностью до регистра, тире и точки в конце; каждое различное название
        документа попадает в список один раз (а не по разу на каждый файл РПД).
        """
        result = []
        for key, entry in sorted(self.entries.items()):
            ref_names = entry.names.get(reference, [])
            if not ref_names:
                continue
            ref_norm = {_norm_name(n): n for n in ref_names}
            for source, names in sorted(entry.names.items()):
                if source == reference:
                    continue
                for name in names:
                    if _norm_name(name) in ref_norm:
                        continue
                    best = max(
                        ref_norm, key=lambda r: SequenceMatcher(None, _norm_name(name), r).ratio()
                    )
                    ratio = SequenceMatcher(None, _norm_name(name), best).ratio()
                    result.append(
                        NameDifference(key, source, name, reference, ref_norm[best], ratio)
                    )
        return result


def _norm_name(name: str) -> str:
    return normalize(name).rstrip(". ")


@dataclass(frozen=True)
class NameDifference:
    canonical: str
    source: str
    name: str
    reference: str
    reference_name: str
    similarity: float


def build_index(sources: dict[str, Iterable[tuple[str, str | None]]]) -> CodeIndex:
    """`{документ: [(код, название), …]}` → индекс."""
    index = CodeIndex()
    for source, items in sources.items():
        for code, name in items:
            index.add(source, code, name)
    return index


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def check_codes(index: CodeIndex, *, reference: str, subsets: Iterable[str]) -> list[str]:
    """Предупреждения: коды документов `subsets`, которых нет в `reference`, и отличия названий.

    Отличие названия — повод для ручной проверки, а не для исправления: это опечатка в одном из
    документов, сокращение («ИТ») или ошибка извлечения; степень сходства помогает отличить.
    """
    warnings = []
    for source in subsets:
        unknown = index.unknown_in(source, reference)
        if unknown:
            refs = [
                ref
                for code in unknown
                for (src, _), values in index.entries[code].references.items()
                if src == source
                for ref in values
            ]
            warnings.append(
                f"{source}: коды, которых нет в «{reference}»: {', '.join(unknown)}"
                + _locations(refs)
            )
    for d in index.name_differences(reference):
        warnings.append(
            f"{d.canonical}: название в «{d.source}» отличается от «{d.reference}» "
            f"(сходство {d.similarity:.2f}): «{_short(d.name)}» ≠ «{_short(d.reference_name)}»"
            + _locations(index.entries[d.canonical].references[d.source, d.name])
        )
    return warnings


def _locations(refs: list[SourceRef]) -> str:
    locations = dict.fromkeys(
        f"{r.file}, {r.section or 'раздел не указан'}, строка {r.line}" for r in refs
    )
    return " [" + "; ".join(locations) + "]" if locations else ""


def check_competence_codes(
    plan: CurriculumPlan,
    ckmv: CkmvDocument | None = None,
    rpd: Iterable[RPDExtraction] = (),
    characteristic: ProgramCharacteristic | None = None,
) -> list[str]:
    """Сверка компетенций ЦКМВ, характеристики и РПД с таблицей 1.1 плана (опорный документ)."""
    sources: dict[str, list[tuple[str, str | None]]] = {
        "план": [(c.code, c.name) for c in plan.competences]
    }
    subsets = []
    if characteristic is not None:
        sources["характеристика"] = [(c.code, c.name) for c in characteristic.competences]
        subsets.append("характеристика")
    if ckmv is not None:
        sources["ЦКМВ"] = [(c.code, c.name) for c in ckmv.competences]
        subsets.append("ЦКМВ")
    rpd_items = [c for r in rpd for c in r.competences]
    index = build_index(sources)
    if rpd_items:
        for c in rpd_items:
            index.add("РПД", c.code, c.name, c.source)
        subsets.append("РПД")
    return check_codes(index, reference="план", subsets=subsets)
