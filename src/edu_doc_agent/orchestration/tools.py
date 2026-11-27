"""Инструменты агентов для чтения данных сессии (ADR-0003, «Инструменты»).

Инструменты обращаются к хранилищу сессии и каталогу правил и только читают данные.
`course_id` — ID документа сессии (РПД, РПП и другие документы).
"""

import difflib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from edu_doc_agent.extraction.curriculum import CurriculumPlan, PlanEntry
from edu_doc_agent.extraction.llm_extract import normalize
from edu_doc_agent.extraction.matrix import CompetenceMatrix, MatrixRow
from edu_doc_agent.extraction.md_utils import MdSection, split_sections
from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.orchestration.edits import edit_text
from edu_doc_agent.orchestration.stores import DOC_TYPE_TITLES, SessionStores, StoredDocument
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.findings import Finding

_WORD_RE = re.compile(r"\w{3,}")
MAX_SECTION = 8000
MAX_PLAN_ENTRIES = 80
# символов данных в ответе get_course: больше агент получил бы обрезанный JSON без пометки,
# каких элементов списка не хватает (предел всего результата — MAX_TOOL_RESULT в agent.py)
MAX_COURSE_DATA = 9_000


_PLAN_ROW = ("name", "block", "credits", "semester", "profile", "control", "competences")
_MATRIX_ROW = ("discipline", "semester", "credits", "profile", "competences")


def _row(item: PlanEntry | MatrixRow, fields: tuple[str, ...]) -> dict[str, Any]:
    """Строка плана или матрицы для модели: только заполненные поля."""
    row = {f: getattr(item, f) for f in fields}
    return {k: v for k, v in row.items() if v not in (None, [])}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _page_lists(
    data: dict[str, Any], match: str | None, offset: int
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Списки в данных: фильтр match, затем элементы с offset, пока данные не превысят
    MAX_COURSE_DATA. Для отфильтрованных и неполных списков — total, returned, next_offset."""
    needle = normalize(match) if match else None
    scalars = {k: v for k, v in data.items() if not isinstance(v, list)}
    budget = MAX_COURSE_DATA - len(_json(scalars))
    out: dict[str, Any] = {}
    lists: dict[str, dict[str, Any]] = {}
    for name, value in data.items():
        if not isinstance(value, list):
            out[name] = value
            continue
        items = [i for i in value if needle is None or needle in normalize(_json(i))]
        page: list[Any] = []
        used = 0
        for item in items[offset:]:
            size = len(_json(item)) + 1
            if page and used + size > budget:
                break
            page.append(item)
            used += size
        budget -= used
        out[name] = page
        if needle is not None or offset or len(page) < len(items):
            end = offset + len(page)
            lists[name] = {
                "total": len(items),
                "returned": len(page),
                "next_offset": end if end < len(items) else None,
            }
    return out, lists


_SERVICE_FIELDS = {"file", "sources", "warnings", "llm", "total_sources"}


def data_fields(d: StoredDocument) -> list[str]:
    """Заполненные поля данных документа для get_course(..., fields); у списков — подполя
    элементов в скобках: «competences[code, name, role]»."""
    out = []
    for name in type(d.data).model_fields:
        value = getattr(d.data, name)
        if name in _SERVICE_FIELDS or value in (None, [], {}, ""):
            continue
        if subs := _subfields(value):
            name = f"{name}[{', '.join(subs)}]"
        out.append(name)
    return out


def _subfields(value: Any) -> list[str]:
    """Подполя элементов списка моделей (или модели), кроме источника значения."""
    item = value[0] if isinstance(value, list) and value else value
    if not isinstance(item, BaseModel):
        return []
    return [k for k in type(item).model_fields if k not in ("source", "sources")]


def _project(name: str, raw: Any, value: Any, subs: list[str]) -> Any:
    """Только подполя subs у элементов списка (или у объекта) — для тяжёлых элементов."""
    if not subs or value is None:
        return value
    allowed = _subfields(raw)
    if unknown := [s for s in subs if s not in allowed]:
        raise ValueError(f"нет подполей {name}.{unknown}; есть: {', '.join(allowed)}")
    items = value if isinstance(value, list) else [value]
    projected = [{k: i[k] for k in subs if k in i} for i in items if isinstance(i, dict)]
    return projected if isinstance(value, list) else projected[0]


def _select(model: BaseModel, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Поля `fields` данных документа; «поле.подполе» — только эти подполя элементов."""
    wanted: dict[str, list[str]] = {}
    for f in fields:
        top, _, sub = f.partition(".")
        wanted.setdefault(top, []).extend([sub] if sub else [])
    known = type(model).model_fields
    if unknown := [f for f in wanted if f not in known]:
        raise ValueError(f"нет полей {unknown}; есть: {', '.join(known)}")
    return {f: _project(f, getattr(model, f), data.get(f), subs) for f, subs in wanted.items()}


def _key(text: str) -> str:
    """Для сравнения кодов блоков: «Блок 2», «Блок.2..прки» → «блок2», «блок2прки»."""
    return re.sub(r"[\s.]", "", normalize(text))


def document_title(d: StoredDocument) -> str:
    title = getattr(d.data, "title", None) or getattr(d.data, "program", None)
    return str(title) if title else d.filename


def plain_data(value: Any) -> Any:
    """Данные документа для модели: источник значения коротко, «раздел, строка N»."""
    if isinstance(value, SourceRef):
        where = ", ".join(
            x for x in (value.section, f"строка {value.line}" if value.line else None) if x
        )
        return where or value.file
    if isinstance(value, BaseModel):
        return {
            k: plain_data(v)
            for k, v in value
            if v not in (None, [], {}, "") and k not in ("sources", "warnings")
        }
    if isinstance(value, list | tuple):
        return [plain_data(v) for v in value]
    if isinstance(value, dict):
        return {k: plain_data(v) for k, v in value.items()}
    return value


def finding_brief(f: Finding) -> dict[str, Any]:
    brief = {
        "id": f.id,
        "rule_id": f.rule_id,
        "rule": f.rule_title,
        "course": f.course or "программа",
        "course_id": f.document_id,
        "status": f.status,
        "severity": f.severity,
        "fixable": f.fixable,
        "message": f.message[:500],
    }
    if f.needs_human:
        brief["needs_human"] = True
    return brief


def sections(document: StoredDocument) -> list[MdSection]:
    return split_sections(document.text) if document.text else []


@dataclass
class SessionTools:
    """Инструменты одного запуска: сессия, каталог правил и данные состояния графа."""

    stores: SessionStores
    session_id: str
    rules: list[Rule]
    finding_ids: list[str]
    change_ids: list[str]
    direction_code: str | None = None

    async def _documents(self) -> list[StoredDocument]:
        return await self.stores.list_documents(self.session_id)

    async def _latest[T: BaseModel](self, kind: type[T], missing: str) -> tuple[str, T]:
        """ID и данные последнего загруженного документа этого вида."""
        found = [d for d in await self._documents() if isinstance(d.data, kind)]
        if not found:
            raise ValueError(missing)
        return found[-1].document_id, found[-1].data

    async def _document(self, course_id: str) -> StoredDocument:
        document = await self.stores.get_document(self.session_id, course_id)
        if document is None:
            raise ValueError(f"нет документа {course_id}; найдите ID через find_course")
        return document

    async def get_current_state(self) -> dict[str, Any]:
        """Документы сессии, итог последней проверки, число открытых замечаний и
        подготовленных изменений."""
        documents = await self._documents()
        findings = await self.stores.get_findings(self.session_id, self.finding_ids)
        return {
            "documents": [
                {
                    "course_id": d.document_id,
                    "type": DOC_TYPE_TITLES.get(d.doc_type, d.doc_type),
                    "file": d.filename,
                    "title": document_title(d),
                }
                for d in documents
            ],
            "direction_code": self.direction_code,
            "open_findings": dict(Counter(f"{f.status}/{f.severity}" for f in findings)),
            "prepared_changes": len(self.change_ids),
            "last_change": await self._last_change(documents),
        }

    async def _last_change(self, documents: list[StoredDocument]) -> dict[str, Any] | None:
        """Последнее изменение сессии, кроме заменённых черновиков."""
        for change_id in reversed(self.change_ids):
            change = await self.stores.get_change(self.session_id, change_id)
            if change is None or change.status == "superseded":
                continue
            titles = {d.document_id: document_title(d) for d in documents}
            return {
                "status": change.status,
                "edits": [edit_text(e, titles) for e in change.edits],
                "finding_ids": change.finding_ids,
                "skipped_finding_ids": change.skipped_finding_ids,
            }
        return None

    async def get_findings(
        self,
        status: str | None = None,
        severity: str | None = None,
        fixable: bool | None = None,
        rule_id: str | None = None,
        course_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Замечания сессии, подходящие под фильтры: status (violation, not_checked),
        severity (error, warning), fixable, rule_id, course_id (ID документа)."""
        findings = await self.stores.get_findings(self.session_id, self.finding_ids)
        return [
            finding_brief(f)
            for f in findings
            if (status is None or f.status == status)
            and (severity is None or f.severity == severity)
            and (fixable is None or f.fixable == fixable)
            and (rule_id is None or f.rule_id == rule_id)
            and (course_id is None or f.document_id == course_id)
        ]

    async def get_rule(self, rule_id: str) -> dict[str, Any]:
        """Правило проверки системы по rule_id из замечания: что проверяется и почему, с
        цитатой пункта норматива. Не источник требований: требования ФГОС и норм СПбГУ ищи
        поиском по нормативам."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule.model_dump(mode="json", exclude_none=True)
        raise ValueError(f"нет правила {rule_id}")

    async def find_course(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Документы (дисциплины, практики и документы программы) по неточному названию со
        степенью совпадения от 0 до 1."""
        q = normalize(query)
        scored = []
        for d in await self._documents():
            title = normalize(document_title(d))
            score = difflib.SequenceMatcher(None, q, title).ratio()
            if q and (q in title or title in q):
                score = max(score, 0.9)
            scored.append((round(score, 2), d))
        scored.sort(key=lambda x: -x[0])
        return [
            {
                "course_id": d.document_id,
                "title": document_title(d),
                "type": DOC_TYPE_TITLES.get(d.doc_type, d.doc_type),
                "file": d.filename,
                "fields": data_fields(d),
                "score": s,
            }
            for s, d in scored[:limit]
        ]

    async def get_plan_summary(self) -> dict[str, Any]:
        """Сводка учебного плана: з.е. блоков по профилям (общие строки входят в каждый
        профиль), ГИА, число строк плана. Строки плана целиком — get_course(..., ["entries"])."""
        plan_id, plan = await self._latest(CurriculumPlan, "учебный план не загружен")
        profiles = sorted({e.profile for e in plan.entries if e.profile}) or [None]
        blocks: dict[str, dict[str, float]] = {}
        for e in plan.entries:
            if not e.block or e.block == "-":
                continue
            for p in profiles if e.profile is None else [e.profile]:
                by_profile = blocks.setdefault(e.block, {})
                key = p or "план"
                by_profile[key] = by_profile.get(key, 0) + (e.credits or 0)
        return {
            "course_id": plan_id,
            "program": plan.program,
            "direction": plan.direction,
            "blocks_credits_by_profile": blocks,
            "note": "строки по выбору и альтернативы не исключены: сумма блока 1 завышена",
            "final_attestation": [
                {"name": a.name, "credits": a.credits} for a in plan.final_attestation
            ],
            "entries": len(plan.entries),
        }

    async def get_plan_entries(
        self, block: str | None = None, name: str | None = None, profile: str | None = None
    ) -> dict[str, Any]:
        """Строки учебного плана: дисциплины, практики, ГИА с з.е., семестром, профилем и
        компетенциями. Фильтры — часть кода блока («Блок.1» — дисциплины, «Блок.2» —
        практики, «Блок.3» — ГИА), часть названия, часть названия профиля."""
        plan_id, plan = await self._latest(CurriculumPlan, "учебный план не загружен")
        rows = [
            e
            for e in plan.entries
            if (block is None or _key(block) in _key(e.block or ""))
            and (name is None or normalize(name) in normalize(e.name))
            and (profile is None or normalize(profile) in normalize(e.profile or ""))
        ]
        return {
            "course_id": plan_id,
            "found": len(rows),
            "credits_total": sum(e.credits or 0 for e in rows),
            "entries": [_row(e, _PLAN_ROW) for e in rows[:MAX_PLAN_ENTRIES]],
            **(
                {"note": f"показаны первые {MAX_PLAN_ENTRIES}: уточните фильтр"}
                if len(rows) > MAX_PLAN_ENTRIES
                else {}
            ),
        }

    async def get_matrix_rows(
        self,
        discipline: str | None = None,
        competence: str | None = None,
        profile: str | None = None,
    ) -> dict[str, Any]:
        """Строки матрицы компетенций: дисциплина, семестр, з.е., профиль и формируемые
        компетенции. Фильтры — часть названия дисциплины, код компетенции (полностью, точно),
        часть названия профиля. Без фильтров — число дисциплин по каждой компетенции."""
        matrix_id, matrix = await self._latest(CompetenceMatrix, "матрица компетенций не загружена")
        code = competence and normalize(competence).replace(" ", "")
        rows = [
            r
            for r in matrix.rows
            if (discipline is None or normalize(discipline) in normalize(r.discipline))
            and (code is None or code in {normalize(c).replace(" ", "") for c in r.competences})
            and (profile is None or normalize(profile) in normalize(r.profile or ""))
        ]
        result: dict[str, Any] = {"course_id": matrix_id, "found": len(rows)}
        if discipline is None and competence is None and profile is None:
            result["disciplines_per_competence"] = matrix.totals
            return result
        result["rows"] = [_row(r, _MATRIX_ROW) for r in rows[:MAX_PLAN_ENTRIES]]
        return result

    async def get_course(
        self,
        course_id: str,
        fields: list[str] | None = None,
        match: str | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Данные документа с источником каждого значения; fields — только эти поля, у
        элементов списка — только эти подполя: ["список.подполе"]. Списки в данных
        фильтруются: match — элементы, где встречается этот текст; длинные списки приходят
        частями: lists — сколько всего, сколько выдано и offset следующей части."""
        document = await self._document(course_id)
        data = plain_data(document.data)
        if fields:
            data = _select(document.data, data, fields)
        data, lists = _page_lists(data, match, offset)
        result = {"course_id": course_id, "file": document.filename, "data": data}
        if lists:
            result["lists"] = lists
        return result

    async def get_field_values(
        self, field: str, course_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Значения поля во всех документах (или в course_ids) с источниками."""
        out = []
        for d in await self._documents():
            if course_ids and d.document_id not in course_ids:
                continue
            if field in type(d.data).model_fields:
                out.append(
                    {
                        "course_id": d.document_id,
                        "title": document_title(d),
                        "value": plain_data(getattr(d.data, field)),
                    }
                )
        return out

    async def search_document(
        self, query: str, course_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Фрагменты текста документов с номером раздела: по словам запроса."""
        words = set(_WORD_RE.findall(normalize(query)))
        if not words:
            raise ValueError("пустой запрос")
        hits = []
        documents = [await self._document(course_id)] if course_id else await self._documents()
        for d in documents:
            for s in sections(d):
                text = normalize(s.full_text or s.text)
                score = sum(1 for w in words if w in text)
                if score:
                    hits.append((score, d, s))
        hits.sort(key=lambda x: -x[0])
        return [
            {
                "course_id": d.document_id,
                "title": document_title(d),
                "section_id": s.number or s.heading,
                "heading": s.heading,
                "text": (s.full_text or s.text)[:600],
            }
            for _, d, s in hits[:limit]
        ]

    async def get_document_outline(self, course_id: str) -> list[dict[str, Any]]:
        """Разделы документа: номер, заголовок, уровень."""
        document = await self._document(course_id)
        return [
            {"section_id": s.number or s.heading, "heading": s.heading, "level": s.level}
            for s in sections(document)
        ]

    async def read_section(self, course_id: str, section_id: str) -> dict[str, Any]:
        """Текст раздела по номеру, например «3.1.4», или по заголовку."""
        document = await self._document(course_id)
        for s in sections(document):
            if section_id in (s.number, s.heading) or (
                s.number is None and normalize(section_id) in normalize(s.heading)
            ):
                text = s.full_text or s.text
                return {"section_id": section_id, "heading": s.heading, "text": text[:MAX_SECTION]}
        raise ValueError(f"нет раздела {section_id}; список — get_document_outline")

    def as_tools(self, names: list[str]) -> list[BaseTool]:
        return [function_tool(getattr(self, name)) for name in names]


def function_tool(function: Callable[..., Any], name: str | None = None) -> BaseTool:
    """Async-функция с docstring → инструмент LangChain."""
    return StructuredTool.from_function(
        coroutine=function,
        name=name or function.__name__,
        description=(function.__doc__ or name or function.__name__).strip(),
    )
