"""Правки данных документов: применение к данным извлечения и текст для пользователя."""

import copy
import json
from dataclasses import replace
from typing import Any

from pydantic import ValidationError

from edu_doc_agent.orchestration.stores import Change, ChangeEdit, StoredDocument
from edu_doc_agent.validation.catalog import Rule
from edu_doc_agent.validation.checks import CHECK_FIELDS


def path_parts(path: str) -> list[str | int]:
    return [int(p) if p.isdigit() else p for p in path.strip("/").split("/") if p]


def get_value(data: Any, path: str) -> Any:
    for part in path_parts(path):
        data = data[part]
    return data


def set_value(data: Any, path: str, value: Any) -> None:
    parts = path_parts(path)
    if not parts:
        raise KeyError("пустой путь")
    for part in parts[:-1]:
        data = data[part]
    last = parts[-1]
    if isinstance(data, dict) and last not in data:
        raise KeyError(last)
    data[last] = value


def apply_edits(
    documents: list[StoredDocument], edits: list[ChangeEdit]
) -> tuple[list[StoredDocument], list[str]]:
    """Документы с правками и ошибки правок (неизвестный документ, путь, тип значения)."""
    by_id = {d.document_id: d for d in documents}
    dumps: dict[str, dict] = {}
    errors = []
    for e in edits:
        document = by_id.get(e.document_id)
        if document is None:
            errors.append(f"нет документа {e.document_id}")
            continue
        dump = dumps.setdefault(e.document_id, document.data.model_dump(mode="json"))
        try:
            if e.op == "set":
                set_value(dump, e.path, copy.deepcopy(e.new))
            elif e.op == "append":
                target = get_value(dump, e.path)
                if not isinstance(target, list):
                    raise TypeError(e.path)
                target.append(copy.deepcopy(e.new))
        except (KeyError, IndexError, TypeError):
            errors.append(f"{document.filename}: нет значения по пути «{e.path}»")
    # удаления после остальных правок, с конца списка: номера элементов — как в исходных данных
    removals = sorted(
        (e for e in edits if e.op == "remove" and e.document_id in dumps),
        key=lambda e: path_parts(e.path),
        reverse=True,
    )
    for e in removals:
        *parent, index = path_parts(e.path)
        try:
            container = get_value(dumps[e.document_id], "/".join(map(str, parent)))
            if not isinstance(container, list) or not isinstance(index, int):
                raise TypeError(e.path)
            del container[index]
        except (KeyError, IndexError, TypeError):
            errors.append(f"{by_id[e.document_id].filename}: нет элемента списка «{e.path}»")
    out = []
    for d in documents:
        if d.document_id in dumps:
            try:
                d = replace(d, data=type(d.data).model_validate(dumps[d.document_id]))
            except ValidationError as error:
                errors.append(f"{d.filename}: значение не подходит по типу: {error}")
        out.append(d)
    return out, errors


def top_field(path: str) -> str:
    return str(path_parts(path)[0])


def rules_using(rules: list[Rule], fields: set[str]) -> list[Rule]:
    def used(rule: Rule) -> set[str]:
        if rule.method == "llm":
            return set(rule.fields)
        return set(CHECK_FIELDS.get(rule.check or "", ()))

    return [r for r in rules if used(r) & fields]


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # 6.0 → «6»
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 300 else text[:300] + "…"


def edit_text(e: ChangeEdit, titles: dict[str, str]) -> str:
    where = f"{titles.get(e.document_id, e.document_id)}, «{e.path}»"
    if e.op == "remove":
        return f"{where}: удалить «{_fmt(e.old)}»"
    if e.op == "append":
        return f"{where}: добавить «{_fmt(e.new)}»"
    if isinstance(e.old, dict) and isinstance(e.new, dict):  # элемент списка целиком
        changed = [
            f"{key}: «{_fmt(e.old.get(key))}» → «{_fmt(value)}»"
            for key, value in e.new.items()
            if key != "source" and e.old.get(key) != value
        ]
        return f"{where}: {'; '.join(changed) or 'без изменений'}"
    return f"{where}: было «{_fmt(e.old)}», стало «{_fmt(e.new)}»"


def change_text(change: Change, titles: dict[str, str]) -> str:
    lines = ["Предлагаемое изменение:"]
    for i, e in enumerate(change.edits, start=1):
        lines.append(f"{i}. {edit_text(e, titles)}")
    if change.rationale:
        lines.append(f"Обоснование: {change.rationale}")
    if change.consequences:
        lines.append("Последствия:")
        lines += [f"- {c}" for c in change.consequences]
    return "\n".join(lines)
