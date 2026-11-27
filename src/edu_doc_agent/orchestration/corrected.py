"""Применение подтверждённого изменения: правки в данные документов, перепроверка правилами
`method: code` и исправленный DOCX (правки в исходном файле) — в хранилище сессии."""

from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel

from edu_doc_agent.extraction.source import SourceRef
from edu_doc_agent.generation.document import Patch, patch_docx, source_for
from edu_doc_agent.orchestration.check import select_rules
from edu_doc_agent.orchestration.deps import GraphDeps
from edu_doc_agent.orchestration.edits import apply_edits, rules_using, top_field
from edu_doc_agent.orchestration.program import invalidate, load_program
from edu_doc_agent.orchestration.state import OutputFile, empty_scope
from edu_doc_agent.orchestration.stores import Change, ChangeEdit, StoredDocument
from edu_doc_agent.validation.findings import finding_id
from edu_doc_agent.validation.runner import check_with_code


async def save_corrected(
    deps: GraphDeps,
    session_id: str,
    change: Change,
    before: list[StoredDocument],
    after: list[StoredDocument],
) -> tuple[list[OutputFile], list[str]]:
    """Исправленные документы: правки изменения в исходном DOCX (`generation/document.py`),
    файл — в хранилище сессии. `before` — данные до правок (места значений в документе),
    `after` — после. Возвращает файлы и заметки: что внести в документ вручную."""
    if deps.storage is None:
        return [], []
    by_id = {d.document_id: d for d in before}
    changed = {d.document_id: d for d in after}
    files: list[OutputFile] = []
    notes: list[str] = []
    for doc_id in dict.fromkeys(e.document_id for e in change.edits):
        document = by_id[doc_id]
        edits = [e for e in change.edits if e.document_id == doc_id]
        content = await _current_file(deps, document)
        if content is None or not document.filename.lower().endswith(".docx"):
            reason = "исходный файл не найден" if content is None else "файл не DOCX"
            notes.append(f"{document.filename}: {reason}, правки ({len(edits)}) внесите вручную")
            continue
        patches = _document_patches(document, edits)
        result = patch_docx(content, patches)
        name = f"{Path(document.filename).stem} (исправлено).docx"
        key = await deps.storage.save(
            session_id=session_id,
            doc_id=f"corrected-{change.id}",
            filename=name,
            content=result.content,
        )
        await deps.stores.save_document(replace(changed[doc_id], file_key=key))
        files.append(
            {
                "storage_key": key,
                "filename": name,
                "kind": "corrected_document",
                "derived_from": document.filename,
            }
        )
        note = f"{name}: перенесено правок {len(result.applied)} из {len(patches)}"
        if result.manual:
            note += f"; внесите вручную: {'; '.join(result.manual)}"
        notes.append(note)
    return files, notes


_SHARED = "место в документе общее с другими значениями: удалите вручную"
_REPLACED = "элемент заменён другим: удалите старый и добавьте новый вручную"


def _document_patches(document: StoredDocument, edits: list[ChangeEdit]) -> list[Patch]:
    """Правки для файла документа. Удаление переносится, только если документ не потеряет
    другое содержимое: в том же списке нет добавления (это замена элемента), и место (строка,
    абзац) не общее с элементами, которые остаются, в том числе с теми, чьё удаление само
    не переносится."""
    appended = {e.path for e in edits if e.op == "append"}
    places = _places(document.data)

    def quote(path: str) -> str | None:
        source = source_for(document.data, path)
        return source.quote if source else None

    removals = {e.path for e in edits if e.op == "remove"}
    reasons = {p: _REPLACED for p in removals if p.rsplit("/", 1)[0] in appended}
    while True:  # отказ от одного удаления может сделать место общим для других
        effective = removals - set(reasons)
        shared = {p for p in effective if (q := quote(p)) and set(places.get(q, [])) - effective}
        if not shared:
            break
        reasons |= dict.fromkeys(shared, _SHARED)
    patches = []
    for e in edits:
        if e.op == "remove" and e.path in reasons:
            patches.append(Patch(e.path, e.op, e.old, e.new, None, reasons[e.path]))
        else:
            patches += _patches(document, e)
    return patches


def _places(data: BaseModel) -> dict[str, list[str]]:
    """Цитата места в документе → пути элементов списков, которые там записаны."""
    places: dict[str, list[str]] = {}
    for name in type(data).model_fields:
        value = getattr(data, name)
        if not isinstance(value, list):
            continue
        for i, item in enumerate(value):
            source = getattr(item, "source", None)
            if isinstance(source, SourceRef) and source.quote:
                places.setdefault(source.quote, []).append(f"{name}/{i}")
    return places


def _patches(document: StoredDocument, edit: ChangeEdit) -> list[Patch]:
    """Правки для файла. Замена элемента списка целиком («competences/0» → {…}) — это правки
    изменившихся полей элемента: в документе меняется код, а не вся строка."""
    source = source_for(document.data, edit.path)
    quote = source.quote if source else None
    if edit.op == "set" and isinstance(edit.old, dict) and isinstance(edit.new, dict):
        return [
            Patch(f"{edit.path}/{key}", "set", edit.old.get(key), value, quote)
            for key, value in edit.new.items()
            if key != "source" and edit.old.get(key) != value and not isinstance(value, dict | list)
        ]
    return [Patch(edit.path, edit.op, edit.old, edit.new, quote)]


async def _current_file(deps: GraphDeps, document: StoredDocument) -> bytes | None:
    """Текущая версия файла: последний исправленный или исходный."""
    if document.file_key and deps.storage is not None:
        return await deps.storage.read(document.file_key)
    path = Path(document.path)
    return path.read_bytes() if document.path and path.is_file() else None


async def apply_draft(
    deps: GraphDeps,
    session_id: str,
    run_id: str,
    change: Change,
    direction_code: str | None,
    finding_ids: list[str],
) -> tuple[list[str], list[OutputFile]]:
    """Правки черновика в данные и файлы документов, затронутые документы перепроверяются
    правилами `method: code`. Возвращает открытые замечания сессии и исправленные файлы."""
    before = await deps.stores.list_documents(session_id)
    after, errors = apply_edits(before, change.edits)
    if errors:
        await deps.stores.save_change(
            change.model_copy(update={"status": "failed", "problems": errors})
        )
        return finding_ids, []
    affected = {e.document_id for e in change.edits}
    changed = [d for d in after if d.document_id in affected]
    for d in changed:
        await deps.stores.save_document(d)
    invalidate(deps, session_id)
    program, _ = await load_program(deps, session_id)
    rules = select_rules(deps.rules_dir, direction_code, empty_scope())
    results = check_with_code(
        rules, program, session_id=session_id, run_id=run_id, document_ids=affected
    )
    # правила method: llm по изменённым полям проверены на черновике и выполнены
    fields = {top_field(e.path) for e in change.edits}
    passed = results.passed_ids + [
        finding_id(r.id, doc_id)
        for r in rules_using(rules, fields)
        if r.method == "llm"
        for doc_id in affected
    ]
    await deps.stores.delete_findings(session_id, passed)
    await deps.stores.save_findings(session_id, results.findings)
    evaluated = set(passed) | {f.id for f in results.findings}
    open_ids = [i for i in finding_ids if i not in evaluated] + [f.id for f in results.findings]
    files, notes = await save_corrected(deps, session_id, change, before, changed)
    await deps.stores.save_change(
        change.model_copy(update={"status": "applied", "document_notes": notes})
    )
    return open_ids, files
