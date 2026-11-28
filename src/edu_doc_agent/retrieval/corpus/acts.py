"""Чанкер федеральных актов из ГАРАНТа: закон по статьям и частям, приказ по пунктам (#50).

Закон (`layout: law`, 273-ФЗ):

- «Глава 2. Система образования» и «Статья 13. Общие требования…» идут в `path`;
- фрагмент — часть статьи («5. Количество зачетных единиц…»): id `273-fz:ст.13:ч.5`,
  номер `ст.13:ч.5`; пункты части («1) …», «а) …») остаются в ней;
- в статье без частей фрагмент — пункт («6) образовательная программа - …», ст. 2): id
  `273-fz:ст.2:п.6`; текст статьи до пунктов — фрагмент `273-fz:ст.2`, он же `context`
  пунктов («Для целей настоящего Федерального закона применяются следующие основные
  понятия:»); статья без частей и пунктов — один фрагмент `273-fz:ст.N`.

Приказ (`layout: order`, приказы № 245, 885/390): в корпус идёт только утверждённый документ
(«Порядок…», «Положение…» после «УТВЕРЖДЕН»), без текста самого приказа, подписей и
приложений-форм. Фрагмент — пункт («15.», «17(1).»): id `order-245:15`; абзацы без номера
дописываются к пункту, разделы «II. …» идут в `path`.

Общее: служебные вставки ГАРАНТа убираются (`garant.py`), «Информация об изменениях» ->
`changed_by` фрагмента, утратившие силу части и пункты не берутся, сноски после черты
«──────» отбрасываются.
"""

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus import garant
from edu_doc_agent.retrieval.corpus.models import (
    ChangeNote,
    CorpusDocument,
    DocumentInfo,
    Fragment,
)

ActLayout = Literal["law", "order"]
LAYOUTS: tuple[str, ...] = ActLayout.__args__

_CHAPTER = re.compile(r"^Глава\s+\d+(?:\.\d+)?\.\s")
_ARTICLE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$", re.S)
_PART = re.compile(r"^(\d+(?:\.\d+)?)\.\s+(?=\S)")
_ITEM = re.compile(r"^(\d+(?:\.\d+)?)\)\s")
_POINT = re.compile(r"^(\d+(?:\(\d+\))?)\.\s*(?=[«\"(]?[А-ЯЁA-Zа-яё])")
_SECTION = re.compile(r"^[IVX]+\.\s+\S")
_APPROVED = re.compile(r"^УТВЕРЖДЕН[АОЫ]?\b|^Утвержден[аоы]?$", re.IGNORECASE)
_DOCUMENT_TITLE = re.compile(r"^(?:Порядок|Положение|Правила|Требования)\b")
_APPENDIX = re.compile(r"^Приложение\b")
# пункт части, утративший силу: «3) утратил силу с 5 мая 2020 г. - …»
_REPEALED_ITEM = re.compile(r"^(?:\d+(?:\.\d+)?|[а-я])\)\s*утратил[аио]?\s+силу", re.I)
# две части в одном абзаце (парсер не увидел границу): «3. … образования. 4. В Российской…»
_GLUED_PART = re.compile(r"(?<=[.;:])\s+(\d+(?:\.\d+)?)\.\s+(?=[«\"(]?[А-ЯЁ])")
_REPEALED = re.compile(r"^(?:\d+(?:\.\d+)?(?:\(\d+\))?[.)]\s*)?Утратил[аио]?\s+силу", re.I)
# цели отметок об изменениях
_NOTE_PART = re.compile(r"^Част(?:ь|и)\s+(\d+(?:\.\d+)?)(?:\s+статьи\s+(\d+(?:\.\d+)?))?")
_NOTE_ADDED_PART = re.compile(
    r"^Статья\s+(\d+(?:\.\d+)?)\s+дополнена\s+част(?:ью|ями)\s+(\d+(?:\.\d+)?)"
)
_NOTE_ITEM = re.compile(r"^Пункт\s+(\d+(?:\.\d+)?)(?:\s+части\s+(\d+(?:\.\d+)?))?")
_NOTE_ADDED_ITEM = re.compile(r"пункт(?:ом|ами)\s+(\d+(?:\.\d+)?)")
_NOTE_POINT = re.compile(r"(?:^Пункт|пунктом)\s+(\d+(?:\(\d+\))?)")


@dataclass
class _Unit:
    """Будущий фрагмент: часть статьи, пункт статьи без частей, вводный текст статьи или
    пункт приказа."""

    key: tuple[str, ...]  # («13», «ч», «5») / («2», «п», «6») / («2»,) / («15»,)
    path: str
    lines: list[str] = field(default_factory=list)
    changes: list[ChangeNote] = field(default_factory=list)


def chunk_act(parsed: ParsedDocument, info: DocumentInfo, layout: ActLayout) -> CorpusDocument:
    warnings: list[str] = []
    blocks = _unglued(parsed.blocks)
    if layout == "law":
        units, notes = _law_units(blocks, warnings)
    else:
        units, notes = _order_units(blocks, warnings)
    fragments = _fragments(units, notes, info, layout, warnings)
    return CorpusDocument(document=info, fragments=fragments, warnings=warnings)


def _unglued(blocks: list[Block]) -> list[Block]:
    """Абзац «3. … 4. …» -> два абзаца, если второй номер — следующий по порядку."""
    out: list[Block] = []
    for block in blocks:
        start = _PART.match(block.text) if block.kind != BlockKind.TABLE else None
        while start:
            expected = _next_number(start.group(1))
            glued = next(
                (m for m in _GLUED_PART.finditer(block.text) if m.group(1) == expected), None
            )
            if glued is None:
                break
            out.append(replace(block, text=block.text[: glued.start()]))
            block = replace(block, text=block.text[glued.start() :].lstrip())
            start = _PART.match(block.text)
        out.append(block)
    return out


def _next_number(number: str) -> str:
    """«3» -> «4», «3.1» -> «4» (за дополнительной частью идёт следующая целая)."""
    return str(int(number.split(".")[0]) + 1)


# Закон


def _law_units(
    blocks: list[Block], warnings: list[str]
) -> tuple[list[_Unit], dict[tuple, list[ChangeNote]]]:
    units: list[_Unit] = []
    notes: dict[tuple, list[ChangeNote]] = {}
    pending: list[ChangeNote] = []  # отметка без явной цели — к следующему фрагменту
    chapter = article_title = article = ""
    has_parts = title_open = False
    started = False

    def open_unit(key: tuple[str, ...], label: str) -> None:
        nonlocal pending
        path = " › ".join(p for p in (chapter, article_title, label) if p)
        units.append(_Unit(key, path, changes=pending))
        pending = []

    for block in blocks:
        if started and (found := garant.change_note(block)):
            target, note = found
            key = _law_note_target(target, article, units[-1].key if units else None)
            if key is None:
                pending.append(note)
            else:
                notes.setdefault(key, []).append(note)
            continue
        if block.kind == BlockKind.TABLE:
            if started and not garant.is_service_table(block) and units:
                units[-1].lines.extend(_row_text(r) for r in block.rows if any(r))
            continue
        text = garant.clean_text(block.text)
        if garant.FOOTNOTE_RULE.match(text):
            break
        if _CHAPTER.match(text):
            started, chapter, article, title_open = True, text, "", False
            continue
        if not started or garant.is_service_paragraph(text):
            continue
        if match := _ARTICLE.match(text):
            article, article_title = match.group(1), text
            has_parts, title_open = False, True
            if _REPEALED.match(match.group(2)):
                article = ""  # статья утратила силу: её текст не берём
            continue
        if not article:
            continue
        if title_open and text[:1].islower():  # перенос длинного заголовка статьи
            article_title = f"{article_title} {text}"
            continue
        title_open = False
        part, item = _PART.match(text), _ITEM.match(text)
        if part and (not units or units[-1].key[0] != article or _after(units, part.group(1))):
            has_parts = True
            open_unit((article, "ч", part.group(1)), f"ч. {part.group(1)}")
        elif item and not has_parts and _new_item(units, article, item.group(1)):
            if not units or units[-1].key[0] != article:
                warnings.append(f"ст. {article}: пункт {item.group(1)} без вводного текста")
            open_unit((article, "п", item.group(1)), f"п. {item.group(1)}")
        elif not units or units[-1].key[0] != article:
            open_unit((article,), "")
        if not _REPEALED_ITEM.match(text):
            units[-1].lines.append(text)
    if not started:
        warnings.append("ОШИБКА: не найдено начало закона («Глава 1. …»)")
    return units, notes


def _after(units: list[_Unit], number: str) -> bool:
    """Новая часть той же статьи: после вводного текста или номер больше открытой части
    (иначе «1.» внутри части — не новая часть)."""
    last = units[-1].key
    if len(last) == 1:
        return True
    return last[1] == "ч" and _num(number) > _num(last[2])


def _new_item(units: list[_Unit], article: str, number: str) -> bool:
    if not units or units[-1].key[0] != article or len(units[-1].key) != 3:
        return True
    return units[-1].key[1] == "п" and _num(number) > _num(units[-1].key[2])


def _law_note_target(target: str, article: str, current: tuple | None) -> tuple | None:
    if match := _NOTE_ADDED_PART.match(target):
        return (match.group(1), "ч", match.group(2))
    if match := _NOTE_PART.match(target):
        return (match.group(2) or article, "ч", match.group(1))
    if match := _NOTE_ITEM.match(target):
        if match.group(2):  # «Пункт 2 части 7» — изменение внутри части
            return (article, "ч", match.group(2))
        if current is not None and len(current) == 3 and current[1] == "ч":
            return current  # пункт внутри открытой части
        return (article, "п", match.group(1))
    if (match := _NOTE_ADDED_ITEM.search(target)) and target.startswith("Статья"):
        return (article, "п", match.group(1))
    return None


# Приказ


def _order_units(
    blocks: list[Block], warnings: list[str]
) -> tuple[list[_Unit], dict[tuple, list[ChangeNote]]]:
    start = _order_body(blocks)
    if start is None:
        warnings.append("ОШИБКА: не найден утверждённый документ («УТВЕРЖДЕН… Порядок/Положение»)")
        return [], {}
    title = _short(garant.clean_text(blocks[start].text))
    units: list[_Unit] = []
    notes: dict[tuple, list[ChangeNote]] = {}
    pending: list[ChangeNote] = []
    section = ""
    for block in blocks[start + 1 :]:
        if found := garant.change_note(block):
            target, note = found
            if match := _NOTE_POINT.search(target):
                notes.setdefault((match.group(1),), []).append(note)
            elif "сноск" not in target:
                pending.append(note)
            continue
        if block.kind == BlockKind.TABLE:
            if not garant.is_service_table(block) and units:
                units[-1].lines.extend(_row_text(r) for r in block.rows if any(r))
            continue
        text = garant.clean_text(block.text)
        if garant.FOOTNOTE_RULE.match(text) or (_APPENDIX.match(text) and units):
            break
        if garant.is_service_paragraph(text) or text.startswith(garant.EDITION_HEADER):
            continue
        if block.kind == BlockKind.HEADING and _SECTION.match(text):
            section = text
            continue
        point = _POINT.match(text)
        if point and (not units or _num(point.group(1)) > _num(units[-1].key[0])):
            number = point.group(1)
            path = " › ".join(p for p in (title, section, f"п. {number}") if p)
            units.append(_Unit((number,), path, changes=pending))
            pending = []
        elif not units:
            warnings.append(f"текст до первого пункта пропущен: {text[:60]}")
            continue
        if not _REPEALED_ITEM.match(text):
            units[-1].lines.append(text)
    return units, notes


def _short(title: str, limit: int = 80) -> str:
    """Название утверждённого документа для `path`: длинное — до границы слова."""
    if len(title) <= limit:
        return title
    return title[:limit].rsplit(" ", 1)[0] + "…"


def _order_body(blocks: list[Block]) -> int | None:
    """Индекс заголовка утверждённого документа: первый «Порядок…/Положение…» после
    «УТВЕРЖДЕН»."""
    approved = False
    for i, block in enumerate(blocks):
        if block.kind == BlockKind.TABLE:
            continue
        if _APPROVED.match(block.text):
            approved = True
        elif approved and _DOCUMENT_TITLE.match(block.text):
            return i
    return None


# Фрагменты


def _fragments(
    units: list[_Unit],
    notes: dict[tuple, list[ChangeNote]],
    info: DocumentInfo,
    layout: ActLayout,
    warnings: list[str],
) -> list[Fragment]:
    from edu_doc_agent.retrieval.corpus.chunker import split_long

    intros = {u.key[0]: "\n".join(u.lines) for u in units if len(u.key) == 1}
    fragments: list[Fragment] = []
    seen: set[str] = set()
    for unit in units:
        text = "\n".join(unit.lines).strip()
        if not text or _REPEALED.match(text):  # пункт утратил силу — от него ничего не осталось
            continue
        number = _number(unit.key, layout)
        fid = f"{info.doc_id}:{number}"
        if fid in seen:
            warnings.append(f"{number}: повтор номера — текст присоединён к первому")
            i = next(i for i, f in enumerate(fragments) if f.id == fid)
            fragments[i] = fragments[i].model_copy(update={"text": f"{fragments[i].text}\n{text}"})
            continue
        seen.add(fid)
        context = intros.get(unit.key[0], "") if layout == "law" and unit.key[1:2] == ("п",) else ""
        fragments.append(
            Fragment(
                id=fid,
                doc_type=info.doc_type,
                doc_id=info.doc_id,
                edition=info.edition,
                kind="clause",
                number=number,
                path=unit.path,
                text=text,
                context=context,
                changed_by=[*unit.changes, *notes.get(unit.key, [])],
            )
        )
    if not fragments:
        warnings.append("ОШИБКА: нет ни одного фрагмента")
    return [part for f in fragments for part in split_long(f)]


def _number(key: tuple[str, ...], layout: ActLayout) -> str:
    """(«13», «ч», «5») -> «ст.13:ч.5»; («2»,) -> «ст.2»; у приказа («15»,) -> «15»."""
    if layout == "order":
        return key[0]
    article, *rest = key
    return f"ст.{article}:{rest[0]}.{rest[1]}" if rest else f"ст.{article}"


def _num(number: str) -> tuple[int, ...]:
    """«3.1» -> (3, 1); «17(1)» -> (17, 1) — для сравнения порядка номеров."""
    return tuple(int(x) for x in re.findall(r"\d+", number))


def _row_text(row: list[str]) -> str:
    return " | ".join(c.replace("\n", " ") for c in row if c)
