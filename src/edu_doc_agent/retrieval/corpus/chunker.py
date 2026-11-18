"""Чанкер нормативов: блоки парсера (`ParsedDocument`) -> фрагменты корпуса (ADR-0005).

Формат исходника (PDF, DOCX) чанкеру неважен — он работает с блоками `parsing/`.
Как режется документ:

1. Всё до начала тела (преамбула приказа, титул «УТВЕРЖДЕН …») пропускается; из этой части
   берутся метаданные ФГОС (код направления, приказ, редакция), если их нет в манифесте.
2. Заголовки разделов («III. Требования…», «2. Требования…») идут в `path`, фрагментами
   не становятся.
3. Абзац с номером («3.3.», «2.2.1.6.») открывает пункт; абзацы без номера, подписи и таблицы
   дописываются к текущему пункту. Номер меньше предыдущего — опечатка нумерации, абзац
   остаётся в текущем пункте (с предупреждением).
4. Компетенции — отдельные фрагменты `<пункт>:<код>`: строки таблицы компетенций и абзацы,
   начинающиеся с кода («ОПК-1. Способен…»). Пункт хранит вводную фразу.
5. «Информация об изменениях: Пункт 3.3 изменен…» -> `changed_by` пункта 3.3.
6. Приложение: у ФГОС — перечень профстандартов (фрагмент на профстандарт), у ОС СПбГУ —
   отбрасывается (таблица без рамок; УКБ добавляются через overrides).
7. Длинный пункт делится по предложениям на части без потерь (`part`).
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus.codes import (
    COMPETENCY_PREFIX,
    DASH,
    PROF_STANDARD_CODE,
    find_competency_codes,
    normalize_code,
)
from edu_doc_agent.retrieval.corpus.models import (
    ChangeNote,
    CorpusDocument,
    DocType,
    DocumentInfo,
    Fragment,
    Overrides,
)

MAX_PART_CHARS = 1500  # длиннее — пункт делится на части по предложениям
MIN_PART_CHARS = 200  # хвост короче присоединяется к предыдущей части


@dataclass(frozen=True)
class Profile:
    """Чем документы одного типа отличаются при нарезке."""

    doc_type: DocType
    body_start: re.Pattern[str]  # первый заголовок тела
    section: re.Pattern[str]  # заголовок раздела
    appendix: re.Pattern[str]  # начало приложения (после тела)
    appendix_mode: Literal["prof_standards", "skip"]


PROFILES: dict[str, Profile] = {
    "fgos": Profile(
        doc_type="fgos",
        body_start=re.compile(r"^I\.\s+Общие положения"),
        section=re.compile(r"^[IVX]+\.\s+\S"),
        appendix=re.compile(r"^Приложение\b"),
        appendix_mode="prof_standards",
    ),
    "spbu_standard": Profile(
        doc_type="spbu_standard",
        body_start=re.compile(r"^1\.\s+Общие положения"),
        section=re.compile(r"^\d{1,2}\.\s+[А-ЯЁ]"),
        appendix=re.compile(r"^Приложение к Образовательному стандарту"),
        appendix_mode="skip",
    ),
}

# «3.3. Программа…», «4.4.3.Не менее…» (без пробела), «2.4 Организация…» (без точки)
_CLAUSE = re.compile(r"^(\d+(?:\.\d+)+)(?:\.\s*|\s+)(?=[«\"(]?[А-ЯЁA-Zа-яё])")
# абзац или ячейка, начинающиеся с кода: «ОПК-1. Способен…», «УКБ-1: Способен…»
_COMPETENCY = re.compile(rf"^({COMPETENCY_PREFIX}{DASH}\d+)[.:]?\s+\S", re.S)
_CHANGE_HEADER = "Информация об изменениях"
_CHANGE = re.compile(
    r"^(?:Пункт|Подпункт|Абзац|Раздел|Таблица|Приложение)\b.*?"
    r"(?:изменен|дополнен|утратил|признан|исключен)"
)
_CHANGE_TARGET = re.compile(r"^(?:Пункт|Подпункт)\s+(\d+(?:\.\d+)+)")
_CHANGE_SINCE = re.compile(r"\b(с \d{1,2} [а-я]+ \d{4} г\.)")
_CHANGE_ACT = re.compile(r"\s-\s(.+?)(?:\s+См\..*)?$")
_PROF_GROUP = re.compile(r"^\d{2}\s+\D")  # «06 Связь, информационные и коммуникационные технологии»
_SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[«\"(]?[А-ЯЁA-Z0-9])|\n")


def chunk_document(
    parsed: ParsedDocument, info: DocumentInfo, overrides: Overrides | None = None
) -> CorpusDocument:
    """Нарезать документ на фрагменты.

    `info` — метаданные (см. `detect_info`), `overrides` — ручные правки этого документа.
    """
    profile = PROFILES[info.doc_type]
    state = _State(info, profile)
    for block in _body(parsed.blocks, profile, state.warnings):
        state.feed(block)
    state.close_clause()
    fragments = state.bind_changes()
    if overrides is not None:
        fragments = _apply(fragments, overrides, info, state.warnings)
    fragments = [part for fragment in fragments for part in _split(fragment)]
    state.warnings.extend(_check(fragments, info))
    return CorpusDocument(document=info, fragments=fragments, warnings=state.warnings)


def _apply(
    fragments: list[Fragment], overrides: Overrides, info: DocumentInfo, warnings: list[str]
) -> list[Fragment]:
    by_id = {f.id: f for f in fragments}
    for fid in [*overrides.replace, *overrides.drop]:
        if fid not in by_id:
            warnings.append(f"overrides: фрагмента {fid} нет")
    result = []
    for f in fragments:
        if f.id in overrides.drop:
            continue
        if f.id in overrides.replace:
            patch = overrides.replace[f.id]
            update = {"text": patch} if isinstance(patch, str) else dict(patch)
            unknown = update.keys() - _REPLACEABLE
            if unknown:
                warnings.append(f"overrides: у {f.id} нельзя менять {', '.join(sorted(unknown))}")
                update = {k: v for k, v in update.items() if k in _REPLACEABLE}
            if f.kind == "clause" and "text" in update:
                update["codes"] = find_competency_codes(update["text"])
            f = f.model_copy(update=update)
        result.append(f)
    for extra in overrides.add:
        data = {"doc_type": info.doc_type, "doc_id": info.doc_id, "edition": info.edition}
        fragment = Fragment(**(data | extra))
        if not fragment.codes:
            fragment.codes = find_competency_codes(fragment.text)
        result.insert(_position(result, fragment), fragment)
    return result


def _position(fragments: list[Fragment], new: Fragment) -> int:
    """Куда вставить добавленный фрагмент, чтобы порядок совпадал с документом:
    среди фрагментов того же пункта, компетенция — по порядку кодов."""
    same = [i for i, f in enumerate(fragments) if f.number == new.number]
    if not same:
        return len(fragments)
    if new.kind == "competency" and new.codes:
        for i in same:
            f = fragments[i]
            if (
                f.kind == "competency"
                and f.codes
                and _code_key(f.codes[0]) > _code_key(new.codes[0])
            ):
                return i
    return same[-1] + 1


def _code_key(code: str) -> tuple[str, int]:
    prefix, _, number = code.partition("-")
    return prefix, int(number) if number.isdigit() else 0


_REPLACEABLE = {"text", "group", "path"}


# Тело документа


def _body(blocks: list[Block], profile: Profile, warnings: list[str]) -> list[Block]:
    """Блоки от начала тела (включительно) до конца документа."""
    for i, block in enumerate(blocks):
        if block.kind != BlockKind.TABLE and profile.body_start.match(block.text):
            return blocks[i:]
    warnings.append("не найдено начало тела документа — нарезан весь документ")
    return blocks


@dataclass
class _Clause:
    number: str
    section: str
    lines: list[str] = field(default_factory=list)
    changes: list[ChangeNote] = field(default_factory=list)
    children: list[Fragment] = field(default_factory=list)


class _State:
    def __init__(self, info: DocumentInfo, profile: Profile) -> None:
        self.info = info
        self.profile = profile
        self.fragments: list[Fragment] = []
        self.warnings: list[str] = []
        self.section = ""
        self.after_section = False  # предыдущий блок — заголовок раздела
        self.clause: _Clause | None = None
        self.last_key: tuple[int, ...] = ()
        self.changes: dict[str, list[ChangeNote]] = {}
        self.unbound_changes: list[ChangeNote] = []  # изменения без номера пункта
        self.appendix = False
        self.appendix_title: list[str] = []
        self.prof_group: str | None = None

    # маршрутизация блоков

    def feed(self, block: Block) -> None:
        if self.appendix:
            self._appendix(block)
        elif block.kind == BlockKind.TABLE:
            self._table(block)
        elif block.kind == BlockKind.HEADING:
            self._heading(block.text)
        else:
            self._paragraph(block.text)

    def _heading(self, text: str) -> None:
        if self.profile.appendix.match(text) and self.last_key:
            self.close_clause()
            self.appendix = True
            return
        if self.profile.section.match(text):
            self.close_clause()
            self.section, self.after_section = text, True
            return
        if self.after_section and text[:1].islower():  # перенос длинного заголовка раздела
            self.section = f"{self.section} {text}"
            return
        self.after_section = False
        self._paragraph(text)  # подпись («Таблица», «Структура и объем …») — часть пункта

    def _paragraph(self, text: str) -> None:
        self.after_section = False
        if text.startswith(_CHANGE_HEADER):  # сама отметка — следующим абзацем (или её нет)
            return
        if _CHANGE.match(text):  # раньше приложения: «Приложение изменено…»
            self._change(text)
            return
        if self.profile.appendix.match(text) and self.last_key:
            self.close_clause()
            self.appendix = True
            return
        if match := _CLAUSE.match(text):
            key = _key(match.group(1))
            if key > self.last_key:
                self.close_clause()
                self.last_key = key
                number = match.group(1)
                self.clause = _Clause(number, self.section, [text], self.unbound_changes)
                self.unbound_changes = []
                return
            self.warnings.append(f"номер {match.group(1)} не по порядку — оставлен в тексте")
        if self.clause is None:
            self.warnings.append(f"текст вне пункта пропущен: {text[:60]}")
            return
        if match := _COMPETENCY.match(text):
            self._competency(match.group(1), text, group=None)
            return
        self.clause.lines.append(text)

    def _change(self, text: str) -> None:
        since = _CHANGE_SINCE.search(text)
        act = _CHANGE_ACT.search(text)
        note = ChangeNote(
            since=since.group(1) if since else None,
            act=act.group(1) if act else text,
        )
        if target := _CHANGE_TARGET.match(text):
            self.changes.setdefault(target.group(1), []).append(note)
        else:
            self.unbound_changes.append(note)

    def _table(self, block: Block) -> None:
        if self.clause is None:
            self.warnings.append("таблица вне пункта пропущена")
            return
        if _is_competency_table(block):
            for row in block.rows:
                cell = next((i for i, c in enumerate(row) if _COMPETENCY.match(c)), None)
                if cell is None:
                    if any(row) and "компетенц" not in " ".join(row).lower():  # не шапка
                        self.warnings.append(
                            f"п. {self.clause.number}: строка таблицы компетенций без кода: "
                            f"{' | '.join(c for c in row if c)[:80]}"
                        )
                    continue
                code = _COMPETENCY.match(row[cell]).group(1)
                group = row[0] if cell > 0 and row[0] else None
                self._competency(code, row[cell], group)
            return
        self.clause.lines.extend(_row_text(row) for row in block.rows if any(row))

    def _competency(self, raw_code: str, text: str, group: str | None) -> None:
        assert self.clause is not None
        code = normalize_code(raw_code)
        parent = self._clause_id(self.clause.number)
        self.clause.children.append(
            Fragment(
                id=f"{parent}:{code}",
                doc_type=self.info.doc_type,
                doc_id=self.info.doc_id,
                edition=self.info.edition,
                kind="competency",
                number=self.clause.number,
                path=_path(self.clause.section, self.clause.number, code),
                text=text,
                codes=[code],
                parent=parent,
                group=group,
                changed_by=self.clause.changes,
            )
        )

    # приложение

    def _appendix(self, block: Block) -> None:
        if self.profile.appendix_mode == "skip":
            return
        if block.kind != BlockKind.TABLE:
            if block.text.startswith("Перечень") or self.appendix_title:
                self.appendix_title.append(block.text)
            return
        title = " ".join(self.appendix_title) or "Перечень профессиональных стандартов"
        self.appendix_title = []
        for row in block.rows:
            filled = [c for c in row if c]
            if len(filled) == 1 and _PROF_GROUP.match(filled[0]):
                self.prof_group = filled[0]
                continue
            code_cell = next(
                (i for i, c in enumerate(row) if PROF_STANDARD_CODE.fullmatch(c)), None
            )
            if code_cell is None:
                continue  # шапка
            code = row[code_cell]
            text = " ".join(c for c in row[code_cell + 1 :] if c) or code
            self.fragments.append(
                Fragment(
                    id=f"{self.info.doc_id}:app:{code}",
                    doc_type=self.info.doc_type,
                    doc_id=self.info.doc_id,
                    edition=self.info.edition,
                    kind="prof_standard_ref",
                    number="app",
                    path=f"Приложение › {title} › {code}",
                    text=text,
                    codes=[code],
                    group=self.prof_group,
                )
            )

    # завершение пункта

    def close_clause(self) -> None:
        clause, self.clause = self.clause, None
        if clause is None:
            return
        text = "\n".join(clause.lines)
        self.fragments.append(
            Fragment(
                id=self._clause_id(clause.number),
                doc_type=self.info.doc_type,
                doc_id=self.info.doc_id,
                edition=self.info.edition,
                kind="clause",
                number=clause.number,
                path=_path(clause.section, clause.number),
                text=text,
                codes=find_competency_codes(text),
                changed_by=clause.changes,
            )
        )
        self.fragments.extend(clause.children)

    def bind_changes(self) -> list[Fragment]:
        """Отметки «Пункт N изменен…» -> `changed_by` пункта N и его компетенций.

        Отметка стоит то перед пунктом, то после него, поэтому привязка — в конце.
        """
        result = []
        for f in self.fragments:
            if notes := self.changes.get(f.number):
                f = f.model_copy(update={"changed_by": [*f.changed_by, *notes]})
            result.append(f)
        numbers = {f.number for f in self.fragments}
        for number in self.changes.keys() - numbers:
            self.warnings.append(f"отметка об изменении пункта {number}, которого нет")
        return result

    def _clause_id(self, number: str) -> str:
        return f"{self.info.doc_id}:{number}"


def _key(number: str) -> tuple[int, ...]:
    return tuple(int(x) for x in number.split("."))


def _path(section: str, number: str, code: str | None = None) -> str:
    parts = [p for p in (section, number, code) if p]
    return " › ".join(parts)


def _row_text(row: list[str]) -> str:
    return " | ".join(c.replace("\n", " ") for c in row if c)


def _is_competency_table(block: Block) -> bool:
    """Шапка про компетенции, или большинство строк начинаются с кода компетенции
    (продолжение таблицы, оторванное от шапки)."""
    with_code = [any(_COMPETENCY.match(c) for c in row) for row in block.rows]
    head = " ".join(block.rows[0]).lower() if block.rows else ""
    if "компетенц" in head:
        return any(with_code)
    return sum(with_code) * 2 > len(with_code)


# Части длинного пункта


def _split(fragment: Fragment) -> list[Fragment]:
    """Пункт длиннее MAX_PART_CHARS -> части по границам предложений; склейка = исходный текст."""
    text = fragment.text
    if len(text) <= MAX_PART_CHARS:
        return [fragment]
    cuts = [m.end() for m in _SENTENCE_END.finditer(text)]
    parts, start, last = [], 0, 0
    for cut in [*cuts, len(text)]:
        if cut - start > MAX_PART_CHARS and last > start:
            parts.append(text[start:last])
            start = last
        last = cut
    parts.append(text[start:])
    if len(parts) > 1 and len(parts[-1]) < MIN_PART_CHARS:  # короткий хвост — к предыдущей части
        parts[-2:] = [parts[-2] + parts[-1]]
    return [fragment.model_copy(update={"part": i, "text": t}) for i, t in enumerate(parts)]


# Проверки


def _check(fragments: list[Fragment], info: DocumentInfo) -> list[str]:
    warnings = []
    ids = [(f.id, f.part) for f in fragments]
    if len(ids) != len(set(ids)):
        warnings.append("ОШИБКА: повторяющиеся id фрагментов")
    numbers = [f.number for f in fragments if f.kind == "clause" and f.part == 0]
    by_parent: dict[str, list[int]] = {}
    for n in numbers:
        *parent, last = n.split(".")
        by_parent.setdefault(".".join(parent), []).append(int(last))
    for parent, xs in by_parent.items():
        if xs != list(range(1, len(xs) + 1)):
            warnings.append(f"нумерация внутри {parent}: {xs}")
    if info.doc_type == "fgos":
        for category in ("УК", "ОПК"):
            codes = [
                f.codes[0]
                for f in fragments
                if f.kind == "competency" and f.codes[0].startswith(f"{category}-")
            ]
            expected = [f"{category}-{i}" for i in range(1, len(codes) + 1)]
            if not codes:
                warnings.append(f"не найдены {category}")
            elif codes != expected:
                warnings.append(f"{category} не подряд: {', '.join(codes)}")
    for f in fragments:
        if not f.text.strip():
            warnings.append(f"{f.id}: пустой текст")
    return warnings


# Метаданные ФГОС по тексту

_MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря"
_DIRECTION = re.compile(r"по направлению подготовки\s+(\d{2}\.(\d{2})\.\d{2})\s+([^\"«»(]+)")
_ORDER = re.compile(r"от\s+(\d{1,2}\s+[а-я]+\s+\d{4}|\d{2}\.\d{2}\.\d{4})\s*г?\.?\s*N\s*(\d+)")
_EDITION_CONSULTANT = re.compile(r"\(ред\.\s*от\s*(\d{2}\.\d{2}\.\d{4})\)")
_EDITION_GARANT = re.compile(
    r"С изменениями и дополнениями от:\s*((?:\d{1,2} [а-я]+ \d{4} г\.,?\s*)+)"
)
_RU_DATE = re.compile(r"(\d{1,2})\s+([а-я]+)\s+(\d{4})")
_LEVELS = {"03": "бакалавриат", "04": "магистратура", "05": "специалитет"}


def detect_info(
    parsed: ParsedDocument, doc_type: DocType, source_file: str, **given: str
) -> DocumentInfo:
    """Метаданные документа: заданные явно (`given`, из манифеста) или найденные в тексте.

    Для ФГОС по тексту определяются код и название направления, уровень, приказ и редакция.
    Для остальных типов `doc_id`, `title`, `edition` нужно задать в манифесте.
    """
    found: dict[str, str] = {}
    if doc_type == "fgos":
        head = " ".join(b.text for b in parsed.blocks[:60] if b.kind != BlockKind.TABLE)
        if m := _DIRECTION.search(head):
            code, level_code, name = m.group(1), m.group(2), m.group(3).strip(" .,")
            level = _LEVELS.get(level_code, "")
            found |= {
                "doc_id": f"fgos-{code}",
                "direction_code": code,
                "direction_name": name,
                "level": level,
                "title": f"ФГОС ВО — {level} по направлению {code} {name}".replace("  ", " "),
            }
        if m := _ORDER.search(head):
            date, number = _date(m.group(1)), m.group(2)
            found["order"] = f"Приказ Минобрнауки России от {date} N {number}"
            edition = f"{number} от {date}"
            if e := _EDITION_CONSULTANT.search(head):
                edition = f"{number} ред. от {e.group(1)}"
            elif e := _EDITION_GARANT.search(head):
                dates = _RU_DATE.findall(e.group(1))
                if dates:
                    edition = f"{number} ред. от {_date(' '.join(dates[-1]))}"
            found["edition"] = edition
    values = {**found, **{k: v for k, v in given.items() if v}}
    missing = [k for k in ("doc_id", "title", "edition") if k not in values]
    if missing:
        raise ValueError(f"{source_file}: не определены {', '.join(missing)} — задайте в манифесте")
    return DocumentInfo(doc_type=doc_type, source_file=source_file, **values)


def _date(text: str) -> str:
    if m := _RU_DATE.fullmatch(text.strip()):
        day, month, year = m.groups()
        return f"{int(day):02d}.{_MONTHS.split().index(month) + 1:02d}.{year}"
    return text
