"""Профессиональный стандарт -> фрагменты корпуса (#50).

Вход — текст в формате карточки: подписи полей («Наименование», «Код», «Трудовые действия»,
«Необходимые умения»…) стоят на своих строках, значения — на следующих. Так выглядит страница
профстандарта на сайте, сохранённая как `.txt`; PDF приказа с текстовым слоем приводится к
тому же виду в `card_from_pdf` (скан без текста не поддерживается — нужен OCR). Разбирается
раздел III; функциональная карта (раздел II) повторяет его кратко и не нужна, раздел IV
(разработчики) отбрасывается.

Фрагменты:

- общие сведения: вид деятельности, основная цель, перечень обобщённых трудовых функций —
  id `ps-06.001:general`, kind `prof_standard`, код `06.001`;
- обобщённая трудовая функция: название, уровень, должности, требования к образованию и
  опыту, особые условия, коды ОКСО — id `ps-06.001:A`, kind `generalized_function`, код
  `06.001:A`;
- трудовая функция: название, уровень, трудовые действия, необходимые умения и знания —
  id `ps-06.001:A/01.3`, kind `labor_function`, код `06.001:A/01.3`.

Буквы кодов функций приводятся к латинице: в документах ОП пишут и «А/01.3» кириллицей.
Редакция — приказ Минтруда («424н от 20.07.2022»), срок действия — в `approval_act`.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from edu_doc_agent.parsing.pdf_parser import NoTextLayerError
from edu_doc_agent.retrieval.corpus.chunker import split_long
from edu_doc_agent.retrieval.corpus.codes import PROF_STANDARD_CODE
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, DocumentInfo, Fragment

# кириллические буквы, похожие на латинские, в кодах функций: «А/01.3» -> «A/01.3»
_LATIN = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")
_HEADER = re.compile(r"^Профстандарт:\s*(\d{2}\.\d{3})")
_PERIOD = re.compile(r"действует с (\d{2}\.\d{2}\.\d{4}) по (\d{2}\.\d{2}\.\d{4})")
_MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря"
# «от 20.07.2022 № 424н» (карточка на сайте), «от 20 июля 2022 г. № 424н» (текст приказа)
_ORDER = re.compile(
    rf"от\s+(\d{{2}}\.\d{{2}}\.\d{{4}}|\d{{1,2}}\s+(?:{_MONTHS.replace(' ', '|')})\s+\d{{4}})"
    r"\s*(?:г\.)?\s*(?:№|N)\s*(\d+н)"
)
_DASHES = {"-", "–", "—"}  # «нет значения» в таблицах
_GENERALIZED = re.compile(r"^3\.(\d+)\.\s*Обобщенная трудовая функция")
_FUNCTION = re.compile(r"^3\.(\d+)\.(\d+)\.\s*Трудовая функция")
_END = re.compile(r"^IV\.\s")
_CLASSIFIER = re.compile(r"^(ОКЗ|ЕКС|ЕТКС|ОКПДТР|ОКСО)\b")
FUNCTION_CODE = re.compile(r"^[A-ZА-Я](?:/\d{2}\.\d(?:\.\d)?)?$")

# подписи полей; значения — строки до следующей подписи. После «Дополнительные
# характеристики» — таблица классификаторов со своими подписями: она читается целиком.
_LABELS = {
    "Наименование": "name",
    "Код": "code",
    "Уровень квалификации": "level",
    "Уровень (подуровень) квалификации": "level",
    "Возможные наименования должностей, профессий": "positions",
    "Требования к образованию и обучению": "education",
    "Требования к опыту практической работы": "experience",
    "Особые условия допуска к работе": "conditions",
    "Другие характеристики": "other",
    "Трудовые действия": "actions",
    "Необходимые умения": "skills",
    "Необходимые знания": "knowledge",
    "Дополнительные характеристики": "classifiers",
}
# служебные строки шапок таблиц: в текст не идут
_SKIP = {
    "Оригинал",
    "X",
    "Заимствовано из оригинала",
    "Код оригинала",
    "Регистрационный номер профессионального стандарта",
    "Происхождение обобщенной трудовой функции",
    "Происхождение трудовой функции",
}
_GENERALIZED_FIELDS = (
    ("positions", "Возможные наименования должностей, профессий"),
    ("education", "Требования к образованию и обучению"),
    ("experience", "Требования к опыту практической работы"),
    ("conditions", "Особые условия допуска к работе"),
    ("other", "Другие характеристики"),
)
_FUNCTION_FIELDS = (
    ("actions", "Трудовые действия"),
    ("skills", "Необходимые умения"),
    ("knowledge", "Необходимые знания"),
    ("other", "Другие характеристики"),
)


@dataclass
class _Block:
    number: str  # «3.1», «3.1.2»
    generalized: bool
    fields: dict[str, list[str]] = field(default_factory=dict)

    def value(self, key: str) -> str:
        return " ".join(self.fields.get(key, [])).strip()


def chunk_prof_standard(text: str, source_file: str, **given: str) -> CorpusDocument:
    """Нарезать профстандарт. `given` — метаданные из манифеста, они важнее найденных."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    warnings: list[str] = []
    info = _info(lines, source_file, given)
    code = info.doc_id.removeprefix("ps-")
    blocks = _blocks(lines)
    fragments = [_general(lines, info, code, blocks)]
    current = ""  # обобщённая функция открытого блока
    for block in blocks:
        name, fcode = block.value("name"), _function_code(block.value("code"))
        if not name or not FUNCTION_CODE.match(fcode):
            warnings.append(f"п. {block.number}: не найдены наименование или код функции")
            continue
        level = block.value("level")
        if block.generalized:
            current = f"Обобщенная трудовая функция {fcode} «{name}», уровень квалификации {level}"
            head, fields, kind = current, _GENERALIZED_FIELDS, "generalized_function"
            title = f"{block.number} Обобщенная трудовая функция {fcode}"
            context = ""
        else:
            head = f"Трудовая функция {fcode} «{name}», уровень (подуровень) квалификации {level}"
            fields, kind = _FUNCTION_FIELDS, "labor_function"
            title = f"{block.number} Трудовая функция {fcode}"
            context = current
        body = [head]
        for key, label in fields:
            values = [v for v in block.fields.get(key, []) if v not in _DASHES]
            if values:
                body += [f"{label}:", *values]
        if block.generalized and (okso := _okso(block.fields.get("classifiers", []))):
            body += ["ОКСО:", *okso]
        fragments.append(
            Fragment(
                id=f"{info.doc_id}:{fcode}",
                doc_type="prof_standard",
                doc_id=info.doc_id,
                edition=info.edition,
                kind=kind,
                number=block.number,
                path=f"{info.title} › {title}",
                text="\n".join(body),
                context=context,
                codes=[f"{code}:{fcode}"],
            )
        )
    if len(fragments) == 1:
        warnings.append("ОШИБКА: не найдено ни одной трудовой функции")
    ids = [f.id for f in fragments]
    if len(ids) != len(set(ids)):
        warnings.append("ОШИБКА: повторяющиеся коды функций")
    fragments = [part for f in fragments for part in split_long(f)]
    return CorpusDocument(document=info, fragments=fragments, warnings=warnings)


def _info(lines: list[str], source_file: str, given: dict[str, str]) -> DocumentInfo:
    head = " ".join(lines[:40])
    found: dict[str, str] = {}
    code = next((m.group(1) for ln in lines[:5] if (m := _HEADER.match(ln))), None)
    if code is None:
        code = next((ln for ln in lines[:40] if PROF_STANDARD_CODE.fullmatch(ln)), None)
    if code:
        found["doc_id"] = f"ps-{code}"
        name = _after(lines, f"Профстандарт: {code}") or _after(lines, code)
        if name:
            found["title"] = f"Профессиональный стандарт {code} «{name}»"
    if order := _ORDER.search(head):
        date, number = _numeric_date(order.group(1)), order.group(2)
        found["edition"] = f"{number} от {date}"
        act = f"Приказ Минтруда России от {date} № {number}"
        if period := _PERIOD.search(head):
            act += f"; действует с {period.group(1)} по {period.group(2)}"
        found["approval_act"] = act
    values = {**found, **{k: v for k, v in given.items() if v}}
    missing = [k for k in ("doc_id", "title", "edition") if k not in values]
    if missing:
        raise ValueError(f"{source_file}: не определены {', '.join(missing)} — задайте в манифесте")
    return DocumentInfo(doc_type="prof_standard", source_file=source_file, **values)


def _after(lines: list[str], marker: str) -> str | None:
    for i, line in enumerate(lines[:-1]):
        if line == marker:
            return lines[i + 1]
    return None


def _blocks(lines: list[str]) -> list[_Block]:
    """Раздел III -> блоки обобщённых и трудовых функций с полями по подписям."""
    blocks: list[_Block] = []
    label: str | None = None
    for line in lines:
        if _END.match(line):
            break
        if match := _GENERALIZED.match(line):
            blocks.append(_Block(f"3.{match.group(1)}", generalized=True))
            label = None
            continue
        if match := _FUNCTION.match(line):
            blocks.append(_Block(f"3.{match.group(1)}.{match.group(2)}", generalized=False))
            label = None
            continue
        if not blocks or line in _SKIP:
            continue
        fields = blocks[-1].fields
        if label == "classifiers":  # таблица ОКЗ/ЕКС/ОКСО — до конца блока, свои «Код»
            fields["classifiers"].append(line)
        elif line in _LABELS:
            label = _LABELS[line]
            fields.setdefault(label, [])
        elif label is not None:
            fields[label].append(line)
    return blocks


def _general(lines: list[str], info: DocumentInfo, code: str, blocks: list[_Block]) -> Fragment:
    body: list[str] = []
    if activity := _after(lines, "I. Общие сведения"):
        activity = activity.removesuffix(code).strip()  # в PDF код стоит в той же строке
        body.append(f"Вид профессиональной деятельности: {activity}")
    if goal := _goal(lines):
        body.append(f"Основная цель вида профессиональной деятельности: {goal}")
    generalized = [
        f"{_function_code(b.value('code'))} {b.value('name')} (уровень {b.value('level')})"
        for b in blocks
        if b.generalized
    ]
    if generalized:
        body += ["Обобщенные трудовые функции:", *generalized]
    return Fragment(
        id=f"{info.doc_id}:general",
        doc_type="prof_standard",
        doc_id=info.doc_id,
        edition=info.edition,
        kind="prof_standard",
        number="1",
        path=f"{info.title} › I. Общие сведения",
        text="\n".join(body) or info.title,
        codes=[code],
    )


def _goal(lines: list[str]) -> str:
    """Строки после «Основная цель…:» до «Группа занятий:» (в PDF цель переносится)."""
    label = "Основная цель вида профессиональной деятельности:"
    if label not in lines:
        return ""
    goal: list[str] = []
    for line in lines[lines.index(label) + 1 :]:
        if line.startswith(("Группа занятий", "II.", "III.")):
            break
        goal.append(line)
    return " ".join(goal)


def _numeric_date(text: str) -> str:
    """«20 июля 2022» -> «20.07.2022»; «20.07.2022» — как есть."""
    if match := re.fullmatch(r"(\d{1,2})\s+(\S+)\s+(\d{4})", text.strip()):
        day, month, year = match.groups()
        return f"{int(day):02d}.{_MONTHS.split().index(month) + 1:02d}.{year}"
    return text


def _function_code(raw: str) -> str:
    return raw.split()[0].translate(_LATIN) if raw else ""


def _okso(values: list[str]) -> list[str]:
    """Коды и названия ОКСО из таблицы дополнительных характеристик."""
    out: list[str] = []
    inside = False
    pending: str | None = None
    for value in values:
        if match := _CLASSIFIER.match(value):
            inside, pending = match.group(1) == "ОКСО", None
            continue
        if not inside:
            continue
        if pending is None and re.fullmatch(r"\d\.\d{2}\.\d{2}\.\d{2}", value):
            pending = value
        elif pending is not None:
            out.append(f"{pending} {value}")
            pending = None
    return out


# PDF


_MARKER = re.compile(r"^3\.\d+\.(?:\d+\.)?\s*(?:Обобщенная трудовая|Трудовая) функция")
_TOC = re.compile(r"\.{5,}")  # строка оглавления с отточием
_CODE_IN_LINE = re.compile(r"(\d{2}\.\d{3})$")


def card_from_pdf(path: Path) -> str:
    """PDF профстандарта с текстовым слоем -> текст в формате карточки (см. начало модуля).

    Таблицы (`pdfplumber.find_tables`) дают подписи и значения по ячейкам: «Трудовые
    действия» и каждое действие — отдельной строкой. Шапка функции («Наименование … Код
    A/01.3 … Уровень … 3») набрана без рамок в три колонки, строки колонок перемешаны —
    её поля собираются по координатам слов. В разделе III, кроме шапок и таблиц, берутся
    только «Дополнительные характеристики» и начало раздела IV: остальное — обрывки шапок
    таблиц («оригинал профессионального»). Скан без текста — `NoTextLayerError`.
    """
    import pdfplumber

    out: list[str] = []
    in_functions = toc = False
    pending_head: list[dict] = []  # слова шапки функции, перенесённой на следующую страницу
    last_row: dict[int, int] = {}  # колонка -> индекс строки в out для последней строки таблицы
    with pdfplumber.open(path) as pdf:
        if not any(page.chars for page in pdf.pages):
            raise NoTextLayerError(f"{path.name}: нет текстового слоя, нужен OCR")
        for page in pdf.pages:
            tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
            boxes = [t.bbox for t in tables]

            def outside(obj: dict, boxes: list = boxes) -> bool:
                cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
                return not any(x0 <= cx <= x1 and t <= cy <= b for x0, t, x1, b in boxes)

            free = page.filter(outside)
            words = free.extract_words()
            events = [(ln["top"], 0, ln) for ln in free.extract_text_lines(strip=True)]
            events += [(t.bbox[1], 1, t) for t in tables]
            skip_until = -1.0
            if pending_head:
                end = tables[0].bbox[1] if tables else page.height
                shifted = [{**w, "top": w["top"] + page.height} for w in words if w["top"] < end]
                out += _function_head(pending_head + shifted)
                pending_head, skip_until = [], end
            first_row = True
            events.sort(key=lambda e: (e[0], e[1]))
            for top, kind, obj in events:
                if kind == 1:
                    for row in obj.extract():
                        cells = {i: c for i, c in enumerate(_cell(c) for c in row) if c}
                        if first_row and _continues(cells, last_row):
                            for i, c in cells.items():  # строка таблицы разорвана страницей
                                out[last_row[i]] += " " + c
                        elif cells and not cells[min(cells)].startswith("Оригинал"):
                            last_row = {}
                            for i, c in cells.items():
                                last_row[i] = len(out)
                                out.append(c)
                        first_row = False
                    continue
                first_row = False
                text = obj["text"].strip()
                if text == "Содержание":  # оглавление — до заголовка «I. Общие сведения»
                    toc = True
                if toc and text != "I. Общие сведения":
                    continue
                toc = False
                if top < skip_until or _TOC.search(text):
                    continue
                if _MARKER.match(text):
                    in_functions = True
                    out.append(text)
                    start = obj["bottom"]
                    if not _balanced(text):  # название ОТФ перенеслось на следующие строки
                        quoted = [text]
                        for line in (e[2] for e in events if e[1] == 0 and e[0] > top):
                            quoted.append(line["text"].strip())
                            start = line["bottom"]
                            if _balanced(" ".join(quoted)):
                                break
                        out[-1], quoted = " ".join(quoted), []
                    end = next((t.bbox[1] for t in tables if t.bbox[1] > start), None)
                    head = [w for w in words if start < w["top"] < (end or page.height)]
                    if end is None:  # шапка может продолжиться на следующей странице
                        pending_head, skip_until = head, page.height
                    else:
                        out += _function_head(head)
                        skip_until = end
                    continue
                if (
                    in_functions
                    and text != "Дополнительные характеристики"
                    and not _END.match(text)
                ):
                    continue
                out.append(text)
    return "\n".join(_card_header(out) + out)


def _continues(cells: dict[int, str], last_row: dict[int, int]) -> bool:
    """Первая строка таблицы на странице — продолжение строки с прошлой страницы: все её
    ячейки начинаются со строчной буквы и есть в прошлой строке."""
    return bool(cells) and all(i in last_row and c[0].islower() for i, c in cells.items())


def _balanced(text: str) -> bool:
    return text.count("«") <= text.count("»")


def _cell(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _function_head(words: list[dict]) -> list[str]:
    """Шапка функции по координатам: название — между «Наименование» и «Код», код — после
    «Код», уровень — правее подписи «Уровень … квалификации»."""
    origin = next((w["top"] for w in words if w["text"] == "Происхождение"), None)
    if origin is not None:  # строка «Происхождение обобщенной …» — уже не шапка
        words = [w for w in words if w["top"] < origin - 2]
    label = next((w for w in words if w["text"] == "Наименование"), None)
    code_label = next((w for w in words if w["text"] == "Код"), None)
    level_label = next((w for w in words if w["text"] == "Уровень"), None)
    if label is None or code_label is None:
        return []
    level_x = max(
        (w["x1"] for w in words if w["text"] in ("Уровень", "квалификации", "(подуровень)")),
        default=code_label["x1"],
    )
    by_line = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    name = [w["text"] for w in by_line if label["x1"] < w["x0"] and w["x1"] < code_label["x0"]]
    right = level_label["x0"] if level_label else level_x
    code = [w["text"] for w in by_line if code_label["x1"] < w["x0"] and w["x1"] < right]
    level = [w["text"] for w in by_line if w["x0"] > level_x and w["text"][:1].isdigit()]
    if not name or not code:
        return []
    return ["Наименование", " ".join(name), "Код", code[0], "Уровень квалификации", *level[:1]]


def _card_header(lines: list[str]) -> list[str]:
    """«Профстандарт: 06.001» и название — как в начале карточки на сайте."""
    code = next((m.group(1) for ln in lines if (m := _CODE_IN_LINE.search(ln))), None)
    name = _after(lines, "ПРОФЕССИОНАЛЬНЫЙ СТАНДАРТ")
    if code is None or name is None:
        return []
    return [f"Профстандарт: {code}", name]
