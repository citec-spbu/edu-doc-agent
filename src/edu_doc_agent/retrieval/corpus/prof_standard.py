"""Профессиональный стандарт из текста карточки на сайте -> фрагменты корпуса (#50).

Вход — текст страницы профстандарта, сохранённый как `.txt`: подписи полей («Наименование»,
«Код», «Трудовые действия», «Необходимые умения»…) стоят на своих строках, значения — на
следующих. Разбирается раздел III; функциональная карта (раздел II) повторяет его кратко и не
нужна, раздел IV (разработчики) отбрасывается.

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

from edu_doc_agent.retrieval.corpus.chunker import split_long
from edu_doc_agent.retrieval.corpus.codes import PROF_STANDARD_CODE
from edu_doc_agent.retrieval.corpus.models import CorpusDocument, DocumentInfo, Fragment

# кириллические буквы, похожие на латинские, в кодах функций: «А/01.3» -> «A/01.3»
_LATIN = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")
_HEADER = re.compile(r"^Профстандарт:\s*(\d{2}\.\d{3})")
_PERIOD = re.compile(r"действует с (\d{2}\.\d{2}\.\d{4}) по (\d{2}\.\d{2}\.\d{4})")
_ORDER = re.compile(r"от\s+(\d{2}\.\d{2}\.\d{4})\s*(?:№|N)\s*(\d+н)")
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
            values = [v for v in block.fields.get(key, []) if v != "-"]
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
        date, number = order.groups()
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
        body.append(f"Вид профессиональной деятельности: {activity}")
    goal = _after(lines, "Основная цель вида профессиональной деятельности:")
    if goal:
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
