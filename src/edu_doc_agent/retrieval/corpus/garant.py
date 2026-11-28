"""Выгрузки нормативов из системы ГАРАНТ: служебные вставки, отметки об изменениях, реквизиты.

ГАРАНТ вставляет в текст документа свои примечания. В корпус они не идут:

- рамки «С изменениями и дополнениями от: …», «ГАРАНТ: См. …» (парсер читает их таблицами;
  у залитых рамок буквы двух слоёв перемешаны — такие таблицы тоже служебные);
- «ГАРАНТ: См. комментарии к статье 13…», приклеенное к заголовку статьи;
- «См. предыдущую редакцию», «См. текст пункта…» и прочие ссылки «См. …»;
- сноски в конце документа — после черты «──────».

«Информация об изменениях: Часть 5 изменена с 1 сентября 2021 г. - Федеральный закон …» —
не мусор, а `changed_by` фрагмента (`change_note`).
"""

import re
from datetime import date

from edu_doc_agent.parsing.models import Block, BlockKind, ParsedDocument
from edu_doc_agent.retrieval.corpus.models import ChangeNote

CHANGE_HEADER = "Информация об изменениях"
EDITION_HEADER = "С изменениями и дополнениями от"
FOOTNOTE_RULE = re.compile(r"^─{5,}")
_GARANT_TAIL = re.compile(r"\s*(?:ГАРАНТ|Информация об изменениях):.*$", re.S)
_SEE = re.compile(r"^См\.\s")
_SINCE = re.compile(r"\b(с \d{1,2} [а-я]+ \d{4} г\.)")
_ACT = re.compile(r"^(?:Федеральн|Приказ|Постановлени|Указ|Распоряжени)")
_MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря"
_RU_DATE = re.compile(rf"(\d{{1,2}})\s+({_MONTHS.replace(' ', '|')})\s+(\d{{4}})")
_EXPIRY = re.compile(
    rf"документ признан утратившим силу с (\d{{1,2}} (?:{_MONTHS.replace(' ', '|')}) \d{{4}})"
)
_NUMBER = re.compile(r"(?:N|№)\s*(\d+(?:/\d+)?(?:-ФЗ)?)")
_ACT_DATE = re.compile(rf"от\s+(\d{{1,2}}\s+(?:{_MONTHS.replace(' ', '|')})\s+\d{{4}})")


def is_service_table(block: Block) -> bool:
    """Таблица-рамка ГАРАНТа, а не таблица документа.

    Рамки — одна колонка текста (вторая пустая): «С изменениями…», «ГАРАНТ:», «Информация об
    изменениях», а также их перемешанные копии.
    """
    if block.kind != BlockKind.TABLE:
        return False
    filled = {i for row in block.rows for i, cell in enumerate(row) if cell.strip()}
    return len(filled) <= 1


def clean_text(text: str) -> str:
    """Абзац без приклеенных «ГАРАНТ: См. …» и «Информация об изменениях: См. …»."""
    return _GARANT_TAIL.sub("", text).strip()


def is_service_paragraph(text: str) -> bool:
    return not text or text.startswith(("ГАРАНТ:", CHANGE_HEADER)) or bool(_SEE.match(text))


def change_note(block: Block) -> tuple[str, ChangeNote] | None:
    """«Информация об изменениях: Часть 5 изменена …» -> («Часть 5 изменена …», отметка).

    Первая строка рамки может быть обрезана («… с 1 апреля 2024 г. -»): акт тогда ищется в
    следующих строках.
    """
    lines = [row[0].strip() for row in block.rows if row and row[0].strip()]
    if block.kind != BlockKind.TABLE:
        lines = [block.text]
    if not lines or not lines[0].startswith(CHANGE_HEADER):
        return None
    text = lines[0].removeprefix(CHANGE_HEADER).lstrip(": ").strip()
    target, _, act = text.partition(" - ")
    act = act.split(" См.")[0].strip()
    if not act:
        acts = [ln for ln in lines[1:] if _ACT.match(ln)]
        act = max(acts, key=len) if acts else ""
    since = _SINCE.search(target)
    note = ChangeNote(since=since.group(1) if since else None, act=act or target)
    return target, note


def act_requisites(parsed: ParsedDocument) -> dict[str, str]:
    """Название, номер, редакция и срок действия по шапке выгрузки.

    Название — первый абзац и его продолжения (незакрытые кавычки, строчная буква или «(» в
    начале следующего абзаца); редакция — последняя дата из рамки «С изменениями и дополнениями
    от:» (у документа без изменений — дата самого акта); если ГАРАНТ отмечает, что документ
    утрачивает силу, дата дописывается к `approval_act`.
    """
    texts = [b.text for b in parsed.blocks[:10] if b.kind != BlockKind.TABLE and b.text]
    head = texts[:1]
    for text in texts[1:]:
        if " ".join(head).count('"') % 2 == 0 and not text.startswith("(") and text[:1].isupper():
            break
        head.append(text)
    title = re.sub(r"\s*\(с изменениями и дополнениями\).*$", "", " ".join(head), flags=re.S)
    title = title.strip()
    found: dict[str, str] = {}
    if title:
        found["title"] = found["approval_act"] = title
    number = _NUMBER.search(title)
    signed = _ACT_DATE.search(title)
    frames = [
        block
        for block in parsed.blocks[:10]
        if block.kind == BlockKind.TABLE
        and block.rows
        and block.rows[0][0].startswith(EDITION_HEADER)
    ]
    changes = [d for block in frames for row in block.rows[1:] for d in _dates(row[0])]
    if expiry := _expiry(parsed.blocks[:10]):
        found["approval_act"] = f"{title}; утрачивает силу с {expiry:%d.%m.%Y}"
    if number:
        if changes:
            found["edition"] = f"{number.group(1)} ред. от {max(changes):%d.%m.%Y}"
        elif signed:
            found["edition"] = f"{number.group(1)} от {_dates(signed.group(1))[0]:%d.%m.%Y}"
    return found


def _expiry(blocks: list[Block]) -> date | None:
    """«ГАРАНТ: … настоящий документ признан утратившим силу с 1 сентября 2027 г.»"""
    texts = [b.text for b in blocks] + [row[0] for b in blocks for row in b.rows if row]
    for text in texts:
        if match := _EXPIRY.search(text):
            return _dates(match.group(1))[0]
    return None


def _dates(text: str) -> list[date]:
    months = _MONTHS.split()
    out = []
    for day, month, year in _RU_DATE.findall(text):
        try:
            out.append(date(int(year), months.index(month) + 1, int(day)))
        except ValueError:
            continue
    return out
