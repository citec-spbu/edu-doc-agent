"""Коды компетенций: каноническая форма, общая для всех модулей (ADR-0005, п. 4).

Нормативы при загрузке в RAG и данные программы при извлечении приводятся к одному виду,
иначе `ПКП-2-ППК-Р2` из ЦКМВ и `ПКП-ППК-Р2` из плана, `УК‑1` из ФГОС и `УК-1` из РПД не
совпадут. Модуль без зависимостей, чтобы его можно было импортировать откуда угодно.

Один и тот же код пишут в двух «диалектах»:

- канонический (учебный план, характеристика, РПД): `ПКП-ППК-Р2`, `ПКП-МО-1`, `ПКП-УНК-03`;
- нумерованный (ЦКМВ, матрица компетенций): `ПКП-2-ППК-Р2`, `ПКП-19-МО-1`, `УНК-03`, где число
  после `ПКП-` — сквозной номер, а у универсальных надпрофессиональных нет префикса `ПКП-`.

Хранимые коды не меняем; `code_key` и `canonical_code` нужны только для сопоставления.
"""

import re

_LEADING_ZEROS_RE = re.compile(r"(?<=[-.])0+(?=[0-9])")


def code_key(code: str) -> str:
    """Ключ для сопоставления кодов: «ПКП-УНК-03.01» и «ПКП-УНК-3.1» дают один ключ.

    Хранимые коды не меняем (см. выше); ключ нужен только для поиска соответствий.
    """
    return _LEADING_ZEROS_RE.sub("", " ".join(code.split()).upper())


_NUMBER_INFIX_RE = re.compile(r"^ПКП-\d+-(?=[А-ЯЁ])")
_BARE_UNK_RE = re.compile(r"^УНК-")
_OUTCOME_TAIL_RE = re.compile(r"^([ЗУВ3])-(\d+)\.?$")


def _canonical_token(token: str) -> str:
    token = token.strip().rstrip(".")
    token = _NUMBER_INFIX_RE.sub("ПКП-", token.upper())
    token = _BARE_UNK_RE.sub("ПКП-УНК-", token)
    return code_key(token)


def canonical_code(code: str) -> str:
    """Единый вид кода: без сквозного номера, с `ПКП-` у УНК, без ведущих нулей.

    Работает для компетенции (`ПКП-2-ППК-Р2`), индикатора (`УНК-01.01`) и результата
    (`ПКП-МО-1.1 З-1`, в том числе с опечаткой «3-1»); результат остаётся «код буква-номер».
    """
    parts = clean_dashes(code).split()
    if not parts:
        return ""
    head = _canonical_token(parts[0])
    if len(parts) == 2 and (m := _OUTCOME_TAIL_RE.match(parts[1].upper())):
        letter = "З" if m.group(1) == "3" else m.group(1)
        return f"{head} {letter}-{m.group(2)}"
    return head if len(parts) == 1 else " ".join([head, *parts[1:]])


def clean_dashes(text: str) -> str:
    return text.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"})).replace("\xa0", " ").strip()


def code_dialect(code: str) -> str:
    """`numbered`, если в коде есть сквозной номер или нет `ПКП-` у УНК; иначе `canonical`."""
    token = clean_dashes(code).split()[0].upper() if code.strip() else ""
    return "numbered" if _NUMBER_INFIX_RE.match(token) or _BARE_UNK_RE.match(token) else "canonical"
