"""Контрольная разметка известного XLSX-плана через OOXML, без парсера и экстрактора.

Это привязанный к одному исходнику reference reader, а не универсальный извлекатель.
Номера разделов, столбцы и правила области действия контекста сверены с исходником.
Сгенерированный эталон остаётся в data/evaluation; независимой проверки человеком нет.
"""

import hashlib
import io
import posixpath
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET
from zipfile import ZipFile

PLAN_SHA256 = "c564673063c1a866be20d1c6b8a9eac4743dca5db764073abcf31b51d20dcec0"
_S = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _col(text: str) -> int:
    number = 0
    for c in text:
        number = number * 26 + ord(c) - ord("A") + 1
    return number


def _position(address: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", address)
    if match is None:
        raise ValueError(f"Неизвестный адрес ячейки: {address}")
    return int(match[2]), _col(match[1])


@dataclass
class XmlSheet:
    name: str
    state: str
    cells: dict[tuple[int, int], object]
    merged: dict[tuple[int, int], tuple[int, int]]
    formulas: dict[tuple[int, int], str]
    errors: dict[tuple[int, int], str]

    def value(self, row: int, col: int, *, inherited: bool = False) -> object:
        position = (row, col)
        if inherited:
            position = self.merged.get(position, position)
        return self.cells.get(position)


def read_xlsx_xml(source: bytes) -> list[XmlSheet]:
    """Только ZIP/XML стандартной библиотеки; читаются сохранённые значения формул."""
    with ZipFile(io.BytesIO(source)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.findall(".//s:t", _S)) for si in root]
        relations = {
            rel.attrib["Id"]: posixpath.normpath(posixpath.join("xl", rel.attrib["Target"])).lstrip(
                "/"
            )
            for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = []
        for meta in ET.fromstring(z.read("xl/workbook.xml")).findall("s:sheets/s:sheet", _S):
            root = ET.fromstring(z.read(relations[meta.attrib[f"{{{_REL}}}id"]]))
            cells, formulas, errors, merged = {}, {}, {}, {}
            for cell in root.findall("s:sheetData/s:row/s:c", _S):
                position = _position(cell.attrib["r"])
                kind = cell.get("t", "n")
                value = cell.findtext("s:v", None, _S)
                if kind == "s" and value is not None:
                    value = shared[int(value)]
                elif kind == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall("s:is//s:t", _S))
                elif value is not None and kind not in {"str", "e", "d"}:
                    value = float(value)
                if value is not None:
                    cells[position] = value
                formula = cell.find("s:f", _S)
                if formula is not None:
                    formulas[position] = formula.text or ""
                if kind == "e":
                    errors[position] = str(value)
            for merge in root.findall("s:mergeCells/s:mergeCell", _S):
                start, end = map(_position, merge.attrib["ref"].split(":"))
                # Пустые объединения не участвуют в семантической разметке.
                if start not in cells:
                    continue
                for row in range(start[0], end[0] + 1):
                    for col in range(start[1], end[1] + 1):
                        merged[row, col] = start
            sheets.append(
                XmlSheet(
                    meta.attrib["name"],
                    meta.get("state", "visible"),
                    cells,
                    merged,
                    formulas,
                    errors,
                )
            )
    return sheets


def build_plan_reference(source: bytes) -> dict:
    """Фиксированные области известной версии; другой файл требует новой разметки."""
    digest = hashlib.sha256(source).hexdigest()
    if digest != PLAN_SHA256:
        raise ValueError("Для этой версии XLSX-плана нет проверенной разметки областей")
    sheet = read_xlsx_xml(source)[0]
    value = sheet.value
    # Область раздела 2: 86:705. Запись имеет название в D и числовую нагрузку F:S.
    # Это включает все 20 дисциплин без кода и исключает шапки выбора/переводы.
    entry_rows = [
        row
        for row in range(89, 706)
        if value(row, 4) is not None
        and any(isinstance(value(row, col), (int, float)) for col in range(6, 20))
    ]
    events: dict[str, list[tuple[int, object]]] = {
        name: [] for name in ("year", "semester", "part", "group", "profile", "trajectory")
    }
    for row in range(86, 706):
        label = _text(value(row, 1))
        if re.fullmatch(r"\d+ год обучения", label):
            events["year"].append((row, int(label.split()[0])))
        elif re.fullmatch(r"С\d+\. Семестр \d+", label):
            events["semester"].append((row, int(label.split()[-1])))
        elif label in {
            "Базовая часть периода обучения",
            "Вариативная часть периода обучения",
            "Факультативные занятия",
        }:
            events["part"].append((row, label))
        elif label == "Блок(и) дисциплин":
            events["group"].append((row, None))
        elif label.startswith("Блок дисциплин "):
            events["group"].append((row, label))
        elif label.startswith("Профиль "):
            events["profile"].append((row, label))
        elif label.startswith("Траектория "):
            events["trajectory"].append((row, label))
        if _text(value(row, 4)) == "Дисциплины по выбору:":
            events["group"].append((row, "Блок дисциплин по выбору: " + _text(value(row + 1, 4))))

    def latest(name: str, row: int) -> tuple[int, object]:
        return max((e for e in events[name] if e[0] < row), default=(0, None))

    # Области действия: семестр ограничивает часть, часть ограничивает группу,
    # группа ограничивает профиль, профиль ограничивает траекторию.
    def context(row: int) -> dict:
        result, boundary = {}, 0
        for name in ("year", "semester", "part", "group", "profile", "trajectory"):
            start, item = latest(name, row)
            result[name] = item if start >= boundary else None
            if name != "year":
                boundary = max(boundary, start)
        return result

    contact = {col: _text(value(85, col)).lower() for col in range(6, 15)}
    independent = {col: _text(value(85, col)).lower() for col in range(15, 20)}
    entries = []
    for row in entry_rows:
        label = _text(value(row, 4))
        match = re.match(r"^\[(\d+)\]\s*(.*)$", label)
        entries.append(
            {
                "line": row,
                "block": _text(value(row, 1, inherited=True)) or None,
                "code": match[1] if match else None,
                "name": match[2] if match else label,
                "name_en": _text(value(row + 1, 4)) or None,
                "credits": value(row, 2, inherited=True),
                "competences": [
                    s.strip() for s in _text(value(row, 3, inherited=True)).split(",") if s.strip()
                ],
                "control": [
                    s.strip().lower()
                    for s in _text(value(row, 5, inherited=True)).split(",")
                    if s.strip()
                ],
                "contact": {
                    name: value(row, col)
                    for col, name in contact.items()
                    if isinstance(value(row, col), (int, float))
                },
                "self": {
                    name: value(row, col)
                    for col, name in independent.items()
                    if isinstance(value(row, col), (int, float))
                },
                "active": value(row, 20),
                **context(row),
            }
        )
    return {
        "version": 2,
        "source_sha256": digest,
        "sheet": sheet.name,
        "annotation": "OOXML reference reader; фиксированные области одного файла; "
        "без вывода экстрактора и без независимой проверки человеком",
        "title": {
            "program": _text(value(10, 1)) + " " + _text(value(11, 1)),
            "level": _text(value(12, 6)),
            "direction": _text(value(13, 6)),
            "study_form": _text(value(15, 10)),
            "languages": _text(value(16, 10)),
            "duration": _text(value(17, 10)),
            "registration_number": _text(value(25, 8)),
        },
        "competences": [
            {"line": row, "code": _text(value(row, 1)), "name": _text(value(row, 2))}
            for row in range(33, 82)
        ],
        "entries": entries,
        "final_attestation": [
            {
                "line": 711,
                "block": _text(value(711, 1)),
                "credits": value(711, 2),
                "name": _text(value(711, 3)),
                "name_en": _text(value(712, 3)),
                "competences": [s.strip() for s in _text(value(711, 12)).split(",")],
            }
        ],
        "totals": {"competences": 49, "entries": 222, "final_attestation": 1},
    }
