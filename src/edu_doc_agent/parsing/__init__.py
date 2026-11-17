"""Parser Service: документы ОП -> ParsedDocument (список блоков) -> Markdown"""

from pathlib import Path

from edu_doc_agent.parsing.docx_parser import DocxParser
from edu_doc_agent.parsing.models import BaseParser, Block, BlockKind, CellSpan, ParsedDocument
from edu_doc_agent.parsing.pdf_parser import NoTextLayerError, PdfParser
from edu_doc_agent.parsing.xlsx_parser import XlsxParser

__all__ = [
    "BaseParser",
    "Block",
    "BlockKind",
    "CellSpan",
    "NoTextLayerError",
    "ParsedDocument",
    "UnsupportedFormatError",
    "parse_document",
    "supported_suffixes",
]


class UnsupportedFormatError(ValueError):
    """Нет парсера для файла"""


PARSERS: dict[str, BaseParser] = {
    ".docx": DocxParser(),
    ".pdf": PdfParser(),
    ".xlsx": XlsxParser(),
}


def supported_suffixes() -> tuple[str, ...]:
    return tuple(PARSERS)


def parse_document(path: str | Path) -> ParsedDocument:
    path = Path(path)
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise UnsupportedFormatError(
            f"Нет парсера для {path.suffix!r}, доступны: {', '.join(PARSERS)}"
        )
    return parser.parse(path)
