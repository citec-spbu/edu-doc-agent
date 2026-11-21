"""Командная строка: `python -m edu_doc_agent.extraction файл.docx [--llm]` печатает JSON."""

import argparse
import sys
from pathlib import Path

from edu_doc_agent.extraction.curriculum import extract_plan_file
from edu_doc_agent.extraction.rpd import extract_rpd_file
from edu_doc_agent.llm import OpenAICompatibleClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Извлечение сущностей из РПД/РПП или учебного плана (DOCX, XLSX, Markdown)"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--type", choices=("rpd", "plan"), default="rpd", help="Тип документа")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="добавить смысловые поля через LLM (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL в .env)",
    )
    args = parser.parse_args(argv)
    if args.type == "plan":
        if args.llm:
            parser.error("для учебного плана --llm не поддерживается")
        result = extract_plan_file(args.path)
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
        return 0
    client = OpenAICompatibleClient.from_settings() if args.llm else None
    result = extract_rpd_file(args.path, llm=client)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
