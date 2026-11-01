"""Командная строка: `python -m edu_doc_agent.extraction файл.md [--llm]` печатает JSON."""

import argparse
import sys
from pathlib import Path

from edu_doc_agent.extraction.rpd import extract_rpd_file
from edu_doc_agent.llm import OpenAICompatibleClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Извлечение сущностей из md-файла РПД/РПП")
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="добавить смысловые поля через LLM (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL в .env)",
    )
    args = parser.parse_args(argv)
    client = OpenAICompatibleClient.from_settings() if args.llm else None
    result = extract_rpd_file(args.path, llm=client)
    sys.stdout.write(result.model_dump_json(indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
