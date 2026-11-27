"""Чат с графом в терминале: ручная проверка, пока воркер не вызывает граф.

    uv run python -m edu_doc_agent.orchestration.cli [файлы...] [--out ПАПКА] [--action ВИД]

Тот же вход, что у воркера (`run_turn`), но состояние — в памяти процесса: PostgreSQL и Redis
не нужны, нужен ключ LLM в `.env`. На вопрос графа можно ответить номером варианта или своим
текстом. Файлы ответа (протокол проверки, исправленные документы) сохраняются в `--out`.
Команды: `/file путь` — добавить документ, `/exit` — выйти.
"""

import argparse
import asyncio
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from edu_doc_agent.orchestration.deps import GraphDeps, default_deps
from edu_doc_agent.orchestration.graph import build_graph
from edu_doc_agent.orchestration.session import Turn, run_turn


def render(turn: Turn) -> tuple[str, list[str]]:
    """Текст для терминала и варианты ответа (если граф задал вопрос)."""
    if turn.question is None:
        return turn.reply or "(пустой ответ)", []
    options = list(turn.question.get("options") or [])
    lines = [f"[вопрос] {turn.question['question']}"]
    lines += [f"  {i}. {option}" for i, option in enumerate(options, start=1)]
    if options:
        lines.append("  (номер варианта или свой ответ)")
    return "\n".join(lines), options


def answer_text(text: str, options: list[str]) -> str:
    """Номер варианта — сам вариант; остальное — свой ответ пользователя."""
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    return text


async def save_files(deps: GraphDeps, turn: Turn, out: Path) -> list[Path]:
    saved = []
    if deps.storage is None:
        return saved
    for file in turn.files:
        path = out / file["filename"]
        path.write_bytes(await deps.storage.read(file["storage_key"]))
        saved.append(path)
    return saved


async def chat(files: list[Path], out: Path, action: str | None) -> None:
    deps = default_deps()
    graph = build_graph(InMemorySaver(), deps)
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    paths = [str(f.resolve()) for f in files]
    out.mkdir(parents=True, exist_ok=True)
    print(f"Сессия {session_id}. Документов: {len(paths)}. Файлы ответа — в {out}/.")
    options: list[str] = []
    while True:
        try:
            text = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            return
        if text in ("", "/exit"):
            if text:
                return
            continue
        if text.startswith("/file "):
            path = Path(text.removeprefix("/file ").strip())
            if path.is_file():
                paths.append(str(path.resolve()))
                print(f"Добавлен: {path.name} (обработается со следующим сообщением)")
            else:
                print(f"Нет файла: {path}")
            continue
        message = answer_text(text, options)
        turn = await run_turn(
            graph, session_id, uuid.uuid4().hex, message, file_paths=paths, action=action
        )
        reply, options = render(turn)
        print(reply)
        for path in await save_files(deps, turn, out):
            print(f"Файл: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Чат с графом оркестрации в терминале")
    parser.add_argument("files", nargs="*", type=Path, help="документы программы")
    parser.add_argument(
        "--out", type=Path, default=Path("storage/chat-output"), help="папка файлов ответа"
    )
    parser.add_argument(
        "--action",
        choices=["check", "qa", "generate"],
        help="вид запроса без оркестратора (по умолчанию оркестратор решает сам)",
    )
    args = parser.parse_args()
    asyncio.run(chat(args.files, args.out, args.action))


if __name__ == "__main__":
    main()
