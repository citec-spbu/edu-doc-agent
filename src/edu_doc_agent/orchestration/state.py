"""Состояние графа оркестрации (ADR-0003).

В состоянии лежат только ID документов, замечаний и изменений, сами данные берутся из
хранилища (`stores.py`). Поля делятся на три группы:

- вход запуска: все ключи `GraphInput` передаются при каждом запуске, поэтому значения
  прошлого запуска не просачиваются в новый;
- данные сессии: копятся между запусками в одной сессии (`thread_id`);
- поля запуска: узел `start` сбрасывает их в начале каждого запуска.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# вид запроса: что запускать
RequestKind = Literal["question", "check", "change", "check_and_fix"]
# маршрут, заданный интерфейсом без LLM
Action = Literal["check", "generate", "qa"]
QuestionType = Literal["clarification", "missing_data", "confirmation"]


class Scope(TypedDict):
    """Объём работы подграфов: пустой список означает «все»."""

    courses: list[str]
    rule_sets: list[str]
    finding_ids: list[str]
    change_goal: str | None


class PendingQuestion(TypedDict):
    """Вопрос пользователю: узел `ask_user` ставит граф на паузу и передаёт ответ дальше."""

    type: QuestionType
    question: str
    options: list[str]
    resume_to: str  # узел, который получает ответ
    call_id: str | None  # вызов ask_user агента: ответ — его результат в `work`
    draft: str | None  # текст для документа, предложенный агентом в вопросе
    change_id: str | None


class CheckSummary(TypedDict):
    """Итог проверки в этом запуске."""

    rules: int  # правил в проверке
    rule_sets: list[str]  # наборы правил из Scope; пустой — все
    passed: int  # пар «правило, единица проверки» без замечаний
    finding_ids: list[str]  # нарушения и not_checked этого запуска
    missing: list[str]  # почему правила ФГОС не применялись: direction_code, fgos_rules, …
    direction_code: str | None  # код направления проверки
    unknown_courses: list[str]  # дисциплины из запроса, которых нет среди документов
    catalog_version: str
    editions: list[str]  # редакции нормативов в правилах проверки


class OutputFile(TypedDict):
    """Файл, который запуск сохранил в хранилище сессии: протокол проверки и т. п."""

    storage_key: str  # ключ FileStorage (`dao/storage.py`)
    filename: str
    kind: str  # «check_protocol», «corrected_document»
    derived_from: str | None  # у исправленного документа — имя исходного файла


class GraphInput(TypedDict):
    run_id: str
    message: str
    action: Action | None  # если задано, маршрут без LLM
    file_paths: list[str]  # файлы сессии в /storage; уже обработанные пропускаются


class GraphState(GraphInput):
    # данные сессии
    document_ids: list[str]
    ingested_paths: list[str]  # обработанные файлы
    finding_ids: list[str]
    change_ids: list[str]
    direction_code: str | None  # "" — пользователь отказался указать код, ФГОС не проверяется
    # диалог с пользователем: реплики без работы агентов (`history.py`)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str | None  # краткое содержание реплик, удалённых из `messages`
    # поля запуска
    kind: RequestKind | None
    scope: Scope | None
    pending: PendingQuestion | None
    answer: str | None  # последний ответ пользователя на вопрос графа
    # работа агента, который ждёт ответа пользователя или подтверждения своего черновика:
    # сообщения с вызовами инструментов; агент продолжает с ними после ответа
    work: list[AnyMessage]
    work_agent: str | None  # чья это работа (агент, который ждёт ответа): «generator»
    ingest_errors: list[str]
    check_summary: CheckSummary | None
    change_id: str | None  # изменение этого запуска
    change_explanation: str | None  # пояснение генератора к черновику по вопросу пользователя
    reply: str | None
    files: list[OutputFile]  # файлы ответа этого запуска
    run_change_ids: list[str]  # изменения запуска; при «проверке с исправлением» — по документу
    # «проверка с исправлением»: исправимые замечания по документам, ещё не обработанные;
    # None — очередь ещё не составлена
    fix_queue: list[list[str]] | None
    fix_targets: list[str]  # замечания документа, который исправляется сейчас


class GraphOutput(TypedDict):
    run_id: str
    kind: RequestKind | None
    reply: str | None
    files: list[OutputFile]


def make_input(
    run_id: str,
    message: str,
    *,
    action: Action | None = None,
    file_paths: list[str] | None = None,
) -> GraphInput:
    """Вход запуска со всеми ключами."""
    return {
        "run_id": run_id,
        "message": message,
        "action": action,
        "file_paths": file_paths or [],
    }


def empty_scope() -> Scope:
    return {"courses": [], "rule_sets": [], "finding_ids": [], "change_goal": None}


def question(
    type: QuestionType,
    text: str,
    resume_to: str,
    *,
    options: list[str] | None = None,
    call_id: str | None = None,
    draft: str | None = None,
    change_id: str | None = None,
) -> PendingQuestion:
    return {
        "type": type,
        "question": text,
        "options": options or [],
        "resume_to": resume_to,
        "call_id": call_id,
        "draft": draft,
        "change_id": change_id,
    }
