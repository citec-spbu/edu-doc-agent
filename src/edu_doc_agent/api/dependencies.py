from arq.connections import ArqRedis
from fastapi import Request

from edu_doc_agent.dao import SessionRunDAO


def get_arq_pool(request: Request) -> ArqRedis:
    """Пул подключений к Redis (ARQ), инициализируется в lifespan (main.py)."""
    return request.app.state.redis_pool


def get_session_run_dao() -> SessionRunDAO:
    """DAO для Session и Run. В тестах подменяется через app.dependency_overrides."""
    return SessionRunDAO()
