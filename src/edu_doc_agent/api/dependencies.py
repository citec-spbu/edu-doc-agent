"""
Инъекции зависимостей для FastAPI.
Здесь лежат функции-помощники, которые FastAPI автоматически
выполнит перед тем, как передать запрос в основной роут.
"""

from arq.connections import ArqRedis
from fastapi import Request

from edu_doc_agent.dao import SessionRunDAO


def get_arq_pool(request: Request) -> ArqRedis:
    """
    Достает пул подключений к Redis (ARQ) из глобального состояния приложения.
    Пул инициализируется в lifespan (main.py).

    Использование в роуте:
    def my_endpoint(redis_pool: ArqRedis = Depends(get_arq_pool)):
    """
    return request.app.state.redis_pool


def get_session_run_dao(request: Request) -> SessionRunDAO:
    """
    Возвращает DAO для работы с Session и Run.
    Если в app.state.session_run_dao задан переопределённый экземпляр (например, в тестах),
    использует его; иначе создаёт стандартный SessionRunDAO.
    """
    dao = getattr(request.app.state, "session_run_dao", None)
    if isinstance(dao, SessionRunDAO):
        return dao
    return SessionRunDAO()
