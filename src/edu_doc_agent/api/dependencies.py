"""
Инъекции зависимостей для FastAPI.
Здесь лежат функции-помощники, которые FastAPI автоматически
выполнит перед тем, как передать запрос в основной роут.
"""

from arq.connections import ArqRedis
from fastapi import Request


def get_arq_pool(request: Request) -> ArqRedis:
    """
    Достает пул подключений к Redis (ARQ) из глобального состояния приложения.
    Пул инициализируется в lifespan (main.py).

    Использование в роуте:
    def my_endpoint(redis_pool: ArqRedis = Depends(get_arq_pool)):
    """
    return request.app.state.redis_pool
