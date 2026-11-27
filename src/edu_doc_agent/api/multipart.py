"""Ограничение файловых частей до их записи multipart-парсером."""

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from python_multipart.multipart import parse_options_header
from starlette.formparsers import MultiPartException, MultiPartParser

from edu_doc_agent.config import get_settings


class BoundedMultiPartParser(MultiPartParser):
    def __init__(self, *args, max_file_bytes: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_file_bytes = max_file_bytes
        self._file_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._file_bytes += end - start
            if self._file_bytes > self.max_file_bytes:
                # До очереди записи Starlette; parse() закроет все временные
                # файлы при исключении и больше не запросит следующий chunk.
                raise HTTPException(
                    status_code=413,
                    detail=f"Размер файла превышает допустимый лимит ({self.max_file_bytes} байт).",
                )
        super().on_part_data(data, start, end)


class BoundedMultipartRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def receive_form(request: Request):
            content_type, _ = parse_options_header(request.headers.get("content-type", ""))
            if request.method != "POST" or content_type != b"multipart/form-data":
                return await handler(request)

            parser = BoundedMultiPartParser(
                request.headers,
                request.stream(),
                max_file_bytes=get_settings().max_upload_file_bytes,
            )
            try:
                form = await parser.parse()
            except MultiPartException as exc:
                raise HTTPException(status_code=400, detail=exc.message) from exc

            # FastAPI использует уже разобранную форму и сохраняет обычную
            # валидацию File/Form и OpenAPI-схему для Swagger UI.
            request._form = form
            try:
                return await handler(request)
            finally:
                await form.close()

        return receive_form
