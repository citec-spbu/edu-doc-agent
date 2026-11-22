"""Контракт локального файлового хранилища и регрессии его адресации."""

import asyncio
import errno
import os
import stat
import threading
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

import pytest

from edu_doc_agent.dao import (
    FileStorage,
    InvalidStoragePathError,
    LocalFileStorage,
    StorageFileNotFoundError,
)

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_path / "storage")


def test_save_read_and_delete_in_separate_sessions(storage: LocalFileStorage) -> None:
    async def scenario() -> None:
        writer: FileStorage = storage
        reader: FileStorage = LocalFileStorage(storage.base_dir)
        args = {"doc_id": DOCUMENT_ID, "filename": "Матрица.docx"}
        first = await writer.save(session_id="first", **args, content=b"\x00original\xff")
        second = await writer.save(session_id="second", **args, content=b"")

        assert Path(first).parent != Path(second).parent
        assert await reader.read(first) == b"\x00original\xff"
        assert await reader.read(second) == b""
        assert await reader.exists(first)

        assert await reader.delete(first)
        assert not await writer.exists(first)
        assert not await reader.delete(first)
        with pytest.raises(StorageFileNotFoundError):
            await writer.read(first)
        assert await writer.exists(second)

    asyncio.run(scenario())


def test_repeated_save_replaces_contents_for_the_same_key(storage: LocalFileStorage) -> None:
    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        key = await storage.save(**args, content=b"original")
        assert await storage.save(**args, content=b"updated") == key
        assert await storage.read(key) == b"updated"

    asyncio.run(scenario())


def test_open_reader_keeps_complete_old_version_during_replacement(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        original = b"original record\n" * 8192
        replacement = b"replacement record\n" * 16384
        key = await storage.save(**args, content=original)
        loop = asyncio.get_running_loop()
        reader_opened = asyncio.Event()
        resume_reader = threading.Event()
        original_fdopen = os.fdopen

        def pause_after_open(fd: int, mode: str = "r", *, closefd: bool = True) -> BinaryIO:
            if mode == "rb":
                loop.call_soon_threadsafe(reader_opened.set)
                assert resume_reader.wait(timeout=5), "Не получен сигнал продолжить чтение"
            return original_fdopen(fd, mode, closefd=closefd)

        with monkeypatch.context() as patch:
            patch.setattr("edu_doc_agent.dao.storage.os.fdopen", pause_after_open)
            read_task = asyncio.create_task(storage.read(key))
            try:
                await asyncio.wait_for(reader_opened.wait(), timeout=5)
                assert await storage.save(**args, content=replacement) == key
            finally:
                resume_reader.set()

            assert await asyncio.wait_for(read_task, timeout=5) == original

        assert await storage.read(key) == replacement

    asyncio.run(scenario())


def test_new_reader_sees_complete_version_during_publication(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        original = b"original record\n" * 8192
        replacement = b"replacement record\n" * 16384
        key = await storage.save(**args, content=original)
        name = Path(key).name
        loop = asyncio.get_running_loop()
        original_replace = os.replace
        original_unlink = os.unlink
        observed: list[bytes] = []

        def read_current_key() -> None:
            pending = asyncio.run_coroutine_threadsafe(storage.read(key), loop)
            try:
                version = pending.result(timeout=5)
            finally:
                pending.cancel()
            assert version in (original, replacement)
            observed.append(version)

        def observe_replace(
            source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int
        ) -> None:
            read_current_key()
            original_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            read_current_key()

        def observe_unlink(path: Path | str, *, dir_fd: int | None = None) -> None:
            original_unlink(path, dir_fd=dir_fd)
            if Path(path).name == name:
                read_current_key()

        with monkeypatch.context() as patch:
            patch.setattr("edu_doc_agent.dao.storage.os.replace", observe_replace)
            # Проба после unlink ловит разрыв при схеме «удалить, затем установить».
            patch.setattr("edu_doc_agent.dao.storage.os.unlink", observe_unlink)
            assert await storage.save(**args, content=replacement) == key

        assert observed
        assert await storage.read(key) == replacement

    asyncio.run(scenario())


def test_case_distinct_sessions_have_distinct_storage_directories(
    storage: LocalFileStorage,
) -> None:
    async def scenario() -> None:
        args = {"doc_id": DOCUMENT_ID, "filename": "report.docx"}
        first = await storage.save(session_id="Group", **args, content=b"first")
        second = await storage.save(session_id="group", **args, content=b"second")

        assert Path(first).parent.as_posix().casefold() != Path(second).parent.as_posix().casefold()
        assert await storage.read(first) == b"first"
        assert await storage.read(second) == b"second"

    asyncio.run(scenario())


def test_session_id_matching_root_file_can_save(storage: LocalFileStorage) -> None:
    storage.base_dir.mkdir()
    root_file = storage.base_dir / "README.md"
    root_file.write_bytes(b"storage documentation")

    async def scenario() -> None:
        key = await storage.save(
            session_id=root_file.name, doc_id=DOCUMENT_ID, filename="report.docx", content=b"report"
        )
        assert Path(key).parts[0] == "sessions"
        assert await storage.read(key) == b"report"
        for operation in (storage.read, storage.exists, storage.delete):
            with pytest.raises(InvalidStoragePathError):
                await operation(root_file.name)

    asyncio.run(scenario())
    assert root_file.read_bytes() == b"storage documentation"


@pytest.mark.parametrize("mask", [0o077, 0o022], ids=["umask-077", "umask-022"])
def test_save_preserves_private_file_mode(storage: LocalFileStorage, mask: int) -> None:
    previous_umask = os.umask(mask)
    try:
        key = asyncio.run(
            storage.save(
                session_id="synthetic",
                doc_id=DOCUMENT_ID,
                filename="report.docx",
                content=b"report",
            )
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((storage.base_dir / key).stat().st_mode) == 0o600


def test_failed_publication_keeps_original_and_removes_temporary_file(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        key = await storage.save(**args, content=b"original")
        target = storage.resolve_path(key)

        def failed_replace(
            source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int
        ) -> None:
            assert src_dir_fd == dst_dir_fd
            assert Path(source).name == source
            assert destination == target.name
            assert os.path.samestat(os.fstat(dst_dir_fd), target.parent.stat())
            raise OSError("synthetic failure")

        with monkeypatch.context() as patch:
            patch.setattr("edu_doc_agent.dao.storage.os.replace", failed_replace)
            with pytest.raises(OSError, match="synthetic failure"):
                await storage.save(**args, content=b"updated")

        assert await storage.read(key) == b"original"
        assert list(target.parent.iterdir()) == [target]

    asyncio.run(scenario())


def test_read_translates_file_disappearing_during_open(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        key = await storage.save(
            session_id="synthetic", doc_id=DOCUMENT_ID, filename="report.docx", content=b"report"
        )
        target = storage.resolve_path(key)
        original_open = os.open

        def open_after_delete(
            path: Path | str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            if path == target.name:
                assert dir_fd is not None and flags & os.O_NOFOLLOW
                os.unlink(path, dir_fd=dir_fd)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr("edu_doc_agent.dao.storage.os.open", open_after_delete)
        with pytest.raises(StorageFileNotFoundError) as exc:
            await storage.read(key)
        assert isinstance(exc.value.__cause__, FileNotFoundError)
        assert not await storage.exists(key)

    asyncio.run(scenario())


def test_delete_refuses_directories(storage: LocalFileStorage) -> None:
    key = "synthetic/report.docx"
    directory = storage.base_dir / key
    directory.mkdir(parents=True)

    with pytest.raises(InvalidStoragePathError):
        asyncio.run(storage.delete(key))

    assert directory.is_dir()


@pytest.mark.parametrize(
    "unlink_error",
    [
        pytest.param(None, id="unlink-succeeds"),
        pytest.param(OSError(errno.EIO, "synthetic I/O error"), id="unlink-fails"),
    ],
)
def test_cancelled_delete_finishes_before_file_is_saved_again(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch, unlink_error: OSError | None
) -> None:
    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        key = await storage.save(**args, content=b"original")
        name = Path(key).name
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()
        original_unlink = os.unlink

        def paused_unlink(path: Path | str, *, dir_fd: int | None = None) -> None:
            if path == name and dir_fd is not None:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(timeout=5), "Не получен сигнал завершить unlink"
                if unlink_error is not None:
                    raise unlink_error
            original_unlink(path, dir_fd=dir_fd)

        with monkeypatch.context() as patch:
            patch.setattr("edu_doc_agent.dao.storage.os.unlink", paused_unlink)
            task = asyncio.create_task(storage.delete(key))
            try:
                await asyncio.wait_for(entered.wait(), timeout=5)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "Отмена не должна обгонять удаление"
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done(), "Повторная отмена не должна обгонять удаление"
            finally:
                release.set()

            with pytest.raises(asyncio.CancelledError) as exc:
                await asyncio.wait_for(task, timeout=5)

        if unlink_error is None:
            assert exc.value.__cause__ is None
            assert not await storage.exists(key)
        else:
            cause = exc.value.__cause__
            assert isinstance(cause, OSError)
            assert cause.errno == errno.EIO
            assert await storage.read(key) == b"original"
        assert await storage.save(**args, content=b"replacement") == key
        assert await storage.read(key) == b"replacement"

    asyncio.run(scenario())


def test_file_operations_leave_fifo_in_place(storage: LocalFileStorage) -> None:
    key = "synthetic/report.docx"
    fifo = storage.base_dir / key
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo, mode=0o600)

    async def scenario() -> None:
        assert not await storage.exists(key)
        with pytest.raises(StorageFileNotFoundError):
            await storage.read(key)
        assert not await storage.delete(key)

    asyncio.run(scenario())
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


def test_file_operations_handle_regular_file_in_place_of_session_directory(
    storage: LocalFileStorage,
) -> None:
    storage.base_dir.mkdir()
    parent = storage.base_dir / "synthetic"
    parent.write_bytes(b"regular file")

    async def scenario() -> None:
        key = "synthetic/report.docx"
        assert not await storage.exists(key)
        with pytest.raises(StorageFileNotFoundError):
            await storage.read(key)
        assert not await storage.delete(key)

    asyncio.run(scenario())
    assert parent.read_bytes() == b"regular file"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("session_id", "", id="empty-session"),
        pytest.param("session_id", "synthetic\x00", id="nul-in-session"),
        pytest.param("session_id", "synthetic\ud800", id="invalid-utf8-session"),
        pytest.param("doc_id", "folder/doc", id="nested-document"),
        pytest.param("doc_id", "report\n", id="document-newline"),
        pytest.param("filename", "../report.docx", id="relative-name"),
        pytest.param("filename", "folder\\report.docx", id="windows-separator"),
        pytest.param("filename", "report\x00.docx", id="nul-in-name"),
        pytest.param("filename", "report\x7f.docx", id="ascii-del-in-name"),
        pytest.param("filename", "report\ud800.docx", id="invalid-utf8"),
        pytest.param("filename", "  ", id="blank-name"),
    ],
)
def test_save_refuses_invalid_components(storage: LocalFileStorage, field: str, value: str) -> None:
    args = {"session_id": "synthetic", "doc_id": "source", "filename": "report.docx"}
    args[field] = value

    with pytest.raises(InvalidStoragePathError):
        asyncio.run(storage.save(**args, content=b""))

    assert not storage.base_dir.exists()


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/report.docx",
        "first/../second/report.docx",
        "synthetic//report.docx",
        "synthetic\\report.docx",
        "synthetic/report\x00.docx",
        "synthetic/" + "я" * 128,
    ],
    ids=["empty", "absolute", "parent", "empty-segment", "backslash", "nul", "overlong-component"],
)
def test_file_operations_refuse_invalid_keys(storage: LocalFileStorage, key: str) -> None:
    async def scenario() -> None:
        for operation in (storage.read, storage.exists, storage.delete):
            with pytest.raises(InvalidStoragePathError):
                await operation(key)

    asyncio.run(scenario())


def test_save_refuses_session_directory_symlink(storage: LocalFileStorage) -> None:
    async def scenario() -> None:
        args = {"session_id": "first", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        key = await storage.save(**args, content=b"report")
        directory = (storage.base_dir / key).parent
        assert await storage.delete(key)
        directory.rmdir()

        target = storage.base_dir / "second"
        target.mkdir()
        directory.symlink_to(target, target_is_directory=True)

        with pytest.raises(InvalidStoragePathError):
            await storage.save(**args, content=b"report")
        for operation in (storage.read, storage.exists, storage.delete):
            with pytest.raises(InvalidStoragePathError):
                await operation(key)

        assert not list(target.iterdir())

    asyncio.run(scenario())


def test_file_operations_refuse_dangling_file_symlink(storage: LocalFileStorage) -> None:
    directory = storage.base_dir / "synthetic"
    directory.mkdir(parents=True)
    alias = directory / "alias.docx"
    alias.symlink_to(directory / "missing.docx")

    async def scenario() -> None:
        for operation in (storage.read, storage.exists, storage.delete):
            with pytest.raises(InvalidStoragePathError):
                await operation("synthetic/alias.docx")

    asyncio.run(scenario())
    assert alias.is_symlink()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            ("revision_final", "report.docx"), ("revision", "final_report.docx"), id="ids"
        ),
        pytest.param(
            ("source", "я" * 300 + "-a.docx"), ("source", "я" * 300 + "-b.docx"), id="long"
        ),
        pytest.param(("source", "report.docx"), ("source", " report.docx "), id="whitespace"),
    ],
)
def test_distinct_document_names_keep_distinct_contents(
    storage: LocalFileStorage, first: tuple[str, str], second: tuple[str, str]
) -> None:
    async def scenario() -> None:
        first_key = await storage.save(
            session_id="synthetic", doc_id=first[0], filename=first[1], content=b"first"
        )
        second_key = await storage.save(
            session_id="synthetic", doc_id=second[0], filename=second[1], content=b"second"
        )

        assert first_key != second_key
        assert await storage.read(first_key) == b"first"
        assert await storage.read(second_key) == b"second"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "session_id",
    [
        pytest.param("учебная-сессия", id="unicode"),
        pytest.param("s" * 256, id="overlong"),
        pytest.param("..", id="dots"),
        pytest.param("../neighbor", id="slash"),
        pytest.param("synthetic\n", id="newline"),
    ],
)
def test_encoded_session_directory_does_not_alias_literal_id(
    storage: LocalFileStorage, session_id: str
) -> None:
    async def scenario() -> None:
        args = {"doc_id": DOCUMENT_ID, "filename": "report.docx"}
        first = await storage.save(session_id=session_id, **args, content=b"first")
        encoded_session = Path(first).parent.name
        second = await storage.save(session_id=encoded_session, **args, content=b"second")

        assert encoded_session.isascii()
        assert len(encoded_session) <= 255
        assert Path(second).parent != Path(first).parent
        assert await storage.read(first) == b"first"
        assert await storage.read(second) == b"second"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("filename", "extension"),
    [
        pytest.param("я" * 300 + ".docx", ".docx", id="multibyte-stem"),
        pytest.param("я." + "x" * 187, "." + "x" * 187, id="one-byte-stem-budget"),
        pytest.param("я." + "x" * 230, None, id="oversized-extension"),
    ],
)
def test_save_limits_utf8_component_size(
    storage: LocalFileStorage, filename: str, extension: str | None
) -> None:
    async def scenario() -> None:
        key = await storage.save(
            session_id="synthetic", doc_id="d" * 256, filename=filename, content=b"report"
        )
        stored_name = Path(key).name

        assert len(stored_name.encode("utf-8")) <= 255
        if extension is not None:
            assert stored_name.endswith(extension)
        assert await storage.read(key) == b"report"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "filename", ["report.docx", "report\u0085notes.docx"], ids=["plain", "nel"]
)
def test_previous_storage_keys_remain_usable(storage: LocalFileStorage, filename: str) -> None:
    key = f"synthetic/{DOCUMENT_ID}_{filename}"
    previous = storage.base_dir / key
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous")

    async def scenario() -> None:
        current = await storage.save(
            session_id="synthetic", doc_id=DOCUMENT_ID, filename=filename, content=b"current"
        )
        assert current != key
        assert await storage.read(current) == b"current"
        assert await storage.exists(key)
        assert await storage.read(key) == b"previous"
        assert await storage.delete(key)

    asyncio.run(scenario())
