"""Контракт локального файлового хранилища и регрессии его адресации."""

import asyncio
from pathlib import Path
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


def test_failed_publication_keeps_original_and_removes_temporary_file(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed_replace(_source: str, _destination: Path) -> None:
        raise OSError("synthetic failure")

    async def scenario() -> None:
        args = {"session_id": "synthetic", "doc_id": DOCUMENT_ID, "filename": "report.docx"}
        key = await storage.save(**args, content=b"original")
        target = storage.resolve_path(key)

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
        original_read_bytes = Path.read_bytes

        def read_after_delete(path: Path) -> bytes:
            if path == target:
                path.unlink()
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", read_after_delete)
        with pytest.raises(StorageFileNotFoundError) as exc:
            await storage.read(key)
        assert isinstance(exc.value.__cause__, FileNotFoundError)
        assert not await storage.exists(key)

    asyncio.run(scenario())


def test_delete_refuses_directories(storage: LocalFileStorage) -> None:
    directory = storage.base_dir / "synthetic"
    directory.mkdir(parents=True)

    with pytest.raises(InvalidStoragePathError):
        asyncio.run(storage.delete("synthetic"))

    assert directory.is_dir()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("session_id", "..", id="parent-session"),
        pytest.param("session_id", "../neighbor", id="nested-session"),
        pytest.param("session_id", "synthetic\n", id="session-newline"),
        pytest.param("doc_id", "folder/doc", id="nested-document"),
        pytest.param("doc_id", "report\n", id="document-newline"),
        pytest.param("filename", "../report.docx", id="relative-name"),
        pytest.param("filename", "folder\\report.docx", id="windows-separator"),
        pytest.param("filename", "report\x00.docx", id="nul-in-name"),
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
    target = storage.base_dir / "second"
    target.mkdir(parents=True)
    (storage.base_dir / "first").symlink_to(target, target_is_directory=True)

    with pytest.raises(InvalidStoragePathError):
        asyncio.run(
            storage.save(
                session_id="first", doc_id=DOCUMENT_ID, filename="report.docx", content=b"report"
            )
        )

    assert not list(target.iterdir())


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


@pytest.mark.parametrize("session_id", ["учебная-сессия", "s" * 256], ids=["unicode", "overlong"])
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


def test_previous_storage_keys_remain_usable(storage: LocalFileStorage) -> None:
    key = f"synthetic/{DOCUMENT_ID}_report.docx"
    previous = storage.base_dir / key
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous")

    async def scenario() -> None:
        current = await storage.save(
            session_id="synthetic", doc_id=DOCUMENT_ID, filename="report.docx", content=b"current"
        )
        assert current != key
        assert await storage.read(current) == b"current"
        assert await storage.exists(key)
        assert await storage.read(key) == b"previous"
        assert await storage.delete(key)

    asyncio.run(scenario())
