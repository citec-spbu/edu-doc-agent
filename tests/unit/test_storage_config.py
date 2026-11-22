"""Настройки файлового хранилища на синтетическом окружении."""

import os
from pathlib import Path

import pytest

from edu_doc_agent.config import Settings


@pytest.mark.parametrize("value", [None, "", "   "], ids=["missing", "empty", "whitespace"])
def test_storage_dir_defaults_to_relative_storage(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    for name in list(os.environ):
        if name.lower() == "storage_dir":
            monkeypatch.delenv(name)
    if value is not None:
        monkeypatch.setenv("STORAGE_DIR", value)

    assert Settings(_env_file=None).storage_dir == Path("storage")


def test_storage_dir_can_be_set_in_dotenv_and_overridden_by_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in list(os.environ):
        if name.lower() == "storage_dir":
            monkeypatch.delenv(name)
    dotenv = tmp_path / ".env"
    dotenv.write_text("STORAGE_DIR=from-dotenv\n", encoding="utf-8")
    assert Settings(_env_file=dotenv).storage_dir == Path("from-dotenv")

    override = tmp_path / "from-env"
    monkeypatch.setenv("STORAGE_DIR", str(override))
    assert Settings(_env_file=dotenv).storage_dir == override
