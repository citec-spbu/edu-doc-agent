"""Гибридное извлечение характеристики: синтетика, без сетевых вызовов."""

import json

import pytest

from edu_doc_agent.extraction.characteristic import (
    extract_characteristic,
    extract_characteristic_file,
)
from edu_doc_agent.extraction.characteristic_llm import SYSTEM, extract_characteristic_llm
from edu_doc_agent.llm import FakeLLMClient, LLMError

MD = """### Аннотация
Основная роль — «Инженер морских систем».
Выпускники занимают должности инженеров. Траектория: гидроакустика.
Проект «Морская лаборатория» действует согласно приказу № 42.

### Миссия
Принципы: открытость исследований; сотрудничество с верфями.

# 5. Индивидуализация обучения
Профиль «Морская робототехника».
## 5.1. Выбор траектории
Средства: выбор факультативов и дистанционные модули.
# 6. Другой раздел
Чужая цитата.
"""


def annotation(**overrides):
    output = {
        "roles": [
            {
                "name": "Инженер морских систем",
                "kind": "основная",
                "quote": "Инженер морских систем",
            }
        ],
        "positions": [{"name": "инженер", "quote": "должности инженеров"}],
        "tracks": [{"name": "гидроакустика", "quote": "гидроакустика"}],
        "projects": [
            {
                "name": "Морская лаборатория",
                "kind": "другое",
                "quote": "Проект «Морская лаборатория»",
            }
        ],
        "legal_acts": [{"name": "приказ № 42", "quote": "приказу № 42"}],
    }
    return json.dumps(output | overrides, ensure_ascii=False)


def client():
    return FakeLLMClient(
        replies=[
            annotation(),
            json.dumps(
                {
                    "principles": [
                        {"text": "открытость исследований", "quote": "открытость исследований"},
                        {"text": "сотрудничество с верфями", "quote": "сотрудничество с верфями"},
                    ]
                }
            ),
            json.dumps(
                {
                    "profiles": [
                        {"name": "Морская робототехника", "quote": "Морская робототехника"}
                    ],
                    "means": [{"name": "выбор факультативов", "quote": "выбор факультативов"}],
                }
            ),
        ]
    )


def test_hybrid_keeps_rules_and_adds_all_llm_fields(tmp_path):
    fake = client()
    path = tmp_path / "synthetic.md"
    path.write_text(MD, encoding="utf-8")
    rules = extract_characteristic_file(path)
    hybrid = extract_characteristic_file(path, llm=fake)
    assert rules.llm is None
    assert hybrid.annotation == rules.annotation
    assert hybrid.warnings == rules.warnings
    result = hybrid.llm
    assert result is not None and not result.warnings
    assert len(result.roles) == len(result.positions) == len(result.tracks) == 1
    assert len(result.projects) == len(result.legal_acts) == len(result.profiles) == 1
    assert len(result.principles) == 2 and len(result.individualization_means) == 1
    assert len(fake.calls) == 3 and all(c.system == SYSTEM for c in fake.calls)
    assert "Чужая цитата" not in fake.calls[-1].user
    source = result.individualization_means[0].source
    assert source.file == "synthetic.md" and source.section == "5.1. Выбор траектории"
    assert source.line == 12 and source.quote == MD.splitlines()[11]
    assert "llm" in hybrid.model_dump(mode="json")


@pytest.mark.parametrize(
    "quote",
    ["выдуманная цитата", "Чужая цитата.", "", "   ", "Основная роль…Инженер морских систем"],
)
def test_invalid_or_other_section_evidence_is_rejected(quote):
    fake = FakeLLMClient(
        replies=[
            annotation(
                roles=[{"name": "Инженер морских систем", "kind": "основная", "quote": quote}]
            )
        ]
    )
    result = extract_characteristic_llm(MD, "x.md", fake, ["annotation"])
    assert not result.roles and len(result.warnings) == 1


def test_invalid_json_and_schema_are_retried_with_characteristic_system():
    fake = FakeLLMClient(replies=["не JSON", '{"roles": []}', annotation()])
    result = extract_characteristic_llm(MD, "x.md", fake, ["annotation"])
    assert len(result.roles) == 1 and not result.warnings
    assert len(fake.calls) == 3 and all(c.system == SYSTEM for c in fake.calls)


def test_exhausted_retries_are_reported_and_next_task_runs():
    fake = FakeLLMClient(replies=["{}"] * 3 + ['{"principles": []}'])
    result = extract_characteristic_llm(MD, "x.md", fake, ["annotation", "mission"])
    assert len(fake.calls) == 4 and len(result.warnings) == 1
    assert "3 попыток" in result.warnings[0]


def test_transport_failure_is_reported_without_provider_details():
    def respond(call):
        if call.schema["title"] == "AnnotationOutput":
            raise LLMError("sensitive-provider-details")
        if call.schema["title"] == "MissionOutput":
            return '{"principles": []}'
        return '{"profiles": [], "means": []}'

    fake = FakeLLMClient(respond=respond)
    result = extract_characteristic(MD, "x.md", llm=fake)
    assert len(fake.calls) == 3
    assert any("сбой вызова LLM" in w for w in result.warnings)
    assert "sensitive-provider-details" not in result.model_dump_json()


def test_missing_sections_do_not_call_llm():
    fake = FakeLLMClient()
    result = extract_characteristic_llm("# Документ\nБез нужных разделов", "x.md", fake)
    assert len(result.warnings) == 3 and not fake.calls


def test_empty_selection_and_unknown_task():
    fake = FakeLLMClient()
    assert not extract_characteristic_llm(MD, "x.md", fake, []).warnings
    with pytest.raises(ValueError, match="неизвестные"):
        extract_characteristic_llm(MD, "x.md", fake, ["typo"])
    assert not fake.calls


def test_deduplication_keeps_different_kinds_and_rejects_empty_names():
    role = {"name": "Инженер морских систем", "kind": "основная", "quote": "Инженер морских систем"}
    fake = FakeLLMClient(
        replies=[
            annotation(roles=[role, role, role | {"kind": "опережающая"}, role | {"name": " "}])
        ]
    )
    result = extract_characteristic_llm(MD, "x.md", fake, ["annotation"])
    assert [r.kind for r in result.roles] == ["основная", "опережающая"]
    assert len(result.warnings) == 1


def test_empty_valid_outputs_are_not_errors():
    fake = FakeLLMClient(
        replies=[annotation(roles=[], positions=[], tracks=[], projects=[], legal_acts=[])]
    )
    result = extract_characteristic_llm(MD, "x.md", fake, ["annotation"])
    assert not result.roles and not result.warnings
