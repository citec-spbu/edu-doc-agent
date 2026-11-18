"""Проверка метрики на синтетике: пропуски, лишние сущности, типы и перестановки."""

import json

import pytest

from edu_doc_agent.eval.characteristic_llm_gold import (
    _pairs,
    main,
    score_characteristic_llm,
)
from edu_doc_agent.extraction.characteristic_llm import CharacteristicLLM, LLMEntity
from edu_doc_agent.extraction.source import SourceRef


def gold():
    return {
        "annotation": {
            "roles": [{"name": "инженер морских систем", "kind": "основная"}],
            "positions": [],
            "tracks": [],
            "projects": [],
            "legal_acts": [],
        },
        "mission_principles": ["открытость исследований"],
        "individualization": {"profiles": [], "means": []},
    }


def entity(name, kind=None):
    return LLMEntity(name=name, kind=kind, quote=name, source=SourceRef(file="synthetic.md"))


def perfect():
    return CharacteristicLLM(
        roles=[entity("инженер морских систем", "основная")],
        principles=[entity("открытость исследований")],
    )


def test_perfect_and_empty_negative_groups():
    report = score_characteristic_llm(perfect(), gold())
    assert report.passed() and all(s.f1 == 1 for s in report.scores.values())


def test_missing_duplicate_and_unexpected_entities_are_penalized():
    result = perfect()
    result.roles *= 2
    result.principles.clear()
    result.tracks.append(entity("навигация"))
    report = score_characteristic_llm(result, gold())
    assert not report.passed()
    assert report.scores["roles"].fp == 1
    assert report.scores["principles"].fn == 1
    assert report.scores["tracks"].fp == 1


def test_wrong_kind_and_extraction_failure_fail_threshold():
    result = perfect()
    result.roles[0].kind = "опережающая"
    report = score_characteristic_llm(result, gold())
    assert report.scores["roles"].f1 == 1 and not report.passed()
    result = perfect()
    result.warnings = ["mission: сбой вызова LLM"]
    assert not score_characteristic_llm(result, gold()).passed()


def test_matching_is_one_to_one_and_independent_of_order():
    assert (
        len(
            _pairs(
                ["морская", "морская робототехника"], ["морская робототехника", "морская навигация"]
            )
        )
        == 2
    )
    assert (
        len(_pairs(["морская робототехника и навигация"], ["морская робототехника", "навигация"]))
        == 1
    )
    assert _pairs([""], [""]) == []


def test_cli_offline_and_explicit_online_opt_in(tmp_path, capsys):
    gold_path, prediction = tmp_path / "gold.json", tmp_path / "prediction.json"
    gold_path.write_text(json.dumps(gold()), encoding="utf-8")
    prediction.write_text(perfect().model_dump_json(), encoding="utf-8")
    assert main(["--gold", str(gold_path), "--prediction", str(prediction)]) == 0
    assert "1.000" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--gold", str(gold_path), "--input", str(tmp_path / "absent.md")])
    assert exc.value.code == 2


def test_incomplete_gold_does_not_silently_score_as_empty():
    with pytest.raises(KeyError):
        score_characteristic_llm(perfect(), {"annotation": {}})
