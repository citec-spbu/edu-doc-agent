"""Проверки NER-метрики на синтетических записях и заведомых ошибках."""

import copy

import pytest

from edu_doc_agent.eval.program_ner import (
    anchored_score,
    characteristic_records,
    ckmv_records,
    compare_groups,
    main,
    matrix_records,
    plan_records,
    record,
    score_records,
)


def test_anchored_matching_rejects_position_without_its_head_word():
    score = anchored_score(
        ["руководителей проектов", "групп разработки"],
        ["руководитель проектов", "руководитель групп разработки"],
    )
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)
    assert score.f1 == 0.5
    assert anchored_score(["руководителей проектов"] * 2, ["руководитель проектов"]).fp == 1


def test_missing_extra_and_duplicate_records_are_counted_one_to_one():
    a, b, c = record("УК-1", "Курс А"), record("УК-2", "Курс Б"), record("УК-3", "Курс В")
    score = score_records([a, a, c], [a, b])
    assert (score.tp, score.fp, score.fn) == (1, 2, 1)
    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == 0.5 and score.f1 == pytest.approx(0.4)
    assert score_records([a, a], [a, a]).f1 == 1


def test_text_matching_does_not_hide_missing_words_or_merge_entities():
    assert record("Ёлка — курс") == record("елка - КУРС")
    score = score_records([record("морские системы")], [record("морские системы управления")])
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)
    assert score_records([], []).f1 == 1
    assert score_records([record("лишняя")], []).f1 == 0


def test_plan_relations_are_scoped_by_row_and_penalize_moved_competence():
    gold = {
        "competences": [{"code": "УК-1", "name": "Проверяет учебный маршрут"}],
        "entries": [{"line": 10, "code": "990001", "name": "Навигация", "competences": ["УК-1"]}],
        "final_attestation": [],
    }
    predicted = copy.deepcopy(gold)
    row = predicted["entries"][0]
    row["source"] = {"line": row.pop("line")}
    good = compare_groups(plan_records(predicted), plan_records(gold))
    assert all(s["f1"] == 1 for s in good.values())
    row["source"]["line"] = 11
    bad = compare_groups(plan_records(predicted), plan_records(gold))
    assert bad["entity.discipline"]["f1"] == 1
    assert bad["relation.discipline_competence"]["fp"] == 1
    assert bad["relation.discipline_competence"]["fn"] == 1


def test_ckmv_wrong_parent_is_detected_even_if_entities_match():
    gold = {
        "competences": [
            {
                "code": "УК-1",
                "name": "Управляет учебным судном",
                "line": 3,
                "role": "Инженер",
                "role_section": "Основная роль",
                "group": "База",
                "indicators": [
                    {
                        "code": "УК-1.1",
                        "name": "Считает координаты",
                        "level": "Базовый",
                        "line": 4,
                        "outcomes": [
                            {"code": "УК-1.1 З-1", "kind": "знает", "text": "Знает учебную формулу"}
                        ],
                    }
                ],
            }
        ]
    }
    predicted = copy.deepcopy(gold)
    predicted["competences"][0]["line"] = 9
    scores = compare_groups(ckmv_records(predicted), ckmv_records(gold))
    assert all(s["f1"] == 1 for n, s in scores.items() if n.startswith("entity."))
    assert scores["relation.competence_indicator"]["f1"] == 0
    assert scores["relation.indicator_outcome"]["f1"] == 0


def test_characteristic_wrong_group_is_not_hidden_by_matching_code():
    gold = {
        "competences": [{"code": "УК-1", "name": "Рассчитывает маршрут", "group": "База"}],
        **{
            n: []
            for n in (
                "qualification",
                "areas",
                "standards",
                "employers",
                "objects",
                "activities",
                "mappings",
            )
        },
    }
    predicted = copy.deepcopy(gold)
    predicted["competences"][0]["group"] = "Иная группа"
    score = compare_groups(characteristic_records(predicted), characteristic_records(gold))
    assert score["entity.competence"]["tp"] == 0
    assert len(score["entity.competence"]["false_negative"]) == 1


def test_matrix_scores_positive_links_without_true_negative_inflation():
    gold = {
        "columns": ["УК-1", "УК-2"],
        "rows": [{"number": 1, "discipline": "Навигация", "competences": ["УК-1"]}],
    }
    predicted = copy.deepcopy(gold)
    predicted["columns"] = [{"code": c} for c in gold["columns"]]
    predicted["rows"][0]["competences"].append("УК-2")
    scores = compare_groups(matrix_records(predicted), matrix_records(gold))
    assert scores["entity.competence"]["f1"] == 1
    assert scores["relation.discipline_competence"]["tp"] == 1
    assert scores["relation.discipline_competence"]["fp"] == 1
    assert scores["relation.discipline_competence"]["precision"] == 0.5


def test_unknown_predicted_group_is_not_ignored():
    scores = compare_groups({"entity.unexpected": [record("лишняя")]}, {})
    assert scores["entity.unexpected"]["fp"] == 1


def test_missing_real_data_reports_error_without_fake_scores(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--data-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
                "--report",
                str(tmp_path / "report.md"),
            ]
        )
    assert exc.value.code == 2
    assert "Нет локального файла" in capsys.readouterr().err
    assert not (tmp_path / "result.json").exists()
