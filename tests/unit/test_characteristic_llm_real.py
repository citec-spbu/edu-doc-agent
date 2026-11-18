"""Ручной замер характеристики: только --run-llm, локальные документы и эталон."""

import pytest

from edu_doc_agent.eval.characteristic_llm_gold import load_gold, score_characteristic_llm
from edu_doc_agent.eval.datasets import local_gold
from edu_doc_agent.extraction.characteristic_llm import extract_characteristic_llm
from edu_doc_agent.llm import OpenAICompatibleClient


@pytest.mark.llm
def test_real_characteristic_llm_quality():
    from tests.real_data import real_characteristic_markdown

    path = local_gold("characteristic_llm_gold.json")
    if not path.exists():
        pytest.skip("нет локального data/evaluation/characteristic_llm_gold.json")
    md = real_characteristic_markdown()
    if md is None:
        pytest.skip("нет data/Характеристика.zip")
    client = OpenAICompatibleClient.from_settings()
    result = extract_characteristic_llm(md, "characteristic_corr.docx", client)
    report = score_characteristic_llm(result, load_gold(path))
    # Результат и цитаты остаются рядом с локальным эталоном, вне Git.
    path.with_name("characteristic_llm_prediction.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    path.with_name("characteristic_llm_report.txt").write_text(
        f"model={client.model}; temperature=0\n" + report.summary(), encoding="utf-8"
    )
    print("\n" + report.summary())
    assert report.passed(), report.summary()
