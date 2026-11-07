"""Streamlit-интерфейс edu-doc-agent.

На этом этапе — только проверка доступности API и заглушки для будущих
разделов. Реальный UI появится в M4.
"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="edu-doc-agent",
    page_icon="📚",
    layout="wide",
)

st.title("edu-doc-agent")
st.caption("Агентная ИИ-система анализа и проверки учебно-методических документов ОП")


def _check_api() -> tuple[bool, str]:
    """Пингует /health. Возвращает (жив ли, пояснение)."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"

    data = response.json()
    return True, f"version {data.get('version', '?')}"


with st.sidebar:
    st.subheader("Состояние")
    ok, detail = _check_api()
    if ok:
        st.success(f"API доступен ({detail})")
    else:
        st.error("API недоступен")
        st.caption(detail)
        st.caption(f"URL: {API_URL}")

    st.divider()
    st.caption("Разделы появятся по мере готовности модулей.")

tab_upload, tab_validate, tab_report = st.tabs(["Загрузка пакета", "Проверка", "Протокол"])

with tab_upload:
    st.info("Загрузка пакета документов ОП появится в M2 (задача «сквозной скелет»).")

with tab_validate:
    st.info("Проверка на соответствие ФГОС ВО и профстандартам появится в M3.")

with tab_report:
    st.info("Протокол проверки и карты связей появятся в M3–M4.")

st.divider()
st.caption(f"Документация: https://citec-spbu.github.io/edu-doc-agent/ · API URL: `{API_URL}`")
