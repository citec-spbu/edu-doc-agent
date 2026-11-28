"""Профстандарт из текста карточки и из PDF: общие сведения, обобщённые и трудовые функции."""

from pathlib import Path

import pytest

from edu_doc_agent.parsing.pdf_parser import NoTextLayerError
from edu_doc_agent.retrieval.corpus.prof_standard import card_from_pdf, chunk_prof_standard

FIXTURE = Path(__file__).parents[2] / "fixtures" / "profstandard_normal_01.pdf"

# синтетическая карточка в формате сайта: подписи и значения на отдельных строках
CARD = """Профстандарт: 99.001
Синтетик
Поиск профстандарта
99.001
Синтетик
Настоящий профстандарт действует с 01.03.2023 по 01.03.2029

Утвержден
приказом Министерства труда
и социальной защиты
Российской Федерации
от 20.07.2022 № 999н

ПРОФЕССИОНАЛЬНЫЙ СТАНДАРТ
СИНТЕТИК

I. Общие сведения

Создание синтетических данных
99.001
(наименование вида профессиональной деятельности)
Код
Основная цель вида профессиональной деятельности:
Подготовка данных для проверки программ

II. Описание трудовых функций, входящих в профессиональный стандарт
A
Подготовка данных
3
Сбор данных
A/01.3
3

III. Характеристика обобщенных трудовых функций

3.1. Обобщенная трудовая функция

Наименование
Подготовка данных
Код
А
Уровень квалификации
6

Происхождение обобщенной трудовой функции
Оригинал
X
Заимствовано из оригинала
Код оригинала
Регистрационный номер профессионального стандарта

Возможные наименования должностей, профессий
Инженер данных
Требования к образованию и обучению
Высшее образование - бакалавриат
Требования к опыту практической работы
-
Особые условия допуска к работе
-
Другие характеристики
-

Дополнительные характеристики

Наименование документа
Код
Наименование базовой группы, должности (профессии) или специальности
ОКЗ
2512
Разработчики программного обеспечения
ЕКС <3>
-
Инженер
ОКСО <4>
1.01.03.02
Прикладная математика и информатика
2.09.03.01
Информатика и вычислительная техника

3.1.1. Трудовая функция

Наименование
Сбор и разметка
синтетических данных
Код
А/01.6
Уровень (подуровень) квалификации
6

Происхождение трудовой функции
Оригинал
X

Трудовые действия
Сбор данных
Разметка данных
Необходимые умения
Применять инструменты разметки
Необходимые знания
Форматы данных
Другие характеристики
-

IV. Сведения об организациях - разработчиках профессионального стандарта
4.1. Ответственная организация-разработчик
Синтетический институт
"""


def test_card_is_cut_into_general_info_and_functions() -> None:
    doc = chunk_prof_standard(CARD, "99.001.txt")
    info = doc.document
    assert (info.doc_id, info.doc_type, info.title) == (
        "ps-99.001",
        "prof_standard",
        "Профессиональный стандарт 99.001 «Синтетик»",
    )
    assert info.edition == "999н от 20.07.2022"
    assert info.approval_act.endswith("№ 999н; действует с 01.03.2023 по 01.03.2029")
    assert doc.warnings == []
    general, generalized, function = doc.fragments
    assert (general.id, general.kind, general.codes) == (
        "ps-99.001:general",
        "prof_standard",
        ["99.001"],
    )
    assert "Вид профессиональной деятельности: Создание синтетических данных" in general.text
    assert general.text.endswith("Обобщенные трудовые функции:\nA Подготовка данных (уровень 6)")
    # кириллическая «А» в коде — латинская A
    assert (generalized.id, generalized.kind, generalized.number, generalized.codes) == (
        "ps-99.001:A",
        "generalized_function",
        "3.1",
        ["99.001:A"],
    )
    assert generalized.text == (
        "Обобщенная трудовая функция A «Подготовка данных», уровень квалификации 6\n"
        "Возможные наименования должностей, профессий:\nИнженер данных\n"
        "Требования к образованию и обучению:\nВысшее образование - бакалавриат\n"
        "ОКСО:\n1.01.03.02 Прикладная математика и информатика\n"
        "2.09.03.01 Информатика и вычислительная техника"
    )  # поля «-» пропущены, служебные строки шапок — тоже
    assert (function.id, function.kind, function.codes) == (
        "ps-99.001:A/01.6",
        "labor_function",
        ["99.001:A/01.6"],
    )
    assert function.text == (
        "Трудовая функция A/01.6 «Сбор и разметка синтетических данных», "
        "уровень (подуровень) квалификации 6\n"
        "Трудовые действия:\nСбор данных\nРазметка данных\n"
        "Необходимые умения:\nПрименять инструменты разметки\n"
        "Необходимые знания:\nФорматы данных"
    )
    assert function.context == generalized.text.splitlines()[0]
    assert (
        function.path
        == "Профессиональный стандарт 99.001 «Синтетик» › 3.1.1 Трудовая функция A/01.6"
    )
    assert "Синтетический институт" not in "\n".join(f.text for f in doc.fragments)


def test_metadata_from_manifest_wins_and_missing_is_an_error() -> None:
    doc = chunk_prof_standard(CARD, "x.txt", edition="999н ред. от 01.01.2024")
    assert doc.document.edition == "999н ред. от 01.01.2024"
    with pytest.raises(ValueError, match="не определены"):
        chunk_prof_standard("Текст без реквизитов", "empty.txt")


def test_card_without_functions_is_an_error() -> None:
    head = CARD.split("III. Характеристика")[0]
    doc = chunk_prof_standard(head, "99.001.txt")
    assert [f.kind for f in doc.fragments] == ["prof_standard"]
    assert any(w.startswith("ОШИБКА") for w in doc.warnings)


def test_pdf_is_read_as_card() -> None:
    doc = chunk_prof_standard(card_from_pdf(FIXTURE), FIXTURE.name)
    info = doc.document
    assert (info.doc_id, info.title) == ("ps-99.001", "Профессиональный стандарт 99.001 «Синтетик»")
    assert info.edition == "999н от 01.03.2030"  # дата приказа прописью
    assert doc.warnings == []
    by_id = {f.id: f for f in doc.fragments}
    assert list(by_id) == [
        "ps-99.001:general",
        "ps-99.001:A",
        "ps-99.001:A/01.6",
        "ps-99.001:A/02.6",
    ]  # оглавление не принято за начало раздела III
    assert "Вид профессиональной деятельности: Создание синтетических данных" in (
        by_id["ps-99.001:general"].text
    )
    generalized = by_id["ps-99.001:A"]
    # название перенесено в заголовке «3.1.» — склеено; строка «Происхождение» не в шапке
    assert generalized.text.splitlines()[0] == (
        "Обобщенная трудовая функция A «Подготовка синтетических данных», уровень квалификации 6"
    )
    assert "Возможные наименования должностей, профессий:\nИнженер данных" in generalized.text
    assert generalized.text.endswith("ОКСО:\n2.09.03.04 Программная инженерия")
    collect = by_id["ps-99.001:A/01.6"]
    # строка таблицы, разорванная страницей, склеена; новое знание на новой странице — нет
    assert collect.text.endswith(
        "Необходимые знания:\nМетоды генерации данных для проверки программ\n"
        "Форматы хранения данных"
    )
    check = by_id["ps-99.001:A/02.6"]
    # шапка функции начинается внизу страницы и продолжается на следующей
    assert check.text.splitlines()[0] == (
        "Трудовая функция A/02.6 «Проверка синтетических данных на полноту», "
        "уровень (подуровень) квалификации 6"
    )
    assert "Синтетический институт" not in "\n".join(f.text for f in doc.fragments)


def test_scanned_pdf_is_an_error(tmp_path: Path) -> None:
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )
    with pytest.raises(NoTextLayerError, match="OCR"):
        card_from_pdf(scan)
