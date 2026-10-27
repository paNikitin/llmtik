"""Тесты к контракту journal.py для задачи t03.

Файл можно запускать напрямую: `python my_tests.py` посчитает мутационный балл.
Внутри песочницы задачи он импортируется как тестовый модуль, поэтому импорт
`journal` сделан внутри тестов, а запуск спрятан под `__main__`.
"""


def v(event_id, revision, operation, **fields):
    return {
        "event_id": event_id,
        "revision": revision,
        "operation": operation,
        "fields": fields,
    }


# ------------------------------------------------------------------ add


def test_add_returns_new_list_and_keeps_original():
    from journal import add

    original = []
    journal, status = add(original, v("e", 1, "create", start=10, length=2))
    assert status == "added"
    assert journal == [v("e", 1, "create", start=10, length=2)]
    assert original == []
    assert journal is not original


def test_add_duplicate_keeps_journal():
    from journal import add

    version = v("e", 1, "create", start=10, length=2)
    journal, _ = add([], version)
    again, status = add(journal, version)
    assert status == "duplicate"
    assert again == journal
    assert len(again) == 1


def test_add_conflict_keeps_journal():
    from journal import add

    journal, _ = add([], v("e", 1, "create", start=10, length=2))
    other = v("e", 1, "create", start=99, length=2)
    result, status = add(journal, other)
    assert status == "conflict"
    assert result == journal
    assert len(result) == 1


def test_add_accepts_versions_out_of_order():
    from journal import add

    journal, _ = add([], v("e", 2, "update", start=20))
    journal, status = add(journal, v("e", 1, "create", start=10))
    assert status == "added"
    assert len(journal) == 2


# ---------------------------------------------------------------- state


def test_state_applies_versions_by_revision():
    from journal import add, state

    journal = []
    for version in (
        v("e", 1, "create", start=10, length=2),
        v("e", 3, "update", start=30),
        v("e", 2, "update", length=5),
    ):
        journal, _ = add(journal, version)
    assert state(journal, "e") == {"start": 30, "length": 5}


def test_state_requires_create_revision_one():
    from journal import add, state

    journal, _ = add([], v("e", 2, "create", start=10, length=2))
    assert state(journal, "e") is None


def test_state_none_without_create():
    from journal import add, state

    journal, _ = add([], v("e", 2, "update", start=10))
    assert state(journal, "e") is None


def test_state_none_for_unknown_event():
    from journal import add, state

    journal, _ = add([], v("e", 1, "create", start=10, length=2))
    assert state(journal, "other") is None


def test_state_cancel_is_terminal():
    from journal import add, state

    journal = []
    for version in (
        v("e", 1, "create", start=10, length=2, direction="up"),
        v("e", 2, "cancel"),
        v("e", 3, "update", start=50),
    ):
        journal, _ = add(journal, version)
    assert state(journal, "e") is None


# --------------------------------------------------------------- active


def test_active_interval_boundaries():
    from journal import add, active

    journal, _ = add([], v("e", 1, "create", start=10, length=2))
    assert active(journal, 9) == ["e"]
    assert active(journal, 10) == ["e"]
    assert active(journal, 11) == ["e"]
    assert active(journal, 12) == []


def test_active_orders_by_start():
    from journal import add, active

    journal = []
    for version in (
        v("a", 1, "create", start=20, length=2),
        v("b", 1, "create", start=10, length=2),
    ):
        journal, _ = add(journal, version)
    assert active(journal, 0) == ["b", "a"]


def test_active_orders_by_id_on_equal_start():
    from journal import add, active

    journal = []
    for version in (
        v("b", 1, "create", start=10, length=2),
        v("a", 1, "create", start=10, length=2),
    ):
        journal, _ = add(journal, version)
    assert active(journal, 0) == ["a", "b"]


def test_active_skips_incomplete_event():
    from journal import add, active

    journal = []
    for version in (
        v("full", 1, "create", start=10, length=2),
        v("partial", 1, "create", start=10),
    ):
        journal, _ = add(journal, version)
    assert active(journal, 0) == ["full"]


def test_active_excludes_cancelled():
    from journal import add, active

    journal = []
    for version in (
        v("e", 1, "create", start=10, length=2),
        v("e", 2, "cancel"),
    ):
        journal, _ = add(journal, version)
    assert active(journal, 0) == []


if __name__ == "__main__":
    from polygon2 import check_tests

    check_tests("t03", open(__file__).read())
