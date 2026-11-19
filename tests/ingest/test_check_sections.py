from ingest.check_sections import check


def chunk(section, form="10-K"):
    return {"section_id": section, "form_type": form}


def test_check_passes_clean_10k():
    chunks = [chunk("cover")] + [
        chunk(s) for s in ("item_1a", "item_7", "item_8") for _ in range(5)
    ]
    assert check(chunks) == []


def test_check_flags_tiny_section_and_big_cover():
    chunks = [chunk("cover", "10-Q")] * 10 + [chunk("part_i_item_1", "10-Q")] * 5
    problems = check(chunks)
    assert "part_i_item_2 has 0 chunks" in problems
    assert any(p.startswith("cover has 10/15") for p in problems)


def test_check_ignores_8k():
    assert check([chunk("exhibit_99_1", "8-K")]) == []
