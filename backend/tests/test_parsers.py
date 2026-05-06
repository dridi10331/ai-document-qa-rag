from app.services.parsers import parse_md


def test_parse_md_extracts_headings(tmp_path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\n\nBody text", encoding="utf-8")

    parsed = parse_md(file_path)

    assert parsed.pages
    assert parsed.metadata["type"] == "md"
    assert parsed.metadata["headings"][0] == "Title"
