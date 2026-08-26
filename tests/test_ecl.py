from src.ecl import build_mechanics_examples, write_ifrs9_summary


def test_mechanics_examples_cover_each_ifrs9_stage() -> None:
    examples = build_mechanics_examples()

    assert examples["stage"].tolist() == [1, 2, 3]
    assert examples["ecl"].between(0, examples["ead"]).all()
    assert (
        examples.loc[examples["stage"] == 2, "ecl"].iloc[0]
        > examples.loc[examples["stage"] == 1, "ecl"].iloc[0]
    )


def test_mechanics_examples_use_explicit_stage_rules() -> None:
    examples = build_mechanics_examples()

    assert examples["stage_name"].tolist() == [
        "Stage 1 - performing",
        "Stage 2 - significant increase in credit risk",
        "Stage 3 - credit-impaired",
    ]


def test_summary_labels_units_and_limits(tmp_path) -> None:
    examples = build_mechanics_examples()
    output_path = tmp_path / "ifrs9_summary.md"

    write_ifrs9_summary(examples, output_path)

    summary = output_path.read_text(encoding="utf-8")
    assert "monetary units" in summary
    assert "not a portfolio provision" in summary
    assert "(R)" not in summary
    assert "SICR" in summary
