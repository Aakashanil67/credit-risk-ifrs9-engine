import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "check_environment.py"


def load_checker():
    assert SCRIPT_PATH.exists(), "check_environment.py is not implemented"
    spec = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_pins_ignores_comments_and_strips_extras():
    checker = load_checker()

    pins = checker.parse_pins("# comment\n\nuvicorn[standard]==0.52.4\nPandas==2.2.3\n")

    assert pins == {"uvicorn": "0.52.4", "pandas": "2.2.3"}


def test_compare_versions_reports_missing_and_mismatched_packages():
    checker = load_checker()

    issues = checker.compare_versions(
        {"pandas": "2.2.3", "numpy": "1.26.4"},
        {"pandas": "3.0.5", "numpy": None},
    )

    assert issues == [
        "numpy is not installed; expected 1.26.4",
        "pandas is 3.0.5; expected 2.2.3",
    ]


def test_python_version_check_accepts_312_and_rejects_other_versions():
    checker = load_checker()

    assert checker.python_version_issue((3, 12)) is None
    assert checker.python_version_issue((3, 13)) == "Python 3.13 is unsupported; use Python 3.12"
