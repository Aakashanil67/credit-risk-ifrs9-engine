"""Check the interpreter and direct package pins before loading model libraries."""

import importlib.metadata
import re
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PIN_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s#]+)$")


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(text: str) -> dict[str, str]:
    pins = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _PIN_PATTERN.fullmatch(line)
        if match is None:
            continue
        name, pinned_version = match.groups()
        pins[_normalise_name(name)] = pinned_version
    return pins


def compare_versions(pins: Mapping[str, str], installed: Mapping[str, str | None]) -> list[str]:
    installed_normalised = {_normalise_name(name): value for name, value in installed.items()}
    issues = []
    for name, expected in sorted(pins.items()):
        actual = installed_normalised.get(_normalise_name(name))
        if actual is None:
            issues.append(f"{name} is not installed; expected {expected}")
        elif actual != expected:
            issues.append(f"{name} is {actual}; expected {expected}")
    return issues


def python_version_issue(version_info: tuple[int, int]) -> str | None:
    major, minor = version_info
    if (major, minor) == (3, 12):
        return None
    return f"Python {major}.{minor} is unsupported; use Python 3.12"


def main() -> int:
    issue_list = []
    python_issue = python_version_issue(sys.version_info[:2])
    if python_issue:
        issue_list.append(python_issue)

    pins = parse_pins((PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8"))
    installed = {}
    for name in pins:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    issue_list.extend(compare_versions(pins, installed))

    if issue_list:
        for issue in issue_list:
            print(f"ERROR: {issue}")
        return 1

    print(f"Environment matches Python 3.12 and {len(pins)} direct package pins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
