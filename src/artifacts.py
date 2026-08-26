"""Read and write model bundles with the contract needed by a scoring service."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ArtifactBundle:
    model: Any
    category_dtypes: dict[str, Any]
    metadata: dict[str, Any]


def save_artifact_bundle(bundle: ArtifactBundle, directory: Path) -> None:
    """Persist all serving inputs together, rather than relying on neighbouring loose files."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, directory / "model.joblib")
    joblib.dump(bundle.category_dtypes, directory / "category_dtypes.joblib")
    (directory / "metadata.json").write_text(
        json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_artifact_bundle(directory: Path) -> ArtifactBundle:
    """Load a bundle and fail early when its declared feature schema is inconsistent."""
    required = [
        directory / "model.joblib",
        directory / "category_dtypes.joblib",
        directory / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model bundle files: {missing}")

    bundle = ArtifactBundle(
        model=joblib.load(directory / "model.joblib"),
        category_dtypes=joblib.load(directory / "category_dtypes.joblib"),
        metadata=json.loads((directory / "metadata.json").read_text(encoding="utf-8")),
    )
    expected_features = bundle.metadata.get("feature_names")
    model_features = getattr(bundle.model, "feature_name_", None)
    if model_features is not None and expected_features != list(model_features):
        raise ValueError("metadata feature_names do not match the fitted model")
    return bundle
