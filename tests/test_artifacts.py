import json

import pytest

from src.artifacts import ArtifactBundle, load_artifact_bundle, save_artifact_bundle


class ToyModel:
    feature_name_ = ["income", "credit"]


def test_artifact_bundle_round_trips_model_contract_and_metadata(tmp_path):
    """Serving must receive the same schema and category map that training wrote."""
    bundle = ArtifactBundle(
        model=ToyModel(),
        category_dtypes={"contract": ["Cash loans", "Revolving loans"]},
        metadata={"profile": "public_demo", "feature_names": ["income", "credit"]},
    )

    save_artifact_bundle(bundle, tmp_path)
    loaded = load_artifact_bundle(tmp_path)

    assert loaded.model.feature_name_ == ["income", "credit"]
    assert not (tmp_path / "train_medians.joblib").exists()
    assert json.loads((tmp_path / "metadata.json").read_text()) == bundle.metadata


def test_artifact_bundle_rejects_metadata_that_disagrees_with_the_model_schema(tmp_path):
    """A mismatched feature list would otherwise produce a silent serving-time scoring error."""
    bundle = ArtifactBundle(
        model=ToyModel(),
        category_dtypes={},
        metadata={"profile": "public_demo", "feature_names": ["credit", "income"]},
    )

    save_artifact_bundle(bundle, tmp_path)

    with pytest.raises(ValueError, match="feature_names"):
        load_artifact_bundle(tmp_path)
