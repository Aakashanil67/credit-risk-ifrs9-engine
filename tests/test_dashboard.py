from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).parents[1] / "app" / "dashboard.py"


@pytest.mark.filterwarnings("error:Starting a Matplotlib GUI outside of the main thread.*")
def test_local_fallback_scores_without_streamlit_warnings(monkeypatch) -> None:
    """The fallback result must not bury its explanation beneath framework warnings."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    plt.switch_backend("svg")
    app = AppTest.from_file(DASHBOARD).run(timeout=30)

    app.button[0].click().run(timeout=30)

    assert matplotlib.get_backend().lower() == "agg"
    assert not app.exception
    assert not app.warning
    assert any("scoring directly against the saved model" in item.value for item in app.info)
    assert any("PD " in item.value for item in app.success)
