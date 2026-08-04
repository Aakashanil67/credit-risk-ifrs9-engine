"""Run the baseline plus three LightGBM variants, each logged as its own MLflow run.

python run_experiments.py
mlflow ui   # opens http://127.0.0.1:5000 — compare all 4 runs side by side
"""

from src.baseline import run_and_log as run_baseline
from src.config import RANDOM_SEED, TARGET_COL
from src.data_loader import load_application_data
from src.features import build_lgbm_features
from src.preprocessing import split_data
from src.train_lgbm import PARAM_GRID, train_and_log_variant


def main() -> None:
    df = load_application_data()
    train, val, _test = split_data(df, seed=RANDOM_SEED)

    baseline_metrics, _model, _pred = run_baseline(train, val)
    print(f"logistic_baseline: {baseline_metrics}")

    train_X = build_lgbm_features(train)
    val_X = build_lgbm_features(val)
    train_y, val_y = train[TARGET_COL], val[TARGET_COL]

    for params in PARAM_GRID:
        run_name = f"lgbm_lr{params['learning_rate']}_leaves{params['num_leaves']}"
        metrics, best_iteration, _model = train_and_log_variant(
            train_X, train_y, val_X, val_y, params, run_name=run_name
        )
        print(f"{run_name} (stopped at {best_iteration} trees): {metrics}")


if __name__ == "__main__":
    main()
