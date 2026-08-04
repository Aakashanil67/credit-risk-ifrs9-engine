"""Load the Home Credit application table and report its shape before anything touches it."""

from pathlib import Path

import pandas as pd

from src.config import RAW_TRAIN_PATH, REPORTS_DIR, TARGET_COL


def load_application_data(path: Path = RAW_TRAIN_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — download application_train.csv from the Kaggle "
            "Home Credit Default Risk competition and place it in data/."
        )
    return pd.read_csv(path)


def target_balance(df: pd.DataFrame) -> pd.Series:
    counts = df[TARGET_COL].value_counts(normalize=True).sort_index()
    counts.index = ["repaid (0)", "defaulted (1)"]
    return counts


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})


def write_data_dictionary(df: pd.DataFrame, missing: pd.DataFrame, out_path: Path) -> None:
    balance = target_balance(df)
    lines = [
        "# Data dictionary — application_train.csv",
        "",
        f"{df.shape[0]:,} rows, {df.shape[1]} columns. One row is one loan application; "
        f"`{TARGET_COL}` is 1 if the client had a payment more than X days late on at "
        "least one installment (Home Credit's definition of default), 0 otherwise.",
        "",
        f"Target balance: {balance['repaid (0)']:.1%} repaid, {balance['defaulted (1)']:.1%} "
        f"defaulted — a {balance['repaid (0)'] / balance['defaulted (1)']:.1f}:1 imbalance. "
        "Accuracy is meaningless here: a model that predicts 'repaid' for every applicant "
        f"scores {balance['repaid (0)']:.1%} accuracy while catching zero defaults.",
        "",
        f"## Missing values ({len(missing)} of {df.shape[1]} columns affected)",
        "",
        "| column | missing % |",
        "|---|---|",
    ]
    for col, row in missing.head(30).iterrows():
        lines.append(f"| {col} | {row['missing_pct']}% |")
    if len(missing) > 30:
        lines.append(f"| ... {len(missing) - 30} more columns with missing values | |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_application_data()
    print(f"shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"target balance:\n{target_balance(df)}")

    missing = missing_value_report(df)
    print(f"{len(missing)} columns have missing values, worst: "
          f"{missing.index[0]} ({missing.iloc[0]['missing_pct']}%)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_data_dictionary(df, missing, REPORTS_DIR / "data_dictionary.md")


if __name__ == "__main__":
    main()
