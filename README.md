# credit-risk-ifrs9-engine

![status](https://img.shields.io/badge/status-in%20progress-yellow)

PD model, IFRS 9 expected-credit-loss calculator, and SHAP-explained credit scorecard on the Home Credit dataset, served through a FastAPI endpoint and a Streamlit dashboard.

Framed as if deployed at a South African lender under SARB model-risk expectations — see `reports/model_card.md` once that lands.

## Architecture

```mermaid
flowchart LR
    A[application_train.csv] --> B[src/preprocessing.py\nleakage-safe split + impute]
    B --> C[src/baseline.py\nlogistic regression]
    B --> D[src/train_lgbm.py\nLightGBM]
    D --> E[src/scorecard.py\nWoE / optbinning]
    D --> F[src/explain.py\nSHAP + reason codes]
    D --> G[src/ecl.py\nIFRS 9 staged ECL]
    D --> H[api/main.py\nFastAPI /predict]
    H --> I[app/dashboard.py\nStreamlit]
    C -.mlflow.-> J[(MLflow tracking)]
    D -.mlflow.-> J
```

## Status

This is a work in progress, built module by module. The results table, scorecard sample, reason-code example, IFRS 9 example, and run instructions land here once each piece is real — this README gets rewritten once the pipeline runs end to end, not padded with placeholders now.
