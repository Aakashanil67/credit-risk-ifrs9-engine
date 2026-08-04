# IFRS 9 expected credit loss — portfolio summary

LGD assumption: 45% (loss given default — the share of exposure not recovered after a default). EAD: outstanding credit amount at application (`AMT_CREDIT`).

**Total exposure (EAD): R36,753,744,528**
**Total provision (ECL): R1,328,317,872**
**Portfolio coverage ratio: 3.614%** of exposure held as provision.

## By IFRS 9 stage

| stage | loans | EAD (R) | ECL (R) | coverage % |
|---|---|---|---|---|
| 1 — performing | 58,854 | 34,933,557,105 | 1,121,206,522 | 3.210% |
| 2 — SICR (lifetime ECL) | 2,494 | 1,746,208,624 | 183,421,016 | 10.504% |
| 3 — credit-impaired | 154 | 73,978,798 | 23,690,333 | 32.023% |

## Reading this in IFRS 9 language

- **Stage 1** loans get 12-month ECL — the expected loss from default events plausible in the next 12 months.
- **Stage 2** loans have shown a significant increase in credit risk (SICR) since origination and move to lifetime ECL — expected loss over the full remaining term, which is materially larger per loan than a 12-month provision.
- **Stage 3** loans are credit-impaired (PD >= 50%) — also lifetime ECL, and flagged separately for disclosure as IFRS 9 requires.

**Limitation, stated plainly**: SICR is proxied by comparing the tuned LightGBM model's PD ("current") against the logistic baseline's PD ("origination") for the same applicant, because this dataset is one static snapshot with no repeat observations of the same loan over time. A production system would compare a loan's PD today against its own PD at the actual origination date, not two different models' opinions of the same application. Treat the stage split here as illustrative of the mechanics, not a real portfolio's risk migration.
