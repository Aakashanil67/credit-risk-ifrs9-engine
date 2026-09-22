# Monitoring runbook: public-demo risk model

## Purpose and boundary

This runbook describes how the aggregate monitoring outputs in this repository would be reviewed.
It supports an educational portfolio demonstration only. The project has no scheduled production
feed, no automated retraining, and no automated model promotion.

## Status response

| Status | Required action |
|---|---|
| Green | Record the batch result; no model change. |
| Amber | Investigate source-system changes, population mix, missingness, and score movement before the next review cycle. |
| Red | Suspend automatic model promotion, escalate to the model owner and independent validator, assess decision impact, and use an approved fallback policy if this were a real lender. |

The configured thresholds are illustrative: feature and score PSI, missing-rate movement, unseen
category rate, approval-rate movement, and delayed-label performance movement each contribute to
the batch status. They need local calibration before use in a lender.

## Governance roles

These are governance roles, not named people in this solo project.

| Role | Responsibility |
|---|---|
| Model developer | Reproduce diagnostics, document technical findings, and propose no change without evidence. |
| Model owner | Decides whether a finding affects the intended use and approves a controlled response. |
| Independent validator | Challenges evidence, methodology, and release decisions independently of development. |
| Credit policy owner | Assesses customer, portfolio, and policy impact; owns any fallback decision policy. |

## Investigation and recovery guide

| Signal | First checks | Escalation or recovery |
|---|---|---|
| Schema failure | Compare source fields, types, category levels, and versioned request contract. | Stop ingestion or scoring for incompatible batches; correct the source mapping and replay the aggregate check. |
| Unseen category | Confirm source-system code changes and mapping coverage. | Map only through an approved schema change; otherwise route to a controlled fallback or reject invalid input. |
| PSI or missingness drift | Check extract timing, population mix, source changes, and feature distributions. | For red status, suspend automatic promotion and assess decision impact before any model action. |
| Performance deterioration after labels arrive | Verify label maturity, outcome definition, and comparison window. | Escalate to the model owner and independent validator; do not retrain or promote automatically. |
| Service errors or latency | Inspect privacy-safe request logs, health metadata, deployment status, and dependency changes. | Roll back to the approved prior service release or use an approved fallback policy. |
| Model rollback | Confirm the approved bundle and service release, then smoke-test health and a valid request. | Record the incident, reason, approver, and follow-up validation before re-release. |

## Evidence retained by this project

The monitoring reference and replay reports retain only aggregate distributions and batch-level
metrics. The reference window contains 51,503 held-out test rows in aggregate; each demonstration
batch contains 10,000 disjoint replay rows. They do not retain applicant rows, predictions, targets,
identifiers, payloads, or IP addresses. The replay and stress batches are simulations built from
historical Home Credit data, not production observations.
