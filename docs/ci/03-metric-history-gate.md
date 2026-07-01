---

## title: Metric history & regression gate
description: How CI keeps per-test training metrics across runs, runs a two-layer gate against that history, and how to add a gate spec or clean a bad data point.

# Metric history & regression gate

CI keeps each test's per-metric numbers from every run in our own store and runs a two-layer gate against that history — catching the slow drift that fixed `--ci-<metric>` thresholds miss. wandb stays a write-only sink; the gate never reads from it. The baseline lives in our DB.

## Identity: what shares a baseline

The gate compares a number only against earlier numbers of the same kind, from the same test. Two keys decide that:

- **Run series** (the "same test"): `(test_path, backend, suite, test_file_hash)`. `test_file_hash` = sha256 of the test file's **contents**, so editing the test starts a fresh series. Runs differing on any field never share a baseline.
- **Value within a run**: `(metric_key, sub_label)`. `sub_label` separates points under one key (e.g. per-step `ppo_kl` at step 0, 1, …). Step-0 `ppo_kl` is compared only against past step-0 `ppo_kl`, never against `grad_norm`.

The store's baseline query keys on exactly these (plus a `limit` for how many recent points to read): `recent_trusted_values(test_path, backend, suite, metric_key, sub_label, test_file_hash, limit)`.

## Storage: two backends, two tables

- Backends: `SQLiteMetricHistoryStore` is the local/offline backend for unit tests and in-process development; `NeonMetricHistoryStore` is the hosted Postgres backend for CI/prod. Callers use only `MetricHistoryStore`: for the same inputs, both backends must persist the same run and metric fields, return the same trusted baseline rows in newest-first order, and revoke trust for the same runs via `mark_untrusted`.
- `write_run(...)` stores one CI run, its identity/provenance, its run-level `trusted` flag, and all metric values from that run.
- `recent_trusted_values(...)` returns the newest trusted values for one exact run series and one exact metric coordinate; this is the historical-gate baseline read.
- `mark_untrusted(...)` flips matching runs to `trusted = false` by `run_id`, `github_run_id`, or `commit_sha`, so the next baseline read excludes those runs without deleting rows.
- `runs` — one row per CI run of one series: the identity above + provenance (`commit_sha`, `pr_number`, `github_run_id`, `github_run_attempt`, `event_name`, `ref`) + `created_at` + `trusted` (run-level).
- `metric_values` — one row per value: `run_id` FK + `(metric_key, sub_label)` + `value`.
- Read path: composite index `runs(test_path, backend, suite, test_file_hash, trusted, created_at DESC)`.
- Hosted Postgres setup is out-of-band in this round: when `NeonMetricHistoryStore` is implemented, provision the equivalent two tables and application role outside this repo, and keep runtime gate code DML-only. Old-row cleanup policy is a later operational concern, not part of the M0/M1 substrate.

## The gate: two layers

After a test passes, each `(metric_key, sub_label)` value is checked with `|cur - ref| > max(rel * |ref|, abs_floor)` (`rel` default `0.20`; `abs_floor` only matters for metrics near zero, e.g. step-0 `ppo_kl`).

- **Hard gate** — always on. `ref` = a hardcoded safety limit. Runs even with zero history; generalizes today's `--ci-<metric>` thresholds.
- **Historical gate** — activates with ≥1 trusted point in the series. `ref` = mean of the series' trusted runs. Catches drift.
- **Cold start** (0 trusted): historical gate is inactive, hard gate only — not an error.

## Trust, cleanup, who writes

- A run is `trusted` iff it passed **all** active gates. A drifting run is still recorded, with `trusted = false`, so it can't drag the baseline. A test that fails then passes on **retry** is gated on its passing attempt's metrics and trusted normally — needing a retry is not itself a trust penalty.
- **Clean a bad point**: `mark_untrusted` = `UPDATE runs SET trusted = false` on the run. The next gate read excludes it immediately — no rebaseline, no row deletion.
- **Nightly-marked runs write baselines** — either the `schedule` cron (on `main`, post-merge) **or** a PR carrying the `nightly` label (the PR's own pre-merge code). Provenance (`event_name`, `pr_number`) records which, so a label-PR baseline is distinguishable from a post-merge one and can be `mark_untrusted`'d if it turns out bad. Ordinary (unlabeled) PR runs are read-only and only shadow.

## Collection

`CiHistoryBackend` runs alongside `WandbBackend` on the same `log()` fan-out and writes NDJSON snapshots under the harness-assigned per-test attempt directory. After the test passes, the later gate/finalizer consumes those records, assigns identity + provenance, runs the gate, and (on a nightly-marked run only) writes the rows. Nothing is read back from wandb.

## Rollout

Shadow-first: collect, store, and evaluate, but **never block a PR** initially — a historical-gate failure lands as an untrusted row and is surfaced, not enforced. Enforcement arrives later behind a per-test **allowlist** + a global **kill-switch**.

## Map: files & knobs


| Thing                    | Where                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| Enable capture           | set `MILES_CI_GATE_RECORD_DIR` (injected by the CI harness; no CLI flag)               |
| DB connection            | `NEON_DATABASE_URL` (CI secret)                                                        |
| Storage contract         | `tests/ci/metric_history/storage/store.py` (+ `storage/sqlite_store.py` offline, `storage/neon_store.py` prod) |
| Gate logic               | `tests/ci/history_gate.py`                                                             |
| Collection backend       | `miles/utils/tracking_utils/ci_history.py`                                             |
| Declare a gate on a test | `register_ci_gate(...)` in the test file                                               |




## Notes

- Any test-file edit is an intentional baseline reset for that series (the hash changes).
- The nightly trigger (`schedule` cron + `nightly` label) already shipped (#1491); detection here is harness-side via `GITHUB_EVENT_NAME`, so this feature needs **no** `pr-test.yml` **edit**.
- Open: should a brand-new test's first baselines need human confirmation before counting as trusted? (v1: no.) Per-series `rel` / `abs_floor` overrides beyond the global defaults.
