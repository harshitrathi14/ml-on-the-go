# Nu Score On the Go — Usage Guide

**Build ML models with your data.** An internal Northern Arc / AltiFi platform that turns a loan book, bureau pulls and bank-statement variables into a validated, explainable Nu Score — and lets you score new applications the same minute.

| | |
|---|---|
| **Audience** | Credit, risk, analytics and business teams; no data-science background assumed |
| **Access** | `http://10.91.16.86/nuscore_claudecode/` (internal network only; no login) |
| **Version** | Platform v0.5 · Engine cohorts v2 · Nu Score scale v1 (0–1000) |
| **Owner** | Analytics / AI team, Northern Arc |

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [What the product does](#2-what-the-product-does)
3. [Preparing your data](#3-preparing-your-data)
4. [Step-by-step walkthrough](#4-step-by-step-walkthrough)
5. [Reading the results](#5-reading-the-results)
6. [The Nu Score](#6-the-nu-score)
7. [Scoring new applications](#7-scoring-new-applications)
8. [How the engine works](#8-how-the-engine-works)
9. [Governance, privacy and limits](#9-governance-privacy-and-limits)
10. [Troubleshooting and FAQ](#10-troubleshooting-and-faq)
11. [Glossary](#11-glossary)
- [Appendix A — API reference](#appendix-a--api-reference)
- [Appendix B — Operations](#appendix-b--operations)

---

## 1. Executive summary

**The problem.** Building a credit model today takes a data-science team weeks: joining sources, cleaning formats, choosing an algorithm, validating it honestly and packaging it so an underwriter can use it. Most lending teams never get past a spreadsheet.

**The product.** Nu Score On the Go compresses that cycle into minutes. A user uploads the data they already have, names the outcome they want to predict, and the engine does the rest: matches identifiers across files, screens for leakage and personal data, tunes nine model families on GPUs, selects a champion under a validation discipline that a model-risk reviewer would accept, and calibrates the result into a 0–1000 **Nu Score** with risk bands. The trained model is immediately available for scoring new applications — from a form or a batch file — with plain-English reason codes.

**Three things to remember.**

1. **Bring one common identifier.** Loan dump, bureau and bank-statement files just need a shared loan ID / account number; column names can differ, the engine finds the match and shows you the match rate.
2. **The outcome column is yours to choose.** Default flag, 90+ DPD, write-off, churn — anything binary or reducible to binary.
3. **Nothing is stored.** Data is deleted the moment results are delivered; only aggregate metrics and the trained model bundle remain. No PII is needed — and PII-looking columns are dropped automatically.

**Commercials.** Free during the introductory period; thereafter ₹10 per case scored, covering compute on Northern Arc's own servers. Model building, validation, reports and bands are included.

---

## 2. What the product does

```
 Upload sources ─► Match identifiers ─► Analyse & screen ─► Train model zoo ─► Pick champion ─► Nu Score bands ─► Score new applications
```

| Capability | What you get |
|---|---|
| **Multi-source intake** | CSV, bureau JSON (nested trade lines flattened automatically), Parquet; up to 20 GB per file; uploads resume if the connection drops |
| **Identifier matching** | Roles (loan dump / bureau / bank statement) proposed by a local language model; join keys chosen by measured value overlap, shown with match rate and rows-per-case |
| **Data screening** | PII, leakage and identifier columns flagged and excluded by default; Information Value (IV) with WOE bin tables; input clean-up log (₹, %, lakh separators, yes/no) |
| **Model zoo** | Logistic regression, Random Forest, Extra Trees, Histogram GB, XGBoost (GPU), LightGBM, CatBoost, MLP, FT-Transformer (PyTorch) — each tuned with Optuna |
| **Honest validation** | Five cohorts: train / validation / calibration / test / out-of-time. Champion chosen only from the first three; test and OOT are untouched estimates |
| **Nu Score** | Calibrated probability → 0–1000 score (higher = safer) → A+ … E bands learned from the data; cut-off simulator |
| **Explainability** | SHAP drivers, per-record reason codes, plain-English glossary, AI-written narrative |
| **Scoring** | Form for single applications, batch CSV with automatic column harmonisation, drift check on every batch |
| **Reports** | HTML and PDF reports for committees; downloadable scored files |

Use cases beyond credit: churn, collections prioritisation, sales conversion, inventory stock-outs, fraud flags — any table with a binary outcome.

---

## 3. Preparing your data

### 3.1 What to bring

| Source | Status | Typical columns |
|---|---|---|
| **Loan dump** (main table, one row per loan/case) | Required | `loan_id` / `account_no`; outcome to predict (default flag, DPD bucket, write-off, closure); product, amount, tenure, rate, EMI, LTV, disbursal date; repayment history (DPD, overdue, months on book); applicant profile without PII (age band, income, occupation, pincode/district) |
| **Bureau features** | Recommended | Bureau score, enquiries (3/6/12 m), live & closed trade lines, exposure, utilisation, worst DPD, write-offs/settlements, vintage, product mix — flat CSV **or the raw bureau JSON** keyed by the same identifier |
| **Bank-statement variables** | Recommended | Average / minimum / month-end balances; credit & debit counts and amounts; salary regularity; cheque/EMI bounces; cash withdrawals; existing EMI outflow. Monthly rows are fine (the latest month per case is used) |
| **Other signals** | Optional | Collections outcomes, GST/ITR summaries, application-stage decisions, sourcing channel; monthly snapshots enable true out-of-time testing |

### 3.2 Rules of the road

- **One common identifier across files.** Names can differ (`loan_id`, `ref_no`, `account_number`); the engine finds the match and reports the overlap. Aim for > 80 % of loans matched.
- **Outcome column.** A 0/1 flag, yes/no, a class label, or a number you want thresholded (e.g. DPD > 90). You confirm the positive class on screen.
- **Time column.** A disbursal / snapshot date unlocks *true* out-of-time validation. Without one, the "OOT" cohort is a random hold-out and the UI says so.
- **Formats.** CSV (comma, UTF-8), JSON / JSON-lines, Parquet. Values like `₹1,20,000`, `2.5 lakh`, `25%`, `yes/no`, `NA` are normalised automatically and logged.
- **Size.** At least 100 rows and 2 columns; realistically ≥ 2,000 labelled cases with ≥ 100 positives for a stable score. Up to 20 GB per file; training uses up to 1 M rows, evaluation up to 3 M.

### 3.3 Keep out

Do **not** upload: customer name, mobile, email, full address, PAN, Aadhaar, date of birth, or any account number other than the loan identifier. Anything that slips through is flagged as PII and dropped before training.

Also avoid columns that are only known *after* the outcome (write-off amount, settlement flag, closure date). These are leakage; the engine flags them (name patterns and implausibly high IV) and excludes them by default.

---

## 4. Step-by-step walkthrough

### Step 1 — Upload your sources
1. Open the portal and stay on **Upload your data**.
2. Drop one or more files (or click to choose). Each file shows an upload bar, then "Reading and profiling…", then its row/column counts.
3. Remove a file with **×**; add more at any time. Uploads resume automatically after a network drop.

### Step 2 — Analyse files & match identifiers
1. Click **Analyse files & match identifiers**.
2. Each file receives a role badge (Main table / Bureau / Bank statement / Other).
3. In **How your files connect**, confirm the main table and, per secondary file, the key pair (`loan_id = ref_no`) with its match rate and rows-per-case. Override the selects if needed.
4. Click **Confirm & build the modelling table**. With a single file, this step is skipped.

### Step 3 — Choose the outcome
1. Pick the **Outcome column** (the engine pre-selects a likely candidate) and, for text outcomes, the **Positive class** (e.g. `default`, `yes`, `1`).
2. Click **Analyse target**. You will see: positive rate, how the outcome is binarised, the **time column** for out-of-time splits, **columns we recommend leaving out** (PII / leakage / identifier, each with a reason — tick to keep one), the **IV chart** and the **missing-values chart**.
3. Choose a **speed tier**: *Quick* (~3 min on 20 k rows, few tuning trials), *Balanced* (default), *Thorough* (widest search).
4. Click **Train all models**. Progress is live: "XGBoost: trial 12/25 · best CV AUC 0.741", then blend → calibration → SHAP → save.

### Step 4 — Review results
See [Section 5](#5-reading-the-results). Use **Explain with AI** for a narrative, **Download HTML/PDF report** for a committee pack.

### Step 5 — Score new applications
See [Section 7](#7-scoring-new-applications).

> **Synthetic loan book.** The second tab generates a realistic retail loan book (32 credit variables such as bureau score, enquiries, worst DPD, FOIR, LTV, bank balances, bounces) so you can explore the product without data. It is a demo, clearly labelled as such.

---

## 5. Reading the results

### 5.1 Champion card
- **Champion** and *why it won*: a sentence summarising its composite score, strongest components and validation / test / OOT AUC.
- **KPIs**: Validation AUC (used for selection), Test AUC with a bootstrap 95 % interval, OOT AUC tagged *true out-of-time* or *random hold-out*, KS, PR-AUC, Brier, ECE, overfit gap, OOT drop.
- **Composite breakdown**: five bars — discrimination 45 %, calibration 20 %, consistency 15 %, overfit 10 %, simplicity 10 % — computed only on train / validation / calibration.

### 5.2 Validation notes
Amber, plain-English caveats generated from the data: weak AUC (< 0.60), overfit gap > 0.10, validation→OOT drop > 0.05, calibration that worsened Brier, duplicate predictor rows between train and test, constant features, highly correlated pairs, small cohorts. Read these before trusting a score.

### 5.3 All models developed
Every model with composite, validation / test / OOT AUC, KS, PR-AUC, Brier, overfit and OOT-drop tags, tuning trials and best CV AUC, fit time. Click a row for its description, best parameters, Optuna history and per-cohort metrics. Failed models are listed with the error. **NuScore Blend** is the stacked ensemble (logistic meta-learner on out-of-fold predictions of the top four models).

### 5.4 Most important variables
IV per variable with strength (useless / weak / medium / strong / suspicious), the champion's importance, type, missing %, notes, and a **What it means** column from the glossary. Click a row for the WOE bin table (count, events, event rate, WOE, IV contribution). IV above 0.5 deserves a leakage check.

### 5.5 Charts
ROC overlay (champion highlighted), stability across the five cohorts, calibration reliability (before vs after), lift/gains by decile, model comparison, top drivers, confusion matrix at the F1-optimal threshold, SHAP beeswarm and dependence plots.

### 5.6 Diagnostics, drift, rows without an outcome
- **Diagnostics**: bootstrap interval, duplicate rows, constants, correlated pairs, cohort sizes and positives.
- **Stability & drift**: PSI of the score and each feature per cohort versus training (< 0.10 stable, 0.10–0.25 moderate, > 0.25 significant).
- **Rows without an outcome**: cases whose outcome was blank are scored with the final model ("scoring only"), summarised by band and downloadable; they never enter a cohort.

### 5.7 Dataset card
Rows, features, cohort sizes with date ranges, split strategy and OOT status, speed tier, training time, and the **Input clean-up applied** log.

---

## 6. The Nu Score

### 6.1 Scale
- **Range** 0–1000, **higher = safer**.
- **Anchor** 500 = even odds (50 % probability of the adverse outcome).
- **PDO** 60 points: every +60 halves the odds of going bad.
- Computed from the **calibrated** probability (Platt scaling fitted on the calibration cohort), so the score reflects observed bad rates, not raw model output.
- **AI-enhanced**: the probability comes from the stacked blend; the UI shows the champion-only *base score* and the *AI adjustment* separately so the ensemble's effect is visible and governable.

### 6.2 Bands
Six bands **A+, A, B, C, D, E** whose cut-offs are *learned from your data* on the calibration cohort: quantile seeds, then adjacent bands are merged until the bad rate rises monotonically from A+ to E and every band holds at least 5 % of cases. Bands are frozen before test / OOT evaluation. The band table shows, per cohort: score range, share, bad rate, lift, cumulative bad capture and PSI contribution.

### 6.3 Cut-off simulator
Drag the cut-off: approval rate, bad rate among approved and share of bads rejected update live from the test cohort. Use it to translate risk appetite into a policy line before any lending decision — bands are analytical groupings, not approval rules.

### 6.4 Reason codes
For every scored record, the top risk-increasing SHAP contributions, each with the glossary label, the raw variable and its value — adverse-action style.

---

## 7. Scoring new applications

### 7.1 Single application (form)
The form is generated from the trained model's input schema: numeric fields prefilled with training medians and typical ranges, categorical fields as drop-downs (with *Other…*). Click **Score** to get the gauge (0–1000, band colour), probability, base vs AI-adjusted score and reason codes.

### 7.2 Batch file
Drop a CSV of applications. The engine **harmonises columns** automatically — exact names, then normalised names (case, punctuation, units such as "(Rs)", abbreviations, synonyms like *CIBIL → bureau_score*), then the local language model for anything left — and shows the mapping table with a method badge per column (exact / normalised / llm / missing). Values are normalised too (`1,20,000`, `2.5 lakh`, `yes/no`). Missing inputs fall back to training medians/modes and are counted.

The response includes a **drift check** (score PSI and top drifting inputs vs the training population) and a **download** of the scored CSV with `nu_score`, `nu_band`, `probability`, `base_score` and `reason_1…4` appended. Batch files are limited to 200 MB.

### 7.3 Where the model lives
Each training job persists a **model bundle** (preprocessing, champion, blend members, calibrator, bands, SHAP background, drift profiles, glossary) under the job's folder. The training data itself is deleted. Bundles are addressed by `model_id` (shown in the UI and the result JSON).

---

## 8. How the engine works

### 8.1 Cohorts
| Cohort | Share | Purpose |
|---|---|---|
| Train | 55 % | Fitting and hyper-parameter tuning (3-fold stratified CV inside) |
| Validation | 12 % | Model selection, thresholds |
| Calibration | 10 % | Platt calibration, band cut-offs |
| Test | 10 % | Untouched final estimate (bootstrap CI) |
| Out-of-time | 13 % | Latest observation dates when a time column exists — rows sharing a date are never split across cohorts; otherwise a random hold-out labelled as such |

### 8.2 Preprocessing
Median imputation with missing-value indicators; one-hot encoding for low-cardinality categoricals with rare-level pooling; out-of-fold target encoding for high-cardinality ones (pincode, branch); date columns converted to month and days-to-reference; free-text / near-unique text columns dropped. The fitted schema (types, medians, categories) is what the scoring form and harmoniser use.

### 8.3 Model zoo and tuning
Nine families (see Section 2) each get an Optuna TPE search with median pruning and early stopping under the tier's budget (Quick 4 trials / 2 folds, Balanced 25 / 3, Thorough 60 / 3). Tuning runs on up to 200 k rows; final fits on up to 1 M. XGBoost trains on GPU; the FT-Transformer (feature-tokenizer transformer with a CLS readout) trains in PyTorch with early stopping on validation AUC. Models without native importances get permutation importance (validation AUC drop).

### 8.4 Selection and blend
Composite score on selection-safe cohorts only: validation AUC (45 %), validation Brier (20 %), validation→calibration consistency incl. score PSI (15 %), train→validation overfit gap (10 %), simplicity (10 %). The top four models are stacked with a logistic meta-learner on out-of-fold predictions; the blend competes in the leaderboard and supplies the AI-enhanced probability.

### 8.5 Calibration, bands, explainability, drift
Platt scaling on the calibration cohort → Nu Score scaling → learned bands → evaluation on test / OOT. SHAP (TreeExplainer / LinearExplainer / permutation) on the champion; PSI drift per cohort and per scored batch; bootstrap AUC interval and data checks feed the warning engine.

### 8.6 Local AI
Two local language models (Ollama, on-prem, nothing leaves the host) help with **format and language**, never with numbers: assigning file roles and suggesting keys (verified by measured overlap), mapping differently named columns at scoring time, and writing the narrative explanation from aggregate metrics. Only column names, types and summary statistics are ever sent — never rows.

---

## 9. Governance, privacy and limits

### 9.1 Privacy by construction
- Uploaded files are converted to Parquet on the server and **deleted as soon as the training job succeeds**; abandoned sessions expire after 24 hours.
- No PII is required; PII-looking columns (by name and by value pattern: emails, mobiles, PAN, Aadhaar) are excluded by default and never reach the model.
- Category labels in WOE tables are anonymised; reports and the UI contain aggregates only.
- Access is restricted to internal networks at the reverse proxy; there is no application login by design.

### 9.2 Model-risk discipline
- Test and OOT cohorts never influence fitting, selection or calibration; the UI states this next to the champion.
- Every result carries validation notes; the OOT status is explicit ("true out-of-time" vs "random hold-out").
- Selection, calibration and bands are reproducible from the seed; parameters and Optuna trials are retained.
- Bands are analytical groupings, not lending policy; the cut-off simulator is a planning tool.

### 9.3 Current limits
- Binary outcomes only (regression and multi-class are on the roadmap).
- One row per case in the main table; monthly secondary files are reduced to the latest row per case (aggregation windows are on the roadmap).
- Repeated borrowers across cohorts are not grouped; if the same customer appears many times, treat OOT results with care.
- The local LLM steps take 10–30 s; training time scales with rows × tier (Quick ≈ 3 min on 20 k rows).

---

## 10. Troubleshooting and FAQ

| Symptom | Cause / fix |
|---|---|
| "Only .csv, .json/.jsonl and .parquet files are accepted" | Rename or convert Excel files to CSV first |
| "CSV must have at least 100 rows" | The file is too small or was not parsed (check delimiter and encoding) |
| Match rate below 50 % | The key columns hold different identifiers or formats (leading zeros, prefixes); pick another pair in the join step |
| "Target … could not be reduced to exactly 2 classes" | Set the positive class explicitly, or choose a threshold for numeric outcomes |
| OOT shows "random hold-out" | No usable time column: choose one in the target step, or accept stability-only OOT |
| Validation note "Test discrimination is weak" | The variables carry little signal; add bureau / bank features or more history |
| Batch scoring shows "missing" columns | Rename to match the schema or accept median fallback; download the schema from the form |
| Training seems slow | Use *Quick* for exploration; two jobs share the host, so a queued job waits for a free worker |

**Can I keep my data on the platform?** No — by design it is deleted after training. Keep your own copy; the model bundle stays.
**Can I retrain later with new data?** Yes: upload again and train; each job yields a new model bundle.
**Is the Nu Score a bureau score?** No. It is your portfolio's own calibrated score; 500 = even odds on *your* outcome definition.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **AUC / Gini** | Ranking power of the score (0.5 random, 1.0 perfect); Gini = 2·AUC − 1 |
| **KS** | Maximum separation between cumulative good and bad distributions |
| **PR-AUC** | Precision-recall area; more informative than AUC on rare outcomes |
| **Brier / ECE** | Probability accuracy; lower is better (ECE = expected calibration error) |
| **IV / WOE** | Information Value and Weight of Evidence: univariate strength of a variable; IV < 0.02 useless, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong, > 0.5 suspicious |
| **PSI** | Population Stability Index: distribution shift versus training; < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant |
| **SHAP** | Per-prediction attribution of the score to each variable |
| **Platt scaling** | Logistic calibration of raw model log-odds to observed probabilities |
| **PDO** | Points to double the odds (60 on the Nu Score scale) |
| **Cohort** | A disjoint slice of the data with one job (train, validation, calibration, test, OOT) |
| **Leakage** | A variable that reveals the outcome because it is only known afterwards |
| **Overfit gap** | Train AUC minus validation AUC; > 0.10 is a warning |

---

## Appendix A — API reference

All routes are served under `/nuscore_claudecode/api` (and at the root for the mobile app).

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness |
| `/sessions` | POST | Create an upload session → `session_id`, `chunk_bytes` |
| `/sessions/{sid}/files` | POST | Init a file `{filename, size_bytes, chunk_bytes}` → `file_id`, `total_chunks` |
| `/sessions/{sid}/files/{fid}/chunks/{i}` | PUT | Upload chunk *i* (raw body); out-of-order → 409 with `received_chunks` |
| `/sessions/{sid}/files/{fid}/complete` | POST | Convert to Parquet and profile |
| `/sessions/{sid}/analyse` | POST | Roles, suggested target/time column, join key candidates with match rates |
| `/sessions/{sid}/join` | POST | `{primary_file_id, joins:[{file_id, primary_col, file_col}]}` → joined profile |
| `/sessions/{sid}/target` | POST | `{target_col, positive_class?, threshold?}` → IV, flags, time column, split strategy |
| `/sessions/{sid}/train` | POST | `{target_col, exclude_cols, time_col, tier, seed}` → 202 `job_id` |
| `/jobs/train` | POST | Synthetic loan book `{n_rows, default_rate, n_features, tier}` → 202 `job_id` |
| `/jobs/{id}` · `/jobs/{id}/result` | GET | Live status (`message`, `progress.stage`) · full result JSON |
| `/models/{model_id}/schema` | GET | Scoring form fields, bands, champion |
| `/models/{model_id}/score` | POST | `{records:[{...}]}` → score, band, probability, base score, AI adjustment, reasons |
| `/models/{model_id}/score-file?filename=` | POST | Raw CSV body → mapping, band counts, drift, `file_id` |
| `/models/{model_id}/score-file/{file_id}/download` | GET | Scored CSV |
| `/explain` · `/report/html` · `/report/pdf` | POST | AI narrative · reports from a result JSON |
| `/uploads?filename=` · `/upload-csv` · `/train-csv` | POST | Single-file compatibility endpoints (mobile app) |

## Appendix B — Operations

| Item | Value |
|---|---|
| Service | `nuscore-otg.service` (systemd; hardened: MemoryMax 200 G, CPUQuota, NoNewPrivileges, PrivateTmp) |
| Process | uvicorn on `127.0.0.1:8096`, training in a 2-worker process pool, 16 threads per job |
| Reverse proxy | nginx snippet `/etc/nginx/snippets/nuscore-otg.conf` on the `10.91.16.86` server blocks; internal CIDRs only; 20 GB body limit |
| Environment | `/etc/nuscore-otg.env` (template `deploy/nuscore-otg.env.example`): `ROOT_PATH`, `DATA_ROOT`, `AI_PROVIDER=ollama`, `AI_MODEL=qwen2.5:7b`, `CUDA_VISIBLE_DEVICES=2,3`, `JOB_THREADS=16`, caches on `/mnt` |
| Data & logs | `/mnt/harshitrathi/mlpipeline-data/{sessions,jobs,logs}`; sessions deleted after training, 24 h TTL otherwise |
| GPUs | 4 × Tesla T4: 0–1 reserved for Ollama, 2–3 for training |
| Deploy | `sudo systemctl restart nuscore-otg && deploy/smoke_test.sh http://10.91.16.86/nuscore_claudecode` |
| Tests | `.venv/bin/python -m pytest tests -q` · `deploy/session_test.py <url> input/multisource` |
| Repository | `github.com/harshitrathi14/ml-on-the-go`, branch `main` |
