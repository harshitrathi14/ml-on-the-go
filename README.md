# ML on the Go

A full-stack ML sandbox that auto-generates loan/investment-style datasets, splits them into train/test/OOT/ETRC, benchmarks multiple algorithms, and visualizes results in a modern dashboard.

## What you get
- Synthetic dataset with 100+ variables (numeric + categorical) and a binary decision target.
- Automatic split into train/test/oot/etrc.
- Models: Logistic Regression, Random Forest, and XGBoost (fallback to Gradient Boosting if XGBoost is unavailable).
- Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curves.
- Frontend dashboard with leaderboard, model metrics, feature drivers, and diagnostics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

In another terminal:

```bash
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`.

## API
- `GET /health`
- `POST /train`

Example payload:

```json
{
  "n_rows": 5000,
  "n_features": 100,
  "n_categorical": 10,
  "seed": 42,
  "decision_labels": ["default", "no_default"]
}
```

## Notes
- The dataset generator produces a realistic signal-plus-noise target.
- OOT/ETRC are treated as extra holdout sets for robustness checks.
