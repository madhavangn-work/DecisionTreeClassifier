# Decision Tree from Scratch vs. scikit-learn

A side-by-side benchmark of a **CART-style decision tree implemented from scratch in NumPy** against scikit-learn's `DecisionTreeClassifier`, swept across a grid of hyperparameters on the Iris dataset.

For each configuration, the project records training time, memory delta, CPU usage, and four classification metrics (accuracy, precision, recall, F1) so the two implementations can be compared apples-to-apples.

## What this project does

- Trains both classifiers across every combination of:
  - `test_size` ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
  - `max_depth` ∈ {2, 3, …, 10}
  - `min_samples_split` ∈ {2, 3, …, 10}
- That is **5 × 9 × 9 = 405 runs per implementation** (810 total).
- Writes one CSV per implementation to `results/`:
  - `OwnDecisionTreeClassifier_results.csv`
  - `LibraryDecisionTreeClassifier_results.csv`

The from-scratch tree (`src/DecisionTreeClassifier.py`) implements:

- Recursive binary splitting with **Gini impurity** as the split criterion
- `max_depth` and `min_samples_split` early-stopping
- Majority-class voting at leaf nodes
- A `print_tree` helper for inspecting the learned structure

## Project layout

```
.
├── data/
│   └── raw/iris.data                 # UCI Iris dataset (CSV, no header)
├── results/                          # Benchmark CSVs land here
├── src/
│   ├── DecisionTreeClassifier.py     # From-scratch CART implementation
│   ├── benchmark.py                  # Hyperparameter sweep + metric collection
│   ├── data_loader.py                # Iris loader (features/labels)
│   └── GenerateCSV.py                # Thin wrapper around run_benchmarks
├── main.py                           # Entry point: runs both sweeps, writes CSVs
├── pyproject.toml                    # Project metadata + dependencies (uv)
├── requirements.txt                  # Plain pip dependency list
└── uv.lock                           # Pinned lockfile for reproducible installs
```

## Requirements

- Python **3.14+** (see `.python-version` / `pyproject.toml`)
- Dependencies: `numpy`, `pandas`, `scikit-learn`, `psutil`, `pytest`

## Setup

Pick whichever workflow you prefer.

### Option A — `uv` (recommended, matches `uv.lock`)

```bash
uv sync
uv run python main.py
```

### Option B — `pip` + virtualenv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Running the benchmarks

From the project root:

```bash
python main.py
```

This will:

1. Load Iris from `data/raw/iris.data`.
2. Run the full hyperparameter sweep with **scikit-learn**'s `DecisionTreeClassifier`.
3. Run the same sweep with the **from-scratch** `DecisionTree`.
4. Write results to `results/LibraryDecisionTreeClassifier_results.csv` and `results/OwnDecisionTreeClassifier_results.csv`.

Each CSV row contains:

| Column | Description |
| --- | --- |
| `Test-Train Split (%)` | Test fraction × 100 |
| `Max_depth` | Tree depth limit |
| `Min_Sample_Split` | Minimum samples required to split a node |
| `Time taken to train classifier (sec)` | Wall-clock fit time |
| `Memory used (MB)` | RSS delta during fit |
| `CPU used` | CPU% delta during fit |
| `Accuracy Score` | `accuracy_score` on the test set |
| `Precision Score` | Weighted precision |
| `F1-Score` | Weighted F1 |
| `Recall Score` | Weighted recall |

Train/test splits use a fixed `random_state=2023` for reproducibility.

## Using the from-scratch tree directly

```python
import numpy as np
from src.DecisionTreeClassifier import DecisionTree

X = np.array([[2.7, 1.4], [1.5, 2.3], [3.1, 0.9], [0.8, 2.1]])
y = np.array([0, 1, 0, 1])

tree = DecisionTree(max_depth=3, min_samples_split=2)
tree.fit(X, y)

preds = tree.predict(np.array([[2.0, 1.0], [1.0, 2.0]]))
tree.print_tree(tree.root)
```

## Notes & caveats

- The from-scratch tree evaluates **every unique value of every feature** as a candidate threshold, so it is `O(n_features · n_unique_values · n_samples)` per split — fine on Iris, slow on large datasets.
- `Memory used (MB)` and `CPU used` are coarse process-level deltas via `psutil`; treat them as indicative, not authoritative microbenchmarks.
- Only Iris is included out of the box. Adding a new dataset is as simple as writing another loader that returns `(X, y)` as NumPy arrays and calling `run_benchmarks(X, y, classifier_kind=...)`.

## Tech stack

- **NumPy** — array math and the from-scratch tree internals
- **pandas** — data loading and result tables
- **scikit-learn** — reference `DecisionTreeClassifier` and metrics
- **psutil** — process-level memory / CPU sampling

## Author

Madhavan Govindan Namboothiri
