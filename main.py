from pathlib import Path

from src.benchmark import run_benchmarks
from src.data_loader import load_iris_features_and_labels


def main():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "iris.data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_iris_features_and_labels(data_path)
    sklearn_results = run_benchmarks(X, y)
    own_results = run_benchmarks(X, y, classifier_kind="own")

    sklearn_results.to_csv(results_dir / "LibraryDecisionTreeClassifier_results.csv", index=False)
    own_results.to_csv(results_dir / "OwnDecisionTreeClassifier_results.csv", index=False)


if __name__ == "__main__":
    main()
