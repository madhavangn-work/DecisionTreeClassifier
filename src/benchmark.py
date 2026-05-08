import time

import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.DecisionTreeClassifier import DecisionTree

TEST_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5]
MAX_DEPTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10]
MIN_SAMPLE_SPLITS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

RESULT_COLUMNS = [
    "Test-Train Split (%)",
    "Max_depth",
    "Min_Sample_Split",
    "Time taken to train classifier (sec)",
    "Memory used (MB)",
    "CPU used",
    "Accuracy Score",
    "Precision Score",
    "F1-Score",
    "Recall Score",
]


def build_classifier(classifier_kind, max_depth, min_samples_split):
    if classifier_kind == "own":
        return DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
    return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)


def evaluate_configuration(X, y, test_size, max_depth, min_samples_split, classifier_kind=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2023
    )

    current_process = psutil.Process()
    mem_before = current_process.memory_info().rss
    cpu_before = current_process.cpu_percent()

    classifier = build_classifier(classifier_kind, max_depth, min_samples_split)
    start_time = time.time()
    classifier.fit(X_train, y_train)
    end_time = time.time()

    time.sleep(0.01)
    mem_after = current_process.memory_info().rss
    cpu_after = current_process.cpu_percent()
    y_pred = classifier.predict(X_test)

    return {
        "Test-Train Split (%)": test_size * 100,
        "Max_depth": max_depth,
        "Min_Sample_Split": min_samples_split,
        "Time taken to train classifier (sec)": end_time - start_time,
        "Memory used (MB)": (mem_after - mem_before) / (1024**2),
        "CPU used": cpu_after - cpu_before,
        "Accuracy Score": accuracy_score(y_test, y_pred),
        "Precision Score": precision_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "Recall Score": recall_score(y_test, y_pred, average="weighted"),
    }


def run_benchmarks(X, y, classifier_kind=None):
    rows = []
    for test_size in TEST_SIZES:
        for max_depth in MAX_DEPTHS:
            for min_samples_split in MIN_SAMPLE_SPLITS:
                row = evaluate_configuration(
                    X=X,
                    y=y,
                    test_size=test_size,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    classifier_kind=classifier_kind,
                )
                rows.append(row)

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
