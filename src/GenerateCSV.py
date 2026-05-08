from src.benchmark import run_benchmarks


def analytics(X, y, classifier=None):
    return run_benchmarks(X, y, classifier_kind=classifier)