import numpy as np
from sklearn.svm import SVC

def fit_classifier(X, y, C, gamma):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("Classifier features and labels must have matching rows.")
    if not np.isfinite(X).all() or np.unique(y).size < 2:
        raise ValueError("Classifier fitting requires finite features and at least two classes.")
    classifier = SVC(
        kernel="rbf", C=float(C), gamma=float(gamma), random_state=0,
        decision_function_shape="ovr", break_ties=True,
    )
    return classifier.fit(X, y)

def decision_scores(classifier, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2:
        raise ValueError("Classifier queries require a feature matrix.")
    if X.shape[0] == 0:
        return np.empty((0, len(classifier.classes_)), dtype=np.float32)
    scores = np.asarray(classifier.decision_function(X), dtype=np.float32)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    return scores

def decision_margins(classifier, X):
    scores = decision_scores(classifier, X)
    if scores.shape[1] < 2:
        raise ValueError("Decision margins require at least two classifier classes.")
    top_two = np.partition(scores, -2, axis=1)[:, -2:]
    return (top_two.max(axis=1) - top_two.min(axis=1)).astype(np.float32)
