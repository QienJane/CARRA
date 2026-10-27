import copy

import numpy as np

from .classifier import decision_margins
from .quantiles import new_sketch, quantile, update_sketch

_CLASS_FIELDS = (
    "proto_mean", "proto_var", "principal_axes_top2", "sample_count",
    "radius_q95", "margin_q05", "boundary_density", "old_confusion",
    "radius_sketch", "margin_sketch",
)

def _feature_rows(X, y, dimension=None):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.size:
        raise ValueError("Features and labels must have matching rows.")
    if dimension is not None and X.shape[1] != dimension:
        raise ValueError("Feature dimensions differ from the geometry memory.")
    if not np.isfinite(X).all():
        raise ValueError("Geometry memory requires finite features.")
    return X, y

def _update_class(memory, class_id, features, classifier, boundary_threshold):
    features64 = features.astype(np.float64)
    batch_count = features.shape[0]
    batch_mean = features64.mean(axis=0)
    centered = features64 - batch_mean
    batch_scatter = centered.T @ centered
    old_count = int(memory["sample_count"].get(class_id, 0))
    count = old_count + batch_count
    if old_count:
        old_mean = np.asarray(memory["proto_mean"][class_id], dtype=np.float64)
        old_cov = np.asarray(memory["proto_var"][class_id], dtype=np.float64)
        difference = batch_mean - old_mean
        mean = old_mean + difference * (batch_count / count)
        scatter = ((old_count - 1) * old_cov + batch_scatter
                   + np.outer(difference, difference) * old_count * batch_count / count)
    else:
        mean = batch_mean
        scatter = batch_scatter
        for stream, field in enumerate(("radius_sketch", "margin_sketch")):
            seed = np.random.SeedSequence([memory["seed"], class_id, stream])
            memory[field][class_id] = new_sketch(seed)
    covariance = scatter / max(count - 1, 1)
    covariance = (covariance + covariance.T) * 0.5
    _, eigenvectors = np.linalg.eigh(covariance)
    margins = decision_margins(classifier, features)
    errors = np.count_nonzero(classifier.predict(features) != class_id)
    boundary_count = np.count_nonzero(margins < boundary_threshold)
    memory["old_confusion"][class_id] = float(
        (old_count * memory["old_confusion"].get(class_id, 0.0) + errors) / count
    )
    memory["boundary_density"][class_id] = float(
        (old_count * memory["boundary_density"].get(class_id, 0.0) + boundary_count) / count
    )
    memory["sample_count"][class_id] = count
    memory["proto_mean"][class_id] = mean.astype(np.float32)
    memory["proto_var"][class_id] = covariance.astype(np.float32)
    memory["principal_axes_top2"][class_id] = eigenvectors[:, -2:][:, ::-1].astype(np.float32)
    residuals = np.linalg.norm(features64 - mean, axis=1)
    update_sketch(memory["radius_sketch"][class_id], residuals)
    update_sketch(memory["margin_sketch"][class_id], margins)
    memory["radius_q95"][class_id] = quantile(memory["radius_sketch"][class_id], 0.95)
    memory["margin_q05"][class_id] = quantile(memory["margin_sketch"][class_id], 0.05)

def initialize_memory(X, y, clf, seed=0, boundary_threshold=0.25):
    X, y = _feature_rows(X, y)
    if X.shape[0] == 0 or X.shape[1] < 2:
        raise ValueError("Memory initialization requires samples and at least two feature dimensions.")
    class_ids = sorted(int(value) for value in np.unique(y))
    if seed < 0 or any(class_id < 0 for class_id in class_ids):
        raise ValueError("Run seeds and class IDs must be nonnegative.")
    memory = {"version": 2, "seed": int(seed), "class_ids": class_ids}
    memory.update({field: {} for field in _CLASS_FIELDS})
    dispersions = []
    for class_id in class_ids:
        features = X[y == class_id]
        _update_class(memory, class_id, features, clf, boundary_threshold)
        spread = np.linalg.norm(features - features.mean(axis=0), axis=1).mean()
        norm = np.linalg.norm(features, axis=1).mean()
        dispersions.append(float(spread / max(float(norm), 1e-6)))
    
    memory["kappa_ref"] = float(np.quantile(dispersions, 0.95))
    return memory

def update_memory(previous, newX, newY, replayX, replayY, plan, clf, boundary_threshold=0.25):
    if previous.get("version") != 2:
        raise ValueError("Legacy geometry memory must be rebuilt from the base task.")
    dimension = len(previous["proto_mean"][previous["class_ids"][0]])
    newX, newY = _feature_rows(newX, newY, dimension)
    replayX, replayY = _feature_rows(replayX, replayY, dimension)
    old_classes = set(previous["class_ids"])
    new_classes = set(int(value) for value in np.unique(newY))
    if new_classes & old_classes:
        raise ValueError("New-class memory inputs must exclude all old-class samples.")
    selected = set(plan["selected_classes"])
    if not selected <= old_classes or not set(replayY.tolist()) <= selected:
        raise ValueError("Replay memory inputs must belong to selected old classes.")
    for class_id in selected:
        count = int(np.count_nonzero(replayY == class_id))
        quota = int(plan["quotas"][class_id])
        if quota <= 0 or count > quota or count != int(plan["verified_counts"][class_id]):
            raise ValueError("Replay instance counts disagree with the verified replay plan.")
    memory = copy.deepcopy(previous)
    for class_id in sorted(new_classes):
        _update_class(memory, class_id, newX[newY == class_id], clf, boundary_threshold)
    for class_id in sorted(selected):
        if int(plan["verified_counts"][class_id]) == int(plan["quotas"][class_id]):
            _update_class(memory, class_id, replayX[replayY == class_id], clf, boundary_threshold)
    memory["class_ids"] = sorted(old_classes | new_classes)
    return memory
