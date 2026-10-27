from heapq import nsmallest

import numpy as np

from .classifier import decision_scores

EPS = 1e-6

def allocate_quotas(ranked_classes, total_budget, minimum, maximum):
    ranked_classes = [int(class_id) for class_id in ranked_classes]
    if minimum <= 0 or maximum < minimum or total_budget < len(ranked_classes) * minimum:
        raise ValueError("Replay budget cannot satisfy the selected classes' quota bounds.")
    quotas = {class_id: int(minimum) for class_id in ranked_classes}
    remaining = int(total_budget) - len(ranked_classes) * int(minimum)
    for class_id in ranked_classes:
        increment = min(remaining, int(maximum) - quotas[class_id])
        quotas[class_id] += increment
        remaining -= increment
    return quotas

def build_replay_plan(memory, clf, newX, newY, cfg):
    newX = np.asarray(newX, dtype=np.float32)
    newY = np.asarray(newY, dtype=np.int64)
    if newX.ndim != 2 or newY.ndim != 1 or newX.shape[0] != newY.size or not newY.size:
        raise ValueError("Replay planning requires labeled current-task features.")
    if not np.isfinite(newX).all() or memory.get("version") != 2:
        raise ValueError("Planning requires finite features and version-2 geometry memory.")
    old_classes = sorted(int(class_id) for class_id in memory["class_ids"])
    if not old_classes or set(newY.tolist()) & set(old_classes):
        raise ValueError("Planning requires previous classes and disjoint new-class labels.")
    budget = int(cfg.get("total_budget", 48))
    topk = int(cfg.get("topk_old", 3))
    minimum = int(cfg.get("min_budget_per_class", 4))
    maximum = int(cfg.get("max_budget_per_class", 24))
    if topk <= 0 or minimum <= 0 or maximum < minimum or budget < topk * minimum:
        raise ValueError("Expected B >= K*b_min and 0 < b_min <= b_max.")
    prototype_classes = [
        class_id for class_id in old_classes
        if memory["proto_mean"].get(class_id) is not None
    ]
    if prototype_classes:
        means = np.stack([memory["proto_mean"][class_id] for class_id in prototype_classes]).astype(np.float64)
        features64 = newX.astype(np.float64)
        
        distances = features64 @ means.T
        distances *= -2
        distances += np.einsum("ij,ij->i", features64, features64)[:, None]
        distances += np.einsum("ij,ij->i", means, means)[None, :]
        np.maximum(distances, 0, out=distances)
        nearest_indices = np.argmin(distances, axis=1)
        
        nearest_distances = np.linalg.norm(features64 - means[nearest_indices], axis=1)
        radii = np.array([memory["radius_q95"][prototype_classes[index]] for index in nearest_indices])
        distance_ratios = nearest_distances / np.maximum(radii, EPS)
        del features64, means, distances
    else:
        distance_ratios = np.full(newY.size, np.inf)
    score_matrix = decision_scores(clf, newX)
    predicted = np.asarray(clf.classes_)[np.argmax(score_matrix, axis=1)]
    top_two = np.partition(score_matrix, -2, axis=1)[:, -2:]
    margins = top_two.max(axis=1) - top_two.min(axis=1)
    margin_refs = np.array([memory["margin_q05"][int(class_id)] for class_id in predicted])
    novelty = 0.5 * distance_ratios + 0.5 * margin_refs / np.maximum(margins, EPS)
    unknown_ratio = float(np.mean(novelty > float(cfg.get("sample_novelty_thr", 1.0))))
    new_mean = newX.mean(axis=0)
    spread = float(np.linalg.norm(newX - new_mean, axis=1).mean())
    compactness = spread / max(float(np.linalg.norm(newX, axis=1).mean()), EPS)
    relative_dispersion = compactness / max(float(memory["kappa_ref"]), EPS)
    gate = (unknown_ratio >= float(cfg.get("batch_unknown_ratio_thr", 0.60))
            and compactness <= float(cfg.get("batch_compactness_thr", 0.15)))
    plan = {
        "gate": bool(gate), "eligibility": {}, "selected_classes": [],
        "quotas": {}, "axes": {}, "verified_counts": {}, "shortfalls": {},
        "target_budget": budget, "unallocated_budget": budget,
        "scores": {}, "u": unknown_ratio, "kappa": compactness,
        "delta": relative_dispersion, "rho": spread,
    }
    if not gate:
        return plan
    for class_id in old_classes:
        has_geometry = all(memory.get(field, {}).get(class_id) is not None for field in (
            "proto_mean", "proto_var", "principal_axes_top2",
        ))
        consistent = has_geometry and int(clf.predict(memory["proto_mean"][class_id][None, :])[0]) == class_id
        plan["eligibility"][class_id] = "full" if consistent else "unusable"
    eligible = [class_id for class_id in old_classes if plan["eligibility"][class_id] == "full"]
    if not eligible:
        return plan
    cluster_aware = relative_dispersion > float(cfg.get("relative_dispersion_thr", 2.0))
    centers = (np.stack([newX[newY == label].mean(axis=0) for label in sorted(np.unique(newY))])
               if cluster_aware else new_mean[None, :])
    factors = np.zeros((len(eligible), 4), dtype=np.float64)
    axes = {}
    matched_centers = {}
    for index, class_id in enumerate(eligible):
        difference = centers - memory["proto_mean"][class_id]
        center_distances = np.linalg.norm(difference, axis=1)
        overlaps = 1 / (center_distances / max(memory["radius_q95"][class_id] + spread, EPS) + EPS)
        matched = int(np.argmax(overlaps))
        votes = margins[predicted == class_id]
        factors[index] = (
            overlaps[matched], float(np.mean(1 / np.maximum(votes, EPS))) if votes.size else 0.0,
            memory["boundary_density"][class_id], memory["old_confusion"][class_id],
        )
        matched_centers[class_id] = centers[matched].copy()
        if center_distances[matched] > EPS:
            axes[class_id] = (difference[matched] / center_distances[matched]).astype(np.float32)
    ranges = np.ptp(factors, axis=0)
    normalized = np.divide(factors - factors.min(axis=0), ranges,
                           out=np.zeros_like(factors), where=ranges > 0)
    weights = np.asarray(cfg.get("weights", [0.45, 0.30, 0.15, 0.10]), dtype=np.float64)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Planner weights must be four finite nonnegative values.")
    totals = normalized @ weights
    plan["scores"] = {class_id: float(totals[index]) for index, class_id in enumerate(eligible)}
    selected = nsmallest(topk, eligible, key=lambda class_id: (-plan["scores"][class_id], class_id))
    quotas = allocate_quotas(selected, budget, minimum, maximum)
    plan.update({
        "selected_classes": selected, "quotas": quotas,
        "axes": {class_id: axes[class_id] for class_id in selected if class_id in axes},
        "verified_counts": {class_id: 0 for class_id in selected}, "shortfalls": dict(quotas),
        "unallocated_budget": budget - sum(quotas.values()),
        "geometry_mode": "cluster_aware" if cluster_aware else "single_center",
        "matched_centers": {class_id: matched_centers[class_id] for class_id in selected},
    })
    return plan
