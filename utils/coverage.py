import numpy as np

def generate_coverage(memory, plan, n_pseudo=3):
    if int(n_pseudo) != 3:
        raise ValueError("CARRA coverage uses the prototype and its two principal directions.")
    dimension = len(memory["proto_mean"][memory["class_ids"][0]])
    features, labels = [], []
    selected = set(plan["selected_classes"])
    for class_id in sorted(memory["class_ids"]):
        if class_id in selected and plan["verified_counts"][class_id] == plan["quotas"][class_id]:
            continue
        mean = np.asarray(memory["proto_mean"][class_id], dtype=np.float32)
        covariance = np.asarray(memory["proto_var"][class_id], dtype=np.float32)
        axes = np.asarray(memory["principal_axes_top2"][class_id], dtype=np.float32)
        if axes.shape != (dimension, 2):
            raise ValueError("Coverage requires two stored principal directions per class.")
        scale = 0.5 * np.sqrt(max(float(np.trace(covariance)) / dimension, 0.0))
        directions = axes / np.maximum(np.linalg.norm(axes, axis=0), 1e-12)
        features.extend((mean, mean + scale * directions[:, 0], mean + scale * directions[:, 1]))
        labels.extend([class_id] * 3)
    return (np.asarray(features, dtype=np.float32).reshape(-1, dimension),
            np.asarray(labels, dtype=np.int64))
