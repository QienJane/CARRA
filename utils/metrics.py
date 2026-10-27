import numpy as np

def classification_metrics(labels, predictions, new_classes, history):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError('Evaluation requires matching nonempty labels and predictions.')
    per_class = {int(c): float(100 * np.mean(predictions[labels == c] == c))
                 for c in np.unique(labels)}
    new_classes = set(int(c) for c in new_classes)
    new_mask = np.isin(labels, list(new_classes))
    global_accuracy = float(100 * np.mean(predictions == labels))
    local_accuracy = float(100 * np.mean(predictions[new_mask] == labels[new_mask])) if new_mask.any() else None
    old_classes = sorted(set(per_class) - new_classes)
    forgetting, backward = {}, {}
    for class_id in old_classes:
        previous = [task['per_class_accuracy'][class_id] for task in history
                    if class_id in task['per_class_accuracy']]
        if not previous:
            raise ValueError(f'Missing evaluation history for old class {class_id}.')
        forgetting[class_id] = max(previous) - per_class[class_id]
        backward[class_id] = per_class[class_id] - previous[0]
    ifm = None
    if local_accuracy is not None and local_accuracy + global_accuracy > 0:
        ifm = abs(local_accuracy - global_accuracy) / (local_accuracy + global_accuracy) * 100
    tail = sorted(forgetting, key=lambda c: (-forgetting[c], c))[:3]
    return {'G': global_accuracy, 'L': local_accuracy, 'IFM': ifm,
            'AF': float(np.mean(list(forgetting.values()))) if forgetting else None,
            'BWT': float(np.mean(list(backward.values()))) if backward else None,
            'F_top3': float(np.mean([forgetting[c] for c in tail])) if tail else None,
            'tail_classes': tail, 'old_class_count': len(old_classes),
            'per_class_accuracy': per_class, 'forgetting': forgetting,
            'new_classes': sorted(new_classes), 'sample_count': int(labels.size)}

def _mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None

def summarize_run(tasks, plans, budget):
    incremental = tasks[1:]
    count = len(incremental)
    if len(plans) != count:
        raise ValueError('Each incremental task must have one replay plan.')
    active = sum(bool(plan['gate']) for plan in plans)
    realized = sum(sum(plan['verified_counts'].values()) for plan in plans)
    old_count = sum(task['old_class_count'] for task in incremental)
    return {'Mean_G': _mean(task['G'] for task in incremental),
            'Mean_IFM': _mean(task['IFM'] for task in incremental),
            'AF': _mean(task['AF'] for task in incremental),
            'BWT': (sum(task['BWT'] * task['old_class_count'] for task in incremental
                        if task['BWT'] is not None) / old_count) if old_count else None,
            'F_top3': _mean(task['F_top3'] for task in incremental),
            'final_G': tasks[-1]['G'] if tasks else None,
            'AR': active / count if count else None,
            'BFR_act': realized / (budget * active) if active else None,
            'BFR_all': realized / (budget * count) if count else None,
            'verified_total': int(realized),
            'verification_shortfall': int(sum(sum(plan['shortfalls'].values()) for plan in plans)),
            'unallocated_active_budget': int(sum(plan['unallocated_budget'] for plan in plans if plan['gate'])),
            'inactive_unused_budget': int(budget * (count - active)),
            'incremental_tasks': count}

def summarize_runs(summaries):
    
    result = {}
    for key in ('Mean_G', 'Mean_IFM', 'AF', 'BWT', 'F_top3', 'final_G', 'AR',
                'BFR_act', 'BFR_all', 'verified_total', 'verification_shortfall'):
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        result[key] = {'mean': float(np.mean(values)) if values else None,
                       'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       'n': len(values)}
    return result
