import time

import numpy as np
import torch

def invert_feature(model, target, initial, cfg, tolerance):
    
    target = torch.as_tensor(target, dtype=initial.dtype, device=initial.device).reshape(1, -1)
    x = initial.detach().clone().requires_grad_(True)
    velocity = torch.zeros_like(x)
    denominator = target.norm().clamp_min(1e-8)
    steps = 0
    for step in range(int(cfg.get('max_steps', 2000))):
        features = model.forward_feature(x)
        error = (features - target).norm() / denominator
        error_value = float(error.item())
        if not np.isfinite(error_value):
            raise FloatingPointError('Non-finite feature error during inversion.')
        if error_value <= tolerance:
            return x.detach(), features.detach().cpu().numpy()[0], error_value, steps
        gradient, = torch.autograd.grad(error, x)
        with torch.no_grad():
            velocity.mul_(float(cfg.get('momentum', 0.9))).add_(gradient)
            x.add_(velocity, alpha=-float(cfg.get('lr', 1.0)))
        x = x.detach().requires_grad_(True)
        steps = step + 1
    with torch.no_grad():
        features = model.forward_feature(x)
        error = float(((features - target).norm() / denominator).item())
    if not np.isfinite(error):
        raise FloatingPointError('Non-finite final inversion feature.')
    return x.detach(), features.detach().cpu().numpy()[0], error, steps

def boundary_target(classifier, mean, direction, class_id, cfg):
    
    step = float(cfg.get('boundary_step', 0.05))
    distances = step * np.arange(1, int(cfg.get('boundary_steps', 100)) + 1, dtype=np.float32)
    candidates = mean[None, :] + distances[:, None] * direction[None, :]
    outside = np.flatnonzero(classifier.predict(candidates) != class_id)
    if not len(outside):
        return candidates[-1]
    first = int(outside[0])
    target = mean + 0.8 * distances[first] * direction
    if int(classifier.predict(target[None, :])[0]) == class_id:
        return target
    return candidates[first - 1] if first else mean.copy()

def candidate_directions(principal_axes, conflict_axis, rng, count):
    
    basis = []
    if conflict_axis is not None and np.linalg.norm(conflict_axis) > 1e-8:
        basis.append(np.asarray(conflict_axis, dtype=np.float32))
    axes = np.asarray(principal_axes, dtype=np.float32)
    basis.extend(axes[:, index] for index in range(axes.shape[1]))
    basis = np.stack(basis)
    for index in range(count):
        if index < 2 * len(basis):
            direction = basis[index // 2] * (1.0 if index % 2 == 0 else -1.0)
        else:
            direction = rng.normal(size=len(basis)) @ basis
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = basis[0]
            norm = np.linalg.norm(direction)
        yield (direction / max(norm, 1e-8)).astype(np.float32)

def generate_replay(model, classifier, memory, plan, cfg, seed=0):
    
    started = time.perf_counter()
    model.eval()
    parameters = list(model.parameters())
    requires_grad = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    device, dtype = parameters[0].device, parameters[0].dtype
    rng = np.random.default_rng(seed)
    records = []
    plan['attempted_counts'] = {}
    plan['rejected_counts'] = {}
    plan['anchor_steps'] = {}
    plan['optimization_steps'] = {}
    try:
        for class_id in plan['selected_classes']:
            quota = int(plan['quotas'][class_id])
            mean = np.asarray(memory['proto_mean'][class_id], dtype=np.float32)
            initial = torch.zeros((1, *model.get_input_shape()), device=device, dtype=dtype)
            anchor, _, _, anchor_steps = invert_feature(
                model, mean, initial, cfg, float(cfg.get('anchor_tolerance', 1e-3)))
            plan['anchor_steps'][class_id] = anchor_steps
            accepted = attempted = steps = 0
            directions = candidate_directions(
                memory['principal_axes_top2'][class_id], plan['axes'].get(class_id),
                rng, int(cfg.get('max_attempts_per_class', 240)))
            for direction in directions:
                if accepted >= quota:
                    break
                target = boundary_target(classifier, mean, direction, class_id, cfg)
                x, feature, _, iterations = invert_feature(
                    model, target, anchor, cfg, float(cfg.get('tolerance', 1e-2)))
                attempted += 1
                steps += iterations
                if int(classifier.predict(feature[None, :])[0]) != class_id:
                    continue
                records.append({'x': x[0].cpu().numpy().astype(np.float32),
                                'label': int(class_id), 'replay': True})
                accepted += 1
            plan['verified_counts'][class_id] = accepted
            plan['shortfalls'][class_id] = quota - accepted
            plan['attempted_counts'][class_id] = attempted
            plan['rejected_counts'][class_id] = attempted - accepted
            plan['optimization_steps'][class_id] = steps
            print(f'  replay class={class_id}: verified={accepted}/{quota}, attempts={attempted}', flush=True)
    finally:
        for parameter, value in zip(parameters, requires_grad):
            parameter.requires_grad_(value)
    plan['verified_total'] = len(records)
    plan['verification_shortfall'] = int(sum(plan['shortfalls'].values()))
    plan['inversion_seconds'] = time.perf_counter() - started
    return records
