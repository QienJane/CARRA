import math

import numpy as np

def new_sketch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "n": 0,
        "levels": [np.empty(0, dtype=np.float32)],
        "rng_state": rng.bit_generator.state,
    }

def update_sketch(sketch, values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Quantile streams require finite values.")
    if sketch["n"] + values.size >= 2**24:
        raise ValueError("The 128-slot KLL supports fewer than 2**24 observations.")
    if values.size == 0:
        return sketch
    rng = np.random.default_rng()
    rng.bit_generator.state = sketch["rng_state"]
    levels = sketch["levels"]
    for value in values:
        levels[0] = np.append(levels[0], value).astype(np.float32, copy=False)
        sketch["n"] += 1
        if sketch["n"] <= 128:
            continue
        while True:
            height = len(levels)
            capacities = [
                max(2, int(math.ceil(32 * (2 / 3) ** (height - level - 1))))
                for level in range(height)
            ]
            overflowing = next(
                (level for level in range(height) if levels[level].size > capacities[level]),
                None,
            )
            if overflowing is None:
                break
            ordered = np.sort(levels[overflowing])
            even_size = ordered.size - ordered.size % 2
            parity = int(rng.integers(0, 2))
            promoted = ordered[:even_size][parity::2]
            levels[overflowing] = ordered[even_size:].copy()
            if overflowing + 1 == height:
                levels.append(np.empty(0, dtype=np.float32))
            levels[overflowing + 1] = np.concatenate((levels[overflowing + 1], promoted))
    sketch["rng_state"] = rng.bit_generator.state
    return sketch

def quantile(sketch, probability):
    probability = float(probability)
    if not 0 <= probability <= 1 or sketch["n"] == 0:
        raise ValueError("A quantile needs a nonempty stream and probability in [0, 1].")
    values = np.concatenate(sketch["levels"])
    weights = np.concatenate([
        np.full(level_values.size, 2**level, dtype=np.int64)
        for level, level_values in enumerate(sketch["levels"])
    ])
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    position = probability * (sketch["n"] - 1)
    lower, upper = math.floor(position), math.ceil(position)
    lo = values[np.searchsorted(cumulative, lower, side="right")]
    hi = values[np.searchsorted(cumulative, upper, side="right")]
    return float(lo + (float(hi) - float(lo)) * (position - lower))
