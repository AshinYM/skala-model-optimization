import math
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def contact_mask(probabilities, fraction=0.1):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected a nonempty finite probability vector in [0, 1]')
    if not 0 < fraction <= 1:
        raise ValueError('Contact fraction must be in (0, 1]')
    # Floor respects a strict budget. Ties use incoming row order, reproducibly.
    count = math.floor(len(p) * fraction)
    mask = np.zeros(len(p), dtype=bool)
    mask[np.argsort(-p, kind='stable')[:count]] = True
    return mask


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-15)
    # In a tie choose the higher threshold; this rule is fixed before test access.
    return float(thresholds[np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]])


def evaluate(y, probabilities, threshold=0.5, fraction=0.1):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or not set(np.unique(y)) <= {0, 1}:
        raise ValueError('Labels must align with probabilities and be binary')
    selected = contact_mask(p, fraction)
    n, positives, contacted = len(y), int(y.sum()), int(selected.sum())
    hits = int(y[selected].sum())
    prevalence = positives / n
    precision = hits / contacted if contacted else None
    return {'rows': n, 'positives': positives, 'prevalence': prevalence,
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, p)) if positives else None,
            'threshold': float(threshold),
            'precision': float(precision_score(y, p >= threshold, zero_division=0)),
            'recall': float(recall_score(y, p >= threshold, zero_division=0)),
            'f2': float(fbeta_score(y, p >= threshold, beta=2, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, p >= threshold, labels=[0, 1]).tolist(),
            'contact_fraction': float(fraction), 'contacted': contacted, 'captured_churners': hits,
            'precision_at_budget': precision,
            'recall_at_budget': hits / positives if positives else None,
            'lift_at_budget': precision / prevalence if precision is not None and prevalence else None,
            'random_expected_captured': contacted * prevalence}


def bootstrap(y, probabilities, fraction, seed, repeats=300):
    """Uncertainty of fixed predictions, not model refitting or future performance."""
    y, p, rng = np.asarray(y), np.asarray(probabilities), np.random.default_rng(seed)
    samples = {k: [] for k in ['roc_auc', 'average_precision', 'lift_at_budget']}
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) != 2:
            continue
        result = evaluate(y[indices], p[indices], fraction=fraction)
        for name in samples:
            if result[name] is not None:
                samples[name].append(result[name])
    return {name: {'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975)),
                   'resamples': len(values)} for name, values in samples.items() if values}
