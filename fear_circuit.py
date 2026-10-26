"""Aggregate appraised beliefs and update the agent's cumulative mood."""
import math
from typing import Optional

import numpy as np

from method_models import MethodFearResult


def update_mood(current_mood: float, fear: Optional[float], beta: float) -> float:
    if fear is None:
        return current_mood
    return float(np.clip(current_mood + (1.0 - beta) * (fear - current_mood), -1.0, 1.0))


def compute_belief_fear(direct, graph, active_depths, gamma, c=1.0):
    """Count each current belief once; current appraisals replace matched old ones."""
    totals = {'direct_threat': 0., 'direct_safety': 0., 'graph_threat': 0., 'graph_safety': 0.}
    contributions = []
    excluded = {nid for _, nid in direct if nid is not None}

    def add(source, item_id, item):
        amount = item.confidence * item.significance if item.signal in {'Threat', 'Safety'} else 0.
        if item.signal in {'Threat', 'Safety'}:
            totals[f'{source}_{item.signal.lower()}'] += amount
        contributions.append(dict(source=source, item_id=item_id, text=item.text,
                                  signal=item.signal, confidence=item.confidence,
                                  significance=item.significance, amount=amount))

    for index, (belief, _) in enumerate(direct):
        add('direct', index, belief)
    for nid in active_depths:
        if nid not in excluded:
            add('graph', nid, graph.nodes[nid])
    safety = c * (totals['direct_safety'] + totals['graph_safety'])
    fear = math.tanh(gamma * (totals['direct_threat'] + totals['graph_threat'] - safety))
    return MethodFearResult(**totals, fear=fear, contributions=contributions,
                            excluded_old_ids=sorted(excluded & active_depths.keys()),
                            safety_scale=c, scaled_safety=safety)
