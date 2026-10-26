"""Read the small cognition response; semantic judgments belong to the user role."""
import json
import math
from method_models import AppraisedBelief, Cognition


def parse_json(raw):
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(text)


def bounded_float(value):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('expected a finite number') from exc
    if not math.isfinite(number):
        raise ValueError('expected a finite number')
    return min(1., max(0., number))


def parse_belief(data):
    try:
        signal = data['signal']
        if signal not in ('Threat', 'Safety', 'Neutral'):
            return None
        confidence = bounded_float(data['confidence'])
        significance = bounded_float(data['significance'])
        return AppraisedBelief(data['text'], signal, confidence,
                               0. if signal == 'Neutral' else significance)
    except (ValueError, TypeError, KeyError):
        return None


def parse_cognition(data, max_cues):
    return Cognition(
        cues=(data.get('cues') or [])[:max_cues],
        beliefs=[belief for b in (data.get('beliefs') or [])
                 if (belief := parse_belief(b)) is not None],
        graph_update=data.get('graph_update') or {},
        memory=data.get('memory'),
        risk_relevant=data.get('risk_relevant'),
    )
