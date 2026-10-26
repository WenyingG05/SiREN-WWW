"""Apply text-based changes supplied by the simulated user."""
from copy import deepcopy
from dataclasses import asdict
from method_models import BeliefNode, BeliefEdge
from method_parsing import parse_belief


def apply_revision(graph, cognition, matcher, step):
    result = deepcopy(graph)
    changes = cognition.graph_update or {}
    operations = []
    names = {n.text: nid for nid, n in result.nodes.items()}
    original_names = dict(names)
    appraised = {i: b for i, b in enumerate(cognition.beliefs)}
    merged = {}

    def text_ref(value):
        if isinstance(value, dict):
            value = value.get('old_text') or value.get('text')
        return value if isinstance(value, str) and value.strip() else None

    def locate(text, signal=None):
        text = text_ref(text)
        if text is None:
            return None
        if text in names:
            nid = names[text]
            return nid if nid in result.nodes else None
        return matcher.find(text, result.nodes, signal)

    for patch in (changes.get('nodes') or []):
        if isinstance(patch, str):
            patch = {'text': patch}
        if not isinstance(patch, dict):
            continue
        if 'text' in patch and not isinstance(patch['text'], str):
            continue
        target_text = text_ref(patch.get('old_text')) or text_ref(patch.get('text'))
        if target_text is None:
            continue
        candidate = matcher.find(patch['text'], appraised, patch.get('signal')) if 'text' in patch else None
        signal = patch.get('signal', appraised[candidate].signal if candidate is not None else None)
        nid = locate(target_text, None if patch.get('old_text') else signal)
        if patch.get('remove'):
            if nid in result.nodes and nid != result.center_id:
                del result.nodes[nid]
                operations.append(dict(op='remove', node_id=nid, text=target_text))
            continue
        old = result.nodes.get(nid)
        base = asdict(old) if old else {}
        if candidate is not None:
            base.update(asdict(appraised[candidate]))
        base.update({key: patch[key] for key in ('text', 'signal', 'confidence', 'significance') if key in patch})
        belief = parse_belief(base)
        if belief is None:
            continue
        base.update(asdict(belief))
        nid = nid or result.allocate_id()
        result.nodes[nid] = BeliefNode(
            nid, base['text'], base['signal'], float(base['confidence']), float(base['significance']),
            base.get('semantic_type', ''), base.get('evidence_refs', []), step)
        names[target_text] = names[base['text']] = nid
        if isinstance(patch.get('id'), str) and patch['id']:
            names[patch['id']] = nid
        operations.append(dict(op='update' if old else 'add', node=asdict(result.nodes[nid])))
        for source_ref in (patch.get('merge_from') or []):
            source_text = text_ref(source_ref)
            if source_text is None:
                continue
            source = locate(source_text)
            if source is not None and source not in {nid, result.center_id}:
                merged[source] = nid
                del result.nodes[source]
                for text, mapped in list(names.items()):
                    if mapped == source:
                        names[text] = nid
                operations.append(dict(op='merge', source_id=source, source_text=source_text, target_id=nid))

    for raw_text in (changes.get('remove_nodes') or []):
        text = text_ref(raw_text)
        if text is None:
            continue
        nid = original_names.get(text) if text in original_names else locate(text)
        if nid in result.nodes and nid != result.center_id:
            del result.nodes[nid]
            operations.append(dict(op='remove', node_id=nid, text=text))

    def final_id(nid):
        while nid in merged:
            nid = merged[nid]
        return nid

    def pair(edge):
        if isinstance(edge, dict):
            source, target = edge.get('source'), edge.get('target')
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            source, target = edge
        else:
            return None, None
        return locate(source), locate(target)

    removed = {pair(edge) for edge in (changes.get('remove_edges') or [])}
    edges = {}
    for edge in result.edges:
        source, target = final_id(edge.source_id), final_id(edge.target_id)
        if source in result.nodes and target in result.nodes and source != target and (source, target) not in removed:
            edges[source, target] = BeliefEdge(source, target, edge.origin, edge.evidence_refs)
    for relation in (changes.get('edges') or []):
        source, target = pair(relation)
        if source in result.nodes and target in result.nodes and source != target:
            edges[source, target] = BeliefEdge(source, target)
            operations.append(dict(op='add_edge', source_id=source, target_id=target))
    for source, target in removed:
        if source is not None and target is not None:
            operations.append(dict(op='remove_edge', source_id=source, target_id=target))
    result.edges = list(edges.values())
    operations.extend(dict(op='connect_component', **edge) for edge in result.connect_components())
    if operations:
        result.version += 1
    return result, operations
