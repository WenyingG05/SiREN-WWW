"""Replaceable text matching for belief identity, separate from cue relevance."""
import numpy as np


class TextMatcher:
    def __init__(self, encoder=None, threshold=.95):
        self.encoder = encoder
        self.threshold = threshold

    def find(self, text, items, signal=None):
        for key, item in items.items():
            if item.text.strip() == text.strip():
                return key
        # For approximate identity, opposite risk directions stay distinct.
        if signal is not None:
            items = {key: item for key, item in items.items() if item.signal == signal}
        if self.encoder is None or not items:
            return None
        keys = list(items)
        vectors = self.encoder.encode_batch([text] + [items[k].text for k in keys])
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        scores = vectors[1:] @ vectors[0]
        best = int(np.argmax(scores))
        return keys[best] if scores[best] >= self.threshold else None

    def prepare_direct(self, beliefs, graph):
        """Count each current recognition once and map it to the unchanged old graph."""
        unique = {}
        for belief in beliefs:
            if self.find(belief.text, unique, belief.signal) is None:
                unique[len(unique)] = belief
        direct = []
        matched = set()
        for belief in unique.values():
            nid = self.find(belief.text, graph.nodes, belief.signal)
            if nid is None or nid not in matched:
                direct.append((belief, nid))
                matched.add(nid)
        return direct
