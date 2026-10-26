"""Embedding relevance selects cue seeds; scores stay outside the Fear formula."""
import numpy as np


class SeedRetriever:
    def __init__(self, encoder, top_r, threshold, log_top_n=10):
        self.encoder = encoder
        self.top_r = top_r
        self.threshold = threshold
        self.log_top_n = log_top_n

    def retrieve(self, cues, graph):
        if not cues or not graph.nodes:
            return [], []
        ids = sorted(graph.nodes)
        matrix = self.encoder.encode_batch([graph.nodes[i].text for i in ids])
        queries = self.encoder.encode_batch(cues)
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        scores = queries @ matrix.T
        seeds = set()
        diagnostics = []
        for cue, row in zip(cues, scores):
            ranked = sorted(zip(ids, row), key=lambda pair: (-pair[1], pair[0]))
            selected = [nid for nid, score in ranked if score >= self.threshold][:self.top_r]
            seeds.update(selected)
            logged = set(selected) | {nid for nid, _ in ranked[:self.log_top_n]}
            diagnostics.append(dict(cue=cue, seeds=selected,
                                    candidates=[dict(node_id=nid, score=float(score))
                                                for nid, score in ranked if nid in logged]))
        return sorted(seeds), diagnostics
