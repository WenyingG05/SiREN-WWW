"""Belief storage, association traversal and minimal center connections."""
from copy import deepcopy
from dataclasses import asdict
from collections import deque
from method_models import BeliefNode, BeliefEdge


class BeliefGraph:
    def __init__(self, nodes, edges, center_id, version=0, next_id=0):
        self.nodes = {n.node_id: deepcopy(n) for n in nodes}
        self.edges = deepcopy(edges)
        self.center_id = center_id
        self.version = version
        self.next_id = next_id

    @classmethod
    def with_center(cls, text):
        return cls([BeliefNode('center', text, 'Neutral', 1., 0., 'RiskObject')], [], 'center')

    @classmethod
    def from_dict(cls, data):
        return cls([BeliefNode(**n) for n in data['nodes']],
                   [BeliefEdge(**e) for e in data.get('edges', [])], data['center_id'],
                   data.get('version', 0), data.get('next_id', 0))

    def snapshot(self):
        return dict(nodes=[asdict(n) for n in self.nodes.values()],
                    edges=[asdict(e) for e in self.edges], center_id=self.center_id,
                    version=self.version, next_id=self.next_id)

    def context(self, node_ids=None):
        """Give the role texts and scores; node IDs stay in the program."""
        selected = self.nodes.keys() if node_ids is None else set(node_ids) & self.nodes.keys()
        return dict(beliefs=[dict(text=n.text, signal=n.signal,
                                 confidence=n.confidence, significance=n.significance)
                            for nid, n in self.nodes.items() if nid in selected],
                    relations=[dict(source=self.nodes[e.source_id].text,
                                    target=self.nodes[e.target_id].text)
                               for e in self.edges if e.origin != 'structural'
                               and e.source_id in selected and e.target_id in selected])

    def adjacency(self):
        neighbors = {nid: set() for nid in self.nodes}
        for edge in self.edges:
            neighbors[edge.source_id].add(edge.target_id)
            neighbors[edge.target_id].add(edge.source_id)
        return neighbors

    def activate(self, seeds, hops):
        neighbors = self.adjacency()
        depths = {s: 0 for s in seeds if s in neighbors}
        queue = deque(depths)
        while queue:
            current = queue.popleft()
            if hops != 'full_reachable' and depths[current] >= hops:
                continue
            for other in sorted(neighbors[current]):
                if other not in depths:
                    depths[other] = depths[current] + 1
                    queue.append(other)
        return depths

    def connect_components(self):
        connected = set(self.activate([self.center_id], 'full_reachable'))
        added = []
        for nid in sorted(self.nodes):
            if nid not in connected:
                component = set(self.activate([nid], 'full_reachable'))
                edge = BeliefEdge(self.center_id, min(component), 'structural')
                self.edges.append(edge)
                added.append(asdict(edge))
                connected.update(component)
        return added

    def allocate_id(self):
        while True:
            nid = f'b{self.next_id:06d}'
            self.next_id += 1
            if nid not in self.nodes:
                return nid
