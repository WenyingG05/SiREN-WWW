"""One batch cognition per agent per day, followed by numerical processing and action."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
from method_models import MethodStepLog
from belief_matching import TextMatcher
from belief_revision import apply_revision
from fear_circuit import compute_belief_fear, update_mood
from event_memory import update_event_memory, record_action
from method_parsing import bounded_float
from models import Post


def build_action_context(graph, direct, active_depths, operations, matcher):
    """Select surviving updated beliefs and carry temporary direct understanding."""
    selected = set(active_depths)
    merged = {}
    withdrawn = set()
    for operation in operations:
        kind = operation['op']
        if kind in ('add', 'update'):
            selected.add(operation['node']['node_id'])
        elif kind == 'merge':
            merged[operation['source_id']] = operation['target_id']
            selected.add(operation['target_id'])
            withdrawn.add(operation['source_text'])
        elif kind == 'remove':
            withdrawn.add(operation['text'])
        elif kind in ('add_edge', 'remove_edge'):
            selected.update((operation['source_id'], operation['target_id']))

    direct_ids = set()
    temporary = []
    for belief, old_id in direct:
        nid = old_id
        while nid in merged:
            nid = merged[nid]
        if nid is None:
            nid = matcher.find(belief.text, graph.nodes, belief.signal)
            if nid is None and belief.text not in withdrawn:
                temporary.append(asdict(belief))
        if nid in graph.nodes:
            direct_ids.add(nid)
    selected.update(direct_ids)
    local = graph.context(selected)
    local['beliefs'].extend(temporary)
    return dict(graph=local, current_beliefs=graph.context(direct_ids)['beliefs'] + temporary)


class PersonalAgent:
    def __init__(self, agent_id, state, cognitive, retriever, params, max_cues_per_batch,
                 activation_hops, seed, matcher=None):
        self.agent_id = agent_id
        self.state = deepcopy(state)
        self.state.graph.connect_components()
        self.cognitive = cognitive
        self.retriever = retriever
        self.matcher = matcher if matcher is not None else TextMatcher()
        self.params = dict(params)
        self.max_cues_per_batch = max_cues_per_batch
        self.activation_hops = activation_hops
        self.rng = np.random.default_rng(seed)
        self._post_seq = 0

    @property
    def mood(self):
        return self.state.mood

    async def step(self, timeline, step, reading=None):
        old = deepcopy(self.state)
        posts = list({p.post_id: p for p in timeline}.values())
        log = MethodStepLog(self.agent_id, step, 'no_input', [p.post_id for p in posts],
                            old.mood, old.mood, old.graph.version, old.graph.version,
                            memory_before=old.memory.text, memory_after=old.memory.text,
                            reading=deepcopy(reading or {}))
        io_start = len(getattr(self.cognitive, 'io_log', []))
        post = None
        phase = 'cognition'
        try:
            if not posts:
                return None, log
            cognition = await self.cognitive.cognize(posts, old, self.max_cues_per_batch)
            if cognition is None:
                log.status = 'cognition_failed'
                log.errors.append(dict(stage=phase, error='API or response parsing failed'))
                return None, log
            log.cognition = cognition
            if cognition.relevant:
                direct = self.matcher.prepare_direct(cognition.beliefs, old.graph)
                seeds, log.retrieval = self.retriever.retrieve(cognition.cues, old.graph)
                log.active_depths = old.graph.activate(seeds, self.activation_hops)
                log.fear_result = compute_belief_fear(direct, old.graph, log.active_depths,
                                                     self.params['gamma'], self.params.get('c', 1.))
                self.state.graph, log.graph_operations = apply_revision(old.graph, cognition, self.matcher, step)
                self.state.mood = update_mood(old.mood, log.fear_result.fear, self.params['beta'])
            self.state.memory = update_event_memory(old.memory, cognition.memory)
            log.status = 'updated' if cognition.relevant else 'irrelevant'
            if not cognition.relevant:
                return None, log
            payload = dict(profile=self.state.profile,
                           posts=[dict(author=p.author_id, content=p.content) for p in posts],
                           **build_action_context(self.state.graph, direct, log.active_depths,
                                                  log.graph_operations, self.matcher),
                           event_memory=self.state.memory.text,
                           fear=log.fear_result.fear, mood=self.mood)
            phase = 'action'
            log.action_state_version = self.state.graph.version
            probability = await self.cognitive.posting_probability(payload)
            if probability is None:
                log.status = 'action_failed'
                log.errors.append(dict(stage=phase, error='API or response parsing failed'))
                return None, log
            probability = bounded_float(probability)
            log.posting_probability = probability
            log.random_draw = float(self.rng.random())
            log.sampled_post = log.random_draw < probability
            log.status = 'sampled_silence'
            if log.sampled_post:
                phase = 'generation'
                content = await self.cognitive.generate({k: v for k, v in payload.items() if k != 'fear'})
                if not content:
                    log.status = 'generation_failed'
                    log.errors.append(dict(stage=phase, error='API or response parsing failed'))
                    return None, log
                self._post_seq += 1
                post = Post(f'{self.agent_id}_t{step}_{self._post_seq}', self.agent_id, content, step,
                            'personal', published_step=step, visible_step=step + 1)
                self.state.memory = record_action(self.state.memory, step, post)
                log.status = 'posted'
                log.posted = True
                log.new_post = post
            return post, log
        except Exception as exc:
            log.status = f'{phase}_failed'
            log.errors.append(dict(stage=phase, error=str(exc)))
            return None, log
        finally:
            log.mood_after = self.mood
            log.graph_version_after = self.state.graph.version
            log.graph_snapshot = self.state.graph.snapshot()
            log.memory_after = self.state.memory.text
            log.llm_calls = deepcopy(getattr(self.cognitive, 'io_log', [])[io_start:])
