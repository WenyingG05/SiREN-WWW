"""Daily batches, a replaceable contact network, and ordinary state snapshots."""
import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
import numpy as np
import networkx as nx
import yaml
from method_models import AgentState, EventMemory
from belief_graph import BeliefGraph
from belief_matching import TextMatcher
from method_agent import PersonalAgent
from method_llm import APITransport, MethodLLM
from embedding import EmbeddingModel
from retrieval import SeedRetriever
from media_replay import MediaAgent
from social_media import SocialNetwork
from initialization import load_initialization


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_network(ids, media_ids, config, seed):
    net = SocialNetwork()
    net._graph = {aid: set() for aid in ids + media_ids}
    if config.get('precomputed_path'):
        source = json.loads(Path(config['precomputed_path']).read_text(encoding='utf-8'))
        for u, v in source['peer_edges_undirected']:
            net._graph[u].add(v)
            net._graph[v].add(u)
        for mid, aid in source['media_to_personal_arcs']:
            if mid in media_ids:
                net._graph[aid].add(mid)
        return net, source['node_to_agent']
    rng = np.random.default_rng(seed)
    mapping = list(rng.permutation(ids))
    n = len(ids)
    if n <= 1:
        graph = nx.empty_graph(n)
    elif config['type'] == 'barabasi_albert':
        graph = nx.barabasi_albert_graph(n, config['m'], seed=seed)
    elif config['type'] == 'random':
        graph = nx.erdos_renyi_graph(n, config['edge_probability'], seed=seed)
    elif config['type'] == 'small_world':
        graph = nx.watts_strogatz_graph(n, config['neighbors'], config['rewiring_probability'], seed=seed)
    else:
        raise ValueError(f"Unknown network type: {config['type']}")
    for u, v in graph.edges:
        net._graph[mapping[u]].add(mapping[v])
        net._graph[mapping[v]].add(mapping[u])
    for aid in ids:
        for mid in media_ids:
            if rng.random() < config.get('media_follow_prob', 1.):
                net._graph[aid].add(mid)
    return net, {str(i): aid for i, aid in enumerate(mapping)}


def initialize(config, encoder=None, transport_factory=None, matcher=None):
    settings = config['cognition']
    llm_cfg = config['llm']
    seed = config['simulation']['random_seed']
    config.setdefault('reading', dict(max_posts=None, order='input'))
    initial, metadata = load_initialization(config)
    records = config.get('media', {}).get('records', [])
    media_ids = sorted({r['media_agent_id'] for r in records})
    media = [MediaAgent(mid, [r for r in records if r['media_agent_id'] == mid]) for mid in media_ids]
    network, mapping = build_network([s['agent_id'] for s in initial], media_ids, config['network'], seed)
    if encoder is None:
        encoder = EmbeddingModel(settings['embedding_model'])
    retriever = SeedRetriever(encoder, settings['seed_top_r'], settings['retrieval_threshold'],
                              settings.get('retrieval_log_top_n', 10))
    matcher = matcher if matcher is not None else TextMatcher(encoder, settings.get('belief_match_threshold', .95))
    shared_transport = None if transport_factory else APITransport(
        llm_cfg['model'], llm_cfg.get('temperature', .7), llm_cfg.get('max_tokens', 4096),
        llm_cfg.get('timeout_seconds', 60))
    agents = []
    for idx, spec in enumerate(initial):
        graph = BeliefGraph.from_dict(spec['belief_graph']) if 'belief_graph' in spec else BeliefGraph.with_center(spec['center_risk_object'])
        state = AgentState(graph, spec['event_memory'], spec['mood'], spec['profile'])
        transport = transport_factory(spec['agent_id']) if transport_factory else shared_transport
        cognitive = MethodLLM(transport, spec['language'], llm_cfg.get('max_attempts', 3),
                              llm_cfg.get('retry_delay_seconds', 1.), llm_cfg.get('prompt_dir'))
        agents.append(PersonalAgent(spec['agent_id'], state, cognitive, retriever, config['params'],
                                    settings['max_cues_per_batch'], settings['activation_hops'], seed + idx, matcher))
    run_dir = Path(config['output_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir.mkdir(parents=True)
    (run_dir / 'step_logs').mkdir()
    (run_dir / 'checkpoints').mkdir()
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    dump(run_dir / 'network.json', {k: sorted(v) for k, v in network._graph.items()})
    dump(run_dir / 'network_mapping.json', mapping)
    dump(run_dir / 'initial_states.json', snapshot_agents(agents))
    dump(run_dir / 'initialization_log.json', metadata)
    source_dir = Path(__file__).parent
    source_names = ('method_simulation.py', 'method_agent.py', 'method_llm.py',
                    'method_models.py', 'belief_graph.py', 'belief_matching.py',
                    'belief_revision.py', 'fear_circuit.py', 'method_parsing.py',
                    'event_memory.py', 'models.py', 'method_prompts.py',
                    'retrieval.py', 'initialization.py', 'media_replay.py',
                    'social_media.py', 'llm_transport.py', 'embedding.py')
    prompt_dir = Path(llm_cfg.get('prompt_dir') or source_dir / 'prompts')
    file_hashes = {name: file_sha256(source_dir / name) for name in source_names}
    for language in ('zh', 'en'):
        name = f'method_{language}.json'
        file_hashes[f'prompt/{name}'] = file_sha256(prompt_dir / name)
    init_path = config.get('initialization_package_path')
    net_path = config['network'].get('precomputed_path')
    media_hash = hashlib.sha256(json.dumps(records, ensure_ascii=False,
                                            sort_keys=True).encode('utf-8')).hexdigest()
    dump(run_dir / 'method_manifest.json', dict(method_version='risk-belief-graph-v1',
                                               params=config['params'], model=llm_cfg['model'],
                                               cognition=settings,
                                               initial_state_hash=file_sha256(init_path) if init_path else None,
                                               media_hash=media_hash,
                                               precomputed_network_hash=file_sha256(net_path) if net_path else None,
                                               file_hashes=file_hashes))
    return dict(config=config, personal_agents=agents, media_agents=media, network=network, run_dir=run_dir)


def read_batch(agent, pool, network, config):
    followees = network.get_followees(agent.agent_id)
    posts = list({p.post_id: p for p in pool if p.author_id in followees and p.author_id != agent.agent_id}.values())
    available = len(posts)
    if config.get('order', 'input') == 'random' and posts:
        posts = [posts[i] for i in agent.rng.permutation(len(posts))]
    budget = config.get('max_posts')
    selected = posts if budget is None else posts[:budget]
    return selected, dict(available_count=available, read_count=len(selected),
                          truncated_count=available-len(selected))


def snapshot_agents(agents):
    return {a.agent_id: dict(graph=a.state.graph.snapshot(), memory=asdict(a.state.memory),
                            mood=a.mood, profile=deepcopy(a.state.profile),
                            rng_state=a.rng.bit_generator.state, post_seq=a._post_seq) for a in agents}


async def run(components):
    config = components['config']
    run_dir = components['run_dir']
    agents = components['personal_agents']
    media = components['media_agents']
    pool, all_posts, summaries, start_step = [], [], [], 0
    if config.get('resume_checkpoint'):
        pool, all_posts, summaries, start_step = restore_checkpoint(components, Path(config['resume_checkpoint']))
    for step in range(start_step, config['simulation']['total_steps'] + 1):
        # Freeze the previous day's complete feed before this day's publication.
        batches = [read_batch(a, pool, components['network'], config['reading']) for a in agents] if step else []
        media_results = await asyncio.gather(*[m.step(step) for m in media])
        media_posts = [p for group in media_results for p in group]
        results = await asyncio.gather(*[a.step(batch, step, reading=reading)
                                         for a, (batch, reading) in zip(agents, batches)])
        logs = [log for _, log in results]
        generated = [post for post, _ in results if post is not None]
        pool = media_posts + generated
        all_posts.extend(pool)
        fears = [log.fear_result.fear for log in logs if log.fear_result is not None]
        summary = dict(step=step, media_posts=len(media_posts), personal_posts=len(generated),
                       avg_fear=float(np.mean(fears)) if fears else None,
                       avg_mood=float(np.mean([a.mood for a in agents])),
                       failures=sum(log.status.endswith('_failed') for log in logs))
        summaries.append(summary)
        dump(run_dir / 'step_logs' / f'step_{step:03d}.json',
             dict(summary=summary, media_posts=[asdict(p) for p in media_posts], agent_logs=[asdict(log) for log in logs]))
        checkpoint = dict(completed_step=step, status='completed', agents=snapshot_agents(agents),
                          pending_posts=[asdict(p) for p in pool], all_posts=[asdict(p) for p in all_posts], summaries=summaries)
        dump(run_dir / 'checkpoints' / f'step_{step:03d}.json', checkpoint)
        dump(run_dir / 'checkpoint.json', checkpoint)
    dump(run_dir / 'all_posts.json', [asdict(p) for p in all_posts])
    dump(run_dir / 'generated_personal_posts.json', [asdict(p) for p in all_posts if p.source_type == 'personal'])
    dump(run_dir / 'summary.json', summaries)
    return all_posts


def restore_checkpoint(components, path):
    from models import Post
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    for agent in components['personal_agents']:
        stored = checkpoint['agents'][agent.agent_id]
        agent.state = AgentState(BeliefGraph.from_dict(stored['graph']), EventMemory(**stored['memory']),
                                 stored['mood'], stored['profile'])
        agent.rng.bit_generator.state = stored['rng_state']
        agent._post_seq = stored['post_seq']
    return ([Post(**p) for p in checkpoint['pending_posts']], [Post(**p) for p in checkpoint['all_posts']],
            checkpoint['summaries'], checkpoint['completed_step'] + 1)
