"""Load supplied people and initial memories, carrying source metadata as-is."""
import json
from copy import deepcopy
from pathlib import Path
from method_models import EventMemory


def load_initialization(config):
    if config.get('initialization_package_path'):
        package = json.loads(Path(config['initialization_package_path']).read_text(encoding='utf-8'))
        agents = deepcopy(package['agents'])
        metadata = {k: v for k, v in package.items() if k != 'agents'}
    else:
        agents = deepcopy(config['agents'])
        metadata = deepcopy(config.get('initialization', {}))
    for spec in agents:
        memory = spec.get('event_memory', '')
        spec['event_memory'] = EventMemory(memory) if isinstance(memory, str) else EventMemory(**memory)
        spec.setdefault('profile', {})
        spec.setdefault('mood', 0.)
        spec.setdefault('language', 'zh')
        spec.setdefault('stratum_id', None)
    return agents, metadata
