"""Assemble the three runtime system prompts from one language's final prompt file."""
import json
from pathlib import Path

STAGES = ('cognition', 'action', 'generation')


def load_prompts(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return {stage: '\n\n'.join([data['context']] +
                              [data['definitions'][key] for key in data['stages'][stage]['definitions']] +
                              [data['stages'][stage]['task']])
            for stage in STAGES}
