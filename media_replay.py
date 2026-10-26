"""Replay supplied media posts once, preserving their content."""
from copy import deepcopy
from models import Post


class MediaAgent:
    def __init__(self, agent_id, records):
        self.agent_id = agent_id
        self.records = deepcopy(records)

    async def step(self, step):
        return [Post(f"replay:{r['record_id']}", self.agent_id, r['content'], step, 'media',
                     origin='replay', original_record_id=r['record_id'],
                     original_author_id=r.get('original_author_id'),
                     published_step=step, visible_step=step + 1, real_time=r.get('real_time'))
                for r in self.records if r['published_step'] == step]
