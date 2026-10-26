"""Directed follow graph used to assemble each agent's daily reading batch."""


class SocialNetwork:
    def __init__(self):
        self._graph: dict[str, set[str]] = {}

    def get_followees(self, agent_id: str) -> set[str]:
        return self._graph.get(agent_id, set())
