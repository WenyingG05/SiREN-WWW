"""Plain data shared by the user simulation modules."""
from dataclasses import dataclass, field


@dataclass
class BeliefNode:
    node_id: str
    text: str
    signal: str
    confidence: float
    significance: float
    # Optional metadata keeps existing initialization files readable.
    semantic_type: str = ''
    evidence_refs: list[str] = field(default_factory=list)
    updated_step: int = 0


@dataclass
class BeliefEdge:
    source_id: str
    target_id: str
    origin: str = 'text'
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class AppraisedBelief:
    text: str
    signal: str
    confidence: float
    significance: float


@dataclass
class Cognition:
    cues: list[str] = field(default_factory=list)
    beliefs: list[AppraisedBelief] = field(default_factory=list)
    graph_update: dict = field(default_factory=dict)
    memory: str | None = None
    risk_relevant: bool | None = None

    @property
    def relevant(self):
        return self.risk_relevant if self.risk_relevant is not None else bool(self.cues or self.beliefs)


@dataclass
class EventMemory:
    text: str = ''
    version: int = 0
    evidence_refs: list[str] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)


@dataclass
class AgentState:
    graph: object
    memory: EventMemory
    mood: float
    profile: dict


@dataclass
class MethodFearResult:
    direct_threat: float
    direct_safety: float
    graph_threat: float
    graph_safety: float
    fear: float
    contributions: list[dict]
    excluded_old_ids: list[str]
    safety_scale: float = 1.0
    scaled_safety: float = 0.0

    @property
    def neg_act(self):
        return self.direct_threat + self.graph_threat

    @property
    def pos_act(self):
        return self.direct_safety + self.graph_safety


@dataclass
class MethodStepLog:
    agent_id: str
    step: int
    status: str
    input_post_ids: list[str]
    mood_before: float
    mood_after: float
    graph_version_before: int
    graph_version_after: int
    cognition: Cognition | None = None
    memory_before: str = ''
    memory_after: str = ''
    retrieval: list[dict] = field(default_factory=list)
    active_depths: dict[str, int] = field(default_factory=dict)
    fear_result: MethodFearResult | None = None
    graph_operations: list[dict] = field(default_factory=list)
    graph_snapshot: dict = field(default_factory=dict)
    reading: dict = field(default_factory=dict)
    action_state_version: int | None = None
    posting_probability: float | None = None
    random_draw: float | None = None
    sampled_post: bool = False
    posted: bool = False
    new_post: object | None = None
    errors: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
