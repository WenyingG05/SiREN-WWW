"""Post data passed between the media replay and personal agents."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Post:
    post_id: str
    author_id: str
    content: str
    step: int
    source_type: str
    origin: str = "generated"
    original_record_id: Optional[str] = None
    original_author_id: Optional[str] = None
    published_step: Optional[int] = None
    visible_step: Optional[int] = None
    real_time: Optional[str] = None
