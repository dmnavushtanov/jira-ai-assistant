from pydantic import BaseModel
from typing import List, Optional


class Comment(BaseModel):
    author: str
    body: str


class Issue(BaseModel):
    key: str
    summary: str
    description: Optional[str]
    comments: List[Comment] = []
