from pydantic import BaseModel
from typing import List, Tuple, Optional
from fastapi import Query, Body

class ChatWithPdfParams(BaseModel):
    user_query: str
    chat_history: List[Tuple[str, str]] = None
