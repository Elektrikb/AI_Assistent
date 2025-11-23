# File: api/schemas.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class RecommendationResponse(BaseModel):
    answer: str
    recommended_article: Dict
    suggested_actions: List[str]
    session_id: str
    confidence: float

class SessionStatsResponse(BaseModel):
    user_id: str
    created_at: str
    interaction_count: int
    total_reward: float
    avg_reward: float

class TrainingResponse(BaseModel):
    status: str
    episodes_completed: int
    average_reward: float