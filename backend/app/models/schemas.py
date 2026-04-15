
from pydantic import BaseModel
from typing import List, Optional, Any

class ChatRequest(BaseModel):
    message: str

class DataAskRequest(BaseModel):
    question: str
    model: Optional[str] = None
    kg_disabled: bool = False

class PervasiveStatus(BaseModel):
    metric: str
    status: str
    mean_value: float
    recommended_min: Optional[float] = None
    recommended_max: Optional[float] = None
    location: Optional[str] = None

class DataAskResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    note: str
    pervasive_status: Optional[List[PervasiveStatus]] = None
