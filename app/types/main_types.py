from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T')

class ServiceResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    metadata: dict = {}
    error: Optional[str] = None

class KeyState(BaseModel):
    current_index: int
    keys: list[str]
