from typing import Optional
from pydantic import BaseModel


class ActionPrimitive(BaseModel):
    action: str
    obj1: str
    obj2: Optional[str] = None

    @property
    def name(self) -> str:
        return f"({self.action}, {self.obj1}" + (f", {self.obj2}" if self.obj2 else "") + ")"

    def to_legacy_string(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"Action Primitive: {self.name}"
