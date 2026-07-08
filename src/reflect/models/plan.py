from pydantic import BaseModel

from reflect.models.action_primitive import ActionPrimitive


class Plan(BaseModel):
    actions: list[ActionPrimitive]

    def to_legacy_strings(self) -> list[str]:
        return [action.to_legacy_string() for action in self.actions]


class PlanStepCandidate(BaseModel):
    action: str
    obj1: str
    obj2: str | None = None


class PlanStepCandidates(BaseModel):
    candidates: list[PlanStepCandidate]
