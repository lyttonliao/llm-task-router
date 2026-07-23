from typing import Protocol

from llm_task_router.schema import ProviderResult


class Provider(Protocol):
    def invoke(self, prompt: str, model: str) -> ProviderResult: ...
