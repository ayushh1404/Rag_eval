"""
Pydantic request/response models for the standalone POST /extract endpoint.

This endpoint is fully stateless - unlike /chat, it keeps no server-side
memory of its own. Every prior fact it needs (known_data) is passed in on
the request, and every updated fact is handed back in the response for the
caller to persist. See extraction/service.py for the extraction logic and
extraction/router.py for the route + validation.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ExtractionFieldDef(BaseModel):
    id: str
    name: str | None = None
    type: Literal["OPEN", "YES_NO", "CHOICE"]
    instruction: str
    mandatory: bool = False
    options: list[str] | None = None


class KnownDataEntry(BaseModel):
    value: Any = None
    updated_at: str | None = None


class ConversationMessage(BaseModel):
    role: Literal["user", "ai"]
    content: str
    created_at: str | None = None


class ExtractRequest(BaseModel):
    conversation_id: str
    extraction_fields: list[ExtractionFieldDef]
    known_data: dict[str, KnownDataEntry] = Field(default_factory=dict)
    conversation_history: list[ConversationMessage]


class ExtractedFieldEntry(BaseModel):
    value: Any = None
    updated_at: str | None = None


class ChangedField(BaseModel):
    field_id: str
    old_value: Any = None
    new_value: Any = None
    justification: str


class UsageInfo(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None
    model: str | None = None


class ExtractResponse(BaseModel):
    extracted_data: dict[str, ExtractedFieldEntry]
    changed_fields: list[ChangedField]
    usage: UsageInfo
