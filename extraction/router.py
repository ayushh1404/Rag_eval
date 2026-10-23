"""
POST /extract route - a standalone, stateless field-extraction endpoint.
Wired into app.py via app.include_router(router). This file only handles
request validation and turning internal failures into clear HTTP error
responses - see extraction/service.py for the actual extraction logic.
"""

import json
import logging

from fastapi import APIRouter, HTTPException

from .models import (
    ExtractRequest,
    ExtractResponse,
    ExtractedFieldEntry,
    ChangedField,
    UsageInfo,
)
from .service import run_extraction, ExtractionError

log = logging.getLogger("app")

router = APIRouter()

# ANSI escape codes - print the raw /extract request body in blue in the
# console/journal (e.g. `journalctl -u <service> -f` or a raw terminal on the
# Lightsail instance render these directly; log aggregators that strip ANSI
# codes will just show plain text, which is harmless).
_ANSI_BLUE = "\033[94m"
_ANSI_RESET = "\033[0m"


def _validate_fields(extraction_fields: list) -> None:
    if not extraction_fields:
        raise HTTPException(400, "extraction_fields must not be empty")
    seen_ids = set()
    for f in extraction_fields:
        if not f.id or not f.id.strip():
            raise HTTPException(400, "Every extraction_fields entry must have a non-empty id")
        if f.id in seen_ids:
            raise HTTPException(400, f"Duplicate extraction_fields id: {f.id!r}")
        seen_ids.add(f.id)
        if f.type == "CHOICE" and not f.options:
            raise HTTPException(400, f"Field {f.id!r} is type CHOICE but has no options")
        if not f.instruction or not f.instruction.strip():
            raise HTTPException(400, f"Field {f.id!r} is missing an instruction")


@router.post("/extract", response_model=ExtractResponse, tags=["Extraction"])
async def extract(payload: ExtractRequest):
    """
    Stateless field extraction over a full conversation transcript.
    No server-side memory is read or written - previous state (known_data)
    must be passed in, and updated state must be persisted by the caller.
    Safe to call repeatedly with the same input (idempotent).
    """
    print(
        f"{_ANSI_BLUE}POST /extract - REQUEST BODY:\n"
        f"{json.dumps(payload.model_dump(), indent=2, default=str)}"
        f"{_ANSI_RESET}"
    )

    _validate_fields(payload.extraction_fields)

    if not payload.conversation_history:
        raise HTTPException(400, "conversation_history must not be empty")

    if not payload.conversation_id or not payload.conversation_id.strip():
        raise HTTPException(400, "conversation_id is required")

    extraction_fields = [f.model_dump() for f in payload.extraction_fields]
    known_data = {k: v.model_dump() for k, v in payload.known_data.items()}
    conversation_history = [m.model_dump() for m in payload.conversation_history]

    log.info(
        "POST /extract | conversation_id=%s | fields=%d | history_len=%d",
        payload.conversation_id, len(extraction_fields), len(conversation_history),
    )

    try:
        result = run_extraction(extraction_fields, known_data, conversation_history)
    except ExtractionError as e:
        log.error("POST /extract failed for conversation_id=%s: %s", payload.conversation_id, e)
        raise HTTPException(502, f"Extraction failed: {e}")
    except Exception as e:
        log.error(
            "POST /extract unexpected error for conversation_id=%s: %s",
            payload.conversation_id, e, exc_info=True,
        )
        raise HTTPException(500, "Internal error during extraction")

    return ExtractResponse(
        extracted_data={
            fid: ExtractedFieldEntry(**entry) for fid, entry in result["extracted_data"].items()
        },
        changed_fields=[ChangedField(**cf) for cf in result["changed_fields"]],
        usage=UsageInfo(**result["usage"]),
    )
