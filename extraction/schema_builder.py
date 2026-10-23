"""
Builds the strict JSON schema used for the /extract OpenAI structured-output
call. Deliberately separate from dummy_chat/dummy_kb_answer.py's
_build_json_schema(): that one also carries an "answer" field and per-field
_refused/_needs_clarification companions for the turn-by-turn chat-reply use
case. This one is data-only - no answer - and adds a per-field justification
+ correction-detected signal instead, since /extract reconciles an entire
transcript in one shot rather than tracking state turn by turn.
"""

DATE_FIELD_HINTS = ("date", "_when", "schedule")


def is_date_field(field_id: str) -> bool:
    return any(hint in field_id.lower() for hint in DATE_FIELD_HINTS)


# Same principle as dummy_chat/dummy_kb_answer.py's FIELD HANDLING MODE:
# a field whose OWN instruction text asks to classify, categorize, or
# summarize the conversation is fundamentally different from a field that
# extracts a discrete fact someone stated - it should be recomputed fresh
# from the whole transcript every run, not treated as a fact to lock in
# once and only change on an explicit correction. Detected from the
# instruction text, never from the field's id/name, since assistants define
# their own field ids arbitrarily.
DERIVE_FIELD_KEYWORDS = (
    "classify", "categorize", "categorise",
    "summary of", "write a summary", "summarize", "summarise",
)


def is_derive_field(field: dict) -> bool:
    text = (field.get("instruction") or "").lower()
    return any(kw in text for kw in DERIVE_FIELD_KEYWORDS)


def _value_schema(field: dict) -> dict:
    ftype = field["type"]
    if ftype == "OPEN":
        return {"type": ["string", "null"]}
    if ftype == "YES_NO":
        return {"type": ["boolean", "null"]}
    if ftype == "CHOICE":
        options = field.get("options") or []
        return {"type": ["string", "null"], "enum": options + [None]}
    raise ValueError(f"Unknown field type: {ftype}")


def build_extraction_schema(extraction_fields: list[dict]) -> dict:
    properties = {}

    for field in extraction_fields:
        fid = field["id"]
        properties[fid] = _value_schema(field)

        # Only meaningful when known_data already has a value for this
        # field - the service layer ignores this signal entirely for
        # fields with no prior value, but the model must still emit it
        # every run since the schema is strict (all properties required).
        properties[f"{fid}_correction_detected"] = {"type": ["boolean", "null"]}

        # Required whenever a value is newly set, or an existing value is
        # being corrected - lets every change be audited against the
        # transcript rather than trusted blindly.
        properties[f"{fid}_justification"] = {"type": ["string", "null"]}

        if is_date_field(fid):
            properties[f"{fid}_offset_type"] = {
                "type": ["string", "null"],
                "enum": ["today", "tomorrow", "day_after_tomorrow", "this",
                         "next", "next_week", "in_days", "in_weeks", "in_months", None],
            }
            properties[f"{fid}_weekday"] = {
                "type": ["string", "null"],
                "enum": ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday", None],
            }
            properties[f"{fid}_offset_n"] = {"type": ["integer", "null"]}

    return {
        "type": "object",
        "properties": {
            "extracted_fields": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
        },
        "required": ["extracted_fields"],
        "additionalProperties": False,
    }
