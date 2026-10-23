"""
Core logic for POST /extract - a stateless, whole-transcript field
reconciliation. Given known_data (everything already known, from the
caller) and the full conversation_history, calls the model once, applies a
code-level safety net so a field can never change without an explicit
correction signal, resolves any date-type fields deterministically, and
returns the updated picture plus an auditable list of what changed.

Deliberately does not import or depend on dummy_chat/dummy_kb_answer.py's
answer_from_kb() - that function is shaped around the turn-by-turn
"answer + extract" chat use case (it always also writes a conversational
reply), which isn't what this endpoint needs. It does reuse the genuinely
shared, non-chat-specific pieces: the OpenAI client, cost calculation, and
the deterministic date resolver.
"""

import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from dummy_chat.dummy_kb_answer import get_openai_client, calculate_token_cost
from dummy_chat.date_parser import resolve_structured_date

from .schema_builder import build_extraction_schema, is_date_field, is_derive_field
from .prompt_builder import build_extraction_prompt

log = logging.getLogger("app")

MODEL = "gpt-4o-mini"
MAX_COMPLETION_TOKENS = 3000

# ─── Cross-field value-shape guardrail ──────────────────────────────────────
# Code-level backstop for a failure mode the model itself can slip into under
# a strict-JSON single-shot call: writing a value that is shaped like another
# field's data (e.g. an email address) into a field that isn't asking for
# that shape at all (e.g. "Role"). The prompt already forbids this, but
# relying on prompt discipline alone let exactly this happen in production
# (an email address landing in a "Role" field) - and because a field is
# locked in once known_data is non-empty, an unnoticed bad value would
# otherwise persist forever. This check is deliberately conservative: it
# only fires when a value's shape clearly belongs to a different, well-known
# field type (email/phone) than what this field's own id/name asks for.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PHONE_RE = re.compile(r"^\+?[\d][\d\s\-()]{6,}\d$")


def _field_expects(field: dict, keyword: str) -> bool:
    text = f"{field.get('id', '')} {field.get('name', '')}".lower()
    return keyword in text


def _value_shape_mismatch(field: dict, value) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    v = value.strip()
    if _EMAIL_RE.match(v) and not _field_expects(field, "email"):
        return True
    if _PHONE_RE.match(v) and not _field_expects(field, "phone"):
        return True
    return False


class ExtractionError(Exception):
    """Raised on malformed/unrecoverable model output - router.py turns this into a clean HTTP error."""


def _looks_structurally_valid(candidate, expected_field_ids: set) -> bool:
    if not isinstance(candidate, dict):
        return False
    fields = candidate.get("extracted_fields")
    if not isinstance(fields, dict):
        return False
    return expected_field_ids.issubset(fields.keys())


def run_extraction(
    extraction_fields: list[dict],
    known_data: dict,
    conversation_history: list[dict],
) -> dict:
    """
    Returns {"extracted_data": {...}, "changed_fields": [...], "usage": {...}}
    matching the /extract response shape. Raises ExtractionError on
    unrecoverable model-output failures. Pure function of its arguments -
    no reads from or writes to any server-side store.
    """
    openai = get_openai_client()
    schema = build_extraction_schema(extraction_fields)
    system_prompt, user_message = build_extraction_prompt(
        extraction_fields, known_data, conversation_history
    )

    log.info("=" * 80)
    log.info(
        "🔎 EXTRACTION RUN — reconciling %d field(s) against %d message(s) of transcript",
        len(extraction_fields), len(conversation_history),
    )
    log.info("=" * 80)

    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_reconciliation",
                "schema": schema,
                "strict": True,
            },
        },
    }

    resp = openai.chat.completions.create(**kwargs)
    raw_content = resp.choices[0].message.content
    finish_reason = resp.choices[0].finish_reason

    expected_ids = {f["id"] for f in extraction_fields}
    parsed = None
    needs_retry = finish_reason == "length"
    if not needs_retry:
        try:
            parsed = json.loads(raw_content)
            if not _looks_structurally_valid(parsed, expected_ids):
                needs_retry = True
        except json.JSONDecodeError:
            needs_retry = True

    if needs_retry:
        log.warning("run_extraction: retrying with a higher token ceiling (finish_reason=%s)", finish_reason)
        retry_kwargs = dict(kwargs)
        retry_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS * 3
        resp = openai.chat.completions.create(**retry_kwargs)
        raw_content = resp.choices[0].message.content
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"model returned invalid JSON even after retry: {e}") from e

    if not _looks_structurally_valid(parsed, expected_ids):
        raise ExtractionError("model output missing required fields even after retry")

    model_fields = parsed["extracted_fields"]

    # ─── Date resolution (deterministic, same principle as dummy_kb_answer.py:
    # the model only classifies intent, real code does the calendar math) ───
    today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    for field in extraction_fields:
        fid = field["id"]
        if not is_date_field(fid):
            continue
        offset_type = model_fields.get(f"{fid}_offset_type")
        if not offset_type:
            continue
        resolved = resolve_structured_date(
            offset_type=offset_type,
            weekday=model_fields.get(f"{fid}_weekday"),
            offset_n=model_fields.get(f"{fid}_offset_n") or 0,
            today=today_ist,
        )
        if resolved:
            model_fields[fid] = resolved["date"]

    # ─── Code-level safety net: a field NEVER changes unless known_data was
    # empty, or the model explicitly flagged a correction for it. This is
    # enforced here, not just requested in the prompt - the same defense-
    # in-depth principle already used elsewhere in this codebase (carry-
    # forward, date resolution) rather than trusting model discipline alone. ───
    now_iso = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    extracted_data = {}
    changed_fields = []

    for field in extraction_fields:
        fid = field["id"]
        known_entry = known_data.get(fid) or {}
        known_value = known_entry.get("value") if isinstance(known_entry, dict) else known_entry
        known_updated_at = known_entry.get("updated_at") if isinstance(known_entry, dict) else None

        if _value_shape_mismatch(field, known_value):
            # Self-heal: a bad value (e.g. an email address) already stored as
            # known_data for this field from a past run - most likely from
            # before this guardrail existed. Don't keep re-serving it forever;
            # drop it so this field gets a genuine chance to be extracted
            # correctly (or left null) from the transcript again this run.
            log.warning(
                "run_extraction: dropping stale known_data value %r for field %r - "
                "looks like a different field's data type leaked in on a past run",
                known_value, fid,
            )
            known_value = None
            known_updated_at = None

        model_value = model_fields.get(fid)
        correction_detected = bool(model_fields.get(f"{fid}_correction_detected"))
        justification = model_fields.get(f"{fid}_justification")

        if is_derive_field(field):
            # Derive/classify fields (e.g. a running summary, or a category
            # classification) are fundamentally different from fact fields:
            # they should be recomputed fresh from the whole transcript every
            # run, not locked in after first success and only changed on an
            # explicit correction. The prompt tells the model not to bother
            # with "_correction_detected" for these fields, so it's ignored
            # here entirely - only the model's freshly regenerated value
            # matters. Never regress an already-generated value back to null
            # just because this run's regeneration came back empty (e.g. a
            # transient miss) - keep the last good version in that case.
            has_model_value_derive = model_value is not None and model_value != ""
            known_value_str = str(known_value).strip() if known_value not in (None, "") else ""
            model_value_str = str(model_value).strip() if has_model_value_derive else ""

            if has_model_value_derive and model_value_str != known_value_str:
                extracted_data[fid] = {"value": model_value, "updated_at": now_iso}
                changed_fields.append({
                    "field_id": fid,
                    "old_value": known_value,
                    "new_value": model_value,
                    "justification": justification or "(recomputed from full transcript)",
                })
            else:
                extracted_data[fid] = {"value": known_value, "updated_at": known_updated_at}
            continue

        if _value_shape_mismatch(field, model_value):
            log.warning(
                "run_extraction: discarding field %r value %r - looks like a different "
                "field's data type (email/phone) leaking into this field, not real evidence for it",
                fid, model_value,
            )
            model_value = None
            correction_detected = False

        has_known_value = known_value is not None and known_value != ""
        has_model_value = model_value is not None and model_value != ""

        # Don't rely solely on the model remembering to flag "_correction_detected"
        # (that only fires on an EXPLICIT contradiction per the prompt). A user
        # naturally answering a follow-up question with a genuinely different,
        # justified value (e.g. confirming their real email after the assistant
        # asked "please confirm your email") is just as real an update, even
        # though nothing in the transcript reads as an argumentative correction.
        # Without this, a bad value (fabricated or otherwise) that got locked in
        # once can never be overwritten by a later, correct, well-evidenced answer.
        if (
            has_known_value and not correction_detected
            and has_model_value and justification
            and str(model_value).strip().lower() != str(known_value).strip().lower()
        ):
            correction_detected = True

        if has_known_value and not correction_detected:
            # Untouched, regardless of whatever the model output for "value"
            # this run - passive re-mentions must never overwrite this.
            extracted_data[fid] = {"value": known_value, "updated_at": known_updated_at}
            continue

        if has_known_value and correction_detected and has_model_value:
            if str(model_value).strip().lower() == str(known_value).strip().lower():
                # The model flagged this field as corrected (and may even have
                # quoted a real justification), but its own "value" output is
                # identical to what was already known - that's contradictory,
                # since a genuine correction must actually produce a different
                # value. Reporting this as a "changed" field would be
                # misleading (nothing really changed) and, worse, would mask
                # a real correction the model recognized but failed to apply.
                # Keep the existing value untouched and surface this loudly
                # instead of silently treating it as a successful update.
                log.warning(
                    "run_extraction: field %r flagged as corrected, but the model's new "
                    "value (%r) is identical to the existing value - the correction was "
                    "likely detected but not actually applied. Keeping existing value. "
                    "justification given: %r",
                    fid, known_value, justification,
                )
                extracted_data[fid] = {"value": known_value, "updated_at": known_updated_at}
                continue

            if not justification:
                log.warning(
                    "run_extraction: field %r was corrected (%r -> %r) with no justification from the model",
                    fid, known_value, model_value,
                )
            extracted_data[fid] = {"value": model_value, "updated_at": now_iso}
            changed_fields.append({
                "field_id": fid,
                "old_value": known_value,
                "new_value": model_value,
                "justification": justification or "",
            })
            continue

        if not has_known_value and has_model_value:
            if not justification:
                log.warning(
                    "run_extraction: field %r was set to %r with no justification from the model",
                    fid, model_value,
                )
            extracted_data[fid] = {"value": model_value, "updated_at": now_iso}
            changed_fields.append({
                "field_id": fid,
                "old_value": None,
                "new_value": model_value,
                "justification": justification or "",
            })
            continue

        # Neither known before nor found this run.
        extracted_data[fid] = {"value": None, "updated_at": known_updated_at}

    usage_dict = resp.usage.model_dump() if resp.usage else None
    cost_details = calculate_token_cost(usage_dict, resp.model)

    usage = {
        "prompt_tokens": usage_dict.get("prompt_tokens") if usage_dict else None,
        "completion_tokens": usage_dict.get("completion_tokens") if usage_dict else None,
        "cost": cost_details.get("total_cost") if cost_details else None,
        "model": resp.model,
    }

    return {
        "extracted_data": extracted_data,
        "changed_fields": changed_fields,
        "usage": usage,
    }
