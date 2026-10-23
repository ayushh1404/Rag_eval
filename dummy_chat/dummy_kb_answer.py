import os
import json
import re
import logging
from pathlib import Path
from openai import OpenAI

from datetime import datetime
from zoneinfo import ZoneInfo


from dummy_chat.date_parser import resolve_structured_date


_pricing_cache = None

def _build_extraction_instruction_block(fields: list[dict]) -> str:
    lines = []
    for f in fields:
        fid = f.get("id")
        ftype = f.get("type")
        instruction = f.get("instruction", "").strip()
        options = f.get("options") or []

        lines.append(f"Field ID: {fid}")
        lines.append(f"Type: {ftype}")
        lines.append(f"Instruction: {instruction}")

        if ftype == "CHOICE" and options:
            lines.append(f"Allowed Options: {options}")

        lines.append("")  # spacing

    return "\n".join(lines)

def _build_working_memory_block(fields: list[dict], previous_data: dict | None) -> str:
    """
    Generically build the Conversation Working Memory block from
    extraction_fields + previously saved extracted_data.
    No field name or assistant-specific logic is hardcoded here -
    everything is derived from the field definitions passed in.
    """
    fields = fields or []
    previous_data = previous_data if isinstance(previous_data, dict) else {}

    if not fields:
        return ""

    structured_lines = []
    completed = []
    missing_mandatory = []
    pending_clarification = []
    refused = []

    for f in fields:
        fid = f.get("id")
        if not fid:
            continue
        mandatory = bool(f.get("mandatory", False))
        instruction = (f.get("instruction") or "").strip()
        label = f"{fid} ({instruction})" if instruction else fid

        value = previous_data.get(fid)
        is_refused = bool(previous_data.get(f"{fid}_refused"))
        needs_clarification = bool(previous_data.get(f"{fid}_needs_clarification"))

        if is_refused:
            structured_lines.append(f"- {fid}: [user declined to provide this]")
            refused.append(label)
            continue

        has_value = value is not None and value != ""
        if has_value:
            structured_lines.append(f"- {fid}: {value}")
            completed.append(label)
        else:
            structured_lines.append(f"- {fid}: [not yet provided]")
            if mandatory:
                missing_mandatory.append(label)

        if needs_clarification:
            pending_clarification.append(label)

    lines = []
    lines.append("Structured Information:")
    lines.extend(structured_lines if structured_lines else ["- [none]"])
    lines.append("")
    lines.append("Completed Fields:")
    lines.append(", ".join(completed) if completed else "[none]")
    lines.append("")
    lines.append("Missing Mandatory Fields:")
    lines.append(", ".join(missing_mandatory) if missing_mandatory else "[none]")
    lines.append("")
    lines.append("Pending Clarifications:")
    lines.append(", ".join(pending_clarification) if pending_clarification else "[none]")
    lines.append("")
    lines.append("Refused Fields:")
    lines.append(", ".join(refused) if refused else "[none]")

    return "\n".join(lines)


def _build_working_memory_dict(fields: list[dict], data: dict | None) -> dict:
    """
    Structured (JSON-friendly) version of the Conversation Working Memory,
    for debugging/verification purposes only. Same generic logic as
    _build_working_memory_block, but returns a dict instead of prompt text.
    Driven entirely by extraction_fields - no field-specific hardcoding.
    """
    fields = fields or []
    data = data if isinstance(data, dict) else {}

    structured_information = {}
    completed_fields = []
    missing_mandatory_fields = []
    pending_clarifications = []
    refused_fields = []

    for f in fields:
        fid = f.get("id")
        if not fid:
            continue
        label = f.get("name") or fid
        mandatory = bool(f.get("mandatory", False))

        value = data.get(fid)
        is_refused = bool(data.get(f"{fid}_refused"))
        needs_clarification = bool(data.get(f"{fid}_needs_clarification"))

        if is_refused:
            structured_information[label] = None
            refused_fields.append(label)
            continue

        has_value = value is not None and value != ""
        structured_information[label] = value if has_value else ""

        if has_value:
            completed_fields.append(label)
        elif mandatory:
            missing_mandatory_fields.append(label)

        if needs_clarification:
            pending_clarifications.append(label)

    return {
        "structured_information": structured_information,
        "completed_fields": completed_fields,
        "missing_mandatory_fields": missing_mandatory_fields,
        "pending_clarifications": pending_clarifications,
        "refused_fields": refused_fields,
    }


def _build_json_schema(fields: list[dict]) -> dict:
    properties = {}
    DATE_FIELD_HINTS = ("date", "_when", "schedule")

    for f in fields:
        fid = f["id"]
        ftype = f["type"]

        if ftype == "OPEN":
            properties[fid] = {"type": ["string", "null"]}

        elif ftype == "YES_NO":
            properties[fid] = {"type": ["boolean", "null"]}

        elif ftype == "CHOICE":
            options = f.get("options", [])
            properties[fid] = {
                "type": ["string", "null"],
                "enum": options + [None]
            }

        # Additive, generic companion flags used to power Conversation
        # Working Memory (Refused Fields / Pending Clarifications).
        # These are derived the same way for every field, regardless of id.
        properties[f"{fid}_refused"] = {"type": ["boolean", "null"]}
        properties[f"{fid}_needs_clarification"] = {"type": ["boolean", "null"]}

        if any(hint in fid.lower() for hint in DATE_FIELD_HINTS):
            properties[f"{fid}_offset_type"] = {
                "type": ["string", "null"],
                "enum": ["today", "tomorrow", "day_after_tomorrow", "this",
                         "next", "next_week", "in_days", "in_weeks", "in_months", None]
            }
            properties[f"{fid}_weekday"] = {
                "type": ["string", "null"],
                "enum": ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday", None]
            }
            properties[f"{fid}_offset_n"] = {"type": ["integer", "null"]}

    field_ids = [f["id"] for f in fields if f.get("id")]

    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "extracted_data": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
            "current_focus_field": {
                "type": ["string", "null"],
                "enum": field_ids + [None],
                "description": (
                    "The single field id from Extraction Field Definitions that "
                    "your 'answer' is asking the user about right now. Null if "
                    "your answer is not asking about any specific field."
                ),
            },
        },
        "required": ["answer", "extracted_data", "current_focus_field"],
        "additionalProperties": False,
    }

def _carry_forward_known_values(fields: list[dict], new_data: dict, previous_data: dict | None) -> dict:
    """
    Safety net for field-value loss across turns.

    The model is instructed to reproduce already-known Working Memory
    values in its own JSON output every turn (the schema requires every
    field id to be present). If the model ever forgets - because its
    attention was on a different field this turn - it can return null for
    a field that was already genuinely filled, and since the caller
    persists whatever this function returns as the new "previous_extracted_data"
    for the next turn, that silently erases a value the user already gave
    (e.g. a phone number). This is generic over whatever fields are
    configured - it never hardcodes a field id.

    Rule: if the new turn has no value and no fresh refusal for a field,
    but Working Memory already had a genuine value (or an existing refusal)
    for it, keep the existing one. A real value or a real refusal given
    THIS turn always wins over the old state.
    """
    previous_data = previous_data if isinstance(previous_data, dict) else {}
    if not isinstance(new_data, dict):
        return new_data

    for f in fields:
        fid = f.get("id")
        if not fid:
            continue

        new_value = new_data.get(fid)
        old_value = previous_data.get(fid)
        new_refused = bool(new_data.get(f"{fid}_refused"))
        old_refused = bool(previous_data.get(f"{fid}_refused"))

        has_new_value = new_value is not None and new_value != ""
        has_old_value = old_value is not None and old_value != ""

        if not has_new_value and not new_refused:
            if has_old_value:
                new_data[fid] = old_value
            elif old_refused:
                new_data[f"{fid}_refused"] = True

    return new_data



def _default_pricing_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "openai_pricing.json"


def _load_pricing_config() -> dict:
    global _pricing_cache
    if _pricing_cache is not None:
        return _pricing_cache
    pricing_path = os.getenv("OPENAI_PRICING_FILE") or str(
        _default_pricing_path()
    )
    path_obj = Path(pricing_path)
    if not path_obj.is_absolute():
        path_obj = Path(__file__).resolve().parents[1] / path_obj
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            _pricing_cache = json.load(f)
    except Exception:
        _pricing_cache = {}
    return _pricing_cache


def _get_model_pricing(model_name: str | None) -> dict | None:
    if not model_name:
        return None
    config = _load_pricing_config()
    tier = os.getenv("OPENAI_PRICE_TIER") or config.get(
        "default_tier") or "standard"
    tiers = config.get("tiers", {})
    tier_pricing = tiers.get(tier, {})
    if not tier_pricing:
        return None
    aliases = config.get("aliases", {})
    if model_name in aliases:
        model_name = aliases[model_name]
    for key in sorted(tier_pricing.keys(), key=len, reverse=True):
        if model_name == key or model_name.startswith(key):
            return tier_pricing[key]
    return None


def calculate_token_cost(usage: dict | None, model_name: str | None) -> dict | None:
    if not usage:
        return None
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return None
    input_price = pricing.get("input_per_million")
    output_price = pricing.get("output_per_million")
    cached_input_price = pricing.get("cached_input_per_million")
    if input_price is None or output_price is None:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    if isinstance(cached_tokens, int) and cached_input_price is not None:
        non_cached_tokens = max(prompt_tokens - cached_tokens, 0)
        cached_input_cost = (cached_tokens / 1_000_000) * cached_input_price
        non_cached_input_cost = (non_cached_tokens / 1_000_000) * input_price
        input_cost = cached_input_cost + non_cached_input_cost
    else:
        cached_tokens = None
        cached_input_cost = None
        non_cached_input_cost = None
        input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost
    return {
        "model": model_name,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
            "cached_tokens": cached_tokens,
        },
        "pricing_per_million": {
            "input": input_price,
            "cached_input": cached_input_price,
            "output": output_price,
        },
        "input_cost": input_cost,
        "cached_input_cost": cached_input_cost,
        "non_cached_input_cost": non_cached_input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

# ═══════════════════════════════════════════════════════════════════
# LAZY INITIALIZATION
# ═══════════════════════════════════════════════════════════════════


_openai_client = None


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

PLAIN_TEXT_FORMAT_RULES = """
OUTPUT FORMAT POLICY (STRICT):
- Responses must be plain text only.
- Markdown formatting is forbidden.
- Do not use formatting characters such as: *, **, -, •, _, `, #.
- Do not bold or italicize words.
- If listing items, use numbered lists only:
  1. Item one
  2. Item two

NO PLACEHOLDER TEXT (STRICT):
- You are always given the exact CURRENT DATETIME above. If the user asks
  what today's date is, what a weekday/relative date resolves to, or any
  other simple date question, you must compute and state the real answer
  directly using that CURRENT DATETIME - never write a bracketed
  placeholder such as "[insert today's date]", "[insert next Monday's
  date]", "[DATE]", or similar fill-in-the-blank text for a human or
  another system to complete later. Every date you mention must be an
  actual resolved value, not a template.
"""

_PLACEHOLDER_LEAK_PATTERN = re.compile(
    r"\[\s*insert[^\]]{0,80}\]", re.IGNORECASE
)


def _strip_placeholder_leak(answer_text: str) -> str:
    """
    Safety net: if the model ignores the NO PLACEHOLDER TEXT instruction
    above and leaks template fill-in-the-blank brackets (e.g. "[insert
    today's date]") straight into the reply, don't show that broken text
    to the end user. Log it for visibility and fall back to a safe,
    honest response instead of guessing/reconstructing the missing date.
    """
    if not answer_text or not _PLACEHOLDER_LEAK_PATTERN.search(answer_text):
        return answer_text
    logging.getLogger(__name__).warning(
        "answer_from_kb: model leaked placeholder text into answer: %r",
        answer_text,
    )
    return (
        "Sorry, let me get that date confirmed properly - could you tell me "
        "the exact date you'd like?"
    )


def answer_from_kb(
    query: str,
    chunks: list[str],
    include_raw: bool = False,
    user_prompt: str | None = None,
    user_persona: dict | None = None,
    conversation_history: list | None = None,
    kb_topics: list[str] | None = None,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    contact_name: str | None = None,
    preferred_name: str | None = None,
    is_first_message: bool = False,
    extraction_fields: list[dict] | None = None,
    previous_extracted_data: dict | None = None,
    debug: bool = False,
    force_extraction_mode: bool | None = None,
):
    """
    Generate answer from knowledge base chunks using GPT

    Args:
        query: User's question
        chunks: List of relevant text chunks from KB
        force_extraction_mode: overrides the default extraction-mode
            decision (which is otherwise just "were extraction_fields
            passed?"). Pass False to get a plain conversational answer
            (no JSON schema, no forced structured output) while still
            building the Working Memory block from the given
            extraction_fields/previous_extracted_data - lets a fast,
            no-schema call still be aware of what's already known,
            without paying the cost of the full extraction schema.
            None (default) preserves the original behavior everywhere
            else: is_extraction_mode is derived purely from whether
            extraction_fields is non-empty.

    Returns:
        Generated answer
    """
    openai = get_openai_client()

    context = "\n\n---\n\n".join(chunks)

    extraction_fields = extraction_fields or []
    is_extraction_mode = (
        len(extraction_fields) > 0
        if force_extraction_mode is None
        else force_extraction_mode
    )

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    current_datetime = now_ist.strftime("%A, %d %B %Y %H:%M:%S IST")
    today_ist = now_ist.date()

    user_prompt_clean = (user_prompt or "").strip()
    user_persona_clean = user_persona if isinstance(
        user_persona, dict) else None
    history_clean = conversation_history if isinstance(
        conversation_history, list) else []
    topics_clean = [t for t in (kb_topics or [])
                    if isinstance(t, str) and t.strip()]
    # if user_prompt_clean:
    #     system_prompt = (
    #         f"{user_prompt_clean}\n"
    #         "\n"
    #         "CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:\n"
    #         "1. ROLE ADHERENCE:\n"
    #         "   - You MUST strictly follow the role, persona, tone, and behavior defined above\n"
    #         "   - Maintain the expertise level and communication style specified in your role\n"
    #         "   - Stay in character at all times\n"
    #         "\n"
    #         "2. FACTUAL ACCURACY:\n"
    #         "   - Answer ONLY using the provided context for factual claims\n"
    #         "   - NEVER invent or hallucinate information\n"
    #         "   - If factual info is not in context, acknowledge this professionally\n"
    #         "\n"
    #         "3. CONVERSATION FLOW:\n"
    #         "   - ALWAYS analyze the COMPLETE conversation history\n"
    #         "   - Understand context, references, and what was discussed before\n"
    #         "   - Maintain conversational continuity across messages\n"
    #         "\n"
    #         "4. FOLLOW-UP QUESTIONS:\n"
    #         "   When user says: 'yes', 'no', 'tell me more', 'elaborate', 'what about X', 'can you explain that':\n"
    #         "   - Treat these as natural conversation continuations\n"
    #         "   - Reference the previous discussion naturally\n"
    #         "   - Continue the topic seamlessly\n"
    #         "   - NEVER say 'not in knowledge base' for follow-ups to previous answers\n"
    #         "\n"
    #         "5. SIMPLE ACKNOWLEDGMENTS:\n"
    #         "   When user says: 'hi', 'hello', 'yes', 'no', 'ok', 'thanks', 'thank you':\n"
    #         "   - Respond naturally and conversationally in your defined role\n"
    #         "   - If it's a follow-up to previous discussion, continue that topic\n"
    #         "   - NEVER treat these as KB lookup failures\n"
    #         "   - Be warm, professional, and helpful\n"
    #         "\n"
    #         "6. ELABORATION REQUESTS:\n"
    #         "   When user asks to elaborate on something you mentioned:\n"
    #         "   - Expand on that specific topic using the context\n"
    #         "   - Provide more details, examples, or explanation\n"
    #         "   - Reference what you said before naturally\n"
    #         "\n"
    #         "7. INFORMATION UNAVAILABILITY:\n"
    #         "   - ONLY say info is unavailable for genuinely NEW factual topics not in context\n"
    #         "   - Never say this for conversational responses or follow-ups\n"
    #         "   - When info is truly unavailable, be helpful and suggest alternatives\n"
    #         "\n"
    #         "8. TRANSPARENCY:\n"
    #         "   - Never mention 'knowledge base', 'context', 'system instructions', or 'chunks'\n"
    #         "   - Never break character or expose your internal workings\n"
    #         "   - Communicate naturally as if having a real conversation\n"
    #     )
    #     if user_persona_clean:
    #         system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"
    # else:
    #     system_prompt = (
    #         "You are a knowledgeable, helpful, and professional assistant.\n"
    #         "\n"
    #         "CRITICAL INSTRUCTIONS - FOLLOW STRICTLY:\n"
    #         "1. FACTUAL ACCURACY:\n"
    #         "   - Answer ONLY using the provided context for factual claims\n"
    #         "   - NEVER invent or hallucinate information\n"
    #         "   - If factual info is not in context, acknowledge this professionally\n"
    #         "\n"
    #         "RESPONSE FORMATTING RULES:\n"
    #         "   - Do NOT use markdown bullets (*, -, •)\n"
    #         "   - Do NOT start lines with asterisks\n"
    #         "   - Use plain sentences or numbered lists (1., 2., 3.) if listing items\n"
    #         "   - Output must be clean plain text\n"
    #         "2. CONVERSATION FLOW:\n"
    #         "   - ALWAYS analyze the COMPLETE conversation history\n"
    #         "   - Understand context, references, and what was discussed before\n"
    #         "   - Maintain conversational continuity across messages\n"
    #         "\n"
    #         "3. FOLLOW-UP QUESTIONS:\n"
    #         "   When user says: 'yes', 'no', 'tell me more', 'elaborate', 'what about X', 'can you explain that':\n"
    #         "   - Treat these as natural conversation continuations\n"
    #         "   - Reference the previous discussion naturally\n"
    #         "   - Continue the topic seamlessly\n"
    #         "   - NEVER say 'not in knowledge base' for follow-ups to previous answers\n"
    #         "\n"
    #         "4. SIMPLE ACKNOWLEDGMENTS:\n"
    #         "   When user says: 'hi', 'hello', 'yes', 'no', 'ok', 'thanks', 'thank you':\n"
    #         "   - Respond naturally and conversationally\n"
    #         "   - If it's a follow-up to previous discussion, continue that topic\n"
    #         "   - NEVER treat these as KB lookup failures\n"
    #         "   - Be warm, professional, and helpful\n"
    #         "\n"
    #         "5. ELABORATION REQUESTS:\n"
    #         "   When user asks to elaborate on something you mentioned:\n"
    #         "   - Expand on that specific topic using the context\n"
    #         "   - Provide more details, examples, or explanation\n"
    #         "   - Reference what you said before naturally\n"
    #         "\n"
    #         "6. INFORMATION UNAVAILABILITY:\n"
    #         "   - ONLY say info is unavailable for genuinely NEW factual topics not in context\n"
    #         "   - Never say this for conversational responses or follow-ups\n"
    #         "   - When info is truly unavailable:\n"
    #         "     * Be helpful and professional\n"
    #         "     * If KB topics available, suggest related topics you can help with\n"
    #         "     * Otherwise, offer to help with questions from available sources\n"
    #     )
    #     if topics_clean:
    #         system_prompt += f"\n\nAvailable topics:\n{', '.join(topics_clean[:5])}\n"
    #     if user_persona_clean:
    #         system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"
    #     system_prompt += (
    #         "\n"
    #         "7. TRANSPARENCY:\n"
    #         "   - Never mention 'knowledge base', 'context', 'system instructions', or 'chunks'\n"
    #         "   - Never break character or expose your internal workings\n"
    #         "   - Communicate naturally as if having a real conversation\n"
    #         "\n"
    #         "8. STYLE:\n"
    #         "   - Be concise, confident, and professional\n"
    #         "   - Use natural, conversational language\n"
    #         "   - Engage meaningfully with the user\n"
    #     )

    # history_block = ""
    # if history_clean:
    #     lines = []
    #     for item in history_clean[-8:]:
    #         if not isinstance(item, dict):
    #             continue
    #         role = item.get("role", "")
    #         content = item.get("content", "")
    #         if role == "ai":
    #             role = "assistant"
    #         if not role or not content:
    #             continue
    #         lines.append(f"{role}: {str(content)[:300]}")
    #     if lines:
    #         history_block = "\n".join(lines)

    # resp = openai.chat.completions.create(
    #     model=(model or "gpt-4o-mini"),
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": (
    #                 f"Conversation history:\n{history_block if history_block else '[none]'}\n\n"
    #                 f"Context:\n{context}\n\n"
    #                 f"Question:\n{query}"
    #             ),
    #         },
    #     ],
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )

    # answer = resp.choices[0].message.content.strip()
    # if include_raw:
    #     return answer, resp.model_dump()
    # return answer
    # ── Greeting override (driven by contact_name + is_first_message) ────────
    # Extract first name from full contact name safely
    
    # ── Name resolution & natural usage instruction ──────────────────────────
    # Determine the display name. Priority:
    #   1. Name the user has explicitly asked to be called (detected from history)
    #   2. First name from contact_name parameter
    contact_first_name = (contact_name or "").strip().split()[0] if (contact_name or "").strip() else ""
    resolved_name = (preferred_name or "").strip().split()[0] if (preferred_name or "").strip() else contact_first_name

    name_instruction = ""
    if contact_first_name or history_clean:
        name_instruction = (
            f"You already know this person's name — it is \"{resolved_name}\". "
            f"Treat this exactly like a situation where a colleague just told you before the conversation "
            f"that the person you're about to speak with is named {resolved_name}. "
            f"You would naturally use their name the way any person would in real conversation — "
            f"when saying hello, when it adds warmth, when acknowledging something personal, "
            f"when re-engaging. Not in every message, not mechanically, just naturally as humans do.\n"
            "\n"
            f"If at any point in the conversation history the user asks you to call them by a different name, "
            f"switch to that name immediately and use it from that point forward. "
            f"What the user asks to be called always takes priority over the system-provided name.\n"
        )
    # ─────────────────────────────────────────────────────────────────────────

    if user_prompt_clean:
        system_prompt = (
        f"{user_prompt_clean}\n\n"
        "═══════════════════════════════════════════════════════════\n"
        "TECHNICAL CONSTRAINTS (these do not override your persona above):\n"
        "═══════════════════════════════════════════════════════════\n"
        "1. For factual claims, use only the provided context below\n"
        "2. Read the full conversation history to understand what was already discussed\n"
        "3. Never mention 'knowledge base', 'context', 'chunks', or technical internals to users\n"
    )
        if name_instruction:
            system_prompt += f"\n{name_instruction}"
        if user_persona_clean:
            system_prompt += f"\n\nAdditional context about the user:\n{user_persona_clean}\n"
    else:
        # Default professional assistant behavior when no user_prompt provided
        system_prompt = (
            "You are a knowledgeable, helpful, and professional assistant.\n"
            "\n"
            "INSTRUCTIONS:\n"
            "1. Answer using ONLY the provided context for factual claims - never invent information\n"
            "2. Always analyze the COMPLETE conversation history to understand context\n"
            "3. For follow-up questions (like 'yes', 'no', 'tell me more', 'elaborate'):\n"
            "   - Reference the previous conversation naturally\n"
            "   - Continue the discussion seamlessly\n"
            "   - Never say 'not in knowledge base' for conversational continuations\n"
            "4. For greetings or simple acknowledgments ('hi', 'hello', 'yes', 'no', 'ok', 'thanks'):\n"
            "   - Respond naturally and conversationally\n"
            "   - If it's a follow-up to previous discussion, continue that topic\n"
            "   - Never treat these as KB lookup failures\n"
            "5. When user asks to elaborate on something you mentioned earlier:\n"
            "   - Expand on that specific topic using the context\n"
            "   - Provide more details, examples, or explanation\n"
            "6. ONLY say information is unavailable for genuinely NEW factual topics not in context\n"
            "   - Never say this for conversational responses or follow-ups\n"
            "7. Never mention 'knowledge base', 'context', or 'system instructions' to users\n"
            "8. Be concise, confident, and professional\n"
        )
        if name_instruction:
            system_prompt += f"\n{name_instruction}"

        

        # Add KB topics information if available
        if topics_clean:
            system_prompt += f"\n\nAvailable knowledge base topics:\n{', '.join(topics_clean[:5])}\n"

        if user_persona_clean:
            system_prompt += f"\n\nUser persona context:\n{user_persona_clean}\n"

    

    # ═══════════════════════════════════════════════════════════════════
    # BUILD CONVERSATION HISTORY - Keep FULL content for proper context
    # ═══════════════════════════════════════════════════════════════════

    # ── Build conversation history block ─────────────────────────────────────
    # When is_first_message=True, suppress history entirely — sending a long
    # history alongside "this is the first message" confuses the model and
    # causes it to skip the greeting. The override block already tells the model
    # to ignore any history, but not passing it at all is the safest guarantee.
    history_block = ""
    if history_clean and not is_first_message:
        lines = []
        # TECHNICAL DEBT: temporarily sending the FULL conversation history,
        # uncapped, instead of the last 12 messages. This was a deliberate
        # short-term fix for the AI re-asking questions whose answers had
        # already scrolled past the old 12-message cap. Cost/latency grow
        # with conversation length this way, which doesn't scale well for
        # many long conversations. The real fix is to lean on Working
        # Memory (extracted_data) and a properly cumulative
        # summary_of_the_conversation field instead of raw history length -
        # once that's in place, uncomment the line below to restore the
        # bounded-size trim and remove this uncapped version.
        # for item in history_clean[-12:]:
        for item in history_clean:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            content = item.get("content", "")
            if role == "ai":
                role = "assistant"
            if not role or not content:
                continue
            # Keep FULL content - critical for understanding follow-ups and user info collection
            lines.append(f"{role}: {str(content)}")
        if lines:
            history_block = "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    # BUILD USER MESSAGE - Make it crystal clear what needs to be analyzed
    # ═══════════════════════════════════════════════════════════════════

    # Conversation Working Memory - generic, built from extraction_fields +
    # previously saved extracted_data. Injected before conversation history
    # so the model treats it as the authoritative, up-to-date snapshot.
    working_memory_block = _build_working_memory_block(
        extraction_fields, previous_extracted_data
    )
    _working_memory_dict_pre_call = _build_working_memory_dict(
        extraction_fields, previous_extracted_data
    )

    # Previous Question Focus - which single extraction field (if any) the
    # assistant's last turn was actually asking about. Generic: works for
    # any field id, not tied to any specific assistant's fields. This is
    # what lets an ambiguous reply like a bare number or a partial answer
    # be resolved against the right field instead of guessed.
    _prev_focus_id = (previous_extracted_data or {}).get("__current_focus_field__")
    _prev_focus_field_def = None
    if _prev_focus_id:
        for _f in extraction_fields:
            if _f.get("id") == _prev_focus_id:
                _prev_focus_field_def = _f
                break

    user_message_parts = []

    # 0. Conversation Working Memory (source of truth for structured data already collected)
    if working_memory_block:
        user_message_parts.append(
            "CONVERSATION WORKING MEMORY (source of truth for structured data "
            f"already collected in this conversation):\n{working_memory_block}"
        )

    # 0.5 Previous Question Focus (what the CURRENT USER MESSAGE is most likely answering)
    if _prev_focus_field_def:
        user_message_parts.append(
            "PREVIOUS QUESTION FOCUS:\n"
            f"Your previous reply was asking specifically about field '{_prev_focus_field_def.get('id')}' "
            f"({_prev_focus_field_def.get('instruction', '').strip()}).\n"
            "Interpret the CURRENT USER MESSAGE as most likely answering this field first, "
            "especially if it is short, a bare number, or otherwise ambiguous on its own. "
            "Only attribute it to a different field if it clearly does not answer this one."
        )

    # 1. Conversation history (most important for understanding context)
    if history_block:
        user_message_parts.append(
            f"CONVERSATION HISTORY (Read carefully to understand what's been discussed and what user info you have):\n{history_block}"
        )
    else:
        user_message_parts.append(
            "CONVERSATION HISTORY:\n[This is the start of the conversation]")

    # 2. Context from KB
    if context:
        user_message_parts.append(
            f"\nKNOWLEDGE BASE CONTEXT (Use this for answering factual questions):\n{context}"
        )
    else:
        user_message_parts.append(
            "\nKNOWLEDGE BASE CONTEXT:\n[No relevant context found - if user asks about information not in above history, you don't have it]")

    # 3. Current user query (what they're saying right now)
    user_message_parts.append(
        f"\nCURRENT USER MESSAGE:\n{query}\n\n"
        "---\n"
        "Now respond according to your instructions. Remember:\n"
        "- Follow your conversation flow exactly as instructed\n"
        "- Check the conversation history to see what user info you've already collected\n"
        "- Continue your information gathering process if instructed to do so\n"
        "- For follow-up questions about previous topics, elaborate naturally\n"
    )

    user_message = "\n".join(user_message_parts)

    # ═══════════════════════════════════════════════════════════════════
    # CALL OPENAI API (Normal OR Extraction Mode)
    # ═══════════════════════════════════════════════════════════════════

    # enforce consistent output formatting
    system_prompt += "\n\n" + PLAIN_TEXT_FORMAT_RULES

    # Current date/time must be known regardless of mode - not just in
    # extraction mode. Without this, a plain conversational call (e.g. the
    # fast no-schema reply used by /chat_sse) has zero grounding for "what's
    # today's date" style questions and the model ends up guessing/
    # hallucinating a wrong date instead of stating the real one.
    system_prompt += f"\n\nCURRENT DATETIME: {current_datetime}\n"

    # Conversation Working Memory policy - short, generic instruction.
    # Applies whether or not extraction_fields are configured for this assistant.
    system_prompt += """

CONVERSATION WORKING MEMORY POLICY (STRICT):
- The CONVERSATION WORKING MEMORY section in the user message is the source of truth for any structured information already collected.
- Do not ask again for fields already marked as completed, unless the user brings that field up again or wants to change it.
- Respect fields marked as refused - never ask for them again.
- Naturally work through EVERY field listed in Extraction Field Definitions that is still empty and not refused - mandatory fields first, then optional ones, choosing which one to ask next using the NATURAL ORDERING logic defined below (stay on the user's current topic, core details before timing, timing before contact/callback logistics - but always follow the user's own lead if they jump ahead). Do not stop collecting just because the mandatory fields are done; optional fields still need to be worked through too, unless the assistant persona above gives an EXPLICIT instruction to skip that exact field for this situation (e.g. "do not ask company name for job seekers"). A field simply not being mentioned in the persona's own summary or checklist section is NOT an explicit skip instruction - it still needs to be handled per the FIELD HANDLING MODE and MISSING FIELD FOLLOW-UP RULE below.
- Never interrupt or derail answering the user's actual question just to collect data - answer first, then continue collecting naturally afterward.
- A general closing or polite remark from the user (e.g. "thank you", "no thanks", "that's all", "bye") is NOT a refusal of any specific field - it only means they have nothing more to add right now. Do not treat it as blanket permission to stop collecting or to end the conversation while fields remain unfilled and unrefused. Only mark a field refused when the user explicitly declines that exact field when it was being asked about.
- If knowledge base context is provided below, answer the user's question using it first.
- If no knowledge base context is available, continue the conversation normally using the Working Memory and conversation history.
"""

    if is_extraction_mode:
        extraction_block = _build_extraction_instruction_block(extraction_fields)

        system_prompt += f"""

        ═══════════════════════════════════════════════════════════
         EXTRACTION MODE
        ═══════════════════════════════════════════════════════════

        You have TWO responsibilities:
        1) Answer naturally using provided context.
        2) Extract structured data according to field definitions.

        CURRENT DATETIME: {current_datetime}
        DATE INTERPRETATION RULES:
- CURRENT DATETIME: {current_datetime}
- Do NOT compute or output a final calendar date yourself — date
  arithmetic is handled by a separate deterministic system. Your job
  is only to classify the user's scheduling request into structured
  fields:
    meeting_date_offset_type: one of
      "today", "tomorrow", "day_after_tomorrow", "this", "next",
      "next_week", "in_days", "in_weeks", "in_months"
    meeting_date_weekday: the weekday name in lowercase
      (e.g. "friday"), required when offset_type is "this", "next",
      or "next_week"; null otherwise.
    meeting_date_offset_n: an integer, required when offset_type is
      "in_days", "in_weeks", or "in_months" (e.g. "10 days from now"
      -> offset_type="in_days", offset_n=10); null otherwise.
- Examples:
    "next friday" -> offset_type="next", weekday="friday"
    "this friday" / bare "friday" -> offset_type="this", weekday="friday"
    "next week's friday" / "friday next week" -> offset_type="next_week", weekday="friday"
    "day after tomorrow" -> offset_type="day_after_tomorrow"
    "2 weeks from now" -> offset_type="in_weeks", offset_n=2
    "10 days from now" -> offset_type="in_days", offset_n=10
- If the user declines one date and proposes another in the same
  message (e.g. "not friday, this friday instead" or "I'm busy next
  friday, can we do this friday instead"), classify ONLY the date the
  user is finally confirming, not the one they are declining.
- If you cannot confidently classify the request this way, leave
  meeting_date_offset_type null.
- NEVER set both a weekday AND an offset_n on the same field at the same
  time - they belong to mutually exclusive offset_types. weekday is only
  ever non-null when offset_type is "this", "next", or "next_week".
  offset_n is only ever non-null when offset_type is "in_days",
  "in_weeks", or "in_months". Setting both together produces a wrong,
  inconsistent date from the deterministic resolver.
- These offset_type / weekday / offset_n fields are ONE-TURN-ONLY
  instructions to the deterministic resolver, not persistent state - they
  are the opposite of every other field in this schema. Unlike other
  fields, do NOT carry these forward from Working Memory or from what you
  set in a previous turn. Set them ONLY when the CURRENT user message
  itself contains a new scheduling / date request that needs resolving
  right now. On every other turn, leave all three null, even if a date
  field's plain value (e.g. callback_date itself) already has a
  carried-forward value from before.

        Extraction Field Definitions:
        {extraction_block}

        Rules:
        - For OPEN: extract string or return null.
        - For YES_NO: return true, false, or null.
        - For CHOICE:
    • ONLY return a value if the user explicitly mentions one of the allowed options EXACTLY.
    • Do NOT infer.
    • Do NOT interpret.
    • Do NOT autocorrect spelling.
    • Do NOT map synonyms.
    • If the text does NOT exactly match one of the allowed options, return null.
    • Even if the user's intent seems obvious, return null unless exact match.
        - For each field's companion "{{field_id}}_refused": set true ONLY if the user explicitly declines/refuses to provide that field (e.g. "I'd rather not say", "skip that"). Otherwise false or null. Once true, do not ask for that field again.
        - For each field's companion "{{field_id}}_needs_clarification": set true ONLY if the user gave partial, ambiguous, or unclear information for that field that needs a follow-up question. Otherwise false or null.
        - Always include ALL field IDs in extracted_data.
        - Never include extra fields.
        - Output MUST be valid JSON only.

        CURRENT FOCUS FIELD (applies to every extraction, regardless of which fields exist):
        - Set "current_focus_field" to the single field id your "answer" text is asking the user about right now, so the next turn can correctly interpret a short or ambiguous reply as answering that field. Set it to null if your answer is not asking about any specific field (e.g. it is a summary, a free-form answer, or a welcome message).
        - If a "PREVIOUS QUESTION FOCUS" section is present in the user message, treat the CURRENT USER MESSAGE as most likely answering that field first - especially for a bare number, a short phrase, or anything that could plausibly belong to more than one field. Only assign the value to a different field if it clearly does not answer the focus field.
        - If a field's own instruction says to leave it blank when ambiguous (e.g. "if only a number is given without unit, leave blank until clarified"), follow that instruction literally: leave that field null/unclear and ask exactly what the instruction implies is missing, instead of guessing a value or attributing the ambiguous input to a different, unrelated field just because it happens to also accept a similar-looking value (e.g. a number).

        MERGING PARTIAL UPDATES INTO STRUCTURED (JSON) FIELDS (applies to any OPEN field, whatever it is called):
        - If a field's existing value (shown in Structured Information / Conversation Working Memory) is a valid JSON array or JSON object, and the user's latest message only supplies part of that structure (e.g. answers one property, or answers for only some of several items), you MUST merge the new information into the existing structure - update only the specific property/items the message actually addresses, and preserve every other previously known property and item unchanged. Never discard or blank out existing structured data because a message did not repeat it.
        - Never split or distribute an ambiguous answer arbitrarily across multiple items in a structured field (for example, splitting one combined answer evenly across several existing items) unless the user's message explicitly maps each part to a specific item. If it is unclear which item(s) a partial answer belongs to, do not guess a distribution - instead set that field's "{{field_id}}_needs_clarification" to true and ask a short question naming exactly which items still need that information.

        CARRY FORWARD KNOWN VALUES (applies to every field, not just structured ones):
        - The schema requires every field id in your output on every turn - but that does NOT mean re-deriving each one from scratch from only the current message. For any field that already has a genuine value in Structured Information / Conversation Working Memory, and the current user message does not change, correct, or explicitly clear it, you MUST reproduce that exact existing value in your output - never let it silently become null just because this turn's message was about something else. A field going from filled to empty is a real, visible loss of information to the person you're helping - treat it as seriously as you would treat inventing a value that was never given.
        - The one deliberate exception to this is the {{field_id}}_offset_type / {{field_id}}_weekday / {{field_id}}_offset_n trio described in DATE INTERPRETATION RULES above - those are one-turn-only resolver signals and must NOT be carried forward.
        - Only set a field's "{{field_id}}_refused" to true when the user, in the message being processed right now, explicitly and unambiguously declines to share that exact field (e.g. "I'd rather not say", "skip that", "no, I don't want to share my number"). Do not infer a refusal from the user answering a different question instead, going quiet on it, changing the subject, or giving an unrelated reply - that is simply "not answered yet", not "refused". Once a field is genuinely refused, keep carrying that refused state forward the same way you carry forward a filled value, unless the user later volunteers it anyway.

        FIELD HANDLING MODE (determine this for every field from its own instruction text below - never from its id or name):

        1. DERIVE/CLASSIFY fields - the field's instruction asks you to classify, categorize, or generate a summary of the conversation (look for wording like "classify", "categorize", "generate a summary", or similar). For these fields:
           - Never ask the user a dedicated question to fill this - that would sound robotic.
           - Instead, actively attempt to infer/classify the value from the full conversation history and Working Memory on EVERY turn, using whatever has been discussed so far, even if it was never asked about directly.
           - Only leave it null if the conversation genuinely does not yet contain enough information to classify it - keep re-attempting as more is discussed. Do not leave it null indefinitely once the information needed to classify it is actually present in the conversation.
           - These fields never block the missing-field follow-up flow or delay closing the conversation - they are filled silently in the background, not asked about.

        2. CAPTURE-IF-VOLUNTEERED fields - the field's instruction explicitly says to leave it blank if "not applicable" (as distinct from "not yet known/shared/discussed", which still means it should be asked about). For these fields:
           - Do not proactively ask a dedicated question for it.
           - Fill it only if the user mentions it unprompted.
           - These fields never block the missing-field follow-up flow or delay closing the conversation.

        3. ASK-NATURALLY fields - every other field. These are the ones the MISSING FIELD FOLLOW-UP RULE below applies to: they must be actively asked about (see COMPOUND FIELD EXCEPTION below for how "one at a time" applies), until filled or explicitly refused, before the conversation can close.

        MISSING FIELD FOLLOW-UP RULE (applies to every ASK-NATURALLY field in Extraction Field Definitions, not just mandatory ones):

- After extraction, classify every field id into one of the three FIELD HANDLING MODEs above using its own instruction text, then check its state: filled, refused, or still empty.
- For DERIVE/CLASSIFY and CAPTURE-IF-VOLUNTEERED fields: handle per their mode above; they never block closing the conversation.
- For ASK-NATURALLY fields: "mandatory" only controls priority ORDER and how strictly a missing value is enforced before closing - it does NOT mean non-mandatory fields are optional to ask about. EVERY ASK-NATURALLY field, mandatory or not, must eventually be asked about and reach a state of filled or explicitly refused. None may simply be skipped or forgotten.
- Order of priority: mandatory fields first. Keep asking about missing mandatory fields, one at a time, until each is filled or the user explicitly refuses it.
- Once every mandatory ASK-NATURALLY field is filled or refused, do NOT stop. Continue naturally to the remaining OPTIONAL ASK-NATURALLY fields that are still empty and not refused - unless the assistant persona above gives an EXPLICIT instruction to skip that exact field for this situation (e.g. "do not ask company name for job seekers"). The persona's own summary/checklist section simply not mentioning a field is NOT an explicit skip instruction - that field must still be asked about if it is ASK-NATURALLY and still empty.
- Only treat a field as refused, and stop asking about it, when the user explicitly declines that exact field while it was being asked about (e.g. "I'd rather not share my email", "skip that").
- A general closing, thank-you, or "that's all" style remark is NOT a refusal of any specific field and is NOT, by itself, a reason to end the conversation or produce a final summary while ASK-NATURALLY fields remain unfilled and unrefused. Treat it only as "nothing extra to add beyond what's already been said" and continue asking about the next missing field, phrased naturally and briefly.
- Keep it conversational and professional throughout.
- Only once every ASK-NATURALLY field is filled or explicitly refused should you move to a closing/summary response, if the assistant's own instructions above call for one.

        PER-TURN COMPLETENESS CHECK (do this silently, every single turn, before writing "answer"):
- From Conversation Working Memory, list every ASK-NATURALLY field that is still empty and not refused - across the WHOLE Extraction Field Definitions list, not just fields discussed recently.
- If that list is non-empty, "answer" must NOT be a closing/summary/goodbye statement - it must move the conversation toward one of those remaining fields, chosen using the NATURAL ORDERING rule below.
- Do this on every turn of the conversation, however long it runs - a field left empty many turns ago still counts and must not be forgotten.

        NATURAL ORDERING OF WHICH FIELD TO ASK NEXT (generic reasoning - work this out fresh from each field's own instruction text and the live conversation; never hardcode which literal field ids belong in which group, since fields differ per assistant/config):
- Prefer staying on whatever real-world subject the user just raised: if other still-empty fields describe more detail about that same subject, ask the next missing detail of it before jumping to an unrelated field - the way a person naturally keeps following up on what was just mentioned instead of abruptly changing topics.
- As a general default, ONLY when nothing in the conversation suggests a better order: fields describing the core subject/request the user came to discuss tend to be asked first; fields about timing or planning around that subject (e.g. when something will happen) tend to come next; fields whose sole purpose is arranging contact or a follow-up (e.g. a callback date/time, a phone number, or how to reach the person) tend to come last, once there is something substantive to follow up about.
- Always let the user's own lead override this default. If they jump ahead and volunteer information for a field out of the "expected" order, or raise a logistics detail early, accept and record it immediately (extraction already runs on the full message regardless of what was asked), acknowledge it naturally if appropriate, and never ask about it again - then resume with the next still-empty field using the same ordering logic. An out-of-order answer is never a problem, only a reason to update Working Memory and continue smoothly.

        COMPOUND FIELD EXCEPTION to "one field at a time":
- Some OPEN fields are defined, via their own instruction, to hold a single structured value made of several sub-properties (e.g. an object or a list of items with multiple properties each). Asking about such a field may naturally take one question that touches more than one of its sub-properties at once, because they all belong to the value of ONE field, not several - this is not "listing multiple fields together", it is asking one field the way a person naturally would.
- Distinct field ids are still asked one at a time - never combine two or more separate fields' questions into a single message, and never turn missing-field follow-up into a checklist read-out.

        MANDATORY FIELD HANDLING (important, but never rigid):
- "Mandatory" means the field genuinely needs to be filled (or thoughtfully addressed) before the conversation closes - it does NOT mean it must be asked first, immediately, before anything else, or in one fixed canned sentence. Treat it as a field you care more about getting, not as a trigger that overrides everything else happening in the conversation.
- THE USER'S CURRENT MESSAGE ALWAYS COMES FIRST: whatever the user is actually asking or saying right now (a question, a calculation, a request, small talk, anything) must be fully and directly answered in "answer" before or alongside any mandatory-field follow-up. Never skip, delay, or replace a direct answer to the user with a mandatory-field question instead - that reads as ignoring them, which is worse than asking one turn later.
- Once the user's current message has been properly addressed, you may naturally weave in a mandatory-field question - this turn if it fits smoothly, or a later turn if it doesn't. There is no fixed required order among mandatory fields either, or between mandatory and optional ones - use your judgment on what feels natural given the conversation, same as the NATURAL ORDERING guidance above.
- Always phrase the ask in your own natural words based on what the field is for - never recite a field's raw internal "instruction" text to the user verbatim; that text is guidance for you, not a script to read aloud.
- If the user declines to share a mandatory field ("I don't want to say", "skip that", "no"), do NOT immediately mark it refused and drop it the way you would for an optional field. Instead, briefly and politely explain, in your own words, why that piece of information matters for helping them, and ask once more. Only if they still decline after hearing the reason should you set that field's "_refused" flag and let it go gracefully - without pressuring further, unless the persona above explicitly instructs otherwise for this situation.
- Keep every mandatory-field ask brief, warm, and conversational - it should never feel like an interrogation, a form, or a blocker standing between the user and getting help.

        - The "answer" field must ALWAYS contain a natural reply.
        - Never return an empty answer.
        - Output MUST be valid JSON only.
        """

        schema = _build_json_schema(extraction_fields)

        

        kwargs = {
           "model": model or "gpt-4o-mini",
           "messages": [
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": user_message},
          ],
        
        "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction_response",
            "schema": schema,
            "strict": True,
        },
    },
}
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        resp = openai.chat.completions.create(**kwargs)

        raw_content = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason

        # SAFETY NET: this structured-output schema grows with the number
        # of extraction fields (every field carries companion flags, date
        # fields carry an extra offset trio, plus "answer" and any JSON-
        # holding fields like a hiring breakdown). If the caller passes a
        # max_completion_tokens ceiling that's too low for a given turn,
        # OpenAI truncates the response mid-string, which used to crash
        # the whole request with an unhandled JSONDecodeError ("Unterminated
        # string..."). Detect that specific failure mode and retry once
        # with a much higher ceiling before giving up, instead of a 500.
        #
        # Also covers a second failure mode seen in production: valid JSON
        # that is nonetheless missing the required "extracted_data" key
        # entirely. That should be impossible once "strict": true is set
        # on the schema (see below), but this check stays as defense in
        # depth in case of any future schema change that drops strict mode
        # or omits additionalProperties, silently making the guarantee
        # weaker again.
        def _looks_structurally_valid(candidate) -> bool:
            return (
                isinstance(candidate, dict)
                and isinstance(candidate.get("extracted_data"), dict)
                and isinstance(candidate.get("answer"), str)
            )

        needs_retry = finish_reason == "length"
        if not needs_retry:
            try:
                parsed = json.loads(raw_content)
                if not _looks_structurally_valid(parsed):
                    needs_retry = True
            except json.JSONDecodeError:
                needs_retry = True

        if needs_retry:
            retry_kwargs = dict(kwargs)
            current_cap = kwargs.get("max_completion_tokens")
            # Give it clearly more room than whatever was requested (or a
            # safe floor if no cap was set at all but truncation still
            # somehow occurred, e.g. an unusually long conversation).
            retry_kwargs["max_completion_tokens"] = max(
                (current_cap or 0) * 3, 4000
            )
            resp = openai.chat.completions.create(**retry_kwargs)
            raw_content = resp.choices[0].message.content

        try:
            parsed = json.loads(raw_content)
            if not _looks_structurally_valid(parsed):
                raise ValueError("response missing required 'extracted_data'/'answer' keys")
        except (json.JSONDecodeError, ValueError) as e:
            # Even the retry produced invalid/incomplete JSON - don't crash
            # the request with a raw traceback. Log enough to diagnose and
            # return a natural-sounding fallback instead.
            logger = logging.getLogger(__name__)
            logger.error(
                "answer_from_kb: response invalid even after retry (%s). "
                "raw_content length=%d, finish_reason=%s",
                e, len(raw_content or ""), resp.choices[0].finish_reason,
            )
            fallback = {
                "answer": "Sorry, I had trouble processing that - could you say that again?",
                "extracted_data": {f["id"]: None for f in extraction_fields if f.get("id")},
                "current_focus_field": None,
            }
            if include_raw:
                return fallback, resp.model_dump()
            return fallback

        # Safety net: never let a field the model forgot to repeat this
        # turn silently erase a value it already genuinely had.
        if isinstance(parsed.get("extracted_data"), dict):
            parsed["extracted_data"] = _carry_forward_known_values(
                extraction_fields, parsed["extracted_data"], previous_extracted_data
            )

        # Carry current_focus_field through as a reserved key inside
        # extracted_data, so it persists via the same merge/store pipeline
        # app.py already uses for every other extracted field - no separate
        # plumbing needed, and it stays invisible to Working Memory display
        # since it isn't one of extraction_fields' real ids.
        if isinstance(parsed.get("extracted_data"), dict):
            parsed["extracted_data"]["__current_focus_field__"] = parsed.get("current_focus_field")

        DATE_FIELD_HINTS = ("date", "_when", "schedule")
        resolved_date_field_id = None
        resolved_date_info = None

        for field in extraction_fields:
            fid = field.get("id", "")
            if not any(hint in fid.lower() for hint in DATE_FIELD_HINTS):
                continue

            offset_type = parsed.get("extracted_data", {}).get(f"{fid}_offset_type")
            if not offset_type:
                continue
            weekday = parsed.get("extracted_data", {}).get(f"{fid}_weekday")
            offset_n = parsed.get("extracted_data", {}).get(f"{fid}_offset_n") or 0
            
            resolved = resolve_structured_date(
        offset_type=offset_type,
        weekday=weekday,
        offset_n=offset_n,
        today=today_ist,
            )

            if resolved:
                parsed["extracted_data"][fid] = resolved["date"]            # keep meeting_date itself populated too
                parsed["extracted_data"][f"{fid}_iso"] = resolved["date"]
                parsed["extracted_data"][f"{fid}_display"] = resolved["display"]
                resolved_date_field_id = fid
                resolved_date_info = resolved

            # These offset signals are ONE-TURN-ONLY instructions to this
            # resolver, not persistent state. If they are left as-is, the
            # persistence layer carries them into previous_extracted_data,
            # the model (which is told to carry forward known values) keeps
            # echoing them back on every later turn, and this block then
            # fires forever - silently overwriting "answer" with a stale
            # date-confirmation line no matter what the user actually says
            # next (e.g. declining to share a phone number). Reset them to
            # null immediately after use so they only ever fire on a turn
            # where the CURRENT message actually produced a new one.
            parsed["extracted_data"][f"{fid}_offset_type"] = None
            parsed["extracted_data"][f"{fid}_weekday"] = None
            parsed["extracted_data"][f"{fid}_offset_n"] = None

# ── Force the spoken answer to state the exact resolved date ──────────────
# Don't rely on the model to remember to mention it; guarantee it here.
        if resolved_date_info:
            time_field_value = None
            for field in extraction_fields:
                fid = field.get("id", "")
                if "time" in fid.lower() and fid != resolved_date_field_id:
                    time_field_value = parsed.get("extracted_data", {}).get(fid)
                    break

            if time_field_value:
                parsed["answer"] = (
            f"Got it — I'll book your meeting for {resolved_date_info['display']} "
            f"at {time_field_value}. Let me know if that needs to change."
        )
            else:
                parsed["answer"] = (
            f"Got it — that's {resolved_date_info['display']}. "
            f"What time works for you on that day?"
        )

       

        # ───────────────────────────────────────────────
        # STRICT CHOICE VALIDATION + MANDATORY TRACKING
        # ───────────────────────────────────────────────
        # NOTE: This block used to force-overwrite "answer" whenever a
        # mandatory field was missing. That ignored whatever the user
        # actually asked this turn (e.g. a date calculation or a direct
        # question) and replaced it with a canned line built from the raw
        # field "instruction" text - rigid and often unnatural to read.
        # Mandatory-field follow-up (answering the user first, weaving the
        # ask in naturally, explaining why it matters if the user declines,
        # asking in whatever order feels natural) is now the model's job,
        # driven by the MANDATORY FIELD HANDLING rules in the system
        # prompt. This block now only: (a) enforces that CHOICE fields
        # hold nothing but an exact allowed option, and (b) tracks which
        # mandatory fields are still missing, for validation/debug use.

        query_lower = query.lower()
        mandatory_missing = []

        for field in extraction_fields:
            fid = field.get("id")
            ftype = field.get("type")
            mandatory = field.get("mandatory", False)

            if ftype == "CHOICE":
                allowed = field.get("options") or []
                matched_value = None
                for option in allowed:
                    if option.lower() in query_lower:
                        matched_value = option
                        break

                # Only recompute from the current message if the model
                # didn't already carry forward a valid allowed value -
                # otherwise a CHOICE field answered two turns ago would
                # get silently wiped back to null on every later turn
                # that doesn't happen to repeat the option text.
                existing_choice_value = parsed.get("extracted_data", {}).get(fid)
                if existing_choice_value not in allowed:
                    parsed["extracted_data"][fid] = matched_value

            value = parsed.get("extracted_data", {}).get(fid)
            if mandatory and value is None:
                mandatory_missing.append(field)

        # Safety net only, not the primary mechanism: if the model somehow
        # returned a blank answer this turn, give a natural fallback rather
        # than showing the user nothing. This should rarely fire.
        if not (parsed.get("answer") or "").strip():
            if mandatory_missing:
                missing_field = mandatory_missing[0]
                label = (missing_field.get("name") or missing_field.get("id") or "").replace("_", " ").strip().lower()
                if missing_field.get("type") == "CHOICE":
                    options = missing_field.get("options") or []
                    parsed["answer"] = (
                        f"Sorry, could you clarify - please choose one of: {', '.join(options)}."
                    )
                else:
                    parsed["answer"] = f"Sorry, could you tell me a bit more about {label or 'that'}?"
            else:
                parsed["answer"] = "Sorry, could you say that again?"

        parsed["answer"] = _strip_placeholder_leak(parsed.get("answer"))

        if debug:
            parsed["debug"] = {
                "previous_extracted_data": previous_extracted_data or {},
                "conversation_working_memory": _working_memory_dict_pre_call,
                "llm_extracted_data": parsed.get("extracted_data") or {},
            }

        if include_raw:
            return parsed, resp.model_dump()

        return parsed

    else:


        kwargs = {
    "model": model or "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ],
    
}

        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

        resp = openai.chat.completions.create(**kwargs)
        

        answer = _strip_placeholder_leak(resp.choices[0].message.content.strip())

        # Detect if user requested a name change in this exact turn
        # Ask the model to check only the current query, not history
        detected_new_name = None
        if contact_first_name or resolved_name:
            try:
                kwargs = {
                    "model": "gpt-4o-mini",
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Current known name for this user: \"{resolved_name}\".\n"
                            f"User's current message: \"{query}\"\n\n"
                            f"Did the user ask to be called by a different name in this message? "
                            f"If yes, reply with that new name only — single word or short name, no punctuation. "
                            f"If no name change was requested, reply with the single word: NO"
                        )
                    }],
                    
                }
                if max_tokens is not None:
                    kwargs["max_completion_tokens"] = 10
                    
                
                detection_resp = openai.chat.completions.create(**kwargs)
    
                result = detection_resp.choices[0].message.content.strip()
                if result.upper() != "NO" and result:
                    detected_new_name = result.split()[0]
            except Exception:
                pass

        if detected_new_name:
            if include_raw:
                return {"answer": answer, "preferred_name": detected_new_name}, resp.model_dump()
            return {"answer": answer, "preferred_name": detected_new_name}


        if include_raw:
            return answer, resp.model_dump()

        return answer