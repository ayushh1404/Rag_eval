"""
Builds the system + user prompt for the /extract reconciliation call.

Unlike dummy_chat/dummy_kb_answer.py's per-turn extraction (which asks
"what's new about THIS message"), this is a whole-transcript reconciliation
task: given everything already known (known_data) and the FULL, untrimmed
conversation history, decide what should change and nothing more. The
system prompt is deliberately conservative - it explicitly forbids
rewriting a field just because a fact was repeated, and requires a
justification quote for every change. A code-level safety net in
service.py enforces this too, rather than trusting the prompt alone.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from .schema_builder import is_date_field, is_derive_field


def _field_lines(extraction_fields: list[dict]) -> str:
    lines = []
    for f in extraction_fields:
        label = f.get("name") or f["id"]
        lines.append(f"Field ID: {f['id']}")
        lines.append(f"Name: {label}")
        lines.append(f"Type: {f['type']}")
        lines.append(f"Instruction: {(f.get('instruction') or '').strip()}")
        if f["type"] == "CHOICE" and f.get("options"):
            lines.append(f"Allowed Options: {f['options']}")
        lines.append(f"Mandatory: {bool(f.get('mandatory', False))}")
        if is_derive_field(f):
            # Repeated here, right next to this specific field's own
            # definition, not just once in a separate rules section far
            # above - with many fields in one call, a rule stated only once
            # at the top can get diluted/missed by the time the model
            # reaches this particular field. Restating it at the point of
            # use measurably improves compliance (the same pattern already
            # used for the FIELD ISOLATION reminder below).
            lines.append(
                "REMINDER FOR THIS FIELD SPECIFICALLY: this is a derive/classify field - "
                "generate your best current answer for it RIGHT NOW using everything "
                "confirmed so far in the transcript, even if the conversation doesn't feel "
                "fully finished. Do not leave this null just because more details could "
                "still come later - only leave it null if there is truly nothing relevant "
                "anywhere in the transcript yet."
            )
        lines.append("")
    return "\n".join(lines)


def _known_data_block(known_data: dict) -> str:
    lines = []
    for fid, entry in (known_data or {}).items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None or value == "":
            continue
        lines.append(f"- {fid}: {value}")
    return "\n".join(lines) if lines else "[Nothing known yet - this is the first extraction run for this conversation.]"


RECENT_MESSAGE_COUNT = 6


def _format_messages(conversation_history: list[dict]) -> list[str]:
    lines = []
    for msg in conversation_history:
        role = msg.get("role", "")
        role = "assistant" if role == "ai" else role
        content = msg.get("content", "")
        created_at = msg.get("created_at")
        if not role or not content:
            continue
        stamp = f" [{created_at}]" if created_at else ""
        lines.append(f"{role}{stamp}: {content}")
    return lines


def _transcript_block(conversation_history: list[dict]) -> str:
    return "\n".join(_format_messages(conversation_history))


def _recent_messages_block(conversation_history: list[dict]) -> str:
    lines = _format_messages(conversation_history)[-RECENT_MESSAGE_COUNT:]
    return "\n".join(lines) if lines else "[No messages yet.]"


def build_extraction_prompt(
    extraction_fields: list[dict],
    known_data: dict,
    conversation_history: list[dict],
) -> tuple[str, str]:
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    current_datetime = now_ist.strftime("%A, %d %B %Y %H:%M:%S IST")

    has_date_fields = any(is_date_field(f["id"]) for f in extraction_fields)
    derive_fields = [f for f in extraction_fields if is_derive_field(f)]

    system_prompt = f"""You are a precise data-reconciliation engine. You are given a full conversation
transcript and a set of already-known facts (known_data). Your ONLY job is to decide,
for each field below, whether the transcript changes what is already known - nothing
more. You do not write a reply to anyone; your only output is structured data.

CURRENT DATETIME: {current_datetime}

STRICT RULES:
1. If a field already has a non-null value in known_data, treat that value as correct
   and leave it alone by default. Set "{{field_id}}_correction_detected" to true ONLY
   if the transcript contains an EXPLICIT correction or contradiction for that exact
   field (e.g. "actually it's Thane, not Mumbai", "no wait, my number is..."). A
   passive re-mention of the same fact, or unrelated text, is NOT a correction -
   leave correction_detected false/null in that case. CRITICAL: whenever you set
   "{{field_id}}_correction_detected" to true, the field's own "value" you output MUST
   actually be the NEW, updated value - never repeat the old known_data value while
   also flagging a correction. A correction that doesn't change the value is a
   contradiction: if you flag a correction, your value output must genuinely reflect
   what the transcript says the corrected value now is, and it must match what your
   own justification quote describes. If a location was corrected by adding a new
   location (e.g. "Mumbai, Pune" becoming "Mumbai, Pune, Thane"), the new value must
   include ALL locations that are now confirmed - not just the one that was added, and
   not the old list unchanged. Also: reformatting, re-wording, re-capitalizing,
   reordering, or otherwise restating already-known information is NOT a correction if
   no new or contradicting fact actually appears in the transcript - if the substance
   hasn't changed, leave "{{field_id}}_correction_detected" false/null and leave the
   value exactly as known_data already states it, even if you could phrase it more
   consistently or completely. Only flag a correction when the transcript genuinely
   introduces new or contradicting factual content for that exact field.
2. If a field has no value yet in known_data, extract it normally from the transcript
   if it is present anywhere in the conversation. No correction flag is needed here.
3. FIELD ISOLATION (critical - read carefully): every field's answer belongs ONLY
   under that exact field's own "{{field_id}}" key - never write it under a different
   field's key, even when another field's instruction happens to mention the same
   topic. This most commonly goes wrong with a long-form narrative/summary field whose
   instruction lists many topics (e.g. "mention the salary, the timeline, the
   location..."). That instruction means the narrative field's OWN prose should
   reference those facts in sentence form - it does NOT mean the short, exact value
   for "salary" or "timeline" should be written under the narrative field's id instead
   of its own dedicated field's id. When a fact is covered by both a dedicated field
   (e.g. a "Salary" field) and mentioned inside a narrative field's instruction (e.g.
   a "summary" field), you must fill in BOTH independently and correctly: the
   dedicated field gets its own exact short value under its own id, and the narrative
   field gets its own full sentence-form text under its own id that happens to mention
   the same fact in prose. Never relocate a dedicated field's value into a narrative
   field's slot, and never leave a dedicated field null just because its fact already
   appears inside another field's narrative text.
4. Whenever you set a value for the first time, or flag a correction, treat providing
   "{{field_id}}_justification" as part of that SAME action, not a separate optional
   step: it must be a short quote or close paraphrase from the transcript that is the
   actual evidence for that exact value. If you cannot point to real transcript
   evidence for a value, do not set that value at all - leave it null rather than
   setting a value with no justification. Never fabricate a justification, and never
   leave it blank when a value was set or changed this run. The justification for a
   field must be evidence about THAT SAME field's own subject matter specifically -
   never reuse or paraphrase a quote that is actually evidence for a DIFFERENT field,
   even if it appears in the same message as something relevant to this field (e.g. a
   message confirming a location must never be used as the justification for a role,
   headcount, or any other unrelated field).
5. For CHOICE type fields, only ever output one of the exact allowed options, or null -
   never infer, never autocorrect spelling, never guess a close match.
6. Always include every field id in your output, exactly as listed below - this is
   required by the schema regardless of whether that field changed this run.
7. NEVER copy one field's answer into a different field just because you already have
   it in hand from filling in a nearby field (e.g. do not put an email address into a
   "Role" or "Location" field, do not put a phone number into a "Name" field). Each
   field's value must come ONLY from transcript evidence that actually answers THAT
   field's own instruction. If a field has no such evidence anywhere in the transcript,
   its value is null - it is never acceptable to fill it with a plausible-looking value
   borrowed from another field just to avoid leaving it null.
8. DO NOT RESOLVE A STILL-OPEN QUESTION AS IF IT WERE ANSWERED: if the transcript
   presents multiple candidate values for a field (e.g. "day after tomorrow at 5pm or
   tomorrow from 1pm to 3pm") and the user has not yet clearly picked one of them, or
   if the assistant's own last message about that specific field is asking the user to
   choose/confirm and no user reply follows it anywhere in the transcript, that field's
   value is NOT YET DETERMINED. In that case leave the value null (or, if known_data
   already had a genuinely confirmed prior value, leave that untouched) - never guess
   or pick one of the still-open candidates yourself, even if the transcript simply
   ends right after the options were presented. This is different from a value the
   user HAS clearly confirmed (e.g. replying "yes" to a specific proposed value, or
   plainly restating a single clear answer) - a real confirmation like that is valid
   evidence and should be captured normally. The distinction is: was a single answer
   actually settled by the user, or is it still an open offer awaiting their reply?
"""

    if has_date_fields:
        system_prompt += """
9. For date-type fields, do NOT compute the final calendar date yourself - only
   classify offset_type / weekday / offset_n from the transcript (a separate
   deterministic system resolves the actual date from those). Leave these null if no
   new scheduling information is present in the transcript.
"""

    if derive_fields:
        derive_ids = ", ".join(f["id"] for f in derive_fields)
        system_prompt += f"""

DERIVE/CLASSIFY FIELDS - SPECIAL HANDLING (applies ONLY to these field id(s): {derive_ids}):
These field(s) ask you to classify, categorize, or summarize the conversation as a
whole, rather than extract one discrete fact a person stated. All of the STRICT RULES
above about corrections and "leave it alone unless corrected" do NOT apply to these
specific fields - treat them completely differently:
- On every single run, regenerate your SINGLE BEST current version of this field's
  value using the ENTIRE transcript and everything currently known (known_data plus
  the full conversation) - regardless of whatever value it already had before, and
  regardless of whether anything was "corrected". This is a fresh computation every
  time, not an edit of the old value.
- Do NOT wait for the conversation to feel "fully finished" or "completed" before
  producing a value, even if the field's own instruction uses a word like "completed"
  - generate the best possible version RIGHT NOW using whatever is confirmed so far.
  A partial, honest summary of what's known so far is far more useful than leaving
  this null while waiting for a conversation that may never formally "end". Produce a
  more complete version again next run as more information becomes known.
- Only leave the value null if the transcript genuinely does not yet contain enough
  information to say anything meaningful at all for this field.
- Do NOT set "{{field_id}}_correction_detected" for these fields - it is not used
  for this field type and will be ignored either way. You may leave
  "{{field_id}}_justification" null for these fields, or briefly note what the
  summary/classification is based on - it is not required to be a verbatim quote the
  way it is for regular fact fields.
"""

    system_prompt += f"""

FIELD DEFINITIONS:
(Reminder before you read these: match every answer strictly to its own Field ID
below, per FIELD ISOLATION rule 3 above. A field whose instruction lists many topics
- e.g. a summary - is asking you to WRITE PROSE that mentions those topics under ITS
OWN id, not to relocate other fields' exact values into its slot.)

{_field_lines(extraction_fields)}"""

    user_message = f"""ALREADY KNOWN (do not change without real evidence of an update in the transcript):
{_known_data_block(known_data)}

MOST RECENT MESSAGES (last {RECENT_MESSAGE_COUNT} - pay special attention here: if any of
these state a value for a field that differs from ALREADY KNOWN or from earlier in the
full transcript below, these recent messages are almost always the current, correct
state - make sure the "value" you output actually matches what these recent messages
say, not an older value from earlier in the conversation):
{_recent_messages_block(conversation_history)}

FULL CONVERSATION TRANSCRIPT (complete, not trimmed):
{_transcript_block(conversation_history)}

Reconcile the fields now, following the STRICT RULES above. Double-check: for every
field whose value changed, does your output "value" actually match what the MOST
RECENT MESSAGES say - not an earlier, outdated mention?
"""

    return system_prompt, user_message
