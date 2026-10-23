# POST /extract — Standalone Field Extraction API

## What this is

A dedicated, stateless endpoint that reconciles a set of structured fields (name, phone number, salary, callback date, etc.) against a full conversation transcript. It is completely separate from `/chat` and `/chat_sse` — it does not generate a conversational reply, does not do RAG/knowledge-base retrieval, and does not read or write any server-side memory.

It exists because the earlier design (extraction bundled into every `/chat` turn, backed by an in-memory dict keyed by `conversation_id`) doesn't survive server restarts and doesn't work correctly across multiple server processes/replicas. `/extract` fixes this by being a pure function: everything it needs comes in on the request, everything it produces goes out in the response, and the caller (this backend) is responsible for persisting it wherever makes sense (a real database, not this service's memory).

## When to call it

This is **not** meant to be called on every chat turn — it's designed to be called on a **debounce schedule**, e.g. after ~45 seconds of conversation silence, or every N messages, whichever your product logic prefers. Because it always receives the **full, untrimmed conversation history** (see below), calling it too frequently on a fast-growing conversation will cost more than necessary — that tradeoff is intentional and acceptable for infrequent, debounced calls, not for a call-every-turn pattern.

## Statelessness — the most important property of this endpoint

`/extract` reads nothing from and writes nothing to any server-side store. There is no `conversation_id`-keyed memory here (unlike `/chat`'s `_extracted_data_store`). Every prior fact the endpoint needs must be sent in on the request as `known_data`, and every updated fact comes back in the response as `extracted_data` — persisting that between calls is entirely the calling backend's job. This also means:

- It is **idempotent** — calling it twice with the exact same request body should produce an equivalent result (no hidden side effects, no drift from repeated calls).
- It is **safe to retry** on network failure/timeout without any risk of double-applying anything, since there's nothing being applied on this side to begin with.
- It does not care how many times it's been called before, or by whom — there is no session state.

## Request

`POST /extract`
`Content-Type: application/json`

```json
{
  "conversation_id": "conv_extract_test_1",
  "extraction_fields": [
    {
      "id": "name",
      "name": "Full Name",
      "type": "OPEN",
      "instruction": "Extract the caller's full name exactly as shared.",
      "mandatory": true
    },
    {
      "id": "phone_number",
      "name": "Phone Number",
      "type": "OPEN",
      "instruction": "Extract the callback phone number, digits only.",
      "mandatory": true
    },
    {
      "id": "problem_category",
      "name": "Problem Category",
      "type": "CHOICE",
      "instruction": "Classify the enquiry type.",
      "mandatory": true,
      "options": ["Business Enquiry", "Existing Client", "Investor Query", "Media Query"]
    }
  ],
  "known_data": {
    "problem_category": { "value": "Business Enquiry", "updated_at": "2026-07-14T09:05:00+05:30" }
  },
  "conversation_history": [
    { "role": "user", "content": "Hi, I need to hire 50 people", "created_at": "2026-07-14T09:00:00+05:30" },
    { "role": "ai", "content": "Got it - 50 people. What role are you hiring for?", "created_at": "2026-07-14T09:00:10+05:30" },
    { "role": "user", "content": "My name is Aryan Sharma, number is 9876543210", "created_at": "2026-07-14T09:25:00+05:30" }
  ]
}
```

### Field-by-field

| Field | Type | Required | Notes |
|---|---|---|---|
| `conversation_id` | string | yes | Any non-empty identifier for the conversation. Purely used for logging on this side — it is **not** used to look anything up server-side, since this endpoint is stateless. |
| `extraction_fields` | array | yes, non-empty | The full list of fields to extract/reconcile this run. Must be resent in full every call — this endpoint has no memory of what fields exist for a given assistant. |
| `extraction_fields[].id` | string | yes | Unique identifier for the field (e.g. `"phone_number"`). Must be unique within the array — duplicates are rejected. |
| `extraction_fields[].name` | string | no | Human-readable label, used only inside the prompt for clarity. Falls back to `id` if omitted. |
| `extraction_fields[].type` | `"OPEN" \| "YES_NO" \| "CHOICE"` | yes | `OPEN` → free-text string. `YES_NO` → boolean. `CHOICE` → must exactly match one of `options`, never inferred/interpreted/autocorrected. |
| `extraction_fields[].instruction` | string | yes | Natural-language instruction telling the model what to extract for this field and how (e.g. "digits only", "resolve to DD-MMM-YYYY"). |
| `extraction_fields[].mandatory` | bool | no, default `false` | Not currently used to change extraction behavior — informational only. The caller is responsible for deciding what "mandatory but still missing" means for its own flow (see **Known Gaps** below). |
| `extraction_fields[].options` | array of strings | required if `type == "CHOICE"` | Rejected with a 400 if a CHOICE field has no options. |
| `known_data` | object | no, default `{}` | Everything already known for this conversation, keyed by field `id`. Omit entirely (or pass `{}`) on the very first extraction run for a conversation. |
| `known_data.<field_id>.value` | any | — | The previously known value. Can be `null`/omitted if that field just isn't known yet — that's equivalent to not including the key at all. |
| `known_data.<field_id>.updated_at` | ISO8601 string | no | When that value was last set. Passed through unchanged in the response if the field doesn't change this run. |
| `conversation_history` | array | yes, non-empty | The **full, untrimmed** transcript from the start of the conversation. Do not summarize or truncate it before sending — this endpoint is specifically designed to receive everything, since it's called infrequently rather than every turn. |
| `conversation_history[].role` | `"user" \| "ai"` | yes | `"ai"` is normalized internally to the OpenAI `"assistant"` role. |
| `conversation_history[].content` | string | yes | The message text. |
| `conversation_history[].created_at` | ISO8601 string | no | Included in the prompt for the model's context (useful for reasoning about relative dates/times mentioned in the conversation) but not otherwise required. |

## Response

```json
{
  "extracted_data": {
    "name": { "value": "Aryan Sharma", "updated_at": "2026-07-14T09:25:05+05:30" },
    "phone_number": { "value": "9876543210", "updated_at": "2026-07-14T09:25:05+05:30" },
    "problem_category": { "value": "Business Enquiry", "updated_at": "2026-07-14T09:05:00+05:30" }
  },
  "changed_fields": [
    {
      "field_id": "name",
      "old_value": null,
      "new_value": "Aryan Sharma",
      "justification": "My name is Aryan Sharma, number is 9876543210"
    },
    {
      "field_id": "phone_number",
      "old_value": null,
      "new_value": "9876543210",
      "justification": "My name is Aryan Sharma, number is 9876543210"
    }
  ],
  "usage": {
    "prompt_tokens": 640,
    "completion_tokens": 95,
    "cost": 0.00015,
    "model": "gpt-4o-mini"
  }
}
```

| Field | Notes |
|---|---|
| `extracted_data` | The **full current picture** for every field in `extraction_fields` — not just what changed. Always contains one entry per field you sent in, including ones that stayed `null`/unchanged. This is what the caller should persist as the new `known_data` for the *next* call. |
| `changed_fields` | Only the fields that were actually updated this run (newly set, or explicitly corrected). Each entry includes `old_value`, `new_value`, and a `justification` — a quote/paraphrase from the transcript that is the evidence for the change, for audit/debugging purposes. Empty array if nothing changed. |
| `usage` | Token counts and cost (USD) for this single call, plus which model actually served it, so the caller can bill/log this run accurately. |

## The core behavior rule — conservative by default

A field's value **only changes** in one of two cases:
1. It had no value yet in `known_data` and the transcript contains it somewhere → it gets set, and appears in `changed_fields` with `old_value: null`.
2. It already had a value in `known_data`, and the transcript contains an **explicit correction or contradiction** for that exact field (e.g. "actually it's Thane, not Mumbai", "no wait, my number is...") → it gets updated, and appears in `changed_fields` with the real `old_value`.

A passive re-mention of the same already-known fact, or unrelated conversation text, **must never** alter an existing value. This isn't just a prompt instruction — it's enforced in code (`extraction/service.py`): the model's raw output for an already-known field is only trusted if it separately signals a genuine correction; otherwise the existing `known_data` value/timestamp is carried through untouched, byte-for-byte, regardless of what the model returned for that field's raw value.

Date-type fields (any field `id` containing `date`, `_when`, or `schedule`) get special handling: the model is only asked to classify intent (e.g. "next Friday"), and a separate deterministic Python function computes the actual calendar date — because LLMs are unreliable at exact date arithmetic. This mirrors the same mechanism already used in `/chat`'s extraction path.

## Errors

| Status | When |
|---|---|
| `400` | `extraction_fields` empty; a field missing/empty `id` or `instruction`; duplicate field `id`s; a `CHOICE` field with no `options`; `conversation_history` empty; `conversation_id` missing/empty. Standard FastAPI/Pydantic `422` also applies for malformed JSON shapes (e.g. wrong types). |
| `502` | The model's output was invalid/incomplete even after an automatic retry with a higher token ceiling — treat as a transient upstream failure, safe to retry the whole request. |
| `500` | Unexpected internal error. Logged server-side with full traceback for debugging. |

## Known gaps / things intentionally left out of this version

These were open questions during design that got resolved with a default choice — flagging them explicitly so nothing is assumed silently:

- **No `_refused` / `_needs_clarification` tracking.** Unlike `/chat`'s extraction, this endpoint doesn't track "user declined to answer" or "answer was ambiguous" as separate states — a field is simply a value or `null`. If your product needs refusal tracking, that needs to be added.
- **No `missing_mandatory` summary in the response.** The `mandatory` flag on each field is accepted but not acted on — computing "which mandatory fields are still empty" is currently the caller's responsibility (diff `extracted_data` against `extraction_fields[].mandatory` yourself).
- **No per-call model override.** Always uses `gpt-4o-mini` at `temperature=0`. There is no `model` field in the request body.
- **Single conversation per call.** No batch mode — one `conversation_id`/transcript per request.

## Where the code lives

- `extraction/models.py` — Pydantic request/response models (the source of truth for the exact shapes above)
- `extraction/schema_builder.py` — builds the strict JSON schema sent to OpenAI's structured-output API
- `extraction/prompt_builder.py` — builds the system/user prompt (known-data block + full transcript + the conservative-update rules)
- `extraction/service.py` — the actual OpenAI call, retry-on-truncation handling, deterministic date resolution, and the code-level "don't change without an explicit correction" safety net
- `extraction/router.py` — the FastAPI route itself, request validation, and error handling
- Wired into the app in `app.py` via `app.include_router(extraction_router)`
