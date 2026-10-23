"""
evals/eval_generator.py
=======================
Evaluate the answer generator in isolation using known-good golden context.

Run from the project root:
    python -m evals.eval_generator

Optional environment variables:
    GENERATOR_GOLDEN_PATH  Path to the generator golden JSON file
    GENERATOR_MODEL        OpenAI model used by answer_from_kb (default: gpt-4o-mini)
    JUDGE_MODEL            Groq judge model (default: openai/gpt-oss-120b)
    GENERATOR_EVAL_LIMIT   Evaluate only the first N records (default: 0 = all)
    GENERATOR_EVAL_THRESHOLD  Metric pass threshold (default: 0.7)

This script does not call the retriever or an embedding model. Each generation
call receives the context stored directly in the golden dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import AsyncGroq, Groq
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from evals.harness import summarize_by_metric, print_summary
from dummy_chat.dummy_kb_answer import answer_from_kb

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_CANDIDATES = [
    PROJECT_ROOT / "golden_dataset" / "faithfulness_dataset.json",
]

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
THRESHOLD = float(os.getenv("GENERATOR_EVAL_THRESHOLD", "0.7"))
EVAL_LIMIT = int(os.getenv("GENERATOR_EVAL_LIMIT", "2"))


def _resolve_dataset_path() -> Path:
    """Resolve dataset from env var or common project locations."""
    configured = os.getenv("GENERATOR_GOLDEN_PATH")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"GENERATOR_GOLDEN_PATH points to a file that does not exist: {path}"
            )
        return path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.is_file():
            return candidate

    candidates = "\n".join(f"  - {p}" for p in DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the generator golden dataset. Set GENERATOR_GOLDEN_PATH "
        "to its path. Checked:\n" + candidates
    )


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load common JSON list/object dataset layouts."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("goldens", "records", "examples", "data", "test_cases", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            # A single record can also be stored as a JSON object.
            if any(key in payload for key in ("query", "input", "question")):
                records = [payload]
            else:
                raise ValueError(
                    f"Unsupported dataset structure in {path}. Expected a JSON list "
                    "or an object containing a list named goldens/records/examples/data."
                )
    else:
        raise ValueError(f"Expected a JSON list or object in {path}.")

    if not records:
        raise ValueError(f"The generator golden dataset is empty: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every generator golden record must be a JSON object.")
    return records


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _normalise_context(value: Any, record_number: int) -> list[str]:
    """Convert the golden context to the list[str] expected by the generator."""
    if isinstance(value, str):
        context = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        context = []
        for item in value:
            if isinstance(item, str) and item.strip():
                context.append(item.strip())
            elif isinstance(item, dict):
                # Support chunk objects if a dataset stores chunks with a text key.
                chunk_text = _first_present(item, ("text", "content", "page_content", "chunk"))
                if isinstance(chunk_text, str) and chunk_text.strip():
                    context.append(chunk_text.strip())
                else:
                    raise ValueError(
                        f"Dataset record {record_number}: context chunk object has no text/content field."
                    )
            else:
                raise ValueError(
                    f"Dataset record {record_number}: context must contain strings or chunk objects."
                )
    else:
        raise ValueError(
            f"Dataset record {record_number}: expected ideal_context/context to be a string or list."
        )

    if not context:
        raise ValueError(
            f"Dataset record {record_number} has no golden context. Generator isolation requires context."
        )
    return context


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge implemented with Groq, matching the project's retriever evaluator."""

    def __init__(self):
        self.client = Groq()
        self.async_client = AsyncGroq()
        self.model_name = JUDGE_MODEL

    def load_model(self):
        return self.client

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        """Generate a response using Groq, requesting JSON for schemas."""
        if schema is None:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content

        schema_prompt = f"""
{prompt}

You MUST return valid JSON.

The JSON must follow this exact schema:

{schema.model_json_schema()}

Do not return markdown.
Do not return explanations outside the JSON.
Return only the JSON object.
"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching the "
                        "requested schema."
                    ),
                },
                {"role": "user", "content": schema_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        if not output:
            raise ValueError(
                f"Groq judge returned an empty response for {self.model_name}."
            )
        return schema.model_validate_json(output)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        """Async generation used by DeepEval metrics running in async mode."""
        if schema is None:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content

        schema_prompt = f"""
{prompt}

You MUST return valid JSON.

The JSON must follow this exact schema:

{schema.model_json_schema()}

Do not return markdown.
Do not return explanations outside the JSON.
Return only the JSON object.
"""
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON matching the "
                        "requested schema."
                    ),
                },
                {"role": "user", "content": schema_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        if not output:
            raise ValueError(
                f"Groq judge returned an empty response for {self.model_name}."
            )
        return schema.model_validate_json(output)

    def get_model_name(self):
        return self.model_name


def _generate_answer(query: str, context: list[str]) -> str:
    """Call the actual answer_from_kb generator, with no retrieval or extraction."""
    result = answer_from_kb(
        query=query,
        chunks=context,
        model=GENERATOR_MODEL,
        temperature=0,
        extraction_fields=None,
        force_extraction_mode=False,
        conversation_history=[],
        is_first_message=True,
        include_raw=False,
    )

    # In non-extraction mode answer_from_kb normally returns a string. Keep a
    # defensive branch in case a project version returns an answer dictionary.
    if isinstance(result, dict):
        result = result.get("answer", "")
    if not isinstance(result, str) or not result.strip():
        raise ValueError(
            "answer_from_kb returned no text. Check the generator return value and OPENAI_API_KEY."
        )
    return result.strip()


def run() -> dict[str, Any]:
    dataset_path = _resolve_dataset_path()
    records = _load_dataset(dataset_path)
    if EVAL_LIMIT > 0:
        records = records[:EVAL_LIMIT]

    print(f"Generator evaluation dataset: {dataset_path}")
    print(f"Generator model: {GENERATOR_MODEL}")
    print(f"Judge model: Groq / {JUDGE_MODEL}")
    print(f"Golden records to evaluate: {len(records)}")
    print("Retriever calls: 0 | Embedding calls: 0")

    test_cases: list[LLMTestCase] = []
    for index, record in enumerate(records, start=1):
        query = _first_present(record, ("query", "input", "question", "user_query"))
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Dataset record {index} has no query/input/question string.")

        raw_context = _first_present(
            record,
            ("ideal_context", "golden_context", "context", "retrieval_context", "contexts"),
        )
        if raw_context is None:
            raise ValueError(
                f"Dataset record {index} has no context field. Expected ideal_context, "
                "golden_context, context, retrieval_context, or contexts."
            )
        context = _normalise_context(raw_context, index)

        # Crucial isolation rule: pass the dataset's golden context directly.
        # Do not call retrieve_kb_chunks or any embedding model here.
        answer = _generate_answer(query.strip(), context)
        test_cases.append(
            LLMTestCase(
                input=query.strip(),
                actual_output=answer,
                retrieval_context=context,
            )
        )
        print(f"[{index}/{len(records)}] Generated answer for: {query.strip()}")

    judge = GroqJudge()
    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]

    result = evaluate(test_cases=test_cases, metrics=metrics)
    summary = summarize_by_metric(result)
    print_summary("generator", summary)
    return summary


if __name__ == "__main__":
    run()
