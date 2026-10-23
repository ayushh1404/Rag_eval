"""Application-level RAG evaluation using live retrieval/generation and Groq G-Eval judges.

Place this file at ``evals/eval_application.py`` and run from the project root:
    python -m evals.eval_application

The pipeline under test is the project's real retriever -> generator. Groq is used
only for evaluation; the current retriever and answer_from_kb generator still need
their own configured credentials (e.g. OPENAI_API_KEY and MongoDB settings).
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
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from dummy_chat.dummy_kb_answer import answer_from_kb
from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks
from evals.harness import summarize_by_metric, print_summary

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = Path(os.getenv("APP_GOLDEN_PATH", "golden_dataset/correctness_golden.json"))
if not GOLDEN_PATH.is_absolute():
    GOLDEN_PATH = PROJECT_ROOT / GOLDEN_PATH

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
THRESHOLD = float(os.getenv("APP_EVAL_THRESHOLD", "0.7"))
EVAL_LIMIT = int(os.getenv("APP_EVAL_LIMIT", "0"))
WORKSPACE_ID = os.getenv("WORKSPACE_ID", "eval_workspace")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "eval_kb_001")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))


def load_goldens(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}. Put correctness_golden.json in golden_dataset/ "
            "or set APP_GOLDEN_PATH."
        )
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or not payload:
        raise ValueError("The application golden dataset must be a non-empty JSON list.")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("Every golden dataset record must be a JSON object.")
    return payload


def normalize_chunks(retrieved: Any) -> list[str]:
    """Convert the project's retrieved chunk objects into the generator's list[str] input."""
    if not isinstance(retrieved, list):
        raise TypeError(f"retrieve_kb_chunks returned {type(retrieved).__name__}; expected a list.")
    contexts: list[str] = []
    for item in retrieved:
        if isinstance(item, str) and item.strip():
            contexts.append(item.strip())
        elif isinstance(item, dict):
            content = item.get("text") or item.get("content") or item.get("page_content")
            if isinstance(content, str) and content.strip():
                contexts.append(content.strip())
    return contexts


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval model adapter so G-Eval judge calls are sent through Groq."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        self.client = Groq()  # reads GROQ_API_KEY from the environment
        self.async_client = AsyncGroq()
        self.model_name = model_name

    def load_model(self):
        return self.client

    @staticmethod
    def _schema_prompt(prompt: str, schema: type[BaseModel]) -> str:
        return (
            f"{prompt}\n\nReturn valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}\nNo markdown or extra text."
        )

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": self._schema_prompt(prompt, schema)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        if not output:
            raise ValueError("Groq judge returned an empty response.")
        return schema.model_validate_json(output)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": self._schema_prompt(prompt, schema)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        output = response.choices[0].message.content
        if not output:
            raise ValueError("Groq judge returned an empty response.")
        return schema.model_validate_json(output)

    def get_model_name(self) -> str:
        return self.model_name


def generate_answer(question: str, context: list[str]) -> str:
    """Call the same generator function used by the application's RAG pipeline."""
    result = answer_from_kb(
        query=question,
        chunks=context,
        model=GENERATOR_MODEL,
        temperature=0,
        extraction_fields=None,
        force_extraction_mode=False,
        conversation_history=[],
        is_first_message=True,
        include_raw=False,
    )
    if isinstance(result, dict):
        result = result.get("answer", "")
    if not isinstance(result, str) or not result.strip():
        raise ValueError(
            "answer_from_kb returned no text. Check OPENAI_API_KEY and the generator's return format."
        )
    return result.strip()


def build_metrics(judge: GroqJudge) -> list[GEval]:
    """Three separate G-Eval metrics with explicit, non-overlapping scoring rules."""
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Read the question, actual answer, and expected answer carefully.",
            "Judge factual correctness of claims the actual answer explicitly makes, using the expected answer as the reference for this question.",
            "Mark a claim wrong only when it is contradicted by the expected answer or is otherwise clearly factually incorrect.",
            "Do not penalize an answer merely because it is shorter or omits a reference point; omissions belong to Completeness.",
            "Do not penalize additional information if it is relevant and correct. Penalize unsupported certainty or clear factual contradictions.",
            "Score from 0 to 10 according to the rubric, then return the normalized metric score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="All or nearly all stated factual claims are correct; no meaningful contradiction."),
            Rubric(score_range=(7, 8), expected_outcome="Mostly correct, with at most a small imprecision that does not change the main meaning."),
            Rubric(score_range=(5, 6), expected_outcome="Mixed accuracy; one important claim is questionable or inaccurate."),
            Rubric(score_range=(0, 4), expected_outcome="Contains major factual errors, clear contradictions, or substantially misleading claims."),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
        async_mode=False,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the essential concepts and key points needed to answer the question, using the expected answer as the reference checklist.",
            "Check which key points are fully addressed, partially addressed, or missing from the actual answer.",
            "Score coverage of key points only; do not lower this score because a covered point is factually wrong, because Correctness evaluates accuracy separately.",
            "Do not penalize relevant, correct extra information. Do not require exact wording or identical sentence structure.",
            "Give credit for concise answers when they cover the important points; length alone is not completeness.",
            "Score from 0 to 10 according to the rubric, then return the normalized metric score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Covers essentially all important points needed to answer the question."),
            Rubric(score_range=(7, 8), expected_outcome="Covers the main points but omits or under-explains one minor point."),
            Rubric(score_range=(5, 6), expected_outcome="Covers some important points but misses at least one substantial point."),
            Rubric(score_range=(0, 4), expected_outcome="Misses most of the key information or answers only a small part of the question."),
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
        async_mode=False,
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge the actual answer's teaching style and tone only; do not judge factual correctness or coverage.",
            "Reward clear, intuitive, conversational explanations that explain the idea in plain language before relying on jargon.",
            "Reward technical terms that are briefly explained and a natural flow that is easy for a learner to follow.",
            "A relevant analogy or example is a bonus for abstract ideas, but do not penalize an otherwise clear answer for not including one.",
            "Penalize stiff or robotic language, unexplained jargon, confusing phrasing, and bare lists that provide no explanation when explanation is needed.",
            "Do not reward or penalize based on answer length, correctness, completeness, or similarity to the expected answer.",
            "Score from 0 to 10 according to the rubric, then return the normalized metric score.",
        ],
        rubric=[
            Rubric(score_range=(9, 10), expected_outcome="Very clear, intuitive, conversational teaching style with smooth explanation."),
            Rubric(score_range=(7, 8), expected_outcome="Clear and natural explanation; easy to follow even without an analogy."),
            Rubric(score_range=(4, 6), expected_outcome="Understandable but somewhat flat, formal, choppy, or list-heavy."),
            Rubric(score_range=(0, 3), expected_outcome="Robotic, confusing, jargon-heavy, or mostly an unexplained list."),
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=THRESHOLD,
        model=judge,
        strict_mode=False,
        async_mode=False,
    )
    return [correctness, completeness, style]


def run() -> dict[str, Any]:
    records = load_goldens(GOLDEN_PATH)
    if EVAL_LIMIT > 0:
        records = records[:EVAL_LIMIT]

    print("=" * 80)
    print("APPLICATION-LEVEL G-EVAL")
    print(f"Dataset: {GOLDEN_PATH} | records: {len(records)}")
    print(f"Workspace: {WORKSPACE_ID} | KB: {KNOWLEDGE_BASE_ID} | top_k={TOP_K} | min_score={MIN_SCORE}")
    print(f"Generator: {GENERATOR_MODEL} | Groq judge: {JUDGE_MODEL} | threshold: {THRESHOLD}")
    print("Flow under test: live retriever -> live generator -> G-Eval (Correctness, Completeness, Style)")

    test_cases: list[LLMTestCase] = []
    for index, record in enumerate(records, start=1):
        question = record.get("question")
        expected = record.get("ideal_answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Golden record {index} must contain a non-empty 'question'.")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Golden record {index} must contain a non-empty 'ideal_answer'.")
        question = question.strip()
        expected = expected.strip()

        print("\n" + "=" * 80)
        print(f"[{index}/{len(records)}] {record.get('id', f'case-{index}')}: {question}")
        retrieved = retrieve_kb_chunks(
            workspace_id=WORKSPACE_ID,
            query=question,
            knowledge_base_ids=[KNOWLEDGE_BASE_ID],
            top_k=TOP_K,
            min_score=MIN_SCORE,
            include_raw=False,
            fallback_text=False,
        )
        contexts = normalize_chunks(retrieved)
        print(f"Retrieved chunks: {len(contexts)}")
        if not contexts:
            print("WARNING: retriever returned no chunks; generator will receive an empty context.")
        answer = generate_answer(question, contexts)
        print(f"GENERATED ANSWER:\n{answer}")

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=expected,
                retrieval_context=contexts,
            )
        )

    judge = GroqJudge(JUDGE_MODEL)
    metrics = build_metrics(judge)
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "retrieve_kb_chunks",
            "generator": "answer_from_kb",
            "generator_model": GENERATOR_MODEL,
            "judge_provider": "Groq",
            "judge_model": JUDGE_MODEL,
            "workspace_id": WORKSPACE_ID,
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "golden_dataset": str(GOLDEN_PATH),
        },
    )
    summary = summarize_by_metric(result)
    print_summary("application", summary)
    return summary


if __name__ == "__main__":
    run()
