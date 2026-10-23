"""End-to-end RAG pipeline evaluation using Groq as the LLM judge.

Place this file at evals/eval_rag_pipeline.py and run from project root:
    python -m evals.eval_rag_pipeline

The live flow is MongoDB retriever -> answer_from_kb generator.
Contextual claim relevancy = relevant claims / all extracted claims, with up to
five factual claims assessed per retrieved document. Faithfulness and Answer
Relevancy are DeepEval metrics and also use the Groq judge.

Required environment variables:
    GROQ_API_KEY      Groq judge
    OPENAI_API_KEY    Current retriever embeddings + answer_from_kb generator
    MONGODB_URI       MongoDB Atlas retriever

Optional:
    RAG_GOLDEN_PATH=golden_dataset/faithfulness_dataset.json
    WORKSPACE_ID=eval_workspace
    KNOWLEDGE_BASE_ID=eval_kb_001
    GENERATOR_MODEL=gpt-4o-mini
    JUDGE_MODEL=openai/gpt-oss-120b
    RAG_EVAL_THRESHOLD=0.7
    RAG_EVAL_LIMIT=0
    RAG_TOP_K=5
    RAG_MIN_SCORE=0.3
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
from deepeval.metrics import AnswerRelevancyMetric, BaseMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from dummy_chat.dummy_kb_answer import answer_from_kb
from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks
from evals.harness import summarize_by_metric, print_summary

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = Path(os.getenv("RAG_GOLDEN_PATH", "golden_dataset/faithfulness_dataset.json"))
if not GOLDEN_PATH.is_absolute():
    GOLDEN_PATH = PROJECT_ROOT / GOLDEN_PATH
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")
THRESHOLD = float(os.getenv("RAG_EVAL_THRESHOLD", "0.7"))
EVAL_LIMIT = int(os.getenv("RAG_EVAL_LIMIT", "0"))
WORKSPACE_ID = os.getenv("WORKSPACE_ID", "eval_workspace")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID", "eval_kb_001")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))
CLAIMS_PER_DOCUMENT = 3
RAG_EVAL_INDEX = 2


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Golden dataset not found: {path}. Set RAG_GOLDEN_PATH to faithfulness_dataset.json."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next((payload[k] for k in ("goldens", "records", "examples", "data", "test_cases", "items")
                        if isinstance(payload.get(k), list)), None)
        if records is None and any(k in payload for k in ("query", "input", "question", "user_query")):
            records = [payload]
        if records is None:
            raise ValueError("Unsupported golden JSON structure: expected a list of records.")
    else:
        raise ValueError("Golden dataset must be a JSON list or object.")
    if not records or not all(isinstance(r, dict) for r in records):
        raise ValueError("Golden dataset is empty or contains non-object records.")
    return records


def get_query(record: dict[str, Any], number: int) -> str:
    for key in ("query", "input", "question", "user_query"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Record {number} has no query/input/question field.")


def normalize_chunks(retrieved: Any) -> list[str]:
    if not isinstance(retrieved, list):
        raise TypeError(f"Retriever should return a list, got {type(retrieved).__name__}.")
    contexts = []
    for chunk in retrieved:
        if isinstance(chunk, str) and chunk.strip():
            contexts.append(chunk.strip())
        elif isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("content") or chunk.get("page_content")
            if isinstance(text, str) and text.strip():
                contexts.append(text.strip())
    return contexts


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval model adapter that sends judge requests to Groq."""
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.client = Groq()
        self.async_client = AsyncGroq()
        self.model_name = model_name

    def load_model(self):
        return self.client

    @staticmethod
    def _schema_prompt(prompt: str, schema: type[BaseModel]) -> str:
        return (f"{prompt}\n\nReturn valid JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema())}\nNo markdown or extra text.")

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "Return only valid JSON."},
                      {"role": "user", "content": self._schema_prompt(prompt, schema)}],
            temperature=0, response_format={"type": "json_object"})
        output = response.choices[0].message.content
        if not output:
            raise ValueError("Groq judge returned an empty response.")
        return schema.model_validate_json(output)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            response = await self.async_client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content
        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "Return only valid JSON."},
                      {"role": "user", "content": self._schema_prompt(prompt, schema)}],
            temperature=0, response_format={"type": "json_object"})
        output = response.choices[0].message.content
        if not output:
            raise ValueError("Groq judge returned an empty response.")
        return schema.model_validate_json(output)

    def get_model_name(self):
        return self.model_name

    def assess_document_claims(self, query: str, document: str) -> list[dict[str, Any]]:
        """Extract up to five grounded claims from a document and classify query relevance."""
        prompt = f'''Evaluate this retrieved document for the given user query.
USER QUERY: {query}
RETRIEVED DOCUMENT: {document}

Extract up to {CLAIMS_PER_DOCUMENT} distinct, atomic factual claims explicitly supported
by the document (split compound statements if useful). For each claim, decide whether
it directly helps answer the query. Topic overlap alone is not enough. Do not invent
claims. If there are fewer than five genuine claims, return fewer rather than padding.
Return JSON only: {{"claims":[{{"claim":"...","relevant":true,"reason":"brief reason"}}]}}'''
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "Be a careful evaluator. Return valid JSON only."},
                      {"role": "user", "content": prompt}],
            temperature=0, response_format={"type": "json_object"})
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("Groq returned an empty claim-relevance response.")
        try:
            claims = json.loads(raw).get("claims")
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ValueError(f"Invalid claim-relevance JSON: {raw[:400]}") from exc
        if not isinstance(claims, list):
            raise ValueError("Groq claim-relevance response must contain a claims list.")
        cleaned = []
        for item in claims[:CLAIMS_PER_DOCUMENT]:
            if not isinstance(item, dict):
                continue
            claim, relevant = item.get("claim"), item.get("relevant")
            if isinstance(claim, str) and claim.strip() and isinstance(relevant, bool):
                cleaned.append({"claim": claim.strip(), "relevant": relevant,
                                "reason": str(item.get("reason", "")).strip()})
        return cleaned


class ContextualClaimRelevancyMetric(BaseMetric):
    """Relevant extracted claims / total extracted claims across retrieved documents."""
    def __init__(self, judge: GroqJudge, threshold: float = THRESHOLD):
        self.judge = judge
        self.threshold = threshold
        self.score: float | None = None
        self.success: bool | None = None
        self.reason: str | None = None
        self.error: str | None = None
        self.strict_mode = False
        self.async_mode = False
        self.evaluation_details: list[dict[str, Any]] = []

    @property
    def __name__(self) -> str:
        return "Contextual Claim Relevancy"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.evaluation_details = []
        contexts = test_case.retrieval_context or []
        for doc_index, document in enumerate(contexts, start=1):
            claims = self.judge.assess_document_claims(test_case.input, document)
            self.evaluation_details.append({"document_index": doc_index, "claims": claims})
        all_claims = [claim for doc in self.evaluation_details for claim in doc["claims"]]
        total = len(all_claims)
        relevant = sum(claim["relevant"] is True for claim in all_claims)
        self.score = relevant / total if total else 0.0
        self.success = self.score >= self.threshold
        self.reason = (f"{relevant}/{total} claims relevant; score={self.score:.3f}; "
                       f"documents evaluated={len(contexts)}")
        print(f"    Contextual Claim Relevancy: {relevant}/{total} = {self.score:.3f}")
        for doc in self.evaluation_details:
            print(f"      Document {doc['document_index']}:")
            for claim in doc["claims"]:
                status = "RELEVANT" if claim["relevant"] else "NOT RELEVANT"
                print(f"        [{status}] {claim['claim']}")
                if claim["reason"]:
                    print(f"          Reason: {claim['reason']}")
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)


def generate_answer(query: str, context: list[str]) -> str:
    """Use the project's actual generator with context returned by the live retriever."""
    result = answer_from_kb(
        query=query, chunks=context, model=GENERATOR_MODEL, temperature=0,
        extraction_fields=None, force_extraction_mode=False,
        conversation_history=[], is_first_message=True, include_raw=False)
    if isinstance(result, dict):
        result = result.get("answer", "")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("answer_from_kb returned no answer; check OPENAI_API_KEY and generator settings.")
    return result.strip()


def run() -> dict[str, Any]:
    records = load_records(GOLDEN_PATH)
    if RAG_EVAL_INDEX >= 0:
        if RAG_EVAL_INDEX >= len(records):
            raise IndexError(
            f"RAG_EVAL_INDEX={RAG_EVAL_INDEX} is out of range. "
            f"Dataset contains {len(records)} records."
        )
        records = [records[RAG_EVAL_INDEX]]
    elif EVAL_LIMIT > 0: 
        records = records[:EVAL_LIMIT]

    print("=" * 80)
    print("END-TO-END RAG PIPELINE EVALUATION")
    print(f"Dataset: {GOLDEN_PATH} | records: {len(records)}")
    print(f"Workspace: {WORKSPACE_ID} | KB: {KNOWLEDGE_BASE_ID} | top_k={TOP_K} | min_score={MIN_SCORE}")
    print(f"Generator: {GENERATOR_MODEL} | Groq judge: {JUDGE_MODEL}")
    print("Flow: live retriever -> live generator")

    test_cases = []
    for number, record in enumerate(records, start=1):
        query = get_query(record, number)
        print("\n" + "=" * 80 + f"\n[{number}/{len(records)}] QUERY: {query}\n" + "=" * 80)
        retrieved = retrieve_kb_chunks(
            workspace_id=WORKSPACE_ID, query=query,
            knowledge_base_ids=[KNOWLEDGE_BASE_ID], top_k=TOP_K,
            min_score=MIN_SCORE, include_raw=False, fallback_text=False)
        contexts = normalize_chunks(retrieved)
        print(f"Retrieved documents: {len(contexts)}")
        for rank, context in enumerate(contexts, start=1):
            chunk = retrieved[rank - 1] if rank - 1 < len(retrieved) else {}
            if isinstance(chunk, dict):
                print(f"  Rank {rank}: score={chunk.get('score')} | source={chunk.get('source_name')} | chunk={chunk.get('chunk_index')}")
            print(f"    {context[:300]}")
        answer = generate_answer(query, contexts)
        print(f"\nGENERATED ANSWER:\n{answer}")
        test_cases.append(LLMTestCase(input=query, actual_output=answer, retrieval_context=contexts))

    judge = GroqJudge()
    metrics = [
        ContextualClaimRelevancyMetric(judge=judge, threshold=THRESHOLD),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "mongodb_vector_search",
            "embedding_model": "text-embedding-3-small",
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "workspace_id": WORKSPACE_ID,
            "knowledge_base_id": KNOWLEDGE_BASE_ID,
            "golden_set": str(GOLDEN_PATH),
            "contextual_relevancy_strategy": "relevant claims / total claims; up to five claims per document",
        },
    )
    summary = summarize_by_metric(result)
    print_summary("rag_pipeline", summary)
    return summary


def run_local() -> dict[str, Any]:
    return run()


if __name__ == "__main__":
    run_local()
