from pathlib import Path
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import DeepEvalBaseLLM

from groq import Groq, AsyncGroq
from pydantic import BaseModel

from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)


load_dotenv()


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

GOLDEN_PATH = (
    BASE_DIR
    / "golden_dataset"
    / "retriever_goldens.json"
)

JUDGE_MODEL = "openai/gpt-oss-120b"

THRESHOLD = 0.6

WORKSPACE_ID = "eval_workspace"
KNOWLEDGE_BASE_ID = "eval_kb_001"

TOP_K = 5
MIN_SCORE = 0.3


# ============================================================
# GROQ JUDGE
# ============================================================

class GroqJudge(DeepEvalBaseLLM):

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
        """
        Generate a response using Groq.

        When DeepEval supplies a schema, explicitly include that
        schema in the prompt and request JSON output.
        """

        if schema is None:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

            return response.choices[0].message.content

        # ----------------------------------------------------
        # DeepEval supplied a Pydantic schema.
        # Explicitly tell the judge what JSON structure is
        # required.
        # ----------------------------------------------------

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
                {
                    "role": "user",
                    "content": schema_prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

        output = response.choices[0].message.content

        return schema.model_validate_json(output)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        """
        Async generation used by DeepEval because the metrics
        run with async_mode=True.
        """

        if schema is None:

            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )

            return response.choices[0].message.content

        # ----------------------------------------------------
        # DeepEval supplied a Pydantic schema.
        # Explicitly include the schema in the prompt.
        # ----------------------------------------------------

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
                {
                    "role": "user",
                    "content": schema_prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

        output = response.choices[0].message.content

        return schema.model_validate_json(output)

    def get_model_name(self):
        return self.model_name


# ============================================================
# RUN RETRIEVER EVALUATION
# ============================================================

def run():

    # --------------------------------------------------------
    # 1. LOAD GOLDEN DATASET
    # --------------------------------------------------------

    goldens = load_goldens(GOLDEN_PATH)

    goldens = goldens[:10]  # Limit to first 10 for testing

    test_cases = []

    # --------------------------------------------------------
    # 2. RUN THE ACTUAL MONGODB RETRIEVER
    # --------------------------------------------------------

    for g in goldens:

        query = g["query"]

        retrieved = retrieve_kb_chunks(
            workspace_id=WORKSPACE_ID,
            query=query,
            knowledge_base_ids=[KNOWLEDGE_BASE_ID],

            # Retrieve exactly the number of chunks
            # that we want to evaluate.
            top_k=TOP_K,

            min_score=MIN_SCORE,

            # We only need the final retrieved chunks.
            include_raw=False,

            # Disable regex/text fallback.
            fallback_text=False,
        )

        # ----------------------------------------------------
        # Convert retrieved chunks into DeepEval context
        # ----------------------------------------------------

        retrieval_context = [
            chunk["text"]
            for chunk in retrieved
        ]

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print(f"ID: {g['id']}")
        print(f"QUERY: {query}")
        print("=" * 80)

        print(f"Retrieved chunks: {len(retrieved)}")

        for rank, chunk in enumerate(retrieved, start=1):

            print(f"\n--- Rank {rank} ---")
            print(f"Public ID : {chunk.get('public_id')}")
            print(f"Source    : {chunk.get('source_name')}")
            print(f"Chunk     : {chunk.get('chunk_index')}")
            print(f"Score     : {chunk.get('score')}")
            print(
                f"Text      : "
                f"{chunk.get('text', '')[:300]}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # actual_output is intentionally empty.
        #
        # We are evaluating:
        #
        # expected_output
        #        vs
        # retrieval_context
        #
        # No answer generation is being evaluated.
        # ----------------------------------------------------

        test_cases.append(
            LLMTestCase(
                input=query,

                # Human-authored ideal answer
                expected_output=g["ideal_answer"],

                # Chunks returned by YOUR retriever
                retrieval_context=retrieval_context,

                # Generator is intentionally not evaluated.
                actual_output="",
            )
        )

    # --------------------------------------------------------
    # 3. CREATE THE GROQ JUDGE
    # --------------------------------------------------------

    judge = GroqJudge()

    # --------------------------------------------------------
    # 4. DEFINE DEEPEVAL RETRIEVAL METRICS
    # --------------------------------------------------------

    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,

            # IMPORTANT:
            # Pass the actual GroqJudge object.
            model=judge,

            include_reason=True,
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,

            # IMPORTANT:
            # Pass the actual GroqJudge object.
            model=judge,

            include_reason=True,
        ),

    ]

    # --------------------------------------------------------
    # 5. RUN DEEPEVAL
    # --------------------------------------------------------

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,

        hyperparameters={

            # Actual retriever being evaluated
            "retriever": "mongodb_vector_search",

            # Actual embedding model used by retriever
            "embedding_model": "text-embedding-3-small",

            "top_k": TOP_K,

            "min_score": MIN_SCORE,

            "workspace_id": WORKSPACE_ID,

            "knowledge_base_id": KNOWLEDGE_BASE_ID,

            # Model used by the evaluation judge
            "judge_model": JUDGE_MODEL,

            "golden_set": str(GOLDEN_PATH),

            # Explicitly record that these fallbacks
            # were not used.
            "local_vector_fallback": False,
            "text_fallback": False,
        },
    )

    return summarize_by_metric(result)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print_summary(
        "retriever",
        run(),
    )