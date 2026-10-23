from pydantic import BaseModel

from evals.eval_retriever import GroqJudge


class TestOutput(BaseModel):
    verdict: bool
    score: float


def main():

    judge = GroqJudge()

    prompt = """
You are evaluating a retrieval result.

Query:
What is the capital of France?

Retrieved context:
Paris is the capital and largest city of France which is located in America.

Return a JSON object with:
- verdict: true if the context supports the query, otherwise false
- score: a number between 0 and 1 representing how strongly the context supports the query
"""

    result = judge.generate(
        prompt,
        schema=TestOutput,
    )

    print("\nGroq judge test successful!")
    print("Result:")
    print(result)

    print("\nType:")
    print(type(result))


if __name__ == "__main__":
    main()