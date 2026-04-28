import os
from typing import Literal

import pydash
from openai import AzureOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential


class LLMPassFailResult(BaseModel):
    result: Literal["pass", "fail"]
    reason: str


prompt = """
You are a unit testing bot. You will output two lines of text, nothing else. The first line will be 'pass' or 'fail'. The second line will be the reason for the failure.

Example:

Query: The earth is flat
Response:
fail
The earth is round
"""


def get_open_ai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
        azure_endpoint="https://sfc-apim-sweden.azure-api.net",
    )


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20))
def completions_with_backoff(**kwargs):
    openai_client = get_open_ai_client()
    return openai_client.chat.completions.create(**kwargs)


def assert_gpt_4(query: str) -> LLMPassFailResult:
    completion = completions_with_backoff(
        model="sfc-cortex-analyst-dev",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    response = pydash.get(completion, "choices[0].message.content", "{}")
    lines = response.split("\n")
    return LLMPassFailResult(
        result="pass" if "pass" in lines[0] else "fail", reason=lines[1]
    )
