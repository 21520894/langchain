LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-timely-osmosis-38"
API_KEY="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"

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)


# llm = OpenAI(model_name="o4-mini", openai_api_key=API_KEY)
# print(llm("Tell me a joke about data scientist"))

