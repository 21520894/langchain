from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#create prompt template
def create_prompt (template):
    prompt = PromptTemplate (template= template, input_variables=["question"])
    return prompt

#create simple prompt
def create_simple_chain (prompt):
    client = genai.Client(api_key = GEMINI_API_KEY)
    response = client.models.generate_content(
    model='gemini-2.0-flash-001', contents=prompt)
    print(response.text)
    return response

template = """
Bạn là một trợ lý AI đắc lực
Hãy trả lời người dùng một cách chính xác
{question}
"""
if __name__ == "__main__":
    prompt = create_prompt(template)
    question = input("Enter your question: ")
    filled_prompt = prompt.format(question=question)
    print(create_simple_chain(filled_prompt))


