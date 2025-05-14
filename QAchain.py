from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google import genai
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



load_dotenv()

vector_db_path = "./vectorStore"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#create prompt template
def create_prompt (template):
    prompt = PromptTemplate (template= template, input_variables=["context", "question"])
    return prompt

#create simple prompt
def create_qa_chain (prompt, llm, db):
    llm_chain = RetrievalQA.from_llm(
        llm = llm,
        retriever = db.as_retriever(search_kwargs = {"k": 3}),
        prompt = prompt
    )
    return llm_chain

def read_vector_db():
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key = GEMINI_API_KEY)
    db = FAISS.load_local(vector_db_path, embeddings = embedding_model, allow_dangerous_deserialization = True)
    return db

template = """
"system": 
Bạn là một trợ lý AI đắc lực
Hãy trả lời người dùng một cách chính xác
{context}
"user": 
{question}
"""
if __name__ == "__main__":
    db = read_vector_db()
    llm = ChatGoogleGenerativeAI(model = "models/gemini-2.0-flash", google_api_key = GEMINI_API_KEY)
    prompt = create_prompt(template)
    # question = input("Enter your question: ")
    question = "hãy phân tích giúp tôi các công nghệ mà Nami Tech sử dụng và tiềm năng phát triển của nó"

    llm_chain = create_qa_chain(prompt, llm, db)
    response = llm_chain.invoke({"query": question})
    print (response)

  #  """hãy phân tích giúp tôi các công nghệ mà Nami Tech sử dụng và tiềm năng phát triển của nó"""