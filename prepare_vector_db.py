from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

import getpass
import os

# Example text
text = ''' Chào mừng các bạn đến với Mì AI - nơi chúng ta khám phá thế giới của trí tuệ nhân tạo! Trong video hôm nay, chúng ta sẽ đào sâu vào ứng dụng mới và độc đáo của Langchain - một công nghệ tiên tiến đưa trí tuệ nhân tạo lên một tầm cao mới.

🤖 Langchain không chỉ là một hệ thống thông thường, mà còn là một bước đột phá trong xây dựng mô hình hỏi đáp nội dung văn bản. Chúng ta sẽ tìm hiểu về Retrieval Augmented Generation - một kỹ thuật mạnh mẽ đằng sau sự thành công của Langchain.

🌐 Trong video này, chúng ta sẽ:
1️⃣ Hiểu rõ hơn về Retrieval Augmented Generation là gì và làm thế nào nó giúp cải thiện khả năng hỏi đáp của mô hình.
2️⃣ Khám phá cách Langchain tích hợp Retrieval Augmented Generation để xây dựng mô hình mạnh mẽ, linh hoạt và hiệu quả.
3️⃣ Đồng hành cùng Mì AI trong việc thử nghiệm và đánh giá hiệu suất của Langchain khi đối mặt với các thách thức hỏi đáp văn bản.

🔍 Cùng nhau, chúng ta sẽ khám phá những ứng dụng thực tế và tiềm năng mà Retrieval Augmented Generation mang lại trong lĩnh vực trí tuệ nhân tạo. Đừng quên nhấn đăng ký, like và bấm chuông để không bỏ lỡ những video thú vị tiếp theo về thế giới của trí tuệ nhân tạo và công nghệ!"

'''
vector_db_path = "./vectorStore"


# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the text
chunks = text_splitter.split_text(text)

#Embedding textstexts
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004",google_api_key= GEMINI_API_KEY)
vector = embeddings.embed_query("Hello world!")
print (vector)


# #Save local using faiss
db = FAISS.from_texts(texts = chunks, embedding= embeddings)
db.save_local(vector_db_path)

# # Print the chunks
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk}")