# Langchain
- Langchain là một framework để phát triển ứng dụng liên quan đến LLM
- Lợi ích
    - Kết nối LLM đến với nguồi ngữ cảnh (có thể hiểu là nối LLM đến prompt, example, ... để truy xuất nhanh và tiện thao tác)
    - Xây dựng ngữ cảnh cho LLM
# Cung cấp
## Models I/O
- Cung cấp cho người dùng thiết lập sẵn các prompt để truy xuất nhanh
- Các kết nối đến model
- Lựa chọn ví dụ cho prompt -> model học
- Output Parser
## Retrieval
- Retriever: Cho phép người dùng truy vấn trong dữ liệu mà người dùng cấp
- Document Loader: Cho phép người dùng tải tài liệu lên bằng thư viện mà langchain cung cấp sẵn
- Vector store: Vector đặc trưng của văn bản mà từ đó ta có thể query được thông qua câu hỏi
- Text spliter: Chia nhỏ văn bản ra (phù hợp với window của LLM )
- Embedding model: Là model dùng để mã hóa cái văn bản người dùng cấp thành vector store
## Agent tooling:
- Tools:
- Toollist:

# Hoạt động:
Hoạt động như tên, hoạt động bằng cách cho dữ liệu đi qua 1 chuỗi các hành động để xử lý -> đưa ra output mong muốn
Mô hình đơn giản nhất:
Question -> LLM -> Prompt -> Data source -> LLM -> answer

![Minh họa flow của langchain](https://media.licdn.com/dms/image/v2/D5612AQFzLX3ccBu1cA/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1719389475388?e=2147483647&v=beta&t=BDUkc4vIf-Vbg1k3V92Sv07lWYG_6hxBySln5OeB9Fc)

Mô tả: gồm đầu vào bằng các tài liệu pdf mà người dùng muốn được truy xuất vào -> sau đó chia ra thành các chunk để giảm kích thước dữ liệu và tăng tốc độ truy vấn -> sau đó embedding dữ liệu đó thành các embedding vector -> Lưu trữ trong database chuyên dụng ->  Dữ liệu này sẽ nằm trong một hệ thống hỗ trợ người dùng thực hiện truy vấn, cho phép LLM truy cập thêm các thông tin ngoài các thông tin được train trong đó.

# Retrieval Augumented Generation (RAG)
Là một hệ thống hỏi đáp, cho phép truy vấn vào trong 1 vector database.
+ Cho phép người dùng truy xuất những thông tin mà model chưa được pretrain.
+ Cho đỡ tốn thời gian và chi phí hơn đi train thêm thông tin, cũng như có thể xóa bỏ đi bất cứ khi nào.
# VectorDB:
- Giới hạn của LLM là có context window -> không thể đọc hết được toàn bộ data.
- Chia nhỏ văn bản ra -> embedding -> vectorDB. 
- Khi cần truy vấn thì so sánh và lấy ra n vector nào tương đồng với vector truy vấn nhất.
- Đưa qua LLM để đưa ra response mượt nhất.
# Flow:
![Minh họa flow của langchain](https://github.com/21520894/langchain/blob/master/image.png)
## Flow lưu văn bản cần truy xuất:
- Data preparing: xử lý các lỗi căn bản
- Create Vector DB: sử dụng embedding của 1 bên model nào đó (vì các model đó sẽ embedding cho các từ gần giống nhau về thành các vector gần bằng nhau), và spliter để phù hợp với setting DB cũng như tiện truy vấn sau này (Faiss, ChromaDB).
## Flow truy vấn văn bản:
- User query: Người dùng đưa ra câu truy vấn dành cho văn bản đó
- LLM To understand query: Đưa vào LLM để hiểu được câu hỏi (LLM sẽ dựa trên ý nghĩa câu hỏi và chức năng mà các hàm nó đang nắm giữ để xem xét xem có nên sử dụng truy vấn trong vector DB hay không)
- Query DB: Truy vấn trong DB (bằng cách embedding câu query và tìm ra các vector embedding gần giống nhất)
- LLM to generate answer: Đưa thông tin truy vấn được trong DB vào lại trong LLM để LLM dựa vào đó đưa ra câu trả lời cho câu query người dùng
- User response: đưa câu trả lời cho người dùng.

Note: Bài viết này dựa trên nguồn từ Mì AI

# How to run this repo

