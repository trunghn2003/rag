import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# Khởi tạo FastAPI
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost",
        "http://localhost:80",
        "http://api.football1.io.vn",
        "http://157.66.47.51:9000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Xóa và tạo lại bộ sưu tập để đảm bảo tương thích
try:
    chroma_client.delete_collection(name="football_knowledge")
    logger.info("Deleted existing collection 'football_knowledge'")
except Exception:
    pass
collection = chroma_client.get_or_create_collection(
    name="football_knowledge",
    metadata={"hnsw:space": "cosine"}
)
logger.info("Initialized ChromaDB collection 'football_knowledge'")

# Khởi tạo Gemini
key = "AIzaSyCyaBub5taZ9m7ybGCLrH0jv-X-0x4Lv-U"

if not key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-1.5-flash')  # Mô hình hợp lệ
embedding_model = "model/text-embedding-004"  # Sửa từ models/embedding-004 thành text-embedding-004

# Định nghĩa mô hình dữ liệu
class Document(BaseModel):
    id: str
    content: str
    metadata: dict

class Query(BaseModel):
    query: str
    n_results: int = 3
    type: Optional[str] = None

# Endpoint lập chỉ mục tài liệu
@app.post("/index")
async def index_document(document: Document):
    try:
        logger.info(f"Indexing document ID: {document.id}")
        # Tạo embedding cho tài liệu
        embedding = genai.embed_content(
            model=embedding_model,
            content=document.content,
            task_type="retrieval_document",
        )["embedding"]
        logger.info(f"Embedding generated: {len(embedding)} dimensions")

        # Thêm vào ChromaDB
        collection.add(
            documents=[document.content],
            embeddings=[embedding],
            ids=[document.id],
            metadatas=[document.metadata]
        )
        count = collection.count()
        logger.info(f"Total documents in collection: {count}")
        return {"status": "success", "message": f"Document indexed successfully, total documents: {count}"}
    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint truy vấn tài liệu
@app.post("/query")
async def query_documents(query: Query):
    try:
        logger.info(f"Processing query: {query.query}")
        # Tạo embedding cho truy vấn
        query_embedding = genai.embed_content(
            model=embedding_model,
            content=query.query,
            task_type="retrieval_query",
        )["embedding"]
        logger.info(f"Query embedding generated: {len(query_embedding)} dimensions")

        # Tìm tài liệu liên quan
        where_filter = {"type": query.type} if query.type else None
        logger.info(f"Applying filter: {where_filter}")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results,
            where=where_filter
        )
        logger.info(f"Query results: {results['documents']}")

        # Kiểm tra kết quả
        if not results["documents"] or not results["documents"][0]:
            logger.warning("No relevant documents found")
            return {
                "answer": "Không tìm thấy tài liệu liên quan. Vui lòng kiểm tra dữ liệu đã lập chỉ mục hoặc thử truy vấn khác.",
                "documents": [],
                "distances": []
            }

        # Tạo prompt với ngữ cảnh
        context = "\n\n".join(results["documents"][0])
        logger.info(f"Context for Gemini: {context}")
        prompt = f"""Bạn là một trợ lý bóng đá am hiểu, cung cấp thông tin dựa trên ngữ cảnh sau.
        Trả lời bằng giọng điệu thân thiện và dễ hiểu, giải thích các khái niệm phức tạp cho người dùng phổ thông.

        Ngữ cảnh:
        {context}

        Câu hỏi: {query.query}

        Trả lời:"""

        # Tạo phản hồi
        response = model.generate_content(prompt)
        logger.info(f"Gemini response: {response.text}")

        return {
            "answer": response.text,
            "documents": results["documents"][0],
            "distances": results["distances"][0]
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint lập chỉ mục hàng loạt
@app.post("/bulk_index")
async def bulk_index(documents: List[Document]):
    try:
        logger.info(f"Indexing {len(documents)} documents")
        contents = []
        embeddings = []
        ids = []
        metadatas = []

        for doc in documents:
            embedding = genai.embed_content(
                model=embedding_model,
                content=doc.content,
                task_type="retrieval_document",
            )["embedding"]
            contents.append(doc.content)
            embeddings.append(embedding)
            ids.append(doc.id)
            metadatas.append(doc.metadata)

        collection.add(
            documents=contents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        count = collection.count()
        logger.info(f"Total documents in collection: {count}")
        return {"status": "success", "message": f"{len(documents)} documents indexed successfully, total documents: {count}"}
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint kiểm tra tài liệu
@app.get("/documents")
async def get_documents():
    try:
        results = collection.get()
        logger.info(f"Retrieved {len(results['ids'])} documents from collection")
        return {
            "ids": results["ids"],
            "documents": results["documents"],
            "metadatas": results["metadatas"]
        }
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))