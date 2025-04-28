from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://localhost",
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Khởi tạo ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="football_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Khởi tạo Gemini
key = "AIzaSyCyaBub5taZ9m7ybGCLrH0jv-X-0x4Lv-U"
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-2.0-flash')
embedding_model = "models/embedding-001"

class Document(BaseModel):
    id: str
    content: str
    metadata: dict

class Query(BaseModel):
    query: str
    n_results: int = 3
    type: Optional[str] = None

@app.post("/index")
async def index_document(document: Document):
    try:
        # Tạo embedding cho document
        embedding = genai.embed_content(
            model=embedding_model,
            content=document.content,
            task_type="retrieval_document",
        )["embedding"]

        # Thêm vào ChromaDB
        collection.add(
            documents=[document.content],
            embeddings=[embedding],
            ids=[document.id],
            metadatas=[document.metadata]
        )
        return {"status": "success", "message": "Document indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: Query):
    try:
        # Tìm documents liên quan
        where_filter = {"type": query.type} if query.type else None
        results = collection.query(
            query_texts=[query.query],
            n_results=query.n_results,
            where=where_filter
        )

        # Tạo prompt với context
        context = "\n\n".join(results["documents"][0])
        prompt = f"""You are a knowledgeable football assistant that provides information based on the following context.
        Answer in a friendly and conversational tone, breaking down complex concepts for the general audience.

        Context:
        {context}

        Question: {query.query}

        Answer:"""

        # Generate response
        response = model.generate_content(prompt)

        return {
            "answer": response.text,
            "documents": results["documents"][0],
            "distances": results["distances"][0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_index")
async def bulk_index(documents: List[Document]):
    try:
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

        return {"status": "success", "message": f"{len(documents)} documents indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
