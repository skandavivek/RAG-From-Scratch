from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
import fitz
from llama_index.readers.file import PyMuPDFReader

from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

from pydantic import BaseModel
import uvicorn
# from llama_index.core import SimpleDirectoryReader
# from llama_index import VectorStoreIndex, Document
# from llama_index.vector_stores import SimpleVectorStore
# from llama_index.storage.storage_context import StorageContext
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import psycopg2
import tempfile
import os
from dotenv import load_dotenv


app = FastAPI()


# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Load environment variables from .env file
load_dotenv()

# Get the OPENAI_API_KEY
openai_api_key = os.environ["OPENAI_KEYV"]
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o")

Settings.llm = llm
Settings.embed_model = embed_model
client = QdrantClient(":memory:")
#client = qdrant_client.QdrantClient(
    #"<qdrant-url>",
    #api_key="<qdrant-api-key>", # For Qdrant Cloud, None for local instance
#)
# In-memory vector store setup
# vector_store = SimpleVectorStore()
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
#index = None

# PostgreSQL connection
conn = psycopg2.connect(os.environ["DATABASE_URL"])

class Query(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global index
    content = await file.read()
    _, file_extension = os.path.splitext(file.filename)
    
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, mode='wb',suffix=file_extension) as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name


   

    loader = PyMuPDFReader()
    documents = loader.load(file_path=temp_file_path)

    text_parser = SentenceSplitter(
        chunk_size=2048,
    )
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    from llama_index.core.schema import TextNode

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    vector_store = QdrantVectorStore(client=client, collection_name="documents")

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    vector_store.add(nodes)


    # Initialize storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    


    index = VectorStoreIndex.from_vector_store(vector_store,storage_context=storage_context)

    # Remove the temporary file
    #os.unlink(temp_file_path)
    
    return {"message": "File uploaded and indexed successfully"}
    os.unlink(temp_file_path)

@app.post("/query")
async def query(query: Query):
    if not index:
        return {"error": "No document has been uploaded yet"}

    query_engine = index.as_query_engine()
    response = query_engine.query(query.question)

    
#Inspect source nodes
    context=[]
    for node in response.source_nodes:
        print("-----")
        text_fmt = node.node.get_content()
        context.append(text_fmt)
        print(f"Text:\t {text_fmt} ...")
        print(f"Metadata:\t {node.node.metadata}")
        print(f"Score:\t {node.score:.3f}")
    #print(response.source_nodes[0])
    
    # Store query and response in PostgreSQL
    cur = conn.cursor()
    cur.execute("INSERT INTO queries (question, context, answer) VALUES (%s, %s, %s)", (query.question, context, str(response)))
    conn.commit()
    
    return {"answer": str(response)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)