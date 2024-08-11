from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from celery import Celery
from qdrant_client import QdrantClient
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel
import uvicorn
import psycopg2
import tempfile
import os
from dotenv import load_dotenv
import logging
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load environment variables
load_dotenv()

# Set up Celery
celery_app = Celery('tasks', broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"), backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

# OpenAI and Qdrant setup
openai_api_key = os.environ["OPENAI_KEYV"]
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEYV"]
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o")

Settings.llm = llm
Settings.embed_model = embed_model
client = QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    url=os.environ["QDRANT_URL"],
    # otherwise set Qdrant instance with host and port:
    # host="localhost",
    # port=6333
    # set API KEY for Qdrant Cloud
    api_key=os.environ["QDRANT_API_KEY"],
    # path="./db/"
)
index = None

# PostgreSQL connection
conn = psycopg2.connect(os.environ["DATABASE_URL"])

class Query(BaseModel):
    question: str

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_embedding(text):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={"input": text, "model": "text-embedding-3-small"}
        ) as resp:
            data = await resp.json()
            return data['data'][0]['embedding']

async def process_nodes(nodes):
    tasks = [get_embedding(node.get_content(metadata_mode="all")) for node in nodes]
    embeddings = await asyncio.gather(*tasks)
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
    return nodes

@celery_app.task
def process_file(file_content, file_name):
    global index
    logging.info(f"Starting to process file: {file_name}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix=os.path.splitext(file_name)[1]) as temp_file:
        logging.info(f"Starting to write file to cache: {file_name}")
        temp_file.write(file_content)
        logging.info(f"Finished writing file to redis: {file_name}")
        temp_file_path = temp_file.name

    try:
        loader = PyMuPDFReader()
        documents = loader.load(file_path=temp_file_path)
        logging.info(f"Finished reading file text: {file_name}")

        text_parser = SentenceSplitter(chunk_size=2048)
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))

        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)

        logging.info(f"Finished chunking file text: {file_name}")

        loop = asyncio.get_event_loop()
        nodes_with_embeddings = loop.run_until_complete(process_nodes(nodes))

        # Get the list of existing collections
        existing_collections = client.get_collections()

        # Check if the collection exists
        if "documents_basic_rag" in [collection.name for collection in existing_collections.collections]:

            client.delete_collection(collection_name="documents_basic_rag")
        vector_store = QdrantVectorStore(client=client, collection_name="documents_basic_rag")

        vector_store.add(nodes_with_embeddings)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        logging.info(f"Finished processing file: {file_name}")
        return "File processed successfully"
    except Exception as e:
        logging.error(f"Error processing file {file_name}: {str(e)}")
        raise
    finally:
        os.unlink(temp_file_path)

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logging.info(f"Received upload request for file: {file.filename}")
    content = await file.read()
    task = process_file.delay(content, file.filename)
    return {"message": "File upload started", "task_id": str(task.id)}

@app.get("/upload_status/{task_id}")
async def get_upload_status(task_id: str):
    task = process_file.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {'status': 'Processing'}
    elif task.state != 'FAILURE':
        return {'status': 'Completed'}
    else:
        return {'status': 'Failed', 'error': str(task.result)}

@app.post("/query")
async def query(query: Query):
    vector_store = QdrantVectorStore(client=client, collection_name="documents_basic_rag")
    index = VectorStoreIndex.from_vector_store(vector_store)

    try:
        if index is None:
            raise HTTPException(status_code=400, detail="No documents have been processed yet. Please upload a file first.")
        query_engine = index.as_query_engine()
        response = query_engine.query(query.question)

        context = []
        for node in response.source_nodes:
            text_fmt = node.node.get_content()
            context.append(text_fmt)
            logging.info(f"Source node: Text: {text_fmt[:100]}... Metadata: {node.node.metadata} Score: {node.score:.3f}")

        cur = conn.cursor()
        cur.execute("INSERT INTO queries (question, context, answer) VALUES (%s, %s, %s)", (query.question, context, str(response)))
        conn.commit()

        return {"answer": str(response)}
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))