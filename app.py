from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import os, uuid, shutil

from rag_engine import parse_input, build_rag, ask_rag

app = FastAPI(title="Universal RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# GLOBAL STATE
# -------------------------
DOCUMENT_STORE = []   # List[Document]
SOURCE_MAP = {}       # id -> display name
RAG = {"retriever": None, "llm": None}

# -------------------------
# MODELS
# -------------------------
class ChatRequest(BaseModel):
    question: str

class RemoveRequest(BaseModel):
    source_id: str

# -------------------------
# UI
# -------------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# REBUILD RAG
# -------------------------
def rebuild_rag():
    if not DOCUMENT_STORE:
        RAG["retriever"] = None
        RAG["llm"] = None
        return

    retriever, llm = build_rag(DOCUMENT_STORE)
    RAG["retriever"] = retriever
    RAG["llm"] = llm

# -------------------------
# INGEST TEXT / URL
# -------------------------
@app.post("/ingest-text")
def ingest_text(source: str):
    try:
        docs = parse_input(source)
        sid = str(uuid.uuid4())

        DOCUMENT_STORE.extend(docs)
        SOURCE_MAP[sid] = source

        rebuild_rag()

        return {"id": sid, "name": source}

    except Exception as e:
        raise HTTPException(400, str(e))

# -------------------------
# INGEST FILE
# -------------------------
@app.post("/ingest-file")
def ingest_file(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1]
        sid = str(uuid.uuid4())
        fname = f"{sid}{ext}"
        path = os.path.join(UPLOAD_DIR, fname)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        docs = parse_input(path)
        DOCUMENT_STORE.extend(docs)
        SOURCE_MAP[sid] = file.filename

        rebuild_rag()

        return {"id": sid, "name": file.filename}

    except Exception as e:
        raise HTTPException(400, str(e))

# -------------------------
# REMOVE SOURCE
# -------------------------
@app.post("/remove-source")
def remove_source(req: RemoveRequest):
    sid = req.source_id

    if sid not in SOURCE_MAP:
        raise HTTPException(404, "Source not found")

    # rebuild doc store (simple & safe)
    SOURCE_MAP.pop(sid)
    DOCUMENT_STORE.clear()

    for remaining in SOURCE_MAP.values():
        DOCUMENT_STORE.extend(parse_input(remaining))

    rebuild_rag()

    return {"status": "removed"}

# -------------------------
# CHAT
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    if not RAG["retriever"]:
        raise HTTPException(400, "No sources added yet")

    answer = ask_rag(req.question, RAG["retriever"], RAG["llm"])
    return {"answer": answer}
