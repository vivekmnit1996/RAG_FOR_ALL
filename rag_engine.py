import os
import requests
import pandas as pd
import pytesseract

from typing import List
from PIL import Image
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()
# =========================
# SOURCE DETECTION
# =========================
def detect_source_type(user_input: str) -> str:
    user_input = user_input.strip()

    if "youtube.com" in user_input or "youtu.be" in user_input:
        return "youtube"
    if user_input.startswith("http"):
        return "web"

    ext = os.path.splitext(user_input)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in [".png", ".jpg", ".jpeg"]:
        return "image"
    if ext == ".csv":
        return "csv"

    return "code"


def route_input(user_input: str) -> dict:
    return {
        "source_type": detect_source_type(user_input),
        "raw_input": user_input
    }


# =========================
# PARSERS
# =========================
def parse_youtube(input_obj: dict) -> List[Document]:
    url = input_obj["raw_input"]
    video_id = url.split("v=")[-1].split("&")[0]

    try:
        yyt_api = YouTubeTranscriptApi()
        transcript_list = yyt_api.fetch(
            video_id=video_id,
            languages=["en", "hi"]
        )
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts disabled")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found")

    transcript = " ".join(chunk.text for chunk in transcript_list)

    return [Document(
        page_content=transcript,
        metadata={"source": "youtube", "video_id": video_id}
    )]


def parse_pdf(input_obj: dict) -> List[Document]:
    reader = PdfReader(input_obj["raw_input"])
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": "pdf", "page": i}
            ))

    return docs


def parse_web(input_obj: dict) -> List[Document]:
    response = requests.get(input_obj["raw_input"], timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p"))

    return [Document(
        page_content=text,
        metadata={"source": "web"}
    )]


def parse_image(input_obj: dict) -> List[Document]:
    image = Image.open(input_obj["raw_input"])
    text = pytesseract.image_to_string(image)

    return [Document(
        page_content=text,
        metadata={"source": "image"}
    )]


def parse_csv(input_obj: dict) -> List[Document]:
    df = pd.read_csv(input_obj["raw_input"])
    text = df.to_string(index=False)

    return [Document(
        page_content=text,
        metadata={"source": "csv"}
    )]


def parse_code(input_obj: dict) -> List[Document]:
    return [Document(
        page_content=input_obj["raw_input"],
        metadata={"source": "code"}
    )]


# =========================
# MASTER PARSER
# =========================
def parse_input(user_input: str) -> List[Document]:
    routed = route_input(user_input)

    parser_map = {
        "youtube": parse_youtube,
        "pdf": parse_pdf,
        "web": parse_web,
        "image": parse_image,
        "csv": parse_csv,
        "code": parse_code
    }

    return parser_map[routed["source_type"]](routed)


# =========================
# RAG PIPELINE
# =========================
def build_rag(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    return retriever, llm


def ask_rag(question: str, retriever, llm) -> str:
    docs = retriever.invoke(question)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a helpful assistant.
Answer using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content


# # =========================
# # TEST RUN
# # =========================
# if __name__ == "__main__":


#     docs = parse_input(
#         "https://www.youtube.com/watch?v=YdOXS_9_P4U"
#     )

#     retriever, llm = build_rag(docs)

#     answer = ask_rag(
#         "What is the video about?",
#         retriever,
#         llm
#     )

#     print("\nANSWER:\n", answer)
