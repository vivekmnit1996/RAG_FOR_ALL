# Universal-RAG

## Overview

Universal-RAG is a Retrieval-Augmented Generation (RAG) based project that allows users to upload documents or provide online sources and then ask questions based on that data. The system retrieves relevant information from the provided sources and generates accurate, context-aware answers.

This project is created with a student-friendly approach, keeping the code structure simple and the explanation easy to understand.

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines two important ideas:

* **Retrieval** – finding the most relevant information from documents or links
* **Generation** – using an AI model to generate answers using that retrieved information

Instead of relying only on pre-trained knowledge, RAG ensures that answers are grounded in the user’s own data.

---

## Features

* Upload PDF files
* Upload CSV files
* Add website links
* Add YouTube video links
* Ask questions based on uploaded data
* Accurate and data-driven responses

---

## Project Structure

```
Universal-RAG
├── frontend/              # Frontend user interface
├── app.py                 # Main backend application
├── rag_engine.py          # Core RAG processing logic
├── requirements.txt       # Python dependencies
├── .gitignore             # Files ignored by Git
└── README.md              # Project documentation
```

---

## How the System Works

1. User uploads a document or provides a link
2. Text is extracted from the source
3. Important data is converted into embeddings
4. When a question is asked, the system retrieves relevant data
5. The AI model generates an answer using that data

This approach makes responses more reliable and relevant.

---

## Requirements

* Python 3.8 or above
* Required libraries from `requirements.txt`
* Internet connection (for web and YouTube sources)

---

## Installation Steps

1. Clone the repository

```
git clone https://github.com/Pranav-by/Universal-RAG.git
cd Universal-RAG
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the application

```
python app.py
```

4. Open the frontend as instructed in the `frontend` folder

---

## Usage

* Start the application
* Upload PDF or CSV files or provide links
* Enter a question related to the data
* Get answers generated from your own content

---

## Use Cases

* Study assistant for students
* Document-based question answering
* Research help tool
* Knowledge extraction from files and links

---


