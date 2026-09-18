# AcademiaRAG Pro

## Enterprise Knowledge Suite

AcademiaRAG Pro is a local AI-powered university regulation assistant.

It helps students find answers from official university regulation documents without manually searching through long PDF files.

---

## Features

- Read official university PDF documents
- Extract and clean PDF text
- Split documents into searchable chunks
- Semantic search using AI embeddings
- FAISS vector search
- CrossEncoder reranking
- Question intent detection
- Evidence verification
- Source and page references
- Feedback-based learning memory
- Reject unrelated questions
- Local and offline document processing
- Simple desktop dashboard

---

## How It Works

```text
User Question
      |
      v
Intent Detection
      |
      v
Text Embedding
      |
      v
FAISS Semantic Search
      |
      v
Relevant Document Chunks
      |
      v
CrossEncoder Reranking
      |
      v
Evidence Verification
      |
      +------------------+
      |                  |
      v                  v
Reliable Evidence    No Reliable Evidence
      |                  |
      v                  v
Answer             No Relevant Information
      |
      v
Source + Page
      |
      v
User Feedback
      |
      v
Learning Memory
```

---

## Technologies Used

### Programming Language

- Python 3.12

### User Interface

- CustomTkinter
- Tkinter

### AI / Machine Learning

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Semantic Search
- CrossEncoder Reranking
- Intent Detection
- Evidence Verification
- Adaptive Learning

### AI Models

#### Embedding Model

```text
all-MiniLM-L6-v2
```

Used to convert questions and document chunks into numerical embeddings.

Embedding dimension:

```text
384
```

#### Reranking Model

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

Used to rank retrieved document chunks according to their relevance to the user's question.

### Vector Search

- FAISS

### PDF Processing

- pdfplumber

### Database

- SQLite

### Supporting Libraries

- NumPy
- PyTorch
- Pillow

### Development Tools

- Visual Studio Code
- Python
- pip
- Git
- GitHub

---

## Project Structure

```text
AcademiaRAG-Pro/
│
├── app.py
│   └── Main desktop application and dashboard
│
├── rag_engine.py
│   └── RAG pipeline and retrieval system
│
├── make_logo.py
│   └── Application logo generator
│
├── crest.ico
│   └── Application icon
│
├── university_docs/
│   └── Official university regulation PDFs
│
├── learning_memory.db
│   └── Local feedback and learning memory
│
└── README.md
    └── Project documentation
```

---

## Main Components

### app.py

Provides the graphical user interface.

It allows users to:

- Load AI models
- Import university PDFs
- Build the vector database
- Ask questions
- View answers
- View source documents and page numbers
- Provide feedback

### rag_engine.py

Contains the main RAG pipeline.

It handles:

- PDF extraction
- Text cleaning
- Text chunking
- Embedding generation
- FAISS indexing
- Semantic retrieval
- CrossEncoder reranking
- Intent detection
- Evidence verification
- Learning memory

### university_docs

Stores the official university documents used as the knowledge source.

Example:

```text
R22_B.Tech Regulations.pdf
```

---

## Current Knowledge Base

The current project uses the official:

```text
Vignan University R22 B.Tech Academic Regulations
```

The current test system successfully processed:

```text
Pages: 34
Chunks: 155
Embedding Dimension: 384
```

---

## Example Question

```text
What percentage of attendance is required in each course?
```

Example response:

```text
The attendance in each course shall not be less than
75% of the aggregate of all L, T, P sessions conducted
in that course.

Document: R22_B.Tech Regulations.pdf
Page: 16

Evidence Verified: Yes
```

---

## Other Example Questions

```text
What percentage of attendance is required?

How often is attendance reviewed?

Can attendance shortage be condoned?

What is the maximum attendance shortage that can be condoned?

Who receives information about the student's attendance status?

What are supplementary examinations?

What does an R grade mean?
```

---

## Irrelevant Question Handling

The system is designed to answer questions based on the documents in its knowledge base.

For unrelated questions such as:

```text
What is the capital of France?
```

the system can return:

```text
NO RELEVANT INFORMATION FOUND
```

This prevents the system from generating unsupported answers.

---

## Adaptive Learning

AcademiaRAG Pro includes a feedback-based learning memory.

The system stores previous interactions separately.

```text
User Question
      |
      v
Retrieved Answer
      |
      v
User Feedback
      |
      v
Learning Memory
      |
      v
Future Similar Questions
```

The learning memory does not modify the original university PDF.

---

## Evidence Verification

The system checks whether the retrieved information provides sufficient evidence.

A verified answer is displayed with:

```text
Evidence Verified: Yes
```

If sufficient evidence cannot be found:

```text
NO RELEVANT INFORMATION FOUND
```

No unsupported answer is generated.

---

## Source Traceability

Each answer can show its source:

```text
Document: R22_B.Tech Regulations.pdf
Page: 16
```

This allows users to verify the information in the original university document.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/kala0423/AcademiaRAG-Pro.git
```

### 2. Open the Project

```bash
cd AcademiaRAG-Pro
```

### 3. Install Required Packages

```bash
pip install customtkinter pdfplumber sentence-transformers faiss-cpu numpy pillow torch
```

### 4. Run the Application

```bash
python app.py
```

---

## Usage

```text
1. Start the application
        ↓
2. Load AI models
        ↓
3. Select official university PDF
        ↓
4. Build vector database
        ↓
5. Ask a question
        ↓
6. View answer and source
        ↓
7. Give feedback
```

---

## Advantages

- Easy to use
- Local AI processing
- Document-grounded answers
- Semantic search
- Fast vector retrieval
- CrossEncoder reranking
- Evidence verification
- Source and page references
- Feedback-based learning
- Rejects unrelated questions
- Can be extended with additional documents

---

## Future Improvements

- Support multiple regulation documents
- Support R22, R25 and other regulation versions
- Multi-document search
- Better paraphrase understanding
- Chat history
- Voice-based questions
- OCR for scanned PDFs
- Web-based interface
- Mobile application
- Student and faculty dashboards
- Advanced analytics

---

## Privacy

The RAG pipeline is designed to process documents locally.

Feedback and learning information is stored in a local SQLite database.

The original university PDF is not modified by the learning system.

---

## Disclaimer

AcademiaRAG Pro is an academic project prototype.

Users should verify important academic information using the latest official university regulations and notices.

The accuracy of the system depends on the documents available in its knowledge base.

---

## Author

**Kalanjali Kakumanu**

B.Tech Computer Science and Engineering

(Data Science)

Vignan's Foundation for Science, Technology & Research

---

## Project Goal

AcademiaRAG Pro aims to make university regulations easier to search and understand.

Instead of manually searching through a large PDF:

```text
Large PDF
   ↓
Manual Search
   ↓
Find Required Rule
```

the user can simply ask:

```text
What percentage of attendance is required?
```

and the system performs:

```text
Question
   ↓
Semantic Search
   ↓
FAISS Retrieval
   ↓
CrossEncoder Reranking
   ↓
Evidence Verification
   ↓
Answer + Source + Page
```

---

## Project Status

**AcademiaRAG Pro V7.3**

Core RAG pipeline, semantic retrieval, FAISS search, CrossEncoder reranking, intent detection, evidence verification, and adaptive learning memory are implemented and tested.
