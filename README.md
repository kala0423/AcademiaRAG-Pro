# AcademiaRAG Pro

## Enterprise Knowledge Suite

AcademiaRAG Pro is a local AI-powered university regulation assistant.

It helps students quickly find answers from official university regulation documents without manually searching through long PDF files.

The system uses Retrieval-Augmented Generation (RAG) concepts, semantic search, vector search, CrossEncoder reranking, intent detection, evidence verification, and feedback-based learning.

The main goal is to provide answers that are directly grounded in official university documents.

---

## Project Overview

University regulations are usually available as large PDF documents. Students may have difficulty finding specific information such as:

- Attendance requirements
- Attendance shortage rules
- Attendance condonation
- Examination rules
- Supplementary examinations
- R grade information
- Academic regulations

AcademiaRAG Pro solves this problem by allowing the user to ask questions in normal English.

For example:

> What percentage of attendance is required in each course?

The system searches the local university regulation document and returns the relevant answer along with the document name and page number.

---

## Main Features

### 1. PDF Document Processing

The system can read university regulation PDF documents.

It:

1. Extracts text from the PDF.
2. Cleans unnecessary formatting.
3. Divides the document into smaller chunks.
4. Converts the chunks into numerical embeddings.
5. Stores the embeddings for searching.

---

### 2. Semantic Search

The system does not depend only on exact keywords.

It tries to understand the meaning of the question.

For example:

> What percentage of attendance is required?

and

> What is the minimum attendance percentage a student must maintain?

can refer to the same regulation.

This makes the system more useful than simple keyword search.

---

### 3. FAISS Vector Search

FAISS is used to store and search document embeddings.

The process is:

```text
User Question
      |
      v
Question Embedding
      |
      v
FAISS Semantic Search
      |
      v
Relevant Document Chunks
