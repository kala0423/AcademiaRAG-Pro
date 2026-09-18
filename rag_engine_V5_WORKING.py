# ============================================================
# AcademiaRAG Pro V5
# Adaptive Self-Learning RAG Engine
# ============================================================

import os
import re
import math
import sqlite3
import threading
from typing import List, Dict, Optional

import pdfplumber
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DEFAULT_PDF_FOLDER = "university_docs"

LEARNING_DATABASE = "learning_memory.db"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 2

INITIAL_RETRIEVAL_K = 15
FINAL_CHUNK_K = 8

SEMANTIC_WEIGHT = 0.35
RERANK_WEIGHT = 0.65

MIN_EVIDENCE_SCORE = 0.42
LEARNING_SIMILARITY_THRESHOLD = 0.78


# ============================================================
# GLOBAL MODELS
# ============================================================

_embedding_model = None
_reranker_model = None

_model_lock = threading.Lock()


# ============================================================
# MODEL LOADING
# ============================================================

def load_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        with _model_lock:

            if _embedding_model is None:

                print("\nLoading embedding model...")

                _embedding_model = SentenceTransformer(
                    EMBEDDING_MODEL
                )

                print(
                    "Embedding model loaded."
                )

    return _embedding_model


def load_reranker_model():

    global _reranker_model

    if _reranker_model is None:

        with _model_lock:

            if _reranker_model is None:

                print(
                    "\nLoading CrossEncoder reranker..."
                )

                _reranker_model = CrossEncoder(
                    RERANKER_MODEL
                )

                print(
                    "CrossEncoder reranker loaded."
                )

    return _reranker_model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_into_sentences(text: str) -> List[str]:

    text = clean_text(text)

    if not text:
        return []

    # --------------------------------------------------------
    # Preserve useful PDF list markers
    # --------------------------------------------------------

    text = re.sub(
        r"\s+([a-z]\))",
        r"\n\1",
        text
    )

    text = re.sub(
        r"\s+([A-Z]\))",
        r"\n\1",
        text
    )

    text = re.sub(
        r"\s+(\d+\.\d+)",
        r"\n\1",
        text
    )

    lines = text.split("\n")

    sentences = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # ----------------------------------------------------
        # Split after punctuation.
        #
        # IMPORTANT:
        # Do NOT require the next character to be uppercase.
        # PDF extraction can produce:
        #
        # "course. a) The attendance..."
        #
        # ----------------------------------------------------

        parts = re.split(
            r"(?<=[.!?])\s+",
            line
        )

        for part in parts:

            part = part.strip()

            if len(part) < 15:
                continue

            sentences.append(part)

    return sentences


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    page_text: str,
    page_number: int,
    source_name: str
) -> List[Dict]:

    units = split_into_sentences(
        page_text
    )

    if not units:
        return []

    chunks = []

    current_units = []
    current_length = 0

    for unit in units:

        unit_length = len(unit)

        if (
            current_units
            and
            current_length + unit_length
            > CHUNK_SIZE
        ):

            chunk_text = " ".join(
                current_units
            ).strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "source": source_name
                    }
                )

            overlap_units = (
                current_units[
                    -CHUNK_OVERLAP:
                ]
            )

            current_units = (
                overlap_units.copy()
            )

            current_length = sum(
                len(x)
                for x in current_units
            )

        current_units.append(unit)

        current_length += unit_length

    if current_units:

        chunk_text = " ".join(
            current_units
        ).strip()

        if chunk_text:

            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                    "source": source_name
                }
            )

    return chunks


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_chunks(
    pdf_path: str
) -> List[Dict]:

    all_chunks = []

    source_name = os.path.basename(
        pdf_path
    )

    print(
        f"\nReading PDF: {source_name}"
    )

    try:

        with pdfplumber.open(
            pdf_path
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=3
                )

                if not text:
                    continue

                page_chunks = create_chunks(
                    text,
                    page_number,
                    source_name
                )

                all_chunks.extend(
                    page_chunks
                )

        print(
            f"Extracted {len(all_chunks)} chunks "
            f"from {source_name}"
        )

    except Exception as e:

        print(
            f"PDF extraction error: {e}"
        )

    return all_chunks


# ============================================================
# KEYWORDS
# ============================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "for",
    "and",
    "or",
    "does",
    "do",
    "can",
    "how",
    "who",
    "when",
    "where",
    "why",
    "with",
    "be",
    "this",
    "that",
    "student",
    "students",
    "university",
    "give",
    "provide"
}


def extract_keywords(
    text: str
) -> List[str]:

    words = re.findall(
        r"\b[a-zA-Z0-9%]+\b",
        text.lower()
    )

    keywords = []

    for word in words:

        if (
            len(word) >= 3
            and word not in STOP_WORDS
        ):

            keywords.append(word)

    return list(
        dict.fromkeys(keywords)
    )


def keyword_score(
    query: str,
    text: str
) -> float:

    keywords = extract_keywords(
        query
    )

    if not keywords:
        return 0.0

    text_lower = text.lower()

    matches = sum(
        1
        for keyword in keywords
        if keyword in text_lower
    )

    return (
        matches / len(keywords)
    )


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(
    query: str
) -> str:

    q = query.lower()

    if (
        "attendance" in q
        and (
            "how often" in q
            or "review" in q
            or "reviewed" in q
            or "period" in q
        )
    ):
        return "attendance_frequency"

    if (
        "attendance" in q
        and (
            "required" in q
            or "minimum" in q
            or "percentage" in q
            or "%" in q
            or "need" in q
        )
    ):
        return "attendance_requirement"

    if (
        "attendance" in q
        and (
            "condone" in q
            or "condonation" in q
            or "shortage" in q
        )
    ):
        return "attendance_condonation"

    if (
        "attendance" in q
        and (
            "parent" in q
            or "guardian" in q
            or "information" in q
        )
    ):
        return "attendance_status"

    if (
        "supplementary" in q
        or "supplement" in q
    ):
        return "supplementary"

    if (
        "tesla" in q
        or "car" in q
        or "vehicle" in q
    ):
        return "reward_vehicle"

    if (
        "laptop" in q
        or "reward" in q
        or "gift" in q
    ):
        return "reward"

    return "general"


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {

    "attendance_requirement": [
        "attendance",
        "75",
        "percentage",
        "required",
        "minimum"
    ],

    "attendance_frequency": [
        "attendance",
        "review",
        "reviewed",
        "every",
        "weeks"
    ],

    "attendance_condonation": [
        "attendance",
        "shortage",
        "condoned",
        "condonation",
        "10%"
    ],

    "attendance_status": [
        "attendance",
        "parents",
        "guardian",
        "status"
    ],

    "supplementary": [
        "supplementary",
        "examination",
        "exam"
    ],

    "reward_vehicle": [
        "tesla",
        "car",
        "vehicle",
        "attendance"
    ],

    "reward": [
        "laptop",
        "reward",
        "gift"
    ]
}


def intent_score(
    query: str,
    text: str,
    intent: str
) -> float:

    keywords = INTENT_KEYWORDS.get(
        intent,
        []
    )

    if not keywords:
        return 0.0

    text_lower = text.lower()

    matches = sum(
        1
        for keyword in keywords
        if keyword in text_lower
    )

    return (
        matches / len(keywords)
    )


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(
    x: float
) -> float:

    try:

        return 1.0 / (
            1.0 + math.exp(-x)
        )

    except OverflowError:

        return (
            0.0
            if x < 0
            else 1.0
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(
    text: str
) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9% ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DATABASE
# ============================================================

class LearningMemory:

    def __init__(
        self,
        database_path=LEARNING_DATABASE
    ):

        self.database_path = (
            database_path
        )

        self.lock = threading.Lock()

        self.create_database()

    # --------------------------------------------------------
    # CREATE TABLE
    # --------------------------------------------------------

    def create_database(self):

        with sqlite3.connect(
            self.database_path
        ) as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS
                learning_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    source TEXT,
                    page INTEGER,
                    feedback TEXT,
                    correction TEXT,
                    verified INTEGER DEFAULT 0,
                    created_at TIMESTAMP
                    DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.commit()

    # --------------------------------------------------------
    # SAVE FEEDBACK
    # --------------------------------------------------------

    def save_feedback(
        self,
        question: str,
        answer: str,
        source: Optional[str],
        page: Optional[int],
        feedback: str,
        correction: Optional[str] = None
    ):

        with self.lock:

            with sqlite3.connect(
                self.database_path
            ) as conn:

                cursor = conn.cursor()

                verified = (
                    1
                    if feedback == "helpful"
                    or correction
                    else 0
                )

                cursor.execute(
                    """
                    INSERT INTO learning_memory
                    (
                        question,
                        answer,
                        source,
                        page,
                        feedback,
                        correction,
                        verified
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        question,
                        answer,
                        source,
                        page,
                        feedback,
                        correction,
                        verified
                    )
                )

                conn.commit()

    # --------------------------------------------------------
    # GET VERIFIED MEMORIES
    # --------------------------------------------------------

    def get_verified_memories(
        self
    ):

        with sqlite3.connect(
            self.database_path
        ) as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    question,
                    answer,
                    source,
                    page,
                    correction
                FROM learning_memory
                WHERE verified = 1
                """
            )

            return cursor.fetchall()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self):

        with sqlite3.connect(
            self.database_path
        ) as conn:

            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM learning_memory
                """
            )

            total = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM learning_memory
                WHERE feedback = 'helpful'
                """
            )

            helpful = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM learning_memory
                WHERE feedback = 'not_helpful'
                """
            )

            not_helpful = (
                cursor.fetchone()[0]
            )

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM learning_memory
                WHERE correction IS NOT NULL
                AND correction != ''
                """
            )

            corrections = (
                cursor.fetchone()[0]
            )

        return {
            "total": total,
            "helpful": helpful,
            "not_helpful": not_helpful,
            "corrections": corrections
        }


# ============================================================
# MAIN RAG ENGINE
# ============================================================

class LocalRAGEngine:

    def __init__(
        self,
        pdf_folder=DEFAULT_PDF_FOLDER
    ):

        self.pdf_folder = pdf_folder

        self.embedding_model = None
        self.reranker = None

        self.chunks = []
        self.documents_metadata = []

        self.index = None

        self.ready = False
        self.model_ready = False

        self.learning_memory = (
            LearningMemory()
        )

        self.learning_embeddings = None
        self.learning_records = []

        print("\n" + "=" * 60)
        print("AcademiaRAG Pro V5")
        print("Adaptive Self-Learning RAG Engine")
        print("=" * 60)

    # ========================================================
    # MODEL API
    # ========================================================

    def load_model(self):

        try:

            self.embedding_model = (
                load_embedding_model()
            )

            self.reranker = (
                load_reranker_model()
            )

            self.model_ready = True

            return True

        except Exception as e:

            print(
                f"Model loading error: {e}"
            )

            return False

    def is_model_ready(self):

        return self.model_ready

    # ========================================================
    # BUILD DATABASE API
    # ========================================================

    def build_vector_db(
        self,
        folder=None
    ):

        try:

            if not self.model_ready:

                if not self.load_model():

                    return False

            if folder is None:

                folder = self.pdf_folder

            if not os.path.exists(folder):

                print(
                    f"Folder not found: {folder}"
                )

                return False

            pdf_files = [

                os.path.join(
                    folder,
                    filename
                )

                for filename
                in os.listdir(folder)

                if filename.lower().endswith(
                    ".pdf"
                )
            ]

            if not pdf_files:

                print(
                    "No PDF files found."
                )

                return False

            self.chunks = []

            for pdf in pdf_files:

                extracted = (
                    extract_pdf_chunks(pdf)
                )

                self.chunks.extend(
                    extracted
                )

            if not self.chunks:

                print(
                    "No readable PDF content."
                )

                return False

            print(
                f"\nTotal chunks: "
                f"{len(self.chunks)}"
            )

            texts = [
                chunk["text"]
                for chunk in self.chunks
            ]

            print(
                "\nCreating embeddings..."
            )

            embeddings = (
                self.embedding_model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            )

            embeddings = embeddings.astype(
                "float32"
            )

            dimension = (
                embeddings.shape[1]
            )

            self.index = (
                faiss.IndexFlatIP(
                    dimension
                )
            )

            self.index.add(
                embeddings
            )

            self.documents_metadata = (
                self.chunks
            )

            self.ready = True

            print(
                "\nVector database ready."
            )

            print(
                f"Embedding dimension: "
                f"{dimension}"
            )

            print(
                f"Indexed documents: "
                f"{len(self.chunks)}"
            )

            # Load previous learning memory
            self.refresh_learning_memory()

            return True

        except Exception as e:

            print(
                f"\nVector database error: {e}"
            )

            return False

    def is_index_ready(self):

        return self.ready

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve_candidates(
        self,
        query,
        top_k=INITIAL_RETRIEVAL_K
    ):

        if not self.ready:

            return []

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        )

        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )

        k = min(
            top_k,
            len(self.chunks)
        )

        distances, indices = (
            self.index.search(
                query_embedding,
                k
            )
        )

        candidates = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index < 0:
                continue

            chunk = dict(
                self.chunks[index]
            )

            chunk[
                "semantic_score"
            ] = float(distance)

            candidates.append(
                chunk
            )

        return candidates

    # ========================================================
    # CROSS ENCODER
    # ========================================================

    def rerank(
        self,
        query,
        candidates
    ):

        if not candidates:

            return []

        pairs = [

            [
                query,
                candidate["text"]
            ]

            for candidate in candidates
        ]

        scores = (
            self.reranker.predict(
                pairs
            )
        )

        results = []

        for candidate, raw_score in zip(
            candidates,
            scores
        ):

            candidate = dict(
                candidate
            )

            semantic = (
                candidate[
                    "semantic_score"
                ]
            )

            reranker_probability = (
                sigmoid(
                    float(raw_score)
                )
            )

            keyword = keyword_score(
                query,
                candidate["text"]
            )

            intent = detect_intent(
                query
            )

            intent_match = intent_score(
                query,
                candidate["text"],
                intent
            )

            final_score = (

                0.30 * semantic

                +

                0.45 *
                reranker_probability

                +

                0.15 * keyword

                +

                0.10 * intent_match
            )

            candidate[
                "reranker_score"
            ] = float(raw_score)

            candidate[
                "reranker_probability"
            ] = reranker_probability

            candidate[
                "keyword_score"
            ] = keyword

            candidate[
                "intent_score"
            ] = intent_match

            candidate[
                "score"
            ] = final_score

            results.append(
                candidate
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results

    # ========================================================
    # LEARNING MEMORY
    # ========================================================

    def refresh_learning_memory(self):

        records = (
            self.learning_memory
            .get_verified_memories()
        )

        self.learning_records = []

        if not records:

            self.learning_embeddings = None

            print(
                "Learning memory: empty"
            )

            return

        questions = []

        for row in records:

            question = row[0]
            answer = row[1]
            source = row[2]
            page = row[3]
            correction = row[4]

            self.learning_records.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "page": page,
                    "correction": correction
                }
            )

            questions.append(
                question
            )

        self.learning_embeddings = (
            self.embedding_model.encode(
                questions,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
        )

        print(
            f"Learning memory loaded: "
            f"{len(questions)} verified interactions"
        )

    # ========================================================
    # SEARCH LEARNING MEMORY
    # ========================================================

    def search_learning_memory(
        self,
        query
    ):

        if (
            self.learning_embeddings
            is None
            or not self.learning_records
        ):

            return None

        query_embedding = (
            self.embedding_model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
        )

        scores = (
            np.dot(
                self.learning_embeddings,
                query_embedding[0]
            )
        )

        best_index = int(
            np.argmax(scores)
        )

        best_score = float(
            scores[best_index]
        )

        if (
            best_score
            >= LEARNING_SIMILARITY_THRESHOLD
        ):

            memory = dict(
                self.learning_records[
                    best_index
                ]
            )

            memory[
                "similarity"
            ] = best_score

            return memory

        return None

    # ========================================================
    # FEEDBACK
    # ========================================================

    def save_feedback(
        self,
        question,
        answer,
        source,
        page,
        feedback,
        correction=None
    ):

        self.learning_memory.save_feedback(
            question=question,
            answer=answer,
            source=source,
            page=page,
            feedback=feedback,
            correction=correction
        )

        self.refresh_learning_memory()

        print(
            "\nLearning memory updated."
        )

    # ========================================================
    # LEARNING STATISTICS
    # ========================================================

    def get_learning_statistics(self):

        return (
            self.learning_memory
            .statistics()
        )

    # ========================================================
    # CLEAN ANSWER EXTRACTION
    # ========================================================

    def extract_answer_sentences(
        self,
        query,
        results,
        max_sentences=3
    ):

        if not results:

            return []

        intent = detect_intent(
            query
        )

        sentence_records = []

        # ----------------------------------------------------
        # Collect sentences
        # ----------------------------------------------------

        for result in results:

            sentences = split_into_sentences(
                result["text"]
            )

            for sentence in sentences:

                if len(sentence) < 20:
                    continue

                sentence_records.append(
                    {
                        "text": sentence,
                        "source":
                            result["source"],
                        "page":
                            result["page"],
                        "chunk_score":
                            result["score"]
                    }
                )

        if not sentence_records:

            return []

        # ----------------------------------------------------
        # CrossEncoder sentence scoring
        # ----------------------------------------------------

        pairs = [

            [
                query,
                record["text"]
            ]

            for record in sentence_records
        ]

        scores = (
            self.reranker.predict(
                pairs
            )
        )

        ranked = []

        for record, raw_score in zip(
            sentence_records,
            scores
        ):

            semantic = sigmoid(
                float(raw_score)
            )

            keyword = keyword_score(
                query,
                record["text"]
            )

            intent_match = intent_score(
                query,
                record["text"],
                intent
            )

            score = (

                0.55 * semantic

                +

                0.25 * keyword

                +

                0.20 * intent_match
            )

            ranked.append(
                {
                    **record,
                    "score": score
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # ----------------------------------------------------
        # Intent-specific filtering
        # ----------------------------------------------------

        selected = []

        for record in ranked:

            text_lower = (
                record["text"].lower()
            )

            include = True

            if intent == (
                "attendance_requirement"
            ):

                include = (
                    "attendance"
                    in text_lower
                    and "%"
                    in text_lower
                )

            elif intent == (
                "attendance_frequency"
            ):

                include = (
                    "attendance"
                    in text_lower
                    and (
                        "every"
                        in text_lower
                        or "weeks"
                        in text_lower
                        or "periodically"
                        in text_lower
                    )
                )

            elif intent == (
                "attendance_condonation"
            ):

                include = (
                    "shortage"
                    in text_lower
                    or "condoned"
                    in text_lower
                )

            elif intent == (
                "attendance_status"
            ):

                include = (
                    "parent"
                    in text_lower
                    or "guardian"
                    in text_lower
                )

            if not include:
                continue

            # ------------------------------------------------
            # Remove duplicates
            # ------------------------------------------------

            normalized = normalize_text(
                record["text"]
            )

            duplicate = False

            for existing in selected:

                existing_normalized = (
                    normalize_text(
                        existing["text"]
                    )
                )

                if (
                    normalized
                    == existing_normalized
                ):

                    duplicate = True
                    break

                if (
                    normalized
                    in existing_normalized
                    or
                    existing_normalized
                    in normalized
                ):

                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(
                record
            )

            if len(selected) >= max_sentences:
                break

        return selected

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def semantic_search(
        self,
        query,
        top_k=5,
        threshold=0.30
    ):

        if not query:

            return []

        query = query.strip()

        if not query:

            return []

        if not self.ready:

            return []

        # ----------------------------------------------------
        # First check learning memory
        # ----------------------------------------------------

        memory = (
            self.search_learning_memory(
                query
            )
        )

        if memory:

            print(
                "\n✓ Learned answer matched."
            )

            print(
                f"Learning similarity: "
                f"{memory['similarity']:.3f}"
            )

            learned_answer = (
                memory["correction"]
                if memory["correction"]
                else memory["answer"]
            )

            return [
                {
                    "text": learned_answer,
                    "source":
                        memory["source"]
                        or "Learning Memory",
                    "page":
                        memory["page"]
                        or "-",
                    "score":
                        memory["similarity"],
                    "learned": True
                }
            ]

        # ----------------------------------------------------
        # Normal RAG retrieval
        # ----------------------------------------------------

        candidates = (
            self.retrieve_candidates(
                query,
                INITIAL_RETRIEVAL_K
            )
        )

        if not candidates:

            return []

        reranked = self.rerank(
            query,
            candidates
        )

        # ----------------------------------------------------
        # Evidence safeguard
        # ----------------------------------------------------

        verified_results = []

        intent = detect_intent(
            query
        )

        for result in reranked:

            score = result["score"]

            text_lower = (
                result["text"].lower()
            )

            # Reward questions need stronger evidence.
            if intent == "reward_vehicle":

                if not any(
                    term in text_lower
                    for term in [
                        "tesla",
                        "vehicle",
                        "car"
                    ]
                ):
                    continue

            if intent == "reward":

                if not any(
                    term in text_lower
                    for term in [
                        "laptop",
                        "reward",
                        "gift"
                    ]
                ):
                    continue

            if score >= threshold:

                verified_results.append(
                    result
                )

        if not verified_results:

            return []

        return verified_results[
            :top_k
        ]


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("ACADEMIARAG PRO V5 TEST")
    print("ADAPTIVE SELF-LEARNING RAG")
    print("=" * 70)

    engine = LocalRAGEngine()

    if not engine.load_model():

        print(
            "Model loading failed."
        )

        raise SystemExit

    if not engine.build_vector_db():

        print(
            "Vector database build failed."
        )

        raise SystemExit

    questions = [

        "What percentage of attendance is required in each course?",

        "How often is attendance reviewed?",

        "Can attendance shortage be condoned?",

        "What is the maximum attendance shortage that can be condoned?",

        "Who receives information about the student's attendance status?",

        "What are supplementary examinations?",

        "Does Vignan University provide a Tesla car to students with 100% attendance?",

        "Does Vignan University give students a laptop reward?"
    ]

    for question in questions:

        print("\n")
        print("-" * 70)

        print(
            "QUESTION:"
        )

        print(
            question
        )

        results = (
            engine.semantic_search(
                question,
                top_k=5,
                threshold=0.30
            )
        )

        print("\nANSWER:")

        answers = (
            engine.extract_answer_sentences(
                question,
                results,
                max_sentences=2
            )
        )

        if answers:

            for answer in answers:

                print(
                    f"• {answer['text']}"
                )

            print(
                f"\nSOURCE: "
                f"{answers[0]['source']} "
                f"| Page "
                f"{answers[0]['page']}"
            )

        else:

            print(
                "I could not find reliable "
                "evidence for that question "
                "in the official university "
                "documents."
            )

    print("\n")
    print("=" * 70)
    print("V5 TEST COMPLETED")
    print("=" * 70)