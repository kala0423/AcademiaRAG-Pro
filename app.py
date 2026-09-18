# ================================================================
# AcademiaRAG Pro V7.3
# Enterprise University Knowledge Suite
# GUI connected directly to rag_engine.py
# ================================================================

import os
import threading
import shutil
from pathlib import Path

import customtkinter as ctk

from tkinter import filedialog, messagebox

import rag_engine


# ================================================================
# UI SETTINGS
# ================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ================================================================
# PATH CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "university_docs"


# ================================================================
# MAIN APPLICATION
# ================================================================

class UniversityRAGApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        # --------------------------------------------------------
        # WINDOW
        # --------------------------------------------------------

        self.title(
            "AcademiaRAG Pro v7.3 - Enterprise Knowledge Suite"
        )

        self.geometry(
            "1150x780"
        )

        self.minsize(
            1000,
            700
        )

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

        # --------------------------------------------------------
        # CREATE DOCUMENT FOLDER
        # --------------------------------------------------------

        DOCS_DIR.mkdir(
            exist_ok=True
        )

        # --------------------------------------------------------
        # APPLICATION STATE
        # --------------------------------------------------------

        self.model_ready = False
        self.index_ready = False
        self.indexing = False
        self.searching = False

        # --------------------------------------------------------
        # CURRENT ANSWER
        # --------------------------------------------------------

        self.current_question = None
        self.current_answer = None
        self.current_source = None
        self.current_page = None
        self.current_result = None

        # --------------------------------------------------------
        # BUILD GUI
        # --------------------------------------------------------

        self.create_interface()

        # --------------------------------------------------------
        # START BACKEND
        # --------------------------------------------------------

        self.after(
            300,
            self.start_backend
        )


    # ============================================================
    # CREATE INTERFACE
    # ============================================================

    def create_interface(self):

        # ========================================================
        # LEFT SIDEBAR
        # ========================================================

        self.left_panel = ctk.CTkFrame(
            self,
            width=285,
            corner_radius=0
        )

        self.left_panel.pack(
            side="left",
            fill="y"
        )

        self.left_panel.pack_propagate(
            False
        )


        # ========================================================
        # BRANDING
        # ========================================================

        self.panel_title = ctk.CTkLabel(
            self.left_panel,
            text="ACADEMIARAG PRO",
            font=ctk.CTkFont(
                size=21,
                weight="bold"
            )
        )

        self.panel_title.pack(
            pady=(28, 3)
        )


        self.panel_subtitle = ctk.CTkLabel(
            self.left_panel,
            text="Enterprise Knowledge Suite",
            text_color="#888888",
            font=ctk.CTkFont(
                size=11
            )
        )

        self.panel_subtitle.pack(
            pady=(0, 28)
        )


        # ========================================================
        # KNOWLEDGE BASE
        # ========================================================

        self.knowledge_title = ctk.CTkLabel(
            self.left_panel,
            text="KNOWLEDGE BASE",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        self.knowledge_title.pack(
            pady=(0, 8),
            padx=20,
            anchor="w"
        )


        # --------------------------------------------------------
        # SELECT PDF
        # --------------------------------------------------------

        self.import_btn = ctk.CTkButton(
            self.left_panel,
            text="Select PDF Document",
            height=38,
            command=self.import_pdf_file
        )

        self.import_btn.pack(
            pady=6,
            padx=20,
            fill="x"
        )


        # --------------------------------------------------------
        # BUILD DATABASE
        # --------------------------------------------------------

        self.index_btn = ctk.CTkButton(
            self.left_panel,
            text="Build Vector Database",
            height=38,
            fg_color="#2b7a4b",
            hover_color="#1e5433",
            command=self.start_indexing_thread
        )

        self.index_btn.pack(
            pady=6,
            padx=20,
            fill="x"
        )


        # ========================================================
        # DOCUMENT STATUS
        # ========================================================

        self.document_title = ctk.CTkLabel(
            self.left_panel,
            text="DOCUMENT STATUS",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        self.document_title.pack(
            pady=(25, 5),
            padx=20,
            anchor="w"
        )


        self.document_lbl = ctk.CTkLabel(
            self.left_panel,
            text="No document indexed",
            text_color="#aaaaaa",
            wraplength=235,
            justify="left"
        )

        self.document_lbl.pack(
            pady=5,
            padx=20,
            anchor="w"
        )


        # ========================================================
        # SELF LEARNING
        # ========================================================

        self.learning_title = ctk.CTkLabel(
            self.left_panel,
            text="SELF-LEARNING",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        self.learning_title.pack(
            pady=(25, 5),
            padx=20,
            anchor="w"
        )


        self.learning_lbl = ctk.CTkLabel(
            self.left_panel,
            text="Interactions: 0\n"
                 "Helpful: 0\n"
                 "Not Helpful: 0\n"
                 "Corrections: 0",
            text_color="#aaaaaa",
            wraplength=235,
            justify="left"
        )

        self.learning_lbl.pack(
            pady=5,
            padx=20,
            anchor="w"
        )


        # ========================================================
        # SYSTEM STATUS
        # ========================================================

        self.status_title = ctk.CTkLabel(
            self.left_panel,
            text="SYSTEM STATUS",
            font=ctk.CTkFont(
                size=13,
                weight="bold"
            )
        )

        self.status_title.pack(
            pady=(25, 5),
            padx=20,
            anchor="w"
        )


        self.status_lbl = ctk.CTkLabel(
            self.left_panel,
            text="Initializing...",
            text_color="#aaaaaa",
            wraplength=235,
            justify="left"
        )

        self.status_lbl.pack(
            pady=5,
            padx=20,
            anchor="w"
        )


        # ========================================================
        # TECHNOLOGY STACK
        # ========================================================

        self.info_lbl = ctk.CTkLabel(
            self.left_panel,
            text=(
                "ADAPTIVE LOCAL RAG\n\n"
                "• Official University PDF\n"
                "• PDF Text Extraction\n"
                "• SentenceTransformer\n"
                "• FAISS Semantic Search\n"
                "• CrossEncoder Reranking\n"
                "• Intent Detection\n"
                "• Evidence Verification\n"
                "• Answer Extraction\n"
                "• Learning Memory\n"
                "• User Feedback"
            ),
            text_color="#777777",
            justify="left",
            wraplength=235
        )

        self.info_lbl.pack(
            side="bottom",
            pady=20,
            padx=20,
            anchor="w"
        )


        # ========================================================
        # MAIN WORKSPACE
        # ========================================================

        self.main_workspace = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.main_workspace.pack(
            side="right",
            fill="both",
            expand=True,
            padx=22,
            pady=20
        )


        # ========================================================
        # HEADER
        # ========================================================

        self.app_header = ctk.CTkLabel(
            self.main_workspace,
            text="ACADEMIC REGULATIONS ASSISTANT",
            font=ctk.CTkFont(
                size=23,
                weight="bold"
            )
        )

        self.app_header.pack(
            pady=(0, 3)
        )


        self.sub_header = ctk.CTkLabel(
            self.main_workspace,
            text=(
                "Feedback-Driven Institutional "
                "Knowledge Retrieval"
            ),
            text_color="#888888",
            font=ctk.CTkFont(
                size=12
            )
        )

        self.sub_header.pack(
            pady=(0, 15)
        )


        # ========================================================
        # CHAT CONSOLE
        # ========================================================

        self.chat_console = ctk.CTkTextbox(
            self.main_workspace,
            font=ctk.CTkFont(
                size=13
            ),
            wrap="word",
            corner_radius=10
        )

        self.chat_console.pack(
            fill="both",
            expand=True,
            pady=(0, 10)
        )


        self.chat_console.insert(
            "0.0",
            (
                "ACADEMIARAG PRO V7.3\n"
                "=====================\n\n"
                "RAG Advisor\n\n"
                "Welcome to AcademiaRAG Pro.\n\n"
                "This system answers questions "
                "using official university "
                "regulation documents stored "
                "locally on your computer.\n\n"
                "Backend initialization is in progress...\n"
            )
        )

        self.chat_console.configure(
            state="disabled"
        )


        # ========================================================
        # FEEDBACK BAR
        # ========================================================

        self.feedback_frame = ctk.CTkFrame(
            self.main_workspace,
            fg_color="transparent"
        )

        self.feedback_frame.pack(
            fill="x",
            pady=(0, 10)
        )


        # --------------------------------------------------------
        # HELPFUL
        # --------------------------------------------------------

        self.helpful_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Helpful",
            width=120,
            height=34,
            fg_color="#2b7a4b",
            hover_color="#1e5433",
            command=self.mark_helpful,
            state="disabled"
        )

        self.helpful_btn.pack(
            side="left",
            padx=(0, 8)
        )


        # --------------------------------------------------------
        # NOT HELPFUL
        # --------------------------------------------------------

        self.not_helpful_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Not Helpful",
            width=120,
            height=34,
            fg_color="#8b3a3a",
            hover_color="#642727",
            command=self.mark_not_helpful,
            state="disabled"
        )

        self.not_helpful_btn.pack(
            side="left"
        )


        # --------------------------------------------------------
        # CLEAR
        # --------------------------------------------------------

        self.clear_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Clear",
            width=100,
            height=34,
            command=self.clear_console
        )

        self.clear_btn.pack(
            side="left",
            padx=8
        )


        # --------------------------------------------------------
        # LEARNING STATS
        # --------------------------------------------------------

        self.learning_stats_btn = ctk.CTkButton(
            self.feedback_frame,
            text="Learning Stats",
            width=135,
            height=34,
            command=self.show_learning_stats
        )

        self.learning_stats_btn.pack(
            side="right"
        )


        # ========================================================
        # QUESTION AREA
        # ========================================================

        self.interaction_frame = ctk.CTkFrame(
            self.main_workspace,
            fg_color="transparent"
        )

        self.interaction_frame.pack(
            fill="x"
        )


        # --------------------------------------------------------
        # QUESTION ENTRY
        # --------------------------------------------------------

        self.query_entry = ctk.CTkEntry(
            self.interaction_frame,
            placeholder_text=(
                "Ask about attendance, "
                "R grade, supplementary exams..."
            ),
            height=42,
            font=ctk.CTkFont(
                size=13
            )
        )

        self.query_entry.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(0, 10)
        )


        self.query_entry.bind(
            "<Return>",
            lambda event: self.execute_search()
        )


        # --------------------------------------------------------
        # ASK BUTTON
        # --------------------------------------------------------

        self.submit_btn = ctk.CTkButton(
            self.interaction_frame,
            text="Ask Assistant",
            width=145,
            height=42,
            command=self.execute_search,
            state="disabled"
        )

        self.submit_btn.pack(
            side="right"
        )


    # ============================================================
    # CONSOLE OUTPUT
    # ============================================================

    def print_to_console(
        self,
        emitter,
        message
    ):

        self.chat_console.configure(
            state="normal"
        )

        self.chat_console.insert(
            "end",
            f"\n\n{emitter}\n"
            f"{'=' * len(emitter)}\n\n"
        )

        self.chat_console.insert(
            "end",
            f"{message}\n"
        )

        self.chat_console.see(
            "end"
        )

        self.chat_console.configure(
            state="disabled"
        )


    # ============================================================
    # CLEAR CONSOLE
    # ============================================================

    def clear_console(self):

        self.chat_console.configure(
            state="normal"
        )

        self.chat_console.delete(
            "0.0",
            "end"
        )

        self.chat_console.insert(
            "0.0",
            (
                "ACADEMIARAG PRO V7.3\n"
                "=====================\n\n"
                "RAG Advisor\n\n"
                "Console cleared.\n\n"
                "Ask a question about the "
                "university regulations."
            )
        )

        self.chat_console.configure(
            state="disabled"
        )


    # ============================================================
    # BACKEND STARTUP
    # ============================================================

    def start_backend(self):

        self.status_lbl.configure(
            text="Loading local AI models..."
        )

        self.import_btn.configure(
            state="disabled"
        )

        self.index_btn.configure(
            state="disabled"
        )

        self.submit_btn.configure(
            state="disabled"
        )

        threading.Thread(
            target=self.load_backend_thread,
            daemon=True
        ).start()


    # ============================================================
    # LOAD BACKEND
    # ============================================================

    def load_backend_thread(self):

        try:

            rag_engine.load_models()
            rag_engine.init_database()

            success = True

        except Exception as e:

            print(
                "BACKEND ERROR:",
                e
            )

            success = False

            error_message = str(e)

        if success:

            self.after(
                0,
                self.backend_ready
            )

        else:

            self.after(
                0,
                lambda: self.backend_failed(
                    error_message
                )
            )


    # ============================================================
    # BACKEND READY
    # ============================================================

    def backend_ready(self):

        self.model_ready = True

        self.import_btn.configure(
            state="normal"
        )

        self.index_btn.configure(
            state="normal"
        )

        self.submit_btn.configure(
            state="disabled"
        )

        self.status_lbl.configure(
            text=(
                "AI models ready.\n"
                "Import official PDF and "
                "build vector database."
            )
        )

        self.update_learning_stats()

        self.print_to_console(
            "RAG Advisor",
            (
                "Local AI models loaded successfully.\n\n"
                "Embedding model: all-MiniLM-L6-v2\n"
                "Reranker: ms-marco-MiniLM-L6-v2\n\n"
                "Next step:\n"
                "Select the official university PDF "
                "and build the vector database."
            )
        )


    # ============================================================
    # BACKEND FAILED
    # ============================================================

    def backend_failed(
        self,
        error
    ):

        self.status_lbl.configure(
            text="AI model loading failed."
        )

        messagebox.showerror(
            "Backend Error",
            (
                "The local AI backend could not "
                "be initialized.\n\n"
                f"Error:\n{error}"
            )
        )


    # ============================================================
    # IMPORT PDF
    # ============================================================

    def import_pdf_file(self):

        file_path = filedialog.askopenfilename(
            title="Select Official University PDF",
            filetypes=[
                (
                    "PDF Documents",
                    "*.pdf"
                )
            ]
        )

        if not file_path:
            return

        filename = os.path.basename(
            file_path
        )

        destination = DOCS_DIR / filename

        try:

            shutil.copy2(
                file_path,
                destination
            )

            self.document_lbl.configure(
                text=(
                    f"Selected:\n"
                    f"{filename}\n\n"
                    "Ready for indexing."
                )
            )

            self.status_lbl.configure(
                text=(
                    "PDF imported.\n"
                    "Build the vector database."
                )
            )

            self.print_to_console(
                "Knowledge Base",
                (
                    "PDF imported successfully.\n\n"
                    f"Document: {filename}\n"
                    f"Location: {destination}"
                )
            )

            messagebox.showinfo(
                "PDF Imported",
                (
                    f"{filename}\n\n"
                    "has been added to the "
                    "university_docs folder."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Import Error",
                str(e)
            )


    # ============================================================
    # BUILD VECTOR DATABASE
    # ============================================================

    def start_indexing_thread(self):

        if not self.model_ready:

            messagebox.showwarning(
                "Model Not Ready",
                "Please wait for the AI models to load."
            )

            return


        # --------------------------------------------------------
        # Find PDFs
        # --------------------------------------------------------

        pdf_files = list(
            DOCS_DIR.glob("*.pdf")
        )

        if not pdf_files:

            messagebox.showwarning(
                "No PDF Found",
                (
                    "No PDF document was found.\n\n"
                    "Use 'Select PDF Document' first."
                )
            )

            return


        # --------------------------------------------------------
        # Prefer R22 regulation
        # --------------------------------------------------------

        selected_pdf = None

        for pdf in pdf_files:

            if "R22" in pdf.name.upper():

                selected_pdf = pdf

                break


        if selected_pdf is None:

            selected_pdf = pdf_files[0]


        self.indexing = True

        self.index_btn.configure(
            state="disabled"
        )

        self.import_btn.configure(
            state="disabled"
        )

        self.submit_btn.configure(
            state="disabled"
        )

        self.status_lbl.configure(
            text=(
                "Building vector database...\n"
                "Extracting PDF and generating embeddings."
            )
        )

        self.document_lbl.configure(
            text=(
                f"Indexing:\n"
                f"{selected_pdf.name}"
            )
        )

        self.print_to_console(
            "Knowledge Base",
            (
                "Vector database construction started.\n\n"
                f"Document: {selected_pdf.name}\n\n"
                "Pipeline:\n"
                "1. PDF extraction\n"
                "2. Text cleaning\n"
                "3. Chunk creation\n"
                "4. SentenceTransformer embeddings\n"
                "5. FAISS indexing"
            )
        )

        threading.Thread(
            target=self.build_database_thread,
            args=(selected_pdf,),
            daemon=True
        ).start()


    # ============================================================
    # BUILD DATABASE WORKER
    # ============================================================

    def build_database_thread(
        self,
        pdf_path
    ):

        try:

            info = rag_engine.build_knowledge_base(
                pdf_path
            )

            success = True

        except Exception as e:

            print(
                "INDEXING ERROR:",
                e
            )

            success = False
            error_message = str(e)
            info = None

        self.after(
            0,
            lambda: self.indexing_finished(
                success,
                info,
                error_message
                if not success
                else None
            )
        )


    # ============================================================
    # INDEXING FINISHED
    # ============================================================

    def indexing_finished(
        self,
        success,
        info,
        error
    ):

        self.indexing = False

        self.index_btn.configure(
            state="normal"
        )

        self.import_btn.configure(
            state="normal"
        )


        if success:

            self.index_ready = True

            chunks = info.get(
                "chunks",
                0
            )

            dimension = info.get(
                "dimension",
                0
            )

            pdf_files = list(
                DOCS_DIR.glob("*.pdf")
            )

            if pdf_files:

                selected_pdf = pdf_files[0]

                for pdf in pdf_files:

                    if "R22" in pdf.name.upper():

                        selected_pdf = pdf

                        break

                document_name = selected_pdf.name

            else:

                document_name = "University PDF"


            self.document_lbl.configure(
                text=(
                    f"Document:\n"
                    f"{document_name}\n\n"
                    f"Chunks: {chunks}\n"
                    f"Vector dimension: {dimension}\n\n"
                    "FAISS: READY"
                )
            )

            self.status_lbl.configure(
                text=(
                    "Vector database ready.\n"
                    f"{chunks} chunks indexed."
                )
            )

            self.submit_btn.configure(
                state="normal"
            )

            self.print_to_console(
                "RAG Advisor",
                (
                    "Vector database built successfully.\n\n"
                    f"Indexed chunks: {chunks}\n"
                    f"Embedding dimension: {dimension}\n\n"
                    "Semantic retrieval: ACTIVE\n"
                    "CrossEncoder reranking: ACTIVE\n"
                    "Intent detection: ACTIVE\n"
                    "Evidence verification: ACTIVE\n"
                    "Learning memory: ACTIVE\n\n"
                    "The system is ready for questions."
                )
            )

            messagebox.showinfo(
                "Database Ready",
                (
                    "Vector database built successfully.\n\n"
                    f"Chunks indexed: {chunks}\n"
                    f"Embedding dimension: {dimension}"
                )
            )

        else:

            self.index_ready = False

            self.submit_btn.configure(
                state="disabled"
            )

            self.status_lbl.configure(
                text="Vector database build failed."
            )

            messagebox.showerror(
                "Build Failed",
                (
                    "The vector database could not "
                    "be built.\n\n"
                    f"Error:\n{error}"
                )
            )


    # ============================================================
    # SEARCH
    # ============================================================

    def execute_search(self):

        query = (
            self.query_entry
            .get()
            .strip()
        )

        if not query:

            messagebox.showwarning(
                "Question Required",
                "Please enter a question."
            )

            return


        if not self.model_ready:

            messagebox.showwarning(
                "Model Not Ready",
                "The AI models are still loading."
            )

            return


        if not self.index_ready:

            messagebox.showwarning(
                "Database Not Ready",
                (
                    "Please build the Vector Database "
                    "before asking questions."
                )
            )

            return


        if self.searching:

            return


        self.searching = True


        # --------------------------------------------------------
        # Store current question
        # --------------------------------------------------------

        self.current_question = query
        self.current_answer = None
        self.current_source = None
        self.current_page = None
        self.current_result = None


        # --------------------------------------------------------
        # Display question
        # --------------------------------------------------------

        self.print_to_console(
            "You",
            query
        )


        self.query_entry.delete(
            0,
            "end"
        )


        # --------------------------------------------------------
        # Disable controls
        # --------------------------------------------------------

        self.submit_btn.configure(
            state="disabled",
            text="Searching..."
        )

        self.helpful_btn.configure(
            state="disabled"
        )

        self.not_helpful_btn.configure(
            state="disabled"
        )

        self.status_lbl.configure(
            text="Searching knowledge base..."
        )


        # --------------------------------------------------------
        # Search in background
        # --------------------------------------------------------

        threading.Thread(
            target=self.search_thread,
            args=(query,),
            daemon=True
        ).start()


    # ============================================================
    # SEARCH WORKER
    # ============================================================

    def search_thread(
        self,
        query
    ):

        try:

            result = rag_engine.semantic_search(
                query
            )

            success = True

        except Exception as e:

            print(
                "SEARCH ERROR:",
                e
            )

            result = None
            success = False
            error_message = str(e)

        self.after(
            0,
            lambda: self.display_search_result(
                query,
                result,
                success,
                error_message
                if not success
                else None
            )
        )


    # ============================================================
    # DISPLAY SEARCH RESULT
    # ============================================================

    def display_search_result(
        self,
        query,
        result,
        success,
        error
    ):

        self.searching = False

        self.submit_btn.configure(
            state="normal",
            text="Ask Assistant"
        )


        # ========================================================
        # ERROR
        # ========================================================

        if not success:

            self.status_lbl.configure(
                text="Search failed."
            )

            self.print_to_console(
                "RAG Advisor",
                (
                    "An error occurred while "
                    "processing the question.\n\n"
                    f"Error:\n{error}"
                )
            )

            return


        # ========================================================
        # SAVE RESULT
        # ========================================================

        self.current_result = result


        # ========================================================
        # NO VERIFIED ANSWER
        # ========================================================

        if not result:

            self.show_no_answer()

            return


        answer = result.get(
            "answer"
        )

        verified = result.get(
            "verified",
            False
        )

        score = result.get(
            "score",
            0
        )

        intent = result.get(
            "intent",
            "general"
        )

        source = result.get(
            "source"
        )

        page = result.get(
            "page"
        )

        learned = result.get(
            "learned",
            False
        )

        learning_similarity = result.get(
            "learning_similarity",
            0
        )


        # ========================================================
        # SAFETY CHECK
        # ========================================================

        if not verified or not answer:

            self.current_answer = None

            self.current_source = None

            self.current_page = None

            self.show_no_answer(
                score=score,
                intent=intent
            )

            return


        # ========================================================
        # SAVE CURRENT ANSWER
        # ========================================================

        self.current_answer = answer
        self.current_source = source
        self.current_page = page


        # ========================================================
        # BUILD OUTPUT
        # ========================================================

        output = (
            "ANSWER\n"
            "======\n\n"
        )

        output += (
            f"{answer}\n"
        )


        # ========================================================
        # SOURCE
        # ========================================================

        output += (
            "\n\nSOURCE\n"
            "======\n\n"
        )

        if source:

            output += (
                f"Document: {source}\n"
            )

        if page:

            output += (
                f"Page: {page}\n"
            )


        # ========================================================
        # RETRIEVAL INFORMATION
        # ========================================================

        output += (
            "\nRETRIEVAL INFORMATION\n"
            "======================\n\n"
            f"Confidence Score: {score:.3f}\n"
            f"Intent: {intent}\n"
            f"Evidence Verified: "
            f"{'Yes' if verified else 'No'}\n"
        )


        # ========================================================
        # LEARNING INFORMATION
        # ========================================================

        output += (
            "\nADAPTIVE LEARNING\n"
            "=================\n\n"
        )

        if learned:

            output += (
                "Previously learned interaction detected.\n"
                f"Learning similarity: "
                f"{learning_similarity:.3f}\n"
                "Learned knowledge influenced retrieval."
            )

        else:

            output += (
                "New interaction detected.\n"
                "User feedback can improve future "
                "similar queries."
            )


        # ========================================================
        # SAFEGUARD
        # ========================================================

        output += (
            "\n\nRETRIEVAL SAFEGUARD\n"
            "===================\n\n"
            "The answer is grounded in the retrieved "
            "local university document.\n\n"
            "Feedback is stored separately as learning "
            "memory and does not modify the official PDF."
        )


        # ========================================================
        # DISPLAY
        # ========================================================

        self.status_lbl.configure(
            text="Search completed."
        )

        self.print_to_console(
            "RAG Advisor",
            output
        )


        # ========================================================
        # ENABLE FEEDBACK
        # ========================================================

        self.helpful_btn.configure(
            state="normal"
        )

        self.not_helpful_btn.configure(
            state="normal"
        )


        self.update_learning_stats()


    # ============================================================
    # NO ANSWER
    # ============================================================

    def show_no_answer(
        self,
        score=0,
        intent="general"
    ):

        self.current_answer = None
        self.current_source = None
        self.current_page = None

        self.status_lbl.configure(
            text="No verified answer found."
        )

        output = (
            "NO RELEVANT INFORMATION FOUND\n"
            "==============================\n\n"
            "I could not find sufficiently reliable "
            "evidence for this question in the "
            "indexed university documents.\n\n"
            "No unsupported answer was generated.\n\n"
            f"Retrieval score: {score:.3f}\n"
            f"Detected intent: {intent}\n\n"
            "Try asking the question using terminology "
            "that appears in the university regulations."
        )

        self.print_to_console(
            "RAG Advisor",
            output
        )


    # ============================================================
    # HELPFUL FEEDBACK
    # ============================================================

    def mark_helpful(self):

        if not self.current_question:
            return

        if not self.current_answer:
            return

        try:

            rag_engine.save_feedback(
                question=self.current_question,
                answer=self.current_answer,
                source=self.current_source,
                page=self.current_page,
                helpful=1,
                correction=""
            )

            self.helpful_btn.configure(
                state="disabled"
            )

            self.not_helpful_btn.configure(
                state="disabled"
            )

            self.print_to_console(
                "Learning System",
                (
                    "Helpful feedback recorded.\n\n"
                    "This interaction has been added "
                    "to the local adaptive learning memory."
                )
            )

            self.update_learning_stats()

        except Exception as e:

            messagebox.showerror(
                "Feedback Error",
                str(e)
            )


    # ============================================================
    # NOT HELPFUL
    # ============================================================

    def mark_not_helpful(self):

        if not self.current_question:
            return

        if not self.current_answer:
            return


        correction_dialog = ctk.CTkInputDialog(
            text=(
                "Enter the corrected answer.\n\n"
                "Leave blank if you only want to "
                "record negative feedback."
            ),
            title="Improve AcademiaRAG"
        )


        correction = (
            correction_dialog
            .get_input()
        )


        if correction is None:

            correction = ""


        try:

            rag_engine.save_feedback(
                question=self.current_question,
                answer=self.current_answer,
                source=self.current_source,
                page=self.current_page,
                helpful=0,
                correction=correction
            )

            self.helpful_btn.configure(
                state="disabled"
            )

            self.not_helpful_btn.configure(
                state="disabled"
            )


            if correction:

                message = (
                    "Correction stored successfully.\n\n"
                    "The correction is stored separately "
                    "in the adaptive learning memory."
                )

            else:

                message = (
                    "Negative feedback recorded.\n\n"
                    "No correction was provided."
                )


            self.print_to_console(
                "Learning System",
                message
            )

            self.update_learning_stats()

        except Exception as e:

            messagebox.showerror(
                "Feedback Error",
                str(e)
            )


    # ============================================================
    # UPDATE LEARNING STATS
    # ============================================================

    def update_learning_stats(self):

        try:

            total = rag_engine.get_learning_count()

            # ----------------------------------------------------
            # Read detailed statistics directly from SQLite
            # ----------------------------------------------------

            import sqlite3

            conn = sqlite3.connect(
                rag_engine.DB_PATH
            )

            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN helpful = 1
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN helpful = 0
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN correction IS NOT NULL
                                AND correction != ''
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    )
                FROM feedback_memory
                """
            ).fetchone()

            conn.close()


            total = row[0]
            helpful = row[1]
            not_helpful = row[2]
            corrections = row[3]


            self.learning_lbl.configure(
                text=(
                    f"Interactions: {total}\n"
                    f"Helpful: {helpful}\n"
                    f"Not Helpful: {not_helpful}\n"
                    f"Corrections: {corrections}\n\n"
                    "Adaptive memory: ACTIVE"
                )
            )

        except Exception as e:

            print(
                "LEARNING STATS ERROR:",
                e
            )

            self.learning_lbl.configure(
                text=(
                    "Adaptive memory: ACTIVE"
                )
            )


    # ============================================================
    # LEARNING STATISTICS WINDOW
    # ============================================================

    def show_learning_stats(self):

        try:

            import sqlite3

            conn = sqlite3.connect(
                rag_engine.DB_PATH
            )

            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN helpful = 1
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN helpful = 0
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    ),
                    COALESCE(
                        SUM(
                            CASE
                                WHEN correction IS NOT NULL
                                AND correction != ''
                                THEN 1
                                ELSE 0
                            END
                        ),
                        0
                    )
                FROM feedback_memory
                """
            ).fetchone()

            conn.close()


            total = row[0]
            helpful = row[1]
            not_helpful = row[2]
            corrections = row[3]


            messagebox.showinfo(
                "AcademiaRAG Learning Statistics",
                (
                    "SELF-LEARNING STATUS\n\n"
                    f"Total interactions: {total}\n\n"
                    f"Helpful responses: {helpful}\n\n"
                    f"Not helpful responses: "
                    f"{not_helpful}\n\n"
                    f"Corrections: {corrections}\n\n"
                    "Learning memory is stored locally "
                    "in SQLite.\n\n"
                    "The official university PDF "
                    "is never modified by feedback."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Learning Statistics",
                str(e)
            )


    # ============================================================
    # CLOSE APPLICATION
    # ============================================================

    def close_application(self):

        self.destroy()


# ================================================================
# APPLICATION START
# ================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ACADEMIARAG PRO V7.3")
    print("ENTERPRISE UNIVERSITY KNOWLEDGE SUITE")
    print("=" * 70)

    app = UniversityRAGApp()

    app.mainloop()