# ============================================================
# AcademiaRAG Pro V5
# Adaptive Self-Learning GUI
# ============================================================

import os
import threading

import customtkinter as ctk

from tkinter import filedialog, messagebox

from rag_engine import LocalRAGEngine


# ============================================================
# UI SETTINGS
# ============================================================

ctk.set_appearance_mode(
    "Dark"
)

ctk.set_default_color_theme(
    "blue"
)


# ============================================================
# MAIN APPLICATION
# ============================================================

class UniversityRAGApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "🎓 Academia RAG Enterprise Advisor v2.0"
        )

        self.geometry(
            "1000x720"
        )

        self.resizable(
            False,
            False
        )

        # ----------------------------------------------------
        # BACKEND
        # ----------------------------------------------------

        self.rag_engine = (
            LocalRAGEngine()
        )

        self.target_folder = (
            "university_docs"
        )

        if not os.path.exists(
            self.target_folder
        ):

            os.makedirs(
                self.target_folder
            )

        # Current answer information
        self.current_question = None
        self.current_answer = None
        self.current_source = None
        self.current_page = None

        self.create_interface()

        self.after(
            300,
            self.start_model_loading
        )

    # ========================================================
    # INTERFACE
    # ========================================================

    def create_interface(self):

        # ----------------------------------------------------
        # SIDEBAR
        # ----------------------------------------------------

        self.left_panel = ctk.CTkFrame(
            self,
            width=250,
            corner_radius=0
        )

        self.left_panel.pack(
            side="left",
            fill="y"
        )

        self.left_panel.pack_propagate(
            False
        )

        self.panel_title = ctk.CTkLabel(
            self.left_panel,
            text="KNOWLEDGE BASE",
            font=ctk.CTkFont(
                size=16,
                weight="bold"
            )
        )

        self.panel_title.pack(
            pady=20
        )

        # ----------------------------------------------------
        # IMPORT
        # ----------------------------------------------------

        self.import_btn = ctk.CTkButton(
            self.left_panel,
            text="📁 Select PDF Document",
            command=self.import_pdf_file
        )

        self.import_btn.pack(
            pady=15,
            padx=20,
            fill="x"
        )

        # ----------------------------------------------------
        # BUILD
        # ----------------------------------------------------

        self.index_btn = ctk.CTkButton(
            self.left_panel,
            text="⚡ Build Vector Database",
            fg_color="#2b7a4b",
            hover_color="#1e5433",
            command=self.start_indexing_thread
        )

        self.index_btn.pack(
            pady=10,
            padx=20,
            fill="x"
        )

        # ----------------------------------------------------
        # LEARNING STATUS
        # ----------------------------------------------------

        self.learning_title = ctk.CTkLabel(
            self.left_panel,
            text="Self-Learning",
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
            text="Learning memory: 0",
            text_color="#aaaaaa",
            wraplength=210,
            justify="left"
        )

        self.learning_lbl.pack(
            pady=5,
            padx=20,
            anchor="w"
        )

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        self.status_title = ctk.CTkLabel(
            self.left_panel,
            text="System Status",
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
            wraplength=210,
            justify="left"
        )

        self.status_lbl.pack(
            pady=5,
            padx=20,
            anchor="w"
        )

        # ----------------------------------------------------
        # DESCRIPTION
        # ----------------------------------------------------

        self.info_lbl = ctk.CTkLabel(
            self.left_panel,
            text=(
                "ADAPTIVE LOCAL RAG\n\n"
                "✓ PDF extraction\n"
                "✓ MiniLM embeddings\n"
                "✓ FAISS retrieval\n"
                "✓ CrossEncoder reranking\n"
                "✓ Clean answer extraction\n"
                "✓ Source verification\n"
                "✓ Learning memory\n"
                "✓ User feedback"
            ),
            text_color="#777777",
            justify="left",
            wraplength=210
        )

        self.info_lbl.pack(
            side="bottom",
            pady=25,
            padx=20,
            anchor="w"
        )

        # ----------------------------------------------------
        # MAIN AREA
        # ----------------------------------------------------

        self.main_workspace = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        self.main_workspace.pack(
            side="right",
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        self.app_header = ctk.CTkLabel(
            self.main_workspace,
            text=(
                "ACADEMIC REGULATIONS "
                "SELF-LEARNING RAG SYSTEM"
            ),
            font=ctk.CTkFont(
                size=20,
                weight="bold"
            )
        )

        self.app_header.pack(
            pady=(0, 4)
        )

        self.sub_header = ctk.CTkLabel(
            self.main_workspace,
            text=(
                "Feedback-Driven Institutional "
                "Knowledge Retrieval"
            ),
            text_color="#888888"
        )

        self.sub_header.pack(
            pady=(0, 15)
        )

        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

        self.chat_console = ctk.CTkTextbox(
            self.main_workspace,
            font=ctk.CTkFont(
                size=13
            )
        )

        self.chat_console.pack(
            fill="both",
            expand=True,
            pady=(0, 10)
        )

        self.chat_console.insert(
            "0.0",
            (
                "🤖 RAG Advisor:\n\n"
                "Welcome to AcademiaRAG Pro.\n\n"
                "This system uses official "
                "institutional documents as its "
                "primary knowledge source.\n\n"
                "The system also learns from "
                "verified user feedback and "
                "corrections.\n\n"
            )
        )

        self.chat_console.configure(
            state="disabled"
        )

        # ----------------------------------------------------
        # FEEDBACK BUTTONS
        # ----------------------------------------------------

        self.feedback_frame = ctk.CTkFrame(
            self.main_workspace,
            fg_color="transparent"
        )

        self.feedback_frame.pack(
            fill="x",
            pady=(0, 10)
        )

        self.helpful_btn = ctk.CTkButton(
            self.feedback_frame,
            text="👍 Helpful",
            width=120,
            fg_color="#2b7a4b",
            hover_color="#1e5433",
            command=self.mark_helpful,
            state="disabled"
        )

        self.helpful_btn.pack(
            side="left",
            padx=(0, 8)
        )

        self.not_helpful_btn = ctk.CTkButton(
            self.feedback_frame,
            text="👎 Not Helpful",
            width=120,
            fg_color="#8b3a3a",
            hover_color="#642727",
            command=self.mark_not_helpful,
            state="disabled"
        )

        self.not_helpful_btn.pack(
            side="left"
        )

        self.learning_stats_btn = ctk.CTkButton(
            self.feedback_frame,
            text="🧠 Learning Stats",
            width=140,
            command=self.show_learning_stats
        )

        self.learning_stats_btn.pack(
            side="right"
        )

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        self.interaction_frame = ctk.CTkFrame(
            self.main_workspace,
            fg_color="transparent"
        )

        self.interaction_frame.pack(
            fill="x"
        )

        self.query_entry = ctk.CTkEntry(
            self.interaction_frame,
            placeholder_text=(
                "Ask a question about "
                "university regulations..."
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
            lambda event:
                self.execute_search()
        )

        self.submit_btn = ctk.CTkButton(
            self.interaction_frame,
            text="Ask Assistant",
            width=130,
            command=self.execute_search
        )

        self.submit_btn.pack(
            side="right"
        )

    # ========================================================
    # CONSOLE
    # ========================================================

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
            f"\n{emitter}: {message}\n"
        )

        self.chat_console.see(
            "end"
        )

        self.chat_console.configure(
            state="disabled"
        )

    # ========================================================
    # MODEL
    # ========================================================

    def start_model_loading(self):

        self.status_lbl.configure(
            text=(
                "⏳ Loading local AI models..."
            )
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
            target=self.load_model_background,
            daemon=True
        ).start()

    def load_model_background(self):

        success = (
            self.rag_engine.load_model()
        )

        self.after(
            0,
            lambda:
                self.model_loaded_ui(
                    success
                )
        )

    def model_loaded_ui(
        self,
        success
    ):

        if success:

            self.status_lbl.configure(
                text=(
                    "✅ Local AI models ready.\n"
                    "Upload your PDF documents."
                )
            )

            self.import_btn.configure(
                state="normal"
            )

            self.index_btn.configure(
                state="normal"
            )

            self.submit_btn.configure(
                state="normal"
            )

            self.update_learning_stats()

            self.print_to_console(
                "🤖 RAG Advisor",
                (
                    "Embedding model and "
                    "CrossEncoder loaded successfully."
                )
            )

        else:

            self.status_lbl.configure(
                text="❌ Model loading failed."
            )

            messagebox.showerror(
                "Model Error",
                (
                    "AI models could not be loaded.\n\n"
                    "Check the terminal."
                )
            )

    # ========================================================
    # PDF IMPORT
    # ========================================================

    def import_pdf_file(self):

        file_path = (
            filedialog.askopenfilename(
                filetypes=[
                    (
                        "PDF Documents",
                        "*.pdf"
                    )
                ]
            )
        )

        if not file_path:
            return

        filename = os.path.basename(
            file_path
        )

        destination = os.path.join(
            self.target_folder,
            filename
        )

        try:

            import shutil

            shutil.copy2(
                file_path,
                destination
            )

            self.status_lbl.configure(
                text=(
                    f"📄 Added:\n"
                    f"{filename}\n\n"
                    "Build the vector database."
                )
            )

            messagebox.showinfo(
                "PDF Imported",
                (
                    f"{filename}\n\n"
                    "was added to the local "
                    "knowledge base."
                )
            )

        except Exception as e:

            messagebox.showerror(
                "Import Error",
                str(e)
            )

    # ========================================================
    # BUILD DATABASE
    # ========================================================

    def start_indexing_thread(self):

        if not self.rag_engine.is_model_ready():

            messagebox.showwarning(
                "Model",
                "Local AI model is not ready."
            )

            return

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
                "⏳ Extracting PDF text...\n"
                "Generating embeddings..."
            )
        )

        threading.Thread(
            target=self.run_vector_indexing,
            daemon=True
        ).start()

    def run_vector_indexing(self):

        success = (
            self.rag_engine.build_vector_db(
                self.target_folder
            )
        )

        self.after(
            0,
            lambda:
                self.indexing_finished(
                    success
                )
        )

    def indexing_finished(
        self,
        success
    ):

        self.index_btn.configure(
            state="normal"
        )

        self.import_btn.configure(
            state="normal"
        )

        self.submit_btn.configure(
            state="normal"
        )

        if success:

            chunks = len(
                self.rag_engine.documents_metadata
            )

            self.status_lbl.configure(
                text=(
                    "🎯 FAISS Database Live!\n"
                    f"{chunks} chunks indexed."
                )
            )

            self.update_learning_stats()

            self.print_to_console(
                "🤖 RAG Advisor",
                (
                    "Vector database built successfully.\n"
                    f"Indexed chunks: {chunks}\n"
                    "Self-learning memory loaded.\n"
                    "Ready for questions."
                )
            )

            messagebox.showinfo(
                "Database Ready",
                (
                    "Local vector database "
                    "built successfully.\n\n"
                    f"Indexed chunks: {chunks}"
                )
            )

        else:

            self.status_lbl.configure(
                text="❌ Database build failed."
            )

            messagebox.showwarning(
                "Build Failed",
                (
                    "No readable PDF content "
                    "was found."
                )
            )

    # ========================================================
    # SEARCH
    # ========================================================

    def execute_search(self):

        query = (
            self.query_entry
            .get()
            .strip()
        )

        if not query:
            return

        if not self.rag_engine.is_model_ready():

            messagebox.showwarning(
                "Model",
                "AI model is not ready."
            )

            return

        if not self.rag_engine.is_index_ready():

            messagebox.showwarning(
                "Database",
                (
                    "Build the Vector Database "
                    "before asking questions."
                )
            )

            return

        self.print_to_console(
            "👤 You",
            query
        )

        self.query_entry.delete(
            0,
            "end"
        )

        self.submit_btn.configure(
            state="disabled"
        )

        self.helpful_btn.configure(
            state="disabled"
        )

        self.not_helpful_btn.configure(
            state="disabled"
        )

        self.status_lbl.configure(
            text="🔎 Searching..."
        )

        # Save current question
        self.current_question = query

        threading.Thread(
            target=self.run_search,
            args=(query,),
            daemon=True
        ).start()

    def run_search(
        self,
        query
    ):

        results = (
            self.rag_engine.semantic_search(
                query,
                top_k=5,
                threshold=0.30
            )
        )

        answer_sentences = (
            self.rag_engine
            .extract_answer_sentences(
                query,
                results,
                max_sentences=2
            )
        )

        self.after(
            0,
            lambda:
                self.display_results(
                    query,
                    results,
                    answer_sentences
                )
        )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    def display_results(
        self,
        query,
        results,
        answer_sentences
    ):

        self.submit_btn.configure(
            state="normal"
        )

        self.status_lbl.configure(
            text="🎯 Search completed."
        )

        if not results:

            self.current_answer = None
            self.current_source = None
            self.current_page = None

            self.print_to_console(
                "🤖 RAG Advisor",
                (
                    "I could not find reliable "
                    "evidence for that question "
                    "in the official university "
                    "documents.\n\n"
                    "No unsupported answer was generated."
                )
            )

            return

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        output = (
            "📌 Answer:\n\n"
        )

        if answer_sentences:

            for answer in answer_sentences:

                output += (
                    f"• {answer['text']}\n"
                )

            best_answer = (
                answer_sentences[0]
            )

            self.current_answer = (
                " ".join(
                    answer["text"]
                    for answer in answer_sentences
                )
            )

            self.current_source = (
                best_answer["source"]
            )

            self.current_page = (
                best_answer["page"]
            )

        else:

            self.current_answer = None
            self.current_source = None
            self.current_page = None

            output += (
                "Relevant passages were retrieved, "
                "but a reliable concise answer "
                "could not be extracted.\n"
            )

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        if (
            self.current_source
            and self.current_page
        ):

            output += (
                "\n\n📋 Primary Source:\n"
                f"📍 {self.current_source}\n"
                f"📄 Page {self.current_page}\n"
            )

        # ----------------------------------------------------
        # LEARNING STATUS
        # ----------------------------------------------------

        learned = any(
            result.get(
                "learned",
                False
            )
            for result in results
        )

        if learned:

            output += (
                "\n🧠 Learning Memory:\n"
                "This answer was retrieved from "
                "a previously verified interaction."
            )

        else:

            output += (
                "\n🧠 Learning Status:\n"
                "New answer — your feedback can "
                "help improve future retrieval."
            )

        # ----------------------------------------------------
        # SAFEGUARD
        # ----------------------------------------------------

        output += (
            "\n\n🔐 Retrieval Safeguard:\n"
            "Answers are grounded in retrieved "
            "local institutional documents. "
            "User feedback is stored separately "
            "as learning memory and does not "
            "rewrite official documents."
        )

        self.print_to_console(
            "🤖 RAG Advisor",
            output
        )

        # Enable feedback
        if self.current_answer:

            self.helpful_btn.configure(
                state="normal"
            )

            self.not_helpful_btn.configure(
                state="normal"
            )

    # ========================================================
    # HELPFUL
    # ========================================================

    def mark_helpful(self):

        if not self.current_question:
            return

        if not self.current_answer:
            return

        self.rag_engine.save_feedback(
            question=self.current_question,
            answer=self.current_answer,
            source=self.current_source,
            page=self.current_page,
            feedback="helpful"
        )

        self.helpful_btn.configure(
            state="disabled"
        )

        self.not_helpful_btn.configure(
            state="disabled"
        )

        self.print_to_console(
            "🧠 Learning System",
            (
                "✓ Feedback recorded.\n"
                "This verified interaction can "
                "be used to improve future retrieval."
            )
        )

        self.update_learning_stats()

    # ========================================================
    # NOT HELPFUL
    # ========================================================

    def mark_not_helpful(self):

        if not self.current_question:
            return

        if not self.current_answer:
            return

        correction = ctk.CTkInputDialog(
            text=(
                "How should the answer be corrected?\n\n"
                "Enter the correct answer, or press "
                "Cancel if you only want to record "
                "negative feedback."
            ),
            title="Improve AcademiaRAG"
        ).get_input()

        self.rag_engine.save_feedback(
            question=self.current_question,
            answer=self.current_answer,
            source=self.current_source,
            page=self.current_page,
            feedback="not_helpful",
            correction=correction
        )

        self.helpful_btn.configure(
            state="disabled"
        )

        self.not_helpful_btn.configure(
            state="disabled"
        )

        if correction:

            self.print_to_console(
                "🧠 Learning System",
                (
                    "✓ Correction stored.\n"
                    "The corrected answer can be "
                    "used for future similar questions."
                )
            )

        else:

            self.print_to_console(
                "🧠 Learning System",
                (
                    "✓ Negative feedback recorded.\n"
                    "No correction was added."
                )
            )

        self.update_learning_stats()

    # ========================================================
    # LEARNING STATS
    # ========================================================

    def update_learning_stats(self):

        try:

            stats = (
                self.rag_engine
                .get_learning_statistics()
            )

            self.learning_lbl.configure(
                text=(
                    f"Interactions: "
                    f"{stats['total']}\n"
                    f"Helpful: "
                    f"{stats['helpful']}\n"
                    f"Corrections: "
                    f"{stats['corrections']}\n\n"
                    "🧠 Adaptive memory active"
                )
            )

        except Exception:

            self.learning_lbl.configure(
                text="Learning memory: active"
            )

    def show_learning_stats(self):

        stats = (
            self.rag_engine
            .get_learning_statistics()
        )

        messagebox.showinfo(
            "AcademiaRAG Learning Statistics",
            (
                "🧠 SELF-LEARNING STATUS\n\n"
                f"Total interactions: "
                f"{stats['total']}\n\n"
                f"Helpful responses: "
                f"{stats['helpful']}\n\n"
                f"Not helpful responses: "
                f"{stats['not_helpful']}\n\n"
                f"Verified corrections: "
                f"{stats['corrections']}\n\n"
                "Learning memory is stored "
                "locally in SQLite."
            )
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = UniversityRAGApp()

    app.mainloop()