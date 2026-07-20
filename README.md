# 2026 F1 Spec RAG Agent

A Retrieval-Augmented Generation (RAG) agent specialized in the **2026 FIA Formula 1 Technical Regulations**. It provides hyper-specific technical answers with exact **Article Number citations** from the official FIA document.

## 🚀 Features
*   **Official Data Source:** Queries the latest 2026 FIA Technical Regulations.
*   **Exact Citations:** Every technical answer includes the corresponding Article Number (e.g., Article 3.2.1).
*   **Up-to-Date Answer Skill:** Detects "latest/current/update" style questions and injects live web snippets with URL citations.
*   **Dual Interface:** Includes both a **Terminal CLI** and a **Streamlit Web UI**.
*   **Powered by Gemini:** Uses Google Gemini models for high-quality technical reasoning.

## 🛠️ Setup

1.  **Clone or create the project directory.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure API Key:**
    Create a `.env` file in the root directory and add your Google Gemini API Key:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key_here
    ```
4.  **Ingest Regulations:**
    Run the ingestion script to download the PDF and build the vector database:
    ```bash
    python src/ingestion.py
    ```

## 🖥️ Usage

### Terminal CLI
Run the interactive terminal agent:
```bash
python src/cli.py
```

### Streamlit Web UI
Launch the browser-based interface:
```bash
streamlit run src/app.py
```

## 📁 Project Structure
*   `src/ingestion.py`: Downloads PDF and populates the ChromaDB vector store.
*   `src/rag_agent.py`: Core RAG pipeline with strict citation prompting.
*   `src/up_to_date_skill.py`: Live web enrichment skill for current-event and latest-update questions.
*   `src/cli.py`: Interactive command-line interface.
*   `src/app.py`: Streamlit-based web dashboard.
*   `vector_db/`: Persistent storage for regulatory document embeddings.
*   `data/`: Stores the official FIA PDF document.
