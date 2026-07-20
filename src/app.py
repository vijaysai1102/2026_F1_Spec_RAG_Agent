import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from rag_agent import F1RAGAgent
from ingestion import ingest_data

load_dotenv()

st.set_page_config(page_title="2026 F1 Spec RAG Agent", page_icon="🏎️", layout="wide")

st.title("🏎️ 2026 F1 Spec Technical RAG Agent")
st.markdown("""
Ask specific technical questions about the **2026 FIA Formula 1 Technical Regulations**.
The agent will answer based on the official PDF and provide **Article Number** citations.
""")

# Initialize RAG agent only once
@st.cache_resource
def load_rag_agent():
    if not os.path.exists("vector_db"):
        with st.spinner("Initializing vector database from regulations... This may take a minute."):
            try:
                ingest_data()
                st.success("Database initialized!")
            except Exception as e:
                st.error(f"Failed to initialize database: {e}")
                return None
    
    try:
        return F1RAGAgent()
    except Exception as e:
        st.error(f"Error loading RAG agent: {e}")
        return None

agent = load_rag_agent()

if agent is None:
    st.warning("⚠️ Vector database not found. Please run the ingestion script first: `python src/ingestion.py`")
else:
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.info("This agent uses a RAG pipeline with Google Gemini to query the 2026 FIA Technical Regulations.")
        st.subheader("Key Features")
        st.markdown(
            "- **Direct PDF Querying**\n"
            "- **Mandatory Citations**\n"
            "- **Live Web Skill for Up-to-Date Questions**"
        )
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ex: What are the active aero modes (X-mode/Z-mode)?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing regulations and checking live updates when needed..."):
                try:
                    response = agent.query(prompt)
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Store response in history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show sources in an expander
                    with st.expander("📚 View Cited Sources"):
                        for i, doc in enumerate(response["context"]):
                            page_num = doc.metadata.get('page', 'N/A')
                            if isinstance(page_num, int):
                                page_num += 1
                            st.write(f"**Source {i+1} (Page {page_num})**")
                            st.write(doc.page_content)
                            st.divider()

                    if response.get("skill_active"):
                        with st.expander("🌐 Live Web Sources (Up-to-Date Skill)"):
                            if response.get("skill_error"):
                                st.warning(response["skill_error"])
                            elif not response.get("web_sources"):
                                st.info("No live web snippets were returned for this question.")
                            else:
                                for i, source in enumerate(response["web_sources"], start=1):
                                    st.markdown(
                                        f"**{i}. {source['title']}**  \n"
                                        f"{source['url']}  \n"
                                        f"{source['snippet']}"
                                    )
                                    st.divider()
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
