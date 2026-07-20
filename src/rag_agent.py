import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from up_to_date_skill import UpToDateAnswerSkill

load_dotenv()

VECTOR_DB_DIR = "vector_db"

class F1RAGAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
        )
        self.up_to_date_skill = UpToDateAnswerSkill()

    def query(self, question: str):
        # 1. Retrieve
        docs = self.vectorstore.similarity_search(question, k=10)
        context = "\n\n".join([doc.page_content for doc in docs])
        skill_result = self.up_to_date_skill.run(question)
        web_context = skill_result.web_context if skill_result.active else ""

        # 2. Construct Prompt
        system_prompt = (
            "You are an expert on the 2026 FIA Formula 1 Technical Regulations and Formula 1 in general. "
            "Your primary task is to answer technical questions accurately using the provided context and provide citations for your claims. "
            "If the provided context contains the answer, you MUST use it and cite the relevant Article Number (e.g., Article 3.2.1). "
            "If the article number is not clear, mention the page number. "
            "\n\n"
            "CRITICAL: If the provided context DOES NOT contain the answer (e.g., questions about the MGU-H elimination, general F1 concepts, or historical rules), "
            "you should use your general expert knowledge to answer the question. However, you MUST explicitly state that your answer is based on general knowledge and not found directly in the provided 2026 regulations context."
            "\n\n"
            "Technical Guidance: "
            "- '50/50 power split' refers to the balance between ICE (Internal Combustion Engine) and MGU-K (Electrical) power. "
            "- Active Aero modes are officially 'Straight-Line Mode' and 'Corner Mode'. "
            "\n\n"
            "If a live web context block is provided, use it only for up-to-date facts and explicitly cite the URL next to those facts. "
            "If live web context is empty or unavailable, say that no live sources were available and avoid pretending to have current-web verification. "
            "\n\n"
            f"Regulation Context: {context}\n\n"
            f"Live Web Context Timestamp (UTC): {skill_result.generated_at_utc}\n"
            f"Live Web Context: {web_context if web_context else 'None'}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        # 3. Generate
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "context": docs,
            "skill_active": skill_result.active,
            "web_sources": [
                {
                    "title": source.title,
                    "url": source.url,
                    "snippet": source.snippet,
                }
                for source in skill_result.sources
            ],
            "skill_error": skill_result.error,
        }

def ask_question(question: str):
    agent = F1RAGAgent()
    return agent.query(question)

if __name__ == "__main__":
    # Test question
    q = "What is the new wheelbase limit for the 2026 cars?"
    res = ask_question(q)
    print(f"Question: {q}")
    print(f"Answer: {res['answer']}")
