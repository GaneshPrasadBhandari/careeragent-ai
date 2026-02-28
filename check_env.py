# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()

# def check_service(name, status):
#     icon = "✅" if status else "❌"
#     print(f"{name:<25} {icon} {'Active' if status else 'Failed/Missing'}")

# print("\n--- CareerAgent-AI: Full Stack Diagnostic ---")

# # 1. LLM & Generation
# gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
# openai_ok = bool(os.getenv("OPENAI_API_KEY"))
# ollama_ok = os.getenv("OLLAMA_HOST") is not None
# check_service("Google Gemini API", gemini_ok)
# check_service("OpenAI API", openai_ok)
# check_service("Local Ollama", ollama_ok)

# # 2. Search & Discovery (L2 Hunt)
# serper_ok = bool(os.getenv("SERPER_API_KEY"))
# tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
# scraping_ok = bool(os.getenv("SCRAPINGBEE_API_KEY"))
# check_service("Serper.dev", serper_ok)
# check_service("Tavily AI Search", tavily_ok)
# check_service("ScrapingBee", scraping_ok)

# # 3. Databases (L0 & L4 RAG)
# db_cloud = "sqlitecloud" in os.getenv("DATABASE_URL", "")
# qdrant_ok = bool(os.getenv("QDRANT_API_KEY"))
# check_service("SQLite Cloud (Auth)", db_cloud)
# check_service("Qdrant Vector DB", qdrant_ok)

# # 4. Observability & MCP
# ls_ok = "lsv2" in os.getenv("LANGSMITH_API_KEY", "")
# mcp_ok = bool(os.getenv("MCP_URL"))
# check_service("LangSmith Tracing", ls_ok)
# check_service("MCP Server (Tavily)", mcp_ok)

# # 5. Communication
# sendgrid_ok = bool(os.getenv("SENDGRID_API_KEY"))
# check_service("SendGrid Email", sendgrid_ok)

# print("-" * 45)




# # Add these to your check_env.py
# # 3.1 Persistent Memory (Chroma check)
# chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "outputs/phase3/chroma")
# chroma_ok = os.path.exists(chroma_dir)
# check_service("Local Chroma DB", chroma_ok)

# # 4.1 Reasoning Engine (Hugging Face)
# hf_ok = bool(os.getenv("HF_TOKEN"))
# check_service("Hugging Face Inference", hf_ok)

# # 5.1 Communication (Resend check)
# resend_ok = bool(os.getenv("RESEND_API_KEY"))
# check_service("Resend Email Engine", resend_ok)



import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def check_service(name, status, details=""):
    icon = "✅" if status else "❌"
    detail_str = f" ({details})" if details else ""
    print(f"{name:<25} {icon} {'Active' if status else 'Failed/Missing'}{detail_str}")

print("\n--- CareerAgent-AI: Beta Readiness Diagnostic ---")

# 1. LLM & Reasoning Engine
# Hugging Face logic: Checks all 3 common variable names
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
gemini_ok = bool(os.getenv("GEMINI_API_KEY"))

check_service("OpenAI (Orchestrator)", openai_ok)
check_service("Google Gemini (Backup)", gemini_ok)
check_service("Hugging Face (Inference)", bool(hf_token and hf_token.startswith("hf_")))

# 2. Vector Memory (L0 & L4)
# Verifies if Qdrant Cloud is actually reachable, not just if the key exists
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_active = False
if qdrant_url and qdrant_key:
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=5)
        client.get_collections()
        qdrant_active = True
    except:
        qdrant_active = False

chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "outputs/phase3/chroma")
check_service("Qdrant Cloud (Global)", qdrant_active)
check_service("Local Chroma (Cache)", os.path.exists(chroma_dir))

# 3. Search & Discovery (L2 Hunt)
check_service("Tavily AI Search", bool(os.getenv("TAVILY_API_KEY")))
check_service("Serper.dev", bool(os.getenv("SERPER_API_KEY")))

# 4. Observability (LangSmith)
# Ensures key is the modern v2 format
ls_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
check_service("LangSmith Tracing", bool(ls_key and "lsv2" in ls_key))

# 5. Communication (L9 State Flush)
# Identifies which email engine is currently "Live"
resend_key = os.getenv("RESEND_API_KEY")
sendgrid_key = os.getenv("SENDGRID_API_KEY")
check_service("Resend Engine", bool(resend_key))
check_service("SendGrid Engine", bool(sendgrid_key))

print("-" * 55)
if not (openai_ok or hf_token):
    print("⚠️  CRITICAL: No LLM detected. The agent will have no 'brain'.")
if not qdrant_active:
    print("⚠️  WARNING: Qdrant offline. Long-term memory is disabled.")