import os
import requests
from dotenv import load_dotenv

load_dotenv()

def check_service(name, status):
    icon = "✅" if status else "❌"
    print(f"{name:<25} {icon} {'Active' if status else 'Failed/Missing'}")

print("\n--- CareerAgent-AI: Full Stack Diagnostic ---")

# 1. LLM & Generation
gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
ollama_ok = os.getenv("OLLAMA_HOST") is not None
check_service("Google Gemini API", gemini_ok)
check_service("OpenAI API", openai_ok)
check_service("Local Ollama", ollama_ok)

# 2. Search & Discovery (L2 Hunt)
serper_ok = bool(os.getenv("SERPER_API_KEY"))
tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
scraping_ok = bool(os.getenv("SCRAPINGBEE_API_KEY"))
check_service("Serper.dev", serper_ok)
check_service("Tavily AI Search", tavily_ok)
check_service("ScrapingBee", scraping_ok)

# 3. Databases (L0 & L4 RAG)
db_cloud = "sqlitecloud" in os.getenv("DATABASE_URL", "")
qdrant_ok = bool(os.getenv("QDRANT_API_KEY"))
check_service("SQLite Cloud (Auth)", db_cloud)
check_service("Qdrant Vector DB", qdrant_ok)

# 4. Observability & MCP
ls_ok = "lsv2" in os.getenv("LANGSMITH_API_KEY", "")
mcp_ok = bool(os.getenv("MCP_URL"))
check_service("LangSmith Tracing", ls_ok)
check_service("MCP Server (Tavily)", mcp_ok)

# 5. Communication
sendgrid_ok = bool(os.getenv("SENDGRID_API_KEY"))
check_service("SendGrid Email", sendgrid_ok)

print("-" * 45)