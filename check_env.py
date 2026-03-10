import os
import requests
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None

load_dotenv()


def check_service(name, status, details=""):
    icon = "✅" if status else "❌"
    detail_str = f" ({details})" if details else ""
    print(f"{name:<28} {icon} {'Active' if status else 'Failed/Missing'}{detail_str}")


print("\n--- CareerAgent-AI: Beta Readiness Diagnostic ---")

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
check_service("OpenAI (Orchestrator)", openai_ok)
check_service("Google Gemini (Backup)", gemini_ok)
check_service("Hugging Face (Inference)", bool(hf_token and hf_token.startswith("hf_")))

qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")
qdrant_active = False
if qdrant_url and qdrant_key and QdrantClient:
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=5)
        client.get_collections()
        qdrant_active = True
    except Exception:
        qdrant_active = False

chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "outputs/phase3/chroma")
check_service("Qdrant Cloud (Global)", qdrant_active)
check_service("Local Chroma (Cache)", os.path.exists(chroma_dir), chroma_dir)

check_service("Tavily AI Search", bool(os.getenv("TAVILY_API_KEY")))
check_service("Serper.dev", bool(os.getenv("SERPER_API_KEY")))

ls_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
check_service("LangSmith Tracing", bool(ls_key and "lsv2" in ls_key))

resend_key = os.getenv("RESEND_API_KEY")
sendgrid_key = os.getenv("SENDGRID_API_KEY")
check_service("Resend Engine", bool(resend_key))
check_service("SendGrid Engine", bool(sendgrid_key))

langsmith_project = os.getenv("LANGSMITH_PROJECT", "")
langchain_project = os.getenv("LANGCHAIN_PROJECT", "")
project_match = bool(langsmith_project) and langsmith_project == langchain_project
check_service("LangSmith Project Sync", project_match, f"ls={langsmith_project or 'unset'} lc={langchain_project or 'unset'}")


def _quick_get(url: str, headers=None):
    try:
        r = requests.get(url, headers=headers or {}, timeout=6)
        return r.status_code
    except Exception:
        return None

serper_status = None
if os.getenv("SERPER_API_KEY"):
    serper_status = _quick_get("https://google.serper.dev/search", headers={"X-API-KEY": os.getenv("SERPER_API_KEY", "")})
openai_status = None
if os.getenv("OPENAI_API_KEY"):
    openai_status = _quick_get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"})

check_service("Serper Reachability", bool(serper_status and serper_status in (200, 400, 401, 403)), f"HTTP {serper_status}" if serper_status else "not_checked")
check_service("OpenAI Reachability", bool(openai_status and openai_status in (200, 401)), f"HTTP {openai_status}" if openai_status else "not_checked")

print("-" * 60)
if not (openai_ok or hf_token):
    print("⚠️  CRITICAL: No LLM detected. The agent will have no 'brain'.")
if not qdrant_active:
    print("⚠️  WARNING: Qdrant offline. Long-term memory is disabled.")
