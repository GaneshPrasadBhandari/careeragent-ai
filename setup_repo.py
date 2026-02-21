import os
from pathlib import Path

def setup_careeragent_ai():
    # Define the 10-Layer Structure + MLOps folders
    # This aligns with the "Recursive Gate" and "Evaluator Agent" logic
    structure = [
        "src/careeragent/core",          # L0: Infra/Settings/Logging
        "src/careeragent/database",      # L1: Azure Blob/VectorDB/SQL
        "src/careeragent/schemas",       # L0: Pydantic V2 Models
        "src/careeragent/agents",        # L2-L4: Generator & Evaluator Agents
        "src/careeragent/models",        # L1: LLM/Embedding Factories
        "src/careeragent/orchestration", # L5-L7: LangGraph/State Management
        "src/careeragent/utils",         # Helpers (PDF/Text)
        "app/api",                       # L9: FastAPI Endpoints
        "app/ui",                        # L8: Gradio/Reflex Dashboard
        "tests/unit",                    # Testing each agent
        "tests/integration",             # Testing Layer-to-Layer loops
        "data/raw", "data/gold",         # DVC Data storage
        "artifacts",                     # JSON/MD Versioned Artifacts
        "notebooks",                     # Staging for ChatGPT code experiments
        ".github/workflows"              # CI/CD for AWS/Azure
    ]

    for path in structure:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Create __init__.py to make it a Python package
        (p / "__init__.py").touch()
    
    # Create the Experiment Notebook
    (Path("notebooks") / "migration_experiments.ipynb").touch()
    
    # Create standard root files
    (Path(".env")).touch()
    (Path(".gitignore")).touch()

    print("\n✅ CareerAgent-AI Skeleton Ready. Open VS Code to start Batch 1.")

if __name__ == "__main__":
    setup_careeragent_ai()