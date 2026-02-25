from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from careeragent.core.state import AgentState
from careeragent.services.health_service import get_artifacts_root


FeedbackLabel = Literal["spam_fake", "legitimate_bug"]


class FeedbackItem(BaseModel):
    """
    Description: User/employer feedback payload for downstream triage and refinement.
    Layer: L8
    Input: Free-text feedback
    Output: Normalized feedback item
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["user", "employer", "system"] = "user"
    text: str
    context: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class FeedbackClassification(BaseModel):
    """
    Description: Classifier output distinguishing spam/fake issues vs legitimate bugs.
    Layer: L8
    Input: FeedbackItem.text
    Output: Label + confidence + reasons
    """

    model_config = ConfigDict(extra="forbid")

    label: FeedbackLabel
    confidence: float
    reasons: List[str] = Field(default_factory=list)


class FeedbackIngestResult(BaseModel):
    """
    Description: Result of feedback ingestion into the RAG refinement store.
    Layer: L8
    Input: FeedbackItem + classification
    Output: persisted flag + doc_id
    """

    model_config = ConfigDict(extra="forbid")

    stored: bool
    doc_id: Optional[str] = None
    classification: FeedbackClassification


class LocalJsonlVectorStore:
    """
    Description: Minimal local "RAG vector store" placeholder that persists feedback into JSONL.
                 (Swappable later with Chroma/FAISS/Azure AI Search.)
    Layer: L8
    Input: Text + metadata
    Output: JSONL file under artifacts/rag/
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        """
        Description: Initialize store path.
        Layer: L0
        Input: artifacts root (optional)
        Output: LocalJsonlVectorStore
        """
        base = root or get_artifacts_root()
        self._dir = base / "rag"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "feedback_store.jsonl"

    def add_text(self, *, text: str, metadata: Dict[str, Any]) -> str:
        """
        Description: Persist feedback as JSONL row (acts as RAG memory).
        Layer: L8
        Input: text + metadata
        Output: doc_id
        """
        doc_id = sha256((text or "").encode("utf-8")).hexdigest()[:24]
        row = {"doc_id": doc_id, "text": text, "metadata": metadata}
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return doc_id


class FeedbackEvaluatorService:
    """
    Description: Triage feedback into Spam/Fake vs Legitimate Bugs and store valid feedback into RAG.
    Layer: L8
    Input: FeedbackItem
    Output: FeedbackIngestResult + state.meta updates
    """

    _BUG_SIGNALS = (
        "traceback",
        "exception",
        "error:",
        "failed",
        "stack trace",
        "reproduce",
        "steps",
        "expected",
        "actual",
        "http 4",
        "http 5",
        "timeout",
        "null",
        "none",
        "typeerror",
        "valueerror",
        "pydantic",
        "langgraph",
    )
    _SPAM_SIGNALS = (
        "crypto",
        "bitcoin",
        "investment",
        "guaranteed",
        "gift card",
        "click here",
        "free money",
        "adult",
        "casino",
        "whatsapp",
        "telegram",
    )

    def __init__(self, store: Optional[LocalJsonlVectorStore] = None) -> None:
        """
        Description: Initialize feedback evaluator with a backing RAG store.
        Layer: L0
        Input: Optional LocalJsonlVectorStore
        Output: FeedbackEvaluatorService
        """
        self._store = store or LocalJsonlVectorStore()

    def classify(self, *, item: FeedbackItem) -> FeedbackClassification:
        """
        Description: Rule-based classifier to detect spam/fake issues vs legitimate bugs.
        Layer: L8
        Input: FeedbackItem
        Output: FeedbackClassification
        """
        t = (item.text or "").strip()
        low = t.lower()
        reasons: List[str] = []

        spam_hits = sum(1 for s in self._SPAM_SIGNALS if s in low)
        bug_hits = sum(1 for s in self._BUG_SIGNALS if s in low)

        # Heuristics: presence of structured error context boosts legitimacy
        has_code_block = "```" in t
        has_file_hint = bool(re.search(r"\b(src/|trace|line \d+|\.py)\b", low))

        # Score
        score_legit = (0.35 * min(1.0, bug_hits / 3.0)) + (0.35 * (1.0 if has_code_block else 0.0)) + (0.30 * (1.0 if has_file_hint else 0.0))
        score_spam = min(1.0, spam_hits / 2.0)

        if score_spam > 0.55 and score_legit < 0.55:
            reasons.append("Spam indicators detected (promotional/irrelevant keywords).")
            return FeedbackClassification(label="spam_fake", confidence=round(score_spam, 3), reasons=reasons)

        # Default to legitimate if it contains bug signals or structured context
        if score_legit >= 0.45 or bug_hits >= 1:
            reasons.append("Contains bug signals (errors/reproduction context).")
            return FeedbackClassification(label="legitimate_bug", confidence=round(max(score_legit, 0.60), 3), reasons=reasons)

        # Otherwise treat as spam/fake (low-signal complaint)
        reasons.append("Low-signal feedback without reproducible details; treated as spam/fake by policy.")
        return FeedbackClassification(label="spam_fake", confidence=0.60, reasons=reasons)

    def ingest(self, *, state: AgentState, item: FeedbackItem) -> FeedbackIngestResult:
        """
        Description: Store legitimate feedback into RAG store for refinement and log classification to state.
        Layer: L8
        Input: state + feedback item
        Output: FeedbackIngestResult
        """
        cls = self.classify(item=item)

        state.meta.setdefault("feedback_events", [])
        state.meta["feedback_events"].append({"label": cls.label, "confidence": cls.confidence, "text": item.text[:300]})
        state.touch()

        if cls.label != "legitimate_bug":
            return FeedbackIngestResult(stored=False, doc_id=None, classification=cls)

        doc_id = self._store.add_text(
            text=item.text,
            metadata={"source": item.source, "context": item.context, **(item.meta or {})},
        )
        return FeedbackIngestResult(stored=True, doc_id=doc_id, classification=cls)
