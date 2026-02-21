from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from careeragent.orchestration.state import OrchestrationState


def get_artifacts_root() -> Path:
    """
    Description: Resolve the canonical artifacts root directory required by the platform.
    Layer: L0
    Input: None
    Output: Path to src/careeragent/artifacts
    """
    here = Path(__file__).resolve()
    # src/careeragent/services/health_service.py -> src/careeragent
    careeragent_dir = here.parents[1]
    root = careeragent_dir / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


class EnvHealthCheck(BaseModel):
    """
    Description: Environment key health report for API gateway readiness.
    Layer: L0
    Input: os.environ (optionally loaded from .env)
    Output: Health report for UI/API
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    missing_keys: List[str] = Field(default_factory=list)
    tracing_enabled: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class UIAlerter:
    """
    Description: UI alert sink for user-facing notifications (Streamlit/Gradio can read from state.meta).
    Layer: L1
    Input: OrchestrationState + message payload
    Output: Appends structured alerts to state.meta['ui_alerts']
    """

    @staticmethod
    def alert(state: OrchestrationState, *, severity: Literal["info", "warning", "error"], title: str, message: str) -> None:
        """
        Description: Append a structured alert to OrchestrationState.meta for UI rendering.
        Layer: L1
        Input: OrchestrationState + alert payload
        Output: None
        """
        state.meta.setdefault("ui_alerts", [])
        state.meta["ui_alerts"].append(
            {"severity": severity, "title": title, "message": message}
        )
        state.touch()


class QuotaUsageSnapshot(BaseModel):
    """
    Description: Persistent quota usage snapshot for API providers (e.g., Serper).
    Layer: L0
    Input: Aggregated request metadata
    Output: JSON-serializable snapshot persisted in artifacts/quota/
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    total_requests: int = 0
    total_errors: int = 0
    last_status_code: Optional[int] = None
    blocked: bool = False
    last_error: Optional[str] = None


class QuotaManager:
    """
    Description: Tracks API usage and enforces quota-aware blocking.
    Layer: L0
    Input: API response codes + orchestration step context
    Output: Persisted counters + state transitions (blocked/api_failure)
    """

    def __init__(self, artifacts_root: Optional[Path] = None) -> None:
        """
        Description: Initialize quota manager with persistent storage.
        Layer: L0
        Input: artifacts_root (optional)
        Output: QuotaManager
        """
        self._root = (artifacts_root or get_artifacts_root()) / "quota"
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / "quota_usage.json"
        self._data: Dict[str, QuotaUsageSnapshot] = {}
        self._load()

    def _load(self) -> None:
        """
        Description: Load quota usage from disk.
        Layer: L0
        Input: artifacts/quota/quota_usage.json
        Output: In-memory quota map
        """
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for k, v in (raw or {}).items():
                self._data[k] = QuotaUsageSnapshot(**v)
        except Exception:
            # fail open; keep empty
            self._data = {}

    def persist(self) -> None:
        """
        Description: Persist quota usage to disk.
        Layer: L0
        Input: In-memory quota map
        Output: artifacts/quota/quota_usage.json updated
        """
        payload = {k: v.model_dump() for k, v in self._data.items()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record(self, *, provider: str, status_code: int, error: Optional[str] = None) -> None:
        """
        Description: Record a provider call result.
        Layer: L0
        Input: provider + status_code (+ optional error)
        Output: Updates persistent counters
        """
        snap = self._data.get(provider) or QuotaUsageSnapshot(provider=provider)
        snap.total_requests += 1
        snap.last_status_code = int(status_code)
        if int(status_code) >= 400:
            snap.total_errors += 1
            snap.last_error = error or f"HTTP {status_code}"
        self._data[provider] = snap
        self.persist()

    def mark_blocked(self, *, provider: str, reason: str) -> None:
        """
        Description: Mark a provider as blocked due to quota or access errors.
        Layer: L0
        Input: provider + reason
        Output: Updates snapshot.blocked
        """
        snap = self._data.get(provider) or QuotaUsageSnapshot(provider=provider)
        snap.blocked = True
        snap.last_error = reason
        self._data[provider] = snap
        self.persist()

    def handle_serper_response(
        self,
        *,
        state: OrchestrationState,
        step_id: str,
        status_code: int,
        tool_name: str = "serper.search",
        error_detail: Optional[str] = None,
    ) -> bool:
        """
        Description: Enforce quota policy for Serper. If HTTP 403 occurs, block the step and alert UI.
        Layer: L0
        Input: OrchestrationState + step context + status_code
        Output: True if blocked, else False
        """
        provider = "serper"
        self.record(provider=provider, status_code=status_code, error=error_detail)

        if int(status_code) == 403:
            # Step-level block (audit)
            state.end_step(
                step_id,
                status="blocked",
                output_ref={"provider": provider, "status_code": 403},
                message="SERPER_QUOTA_EXCEEDED",
            )
            # Run-level block
            state.status = "blocked"
            state.meta["run_failure_code"] = "API_FAILURE"
            state.meta["run_failure_provider"] = provider
            state.touch()

            self.mark_blocked(provider=provider, reason="403 quota exceeded / forbidden")

            UIAlerter.alert(
                state,
                severity="error",
                title="Search quota exceeded",
                message="Serper returned 403 (quota exceeded). Your run is blocked. Update your Serper plan/key or reduce search frequency.",
            )
            return True

        if int(status_code) >= 500:
            # API_FAILURE but not necessarily quota
            state.status = "api_failure"
            state.meta["run_failure_code"] = "API_FAILURE"
            state.meta["run_failure_provider"] = provider
            state.touch()
        return False

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Description: Return a JSON-serializable snapshot for monitoring.
        Layer: L0
        Input: None
        Output: dict snapshot
        """
        return {k: v.model_dump() for k, v in self._data.items()}


@dataclass(frozen=True)
class RequiredEnvKeys:
    """
    Description: Canonical API key names to check for production readiness.
    Layer: L0
    Input: None
    Output: Key registry
    """

    ollama: tuple[str, ...] = ("OLLAMA_BASE_URL", "OLLAMA_HOST")
    serper: tuple[str, ...] = ("SERPER_API_KEY",)
    twilio: tuple[str, ...] = ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER")
    langsmith: tuple[str, ...] = ("LANGSMITH_API_KEY",)
    huggingface: tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


class HealthService:
    """
    Description: API gateway health and monitoring utilities.
    Layer: L0
    Input: .env + environment variables
    Output: EnvHealthCheck + tracing bootstrap + quota manager
    """

    def __init__(self, *, artifacts_root: Optional[Path] = None) -> None:
        """
        Description: Initialize health service.
        Layer: L0
        Input: Optional artifacts_root
        Output: HealthService
        """
        self._artifacts_root = artifacts_root or get_artifacts_root()
        self.quota = QuotaManager(self._artifacts_root)

    def load_env(self, *, dotenv_path: str = ".env") -> None:
        """
        Description: Load environment variables from .env (non-fatal if missing).
        Layer: L0
        Input: dotenv_path
        Output: os.environ updated
        """
        load_dotenv(dotenv_path=dotenv_path, override=False)

    def enable_langsmith_tracing(self, *, project: str = "careeragent-ai") -> bool:
        """
        Description: Enable LangSmith tracing using environment variables (best-effort).
        Layer: L0
        Input: project name
        Output: True if tracing enabled, else False
        """
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            return False

        # Preferred LangSmith-native envs
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project)

        # Backward-compatible LangChain tracing env (some stacks still use these)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        # Some setups use LANGCHAIN_API_KEY; we don't set it here to avoid overwriting.
        return True

    def check_env(self) -> EnvHealthCheck:
        """
        Description: Validate required environment variables for production integration.
        Layer: L0
        Input: os.environ
        Output: EnvHealthCheck
        """
        req = RequiredEnvKeys()
        missing: List[str] = []

        def any_present(keys: tuple[str, ...]) -> bool:
            return any(os.getenv(k) for k in keys)

        if not any_present(req.ollama):
            missing.append("OLLAMA_BASE_URL or OLLAMA_HOST")
        for k in req.serper:
            if not os.getenv(k):
                missing.append(k)
        for k in req.twilio:
            if not os.getenv(k):
                missing.append(k)
        for k in req.langsmith:
            if not os.getenv(k):
                missing.append(k)
        if not any_present(req.huggingface):
            missing.append("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")

        tracing = self.enable_langsmith_tracing(project=os.getenv("LANGSMITH_PROJECT", "careeragent-ai"))

        return EnvHealthCheck(
            ok=(len(missing) == 0),
            missing_keys=missing,
            tracing_enabled=tracing,
            details={"quota_snapshot": self.quota.snapshot()},
        )
