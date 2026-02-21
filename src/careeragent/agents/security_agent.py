from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from careeragent.orchestration.state import OrchestrationState, _iso_utc, _utc_now


@dataclass(frozen=True)
class SecurityConfig:
    """
    Description: Security configuration for prompt-injection detection.
    Layer: L0
    Input: Optional config overrides from env/state.meta
    Output: Deterministic security behavior
    """
    max_snippet_chars: int = 240


class SanitizeAgent:
    """
    Description: L0 security guard that sanitizes inputs before any LLM call.
    Layer: L0
    Input: user text / prompts
    Output: sanitized text OR blocks run + writes artifacts/security_audit.json
    """

    _INJECTION_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"\b(ignore|disregard)\b.*\b(previous|above|system|developer|instructions)\b", re.I),
        re.compile(r"\b(system\s*prompt|developer\s*message|hidden\s*instructions)\b", re.I),
        re.compile(r"\b(jailbreak|do\s*anything\s*now|dan)\b", re.I),
        re.compile(r"\bBEGIN\s*(SYSTEM|PROMPT|INSTRUCTIONS)\b", re.I),
        re.compile(r"\b(exfiltrate|leak)\b.*\b(prompt|keys|secrets)\b", re.I),
    )

    def __init__(self, *, artifacts_root: Optional[Path] = None, config: Optional[SecurityConfig] = None) -> None:
        """
        Description: Initialize the sanitize agent.
        Layer: L0
        Input: artifacts_root + config
        Output: SanitizeAgent
        """
        self._root = artifacts_root or Path(__file__).resolve().parents[1] / "artifacts"
        self._root.mkdir(parents=True, exist_ok=True)
        self._audit_path = self._root / "security_audit.json"
        self._cfg = config or SecurityConfig()

    def sanitize_before_llm(
        self,
        *,
        state: OrchestrationState,
        step_id: str,
        tool_name: str,
        user_text: str,
        context: str = "generic",
    ) -> Optional[str]:
        """
        Description: Inspect input for prompt injections; block run if detected.
        Layer: L0
        Input: OrchestrationState + step_id + tool_name + user_text
        Output: sanitized text (same as input) OR None if blocked
        """
        txt = (user_text or "").strip()

        for rx in self._INJECTION_PATTERNS:
            m = rx.search(txt)
            if m:
                # Step trace + run status transition
                state.end_step(
                    step_id,
                    status="blocked",
                    output_ref={"security": "prompt_injection", "pattern": rx.pattern},
                    message="PROMPT_INJECTION_BLOCKED",
                )
                state.status = "blocked"
                state.meta["run_failure_code"] = "SECURITY_BLOCK"
                state.meta["security_block_reason"] = "prompt_injection"
                state.touch()

                # Persist audit log
                self._append_audit(
                    {
                        "run_id": state.run_id,
                        "ts_utc": _iso_utc(_utc_now()),
                        "step_id": step_id,
                        "tool_name": tool_name,
                        "context": context,
                        "matched_pattern": rx.pattern,
                        "snippet": txt[: self._cfg.max_snippet_chars],
                    }
                )
                return None

        return txt

    def _append_audit(self, record: Dict[str, Any]) -> None:
        """
        Description: Append a record into artifacts/security_audit.json.
        Layer: L0
        Input: record dict
        Output: file updated
        """
        existing: List[Dict[str, Any]] = []
        if self._audit_path.exists():
            try:
                existing = json.loads(self._audit_path.read_text(encoding="utf-8")) or []
            except Exception:
                existing = []

        existing.append(record)
        self._audit_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
