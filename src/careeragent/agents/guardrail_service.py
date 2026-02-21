from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from careeragent.orchestration.state import OrchestrationState
from careeragent.agents.parser_agent_service import ExtractedResume
from careeragent.agents.matcher_agent_schema import MatchReport


GuardAction = Literal["allow", "redact", "block", "needs_revision"]


class GuardResult(BaseModel):
    """
    Description: Output of guardrails check with action + issues + sanitized text (if applicable).
    Layer: L0
    Input: Raw input/output text
    Output: GuardResult used for blocking or loop-back feedback
    """

    model_config = ConfigDict(extra="forbid")

    action: GuardAction
    issues: List[str] = Field(default_factory=list)
    sanitized_text: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class InputGuard:
    """
    Description: Pre-LLM guard that detects prompt injections and risky PII before model calls.
    Layer: L0
    Input: User text + context
    Output: GuardResult (allow/redact/block)
    """

    _INJECTION_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"\b(ignore|disregard)\b.*\b(previous|above|system|developer)\b", re.I),
        re.compile(r"\b(system\s*prompt|developer\s*message|hidden\s*instructions)\b", re.I),
        re.compile(r"\b(jailbreak|do\s*anything\s*now|dan)\b", re.I),
        re.compile(r"\bBEGIN\s*(SYSTEM|PROMPT|INSTRUCTIONS)\b", re.I),
    )

    _EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    _PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{8,}\d)")
    _SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

    def inspect(
        self,
        *,
        state: OrchestrationState,
        text: str,
        context: Literal["resume", "job", "chat", "feedback"] = "chat",
        allow_resume_contact_pii: bool = True,
    ) -> GuardResult:
        """
        Description: Inspect text for injection and PII. Redact disallowed PII or block on injection.
        Layer: L0
        Input: state + text + context
        Output: GuardResult
        """
        t = text or ""
        issues: List[str] = []

        # Prompt injection detection: block immediately.
        for rx in self._INJECTION_PATTERNS:
            if rx.search(t):
                issues.append("Prompt injection detected (attempt to override system/developer instructions).")
                self._log_security_event(state, event_type="prompt_injection_block", details={"context": context})
                state.status = "blocked"
                state.meta["run_failure_code"] = "SECURITY_BLOCK"
                state.touch()
                return GuardResult(action="block", issues=issues, sanitized_text=None)

        # PII detection and redaction
        pii_hits = {
            "email": bool(self._EMAIL_RE.search(t)),
            "phone": bool(self._PHONE_RE.search(t)),
            "ssn": bool(self._SSN_RE.search(t)),
            "credit_card": bool(self._CC_RE.search(t)),
        }
        disallowed = []
        if pii_hits["ssn"]:
            disallowed.append("ssn")
        if pii_hits["credit_card"]:
            disallowed.append("credit_card")

        if disallowed:
            issues.append(f"Disallowed PII detected: {', '.join(disallowed)}. Blocking.")
            self._log_security_event(state, event_type="pii_block", details={"pii": disallowed, "context": context})
            state.status = "blocked"
            state.meta["run_failure_code"] = "SECURITY_BLOCK"
            state.touch()
            return GuardResult(action="block", issues=issues)

        # Redact contact PII when context isn't resume (or when policy wants masking).
        sanitized = t
        if context != "resume" or not allow_resume_contact_pii:
            if pii_hits["email"]:
                sanitized = self._EMAIL_RE.sub("[REDACTED_EMAIL]", sanitized)
            if pii_hits["phone"]:
                sanitized = self._PHONE_RE.sub("[REDACTED_PHONE]", sanitized)
            if sanitized != t:
                issues.append("Contact PII redacted before LLM call.")
                self._log_security_event(state, event_type="pii_redact", details={"context": context})
                return GuardResult(action="redact", issues=issues, sanitized_text=sanitized)

        return GuardResult(action="allow", issues=issues, sanitized_text=t)

    @staticmethod
    def _log_security_event(state: OrchestrationState, *, event_type: str, details: Dict[str, Any]) -> None:
        """
        Description: Record security events for compliance and deep analytics.
        Layer: L0
        Input: state + event payload
        Output: state.meta['security_events'] appended
        """
        state.meta.setdefault("security_events", [])
        state.meta["security_events"].append({"type": event_type, "details": details})
        state.touch()


class OutputGuard:
    """
    Description: Post-generation guard that checks for hallucinations and bias in cover letter drafts.
    Layer: L0
    Input: Draft text + grounding evidence (resume + match report)
    Output: GuardResult (pass/needs_revision/block)
    """

    _BIAS_FLAGS: Tuple[str, ...] = (
        "young and energetic",
        "native english speaker",
        "must be a citizen",
        "male candidate",
        "female candidate",
        "religion",
        "caste",
    )

    _RISKY_CLAIMS: Tuple[re.Pattern, ...] = (
        re.compile(r"\bphd\b", re.I),
        re.compile(r"\b10\+?\s*years\b", re.I),
        re.compile(r"\bpatent\b", re.I),
        re.compile(r"\bnobel\b", re.I),
    )

    def check_cover_letter(
        self,
        *,
        state: OrchestrationState,
        draft_text: str,
        resume: ExtractedResume,
        match_report: MatchReport,
        country_code: str = "US",
    ) -> GuardResult:
        """
        Description: Validate cover letter for hallucinations/bias and missing required fields.
        Layer: L0
        Input: state + draft_text + resume + match_report
        Output: GuardResult (allow/needs_revision/block)
        """
        txt = (draft_text or "").strip()
        issues: List[str] = []

        # Bias / protected attribute language
        low = txt.lower()
        for phrase in self._BIAS_FLAGS:
            if phrase in low:
                issues.append(f"Potential bias flag detected: '{phrase}'. Remove protected-attribute language.")

        # Hallucination heuristic: risky claims not grounded
        for rx in self._RISKY_CLAIMS:
            if rx.search(txt):
                issues.append("Potential hallucination: high-risk credential/tenure claim detected. Verify grounding.")

        # Grounding heuristic: highlighted/matched skills should dominate; unknown skill tokens can be risky
        known_skills = set([s.strip().lower() for s in (resume.skills or [])]) | set(
            [s.strip().lower() for s in (match_report.matched_skills or [])]
        )

        # If user provided a global dictionary, use it to detect “skills mentioned”
        dictionary = state.meta.get("skill_dictionary")
        dict_skills = [str(s).lower() for s in dictionary] if isinstance(dictionary, list) else []
        mentioned = [s for s in dict_skills if s and s in low]
        unknown = [s for s in mentioned if s not in known_skills]
        if len(unknown) >= 3:
            issues.append(
                "Potential hallucination: cover letter mentions multiple skills not present in resume evidence. "
                f"Review/remove: {', '.join(sorted(set(unknown))[:6])}."
            )

        # Minimal compliance: contact email (if known) should appear somewhere
        if resume.contact.email and resume.contact.email.lower() not in low:
            issues.append("Missing contact email in cover letter. Include it in header or signature.")

        if issues:
            # Needs revision rather than block by default
            state.meta.setdefault("security_events", [])
            state.meta["security_events"].append({"type": "output_guard_flag", "details": {"issues": issues}})
            state.touch()
            return GuardResult(action="needs_revision", issues=issues, sanitized_text=txt)

        return GuardResult(action="allow", issues=[], sanitized_text=txt)
