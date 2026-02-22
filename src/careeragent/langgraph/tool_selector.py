from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from careeragent.langgraph.state import AttemptLog, utc_now
from careeragent.langgraph.tools import ToolResult


ToolCall = Callable[[], Awaitable[ToolResult]]


class ToolSelector:
    """
    Description: Resilient tool selector. Tries A→B→C until confidence >= threshold.
    Layer: L0
    """

    @staticmethod
    async def run(
        *,
        layer_id: str,
        agent: str,
        calls: List[Tuple[str, Optional[str], ToolCall]],
        min_conf: float,
        attempts_log: List[AttemptLog],
    ) -> ToolResult:
        for tool_name, model_name, call in calls:
            attempt_id = uuid.uuid4().hex
            started = utc_now()
            res = await call()
            finished = utc_now()

            status = "ok" if (res.ok and res.confidence >= min_conf) else ("low_conf" if res.ok else "failed")
            attempts_log.append(
                AttemptLog(
                    attempt_id=attempt_id,
                    layer_id=layer_id,
                    agent=agent,
                    tool=tool_name,
                    model=model_name,
                    status=status,
                    confidence=float(res.confidence),
                    started_at_utc=started,
                    finished_at_utc=finished,
                    error=res.error,
                    meta=res.meta,
                )
            )

            if res.ok and res.confidence >= min_conf:
                return res

        # return last (best effort)
        return res  # type: ignore[name-defined]