import re
from typing import List, Optional
from careeragent.core.settings import Settings

# Fixed Regex for L3 Navigation filtering
_NAV_PATTERNS = re.compile(
    r"(sign[- ]?in|log[- ]?in|privacy|terms|cookie|about|contact|help|support|feedback|careers)",
    re.IGNORECASE
)

class ExtractionManager:
    """Manages cleaning and extraction of job data."""
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings

    def is_junk_link(self, text: str) -> bool:
        if not text:
            return True
        return bool(_NAV_PATTERNS.search(text))

class GeoFenceManager:
    """Manages geographic filtering for job leads."""
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings

class LeadScout:
    """The LeadScout agent responsible for finding job opportunities."""
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
