from __future__ import annotations

import re
from typing import Optional

TEAM_ALIASES = {
    "arizona dbacks": "arizona diamondbacks",
    "az diamondbacks": "arizona diamondbacks",
    "chi cubs": "chicago cubs",
    "chi white sox": "chicago white sox",
    "cws": "chicago white sox",
    "la angels": "los angeles angels",
    "los angeles angels of anaheim": "los angeles angels",
    "la dodgers": "los angeles dodgers",
    "ny mets": "new york mets",
    "ny yankees": "new york yankees",
    "oakland athletics": "athletics",
    "oakland as": "athletics",
    "athletics": "athletics",
    "sf giants": "san francisco giants",
    "sd padres": "san diego padres",
    "tb rays": "tampa bay rays",
    "tampa bay devil rays": "tampa bay rays",
    "washington nationals": "washington nationals",
    "wsh nationals": "washington nationals",
}


def normalize_team_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)
