"""Phase 3c: discovery, leaderboard, hall-of-fame queries.

These are read-only aggregations over `matches` / `characters` / `players`
with a single visibility/rating filter shared across every surface.
"""

from app.discovery.hall_of_fame import hall_of_fame_for_character
from app.discovery.leaderboard import (
    LeaderboardWindow,
    character_leaderboard,
    player_leaderboard,
    window_cutoff,
)
from app.discovery.queries import (
    MatchSummary,
    list_live_matches,
    list_recent_matches,
    visible_character_filter,
)

__all__ = [
    "MatchSummary",
    "LeaderboardWindow",
    "list_live_matches",
    "list_recent_matches",
    "visible_character_filter",
    "character_leaderboard",
    "player_leaderboard",
    "window_cutoff",
    "hall_of_fame_for_character",
]
