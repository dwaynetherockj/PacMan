"""
Maze-aware BFS distance utility for curriculum learning.

Computes true walking distance (in tiles, through open corridors only) from
the player's spawn tile to every reachable tile on the board, and provides
a way to sample a random reachable tile at a given target distance. Wall
tiles (BoardStructure values 3-9) are correctly treated as impassable, so
this respects the maze's actual layout rather than straight-line distance.
"""

import random
from collections import deque

from settings import BOARD, PLAYER_X, PLAYER_Y, BLINKY_X, BLINKY_Y

# Walkable board values, per model/board_structure.py's BoardStructure enum:
# EMPTY=0, DOT=1, BIG_DOT=2 are walkable. Everything from 3 upward is a wall
# or gate and is NOT walkable.
WALKABLE_VALUES = {0, 1, 2}


def _compute_distance_map():
    """BFS from the player's spawn tile over all walkable tiles.
    Board indexing is board[row][col], i.e. board[Y][X]."""
    height, width = BOARD.shape
    start = (PLAYER_Y, PLAYER_X)

    distances = {start: 0}
    queue = deque([start])

    while queue:
        row, col = queue.popleft()
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < height and 0 <= n_col < width:
                if BOARD[n_row][n_col] in WALKABLE_VALUES:
                    if (n_row, n_col) not in distances:
                        distances[(n_row, n_col)] = distances[(row, col)] + 1
                        queue.append((n_row, n_col))

    return distances


_DISTANCE_MAP = _compute_distance_map()

_TILES_BY_DISTANCE = {}
for tile, dist in _DISTANCE_MAP.items():
    _TILES_BY_DISTANCE.setdefault(dist, []).append(tile)

# The real maze-aware distance from the player's spawn to Blinky's real
# spawn tile -- this is the target distance for the FINAL curriculum phase.
REAL_BLINKY_DISTANCE = _DISTANCE_MAP.get((BLINKY_Y, BLINKY_X))


def sample_tile_at_distance(target_distance, tolerance=1):
    """Return a random (row, col) tile whose BFS distance from the player's
    spawn tile is within `tolerance` of `target_distance`. Falls back to the
    closest distance bucket that actually has tiles in it if none match
    exactly (some exact distances may not exist depending on maze shape)."""
    candidates = []
    for dist in range(target_distance - tolerance, target_distance + tolerance + 1):
        candidates.extend(_TILES_BY_DISTANCE.get(dist, []))

    if not candidates:
        closest_dist = min(_TILES_BY_DISTANCE.keys(), key=lambda d: abs(d - target_distance))
        candidates = _TILES_BY_DISTANCE[closest_dist]

    return random.choice(candidates)