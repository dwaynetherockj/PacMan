"""
Skill-controlled Pac-Man bot.

skill in [0, 1]:
  - mistake chance at each new tile: 2% (skill 1) up to 30% (skill 0)
  - ghost awareness radius, in walking tiles: 10 (skill 1) down to 2 (skill 0)
The bot heads for the nearest pellet and avoids ghosts inside its radius.
"""

import random
from collections import deque

import numpy as np

from model.direction import Direction

WALKABLE = {0, 1, 2}  # empty, dot, big dot
DIRS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}
OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def bfs_field(board, sources):
    """Walking distance (in tiles) from the nearest source tile to each reachable tile."""
    h, w = board.shape
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and board[nr][nc] in WALKABLE \
                    and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return dist


class SkillBot:
    def __init__(self, skill, seed=None):
        self.skill = skill
        self.mistake_chance = 0.02 + 0.28 * (1 - skill)
        self.radius = round(2 + 8 * skill)
        self.hunt_blue = skill >= 0.4
        self.max_hunts = 4 if skill >= 0.8 else 2   # strong bot no longer unlimited
        self.hunt_range = 5                          # only chase a blue ghost within 5 walking tiles
        self.rng = random.Random(seed)
        self._tile = None
        self._choice = None
        self._stuck_tile = None
        self._stuck_count = 0
        self._hunts_this_powerup = 0
        self._was_powered = False

    def choose(self, engine):
        """Direction to request this tick. Re-decided only when the bot enters a new tile."""
        tw, th = engine.tile_width, engine.tile_height
        player = engine.player
        tile = (int(player.location_y // th), int(player.location_x // tw))

        # Reset the hunt counter each time a new power pellet is eaten.
        powered = bool(engine.player.powerup)
        if powered and not self._was_powered:
            self._hunts_this_powerup = 0
        self._was_powered = powered

        # Track how long we've been on the same tile.
        if tile == self._stuck_tile:
            self._stuck_count += 1
        else:
            self._stuck_tile = tile
            self._stuck_count = 0

        if tile != self._tile or self._choice is None:
            self._tile = tile
            self._choice = self._decide(engine, tile, tw, th)

        # If stuck on one tile too long, force a fresh random open direction.
        if self._stuck_count > 30:
            options = []
            board = player.board
            h, w = board.shape
            for d, (dr, dc) in DIRS.items():
                nr, nc = tile[0] + dr, tile[1] + dc
                if 0 <= nr < h and 0 <= nc < w and board[nr][nc] in WALKABLE:
                    options.append(d)
            if options:
                self._choice = self.rng.choice(options)
            self._stuck_count = 0
        # Count a successful hunt when a ghost has just become "eaten".
        if self.hunt_blue:
            for g in engine.ghosts:
                if g.is_eaten():
                    self._hunts_this_powerup = min(self._hunts_this_powerup + 1,
                                                   self.max_hunts)

        return self._choice

    def _decide(self, engine, tile, tw, th):
        board = engine.player.board  # the same array the game removes dots from
        h, w = board.shape
        current = engine.player.direction

        options = []
        for d, (dr, dc) in DIRS.items():
            nr, nc = tile[0] + dr, tile[1] + dc
            if 0 <= nr < h and 0 <= nc < w and board[nr][nc] in WALKABLE:
                options.append((d, (nr, nc)))
        if not options:
            return current

        if self.rng.random() < self.mistake_chance:
            return self.rng.choice(options)[0]

        dots = [tuple(map(int, rc)) for rc in np.argwhere((board == 1) | (board == 2))]
        dot_dist = bfs_field(board, dots) if dots else {}

        ghost_fields = []       # dangerous ghosts to avoid
        blue_fields = []        # frightened ghosts to chase (if this bot hunts)
        for g in engine.ghosts:
            if g.is_eaten() or g.is_in_house():
                continue
            gt = (int(g.location_y // th), int(g.location_x // tw))
            if g.is_frightened():
                # Only hunt if this bot hunts, hasn't used up its hunts, and
                # the ghost is close enough to reach before blue runs out.
                if (self.hunt_blue
                        and self._hunts_this_powerup < self.max_hunts):
                    field = bfs_field(board, [gt])
                    if field.get(tile, 999) <= self.hunt_range:
                        blue_fields.append(field)
            else:
                ghost_fields.append(bfs_field(board, [gt]))

        best, best_score = None, float("-inf")
        for d, n in options:
            score = -dot_dist.get(n, 999)            # nearer to a pellet is better
            for field in ghost_fields:
                gd = field.get(n, 999)
                if gd < self.radius:
                    score -= 10 * (self.radius - gd)  # nearer to a dangerous ghost is worse
            for field in blue_fields:
                bd = field.get(n, 999)
                score -= 3 * bd                       # nearer to a blue ghost is better (big pull)
            if d == OPPOSITE.get(current):
                score -= 2                            # small push against turning back
            if score > best_score:
                best, best_score = d, score
        return best