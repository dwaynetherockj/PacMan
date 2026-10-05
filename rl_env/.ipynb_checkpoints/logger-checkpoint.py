"""
Per-tick telemetry logger for full Pac-Man games.

Writes one CSV row per tick. The columns are the raw signals the player
skill model will be built from: positions, score, lives, pellets left,
each ghost's mode and distance, and the direction the player chose.
Nothing here changes game behaviour; it only reads state after each tick.
"""

import csv


class TelemetryLogger:
    def __init__(self, engine, path):
        self.engine = engine
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.tick = 0

        self.ghost_names = [type(g).__name__ for g in engine.ghosts]
        header = ["tick", "player_x", "player_y", "player_dir",
                  "score", "lives", "pellets_left", "powerup_active"]
        for n in self.ghost_names:
            header += [f"{n}_x", f"{n}_y", f"{n}_mode", f"{n}_dist"]
        self.writer.writerow(header)

    def log(self):
        """Call once per tick, right after engine.tick()."""
        self.tick += 1
        e = self.engine
        p = e.player
        board = p.board
        pellets_left = int(((board == 1) | (board == 2)).sum())

        row = [self.tick, p.location_x, p.location_y, p.direction.name,
               e.level.score, p.lives, pellets_left, int(bool(p.powerup))]

        for g in e.ghosts:
            mode = ("frightened" if g.is_frightened() else
                    "eaten" if g.is_eaten() else
                    "scatter" if g.is_scatter() else "chase")
            dist = abs(g.location_x - p.location_x) + abs(g.location_y - p.location_y)
            row += [g.location_x, g.location_y, mode, dist]

        self.writer.writerow(row)

    def close(self):
        self.file.close()