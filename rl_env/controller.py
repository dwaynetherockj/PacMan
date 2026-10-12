"""
Adaptive difficulty controller.

Watches a game, and each time the player loses a life it:
  1. summarises that life into the same features the classifier was trained on,
  2. asks the saved player model for a skill band (casual/middle/hardcore),
  3. picks the matching ghost difficulty for the next life.

This changes no game code; it reads game state and adjusts ghost settings.
Call start_life() once at the start, then update() every tick.
"""

import joblib
import numpy as np


OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# Group -> ghost difficulty. One place to change if the design changes.
# For now difficulty is a speed multiplier on the ghosts, as a stand-in until
# the three trained cohort ghosts exist.
BAND_SETTINGS = {
    "casual":   {"ghost_speed_mult": 0.75},   # slower, easier
    "middle":   {"ghost_speed_mult": 1.00},   # normal
    "hardcore": {"ghost_speed_mult": 1.25},   # faster, harder
}


class AdaptiveController:
    def __init__(self, engine, model_path="player_model.joblib"):
        self.engine = engine
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.current_band = "middle"       # start neutral
        self.history = []                  # (life_number, band) for later analysis
        self._life = None
        self._lives_seen = engine.player.lives
        self._prev_dir = None
        self._prev_eaten = set()
        self.start_life()

    def start_life(self):
        e = self.engine
        self._life = {
            "ticks": 0,
            "score_start": e.level.score,
            "reversals": 0,
            "near_ghost_ticks": 0,
            "blue_eaten": 0,
            "powerup_tick": None,
            "reaction_ticks": None,
        }
        self._prev_dir = None

    def update(self):
        """Call once per tick, after engine.tick()."""
        e = self.engine
        p = e.player
        life = self._life
        life["ticks"] += 1

        d = p.direction.name
        if self._prev_dir and OPPOSITE.get(self._prev_dir) == d:
            life["reversals"] += 1
        self._prev_dir = d

        px, py = p.location_x, p.location_y
        nearest = min(
            (abs(g.location_x - px) + abs(g.location_y - py)
             for g in e.ghosts if not (g.is_frightened() or g.is_eaten())),
            default=9999)
        if nearest < e.tile_width:
            life["near_ghost_ticks"] += 1

        if p.powerup and life["powerup_tick"] is None:
            life["powerup_tick"] = life["ticks"]
        if not p.powerup:
            life["powerup_tick"] = None

        now_eaten = {id(g) for g in e.ghosts if g.is_eaten()}
        newly = now_eaten - self._prev_eaten
        life["blue_eaten"] += len(newly)
        if newly and life["powerup_tick"] is not None and life["reaction_ticks"] is None:
            life["reaction_ticks"] = life["ticks"] - life["powerup_tick"]
        self._prev_eaten = now_eaten

        # A life was just lost -> classify it and adapt.
        if p.lives < self._lives_seen:
            self._lives_seen = p.lives
            band = self._classify(life)
            self._apply(band)
            self.history.append((len(self.history) + 1, band))
            self.start_life()

    def _classify(self, life):
        ticks = max(life["ticks"], 1)
        score_rate = (self.engine.level.score - life["score_start"]) / ticks
        reaction = life["reaction_ticks"] if life["reaction_ticks"] is not None else 600.0
        feat = {
            "life_ticks": ticks,
            "blue_eaten": life["blue_eaten"],
            "reaction_filled": reaction,
            "ate_blue": 1.0 if life["blue_eaten"] > 0 else 0.0,
            "near_ghost_ticks": life["near_ghost_ticks"],
            "score_rate": score_rate,
        }
        X = np.array([[feat[f] for f in self.features]])
        return self.model.predict(X)[0]

    def _apply(self, band):
        self.current_band = band
        mult = BAND_SETTINGS[band]["ghost_speed_mult"]
        for g in self.engine.ghosts:
            g.speed_mult = mult