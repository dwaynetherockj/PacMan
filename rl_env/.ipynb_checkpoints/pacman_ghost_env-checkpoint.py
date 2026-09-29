"""
Gymnasium environment for training ONE ghost (index 0, Blinky) with RL.

Options:
  curriculum_distance          spawn the ghost this many walking tiles from the player
  solo_ghost=True              the other three ghosts do not move
  scripted_trained_ghost=True  the trained ghost uses the original scripted AI
  nav_obs=True                 add four open-direction flags (up, down, left, right)
                               to the observation (8 values become 12)
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pacman import Game
from model.direction import Direction
from settings import DISTANCE_FACTOR
from rl_env.maze_distance import sample_tile_at_distance

MAX_STEPS_PER_EPISODE = 800
TRAINED_GHOST_INDEX = 0


class PacmanGhostEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, curriculum_distance=None, solo_ghost=False,
                 scripted_trained_ghost=False, nav_obs=False):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.nav_obs = nav_obs
        n_obs = 12 if nav_obs else 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        self.game = None
        self.engine = None
        self.ghost = None
        self.step_count = 0
        self._prev_lives = None
        self._prev_score = None
        self._catcher = None
        self.curriculum_distance = curriculum_distance
        self.solo_ghost = solo_ghost
        self.scripted_trained_ghost = scripted_trained_ghost

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = Game()
        self.engine = self.game.game_engine
        self.ghost = self.engine.ghosts[TRAINED_GHOST_INDEX]

        # Skip the READY! counter to speed up training.
        self.engine.player.set_to_chase()

        if self.curriculum_distance is not None:
            row, col = sample_tile_at_distance(self.curriculum_distance)
            tile_w = self.engine.tile_width
            tile_h = self.engine.tile_height
            self.ghost.location_x = col * tile_w + tile_w // 2
            self.ghost.location_y = row * tile_h + tile_h // 2

        self.step_count = 0
        self._prev_lives = self.engine.player.lives
        self._prev_score = self.engine.level.score
        self._catcher = None

        return self._get_obs(), {}

    def step(self, action):
        direction = Direction(int(action))

        # Random player movement (engine.direction_command otherwise never changes).
        self.engine.direction_command = random.choice(list(Direction))

        self.engine.render_level()
        self.engine.draw_misc()
        self.engine.render_ghosts()

        if self.engine.player.is_ready():
            self.engine.reset_ghosts()
            self.engine.render_player()
            self.engine.start_counter += 1
            if self.engine.start_counter == self._get_start_trigger():
                self.engine.player.set_to_chase()
                self.engine.start_counter = 0
        elif self.engine.player.is_chasing():
            self.engine.render_player()
            self.engine.move_player()

            for i, g in enumerate(self.engine.ghosts):
                if i == TRAINED_GHOST_INDEX:
                    if self.scripted_trained_ghost:
                        g.follow_target()
                    else:
                        g.follow_target(action_direction=direction)
                elif not self.solo_ghost:
                    g.follow_target()

            self.engine.check_ghosts_and_player_collision()
            if self.engine.player.is_eaten():
                self._catcher = self._find_catcher()
        elif self.engine.player.is_eaten():
            # The real game finishes this inside play_death_animation(), which
            # needs several sprite frames. We apply the same state change directly.
            self.engine.player.set_to_ready()
            self.engine.player.lives -= 1

        self.step_count += 1

        lives_before = self._prev_lives  # captured before _compute_reward() updates it
        reward = self._compute_reward()
        caught = self.engine.player.lives < lives_before

        terminated = caught  # end the episode the moment a catch happens
        truncated = self.step_count >= MAX_STEPS_PER_EPISODE

        mode = "chase"
        if self.ghost.is_scatter():
            mode = "scatter"
        elif self.ghost.is_frightened():
            mode = "frightened"
        elif self.ghost.is_eaten():
            mode = "eaten"

        info = {
            "caught": caught,
            "distance": self._distance_to_player(),
            "mode": mode,
            "caught_by": self._catcher if caught else None,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def _find_catcher(self):
        """Name of the ghost that caught the player (closest one inside the
        game's catch box that is in chase or scatter mode)."""
        px, py = self.engine.player.location_x, self.engine.player.location_y
        best_name, best_dist = None, float("inf")
        for g in self.engine.ghosts:
            if not (g.is_chasing() or g.is_scatter()):
                continue
            dx, dy = abs(g.location_x - px), abs(g.location_y - py)
            if dx < DISTANCE_FACTOR and dy < DISTANCE_FACTOR:
                d = dx * dx + dy * dy
                if d < best_dist:
                    best_name, best_dist = type(g).__name__, d
        return best_name

    def _compute_reward(self):
        reward = 0.0

        lives_now = self.engine.player.lives
        if lives_now < self._prev_lives:
            # Reward only catches made by the trained ghost itself.
            if self._catcher == type(self.ghost).__name__:
                reward += 10.0
        self._prev_lives = lives_now

        dist = self._distance_to_player()
        reward += -0.001 * dist  # closer = less negative = better

        reward -= 0.01  # small per-tick cost so catching sooner is preferred

        return reward

    def _distance_to_player(self):
        gx, gy = self.ghost.location_x, self.ghost.location_y
        px, py = self.engine.player.location_x, self.engine.player.location_y
        return ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5

    def _get_obs(self):
        gx, gy = self.ghost.location_x, self.ghost.location_y
        px, py = self.engine.player.location_x, self.engine.player.location_y

        mode_code = 0.0
        if self.ghost.is_scatter():
            mode_code = 1.0
        elif self.ghost.is_frightened():
            mode_code = 2.0
        elif self.ghost.is_eaten():
            mode_code = 3.0

        values = [gx, gy, px, py, gx - px, gy - py, mode_code,
                  float(self.engine.player.lives)]

        if self.nav_obs:
            # The ghosts share one Turns object (Blinky and Pinky visibly do in
            # level_content_initializer.py), so refresh the flags for THIS
            # ghost's current position before reading them.
            self.ghost._check_borders_ahead()
            t = self.ghost.turns
            values += [float(t.up), float(t.down), float(t.left), float(t.right)]

        return np.array(values, dtype=np.float32)

    def _get_start_trigger(self):
        from settings import START_TRIGGER
        return START_TRIGGER