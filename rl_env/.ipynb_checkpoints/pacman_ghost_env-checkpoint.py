"""
Gymnasium environment for training ONE ghost (Blinky, index 0) using RL.

The other three ghosts keep running their normal built-in AI (follow_target()
with no argument), so Baseline 1 behaviour is completely undisturbed. This
environment only intercepts the trained ghost's decision each tick.

Usage (manual sanity check, not real training):
    env = PacmanGhostEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
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
from rl_env.maze_distance import sample_tile_at_distance

# Direction.RIGHT=0, LEFT=1, UP=2, DOWN=3 -- matches the enum exactly,
# so action (an int 0-3) can be passed straight to Direction(action).

MAX_STEPS_PER_EPISODE = 800
TRAINED_GHOST_INDEX = 0  # Blinky, per __load_ghosts ordering


class PacmanGhostEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, curriculum_distance=None):
        super().__init__()
        self.action_space = spaces.Discrete(4)

        # Observation: [ghost_x, ghost_y, player_x, player_y,
        #               dx, dy, ghost_mode_code, player_lives]
        # Kept simple and unnormalized to start -- normalize once you know
        # real board dimensions, if training is unstable.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        self.game = None
        self.engine = None
        self.ghost = None
        self.step_count = 0
        self._prev_lives = None
        self._prev_score = None 
        self.curriculum_distance = curriculum_distance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = Game()
        self.engine = self.game.game_engine
        self.ghost = self.engine.ghosts[TRAINED_GHOST_INDEX]

        # Let the player start "chasing" immediately rather than waiting
        # through the READY! counter -- speeds up training.
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

        return self._get_obs(), {}

    def step(self, action):
        direction = Direction(int(action))

        # Give the player a random direction each tick, same as
        # measure_catch_time.py -- otherwise the player barely moves at all,
        # since engine.direction_command defaults to LEFT and nothing was
        # ever updating it here.
        self.engine.direction_command = random.choice(list(Direction))

        # Advance the game one tick, but override ONLY the trained ghost's
        # decision. The other three ghosts still call follow_target() with
        # no argument via the normal move_ghosts() path -- we replicate that
        # loop here so we can intercept just one ghost.
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
                    g.follow_target(action_direction=direction)
                else:
                    g.follow_target()

            self.engine.check_ghosts_and_player_collision()
        elif self.engine.player.is_eaten():
            # The real game finishes this transition inside
            # play_death_animation(), which needs several ticks of a sprite
            # animation to run before it decrements lives and resets the
            # player. We don't need the animation headlessly, so we apply
            # the same state change directly and immediately.
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

        return self._get_obs(), reward, terminated, truncated, {"caught": caught, "distance": self._distance_to_player(), "mode": mode}
        
    def _compute_reward(self):
        reward = 0.0

        lives_now = self.engine.player.lives
        if lives_now < self._prev_lives:
            reward += 10.0  # ghost caught the player
        self._prev_lives = lives_now

        # Small shaping reward: closing distance on the player each tick.
        # Encourages proximity-seeking behaviour even between catches.
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

        return np.array([
            gx, gy, px, py,
            gx - px, gy - py,
            mode_code,
            float(self.engine.player.lives),
        ], dtype=np.float32)

    def _get_start_trigger(self):
        from settings import START_TRIGGER
        return START_TRIGGER