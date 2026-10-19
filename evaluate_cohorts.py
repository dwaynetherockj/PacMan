"""
Test casual vs hardcore ghosts against weak and strong players.

All four ghosts run the cohort policy (via engine.ghost_policy). Plays normal
games and measures how hard the ghosts are:
  - catch rate (did the player lose a life)
  - survival time (ticks before caught; capped games count as the cap)
  - average distance the ghosts kept to the player (smaller = more pressure)

Run from the repo root:
    python -u evaluate_cohorts.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import pygame
pygame.time.wait = lambda ms: 0

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from pacman import Game
from rl_env.bots import SkillBot
from rl_env.pacman_ghost_env import PacmanGhostEnv
from model.direction import Direction

GHOSTS = {
    "casual":   ("ghost_casual_s0",   "ghost_casual_s0_stats.pkl"),
    "hardcore": ("ghost_hardcore_s0", "ghost_hardcore_s0_stats.pkl"),
}
PLAYER_SKILLS = {"weak": 0.0, "strong": 1.0}
GAMES = 15
MAX_TICKS = 6000


def load_policy(model_path, stats_path):
    model = PPO.load(model_path)
    dummy = DummyVecEnv([lambda: PacmanGhostEnv(solo_ghost=False)])
    norm = VecNormalize.load(stats_path, dummy)
    norm.training = False
    norm.norm_reward = False
    return model, norm


def make_policy_fn(model, norm, engine):
    def policy(ghost):
        gx, gy = ghost.location_x, ghost.location_y
        px, py = engine.player.location_x, engine.player.location_y
        mode = 0.0
        if ghost.is_scatter(): mode = 1.0
        elif ghost.is_frightened(): mode = 2.0
        elif ghost.is_eaten(): mode = 3.0
        obs = np.array([gx, gy, px, py, gx - px, gy - py, mode,
                        float(engine.player.lives)], dtype=np.float32)
        action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
        return Direction(int(action))
    return policy


def play(model, norm, player_skill, seed):
    random.seed(seed)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    engine.ghost_policy = make_policy_fn(model, norm, engine)

    bot = SkillBot(player_skill, seed=seed)
    start_lives = engine.player.lives
    dist_sum = 0.0

    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()

        px, py = engine.player.location_x, engine.player.location_y
        dist_sum += min(abs(g.location_x - px) + abs(g.location_y - py)
                        for g in engine.ghosts)

        if engine.player.lives < start_lives:
            return {"caught": True, "ticks": t, "avg_dist": dist_sum / t}

    return {"caught": False, "ticks": MAX_TICKS, "avg_dist": dist_sum / MAX_TICKS}


def main():
    print(f"{GAMES} games per cell. Harder ghost = higher catch%, "
          f"shorter survival, smaller distance.\n")
    print(f"{'ghost':<10}{'player':<8}{'catch%':>8}{'survived':>10}{'avg_dist':>10}")
    for gname, (mp, sp) in GHOSTS.items():
        model, norm = load_policy(mp, sp)
        for pname, pskill in PLAYER_SKILLS.items():
            res = [play(model, norm, pskill, seed=s) for s in range(GAMES)]
            catch = 100 * sum(r["caught"] for r in res) / GAMES
            ticks = sum(r["ticks"] for r in res) / GAMES
            dist = sum(r["avg_dist"] for r in res) / GAMES
            print(f"{gname:<10}{pname:<8}{catch:>7.0f}%{ticks:>10.0f}{dist:>10.0f}",
                  flush=True)


if __name__ == "__main__":
    main()