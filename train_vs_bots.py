"""
Train a ghost against the skill BOTS instead of a random player.

Each episode picks a random bot skill from the mix, so the ghost learns to
chase all kinds of players. Uses the same curriculum (spawn close, move out)
and VecNormalize persistence that worked before.

Run from the repo root:
    python -u train_vs_bots.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.time.wait = lambda ms: 0  # skip the 500ms freeze on each catch

import torch
torch.set_num_threads(4)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

BOT_SKILLS = [0.0, 0.25, 0.5, 0.75, 1.0]   # the mix
STEPS_PER_PHASE = 100_000
NUM_PHASES = 6
START_DISTANCE = 3
NAME = "ghost_vs_bots_team"
TRAINED_NAME = "Blinky"

PHASE_DISTANCES = [
    round(START_DISTANCE + (REAL_BLINKY_DISTANCE - START_DISTANCE) * i / (NUM_PHASES - 1))
    for i in range(NUM_PHASES)
]
PHASE_DISTANCES[-1] = REAL_BLINKY_DISTANCE


class CatchCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.own = 0
        self.team = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if info.get("caught"):
                self.team += 1
                if info.get("caught_by") == TRAINED_NAME:
                    self.own += 1
            if done:
                self.episodes += 1
        return True


def make_env(distance):
    def _make():
        return Monitor(PacmanGhostEnv(curriculum_distance=distance,
                                      solo_ghost= False,
                                      player_bot_skills=BOT_SKILLS))
    return DummyVecEnv([_make])


def main():
    print(f"Training vs bot mix {BOT_SKILLS}")
    print(f"Phase distances: {PHASE_DISTANCES}")

    vec_env = VecNormalize(make_env(PHASE_DISTANCES[0]),
                           norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=0)
    tmp = f"{NAME}_tmp.pkl"

    for phase_num, distance in enumerate(PHASE_DISTANCES, start=1):
        if phase_num > 1:
            vec_env.save(tmp)
            vec_env = VecNormalize.load(tmp, make_env(distance))
            vec_env.training = True
            model.set_env(vec_env)

        counter = CatchCounterCallback()
        model.learn(total_timesteps=STEPS_PER_PHASE, callback=counter,
                    reset_num_timesteps=False)
        own_rate = 100 * counter.own / max(counter.episodes, 1)
        team_rate = 100 * counter.team / max(counter.episodes, 1)
        print(f"Phase {phase_num}/{NUM_PHASES} distance {distance:>2}: "
              f"{counter.episodes} episodes | our ghost {counter.own} "
              f"({own_rate:.1f}%) | whole team {counter.team} "
              f"({team_rate:.1f}%)", flush=True)
    model.save(NAME)
    vec_env.save(f"{NAME}_stats.pkl")
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"\nSaved {NAME}.zip and {NAME}_stats.pkl")


if __name__ == "__main__":
    main()