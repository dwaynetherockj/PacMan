"""
Train one cohort ghost (casual or hardcore) at one seed, all four ghosts
active, against the bot mix. Run several of these in parallel for seeds.

Usage:
    python -u train_cohort.py --cohort casual   --seed 0
    python -u train_cohort.py --cohort hardcore --seed 0
"""

import argparse
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.time.wait = lambda ms: 0

import torch
torch.set_num_threads(2)   # keep each run light so parallel runs share the CPU

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

BOT_SKILLS = [0.0, 0.25, 0.5, 0.75, 1.0]
STEPS_PER_PHASE = 100_000
NUM_PHASES = 6
START_DISTANCE = 3
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


def make_env(distance, cohort):
    def _make():
        return Monitor(PacmanGhostEnv(curriculum_distance=distance,
                                      solo_ghost=False,
                                      player_bot_skills=BOT_SKILLS,
                                      cohort=cohort))
    return DummyVecEnv([_make])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=["casual", "hardcore"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    name = f"ghost_{args.cohort}_s{args.seed}"
    print(f"Training {name}: cohort={args.cohort} seed={args.seed}")
    print(f"Phase distances: {PHASE_DISTANCES}")

    vec_env = VecNormalize(make_env(PHASE_DISTANCES[0], args.cohort),
                           norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=args.seed)
    tmp = f"{name}_tmp.pkl"

    for phase_num, distance in enumerate(PHASE_DISTANCES, start=1):
        if phase_num > 1:
            vec_env.save(tmp)
            vec_env = VecNormalize.load(tmp, make_env(distance, args.cohort))
            vec_env.training = True
            model.set_env(vec_env)

        counter = CatchCounterCallback()
        model.learn(total_timesteps=STEPS_PER_PHASE, callback=counter,
                    reset_num_timesteps=False)
        own_rate = 100 * counter.own / max(counter.episodes, 1)
        team_rate = 100 * counter.team / max(counter.episodes, 1)
        print(f"[{name}] Phase {phase_num}/{NUM_PHASES} dist {distance:>2}: "
              f"our {counter.own} ({own_rate:.1f}%) | team {counter.team} "
              f"({team_rate:.1f}%)", flush=True)

    model.save(name)
    vec_env.save(f"{name}_stats.pkl")
    if os.path.exists(tmp):
        os.remove(tmp)
    print(f"[{name}] Saved {name}.zip and {name}_stats.pkl", flush=True)


if __name__ == "__main__":
    main()