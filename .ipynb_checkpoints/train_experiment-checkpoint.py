"""
Curriculum PPO training with switches for controlled experiments.

Run from the repo root, for example:
    python -u train_experiment.py --name solo_base --solo
    python -u train_experiment.py --name solo_nav  --solo --nav

Outputs are named after --name: <name>.zip and <name>_stats.pkl
"""

import argparse
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random

import pygame
pygame.time.wait = lambda ms: 0  # skip the game's 500 ms freeze on each catch

import torch
torch.set_num_threads(4)  # limit CPU threads so parallel runs don't compete

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

NUM_PHASES = 6
START_DISTANCE = 3
TRAINED_NAME = "Blinky"

PHASE_DISTANCES = [
    round(START_DISTANCE + (REAL_BLINKY_DISTANCE - START_DISTANCE) * i / (NUM_PHASES - 1))
    for i in range(NUM_PHASES)
]
PHASE_DISTANCES[-1] = REAL_BLINKY_DISTANCE


class CatchCounterCallback(BaseCallback):
    """Counts episodes, catches by the trained ghost, and catches by other ghosts."""

    def __init__(self):
        super().__init__()
        self.own = 0
        self.other = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if info.get("caught"):
                if info.get("caught_by") == TRAINED_NAME:
                    self.own += 1
                else:
                    self.other += 1
            if done:
                self.episodes += 1
        return True


def make_env(distance, solo, nav):
    def _make():
        return Monitor(PacmanGhostEnv(curriculum_distance=distance,
                                      solo_ghost=solo, nav_obs=nav))
    return DummyVecEnv([_make])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--solo", action="store_true", help="other three ghosts do not move")
    parser.add_argument("--nav", action="store_true", help="add open-direction flags")
    parser.add_argument("--steps", type=int, default=100_000, help="steps per phase")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    tmp_path = f"{args.name}_tmp_stats.pkl"
    print(f"Run '{args.name}': solo={args.solo} nav={args.nav} "
          f"steps/phase={args.steps} seed={args.seed}")
    print(f"Phase distances (tiles): {PHASE_DISTANCES}")

    vec_env = VecNormalize(make_env(PHASE_DISTANCES[0], args.solo, args.nav),
                           norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=args.seed)
    results = []

    for phase_num, distance in enumerate(PHASE_DISTANCES, start=1):
        if phase_num > 1:
            # Keep the normalisation stats across phases (save, then reload).
            vec_env.save(tmp_path)
            vec_env = VecNormalize.load(tmp_path,
                                        make_env(distance, args.solo, args.nav))
            vec_env.training = True
            model.set_env(vec_env)

        counter = CatchCounterCallback()
        model.learn(total_timesteps=args.steps, callback=counter,
                    reset_num_timesteps=False)

        rate = 100 * counter.own / max(counter.episodes, 1)
        print(f"Phase {phase_num}/{NUM_PHASES} distance {distance:>2}: "
              f"{counter.episodes} episodes, Blinky catches {counter.own} "
              f"({rate:.1f}% of episodes), other-ghost catches {counter.other}")
        results.append((distance, counter.episodes, counter.own, counter.other))

    model.save(args.name)
    vec_env.save(f"{args.name}_stats.pkl")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"Saved {args.name}.zip and {args.name}_stats.pkl")


if __name__ == "__main__":
    main()