"""
Isolated test: train at a SINGLE fixed curriculum distance (close spawn,
no phase transitions at all) for a longer run than any single phase got in
train_curriculum.py. This isolates one variable: can the ghost learn to
catch a moving player from close range at all, given real time, with no
VecNormalize resets or environment swaps disrupting it?

Run from the repo root:
    python test_single_distance.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv

FIXED_DISTANCE = 3
TOTAL_TIMESTEPS = 30_000
LOG_FREQ = 2_000


class CatchCounterCallback(BaseCallback):
    def __init__(self, log_freq=LOG_FREQ, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_catches = 0
        self.catches_since_last_log = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("caught"):
                self.total_catches += 1
                self.catches_since_last_log += 1

        if self.num_timesteps % self.log_freq == 0:
            print(f"[Catch counter] Step {self.num_timesteps}: "
                  f"{self.catches_since_last_log} catches in the last {self.log_freq} steps "
                  f"(total so far: {self.total_catches})")
            self.catches_since_last_log = 0

        return True


def main():
    print(f"Testing a SINGLE fixed distance ({FIXED_DISTANCE} tiles), "
          f"no phase changes, for {TOTAL_TIMESTEPS} steps.")

    def make_env():
        return Monitor(PacmanGhostEnv(curriculum_distance=FIXED_DISTANCE))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    catch_callback = CatchCounterCallback()

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=catch_callback)

    print(f"\n--- Result ---")
    print(f"Total catches at fixed distance {FIXED_DISTANCE}: {catch_callback.total_catches}")
    if catch_callback.total_catches == 0:
        print("Even at close range with no phase changes, zero catches. "
              "This points to the moving player (or the reward/task itself) "
              "being the harder obstacle, not the phase-switching mechanism.")
    else:
        print("Catches happened once we removed phase switching. "
              "This points to the VecNormalize-reset-per-phase mechanism "
              "as the real problem in train_curriculum.py.")


if __name__ == "__main__":
    main()