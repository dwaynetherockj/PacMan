"""
Diagnostic v2: tracks the closest distance per episode AND the ghost's
mode (chase/scatter/frightened/eaten) at that closest moment. If close
approaches disproportionately happen during "frightened" mode, that
explains why proximity isn't converting into catches -- frightened ghosts
can't catch the player even at distance 0, per the game's own rules.

Run from the repo root:
    python test_mode_diagnostic.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv

FIXED_DISTANCE = 3
TOTAL_TIMESTEPS = 20_000
LOG_FREQ = 2_000


class ModeAtClosestCallback(BaseCallback):
    def __init__(self, log_freq=LOG_FREQ, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.current_episode_min = float("inf")
        self.current_episode_mode_at_min = None
        self.mode_counts_at_closest = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            dist = info.get("distance")
            mode = info.get("mode")
            if dist is not None and dist < self.current_episode_min:
                self.current_episode_min = dist
                self.current_episode_mode_at_min = mode
            if done:
                if self.current_episode_mode_at_min is not None:
                    key = self.current_episode_mode_at_min
                    self.mode_counts_at_closest[key] = self.mode_counts_at_closest.get(key, 0) + 1
                self.current_episode_min = float("inf")
                self.current_episode_mode_at_min = None

        if self.num_timesteps % self.log_freq == 0 and self.mode_counts_at_closest:
            print(f"[Mode tracker] Step {self.num_timesteps}: "
                  f"mode at closest-approach across episodes so far: {self.mode_counts_at_closest}")

        return True


def main():
    print(f"Mode diagnostic at fixed distance {FIXED_DISTANCE}, {TOTAL_TIMESTEPS} steps.")

    def make_env():
        return Monitor(PacmanGhostEnv(curriculum_distance=FIXED_DISTANCE))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=0)
    tracker = ModeAtClosestCallback()

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=tracker)

    print("\n--- Result ---")
    print(f"Mode at closest-approach, tallied across all episodes: {tracker.mode_counts_at_closest}")

    total = sum(tracker.mode_counts_at_closest.values())
    frightened_count = tracker.mode_counts_at_closest.get("frightened", 0)
    if total > 0 and frightened_count / total > 0.3:
        print(f"Frightened mode accounted for {frightened_count}/{total} closest-approaches "
              f"({100*frightened_count/total:.0f}%) -- this strongly supports the "
              f"'frightened mode is blocking catches' theory.")
    else:
        print("Frightened mode does not dominate closest-approaches -- "
              "the theory needs revisiting.")


if __name__ == "__main__":
    main()