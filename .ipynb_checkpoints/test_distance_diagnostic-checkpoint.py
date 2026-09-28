"""
Diagnostic: tracks the CLOSEST DISTANCE achieved per episode, not just
whether a catch happened. This tells us whether the ghost is getting close
and missing the final precise step, or never meaningfully approaching the
player at all -- two very different problems needing different fixes.

Run from the repo root:
    python test_distance_diagnostic.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv

FIXED_DISTANCE = 3
TOTAL_TIMESTEPS = 30_000
LOG_FREQ = 2_000
CATCH_THRESHOLD_ROUGH = 14  # sqrt(10^2 + 10^2), a rough circular equivalent
                             # of the real square catch test (DISTANCE_FACTOR=10)


class DistanceTrackerCallback(BaseCallback):
    def __init__(self, log_freq=LOG_FREQ, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.current_episode_min = float("inf")
        self.episode_min_distances = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False])

        for info, done in zip(infos, dones):
            dist = info.get("distance")
            if dist is not None and dist < self.current_episode_min:
                self.current_episode_min = dist
            if done:
                self.episode_min_distances.append(self.current_episode_min)
                self.current_episode_min = float("inf")

        if self.num_timesteps % self.log_freq == 0 and self.episode_min_distances:
            recent = self.episode_min_distances[-20:]
            avg_min = sum(recent) / len(recent)
            best_min = min(self.episode_min_distances)
            print(f"[Distance tracker] Step {self.num_timesteps}: "
                  f"avg closest approach (last {len(recent)} eps) = {avg_min:.1f}px, "
                  f"best ever = {best_min:.1f}px "
                  f"(rough catch threshold ~{CATCH_THRESHOLD_ROUGH}px)")

        return True


def main():
    print(f"Distance diagnostic at fixed distance {FIXED_DISTANCE}, {TOTAL_TIMESTEPS} steps.")

    def make_env():
        return Monitor(PacmanGhostEnv(curriculum_distance=FIXED_DISTANCE))

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    tracker = DistanceTrackerCallback()

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=tracker)

    print("\n--- Result ---")
    if tracker.episode_min_distances:
        overall_best = min(tracker.episode_min_distances)
        overall_avg = sum(tracker.episode_min_distances) / len(tracker.episode_min_distances)
        print(f"Best closest-approach across all episodes: {overall_best:.1f}px")
        print(f"Average closest-approach across all episodes: {overall_avg:.1f}px")
        print(f"Rough catch threshold: ~{CATCH_THRESHOLD_ROUGH}px")

        if overall_best < CATCH_THRESHOLD_ROUGH * 2:
            print("The ghost IS getting reasonably close -- this looks like a "
                  "precision/final-approach problem, not a 'not trying' problem.")
        else:
            print("The ghost is NOT getting particularly close even at its best -- "
                  "this looks like a more basic learning/reward problem.")
    else:
        print("No episode data collected -- something went wrong with tracking.")


if __name__ == "__main__":
    main()