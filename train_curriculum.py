"""
Staged curriculum training for PacmanGhostEnv -- CORRECTED persistence fix.

VecNormalize.set_venv() only works on a freshly-loaded (uninitialized)
wrapper, not a live one -- so instead we save the running stats to disk at
each phase boundary and reload them into a NEW VecNormalize wrapping the
next phase's environment. This achieves the same goal (stats persist
across phases) via the officially supported save/load path.

Run from the repo root:
    python train_curriculum.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv
from rl_env.maze_distance import REAL_BLINKY_DISTANCE

STEPS_PER_PHASE = 10_000
NUM_PHASES = 6
START_DISTANCE = 3
STATS_TMP_PATH = "vecnormalize_tmp.pkl"

PHASE_DISTANCES = [
    round(START_DISTANCE + (REAL_BLINKY_DISTANCE - START_DISTANCE) * i / (NUM_PHASES - 1))
    for i in range(NUM_PHASES)
]
PHASE_DISTANCES[-1] = REAL_BLINKY_DISTANCE


class CatchCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_catches = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("caught"):
                self.total_catches += 1
        return True


def make_dummy_vec_env(curriculum_distance):
    def _make():
        return Monitor(PacmanGhostEnv(curriculum_distance=curriculum_distance))
    return DummyVecEnv([_make])


def main():
    print(f"Curriculum phases (maze-aware distances, tiles): {PHASE_DISTANCES}")

    vec_env = make_dummy_vec_env(PHASE_DISTANCES[0])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=0)
    phase_results = []

    for phase_num, distance in enumerate(PHASE_DISTANCES, start=1):
        print(f"\n=== Phase {phase_num}/{NUM_PHASES}: spawn distance = {distance} tiles ===")

        if phase_num > 1:
            vec_env.save(STATS_TMP_PATH)
            new_dummy_env = make_dummy_vec_env(distance)
            vec_env = VecNormalize.load(STATS_TMP_PATH, new_dummy_env)
            vec_env.training = True
            model.set_env(vec_env)

        catch_callback = CatchCounterCallback()
        model.learn(
            total_timesteps=STEPS_PER_PHASE,
            callback=catch_callback,
            reset_num_timesteps=False,
        )

        print(f"Phase {phase_num} catches in {STEPS_PER_PHASE} steps: {catch_callback.total_catches}")
        phase_results.append((distance, catch_callback.total_catches))

    model.save("ppo_blinky_curriculum")
    vec_env.save("vecnormalize_curriculum_stats.pkl")

    print("\n--- Curriculum summary ---")
    for distance, catches in phase_results:
        print(f"Distance {distance:>3} tiles: {catches} catches")

    print("\nModel saved as ppo_blinky_curriculum.zip")


if __name__ == "__main__":
    main()