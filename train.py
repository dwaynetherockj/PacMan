"""
PPO training run for PacmanGhostEnv, with two additions on top of the
original proving run:

1. Observation normalization (VecNormalize) -- puts position/distance
   values on a more consistent scale for the network to learn from.
2. A catch counter -- explicitly tracks how many times the ghost actually
   catches the player during training, rather than inferring it from the
   reward trend alone.

Run from the repo root:
    python train.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env.pacman_ghost_env import PacmanGhostEnv

TOTAL_TIMESTEPS = 50_000
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 2_000
CATCH_LOG_FREQ = 2_000  # print catch count every N steps


class CatchCounterCallback(BaseCallback):
    """Tracks how many times the ghost actually caught the player."""

    def __init__(self, log_freq=CATCH_LOG_FREQ, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.total_catches = 0
        self.catches_since_last_log = 0

    def _on_step(self) -> bool:
        # self.locals["infos"] is a list of info dicts, one per parallel env
        # (we only have one env, so it's a list of length 1).
        for info in self.locals.get("infos", []):
            if info.get("caught"):
                self.total_catches += 1
                self.catches_since_last_log += 1

        if self.num_timesteps % self.log_freq == 0:
            print(f"[Catch counter] Step {self.num_timesteps}: "
                  f"{self.catches_since_last_log} catches in the last {self.log_freq} steps "
                  f"(total so far: {self.total_catches})")
            self.catches_since_last_log = 0

        return True  # returning False would stop training early


def main():
    def make_env():
        return Monitor(PacmanGhostEnv())

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_blinky",
    )
    catch_callback = CatchCounterCallback()

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=None,
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, catch_callback],
    )

    model.save("ppo_blinky_first_run")
    vec_env.save("vecnormalize_stats.pkl")  # needed to reuse the same normalization later

    print(f"\nTraining complete. Model saved as ppo_blinky_first_run.zip")
    print(f"Total catches during training: {catch_callback.total_catches}")


if __name__ == "__main__":
    main()