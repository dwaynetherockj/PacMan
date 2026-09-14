"""
First PPO training run for PacmanGhostEnv.

This is a SHORT run (10,000 steps) meant to prove the whole training loop
works end-to-end -- not to produce a good ghost. Once this completes without
error and the reward trend is visible, scale up total_timesteps and move
this to Colab for the real run.

Run from the repo root:
    python train.py
"""

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_env.pacman_ghost_env import PacmanGhostEnv

TOTAL_TIMESTEPS = 10_000  # short proving run -- raise this later
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_FREQ = 2_000


def main():
    env = PacmanGhostEnv()
    env = Monitor(env)  # tracks episode rewards/lengths for logging

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_blinky",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=None,  # keeping it simple for the first run
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    model.save("ppo_blinky_first_run")
    print("Training complete. Model saved as ppo_blinky_first_run.zip")


if __name__ == "__main__":
    main()