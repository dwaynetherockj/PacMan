"""
Sanity test for PacmanGhostEnv -- NOT training, just proving step() works.

Runs 200 random actions and prints reward stats. If this completes without
crashing, the environment is ready for a real PPO training run.

Run from the repo root:
    python test_env.py
"""

from rl_env.pacman_ghost_env import PacmanGhostEnv

NUM_STEPS = 200


def main():
    env = PacmanGhostEnv()
    obs, info = env.reset()
    print(f"Initial obs: {obs}")

    total_reward = 0.0
    catches = 0

    for step in range(NUM_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward >= 10.0:
            catches += 1

        if terminated or truncated:
            print(f"Episode ended at step {step+1} (terminated={terminated}, truncated={truncated})")
            obs, info = env.reset()

    print("\n--- Environment sanity test results ---")
    print(f"Steps run:       {NUM_STEPS}")
    print(f"Total reward:    {total_reward:.2f}")
    print(f"Average reward:  {total_reward / NUM_STEPS:.4f}")
    print(f"Catches:         {catches}")
    print("PASSED: step() ran without crashing across all steps.")


if __name__ == "__main__":
    main()