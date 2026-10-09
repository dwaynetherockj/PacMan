"""
Generate training data for the player skill model.

Runs many bot games across all skill values. For each LIFE a bot loses,
writes one summary row (the features the classifier reads), labelled with
the bot's skill band (casual / middle / hardcore).

Run from the repo root:
    python -u make_player_data.py
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import csv
import random

import pygame
pygame.time.wait = lambda ms: 0

from pacman import Game
from rl_env.bots import SkillBot
from model.direction import Direction

SKILLS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMES_PER_SKILL = 30
MAX_TICKS = 10000
OUT = "player_data.csv"

OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def band(skill):
    if skill <= 0.33:
        return "casual"
    if skill <= 0.66:
        return "middle"
    return "hardcore"


def play_and_summarise(skill, seed, writer):
    random.seed(seed)
    game = Game()
    engine = game.game_engine
    engine.player.set_to_chase()
    bot = SkillBot(skill, seed=seed)
    board = engine.player.board

    # Per-life accumulators, reset each time a life is lost.
    def fresh():
        return {"ticks": 0, "score_start": engine.level.score,
                "pellets_start": int(((board == 1) | (board == 2)).sum()),
                "reversals": 0, "near_ghost_ticks": 0, "ghost_dist_sum": 0.0,
                "blue_eaten": 0, "distance_moved": 0.0,
                "last_x": engine.player.location_x, "last_y": engine.player.location_y,
                "powerup_tick": None, "reaction_ticks": None}
    life = fresh()
    prev_dir = None
    lives_seen = engine.player.lives
    prev_eaten = set()

    for t in range(1, MAX_TICKS + 1):
        engine.direction_command = bot.choose(engine)
        engine.tick()
        life["ticks"] += 1

        d = engine.player.direction.name
        if prev_dir and OPPOSITE.get(prev_dir) == d:
            life["reversals"] += 1
        prev_dir = d

                # distance the player actually moved this tick
        px0, py0 = engine.player.location_x, engine.player.location_y
        life["distance_moved"] += abs(px0 - life["last_x"]) + abs(py0 - life["last_y"])
        life["last_x"], life["last_y"] = px0, py0

        # note when a power pellet was eaten (powerup just turned on)
        if engine.player.powerup and life["powerup_tick"] is None:
            life["powerup_tick"] = life["ticks"]
        if not engine.player.powerup:
            life["powerup_tick"] = None  # window ended, reset for next pellet

        # nearest dangerous ghost distance this tick
        px, py = engine.player.location_x, engine.player.location_y
        nearest = min(
            (abs(g.location_x - px) + abs(g.location_y - py)
             for g in engine.ghosts
             if not (g.is_frightened() or g.is_eaten())),
            default=9999)
        life["ghost_dist_sum"] += nearest
        if nearest < engine.tile_width:
            life["near_ghost_ticks"] += 1

        # count a blue ghost eaten (ghost newly in "eaten" state)
        now_eaten = {id(g) for g in engine.ghosts if g.is_eaten()}
        newly = now_eaten - prev_eaten
        life["blue_eaten"] += len(newly)
        # record reaction time on the FIRST blue ghost eaten after a power pellet
        if newly and life["powerup_tick"] is not None and life["reaction_ticks"] is None:
            life["reaction_ticks"] = life["ticks"] - life["powerup_tick"]
        prev_eaten = now_eaten

        # a life was just lost -> write the summary, start a new life
        if engine.player.lives < lives_seen:
            write_life(writer, skill, life, engine, board)
            lives_seen = engine.player.lives
            life = fresh()
            prev_dir = None

        if engine.game_over:
            break


def write_life(writer, skill, life, engine, board):
    ticks = max(life["ticks"], 1)
    score_gained = engine.level.score - life["score_start"]
    pellets_now = int(((board == 1) | (board == 2)).sum())
    pellets_eaten = life["pellets_start"] - pellets_now
    writer.writerow([
        round(skill, 2), band(skill),
        ticks,
        round(score_gained / ticks, 4),            # score rate
        pellets_eaten,
        round(pellets_eaten / ticks, 4),           # pellet rate
        life["blue_eaten"],                        # power-pellet reaction (count)
        round(life["ghost_dist_sum"] / ticks, 2),  # avg distance to danger
        life["near_ghost_ticks"],                  # ticks spent in danger
        round(life["reversals"] / ticks, 4),       # dithering rate
        round(pellets_eaten / max(life["distance_moved"], 1), 5),  # pellet efficiency
        life["reaction_ticks"] if life["reaction_ticks"] is not None else "",  # reaction (ticks), blank = never
        1 if life["blue_eaten"] > 0 else 0,        # ate_blue flag
    ])


def main():
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "skill", "band", "life_ticks", "score_rate", "pellets_eaten",
            "pellet_rate", "blue_eaten", "avg_ghost_dist", "near_ghost_ticks",
            "reversal_rate", "pellet_efficiency", "reaction_ticks", "ate_blue"])
        for skill in SKILLS:
            for g in range(GAMES_PER_SKILL):
                play_and_summarise(skill, seed=g, writer=writer)
            print(f"skill {skill:.2f} done", flush=True)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()