"""
Playable ADAPTIVE Pac-Man (study game B) with blind logging.

At game over it writes a file: each life's duration, points and pellets in
plain English, plus a neutral "stat" number for the speed that life was
played at.

Stat key (RESEARCHER ONLY): 0 = normal, 1 = easy, 3 = hard.

Run from the project folder:
    python play_adaptive.py
"""

import uuid

from pacman import Game
from rl_env.controller import AdaptiveController

GAME_LABEL = "B"   # A = original, B = adaptive, C = super-hard
DIFFICULTY_CODE = {"normal": 0, "casual": 1, "skilled": 3}
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def fmt_time(ticks):
    seconds = int(ticks) // 60          # 60 ticks per second
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_log(controller, final_score, won):
    filename = f"session_{uuid.uuid4().hex[:8]}.txt"
    feats = controller.feature_history
    decisions = [str(band) for _, band in controller.history]
    # Life 1 runs at normal speed. Each later life runs at the speed
    # decided after the life before it.
    played_at = ["normal"] + decisions[:-1]

    result = "board cleared" if won else "out of lives"
    lines = [f"Game: {GAME_LABEL}", f"Final score: {final_score}", f"Result: {result}", ""]
    for i, f in enumerate(feats):
        label = ORDINALS[i] if i < len(ORDINALS) else f"{i + 1}th"
        lines.append(f"{label} life duration: {fmt_time(f['life_ticks'])}")
        lines.append(f"{label} life points: {f.get('score_this_life', 'n/a')}")
        lines.append(f"{label} life pellets eaten: {f.get('pellets', 'n/a')}")
        lines.append(f"{label} life stat: {DIFFICULTY_CODE[played_at[i]]}")
        lines.append("")

    with open(filename, "w") as fh:
        fh.write("\n".join(lines))

    # TEST ONLY: show what the classifier saw. Remove before the study.
    for i, (f, d) in enumerate(zip(feats, decisions), start=1):
        print(f"life {i}: decided {d} | {f}")
    return filename


def main():
    game = Game()
    controller = AdaptiveController(game.game_engine)
    logged = False

    game.game_start_sfx.play()
    while True:
        game.check_events()

        # SPACE after game over builds a brand-new game. Attach a fresh
        # controller to it, or the new game would not adapt or be logged.
        if game.game_engine is not controller.engine:
            controller = AdaptiveController(game.game_engine)
            logged = False

        game.update()
        game.draw()
        controller.update()

        if game.game_engine.game_over and not logged:
            if game.game_engine.game_won:
                controller.close_final_life()
            fname = write_log(controller, game.game_engine.level.score,
                              game.game_engine.game_won)
            print(f"Session saved to {fname} -- please send this file to the researcher.")
            logged = True


if __name__ == "__main__":
    main()