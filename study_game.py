"""
Shared code for the three study games (A, B, C).

All three play and log in exactly the same way, so a player cannot tell
them apart. Only the ghost speed rule differs:
  A = normal speed all game (the original game)
  B = adaptive: speed changes after each life
  C = hard speed all game

Stat key (RESEARCHER ONLY): 0 = normal, 1 = easy, 3 = hard.
"""

import uuid

from pacman import Game
from rl_env.controller import AdaptiveController, BAND_SETTINGS

STAT_CODE = {"normal": 0, "casual": 1, "skilled": 3}
ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


class FixedSpeedController(AdaptiveController):
    """Records each life exactly like the adaptive one, but never changes speed."""
    def _apply(self, band):
        pass


def make_controller(game, fixed_band):
    if fixed_band is None:
        return AdaptiveController(game.game_engine)
    controller = FixedSpeedController(game.game_engine)
    mult = 1.0 if fixed_band == "normal" else BAND_SETTINGS[fixed_band]["ghost_speed_mult"]
    for g in game.game_engine.ghosts:
        g.speed_mult = mult
    return controller


def fmt_time(ticks):
    seconds = int(ticks) // 60          # 60 ticks per second
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def write_log(game_label, controller, final_score, won, fixed_band):
    filename = f"session_{uuid.uuid4().hex[:8]}.txt"
    feats = controller.feature_history
    if fixed_band is None:
        # Adaptive: life 1 is normal, each later life uses the speed
        # decided after the life before it.
        decisions = [str(band) for _, band in controller.history]
        played_at = ["normal"] + decisions[:-1]
    else:
        played_at = [fixed_band] * len(feats)

    result = "board cleared" if won else "out of lives"
    lines = [f"Game: {game_label}", f"Final score: {final_score}", f"Result: {result}", ""]
    for i, f in enumerate(feats):
        label = ORDINALS[i] if i < len(ORDINALS) else f"{i + 1}th"
        lines.append(f"{label} life duration: {fmt_time(f['life_ticks'])}")
        lines.append(f"{label} life points: {f.get('score_this_life', 'n/a')}")
        lines.append(f"{label} life pellets eaten: {f.get('pellets', 'n/a')}")
        lines.append(f"{label} life stat: {STAT_CODE[played_at[i]]}")
        lines.append("")

    with open(filename, "w") as fh:
        fh.write("\n".join(lines))
    return filename


def run(game_label, fixed_band=None):
    """fixed_band: None = adaptive, 'normal' or 'skilled' = that speed all game."""
    game = Game()
    controller = make_controller(game, fixed_band)
    logged = False

    game.game_start_sfx.play()
    while True:
        game.check_events()

        # SPACE after game over builds a new game: give it a fresh controller.
        if game.game_engine is not controller.engine:
            controller = make_controller(game, fixed_band)
            logged = False

        game.update()
        game.draw()
        controller.update()

        if game.game_engine.game_over and not logged:
            if game.game_engine.game_won:
                controller.close_final_life()
            fname = write_log(game_label, controller, game.game_engine.level.score,
                              game.game_engine.game_won, fixed_band)
            print(f"Session saved to {fname} -- please send this file to the researcher.")
            logged = True