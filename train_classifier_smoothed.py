"""
Two-class player classifier with multi-life smoothing.

For each life, averages the features over the last N lives WITHIN THE SAME
GAME (grouped by game_id), then classifies casual vs skilled. Tests several
window sizes so we can see accuracy improve as we aggregate more evidence.

Keeps the single-life (N=1) number so we can report the improvement.

Run from the repo root:
    python train_classifier_smoothed.py
"""

import csv
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["life_ticks", "blue_eaten", "reaction_filled", "ate_blue",
            "near_ghost_ticks", "score_rate"]
WINDOWS = [1, 2, 3, 5]


def two_class(band):
    return "casual" if band == "casual" else "skilled"


def row_feats(r):
    reaction = float(r["reaction_ticks"]) if r["reaction_ticks"] != "" else 600.0
    return {
        "life_ticks": float(r["life_ticks"]),
        "blue_eaten": float(r["blue_eaten"]),
        "reaction_filled": reaction,
        "ate_blue": float(r["ate_blue"]),
        "near_ghost_ticks": float(r["near_ghost_ticks"]),
        "score_rate": float(r["score_rate"]),
    }


def load_games():
    rows = list(csv.DictReader(open("player_data.csv")))
    games = defaultdict(list)
    for r in rows:
        games[r["game_id"]].append(r)
    return games


def build_dataset(games, window):
    """For each life, average features over the last `window` lives in its game."""
    X, y, game_ids = [], [], []
    for gid, lives in games.items():
        feat_history = []
        for r in lives:
            feat_history.append(row_feats(r))
            recent = feat_history[-window:]
            avg = {f: np.mean([h[f] for h in recent]) for f in FEATURES}
            X.append([avg[f] for f in FEATURES])
            y.append(two_class(r["band"]))
            game_ids.append(gid)
    return np.array(X), np.array(y), np.array(game_ids)


def main():
    games = load_games()
    gids = list(games.keys())
    cut = int(len(gids) * 0.8)
    train_gids = set(gids[:cut])

    print("Accuracy by smoothing window (lives averaged):\n")
    print(f"{'window':>8}{'accuracy':>12}")
    for w in WINDOWS:
        X, y, g = build_dataset(games, w)
        tr = np.array([gid in train_gids for gid in g])
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[~tr])
        acc = (pred == y[~tr]).mean()
        print(f"{w:>8}{acc*100:>11.1f}%")


if __name__ == "__main__":
    main()
    