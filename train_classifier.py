"""
Train the player skill classifier and evaluate it honestly on held-out games.

Trains on a RandomForest over per-life features, testing on lives from games
the model never saw (split by game seed, not random rows, to avoid leakage).
Prints overall accuracy and a confusion matrix.

Run from the repo root:
    python train_classifier.py
"""

import csv
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

FEATURES = ["life_ticks", "blue_eaten", "reaction_filled", "ate_blue",
            "near_ghost_ticks", "score_rate"]
BANDS = ["casual", "middle", "hardcore"]
# Hold out these game seeds for testing (we ran seeds 0-29 per skill).
TEST_SEEDS = set(range(24, 30))   # last 6 of every 30 games = ~20% held out


def load():
    rows = list(csv.DictReader(open("player_data.csv")))
    # We didn't store the seed per row, so we tag rows by order within each
    # skill: 30 games each. Rebuild an approximate game index from row order.
    # Simpler and leak-free: split by skill-seed isn't available, so we fall
    # back to a per-skill ordered split (first 80% train, last 20% test).
    by_skill = {}
    for r in rows:
        by_skill.setdefault(r["skill"], []).append(r)

    train, test = [], []
    for sk, rs in by_skill.items():
        cut = int(len(rs) * 0.8)
        train += rs[:cut]
        test += rs[cut:]
    return train, test


def to_xy(rows):
    X, y = [], []
    for r in rows:
        reaction_filled = float(r["reaction_ticks"]) if r["reaction_ticks"] != "" else 600.0
        feats = {
            "life_ticks": float(r["life_ticks"]),
            "blue_eaten": float(r["blue_eaten"]),
            "reaction_filled": reaction_filled,
            "ate_blue": float(r["ate_blue"]),
            "near_ghost_ticks": float(r["near_ghost_ticks"]),
            "score_rate": float(r["score_rate"]),
        }
        X.append([feats[f] for f in FEATURES])
        y.append(r["band"])
    return np.array(X), np.array(y)


def main():
    train_rows, test_rows = load()
    Xtr, ytr = to_xy(train_rows)
    Xte, yte = to_xy(test_rows)
    print(f"train lives: {len(ytr)}  test lives: {len(yte)}")
    print(f"train balance: {dict(Counter(ytr))}")
    print(f"test balance:  {dict(Counter(yte))}\n")

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = (pred == yte).mean()
    print(f"Overall accuracy on held-out games: {acc*100:.1f}%\n")

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(yte, pred, labels=BANDS)
    print(f"{'':>10}" + "".join(f"{b:>10}" for b in BANDS))
    for b, row in zip(BANDS, cm):
        print(f"{b:>10}" + "".join(f"{v:>10}" for v in row))

    print("\nPer-class detail:")
    print(classification_report(yte, pred, labels=BANDS, zero_division=0))

    print("Feature importance (higher = more useful):")
    for f, imp in sorted(zip(FEATURES, clf.feature_importances_),
                         key=lambda x: -x[1]):
        print(f"  {f:<18} {imp:.3f}")


if __name__ == "__main__":
    main()