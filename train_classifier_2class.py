"""
Two-class version of the player skill classifier: casual vs skilled.

Collapses the three bands into two:
  casual   = casual   (bot skills 0.0, 0.25)
  skilled  = middle + hardcore (bot skills 0.5, 0.75, 1.0)

Expecting this to score much higher than the 3-class version, because the
hard part before was telling middle from hardcore apart, and now I don't
need to. Keeps the original 3-class script untouched for comparison.

Run from the repo root:
    python train_classifier_2class.py
"""

import csv
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

FEATURES = ["life_ticks", "blue_eaten", "reaction_filled", "ate_blue",
            "near_ghost_ticks", "score_rate"]
BANDS = ["casual", "skilled"]


def two_class(band):
    # middle and hardcore both become "skilled"
    return "casual" if band == "casual" else "skilled"


def load():
    rows = list(csv.DictReader(open("player_data.csv")))
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
        y.append(two_class(r["band"]))
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
    print(f"Overall accuracy on held-out games (2-class): {acc*100:.1f}%\n")

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(yte, pred, labels=BANDS)
    print(f"{'':>10}" + "".join(f"{b:>10}" for b in BANDS))
    for b, row in zip(BANDS, cm):
        print(f"{b:>10}" + "".join(f"{v:>10}" for v in row))

    print("\nPer-class detail:")
    print(classification_report(yte, pred, labels=BANDS, zero_division=0))

    import joblib
    joblib.dump({"model": clf, "features": FEATURES}, "player_model_2class.joblib")
    print("Saved player_model_2class.joblib")


if __name__ == "__main__":
    main()