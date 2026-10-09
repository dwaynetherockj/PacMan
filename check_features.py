import csv
from collections import defaultdict

rows = list(csv.DictReader(open("player_data.csv")))
bysk = defaultdict(list)
for r in rows:
    bysk[r["skill"]].append(r)

print("New features by skill value:\n")
print(f"{'skill':<7}{'rows':>6}{'pellet_eff':>13}{'reaction(s)':>13}{'ate_blue%':>11}")
for sk in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
    rs = bysk.get(sk, [])
    if not rs:
        continue
    eff = sum(float(x["pellet_efficiency"]) for x in rs) / len(rs)
    reacts = [float(x["reaction_ticks"]) for x in rs if x["reaction_ticks"] != ""]
    react_s = (sum(reacts) / len(reacts)) / 60 if reacts else float("nan")
    ate = 100 * sum(int(x["ate_blue"]) for x in rs) / len(rs)
    print(f"{sk:<7}{len(rs):>6}{eff:>13.5f}{react_s:>13.2f}{ate:>11.1f}")