"""
Run after each participant has finished all three games. Moves their
session files into participants/<ID>/ and checks one of each game is there.

Usage (from the project folder):
    python file_sessions.py P01
"""
import glob
import os
import shutil
import sys

if len(sys.argv) != 2:
    sys.exit("Give the participant ID, e.g.:  python file_sessions.py P01")

pid = sys.argv[1]
folder = os.path.join("participants", pid)
if os.path.exists(folder):
    sys.exit(f"{folder} already exists -- pick a new ID so nothing is mixed up.")

files = sorted(glob.glob("session_*.txt"), key=os.path.getmtime)
if not files:
    sys.exit("No session files found. Did the games reach Game Over or Board Cleared?")

games = []
for f in files:
    with open(f) as fh:
        first = fh.readline().strip()          # e.g. "Game: B"
    games.append(first.replace("Game:", "").strip())

os.makedirs(folder)
for f in files:
    shutil.move(f, folder)

print(f"Moved {len(files)} file(s) into {folder}, games in play order: {' '.join(games)}")
if games != ["A", "B", "C"]:
    print("CHECK THIS: expected exactly A B C in that order.")
