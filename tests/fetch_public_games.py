"""Download a few public ARC-AGI-3 games for the offline tests.

SPDX-License-Identifier: MIT-0

Uses the arc-agi toolkit's anonymous API key (free; no account, no cost).
The game files themselves belong to ARC Prize and are NOT redistributed here.
"""
import os
import sys
from pathlib import Path

import arc_agi

GAMES = sys.argv[1:] or ["ls20", "ft09", "vc33"]
target = Path(__file__).resolve().parent / ".arc_envs"
target.mkdir(exist_ok=True)
os.chdir(target)
arcade = arc_agi.Arcade(environments_dir=str(target))
for game in GAMES:
    print(game, "ok" if arcade.make(game) is not None else "FAILED")
