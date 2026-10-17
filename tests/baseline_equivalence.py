"""Gate OFF must reproduce the unpatched harness exactly.

SPDX-License-Identifier: MIT-0

Usage: python baseline_equivalence.py <unpatched_tree> <patched_tree>
Runs the same scripted mock-agent session (Franzen's competition guard
settings, SlideGame) in a fresh subprocess per tree and diffs every tool
result, the opener prompt and the system prompt (timing fields scrubbed).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRANZEN_ENV = {
    "ARC3_BATCH_NOOP_BLOCK": "1", "ARC3_STALE_STATE_BLOCK": "1", "ARC3_ACTION_ECHO": "1",
    "ARC3_NEW_CHANGED_PROMPTS": "1", "ARC3_GUARDS_FROM_LEVEL": "2",
    "ARC3_EXPLAIN_GAMEPLAY_CHANGED": "1", "ARC3_FRAME_DIFF_HINT": "1",
    "ARC3_LEVEL_TRANSFER_GUIDANCE": "1", "ARC3_ACTION_INFO": "1", "EXPOSE_UNDO": "on",
    "ARC3_MEMORY_SECTIONS": "off",
}
SNIPPETS = [
    'r = action(["UP", "RIGHT", "RIGHT"])\nprint(r.get("stop_reason"))',     # no-op stop
    'action(["UP"])\naction(["RIGHT"])',                                     # stale latch
    'r = action(["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])\nprint(r.get("stop_reason"))',
    'try:\n    action(["RIGHT"], expect=lambda p, n: True)\nexcept TypeError as e:\n    print("TypeError", e)',
    'r = action(["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"])\nprint(r.get("level_completed"))',
]
CHILD = r'''
import json, sys, re
sys.path.insert(0, HERE)
from harness_fixtures import make_session, run_snippet
from inference.agent import tool_agent
from inference.agent.runtime_state import load_runtime_state
session, agent = make_session(tmpdir=TMP)
outs = []
for code in SNIPPETS:
    out, d = run_snippet(agent, session, code)
    outs.append(re.sub(r'"(run_elapsed_seconds|time_remaining_seconds)": [0-9.eE+-]+|(run_elapsed_seconds|time_remaining_seconds).: [0-9.eE+-]+', "T", out))
frame, history = load_runtime_state(session.state_path)
opener = agent._build_user_prompt(session.action_count, valid_actions=["UP","DOWN","LEFT","RIGHT"],
    current_frame=frame, history_entries=history, previous_step_summary=agent._last_step_summary)
print(json.dumps({"outs": outs, "opener": opener, "actions": session.action_count,
                  "system": tool_agent._build_system_prompt(tool_output_tokens=3072)}))
'''


def run(tree, tmp):
    env = {**os.environ, **FRANZEN_ENV, "ARC3_TREE": str(tree)}
    env.pop("ARC3_SURPRISE_GATE", None)
    env.pop("SURPRISE_GATE", None)
    code = (CHILD.replace("HERE", repr(str(HERE))).replace("TMP", repr(str(tmp)))
            .replace("SNIPPETS", repr(SNIPPETS)))
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=110)
    if proc.returncode:
        sys.exit(proc.stderr[-3000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    base, patched = Path(sys.argv[1]), Path(sys.argv[2])
    a = run(base, "/tmp/sg_eq_base")
    b = run(patched, "/tmp/sg_eq_patched")
    ok = True
    for key in ("actions", "system", "opener"):
        same = a[key] == b[key]
        ok &= same
        print(f"{key:8s} identical: {same}")
    for i, (x, y) in enumerate(zip(a["outs"], b["outs"])):
        same = x == y
        ok &= same
        print(f"snippet {i} identical: {same}")
        if not same:
            print("  base:   ", x[:600]); print("  patched:", y[:600])
    print("BASELINE EQUIVALENT" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
