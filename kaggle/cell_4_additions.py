# SPDX-License-Identifier: MIT-0
# --- Add to Franzen's cell 4 ("Environment and submission mode") -------------

# (a) right after his `subprocess.run(["git", "apply", ... "/kaggle/harness-changes.patch"], ...)`:
subprocess.run(
    ["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/surprise-gate.patch"],
    cwd=f"{BUNDLE_DIR}/src", check=True,
)
print('surprise-gate patch applied successfully')
# If the patch ships as a Kaggle Dataset instead of a %%writefile cell, point at
# e.g. /kaggle/input/datasets/olsols4/arc3-surprise-gate/surprise-gate.patch.

# (b) inside `setup_env = { ... }`, a new block (all values are strings or ints):
SURPRISE_GATE_ENV = {
    'ARC3_SURPRISE_GATE': '1',              # '0' (or absent) = exact M2 baseline
    'ARC3_SURPRISE_GATE_MAX_ABORTS': 6,     # per level; 0 = shadow (log only); -1 = no cap
    'ARC3_SURPRISE_GATE_DEFAULT': 'changed',# 'none' = only model predictions gate
    'ARC3_SURPRISE_GATE_TIMEOUT': '0.5',    # seconds per predicate, in the sandbox
    'ARC3_SURPRISE_GATE_FROM_LEVEL': 1,     # gate from this level on
    'ARC3_SURPRISE_GATE_PROMPT': '1',       # '0' = enforcement without the prompt addendum
}
# setup_env.update(SURPRISE_GATE_ENV)   # <- before the final os.environ.update(...)

# --- New cell after the benchmark run (cell 9) ---------------------------------
# from inference.agent.surprise_gate import summarize_run_dir
# summary = summarize_run_dir(WORKING_DIR)
# (WORKING_DIR / "surprise_gate_summary.json").write_text(json.dumps(summary, indent=1))
# print({k: summary[k] for k in ("games", "actions", "requested", "aborts", "halts",
#        "actions_saved", "actions_saved_fraction", "capped", "noop_deferred",
#        "model_rates", "default_rates")})
