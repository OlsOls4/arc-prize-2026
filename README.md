# Surprise-gated action batches (ARC-AGI-3, Duck harness)

An opt-in patch for Daniel Franzen's ARC-AGI-3 Milestone 2 solution, which is built on Tufa Labs' Duck harness.

With the patch on, the agent can attach a **prediction** to every action it submits. The harness runs the batch one action at a time and checks each prediction against the frame that actually comes back. At the **first falsified prediction** it skips the rest of the batch and hands the model a compact **surprise report**: the action, what was expected, and what changed.

```python
# `step(dr, dc)` is the model's own helper: "the player moved by (dr, dc)"
action(["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
       expect=[step(1, 0)] + [step(0, 1)] * 4)
```
Real output from the synthetic test game, where the 4th move hits a hidden teleport trap:
```
[surprise] SURPRISE at action 4 of 5 ('RIGHT'): expected `lambda p, n: P(n) == (P(p)[0] + dr, P(p)[1] + dc) [dc=1, dr=0]`,
observed: 2 interior cell(s) changed in rows 6-7, cols 6-8 (colour 0>3 x1, 3>0 x1); 1 edge/HUD cell(s) changed;
colour G object moved [7, 8]->[6, 6]. The remaining 1 action(s) ['RIGHT'] were NOT executed. Your model of the game
was wrong about this action: re-locate objects from current_frame and re-plan. Further action(...) calls in this
snippet raise SurpriseStopError.
```
The per-action trace in the result reads `4: RIGHT -> effect | prediction FAILED`, `5: RIGHT -> SKIPPED (never executed)`.

**Why:** RHAE scores each level by (human_actions / ai_actions)². Every action spent after the model's world model has been refuted costs score quadratically and buys almost nothing. Franzen's no-op stop only catches actions that change *nothing*. This patch also catches actions that change the board **in the wrong way**.

## Contents

| Path | What |
|---|---|
| `surprise-gate.patch` | Unified git diff, same format as Franzen's `harness-changes.patch` (paths `a/ARC3-Inference/...`). Apply it **after** his patch. |
| `module/surprise_gate.py` | Host side: config flags, `BatchGate` (verdict per executed action, abort cap, default predicate), diff summary, counters (`SurpriseGateStats`, `summarize_run_dir`). Also included in the patch as `inference/agent/surprise_gate.py`. |
| `module/surprise_predicates.py` | Sandbox side: normalising, describing and evaluating predicates (tri-state, time-limited). Spliced into the sandbox bootstrap the same way `frame_diff.py` is. Included in the patch as `inference/utils/surprise_predicates.py`. |
| `kaggle/cell_0b_surprise_gate_patch.txt` | Ready-to-paste `%%writefile /kaggle/surprise-gate.patch` notebook cell. |
| `kaggle/cell_4_additions.py` | The `git apply` line, env flags and an end-of-run summary cell for the Kaggle notebook. |
| `tests/` | 33 pytest tests plus a gate-off equivalence check. CPU only, no LLM, no paid API. |
| `run_tests.sh` | Rebuilds Tufa original → Franzen patch → this patch from scratch, then runs everything. |
| `DESIGN.md` | Where it hooks in, and how the Kaggle cell applies it. |
| `LICENSE`, `NOTICE` | MIT-0 for our code; attribution and licences for Franzen and Tufa. |

## Flags

All flags are read at call time.

| Env var | Default | Meaning |
|---|---|---|
| `ARC3_SURPRISE_GATE` (alias `SURPRISE_GATE`) | off | Master switch. Off means the exact M2 baseline: verified identical tool outputs, opener and system prompt. |
| `ARC3_SURPRISE_GATE_MAX_ABORTS` | `6` | Aborts (and snippet halts) allowed per level before surprises stop stopping things. `0` = shadow mode: predictions are scored and logged, but nothing is stopped or reported. `-1` = no cap. |
| `ARC3_SURPRISE_GATE_DEFAULT` | `changed` | Prediction used when the model gives none: "the board interior changes". When `ARC3_BATCH_NOOP_BLOCK=1`, a failure of this default is handed to Franzen's no-op stop (counted as `noop_deferred`). `none` = only model predictions gate. |
| `ARC3_SURPRISE_GATE_TIMEOUT` | `0.5` | Seconds per predicate, enforced inside the sandbox (SIGALRM). |
| `ARC3_SURPRISE_GATE_FROM_LEVEL` | `1` | First level the gate applies to. |
| `ARC3_SURPRISE_GATE_PROMPT` | `1` | `0` keeps enforcement but drops the system-prompt addendum (for ablations). |

## Predicate API (what the model sees)

- `expect=` takes one entry per action, or a single entry that applies to all of them. `None` means no prediction. A list of the wrong length is padded or truncated, with a note printed. It never raises.
- **Callable** `f(p, n)` or `f(p, n, r)`, where:
  - `p` and `n` are the frames before and after that action, with the same API as `current_frame` (`.ascii`, `.level`, `.shape`, `.segmentation`);
  - `r` is that action's result dict.
- **String:** a single expression over `p`, `n`, `r`, plus everything the snippet can see (e.g. `frame_diff`).
- Result: truthy = held, falsy = **falsified**, `None` = unknown.
  - Raising, timing out, or calling `action()` inside a predicate also count as **unknown**, which never stops a batch.
- After a falsified *model* prediction, any further `action(...)` in the same snippet raises `SurpriseStopError`. This is a BaseException, so `except Exception` cannot swallow it. The rule also covers loops that call `action([a])` one action at a time.

## Counters

Each game run directory gets a `surprise_gate_stats.json`, rewritten after every `action(...)` call. It is kept per level and holds:

- `actions`, `requested`, `batches`, `predicted`
- `model` and `default` verdict counts (pass, fail, unknown) and their rates
- `aborts`, `halts`, `actions_saved` (skipped submitted actions, an upper bound on waste avoided)
- `capped`, `noop_deferred`, `shadow_fail`

`summarize_run_dir(WORKING_DIR)` adds all of these up after the run. Each completed level also logs a one-line summary.

## Running the tests

```bash
pip install arc-agi pytest scipy imageio   # arcengine comes with arc-agi
python tests/fetch_public_games.py         # optional: ls20, ft09, vc33 via the free anonymous key
./run_tests.sh                             # clones Franzen's repo if ../franzen is absent
```

Latest run (15 Oct 2026): **33 passed**, plus **BASELINE EQUIVALENT** on the gate-off check. The run included the public games ls20, ft09 and vc33, played offline.

## Attribution

This builds on Daniel Franzen's Milestone 2 solution (https://github.com/da-fr/arc-agi-3-solution, Apache-2.0). That in turn builds on Jeroen Cottaar / Tufa Labs' Duck harness and TAAF (https://github.com/Tufalabs/duck-harness; code MIT, notebook Apache-2.0 per the authors). Their files keep their licences. See `NOTICE`.
