# DESIGN: where the surprise gate hooks in

The patch applies on top of Franzen's M2 harness. Its base is Duck commit `7652836` plus his `harness-changes.patch`. It is checked to be identical to https://github.com/da-fr/arc-agi-3-solution at HEAD on 15 Oct 2026: the notebook patch reverse-applies cleanly to that repo.

## 1. Data flow of one `action(...)` call (gate on)

```
model snippet (sandbox subprocess)          host process
-----------------------------------         ------------------------------------------
action(acts, expect=[f1, f2, "..."])
  normalize_expectations, describe_predicate
  send {"type":"action", actions, stale_after,
        "expect": [desc1, desc2, ...]}  ───────▶ run_sandboxed_python (python_tool_sandbox.py)
                                                  passes expect + expect_eval callback to
                                                ToolAgent._handle_action (tool_agent.py)
                                                  builds surprise_gate.BatchGate, puts it in
                                                  step_arguments["surprise_gate"]
                                                _HarnessGameSession.step_env (solver.py)
                                                  for each action: execute ──▶ game
                                                    BatchGate.check(i, before, after, payload)
  ◀── {"type":"expect_eval", id, index,              model desc? ─▶ expect_eval(...) over the pipe
        before, after, result}                       no desc?    ─▶ default: gameplay_changed
  eval_predicate(f_i, p, n, r)  (SIGALRM 0.5 s)
  send {"type":"expect_result", id, verdict} ──▶     verdict False → abort (stop_reason="surprise")
                                                   ... Franzen's checks continue unchanged:
                                                   terminal stops, then the batch no-op stop
                                                  final payload += "surprise" report and
                                                  "| prediction held/FAILED" on action_trace lines
  ◀── {"type":"action_result", ...}             ToolAgent: counters → surprise_gate_stats.json
  print "[surprise] ..." ; arm latch            _summarize_step_sequence → "surprise_reports"
  next action() → SurpriseStopError             _build_user_prompt → "Note: SURPRISE at ..." in
                                                the next turn's opener
```

## 2. The hook points, file by file

| File | Change | Lines |
|---|---|---|
| `inference/framework/solver.py`, `_HarnessGameSession.step_env` | Directly beside Franzen's guards. `step_env` already runs a batch one action at a time and applies, in order: the per-position `action_guard_hook` (known no-op / known death), the stale-state block, execution, terminal stops, then the **batch no-op stop** (`BATCH_NOOP_BLOCK`, `stop_reason="no_op_action"`). The gate captures the board before each action and calls `BatchGate.check` right after execution. It aborts (`stop_reason="surprise"`) after the terminal stops and **before** the no-op stop. At the end it attaches the report and adds trace suffixes. Nothing runs when `arguments["surprise_gate"]` is absent. | +43 / −0 |
| `inference/agent/tool_agent.py` | `_handle_action` accepts `expect` and `expect_eval`, builds the `BatchGate`, and records counters. `run_sandboxed_python` receives the gate config. `"surprise"` passes through `_compact_action_result`. `_summarize_step_sequence` collects `surprise_reports`. These are rendered in the opener (`_build_user_prompt`, next to the no-op and guard notes) and in the summary-turn context. `_build_system_prompt` appends the addendum when the gate is on. | +91 / −2 |
| `inference/agent/python_tool_sandbox.py` | The bootstrap gets `surprise_predicates.py` spliced in (the same mechanism as `frame_diff.py`). `action(actions, expect=None)` sends predicate descriptions and answers `expect_eval` requests while blocked. The latch and `SurpriseStopError` are added. On the host, `run_sandboxed_python` gets an `_expect_eval` callback over the existing pipe, a pending-line buffer, and ignores late `expect_result` lines. | +150 / −20 |
| `inference/agent/surprise_gate.py` (new) | Flags, prompt addendum, `BatchGate`, diff summary, `SurpriseGateStats`, `summarize_run_dir`. Fails open: an internal error yields "no abort" and is logged. | new |
| `inference/utils/surprise_predicates.py` (new) | Normalise, describe and evaluate predicates. Stdlib only, no `__future__`, so it can be spliced into the sandbox. | new |

### Interaction with Franzen's guards

- **Batch no-op stop** (`ARC3_BATCH_NOOP_BLOCK=1` in the M2 notebook):
  - The default predicate ("interior changes") tests the same thing. When the no-op stop is on, a default failure is deferred to it, so the model gets Franzen's tuned wording once (`noop_deferred`).
  - An explicit model prediction that fails on a no-op action takes precedence as a surprise. The model said what it expected, so that report is the sharper one.
- **Stale-state latch** (`ARC3_STALE_STATE_BLOCK=1`) and **terminal latch** are untouched. The surprise latch sits beside them in `action()`. Order: predicate-context check, gate-off `expect` rejection, terminal latch, surprise latch, then send.
- **Terminal actions** (level completed, game over, win): their predictions are scored for the counters, e.g. `n.level == p.level + 1`. They are never reported or latched, because the batch stops there anyway.
- **`ARC3_GUARDS_FROM_LEVEL=2`**: the gate follows the batch no-op stop's policy and applies from level 1 (`ARC3_SURPRISE_GATE_FROM_LEVEL` to change that).

### Why the sandbox callback (and not host-side eval)

- Model predicates run in the same isolated `python -I -S` subprocess as the model's code. They get the same rlimits, the same restricted builtins and imports, and the same `FrameView`/`segmentation`/`frame_diff` API.
- Python callables never cross the pipe; only short descriptions do.
- Per-predicate time is limited by SIGALRM inside the sandbox. The host waits at most timeout + 2 s (bounded by the tool deadline). On a missed reply it treats the verdict as unknown and drops the late reply.

### Gate off = baseline

With the flag absent, none of these happen:

- no `surprise_gate` key is sent to the sandbox, so it never sends `expect`;
- `expect=` raises the same `TypeError` text Python produced before;
- no step argument is added, no prompt text, and no stats file is written.

`tests/baseline_equivalence.py` runs the same scripted session under Franzen's competition flags on both trees. It compares every tool output, the opener and the system prompt, and finds them identical. The bootstrap source string itself does differ (an extra spliced module).

## 3. Applying it in the Kaggle notebook

Franzen's notebook works like this:

- **Cell 2** is `%%writefile /kaggle/harness-changes.patch`.
- **Cell 4** copies the original Tufa bundle to `/kaggle/taaf-kaggle-source-share`, then runs `git apply --include=ARC3-Inference/* -v /kaggle/harness-changes.patch` with `cwd=BUNDLE_DIR/src`.

To add the gate:

1. **New cell after cell 2.** Paste `kaggle/cell_0b_surprise_gate_patch.txt`. It is `%%writefile /kaggle/surprise-gate.patch` plus the patch. Alternatively, publish `surprise-gate.patch` as a small MIT-0 Kaggle Dataset and reference it from `/kaggle/input/...`.
2. **Cell 4**, immediately after Franzen's `git apply`:
   ```python
   subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/surprise-gate.patch"],
                  cwd=f"{BUNDLE_DIR}/src", check=True)
   ```
3. **Cell 4 `setup_env`**: add `'ARC3_SURPRISE_GATE': '1'` and, optionally, the other knobs (see `kaggle/cell_4_additions.py`). The harness runs in the notebook process and reads the flags at call time. No pickled object gains a field (`HarnessSolver` is unchanged; `ToolAgent` uses `getattr` for its new attribute), so `deploy_target.pkl` and `benchmark_initial.pkl` unpickle as before.
4. **Optional cell after the run:** call `summarize_run_dir(WORKING_DIR)` and write `surprise_gate_summary.json`.

For a clean ablation, use the same notebook with `ARC3_SURPRISE_GATE` set to `0` and to `1`. Shadow mode (`ARC3_SURPRISE_GATE_MAX_ABORTS=0`) separates the cost of writing predicates from the effect of enforcing them.

## 4. Known limitations and risks

- **No LLM was run.** Whether Qwen3.8-Flash-Next writes useful predicates, and whether the gate improves RHAE, is untested. The prompt addendum is about 1,600 characters (≈400 tokens), added to the cached system-prompt prefix.
- **"actions_saved" is an upper bound.** It counts skipped submitted actions. The model may re-issue some of them next turn.
- **A turn per abort.** Each abort turns remaining actions into a new model turn (tokens and wall-clock). The per-level cap (default 6) bounds thrashing, but the right value needs a real run.
- **Trivial or over-strict predicates.** `lambda p, n: True` disables the gate for that action. Over-strict ones burn the cap. Shadow-mode rates (`model_rates`) will show which case applies.
- **SIGALRM** only works on the sandbox's main thread, which is where snippets run. A predicate that spawns no threads cannot escape it. CPU rlimits for the whole snippet still apply.
- **Animations:** predicates see the final frame of each action only, not intermediate animation frames.
- **Upstream drift:** the patch has context lines from Franzen's tree. If his notebook patch changes, regenerate with `run_tests.sh` and resolve conflicts.
