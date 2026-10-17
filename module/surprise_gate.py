"""Surprise-gated action batches (host side).

SPDX-License-Identifier: MIT-0  (our addition on top of the Apache-2.0 harness)

Enabled with ``ARC3_SURPRISE_GATE=1`` (alias ``SURPRISE_GATE=1``). Off - the
default - leaves the harness byte-for-byte on its baseline code paths: no
prompt text, no extra sandbox messages, no extra step arguments.

When on, the model may attach a prediction to each action it submits::

    action(['UP', 'UP', 'LEFT'], expect=[
        lambda p, n: player(n)[0] == player(p)[0] - 1,
        lambda p, n: player(n)[0] == player(p)[0] - 1,
        "frame_diff(p, n)['moved']",
    ])

The solver already executes a batch one action at a time (that is where
Franzen's batch no-op stop lives). After each executed action it now asks a
``BatchGate`` for a verdict. Model predictions are evaluated *inside the
existing Python sandbox subprocess* (rlimits, restricted builtins, per-predicate
SIGALRM time slice), via a callback over the sandbox's own pipe. A prediction
that raises or times out is UNKNOWN, never a failure. At the first FALSE the
rest of the batch is skipped (``stop_reason='surprise'``) and a compact
surprise report - action, expected predicate, observed diff - goes back to the
snippet and into the next turn's opener.

Actions without a model prediction fall back to a default predicate
(``ARC3_SURPRISE_GATE_DEFAULT``): ``changed`` (default) = the board interior
changed, i.e. the same test Franzen's no-op stop uses. When
``ARC3_BATCH_NOOP_BLOCK`` is on, a default-predicate failure is deferred to that
guard so the model is not told the same thing twice.

Aborts are capped per level (``ARC3_SURPRISE_GATE_MAX_ABORTS``, default 6;
negative = unlimited; 0 = shadow mode: predictions are evaluated and logged
but nothing is stopped and nothing is reported to the model).
"""
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional


log = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


# -- configuration (read at call time, like the harness's other opt-ins) ------
def gate_enabled() -> bool:
    for name in ("ARC3_SURPRISE_GATE", "SURPRISE_GATE"):
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip().lower() in _TRUE
    return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def max_aborts_per_level() -> int:
    return _env_int("ARC3_SURPRISE_GATE_MAX_ABORTS", 6)


def default_mode() -> str:
    mode = os.environ.get("ARC3_SURPRISE_GATE_DEFAULT", "changed").strip().lower()
    return mode if mode in {"changed", "none"} else "changed"


def predicate_timeout_seconds() -> float:
    return max(0.05, min(5.0, _env_float("ARC3_SURPRISE_GATE_TIMEOUT", 0.5)))


def from_level() -> int:
    return max(1, _env_int("ARC3_SURPRISE_GATE_FROM_LEVEL", 1))


def prompt_enabled() -> bool:
    return os.environ.get("ARC3_SURPRISE_GATE_PROMPT", "1").strip().lower() in _TRUE


def shadow_mode() -> bool:
    return max_aborts_per_level() == 0


DEFAULT_PREDICATE_TEXT = "the board interior changes (default prediction)"


# -- prompt -------------------------------------------------------------------
def system_prompt_addendum() -> str:
    if not prompt_enabled():
        return ""
    default_line = (
        "Actions without a prediction are checked with a default one: the board "
        "interior must change.\n"
        if default_mode() == "changed" else ""
    )
    return (
        "- Prediction-gated actions: `action(actions, expect=...)` takes one prediction per "
        "action - a Python predicate on the frames before (`p`) and after (`n`) that action, "
        "e.g. `expect=[lambda p, n: n.level == p.level, None, \"frame_diff(p, n)['moved']\"]`. "
        "`p` and `n` have the same API as `current_frame`; a predicate with a third argument "
        "`r` also receives that action's result dict. A single predicate applies to every "
        "action; `None` means no prediction. Strings are one expression over `p`, `n`, `r`.\n"
        "- The harness runs the actions one at a time and checks each prediction. At the first "
        "prediction that returns False the remaining actions are NOT executed, the result has "
        "`stop_reason='surprise'` and a `surprise` report (what you expected, what changed), "
        "and any further `action(...)` call in the same snippet raises `SurpriseStopError`. "
        "A predicate that raises or runs too long counts as unknown and never stops the batch. "
        + default_line
        + "- State the effect your plan relies on - which object moves where, which cells change "
        "colour, that the level advances. Every action is an experiment: a batch continues only "
        "while the world agrees with you.\n"
    )


# -- diff summary -------------------------------------------------------------
def _grid_list(grid: Any) -> list:
    return [list(row) for row in (grid or [])]


def summarize_diff(before: Any, after: Any, *, border: int = 4, max_moves: int = 3) -> str:
    """Compact, model-facing account of what one action changed."""
    a, b = _grid_list(before), _grid_list(after)
    if not a or not b:
        return "no frame available"
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return f"frame shape changed {len(a)}x{len(a[0])} -> {len(b)}x{len(b[0])} (full redraw)"
    rows, cols = len(a), len(a[0])
    interior, edge = [], 0
    transitions: dict[tuple[int, int], int] = {}
    for r in range(rows):
        ra, rb = a[r], b[r]
        if ra == rb:
            continue
        for c in range(cols):
            if ra[c] != rb[c]:
                inside = border <= r < rows - border and border <= c < cols - border
                if inside or border <= 0:
                    interior.append((r, c))
                    key = (ra[c], rb[c])
                    transitions[key] = transitions.get(key, 0) + 1
                else:
                    edge += 1
    if not interior and not edge:
        return "nothing changed anywhere on the board"
    parts = []
    if interior:
        r0 = min(p[0] for p in interior); r1 = max(p[0] for p in interior)
        c0 = min(p[1] for p in interior); c1 = max(p[1] for p in interior)
        top = sorted(transitions.items(), key=lambda kv: -kv[1])[:4]
        parts.append(
            f"{len(interior)} interior cell(s) changed in rows {r0}-{r1}, cols {c0}-{c1}"
            + (" (colour " + ", ".join(f"{x}>{y} x{k}" for (x, y), k in top) + ")" if top else "")
        )
    else:
        parts.append("no change inside the board area")
    if edge:
        parts.append(f"{edge} edge/HUD cell(s) changed")
    moves = _object_moves(a, b, max_moves) if interior else ""
    if moves:
        parts.append(moves)
    return "; ".join(parts)


def _object_moves(a: list, b: list, max_moves: int) -> str:
    """Object-level moves via the harness's own segmentation + frame_diff.
    Best effort: any failure just omits the clause."""
    try:
        from inference.utils.frame_diff import compute_frame_diff
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        from inference.utils.segmentation import segment_layer

        nodes_a = segment_layer(a, ARC_COLOR_CHARS).get("nodes", [])
        nodes_b = segment_layer(b, ARC_COLOR_CHARS).get("nodes", [])
        diff = compute_frame_diff(a, b, nodes_a, nodes_b)
    except Exception:  # noqa: BLE001
        return ""
    bits = []
    for entry in (diff.get("moved") or [])[:max_moves]:
        bits.append(f"colour {entry.get('color')} object moved {entry.get('from')}->{entry.get('to')}")
    counts = [
        f"{len(diff.get(k) or [])} {k}"
        # "resized" left out: it is dominated by backgrounds and HUD bars
        for k in ("appeared", "disappeared", "changed_color")
        if diff.get(k)
    ]
    if counts:
        bits.append(", ".join(counts))
    return "; ".join(bits)


# -- statistics ----------------------------------------------------------------
def _new_level_counters() -> dict[str, Any]:
    return {
        "batches": 0,            # action(...) calls that executed >= 1 action
        "actions": 0,            # executed actions
        "requested": 0,          # submitted actions
        "predicted": 0,          # actions carrying a model prediction
        "model": {PASS: 0, FAIL: 0, UNKNOWN: 0},
        "default": {PASS: 0, FAIL: 0, UNKNOWN: 0},
        "aborts": 0,             # batches cut short by a surprise
        "halts": 0,              # snippets latched after a surprise on a last action
        "actions_saved": 0,      # submitted actions skipped by surprise aborts
        "capped": 0,             # surprises that did not stop (per-level cap reached)
        "noop_deferred": 0,      # default-predicate failures left to the no-op stop
        "shadow_fail": 0,        # failures seen in shadow mode (cap = 0)
    }


class SurpriseGateStats:
    """Per-game counters, kept per level. Thread-safe enough for one game."""

    _registry: "list[SurpriseGateStats]" = []
    _registry_lock = threading.Lock()

    def __init__(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.levels: dict[str, dict[str, Any]] = {}
        self.level: Any = None
        self.created = time.time()
        with SurpriseGateStats._registry_lock:
            SurpriseGateStats._registry.append(self)

    def note_level(self, level: Any) -> dict[str, Any]:
        key = str(level if level is not None else "?")
        self.level = level
        if key not in self.levels:
            self.levels[key] = _new_level_counters()
        return self.levels[key]

    def current(self) -> dict[str, Any]:
        return self.note_level(self.level)

    def aborts_left(self) -> Optional[int]:
        cap = max_aborts_per_level()
        if cap < 0:
            return None
        used = self.current()["aborts"] + self.current()["halts"]
        return max(0, cap - used)

    def record_verdict(self, source: Optional[str], verdict: Optional[bool]) -> None:
        if source not in ("model", "default"):
            return
        bucket = self.current()[source]
        bucket[PASS if verdict is True else FAIL if verdict is False else UNKNOWN] += 1

    def record_batch(self, level: Any, requested: int, executed: int, predicted: int) -> None:
        counters = self.note_level(level)
        counters["requested"] += max(0, int(requested))
        counters["actions"] += max(0, int(executed))
        counters["predicted"] += max(0, int(predicted))
        if executed:
            counters["batches"] += 1

    def bump(self, key: str, amount: int = 1) -> None:
        self.current()[key] += amount

    # -- reporting --------------------------------------------------------------
    def totals(self) -> dict[str, Any]:
        return _sum_levels(self.levels.values())

    def to_dict(self) -> dict[str, Any]:
        return {"game_id": self.game_id, "levels": self.levels, "totals": with_rates(self.totals())}

    def summary_line(self) -> str:
        t = with_rates(self.totals())
        return (
            f"surprise_gate[{self.game_id}] actions={t['actions']} aborts={t['aborts']} "
            f"halts={t['halts']} saved={t['actions_saved']} capped={t['capped']} "
            f"model p/f/u={t['model'][PASS]}/{t['model'][FAIL]}/{t['model'][UNKNOWN]} "
            f"default p/f/u={t['default'][PASS]}/{t['default'][FAIL]}/{t['default'][UNKNOWN]}"
        )

    def write(self, path: Any) -> None:
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=1)
        except OSError:
            log.debug("surprise gate stats not written to %s", path, exc_info=True)

    @classmethod
    def aggregate(cls) -> dict[str, Any]:
        """Totals over every game in this process (for a notebook cell)."""
        with cls._registry_lock:
            games = list(cls._registry)
        levels = [lv for g in games for lv in g.levels.values()]
        out = with_rates(_sum_levels(levels))
        out["games"] = len(games)
        return out


def summarize_run_dir(root: Any) -> dict[str, Any]:
    """Totals over every ``surprise_gate_stats.json`` under ``root``.

    Works after the fact and across processes, unlike ``aggregate()``. One
    file per game run directory; per-game totals are included for analysis.
    """
    from pathlib import Path

    levels, games = [], []
    for path in sorted(Path(root).rglob("surprise_gate_stats.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        games.append({"game_id": data.get("game_id"), "path": str(path),
                      "totals": data.get("totals")})
        levels.extend((data.get("levels") or {}).values())
    out = with_rates(_sum_levels(levels))
    out["games"] = len(games)
    out["per_game"] = games
    return out


def _sum_levels(levels: Any) -> dict[str, Any]:
    total = _new_level_counters()
    for counters in levels:
        for key, value in counters.items():
            if isinstance(value, dict):
                for sub, n in value.items():
                    total[key][sub] = total[key].get(sub, 0) + n
            else:
                total[key] += value
    return total


def with_rates(counters: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(counters))
    for source in ("model", "default"):
        bucket = out[source]
        n = sum(bucket.values())
        out[f"{source}_rates"] = {
            k: (round(v / n, 4) if n else None) for k, v in bucket.items()
        }
    out["actions_saved_fraction"] = (
        round(out["actions_saved"] / out["requested"], 4) if out["requested"] else None
    )
    return out


# -- per-call gate ----------------------------------------------------------------
Evaluator = Callable[[int, dict, dict, dict], dict]


class BatchGate:
    """Judges each executed action of ONE action(...) call.

    Created by the tool agent, handed to the solver through the step
    arguments, and consulted by ``_HarnessGameSession.step_env`` right after
    each executed action (alongside the batch no-op stop).
    """

    def __init__(
        self,
        *,
        descriptions: list,
        stats: SurpriseGateStats,
        evaluator: Optional[Evaluator] = None,
        frame_payload: Optional[Callable[[Any, Any], dict]] = None,
        border: int = 4,
        noop_block_active: bool = False,
    ) -> None:
        self.descriptions = list(descriptions)
        self.stats = stats
        self.evaluator = evaluator
        self.frame_payload = frame_payload or _plain_frame_payload
        self.border = border
        self.noop_block_active = noop_block_active
        self.verdicts: dict[int, tuple[Optional[str], Optional[bool]]] = {}
        self.surprises: list[dict[str, Any]] = []
        self.latch = False
        self.aborted = False
        self.start_level: Any = None

    def predicted_count(self, upto: Optional[int] = None) -> int:
        items = self.descriptions if upto is None else self.descriptions[:upto]
        return sum(1 for d in items if d is not None)

    def check(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fail-open wrapper: a bug in the gate must never cost the game an
        action or crash step_env. See ``_check`` for the contract."""
        try:
            return self._check(*args, **kwargs)
        except Exception:  # noqa: BLE001
            log.warning("surprise gate check failed; continuing ungated", exc_info=True)
            return {"abort": False, "defer_noop": False}

    def _check(
        self,
        index: int,
        batch_size: int,
        before_grid: Any,
        after_grid: Any,
        payload: dict,
        *,
        before_level: Any = None,
        terminal: bool = False,
    ) -> dict[str, Any]:
        """``index`` is 1-based. Returns ``{"abort": bool, "defer_noop": bool}``.

        ``terminal`` marks an action that ended the attempt, the level or the
        run: the solver stops the batch there anyway, so its prediction is only
        scored (for the counters), never reported or latched.
        """
        level = before_level if before_level is not None else payload.get("level")
        if self.start_level is None:
            self.start_level = level
        self.stats.note_level(level)
        try:
            if int(level) < from_level():
                return {"abort": False, "defer_noop": False}
        except (TypeError, ValueError):
            pass
        desc = self.descriptions[index - 1] if 0 < index <= len(self.descriptions) else None
        verdict: Optional[bool] = None
        source: Optional[str] = None
        error = ""
        if desc is not None and self.evaluator is not None:
            source = "model"
            try:
                reply = self.evaluator(
                    index - 1,
                    self.frame_payload(before_grid, payload, before=True, level=level),
                    self.frame_payload(after_grid, payload, before=False,
                                       level=payload.get("level")),
                    _per_action_result(payload),
                ) or {}
            except Exception as exc:  # noqa: BLE001 - a broken channel is an unknown
                reply = {"verdict": None, "error": f"{type(exc).__name__}: {exc}"}
            verdict = reply.get("verdict") if reply.get("verdict") in (True, False) else None
            error = str(reply.get("error") or "")
        elif desc is None and default_mode() == "changed":
            source = "default"
            desc = DEFAULT_PREDICATE_TEXT
            changed = payload.get("gameplay_changed")
            verdict = changed if changed in (True, False) else None
        self.verdicts[index] = (source, verdict)
        self.stats.record_verdict(source, verdict)
        if verdict is not False or terminal:
            return {"abort": False, "defer_noop": False}

        remaining = max(0, batch_size - index)
        if source == "default":
            if remaining == 0:
                return {"abort": False, "defer_noop": False}
            if self.noop_block_active:
                self.stats.bump("noop_deferred")
                return {"abort": False, "defer_noop": True}
        if shadow_mode():
            self.stats.bump("shadow_fail")
            return {"abort": False, "defer_noop": False}

        surprise = {
            "index": index,
            "of": batch_size,
            "action": payload.get("action_display") or payload.get("action_name"),
            "expected": desc,
            "source": source,
            "observed": summarize_diff(before_grid, after_grid, border=self.border),
        }
        if error:
            surprise["error"] = error
        left = self.stats.aborts_left()
        if left is not None and left <= 0:
            surprise["capped"] = True
            self.stats.bump("capped")
            self.surprises.append(surprise)
            return {"abort": False, "defer_noop": False}
        if remaining > 0:
            self.stats.bump("aborts")
            self.stats.bump("actions_saved", remaining)
            self.aborted = True
            surprise["skipped"] = remaining
        else:
            self.stats.bump("halts")
        if source == "model":
            self.latch = True
        self.surprises.append(surprise)
        return {"abort": remaining > 0, "defer_noop": False}

    # -- rendering -----------------------------------------------------------------
    def trace_suffix(self, index: int) -> str:
        source, verdict = self.verdicts.get(index, (None, None))
        if source is None or (source == "default" and verdict is not False):
            return ""  # a held default prediction is the common case: say nothing
        word = {True: "held", False: "FAILED", None: "unknown"}[verdict]
        return f" | prediction {word}" + (" (default)" if source == "default" else "")

    def report(self, skipped_displays: Optional[list] = None) -> Optional[dict[str, Any]]:
        if not self.surprises:
            return None
        return {
            "surprises": self.surprises,
            "latch": self.latch,
            "text": render_report(self.surprises, skipped_displays or [], latch=self.latch),
        }


def render_report(surprises: list, skipped: list, *, latch: bool) -> str:
    lines = []
    for s in surprises[:3]:
        line = (
            f"SURPRISE at action {s.get('index')} of {s.get('of')} ({s.get('action')!r}): "
            f"expected `{s.get('expected')}`, observed: {s.get('observed')}."
        )
        if s.get("capped"):
            line += " (Surprise limit for this level reached, so the batch was NOT stopped.)"
        elif s.get("skipped"):
            line += f" The remaining {s['skipped']} action(s) {skipped!r} were NOT executed."
        lines.append(line)
    if latch:
        lines.append(
            "Your model of the game was wrong about this action: re-locate objects from "
            "current_frame and re-plan. Further action(...) calls in this snippet raise "
            "SurpriseStopError."
        )
    return " ".join(lines)


def _per_action_result(payload: dict) -> dict:
    keys = (
        "action_num", "level", "score", "state", "board_changed", "gameplay_changed",
        "no_op", "done", "level_completed", "game_over", "run_complete",
        "action_name", "action_data", "action_display", "batch_index", "batch_size",
    )
    return {k: payload[k] for k in keys if k in payload}


def _plain_frame_payload(grid: Any, payload: dict, *, before: bool, level: Any = None) -> dict:
    """Frame payload in the sandbox's FrameView wire format."""
    from inference.utils.grid_utils import format_grid_ascii

    rows = [list(row) for row in (grid or [])]
    if level is None:
        level = payload.get("level", 0)
    step = int(payload.get("action_num", 0) or 0) - (1 if before else 0)
    return {
        "ascii": format_grid_ascii(tuple(tuple(r) for r in rows)),
        "step": max(0, step),
        "level": int(level) if isinstance(level, int) else 0,
        "shape": [len(rows), len(rows[0]) if rows else 0],
        "grid": rows,
    }
