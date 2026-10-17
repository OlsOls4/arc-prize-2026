"""Unit tests for the two new modules (no harness, no game).

SPDX-License-Identifier: MIT-0
"""
import threading
import time

import pytest

import harness_fixtures  # noqa: F401  (puts the patched tree on sys.path)
from inference.agent import surprise_gate as sg
from inference.utils import surprise_predicates as sp


# -- surprise_predicates ----------------------------------------------------------
def test_normalize_expectations():
    f = lambda p, n: True  # noqa: E731
    assert sp.normalize_expectations(None, 3) == ([None] * 3, "")
    assert sp.normalize_expectations(f, 2) == ([f, f], "")
    assert sp.normalize_expectations("n.level > 0", 2)[0] == ["n.level > 0"] * 2
    items, note = sp.normalize_expectations([f], 3)
    assert items == [f, None, None] and "missing" in note
    items, note = sp.normalize_expectations([f, f, f], 2)
    assert items == [f, f] and "extra" in note
    items, note = sp.normalize_expectations([f, 7], 2)
    assert items == [f, None] and "ignored" in note
    assert sp.normalize_expectations(42, 1)[0] == [None]


def test_describe_predicate_recovers_source():
    code = (
        "a = lambda p, n: n.level == p.level\n"
        "def moved(p, n):\n    return p.ascii != n.ascii\n"
        "mk = lambda d: (lambda p, n: d > 0)\n"
    )
    scope = {}
    exec(compile(code, "<python_tool>", "exec"), scope)
    assert sp.describe_predicate(scope["a"], code) == "lambda p, n: n.level == p.level"
    assert sp.describe_predicate(scope["moved"], code) == "moved: p.ascii != n.ascii"
    assert sp.describe_predicate(scope["mk"](3), code) == "lambda p, n: d > 0 [d=3]"
    assert sp.describe_predicate("x" * 500).endswith("...")
    assert sp.describe_predicate(None) is None


class _F:
    def __init__(self, level):
        self.level = level


def test_eval_predicate_tristate():
    p, n = _F(1), _F(2)
    assert sp.eval_predicate(lambda p, n: n.level == 2, p, n, {}) == (True, "")
    assert sp.eval_predicate(lambda p, n: n.level == 1, p, n, {}) == (False, "")
    assert sp.eval_predicate(lambda p, n, r: r["x"], p, n, {"x": 1}) == (True, "")
    assert sp.eval_predicate(lambda n: n.level, p, n, {}) == (True, "")
    assert sp.eval_predicate("n.level - p.level == 1", p, n, {}, {"__builtins__": {}}) == (True, "")
    v, err = sp.eval_predicate(lambda p, n: 1 / 0, p, n, {})
    assert v is None and "ZeroDivisionError" in err
    v, err = sp.eval_predicate(lambda p, n: None, p, n, {})
    assert v is None
    v, err = sp.eval_predicate("n.level ==", p, n, {})
    assert v is None and "SyntaxError" in err


def test_eval_predicate_timeout_main_thread():
    def spin(p, n):
        while True:
            pass
    t0 = time.monotonic()
    v, err = sp.eval_predicate(spin, None, None, {}, timeout_seconds=0.1)
    assert v is None and "timed out" in err and time.monotonic() - t0 < 2


def test_eval_predicate_off_main_thread_does_not_crash():
    out = {}
    t = threading.Thread(target=lambda: out.update(r=sp.eval_predicate(lambda p, n: True, 1, 2, {})))
    t.start(); t.join()
    assert out["r"] == (True, "")


# -- surprise_gate ------------------------------------------------------------------
def _grid(player, size=12):
    g = [[0] * size for _ in range(size)]
    g[player[0]][player[1]] = 3
    return g


def test_summarize_diff():
    assert sg.summarize_diff(_grid((5, 5)), _grid((5, 5))) == "nothing changed anywhere on the board"
    text = sg.summarize_diff(_grid((5, 5)), _grid((5, 6)))
    assert "2 interior cell(s) changed in rows 5-5, cols 5-6" in text
    a, b = _grid((5, 5)), _grid((5, 5))
    b[0][0] = 7
    assert sg.summarize_diff(a, b) == "no change inside the board area; 1 edge/HUD cell(s) changed"
    assert "shape changed" in sg.summarize_diff(_grid((5, 5)), _grid((5, 5), size=10))


@pytest.fixture
def env(monkeypatch):
    for k in ("ARC3_SURPRISE_GATE_MAX_ABORTS", "ARC3_SURPRISE_GATE_DEFAULT",
              "ARC3_SURPRISE_GATE_FROM_LEVEL"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _payload(level=1, changed=True, name="RIGHT", **kw):
    return {"level": level, "gameplay_changed": changed, "action_display": name,
            "action_num": 1, **kw}


def _gate(descs, verdicts, stats=None, **kw):
    calls = []

    def evaluator(index, before, after, result):
        calls.append(index)
        return {"verdict": verdicts[index], "error": ""}
    gate = sg.BatchGate(descriptions=descs, stats=stats or sg.SurpriseGateStats("t"),
                        evaluator=evaluator, **kw)
    return gate, calls


def test_gate_aborts_and_counts(env):
    gate, calls = _gate(["a", "b", "c"], [True, False, True])
    g0, g1 = _grid((5, 5)), _grid((5, 6))
    assert gate.check(1, 3, g0, g1, _payload()) == {"abort": False, "defer_noop": False}
    assert gate.check(2, 3, g0, g1, _payload()) == {"abort": True, "defer_noop": False}
    assert calls == [0, 1] and gate.latch
    t = gate.stats.totals()
    assert t["aborts"] == 1 and t["actions_saved"] == 1 and t["model"]["fail"] == 1
    rep = gate.report(["RIGHT"])
    assert "SURPRISE at action 2 of 3" in rep["text"] and "SurpriseStopError" in rep["text"]


def test_gate_default_predicate_and_defer(env):
    gate, _ = _gate([None, None], [None, None])
    g = _grid((5, 5))
    assert gate.check(1, 2, g, g, _payload(changed=False))["abort"] is True
    assert gate.latch is False  # default failures never latch a snippet
    gate2, _ = _gate([None, None], [None, None], noop_block_active=True)
    assert gate2.check(1, 2, g, g, _payload(changed=False)) == {"abort": False, "defer_noop": True}
    env.setenv("ARC3_SURPRISE_GATE_DEFAULT", "none")
    gate3, _ = _gate([None, None], [None, None])
    assert gate3.check(1, 2, g, g, _payload(changed=False))["abort"] is False


def test_gate_cap_shadow_terminal_from_level(env):
    stats = sg.SurpriseGateStats("t")
    env.setenv("ARC3_SURPRISE_GATE_MAX_ABORTS", "1")
    g = _grid((5, 5))
    first, _ = _gate(["x", "x"], [False, False], stats=stats)
    assert first.check(1, 2, g, g, _payload())["abort"] is True
    second, _ = _gate(["x", "x"], [False, False], stats=stats)
    assert second.check(1, 2, g, g, _payload())["abort"] is False
    assert second.surprises[0]["capped"] and stats.totals()["capped"] == 1
    third, _ = _gate(["x", "x"], [False, False], stats=stats)
    assert third.check(1, 2, g, g, _payload(level=2))["abort"] is True  # new level

    env.setenv("ARC3_SURPRISE_GATE_MAX_ABORTS", "0")
    shadow, _ = _gate(["x", "x"], [False, False])
    assert shadow.check(1, 2, g, g, _payload())["abort"] is False
    assert shadow.report() is None and shadow.stats.totals()["shadow_fail"] == 1

    env.setenv("ARC3_SURPRISE_GATE_MAX_ABORTS", "-1")
    term, _ = _gate(["x", "x"], [False, False])
    assert term.check(1, 2, g, g, _payload(), terminal=True)["abort"] is False
    assert term.report() is None and term.stats.totals()["model"]["fail"] == 1

    env.setenv("ARC3_SURPRISE_GATE_FROM_LEVEL", "2")
    early, calls = _gate(["x", "x"], [False, False])
    assert early.check(1, 2, g, g, _payload(level=1))["abort"] is False and calls == []


def test_evaluator_exception_is_unknown(env):
    def boom(*a):
        raise RuntimeError("pipe closed")
    gate = sg.BatchGate(descriptions=["x", "x"], stats=sg.SurpriseGateStats("t"), evaluator=boom)
    g = _grid((5, 5))
    assert gate.check(1, 2, g, g, _payload())["abort"] is False
    assert gate.stats.totals()["model"]["unknown"] == 1


def test_stats_rates_and_aggregate(env, tmp_path):
    stats = sg.SurpriseGateStats("agg-test")
    stats.record_batch(1, 4, 4, 4)
    stats.record_verdict("model", True)
    stats.record_verdict("model", False)
    d = stats.to_dict()["totals"]
    assert d["model_rates"]["pass"] == 0.5 and d["requested"] == 4
    stats.write(tmp_path / "s.json")
    assert (tmp_path / "s.json").read_text().startswith("{")
    agg = sg.SurpriseGateStats.aggregate()
    assert agg["games"] >= 1 and agg["requested"] >= 4


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv("ARC3_SURPRISE_GATE", raising=False)
    monkeypatch.delenv("SURPRISE_GATE", raising=False)
    assert sg.gate_enabled() is False
    monkeypatch.setenv("SURPRISE_GATE", "1")
    assert sg.gate_enabled() is True
    monkeypatch.setenv("ARC3_SURPRISE_GATE", "0")  # the namespaced name wins
    assert sg.gate_enabled() is False


def test_sandbox_bootstrap_contains_injected_module():
    from inference.agent import python_tool_sandbox as pts
    assert "def eval_predicate(" in pts._SANDBOX_BOOTSTRAP
    assert "__SURPRISE_PREDICATES_SOURCE__" not in pts._SANDBOX_BOOTSTRAP
    compile(pts._SANDBOX_BOOTSTRAP, "<bootstrap>", "exec")


def test_summarize_run_dir(tmp_path):
    for i in range(2):
        s = sg.SurpriseGateStats(f"g{i}")
        s.record_batch(1, 5, 3, 5)
        s.bump("aborts"); s.bump("actions_saved", 2)
        (tmp_path / f"g{i}").mkdir()
        s.write(tmp_path / f"g{i}" / "surprise_gate_stats.json")
    out = sg.summarize_run_dir(tmp_path)
    assert out["games"] == 2 and out["aborts"] == 2 and out["actions_saved"] == 4
    assert out["actions_saved_fraction"] == 0.4


def test_check_fails_open(env, monkeypatch):
    gate, _ = _gate(["x", "x"], [False, False])
    monkeypatch.setattr(sg, "summarize_diff", lambda *a, **k: 1 / 0)
    g = _grid((5, 5))
    assert gate.check(1, 2, g, g, _payload()) == {"abort": False, "defer_noop": False}
