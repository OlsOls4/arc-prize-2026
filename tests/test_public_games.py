"""Patched harness against real public ARC-AGI-3 games, offline, no LLM.

SPDX-License-Identifier: MIT-0

Skipped unless the env files were fetched (python fetch_public_games.py).
"""
import json

import pytest

from harness_fixtures import make_session, public_game, run_snippet

GAMES = {
    "ls20": ['"LEFT"', '"LEFT"', '"UP"'],
    "ft09": ['{"action": "MOUSE", "row": 20, "col": 20}',
             '{"action": "MOUSE", "row": 40, "col": 40}',
             '{"action": "MOUSE", "row": 30, "col": 50}'],
    "vc33": ['{"action": "MOUSE", "row": 20, "col": 20}',
             '{"action": "MOUSE", "row": 40, "col": 40}',
             '{"action": "MOUSE", "row": 30, "col": 50}'],
}


def last(out):
    lines = json.loads(out).get("stdout", "").strip().splitlines()
    return lines[-1].strip() if lines else ""


@pytest.fixture
def gate_on(monkeypatch):
    import inference.framework.solver as solver
    monkeypatch.setenv("ARC3_SURPRISE_GATE", "1")
    monkeypatch.setattr(solver, "BATCH_NOOP_BLOCK", False)
    monkeypatch.setenv("ARC3_BATCH_NOOP_BLOCK", "0")
    return monkeypatch


def _session(name, tmp_path):
    game = public_game(name)
    if game is None:
        pytest.skip(f"{name} env files not fetched (run fetch_public_games.py)")
    return make_session(game, tmpdir=tmp_path)


@pytest.mark.parametrize("name", sorted(GAMES))
def test_false_prediction_stops_after_first_action(gate_on, tmp_path, name):
    session, agent = _session(name, tmp_path)
    batch = ", ".join(GAMES[name])
    out, _ = run_snippet(agent, session, f'''
r = action([{batch}], expect=lambda p, n: False)
print(r.get("stop_reason"), r.get("executed_count"))''')
    reason, count = last(out).split()
    assert count == "1"
    assert reason in {"surprise", "game_over", "level_completed"}
    assert session.action_count == 1
    if reason == "surprise":
        t = agent._surprise_stats.totals()
        assert t["aborts"] == 1 and t["actions_saved"] == 2


@pytest.mark.parametrize("name", sorted(GAMES))
def test_true_predictions_change_nothing(gate_on, monkeypatch, tmp_path, name):
    """Same batch with gate OFF and with gate ON + always-true predictions:
    identical action count and identical final board."""
    batch = ", ".join(GAMES[name])
    monkeypatch.delenv("ARC3_SURPRISE_GATE")
    s_off, a_off = _session(name, tmp_path / "off")
    run_snippet(a_off, s_off, f'action([{batch}])')
    monkeypatch.setenv("ARC3_SURPRISE_GATE", "1")
    s_on, a_on = _session(name, tmp_path / "on")
    out, _ = run_snippet(a_on, s_on, f'''
r = action([{batch}], expect=["n.shape == p.shape", lambda p, n: True, lambda p, n, r: r["level"] >= 1])
print(r.get("stop_reason"), r.get("executed_count"))''')
    assert s_on.action_count == s_off.action_count
    assert s_on.current_frame().grid == s_off.current_frame().grid
    t = a_on._surprise_stats.totals()
    assert t["model"]["fail"] == 0 and t["aborts"] == 0
    assert t["model"]["pass"] == s_on.action_count


def test_object_level_predicate_on_ls20(gate_on, tmp_path):
    """A frame_diff-based prediction evaluated in the sandbox on a real game."""
    session, agent = _session("ls20", tmp_path)
    out, _ = run_snippet(agent, session, '''
r = action(["LEFT", "LEFT"], expect="frame_diff(p, n)['changed_cell_count'] >= 0")
print(r.get("stop_reason"), r.get("executed_count"))''')
    assert last(out).endswith("2")
    assert agent._surprise_stats.totals()["model"] == {"pass": 2, "fail": 0, "unknown": 0}
