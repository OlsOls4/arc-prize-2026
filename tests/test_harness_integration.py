"""End-to-end tests through the PATCHED harness, with a scripted mock agent.

SPDX-License-Identifier: MIT-0

Each test feeds Python snippets (what the model's tool calls would contain)
into ToolAgent._run_python_tool. They run in the real sandbox subprocess,
call action(...) over the real pipe protocol, and are executed by the real
_HarnessGameSession.step_env against SlideGame (synthetic, deterministic).
"""
import json

import pytest

from harness_fixtures import make_session, run_snippet

# Player helper available to every snippet: (row, col) of the colour-3 cell.
P = '''
def P(f):
    for r, row in enumerate(f.ascii.split("\\n")):
        c = row.find("G")
        if c >= 0:
            return (r, c)
step = lambda dr, dc: (lambda p, n: P(n) == (P(p)[0] + dr, P(p)[1] + dc))
'''


@pytest.fixture
def gate_on(monkeypatch):
    import inference.framework.solver as solver
    monkeypatch.setenv("ARC3_SURPRISE_GATE", "1")
    monkeypatch.setenv("ARC3_BATCH_NOOP_BLOCK", "0")
    monkeypatch.setattr(solver, "BATCH_NOOP_BLOCK", False)
    monkeypatch.delenv("ARC3_SURPRISE_GATE_MAX_ABORTS", raising=False)
    monkeypatch.delenv("ARC3_SURPRISE_GATE_DEFAULT", raising=False)
    return monkeypatch


def last(out):
    """Last stdout line of a tool call (earlier lines are echoes/reports)."""
    lines = json.loads(out).get("stdout", "").strip().splitlines()
    return lines[-1].strip() if lines else ""


def _result(out):
    """The last action call result as the snippet printed it via json."""
    payload = json.loads(out)
    lines = [l for l in payload.get("stdout", "").splitlines() if l.startswith("RESULT ")]
    return json.loads(lines[-1][len("RESULT "):]) if lines else None, payload


def test_abort_at_first_false_prediction_and_report(gate_on, tmp_path):
    session, agent = make_session(tmpdir=tmp_path)
    # DOWN, then RIGHT x4: the 4th action steps onto a trap (7,9) that
    # teleports the player home. The board changes (no-op guard is blind),
    # but the prediction "col + 1" is falsified there.
    out, _ = run_snippet(agent, session, P + '''
r = action(["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
           expect=[step(1, 0), step(0, 1), step(0, 1), step(0, 1), step(0, 1)])
import json
print("RESULT " + json.dumps({k: r.get(k) for k in ("stop_reason", "executed_count", "skipped_actions", "surprise")}))
''')
    res, payload = _result(out)
    assert res["stop_reason"] == "surprise"
    assert res["executed_count"] == 4
    assert res["skipped_actions"] == ["RIGHT"]
    sur = res["surprise"]["surprises"][0]
    assert sur["index"] == 4 and sur["source"] == "model"
    assert "P(n) ==" in sur["expected"] and "dc=1, dr=0" in sur["expected"]
    assert "interior cell(s) changed" in sur["observed"]
    assert res["surprise"]["latch"] is True
    assert "[surprise] SURPRISE at action 4 of 5" in payload["stdout"]
    # carried into the next turn's opener
    summary = agent._last_step_summary
    assert summary["surprise_reports"] and "SURPRISE at action 4" in summary["surprise_reports"][0]
    from inference.agent.runtime_state import load_runtime_state
    frame, history = load_runtime_state(session.state_path)
    prompt = agent._build_user_prompt(
        session.action_count, valid_actions=["UP", "DOWN", "LEFT", "RIGHT"],
        current_frame=frame, history_entries=history, previous_step_summary=summary,
    )
    assert "Note: SURPRISE at action 4 of 5" in prompt
    # counters on disk
    stats = json.loads((session.state_path.parent / "surprise_gate_stats.json").read_text())
    t = stats["totals"]
    assert t["aborts"] == 1 and t["actions_saved"] == 1 and t["actions"] == 4
    assert t["model"] == {"pass": 3, "fail": 1, "unknown": 0}
    assert stats["levels"]["1"]["predicted"] == 4


def test_latch_stops_single_action_loops(gate_on, tmp_path):
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, P + '''
for a in ["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"]:
    try:
        action([a], expect=step(1, 0) if a == "DOWN" else step(0, 1))
    except Exception:
        pass  # a model's defensive except must not swallow the latch
''')
    payload = json.loads(out)
    assert "SurpriseStopError" in payload["error"]
    assert session.action_count == 4  # DOWN + 3 RIGHT (the trap), then halted
    stats = agent._surprise_stats.totals()
    assert stats["halts"] == 1 and stats["aborts"] == 0


def test_errors_and_timeouts_are_unknown_not_failures(gate_on, tmp_path):
    gate_on.setenv("ARC3_SURPRISE_GATE_TIMEOUT", "0.2")
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, '''
def slow(p, n):
    while True:
        pass
r = action(["RIGHT", "RIGHT", "RIGHT", "RIGHT"],
           expect=[lambda p, n: 1 / 0, slow, lambda p, n: action(["UP"]), "undefined_name > 3"])
print(r.get("stop_reason"), r.get("executed_count"))
''')
    payload = json.loads(out)
    assert payload["stdout"].strip().endswith("None 4")
    t = agent._surprise_stats.totals()
    assert t["model"] == {"pass": 0, "fail": 0, "unknown": 4}
    assert t["aborts"] == 0
    assert session.action_count == 4


def test_default_predicate_and_noop_guard_interplay(gate_on, tmp_path):
    import inference.framework.solver as solver
    # wall above the start: UP is a no-op. No model predictions given.
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, 'r = action(["UP", "RIGHT", "RIGHT"])\nprint(r.get("stop_reason"))')
    assert last(out) == "surprise"
    assert agent._surprise_stats.totals()["default"]["fail"] == 1

    # with Franzen's batch no-op stop on, the default failure defers to it
    gate_on.setenv("ARC3_BATCH_NOOP_BLOCK", "1")
    gate_on.setattr(solver, "BATCH_NOOP_BLOCK", True)
    session, agent = make_session(tmpdir=tmp_path / "b")
    out, _ = run_snippet(agent, session, 'r = action(["UP", "RIGHT", "RIGHT"])\nprint(r.get("stop_reason"))')
    assert last(out) == "no_op_action"
    t = agent._surprise_stats.totals()
    assert t["noop_deferred"] == 1 and t["aborts"] == 0

    # an explicit model prediction still wins over the no-op stop
    session, agent = make_session(tmpdir=tmp_path / "c")
    out, _ = run_snippet(agent, session, P + 'r = action(["UP", "RIGHT"], expect=step(-1, 0))\nprint(r.get("stop_reason"))')
    assert last(out) == "surprise"


def test_default_none_mode(gate_on, tmp_path):
    gate_on.setenv("ARC3_SURPRISE_GATE_DEFAULT", "none")
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, 'r = action(["UP", "RIGHT", "RIGHT"])\nprint(r.get("stop_reason"), r.get("executed_count"))')
    assert last(out) == "None 3"


def test_abort_cap_per_level_and_reset_on_new_level(gate_on, tmp_path):
    gate_on.setenv("ARC3_SURPRISE_GATE_MAX_ABORTS", "1")
    session, agent = make_session(tmpdir=tmp_path)
    never = 'r = action(["RIGHT", "RIGHT"], expect=lambda p, n: False)\nprint(r.get("stop_reason"), r.get("executed_count"))'
    out1, _ = run_snippet(agent, session, never)
    out2, _ = run_snippet(agent, session, never)
    assert last(out1) == "surprise 1"
    assert last(out2) == "None 2"  # capped: batch ran on
    assert "Surprise limit for this level reached" in json.loads(out2)["stdout"]
    t = agent._surprise_stats.totals()
    assert t["aborts"] == 1 and t["capped"] == 2  # both falsified actions of batch 2
    # finish level 1: player at (6,9) -> goal (6,11)
    out, _ = run_snippet(agent, session, P + '''
r = action(["RIGHT", "RIGHT"], expect=[step(0, 1), lambda p, n: n.level == p.level + 1])
print(r.get("stop_reason"), r.get("level_completed"))''')
    assert last(out) == "level_completed True"
    # the level-advance prediction was scored as a pass, not reported
    assert agent._surprise_stats.levels["1"]["model"]["pass"] == 2
    out3, _ = run_snippet(agent, session, never)
    assert last(out3) == "surprise 1"  # new level, new allowance
    assert agent._surprise_stats.levels["2"]["aborts"] == 1


def test_shadow_mode_logs_but_never_stops(gate_on, tmp_path):
    gate_on.setenv("ARC3_SURPRISE_GATE_MAX_ABORTS", "0")
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, 'r = action(["RIGHT", "RIGHT", "RIGHT"], expect=lambda p, n: False)\nprint(r.get("stop_reason"), r.get("executed_count"), "surprise" in r)')
    assert last(out) == "None 3 False"
    assert agent._surprise_stats.totals()["shadow_fail"] == 3


def test_string_predicates_and_frame_diff(gate_on, tmp_path):
    session, agent = make_session(tmpdir=tmp_path)
    out, _ = run_snippet(agent, session, '''
r = action(["RIGHT", "RIGHT"], expect="len(frame_diff(p, n)['moved']) >= 1 and n.level == p.level")
print(r.get("stop_reason"), r.get("executed_count"))''')
    assert last(out) == "None 2"
    assert agent._surprise_stats.totals()["model"]["pass"] == 2


def test_gate_off_is_baseline(monkeypatch, tmp_path):
    import inference.framework.solver as solver
    from inference.agent import tool_agent
    monkeypatch.delenv("ARC3_SURPRISE_GATE", raising=False)
    monkeypatch.delenv("SURPRISE_GATE", raising=False)
    monkeypatch.setattr(solver, "BATCH_NOOP_BLOCK", False)
    session, agent = make_session(tmpdir=tmp_path)
    seen = {}
    real = session.step_env

    def spy(arguments):
        seen.update(arguments)
        return real(arguments)

    agent._step_env_callback = spy
    out, _ = run_snippet(agent, session, 'r = action(["RIGHT", "RIGHT"])\nprint(sorted(r))')
    assert "surprise" not in json.loads(out)["stdout"]
    assert "surprise_gate" not in seen
    out, _ = run_snippet(agent, session, 'action(["RIGHT"], expect=lambda p, n: True)')
    assert "unexpected keyword argument 'expect'" in json.loads(out)["error"]
    assert agent._surprise_stats is None
    assert not (session.state_path.parent / "surprise_gate_stats.json").exists()
    prompt_off = tool_agent._build_system_prompt(tool_output_tokens=3072)
    monkeypatch.setenv("ARC3_SURPRISE_GATE", "1")
    prompt_on = tool_agent._build_system_prompt(tool_output_tokens=3072)
    assert "expect=" not in prompt_off and "Prediction-gated actions" in prompt_on
    assert prompt_on.startswith(prompt_off.split("\n\nTool session rules:")[0][:200])


def test_with_franzen_competition_flags(gate_on, tmp_path):
    """Gate on top of the M2 notebook's guard settings (no-op stop, stale
    latch, action echo, guards from level 2)."""
    import inference.framework.solver as solver
    for name, value in (("BATCH_NOOP_BLOCK", True), ("STALE_STATE_BLOCK", True),
                        ("ACTION_ECHO", True), ("NEW_NOOP_WORDING", True),
                        ("GUARDS_FROM_LEVEL", 2)):
        gate_on.setattr(solver, name, value)
    gate_on.setenv("ARC3_BATCH_NOOP_BLOCK", "1")
    session, agent = make_session(tmpdir=tmp_path)
    # no predictions + wall: Franzen's no-op stop owns it
    out, _ = run_snippet(agent, session, 'r = action(["UP", "RIGHT"])\nprint(r.get("stop_reason"))')
    assert last(out) == "no_op_action"
    # model prediction falsified by the trap: surprise stop, echo still printed
    out, _ = run_snippet(agent, session, P + '''
r = action(["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"], expect=[step(1, 0)] + [step(0, 1)] * 4)
print(r.get("stop_reason"), r.get("executed_count"))''')
    stdout = json.loads(out)["stdout"]
    assert last(out) == "surprise 4"
    assert "[action] batch of 5 actions, 4 executed" in stdout and "stop_reason: surprise" in stdout
    assert "prediction FAILED" in json.dumps(agent._last_action_call_result.get("action_trace"))
    t = agent._surprise_stats.totals()
    assert t["noop_deferred"] == 1 and t["aborts"] == 1
