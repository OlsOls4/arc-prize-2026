"""The full ToolAgent analyzer loop with a scripted fake LLM (no network).

SPDX-License-Identifier: MIT-0

_HarnessGameSession.play() drives ToolAgent.analyze(); only the HTTP call
(_chat_completion) is replaced, by a function that returns canned python tool
calls. This exercises the system prompt, tool dispatch, the sandbox callback,
the solver gate and the tool results that feed the next request.
"""
import json

from harness_fixtures import make_session

P = '''
def P(f):
    for r, row in enumerate(f.ascii.split("\\n")):
        c = row.find("G")
        if c >= 0:
            return (r, c)
step = lambda dr, dc: (lambda p, n: P(n) == (P(p)[0] + dr, P(p)[1] + dc))
'''

SCRIPT = [
    P + 'action(["DOWN", "RIGHT", "RIGHT", "RIGHT", "RIGHT"], expect=[step(1, 0)] + [step(0, 1)] * 4)',
    'action(["RIGHT", "RIGHT", "RIGHT"])',
    'action(["RIGHT"])',
]


def test_full_loop_with_fake_llm(monkeypatch, tmp_path):
    import inference.framework.solver as solver
    from inference.agent.tool_agent import _ChatCompletionResult

    monkeypatch.setenv("ARC3_SURPRISE_GATE", "1")
    monkeypatch.setattr(solver, "BATCH_NOOP_BLOCK", False)
    session, agent = make_session(tmpdir=tmp_path)
    session.solver.max_actions_per_game = 8
    requests = []

    def fake_chat(messages, *, tools, request_timeout_seconds=None, max_tokens=None,
                  enable_thinking=None):
        requests.append(json.loads(json.dumps(messages, default=str)))
        code = SCRIPT[min(len(requests) - 1, len(SCRIPT) - 1)]
        message = {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call{len(requests)}", "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})},
        }]}
        return _ChatCompletionResult(message=message, finish_reason="tool_calls",
                                     usage={"prompt_tokens": 100, "completion_tokens": 10,
                                            "total_tokens": 110})

    monkeypatch.setattr(agent, "_chat_completion", fake_chat)
    session.play()

    assert session.action_count == 8
    assert "Prediction-gated actions" in requests[0][0]["content"]
    # the first tool result (seen by the second request) carries the surprise
    second = json.dumps(requests[1])
    assert "SURPRISE at action 4 of 5" in second
    t = agent._surprise_stats.totals()
    assert t["aborts"] == 1 and t["actions_saved"] == 1
    assert t["model"] == {"pass": 3, "fail": 1, "unknown": 0}
    assert t["actions"] == 8
    stats = json.loads((session.state_path.parent / "surprise_gate_stats.json").read_text())
    assert stats["totals"]["actions"] == 8
