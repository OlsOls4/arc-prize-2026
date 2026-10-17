"""Test fixtures: a tiny synthetic grid game and a real harness session.

SPDX-License-Identifier: MIT-0

No LLM is involved anywhere. The "mock agent" is a list of Python snippets fed
straight into ToolAgent._run_python_tool, i.e. exactly what the model's tool
calls would run, through the patched sandbox and the patched solver.
"""
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("ARC3_TREE", Path(__file__).resolve().parents[2] / "sg_tree"))
for entry in (ROOT / "ARC3-Inference", ROOT / "tufa-arc-agi-framework" / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

# a dummy endpoint: nothing here ever sends a request
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "mock")
os.environ.setdefault("LOCAL_ANALYZER_PROVIDER", "vllm")

import arcengine  # noqa: E402
import taaf.game  # noqa: E402

SIZE = 16
BORDER = 4  # harness default ARC3_NOOP_GUARD_BORDER


@dataclass
class SlideGame(taaf.game.Game):
    """A 16x16 board. A 1-cell player (colour 3) moves with UP/DOWN/LEFT/RIGHT
    (ACTION1-4) inside the interior; walls (colour 5) block it (no-op). A
    'trap' cell (colour 8) teleports the player back to its start: the board
    changes, but not the way a naive mover model predicts. Reaching the goal
    (colour 4) completes the level. Two levels."""

    levels: int = 2
    pos: tuple = field(default=(6, 6), init=False)
    level_index: int = field(default=0, init=False)

    LAYOUTS = (
        {"start": (6, 6), "goal": (6, 11), "walls": {(5, 6)}, "traps": {(7, 9)}},
        {"start": (8, 5), "goal": (8, 10), "walls": set(), "traps": {(8, 7)}},
    )

    def _layout(self):
        return self.LAYOUTS[min(self.level_index, len(self.LAYOUTS) - 1)]

    def _grid(self):
        g = np.zeros((SIZE, SIZE), dtype=np.int8)
        g[0, :] = 2  # a HUD row inside the border, ticks every action
        g[0, : min(SIZE, len(self.game_run.history) % SIZE if self.game_run else 0)] = 9
        lay = self._layout()
        for r, c in lay["walls"]:
            g[r, c] = 5
        for r, c in lay["traps"]:
            g[r, c] = 8
        g[lay["goal"]] = 4
        g[self.pos] = 3
        return g

    def _state(self, state=arcengine.GameState.NOT_FINISHED):
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id, state=state, levels_completed=self.level_index,
            win_levels=self.levels, available_actions=[1, 2, 3, 4],
        )
        raw.frame = [self._grid()]
        return taaf.game.GameState(raw=raw)

    def _start_game(self, session):
        self.game_id = "slide-0001"
        self.number_of_levels = self.levels
        self.base_actions_per_level = [5] * self.levels
        self.grid_size = (SIZE, SIZE)
        self.pos = self._layout()["start"]
        return self._state()

    def _execute_action(self, action):
        if action.id == arcengine.GameAction.RESET:
            self.pos = self._layout()["start"]
            return self._state()
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action.id.value]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        lay = self._layout()
        if BORDER <= r < SIZE - BORDER and BORDER <= c < SIZE - BORDER and (r, c) not in lay["walls"]:
            self.pos = (r, c)
        if self.pos in lay["traps"]:
            self.pos = lay["start"]
        if self.pos == lay["goal"]:
            self.level_index += 1
            if self.level_index >= self.levels:
                self.level_index = self.levels
                return self._state(arcengine.GameState.WIN)
            self.pos = self._layout()["start"]
        return self._state()


def make_session(game=None, tmpdir=None):
    """A real _HarnessGameSession + ToolAgent wired as the solver wires them."""
    from inference.agent.tool_agent import ToolAgent
    from inference.framework.solver import HarnessSolver, _HarnessGameSession

    game = game or SlideGame()
    game.start_game()
    tmp = Path(tmpdir or tempfile.mkdtemp(prefix="sg_test_"))
    run_dir = tmp / game.game_id
    run_dir.mkdir(parents=True, exist_ok=True)
    solver = HarnessSolver(model="local")
    agent = ToolAgent(model="local")
    session = _HarnessGameSession(
        solver=solver, game=game, analyzer=agent, game_index=0, pass_index=0,
        state_path=run_dir / "tool_runtime_state.json",
        transcript_path=run_dir / "transcript.txt",
        analysis_html_relpath="analysis.html",
        stop_event=threading.Event(),
        viewer_data_path=run_dir / "viewer.json",
    )
    session.seed_initial_history()
    session.write_runtime_state()
    agent._step_env_callback = session.step_env
    from inference.agent.action_names import to_model_actions
    from inference.framework.solver import _engine_action_names
    agent._current_valid_actions = to_model_actions(_engine_action_names(game))
    return session, agent


ENV_DIR = Path(__file__).resolve().parent / ".arc_envs"


def public_game(name):
    """A public ARC-AGI-3 game played fully offline from downloaded env files
    (run fetch_public_games.py once; free, anonymous key). None if absent."""
    if not (ENV_DIR / name).is_dir():
        return None
    import arc_agi
    import taaf.game_api as game_api
    spec = game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR)
    )
    return game_api.GameAPI(env_name=name, arcade_spec=spec)


def run_snippet(agent, session, code):
    """One mock 'model' tool call. Returns (tool_output_text, dispatch)."""
    dispatch = agent._run_python_tool(session.state_path, {"code": code})
    return dispatch.content, dispatch
