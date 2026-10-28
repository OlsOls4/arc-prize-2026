# MIT-0 (OlsOls4 additions: surprise-gate apply + scorecard keepalive). Rest derived from Franzen's ARC-AGI-3 Milestone 2 notebook (Apache-2.0).
# Notebook cell 7 of the v8 Kaggle run. Notebook: https://www.kaggle.com/code/olsols4/arc-agi-3-surprise-gated-agent
import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

# Kaggle might cancel the run in no action is done in some timeframe.
# To prevent this, harness will start up even if backend is not yet ready,
# and the first started game will perform a single reset action.
NOTEBOOK_START_TIME = time.time()
SERVER_STARTUP_TIMEOUT = 12 * 60
os.environ['ARC3_WARMUP_ACTION_GAMES'] = '1'
# v7/v8: Kaggle reruns after ~20 Oct load the model several times slower (>1h seen), which made hidden reruns score 0.00;
# wait up to 90 min for the server on each game's first request and tolerate more failures.
os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '5400'
os.environ['ARC3_MAX_ANALYZER_FAILURES'] = '60'

# model and dataset paths
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
ORIG_BUNDLE_DIR   = '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'
SERVED_MODEL_NAME = 'flashnext'

# Apply harness patch
BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')
!rm -Rf $BUNDLE_DIR
!cp -a $ORIG_BUNDLE_DIR $BUNDLE_DIR
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/harness-changes.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('harness patch applied successfully')

# Surprise-gated action batches (OlsOls4, MIT-0): applied on top of Franzen's patch.
subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/surprise-gate.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)
print('surprise-gate patch applied successfully')

# v8 (OlsOls4, MIT-0): scorecard keepalive. Kaggle's gateway closes a scorecard with no action for ~15 min,
# and since ~20 Oct slow model loads (35-60 min) made hidden reruns score 0.00. While the model server is not
# healthy, the warmup game sends a RESET every 7 min (up to 2 h) so the run stays alive until the server is up.
_sp = BUNDLE_DIR / "src" / "ARC3-Inference" / "inference" / "framework" / "solver.py"
_s = _sp.read_text()
_old = """            self._execute_auto_reset()
        try:
            retry_analysis_step: int | None = None
            consecutive_failures = 0
"""
_new = """            self._execute_auto_reset()
            import threading as _kthreading
            def _keepalive(_self=self, _game_id=run.game_id):
                import time as _ktime
                import urllib.request as _kurl
                def _healthy():
                    try:
                        with _kurl.urlopen("http://127.0.0.1:8001/health", timeout=5) as _r:
                            return _r.status == 200
                    except Exception:
                        return False
                _t0 = _ktime.monotonic()
                while _ktime.monotonic() - _t0 < 7200:
                    for _ in range(14):
                        _ktime.sleep(30)
                        if _healthy() or _self.should_stop():
                            return
                    try:
                        if _healthy():
                            return
                        _self._execute_auto_reset()
                        log.warning("%s: keepalive RESET while model server not healthy", _game_id)
                    except Exception as _exc:
                        log.warning("%s: keepalive RESET failed: %r", _game_id, _exc)
            _kthreading.Thread(target=_keepalive, name="scorecard-keepalive", daemon=True).start()
        try:
            retry_analysis_step: int | None = None
            consecutive_failures = 0
"""
assert _s.count(_old)==1, "keepalive anchor missing"
_sp.write_text(_s.replace(_old, _new))
print('keepalive patch applied')

# True only inside a real competition rerun; switches diagnostics + soft deadline.
TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()

# Non-interactive matplotlib backend: diagnostics render plots with no display attached.
os.environ["MPLBACKEND"] = "Agg"
# Marks the run as a (real or emulated) submission so the framework + solver can adjust.
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
# In submission, disable the periodic JSON/HTML diagnostics writes and per-frame logging.
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
# Pin arc_agi's cached level_reset_only before its client is built (RESET keeps the level).
os.environ["ONLY_RESET_LEVELS"] = "true"

# Everything the run produces is written here.
WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"taaf.kaggle: TRUE_SUBMISSION={TRUE_SUBMISSION}")
SERVED_MODEL_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
SERVER_BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}/v1'
USE_PRIORITY_SCHEDULING = True

setup_env = {
    ########## transformers config ##########
    'USE_TF': '0',
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_TORCHVISION': '1',

    ########## serving config ##########
    'LOCAL_ANALYZER_BASE_URL': SERVER_BASE_URL,
    'OPENAI_BASE_URL': SERVER_BASE_URL,
    'LOCAL_ANALYZER_PROVIDER': 'vllm',
    'OPENAI_PROVIDER': 'vllm',
    'LOCAL_ANALYZER_MODEL_ID': SERVED_MODEL_NAME,
    'INFERENCE_ANALYZER_MODEL': SERVED_MODEL_NAME,
    'LOCAL_ANALYZER_APP_NAME': 'ARC3 Agent Harness',
    
    ########## analyzer config ##########
    'ARC3_REASONING_HISTORY_KEY': 'reasoning_content',
    'LOCAL_ANALYZER_ENABLE_THINKING': 'true',
    'LOCAL_ANALYZER_TOOL_STEPS': '0',
    'LOCAL_ANALYZER_TOOL_TIMEOUT': '30',
    'LOCAL_ANALYZER_YIELD_SECONDS': '0',
    'LOCAL_ANALYZER_YIELD_TOKENS': '2048',
    'LOCAL_ANALYZER_TEMPERATURE': '0.7',
    'LOCAL_ANALYZER_TOP_P': '0.95',
    'LOCAL_ANALYZER_TOP_K': '20',

    ########## context pruning ##########
    'LOCAL_ANALYZER_CONTEXT_WINDOW': (116+12)*1024,
    'LOCAL_ANALYZER_MAX_OUTPUT': 12*1024,
    'ARC3_CONTEXT_DRAIN_TOKENS': 58*1024,
    'ARC3_HISTORY_ASSISTANT_TURNS': 150,
    'ARC3_HISTORY_TURN_DRAIN': 30,
    'ARC3_HISTORY_DRAIN_COALESCE': '1',

    ########## tool output ##########
    'LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS': '3072',
    'ARC3_MIDDLE_TRUNCATION': '1',
    'ARC3_ACTION_ECHO': '1',
    
    ########## prompt changes ##########
    'ARC3_EXPLAIN_GAMEPLAY_CHANGED': '1',
    'ARC3_NEW_CHANGED_PROMPTS': '1',
    'ARC3_LEVEL_TRANSFER_GUIDANCE': '1',
    'ARC3_ACTION_INFO': '1',
    'ARC3_DEDUPE_MULTICALL_LINE': '1',
    'ARC3_FRAME_DIFF_HINT': '1',

    ########## vision & animations ##########
    'MULTIMODAL_CONTEXT': 'current_grid',
    'MULTIMODAL_UPSCALE': '10',
    'ARC3_DIFF_IMAGE': '1',
    'ARC3_GAMEOVER_DIFF_IMAGE': '1',
    'ARC3_ANIMATION': '1',

    ########## guards ##########
    'ARC3_GUARDS_FROM_LEVEL': 2,
    'ARC3_BATCH_NOOP_BLOCK': '1',
    'ARC3_STALE_STATE_BLOCK': '1',

    ########## world model ##########
    'ARC3_MEMORY_SECTIONS': 'off',

    ########## persistent functions ##########
    'ARC3_PERSISTENT_FUNCTIONS': '1',
    'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
    'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',

    ########## controls ##########
    'EXPOSE_UNDO': 'on',
}

if USE_PRIORITY_SCHEDULING: setup_env.update({
    'ARC3_MAX_ACTIVE_STREAMS': 10,
    'ARC3_PRIORITY_REFRESH_QUEUE': '1',
    'ARC3_PRIORITY_PACE': '0',
    'ARC3_PRIORITY_TAIL_FADE': '1',
    'ARC3_PRIORITY_TAIL_FADE_FRACTION': '0.2',
    'ARC3_PRIORITY_TAIL_LOOKUP': 'remaining',
    'ARC3_PRIORITY_SCORE_NORMALIZATION': '1',
})

########## surprise gate (ours) ##########
SURPRISE_GATE_ENV = {
    'ARC3_SURPRISE_GATE': '1',              # '0' = exact M2 baseline
    'ARC3_SURPRISE_GATE_MAX_ABORTS': 6,     # per level; 0 = shadow (log only)
    'ARC3_SURPRISE_GATE_DEFAULT': 'changed',
    'ARC3_SURPRISE_GATE_TIMEOUT': '0.5',
    'ARC3_SURPRISE_GATE_FROM_LEVEL': 1,
    'ARC3_SURPRISE_GATE_PROMPT': '1',
}
setup_env.update(SURPRISE_GATE_ENV)
print('surprise gate config:', SURPRISE_GATE_ENV)

os.environ.update({k: str(v) for k,v in setup_env.items()})