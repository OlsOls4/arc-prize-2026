"""Sandbox-side helpers for surprise-gated action batches (ARC3_SURPRISE_GATE).

SPDX-License-Identifier: MIT-0  (our addition on top of the Apache-2.0 harness)

Like frame_diff.py and segmentation.py, this module's source is spliced
verbatim into the sandbox bootstrap (python_tool_sandbox.py), which runs under
``python -I -S``. It therefore has stdlib-only imports and no ``from
__future__`` import. The host imports it too, for tests and for describing
predicates.

A *prediction* is attached to one action and judges that action's effect:

* a callable ``f(p, n)`` or ``f(p, n, r)``, where ``p``/``n`` are the frame views
  before/after the action (same API as ``current_frame``) and ``r`` is that
  action's result dict; or
* a string holding one Python expression over the names ``p``, ``n``, ``r``
  (plus every global the snippet can see, e.g. ``frame_diff``).

Verdicts are tri-state: True (held), False (falsified), None (unknown: the
predicate raised, timed out, returned None, or tried to act).
"""
import ast
import signal


PREDICATE_TEXT_LIMIT = 160


class PredicateTimeout(BaseException):
    """Raised inside a predicate that overruns its time slice.

    BaseException so a predicate's own ``except Exception`` cannot swallow it.
    """


def normalize_expectations(expect, count):
    """Align ``expect`` with a batch of ``count`` actions.

    Returns ``(items, note)``. A single callable/string applies to every
    action; a list is padded with None (or truncated) to the batch length,
    which is reported in ``note`` rather than raised: an error here would cost
    the whole snippet before any action ran, for a bookkeeping mistake.
    """
    count = max(0, int(count))
    if expect is None:
        return [None] * count, ""
    if callable(expect) or isinstance(expect, str):
        return [expect] * count, ""
    if isinstance(expect, (list, tuple)):
        items = list(expect)
        note = ""
        if len(items) != count:
            note = (
                f"expect has {len(items)} entries for {count} actions; "
                + ("missing entries count as no prediction." if len(items) < count
                   else "extra entries were ignored.")
            )
        items = (items + [None] * count)[:count]
        cleaned = []
        for item in items:
            if item is None or callable(item) or isinstance(item, str):
                cleaned.append(item)
            else:
                cleaned.append(None)
                note = (note + " " if note else "") + (
                    "non-callable, non-string expect entries were ignored."
                )
        return cleaned, note
    return [None] * count, "expect must be a callable, a string, or a list; it was ignored."


def _shorten(text, limit=PREDICATE_TEXT_LIMIT):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe_predicate(pred, code_text=""):
    """Short source text for a prediction, for the surprise report.

    Snippet code is exec'd from a string, so ``inspect.getsource`` cannot see
    it. Recover the lambda / def text from the snippet source by line number.
    """
    if pred is None:
        return None
    if isinstance(pred, str):
        return _shorten(pred)
    code = getattr(pred, "__code__", None)
    name = getattr(pred, "__name__", "") or "predicate"
    if code is None or not code_text:
        return _shorten(name)
    line = getattr(code, "co_firstlineno", None)
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return _shorten(name)
    argnames = list(getattr(code, "co_varnames", ())[: getattr(code, "co_argcount", 0)])
    matches = []
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.FunctionDef) and node.name == name:
            matches.append(node)
        elif isinstance(node, ast.Lambda) and name == "<lambda>":
            if [a.arg for a in node.args.args] == argnames:
                matches.append(node)
    if not matches:
        return _shorten(name) + _closure_text(pred)
    # several candidates on one line (e.g. a lambda returning a lambda): the
    # innermost one is the function object we were handed
    node = min(matches, key=lambda m: (getattr(m, "end_col_offset", 0) or 0) - m.col_offset)
    if isinstance(node, ast.FunctionDef):
        body = node.body[-1] if node.body else None
        if isinstance(body, ast.Return) and body.value is not None:
            segment = ast.get_source_segment(code_text, body.value)
            if segment:
                return _shorten(f"{node.name}: {segment}" + _closure_text(pred))
        return _shorten(f"def {node.name}(...) at line {line}" + _closure_text(pred))
    segment = ast.get_source_segment(code_text, node) or f"{name} at line {line}"
    return _shorten(segment + _closure_text(pred))


def _closure_text(fn):
    """Simple captured values, so `step(0, 1)` reads as `... [dr=0, dc=1]`."""
    code = getattr(fn, "__code__", None)
    cells = getattr(fn, "__closure__", None) or ()
    if code is None or not cells:
        return ""
    bits = []
    for var, cell in zip(code.co_freevars, cells):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, (int, float, str, bool, tuple)) and len(repr(value)) <= 24:
            bits.append(f"{var}={value!r}")
    return f" [{', '.join(bits)}]" if bits else ""


def _call_with_arity(fn, p, n, r):
    code = getattr(fn, "__code__", None)
    argcount = getattr(code, "co_argcount", 2) if code is not None else 2
    if code is not None and getattr(fn, "__self__", None) is not None:
        argcount -= 1  # bound method
    if argcount >= 3:
        return fn(p, n, r)
    if argcount == 1:
        return fn(n)
    return fn(p, n)


def _timeout_handler(signum, frame):
    raise PredicateTimeout()


def eval_predicate(pred, p, n, r, globals_dict=None, timeout_seconds=0.5):
    """Evaluate one prediction. Returns ``(verdict, error_text)``.

    ``verdict`` is True / False / None. Never raises (except for interpreter
    shutdown signals): any failure is an unknown, not a falsification.
    The time limit uses SIGALRM, so it applies only on the main thread - which
    is where the sandbox runs snippets. Off the main thread it runs unlimited
    (the host-side tool deadline still bounds the whole snippet).
    """
    try:
        return _eval_predicate(pred, p, n, r, globals_dict, timeout_seconds)
    except PredicateTimeout:  # alarm landed between return and disarm
        return None, f"timed out after {timeout_seconds}s"


def _eval_predicate(pred, p, n, r, globals_dict, timeout_seconds):
    if pred is None:
        return None, "no prediction"
    previous_handler = None
    armed = False
    try:
        if timeout_seconds and timeout_seconds > 0 and hasattr(signal, "setitimer"):
            try:
                previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
                armed = True
            except ValueError:  # not on the main thread
                armed = False
        if isinstance(pred, str):
            scope = dict(globals_dict or {})
            local = {"p": p, "n": n, "r": r, "before": p, "after": n, "result": r}
            value = eval(compile(pred, "<expect>", "eval"), scope, local)
        elif callable(pred):
            value = _call_with_arity(pred, p, n, r)
        else:
            return None, "prediction is not callable"
    except PredicateTimeout:
        return None, f"timed out after {timeout_seconds}s"
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc.msg}"
    except Exception as exc:  # noqa: BLE001 - every failure is an unknown
        return None, _shorten(f"{exc.__class__.__name__}: {exc}", 200)
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    if value is None:
        return None, "predicate returned None"
    try:
        return bool(value), ""
    except Exception as exc:  # noqa: BLE001 - e.g. an ambiguous truth value
        return None, _shorten(f"{exc.__class__.__name__}: {exc}", 200)
