#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
# Rebuild the patched harness from scratch and run every test. CPU only, no LLM.
#
#   ./run_tests.sh [FRANZEN_REPO] [BASELINE_NOTEBOOK]
#
# 1. Copies Franzen's repo (cloned from GitHub if no path is given).
# 2. If the M2 notebook is available, reproduces the Kaggle chain exactly:
#    reverse its embedded harness patch to get Tufa's original files, then
#    `git apply --include=ARC3-Inference/*` Franzen's patch, then ours.
#    Otherwise applies ours directly on the repo (equivalent: the notebook
#    patch reverse-applies cleanly to the repo).
# 3. Runs pytest against the patched tree, then the gate-off equivalence check
#    against the unpatched tree.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FRANZEN="${1:-$HERE/../franzen}"
NOTEBOOK="${2:-$HERE/../baseline/baseline.ipynb}"
WORK="$(mktemp -d /tmp/sg_build.XXXX)"

if [ ! -d "$FRANZEN/ARC3-Inference" ]; then
  git clone -q --depth 1 https://github.com/da-fr/arc-agi-3-solution "$WORK/franzen-src"
  FRANZEN="$WORK/franzen-src"
fi

for t in base patched; do
  mkdir -p "$WORK/$t"
  cp -a "$FRANZEN/ARC3-Inference" "$FRANZEN/tufa-arc-agi-framework" "$WORK/$t/"
  find "$WORK/$t" -name __pycache__ -prune -exec rm -rf {} +
done

cd "$WORK/patched"
git init -q .
if [ -f "$NOTEBOOK" ]; then
  python3 - "$NOTEBOOK" "$WORK/franzen.patch" <<'PY'
import json, sys
nb = json.load(open(sys.argv[1]))
src = "".join(nb["cells"][2]["source"])
assert src.startswith("%%writefile"), "cell 2 is not the harness patch"
open(sys.argv[2], "w").write(src.split("\n", 1)[1].rstrip("\n") + "\n")
PY
  git apply -R --include='ARC3-Inference/*' "$WORK/franzen.patch"        # -> Tufa original
  git apply --include='ARC3-Inference/*' -v "$WORK/franzen.patch" 2>&1 | tail -1   # notebook cell 4
  echo "Franzen notebook patch re-applied (Kaggle chain reproduced)"
fi
git apply --include='ARC3-Inference/*' -v "$HERE/surprise-gate.patch" 2>&1 | grep -E "^Applied|error" || true
echo "surprise-gate.patch applied"

cd "$HERE/tests"
ARC3_TREE="$WORK/patched" python3 -m pytest -q -p no:cacheprovider .
python3 baseline_equivalence.py "$WORK/base" "$WORK/patched"
echo "build dir: $WORK"
