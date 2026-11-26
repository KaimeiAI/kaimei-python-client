"""A tree fitted on rows with missing values may split at an infinite threshold, and skl2onnx's conversion of such a
tree never returns. The ONNX route refuses it at once, with why; it used to hang the caller's process for good.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skl2onnx")

SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from kaimei.structure import StructureError, extract_structure

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    X[rng.random(200) < 0.2, 0] = np.nan
    y = np.isnan(X[:, 0]).astype(int)
    for model in (
        DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y),
        RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(X, y),
    ):
        try:
            extract_structure(model, via="onnx", feature_names=["a", "b", "c"])
            print("read")
        except StructureError as exc:
            print("refused:", exc.message)
    """
)


def test_a_tree_that_learned_where_missing_values_go_is_refused_not_hung() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.path.join(os.path.dirname(__file__), "..", "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, done.stdout
    for line in lines:
        assert line.startswith("refused:") and "infinite threshold" in line, line
