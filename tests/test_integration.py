"""Интеграционный тест: признаки -> обучение с CKKS -> файл результатов."""
import glob
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("tenseal")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args):
    # UTF-8 явно: в выводе кириллица, а кодировка консоли Windows по умолчанию cp1252
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_end_to_end_on_synthetic_features(tmp_path):
    feats = str(tmp_path / "features.npz")
    out = str(tmp_path / "results")
    run(["scripts/prepare_features.py", "--mode", "synthetic",
         "--n_train", "300", "--n_test", "120", "--out", feats])
    run(["run_experiment.py", "--scenario", "single", "--protocol", "cil", "--crypto", "ckks",
         "--seeds", "1", "--tasks", "3", "--rounds", "2", "--batches", "2",
         "--features", feats, "--out", out])
    files = glob.glob(os.path.join(out, "*.json"))
    assert files, "файл результатов не создан"
    res = json.load(open(files[0], encoding="utf-8"))["results"][0]
    assert 0.0 <= res["FAA"] <= 1.0
    assert res["backend"] == "ckks-tenseal"
