"""Юнит-тесты пула подсказок и регрессионный тест шага DP-SGD."""
import copy

import numpy as np

from core.config import ExperimentConfig
from core.pool import Client, PromptPool


def make_client():
    cfg = ExperimentConfig()
    c = Client(0, "test", cfg.pool, cfg.train, np.random.default_rng(0))
    c.ensure_head(0, 3)
    return c, cfg


def test_select_returns_top_n_distinct_cells():
    cfg = ExperimentConfig()
    pool = PromptPool(cfg.pool, np.random.default_rng(0))
    Q = np.random.default_rng(1).normal(size=(8, cfg.pool.D_k))
    idx = pool.select(Q)
    assert idx.shape == (8, cfg.pool.N)
    assert all(len(set(row)) == cfg.pool.N for row in idx)


def test_dpsgd_step_without_noise_equals_plain_step():
    """Регрессия: при бесконечном пороге и нулевом шуме шаг DP-SGD совпадает с обычным."""
    c, cfg = make_client()
    a, b = copy.deepcopy(c), copy.deepcopy(c)
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(16, cfg.pool.D_k))
    y = rng.integers(0, 3, 16)
    a.train_step(Q, y, 0)
    b.train_step_dpsgd(Q, y, 0, None, clip=1e12, z_noise=0.0, divisor=len(Q),
                       noise_rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a.pool.keys, b.pool.keys)
    np.testing.assert_array_equal(a.pool.prompts, b.pool.prompts)
