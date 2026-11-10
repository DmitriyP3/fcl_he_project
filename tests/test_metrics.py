"""Юнит-тесты метрик непрерывного обучения и нормализации обновлений."""
import numpy as np
import pytest

from core.config import DefenseConfig
from core.metrics import (apply_defense, backward_transfer,
                          final_average_accuracy, forgetting)

# R[i][j] — точность на задаче j после обучения задаче i
R = np.array([[0.80, 0.00, 0.00],
              [0.60, 0.70, 0.00],
              [0.50, 0.65, 0.90]])


def test_final_average_accuracy():
    # среднее по последней строке: (0.50 + 0.65 + 0.90) / 3
    assert final_average_accuracy(R) == pytest.approx(0.683333, abs=1e-6)


def test_forgetting_and_backward_transfer():
    # задача 0: пик 0.80 -> 0.50 (0.30); задача 1: пик 0.70 -> 0.65 (0.05)
    assert forgetting(R) == pytest.approx(0.175, abs=1e-9)
    assert backward_transfer(R) == pytest.approx(-0.175, abs=1e-9)


def test_no_forgetting_when_accuracy_is_kept():
    R_stable = np.array([[0.7, 0.0], [0.7, 0.8]])
    assert forgetting(R_stable) == pytest.approx(0.0)


def test_percomponent_normalization_sets_norms_and_keeps_direction():
    rng = np.random.default_rng(0)
    dK, dP = rng.normal(size=1920), rng.normal(size=9600)
    cfg = DefenseConfig(mode="percomponent", C_K=0.3, C_P=0.3)
    k2, p2 = apply_defense(dK, dP, cfg)
    assert np.linalg.norm(k2) == pytest.approx(0.3, rel=1e-9)
    assert np.linalg.norm(p2) == pytest.approx(0.3, rel=1e-9)
    cos = k2 @ dK / (np.linalg.norm(k2) * np.linalg.norm(dK))
    assert cos == pytest.approx(1.0, abs=1e-12)
