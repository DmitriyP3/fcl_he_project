"""
metrics.py — метрики федеративного непрерывного обучения и защита.

Главное здесь — матрица R и метрики забывания, которых не было в прежней версии.

Матрица R размера T x T:
    R[i][j] = точность на задаче j ПОСЛЕ окончания обучения задаче i.

Из неё считаются:
    FAA (Final Average Accuracy) = mean_j R[T-1][j]
        средняя точность по всем задачам в самом конце
    Forgetting = mean_{j<T-1} ( max_{i<T-1} R[i][j] - R[T-1][j] )
        насколько каждая задача просела от своего пика
    BWT (Backward Transfer) = mean_{j<T-1} ( R[T-1][j] - R[j][j] )
        отрицательный BWT = забывание
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


# ============================================================
#  Метрики непрерывного обучения
# ============================================================

def final_average_accuracy(R: np.ndarray) -> float:
    """Средняя точность по всем задачам после обучения последней задаче."""
    return float(np.mean(R[-1, :]))


def forgetting(R: np.ndarray) -> float:
    """
    Среднее забывание: насколько каждая задача просела от своего максимума
    к концу обучения. Считается по всем задачам, кроме последней.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = []
    for j in range(T - 1):
        peak = float(np.max(R[: T - 1, j]))
        vals.append(peak - float(R[T - 1, j]))
    return float(np.mean(vals))


def backward_transfer(R: np.ndarray) -> float:
    """BWT: среднее (итог - точность сразу после изучения задачи)."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    vals = [float(R[T - 1, j] - R[j, j]) for j in range(T - 1)]
    return float(np.mean(vals))


def learning_accuracy(R: np.ndarray) -> float:
    """Средняя точность сразу после изучения каждой задачи (диагональ)."""
    return float(np.mean(np.diag(R)))


def summarize(R: np.ndarray) -> dict:
    return {
        "FAA": final_average_accuracy(R),
        "Forgetting": forgetting(R),
        "BWT": backward_transfer(R),
        "LearningAcc": learning_accuracy(R),
        "R": R.tolist(),
    }


# ============================================================
#  Защита: нормализация обновлений перед шифрованием
# ============================================================

def _scale_to_norm(v: np.ndarray, target: float) -> np.ndarray:
    n = float(np.sqrt(np.sum(v * v)))
    if n < 1e-12:
        return v.copy()
    return v * (target / n)


def apply_defense(dK: np.ndarray, dP: np.ndarray, cfg, rng=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применяет защитное преобразование к обновлению пула.

    none          — без защиты
    global        — общая норма sqrt(||dK||^2+||dP||^2) приводится к C
    percomponent  — ||dK|| -> C_K и ||dP|| -> C_P независимо
    """
    mode = cfg.mode
    if mode == "none":
        return dK, dP

    if mode == "global":
        total = float(np.sqrt(np.sum(dK * dK) + np.sum(dP * dP)))
        if total < 1e-12:
            return dK, dP
        f = cfg.C / total
        return dK * f, dP * f

    if mode == "percomponent":
        return _scale_to_norm(dK, cfg.C_K), _scale_to_norm(dP, cfg.C_P)

    if mode == "dp":
        # Базовый метод сравнения — шумовое возмущение в духе локальной DP
        # на уровне клиента: отсечение нормы до C + гауссов шум N(0,(sigma*C)^2).
        total = float(np.sqrt(np.sum(dK * dK) + np.sum(dP * dP)))
        f = min(1.0, cfg.C / total) if total > 1e-12 else 1.0
        dK, dP = dK * f, dP * f
        sigma = float(getattr(cfg, "sigma", 0.0))
        if sigma > 0:
            rng = rng if rng is not None else np.random.default_rng(0)
            std = sigma * cfg.C
            dK = dK + rng.normal(0.0, std, dK.shape)
            dP = dP + rng.normal(0.0, std, dP.shape)
        return dK, dP

    raise ValueError(f"Неизвестный режим защиты: {mode}")


def norms(dK: np.ndarray, dP: np.ndarray) -> Tuple[float, float]:
    return (float(np.sqrt(np.sum(dK * dK))),
            float(np.sqrt(np.sum(dP * dP))))
