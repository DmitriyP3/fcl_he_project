"""
dp_accountant.py — подсчёт гарантии (eps, delta) для DP-SGD.

Механизм: на каждом шаге каждый снимок попадает в пакет с вероятностью q
(пуассоновская выборка), вклад каждого снимка в градиент ограничен порогом,
к сумме добавляется гауссов шум с множителем z. Расход приватности считается
через дивергенцию Реньи (RDP) для подвыборочного гауссова механизма
(Mironov, Talwar, Zhang, 2019) на целых порядках alpha и переводится в
(eps, delta) улучшенной формулой (Balle et al., 2020) — как в библиотеке
Google dp_accounting.

Важно для непрерывного обучения: снимок пациента используется только во
время своей задачи, данные задач не пересекаются, повторного показа старых
данных нет. Поэтому расход на пациента — это расход шагов его задачи
(параллельная композиция). Это известное свойство; DP в непрерывном обучении
формализовано, например, в Tobaben et al., «Privacy Leakage via Output Label Space
and Differentially Private Continual Learning», TMLR 2026 (arXiv:2411.04680).
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

DEFAULT_ORDERS = list(range(2, 257)) + [300, 400, 512, 768, 1024]


def _logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    return m + math.log(float(np.sum(np.exp(v - m))))


def rdp_sampled_gaussian(q: float, z: float, alpha: int) -> float:
    """RDP одного шага подвыборочного гауссова механизма для целого alpha >= 2."""
    if z <= 0:
        return math.inf
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return alpha / (2.0 * z * z)
    i = np.arange(alpha + 1, dtype=np.float64)
    log_binom = (math.lgamma(alpha + 1)
                 - np.array([math.lgamma(k + 1) for k in i])
                 - np.array([math.lgamma(alpha - k + 1) for k in i]))
    log_terms = (log_binom + i * math.log(q) + (alpha - i) * math.log1p(-q)
                 + (i * i - i) / (2.0 * z * z))
    return _logsumexp(log_terms) / (alpha - 1)


def epsilon(q: float, z: float, steps: int, delta: float,
            orders: Iterable[int] = DEFAULT_ORDERS) -> Tuple[float, int]:
    """
    Гарантия (eps, delta) после steps шагов DP-SGD.
    Возвращает (eps, лучший порядок alpha).
    """
    if steps <= 0:
        return 0.0, 0
    if z <= 0:
        return math.inf, 0
    best, best_a = math.inf, 0
    for a in orders:
        rdp = steps * rdp_sampled_gaussian(q, z, a)
        # улучшенный перевод RDP -> (eps, delta) (Balle et al., 2020)
        eps = (rdp + math.log1p(-1.0 / a)
               - (math.log(delta) + math.log(a)) / (a - 1))
        if eps < best:
            best, best_a = eps, a
    return max(0.0, best), best_a


def noise_for_epsilon(q: float, steps: int, delta: float, target_eps: float,
                      lo: float = 0.3, hi: float = 50.0) -> float:
    """Минимальный множитель шума z, дающий eps <= target_eps (бинарный поиск)."""
    if epsilon(q, hi, steps, delta)[0] > target_eps:
        return math.inf
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if epsilon(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
