"""
leakage.py — продольная (временная) утечка в федеративном непрерывном обучении.

Угроза, специфичная для FCL: честный, но любопытный сервер ничего не
подделывает, а только наблюдает ПОСЛЕДОВАТЕЛЬНОСТЬ того, что присылает
клиника. По ней он пытается узнать:
  1) КОГДА клиника начала осваивать новую задачу (новую группу диагнозов);
  2) КАКУЮ задачу клиника решает сейчас (кластеры наблюдений = задачи?).

Обе вещи — чувствительная информация медицинской организации: новые
направления диагностики, вспышки, изменение профиля пациентов.

Метрики:
  change_point_auc  — AUC обнаружения смены задачи оконным детектором
                      (среднее до против среднего после; 0.5 = угадывание);
  task_id_accuracy  — точность восстановления номера задачи кластеризацией
                      без меток (k-средних) с наилучшим сопоставлением.
Для task_id_accuracy считается эмпирический уровень угадывания:
та же процедура на перемешанных по времени наблюдениях.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np


# ------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(score положительного > score отрицательного) + 0.5*P(равенства)."""
    s = np.round(np.asarray(scores, dtype=np.float64), 12)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _unit(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------
#  1. Обнаружение смены задачи
# ------------------------------------------------------------------
def change_point_scores(F: np.ndarray, task_seq: np.ndarray, w: int = 3):
    """
    Оконный детектор смены режима (стандартная идея скользящих окон):
    оценка раунда r = 1 - cos(среднее F[r-w:r], среднее F[r:r+w]).
    Метка 1, если в раунде r началась новая задача. Раунды ближе w к
    реальной смене исключаются (зона допуска — стандартная практика оценки
    детекторов смены режима: их окно частично захватывает смену).
    Сравниваются средние за несколько раундов, поэтому случайные колебания
    отдельных раундов (выборка пакетов) почти не влияют на оценку.
    """
    U = _unit(F)
    t = np.asarray(task_seq)
    R = len(U)
    switches = [r for r in range(1, R) if t[r] != t[r - 1]]
    scores, labels = [], []
    for r in range(w, R - w + 1):
        a = U[r - w:r].mean(0)
        b = U[r:r + w].mean(0)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if r in switches:
            lab = 1
        elif min([abs(r - sw) for sw in switches] or [R]) < w:
            continue      # зона допуска: окно захватывает реальную смену
        else:
            lab = 0
        scores.append(1.0 - cos)
        labels.append(lab)
    return np.array(scores), np.array(labels)


# ------------------------------------------------------------------
#  2. Определение задачи кластеризацией без меток
# ------------------------------------------------------------------
def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
            n_init: int = 10, iters: int = 50) -> np.ndarray:
    best_lab, best_inertia = None, np.inf
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_lab = inertia, lab
    return best_lab


def _best_match_accuracy(pred: np.ndarray, true: np.ndarray, k: int) -> float:
    best = 0.0
    for perm in itertools.permutations(range(k)):
        acc = float(np.mean(np.array(perm)[pred] == true))
        best = max(best, acc)
    return best


def smooth_sequence(F: np.ndarray, w: int) -> np.ndarray:
    """
    Скользящее среднее по w соседним раундам (окно по центру, у краёв —
    укороченное). Независимый от раунда к раунду шум гасится примерно в
    sqrt(w) раз, а сигнал задачи, который держится много раундов, остаётся.
    """
    F = np.asarray(F, dtype=np.float64)
    if w <= 1:
        return F
    R = len(F)
    out = np.empty_like(F)
    h = w // 2
    for r in range(R):
        a, b = max(0, r - h), min(R, r - h + w)
        out[r] = F[a:b].mean(axis=0)
    return out


def task_id_accuracy(F: np.ndarray, task_seq: np.ndarray,
                     rng: np.random.Generator, shuffle: bool = False,
                     smooth: int = 0) -> float:
    """
    smooth > 1 — атакующий с усреднением по smooth соседним раундам.
    Уровень угадывания (shuffle=True) считается так: раунды перемешиваются
    ДО усреднения. Иначе гладкость усреднённого ряда сама по себе создала бы
    ложную «утечку» на сплошных блоках задач.
    """
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(task_seq)
    if shuffle:
        F = F[rng.permutation(len(F))]
    if smooth and smooth > 1:
        F = smooth_sequence(F, smooth)
    X = _unit(F)
    k = int(len(np.unique(y)))
    lab = _kmeans(X, k, rng)
    return _best_match_accuracy(lab, y, k)


# ------------------------------------------------------------------
#  Сводка по одному уровню наблюдения
# ------------------------------------------------------------------
def sequence_excess(seqs: List[np.ndarray], tasks: List[np.ndarray],
                    seed: int = 0, n_perm: int = 30, smooth: int = 0) -> np.ndarray:
    """Превышение над уровнем угадывания для КАЖДОЙ последовательности."""
    rng = np.random.default_rng(seed)
    out = []
    for F, t in zip(seqs, tasks):
        a = task_id_accuracy(F, t, rng, smooth=smooth)
        nl = np.mean([task_id_accuracy(F, t, rng, shuffle=True, smooth=smooth)
                      for _ in range(n_perm)])
        out.append(a - nl)
    return np.array(out)


def paired_difference(d: np.ndarray, seed: int = 0, n_boot: int = 4000):
    """
    Среднее парных разностей с 95% ДИ (бутстрэп) и p-значением парного
    перестановочного теста со сменой знака (H0: разность симметрична вокруг 0;
    альтернатива: разность > 0).
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = (signs * d[None, :]).mean(axis=1)
    p = (1 + int(np.sum(null >= d.mean()))) / (1 + n_boot)
    return float(d.mean()), float(lo), float(hi), float(p)


def evaluate_sequences(seqs: List[np.ndarray], tasks: List[np.ndarray],
                       seed: int = 0, n_perm: int = 30,
                       n_boot: int = 2000, smooth: int = 0) -> Dict[str, float]:
    """
    Оценка утечки по набору последовательностей наблюдений
    (по одной на клинику и seed) со статистикой:

      task_id_acc        — средняя точность восстановления задачи;
      task_id_acc_chance — средний уровень угадывания: та же процедура на
                           n_perm случайных перестановках наблюдений по времени;
      excess             — превышение над уровнем угадывания;
      excess_ci_lo/hi    — 95% доверительный интервал превышения
                           (бутстрэп по последовательностям);
      p_value            — перестановочный тест: вероятность получить такую
                           точность, если утечки нет (Монте-Карло);
      change_point_auc   — вспомогательная метрика (шумная, выводов не делать).
    """
    rng = np.random.default_rng(seed)
    acc, null = [], []
    all_s, all_l = [], []
    for F, t in zip(seqs, tasks):
        acc.append(task_id_accuracy(F, t, rng, smooth=smooth))
        null.append([task_id_accuracy(F, t, rng, shuffle=True, smooth=smooth)
                     for _ in range(n_perm)])
        s_, l_ = change_point_scores(F, t)
        all_s.append(s_)
        all_l.append(l_)
    acc = np.array(acc)
    null = np.array(null)                      # [последовательности, n_perm]
    excess_i = acc - null.mean(axis=1)

    # бутстрэп доверительного интервала среднего превышения
    n = len(excess_i)
    boots = np.array([excess_i[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    # перестановочный тест для средней точности
    T = acc.mean()
    picks = null[np.arange(n)[None, :], rng.integers(0, n_perm, (n_boot, n))]
    T_null = picks.mean(axis=1)
    p = (1 + int(np.sum(T_null >= T))) / (1 + n_boot)

    cp = np.concatenate(all_s) if all_s else np.array([])
    cl = np.concatenate(all_l) if all_l else np.array([])
    return {
        "task_id_acc": float(acc.mean()),
        "task_id_acc_chance": float(null.mean()),
        "excess": float(excess_i.mean()),
        "excess_ci_lo": float(lo),
        "excess_ci_hi": float(hi),
        "p_value": float(p),
        "change_point_auc": auc_score(cp, cl) if len(cp) else float("nan"),
        "n_sequences": int(n),
        "n_perm": int(n_perm),
    }
