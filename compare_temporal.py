#!/usr/bin/env python3
"""
compare_temporal.py — утечка ИМЕННО задачи: основной прогон против контроля.

Атакующий, восстанавливающий «текущую задачу» по последовательности
обновлений, может ловить не смену диагнозов, а просто стадию обучения
(обновления меняются от начала к концу и без смены задачи). Поэтому:

  основной прогон — клиника меняет задачи (сценарий temporal);
  контроль        — все фазы на ОДНОЙ задаче (temporal --control), те же сиды.

Для каждой клиники и сида: разность превышений (основной − контроль).
Это информация о смене задачи СВЕРХ естественного дрейфа обучения.
Выводится среднее с 95% ДИ и p-значение парного перестановочного теста.

Два p-значения:
  p (послед.) — каждая последовательность (клиника × сид) считается
                независимой. Это завышает уверенность: три последовательности
                одного сида получены в ОДНОМ прогоне, а у клиники-участника
                каждая сумма содержит обновления тех же соседних клиник.
  p (по сидам) — разности сначала усредняются внутри сида, тест идёт по
                 сидам (независимым прогонам). Это корректный уровень
                 повторности; выводы делать по нему.

    python compare_temporal.py --main results/temporal_cil/temporal_observations.npz \\
                               --control results/temporal_cil_control/temporal_observations.npz
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.leakage import sequence_excess, paired_difference, auc_score, _unit


def jump_stat(F, w=3):
    """Резкость самого сильного скачка относительно типичного изменения."""
    U = _unit(F)
    R = len(U)
    vals = []
    for r in range(w, R - w + 1):
        a = U[r - w:r].mean(0)
        b = U[r:r + w].mean(0)
        vals.append(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    vals = np.array(vals)
    return float(vals.max() / (np.median(vals) + 1e-12))


def jump_auc(main_seqs, ctrl_seqs, n_perm=2000, seed=0):
    """AUC атакующего по скачкам: отличить «сменила задачу» от «не меняла»."""
    sm = np.array([jump_stat(F) for F in main_seqs])
    sc = np.array([jump_stat(F) for F in ctrl_seqs])
    sc_all = np.r_[sm, sc]
    y = np.r_[np.ones(len(sm)), np.zeros(len(sc))]
    auc = auc_score(sc_all, y)
    rng = np.random.default_rng(seed)
    null = np.array([auc_score(sc_all, rng.permutation(y)) for _ in range(n_perm)])
    p = (1 + int(np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)))) / (1 + n_perm)
    return auc, p


def per_seed_means(d, n_runs):
    """Среднее по последовательностям одного прогона (сида): [n_runs]."""
    d = np.asarray(d, dtype=np.float64)
    k = len(d) // max(1, n_runs)
    if k < 1 or k * n_runs != len(d):
        return d                      # не удалось разложить по сидам — как есть
    return d.reshape(n_runs, k).mean(axis=1)


ap = argparse.ArgumentParser()
ap.add_argument("--main", required=True)
ap.add_argument("--control", required=True)
ap.add_argument("--n_perm", type=int, default=50)
ap.add_argument("--smooth", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

M = np.load(a.main, allow_pickle=True)["data"].item()
C = np.load(a.control, allow_pickle=True)["data"].item()

print("=" * 100)
print(" УТЕЧКА ЗАДАЧИ = превышение в основном прогоне − превышение в контроле (те же сиды)")
if a.smooth > 1:
    print(f" атакующие усредняют наблюдения по {a.smooth} соседним раундам")
print("=" * 100)
print(f"{'кто и что наблюдает':<58}{'атакующий':<10}{'основной':>9}{'контроль':>10}"
      f"{'разность [95% ДИ]':>26}{'p послед.':>11}{'p по сидам':>12}")
res = {}
for name in M:
    if name not in C:
        continue
    res[name] = {}
    n_runs = min(len(M[name].get("FAA", [])), len(C[name].get("FAA", [])))
    for kind, label in [("weak", "слабый"), ("strong", "сильный")]:
        em = sequence_excess(M[name][kind], M[name]["tasks"], seed=0,
                             n_perm=a.n_perm, smooth=a.smooth)
        ec = sequence_excess(C[name][kind], C[name]["tasks"], seed=0,
                             n_perm=a.n_perm, smooth=a.smooth)
        n = min(len(em), len(ec))
        mean, lo, hi, p = paired_difference(em[:n] - ec[:n])
        ds = per_seed_means(em[:n] - ec[:n], n_runs) if n_runs else em[:n] - ec[:n]
        mean_s, lo_s, hi_s, p_s = paired_difference(ds)
        star = "*" if p < 0.05 else " "
        star_s = "*" if p_s < 0.05 else " "
        res[name][kind] = {"main": float(em.mean()), "control": float(ec.mean()),
                           "diff": mean, "ci_lo": lo, "ci_hi": hi, "p": p, "n": int(n),
                           "ci_lo_seed": lo_s, "ci_hi_seed": hi_s, "p_seed": p_s,
                           "n_seeds": int(len(ds))}
        nm = name if kind == "weak" else ""
        print(f"{nm:<58}{label:<10}{em.mean():>+9.3f}{ec.mean():>+10.3f}"
              f"{mean:>+12.3f} [{lo:+.2f};{hi:+.2f}]{p:>10.3f}{star}{p_s:>11.3f}{star_s}")
print("\n" + "=" * 100)
print(" АТАКУЮЩИЙ ПО СКАЧКАМ: отличить клинику, сменившую задачу, от не менявшей")
print(" (AUC 0.5 = не отличить; p — перестановочный тест, двусторонний)")
print("=" * 100)
print(f"{'кто и что наблюдает':<58}{'слабый AUC':>12}{'p':>8}{'сильный AUC':>14}{'p':>8}")
for name in M:
    if name not in C:
        continue
    aw, pw = jump_auc(M[name]["weak"], C[name]["weak"])
    as_, ps = jump_auc(M[name]["strong"], C[name]["strong"])
    res[name]["jump"] = {"weak_auc": aw, "weak_p": pw, "strong_auc": as_, "strong_p": ps}
    print(f"{name:<58}{aw:>12.3f}{pw:>7.3f}{'*' if pw < 0.05 else ' '}"
          f"{as_:>14.3f}{ps:>7.3f}{'*' if ps < 0.05 else ' '}")

print("\n  основной — превышение над угадыванием при смене задач;")
print("  контроль — то же без смены задач (только дрейф обучения);")
print("  разность — утечка именно информации о задаче; * — значимо (p < 0.05).")
print("  p послед. считает клиники одного сида независимыми и потому занижен;")
print("  выводы делать по «p по сидам» (повторность = независимый прогон).")
print("  Утверждать «сервер узнаёт задачу клиники» можно только по строкам со * в «p по сидам».")
out = a.out or os.path.join(os.path.dirname(a.main), f"compare_temporal_smooth{a.smooth}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(f"\n  сохранено: {out}")
