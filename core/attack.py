"""
attack.py — честная оценка атаки определения членства (в духе PromptMIA).

Игра безопасности:
  1. Федерация немного обучается (прогрев).
  2. Вредоносный сервер выбирает целевой снимок q* и подменяет N ячеек пула
     «ключами-ловушками», направленными на q* (чтобы именно q* их выбирал).
  3. Клиника-жертва делает шаг обучения на пакете данных. В мире «член»
     в пакете есть q*, в мире «не член» — нет (q* взят из тестовой части).
  4. Атакующий по тому, ЧТО ЕМУ РЕАЛЬНО ВИДНО, пытается угадать мир.

Уровни наблюдения (что видит сервер):
  slot_plain     — открытое обновление клиники по ячейкам (FL без шифрования)
  slot_keys_only — зашифрованы только подсказки, ключи открыты
                   (частичное шифрование)
  aggregate_slot — открытая СУММА обновлений всех клиник по ячейкам
                   (обычная безопасная агрегация: сервер видит агрегат)
  total_norm     — только суммарный размер обновления жертвы ||dK||+||dP||
                   (гипотетический канал «агрегатной нормы»)
  ciphertext     — только шифртексты CKKS обновления жертвы
                   (наша схема: сервер без секретного ключа);
                   признак атакующего — средний байт шифртекста
  ciphertext_len — длина сериализованного шифртекста (TenSEAL сжимает
                   сериализацию, длина слегка колеблется — проверяем,
                   не зависит ли она от содержимого)

Для каждого уровня и каждого режима нормализации считаются AUC и
преимущество атакующего Adv = max по порогам |TPR - FPR|.
AUC = 0.5 и Adv ≈ 0 означают, что атакующий угадывает не лучше монетки.
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from .crypto import Aggregator, HAS_TENSEAL
from .federation import build_clients, local_training, _global_labels, _mask
from .metrics import apply_defense

OBSERVATIONS = ["slot_plain", "slot_keys_only", "aggregate_slot",
                "total_norm", "ciphertext", "ciphertext_len"]
DEFENSES = ["none", "global", "percomponent"]


# ------------------------------------------------------------------
#  Метрики атаки
# ------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Ранги с усреднением при равенстве (как scipy.stats.rankdata)."""
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
    """AUC = P(score члена > score не-члена) + 0.5 * P(равенства)."""
    s = np.round(scores.astype(np.float64), 10)   # равенство с точностью 1e-10
    r = _rankdata(s)
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def advantage(scores: np.ndarray, labels: np.ndarray) -> float:
    """Adv = max по порогам |TPR - FPR| (атакующий сам выбирает направление)."""
    s = np.round(scores.astype(np.float64), 10)
    thr = np.unique(s)
    pos = labels == 1
    best = 0.0
    for t in thr:
        tpr = float((s[pos] >= t).mean())
        fpr = float((s[~pos] >= t).mean())
        best = max(best, abs(tpr - fpr))
    return best


# ------------------------------------------------------------------
#  Построение ловушек и сигналов
# ------------------------------------------------------------------
def craft_pool(keys, prompts, q, n_adv, rng, delta_min=0.02, delta=0.05,
               slots=None):
    """
    Ключи-ловушки по Алгоритму 2 статьи PromptMIA: косинусная близость
    ловушки к цели s выбирается чуть выше максимальной близости цели к
    обычным ключам (s ∈ [s_max+delta_min, s_max+delta_min+delta]),
    остальная часть направления — случайная и ортогональная к цели.
    Цель гарантированно выбирает ВСЕ ловушки; чужие снимки могут задеть
    часть ловушек, но не все сразу — на этом различии и держится сигнал.
    Доля чужих снимков, задевших ловушку, выводится как диагностика.
    slots — какие ячейки заменить (по умолчанию последние n_adv).
    Разумный атакующий заменяет наименее используемые ячейки, чтобы обученные
    ключи остались и продолжали обслуживать обычные снимки.
    """
    K = keys.copy()
    M = K.shape[0]
    if slots is None:
        slots = list(range(M - n_adv, M))
    slots = list(slots)
    benign = [m for m in range(M) if m not in slots]
    scale = float(np.linalg.norm(K[benign], axis=1).mean())
    qn = q / (np.linalg.norm(q) + 1e-12)
    Kb = K[benign] / (np.linalg.norm(K[benign], axis=1, keepdims=True) + 1e-12)
    s_max = float((Kb @ qn).max())
    lo = min(max(s_max + delta_min, 0.05), 0.998)
    hi = min(max(s_max + delta_min + delta, lo + 0.001), 0.999)
    for m in slots:
        s = float(rng.uniform(lo, hi))
        r = rng.standard_normal(qn.shape)
        o = r - (r @ qn) * qn
        o /= (np.linalg.norm(o) + 1e-12)
        K[m] = (s * qn + np.sqrt(max(1.0 - s * s, 0.0)) * o) * scale
    return K, prompts.copy(), slots


def slot_ratio(dK, dP, slots, use_prompts=True):
    """Во сколько раз ячейки-ловушки изменились сильнее остальных."""
    M = dK.shape[0]
    mag = np.linalg.norm(dK, axis=1)
    if use_prompts:
        mag = mag + np.linalg.norm(dP.reshape(M, -1), axis=1)
    adv = np.zeros(M, dtype=bool)
    adv[slots] = True
    return float(mag[adv].mean() / (mag[~adv].mean() + 1e-12))


def cipher_features(agg: Aggregator, vec: np.ndarray):
    """То, что сервер реально получает: байты шифртекстов."""
    import tenseal as ts
    blobs = [ts.ckks_vector(agg.ctx, ch.tolist()).serialize()
             for ch in agg._chunks(vec)]
    raw = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    return float(len(raw)), float(raw.mean())


# ------------------------------------------------------------------
#  Основной прогон атаки
# ------------------------------------------------------------------
def run_attack(cfg, clients_data, n_targets=200, attack_batch=8,
               warmup_rounds=5, victim=0, verbose=True) -> Dict:
    rng = np.random.default_rng(cfg.seed + 777)
    task_id = 0
    cil = cfg.protocol == "cil"

    # --- 1. прогрев федерации на задаче 0 (открытая агрегация) ---
    clients = build_clients(cfg, clients_data)
    for c, cd in zip(clients, clients_data):
        c.ensure_head(task_id, len(cd.tasks[task_id].global_classes))
    K_init = clients[victim].pool.keys.copy()
    agg_plain = Aggregator(cfg, mode="plaintext")
    train_rng = np.random.default_rng(cfg.seed)
    for _ in range(warmup_rounds):
        bk = [c.pool.keys.copy() for c in clients]
        bp = [c.pool.prompts.copy() for c in clients]
        local_training(cfg, clients, clients_data, task_id, train_rng)
        ups = [np.concatenate([(c.pool.keys - bk[i]).ravel(),
                               (c.pool.prompts - bp[i]).ravel()])
               for i, c in enumerate(clients)]
        avg = agg_plain.aggregate(ups)
        ksz = cfg.pool.keys_size
        for i, c in enumerate(clients):
            c.pool.keys = bk[i] + avg[:ksz].reshape(bk[i].shape)
            c.pool.prompts = bp[i] + avg[ksz:].reshape(bp[i].shape)

    # --- 2. цели: половина — члены (train), половина — не члены (test) ---
    vd = clients_data[victim].tasks[task_id]
    half = n_targets // 2
    mem_idx = rng.choice(len(vd.X_train), half, replace=False)
    non_idx = rng.choice(len(vd.X_test), half, replace=False)
    targets = ([(vd.X_train[i], vd.y_train[i], 1) for i in mem_idx] +
               [(vd.X_test[i], vd.y_test[i], 0) for i in non_idx])

    # --- 3. атакующий выбирает наименее использованные ячейки ---
    usage = np.linalg.norm(clients[victim].pool.keys - K_init, axis=1)
    trap_slots = sorted(np.argsort(usage)[: cfg.pool.N].tolist())

    agg_ckks = Aggregator(cfg, mode="ckks") if HAS_TENSEAL else None
    # DP-SGD у клиник (z > 0): шаг атакуемого раунда тоже идёт с отсечением и шумом.
    # Пакет фиксированного размера (а не пуассоновский) — это эмпирическая проверка
    # ослабления атаки, формальная гарантия здесь не пересчитывается.
    dp = getattr(cfg, "dpsgd", None)
    use_dp = dp is not None and dp.z > 0
    dp_rng = np.random.default_rng(cfg.seed * 1000 + 555)
    selectivity = []   # доля ЧУЖИХ снимков пакета, выбравших хотя бы одну ловушку
    target_hits = []   # выбрала ли цель все ловушки
    scores = {(o, d): [] for o in OBSERVATIONS for d in DEFENSES}
    labels = []

    def to_head_labels(cd, task, y):
        return _global_labels(task, y) if cil else y

    def train_mask(cd, task):
        return _mask(cd.n_classes_all, task.global_classes) if cil else None

    for ti, (q, y, is_mem) in enumerate(targets):
        labels.append(is_mem)
        K0, P0, slots = craft_pool(clients[victim].pool.keys,
                                   clients[victim].pool.prompts,
                                   q, cfg.pool.N, rng, slots=trap_slots)
        deltas = []
        for ci, (c, cd) in enumerate(zip(clients, clients_data)):
            cc = copy.deepcopy(c)
            cc.pool.keys, cc.pool.prompts = K0.copy(), P0.copy()
            cc._pool_opt = None   # локальный оптимизатор создаётся заново на раунд
            task = cd.tasks[task_id]
            Xb_idx = rng.choice(len(task.X_train), attack_batch, replace=False)
            Xb = task.X_train[Xb_idx].copy()
            yb = task.y_train[Xb_idx].copy()
            if ci == victim and is_mem:
                Xb[0], yb[0] = q, y                    # цель внутри пакета
            if ci == victim:
                sel = cc.pool.select(Xb[1:] if is_mem else Xb)
                selectivity.append(float(np.mean(
                    [len(set(r) & set(slots)) > 0 for r in sel])))
                target_hits.append(set(cc.pool.select(q[None])[0]) == set(slots))
            if use_dp:
                # клиника обучается с DP-SGD: отсечение вклада каждого снимка + шум
                gc = ((dp.clip_head, dp.clip_prompt, dp.clip_key)
                      if getattr(dp, "clip_mode", "global") == "group" else None)
                cc.train_step_dpsgd(Xb, to_head_labels(cd, task, yb), task_id,
                                    train_mask(cd, task), clip=dp.clip, z_noise=dp.z,
                                    divisor=attack_batch, noise_rng=dp_rng, group_clips=gc)
            else:
                cc.train_step(Xb, to_head_labels(cd, task, yb), task_id,
                              train_mask(cd, task))
            deltas.append((cc.pool.keys - K0, cc.pool.prompts - P0))

        for d in DEFENSES:
            dcfg = copy.deepcopy(cfg.defense)
            dcfg.mode = d
            defended = []
            for dK, dP in deltas:
                k2, p2 = apply_defense(dK.ravel(), dP.ravel(), dcfg)
                defended.append((k2.reshape(dK.shape), p2.reshape(dP.shape)))
            vK, vP = defended[victim]
            aK = np.mean([k for k, _ in defended], axis=0)
            aP = np.mean([p for _, p in defended], axis=0)

            scores[("slot_plain", d)].append(slot_ratio(vK, vP, slots, True))
            scores[("slot_keys_only", d)].append(slot_ratio(vK, vP, slots, False))
            scores[("aggregate_slot", d)].append(slot_ratio(aK, aP, slots, True))
            scores[("total_norm", d)].append(
                float(np.linalg.norm(vK) + np.linalg.norm(vP)))
            if agg_ckks is not None:
                clen, byte_mean = cipher_features(
                    agg_ckks, np.concatenate([vK.ravel(), vP.ravel()]))
                scores[("ciphertext", d)].append(byte_mean)
                scores[("ciphertext_len", d)].append(clen)

        if verbose and (ti + 1) % 50 == 0:
            print(f"    цели: {ti + 1}/{len(targets)}")

    labels = np.array(labels)
    table = {}
    for (o, d), v in scores.items():
        if not v:
            continue
        v = np.array(v)
        table[f"{o}|{d}"] = {"AUC": auc_score(v, labels),
                             "Adv": advantage(v, labels)}
    return {"table": table, "n_targets": len(targets),
            "trap_slots": trap_slots,
            "false_trap_rate": float(np.mean(selectivity)),
            "target_selects_traps": float(np.mean(target_hits)),
            "attack_batch": attack_batch, "warmup_rounds": warmup_rounds,
            "protocol": cfg.protocol, "seed": cfg.seed,
            "dpsgd_z": float(dp.z) if use_dp else 0.0,
            "ciphertext_available": agg_ckks is not None}
