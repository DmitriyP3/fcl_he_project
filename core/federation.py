"""
federation.py — главный цикл федеративного НЕПРЕРЫВНОГО обучения.

    для task_id в 0..T-1:
        для round в 0..R-1:
            каждый клиент учится на своей части задачи task_id
            вычисляет обновление пула (дельту)
            применяет нормализацию (если задана)
            отправляет на агрегацию (plaintext или CKKS, сервер без секретного ключа)
            получает усреднённое обновление и применяет к своему пулу
        после задачи: оценка на ВСЕХ задачах 0..task_id -> строка матрицы R

Протокол til: номер задачи известен при тестировании (своя голова на задачу).
Протокол cil: номер задачи НЕ известен; одна голова, выбор среди всех
изученных классов клиента.
"""
from __future__ import annotations

from typing import List
import numpy as np

from .pool import Client
from .crypto import Aggregator
from .metrics import apply_defense, norms, summarize


def _global_labels(task, y_local):
    """Перевод локальных меток задачи (0..K_t-1) в номера классов клиента."""
    return np.asarray(task.global_classes, dtype=np.int64)[y_local]


def _mask(n_all: int, classes) -> np.ndarray:
    m = np.zeros(n_all, dtype=bool)
    m[list(classes)] = True
    return m


def build_clients(cfg, clients_data) -> List[Client]:
    clients = [
        Client(cd.client_id, cd.name, cfg.pool, cfg.train,
               np.random.default_rng(cfg.seed + 100 + cd.client_id),
               protocol=cfg.protocol, n_classes_all=cd.n_classes_all)
        for cd in clients_data
    ]
    # общий стартовый пул у всех (федерация начинает с одинакового состояния)
    k0, p0 = clients[0].pool.flatten()
    for c in clients:
        c.pool.load_flat(k0.copy(), p0.copy())
    return clients


def local_training(cfg, clients, clients_data, task_id, rng):
    """Локальное обучение всех клиентов на текущей задаче (один раунд)."""
    losses = []
    cil = cfg.protocol == "cil"
    data_task = 0 if getattr(cfg, "control_same_task", False) else task_id
    dp = getattr(cfg, "dpsgd", None)
    use_dp = dp is not None and dp.z > 0
    for c, cd in zip(clients, clients_data):
        task = cd.tasks[data_task]
        X, y = task.X_train, task.y_train
        if len(X) == 0:
            continue
        if cil:
            y = _global_labels(task, y)
            mask = _mask(cd.n_classes_all, task.global_classes)
        else:
            mask = None
        if use_dp:
            # DP-SGD: пуассоновская выборка, счёт фактических шагов по задачам
            N = len(X)
            q = min(1.0, cfg.train.batch_size / N)
            if not hasattr(c, "dp_noise_rng"):
                c.dp_noise_rng = np.random.default_rng(cfg.seed * 1000 + 31 + c.id)
                c.dp_log = {}
            for _ in range(cfg.train.batches_per_round):
                sel = np.nonzero(rng.random(N) < q)[0]
                gc = ((dp.clip_head, dp.clip_prompt, dp.clip_key)
                      if getattr(dp, "clip_mode", "global") == "group" else None)
                losses.append(c.train_step_dpsgd(
                    X[sel], y[sel], task_id, mask, clip=dp.clip, z_noise=dp.z,
                    divisor=q * N, noise_rng=c.dp_noise_rng, group_clips=gc))
                qs, st = c.dp_log.get(data_task, (q, 0))
                c.dp_log[data_task] = (q, st + 1)
        else:
            for _ in range(cfg.train.batches_per_round):
                bs = min(cfg.train.batch_size, len(X))
                idx = rng.choice(len(X), bs, replace=False)
                losses.append(c.train_step(X[idx], y[idx], task_id, mask))
    return [l for l in losses if l == l]   # без NaN от пустых пакетов


def privacy_report(cfg, clients) -> dict:
    """
    Гарантия для пациента: снимок участвует только в шагах своей задачи,
    задачи не пересекаются -> eps пациента = eps шагов его задачи
    (параллельная композиция). Для сравнения — eps, если бы снимок
    участвовал во всех шагах всех задач (последовательная композиция).
    """
    from .dp_accountant import epsilon
    dp = cfg.dpsgd
    per = []
    for c in clients:
        for t, (q, steps) in sorted(getattr(c, "dp_log", {}).items()):
            eps, _ = epsilon(q, dp.z, steps, dp.delta)
            per.append({"client": c.id, "task": t, "q": q, "steps": steps, "eps": eps})
    if not per:
        return {}
    worst = max(per, key=lambda r: r["eps"])
    total_steps = sum(r["steps"] for r in per if r["client"] == worst["client"])
    eps_seq, _ = epsilon(worst["q"], dp.z, total_steps, dp.delta)
    return {"z": dp.z, "clip": dp.clip, "delta": dp.delta,
            "clip_mode": getattr(dp, "clip_mode", "global"),
            "eps_per_patient": worst["eps"], "per_task": per,
            "eps_if_sequential_all_tasks": eps_seq}


def evaluate_all(cfg, clients, clients_data, task_id, R):
    """Строка матрицы R: точность на всех задачах 0..task_id."""
    cil = cfg.protocol == "cil"
    for j in range(cfg.n_tasks):
        if j > task_id:
            R[task_id, j] = 0.0
            continue
        accs = []
        for c, cd in zip(clients, clients_data):
            t = cd.tasks[j]
            if cil:
                seen = set()
                for tt in cd.tasks[: task_id + 1]:
                    seen.update(tt.global_classes)
                accs.append(c.evaluate(t.X_test, _global_labels(t, t.y_test),
                                       j, _mask(cd.n_classes_all, seen)))
            else:
                accs.append(c.evaluate(t.X_test, t.y_test, j))
        R[task_id, j] = float(np.mean(accs))


_PROJ_CACHE = {}


def projection_matrix(dim_in: int, dim_out: int = 256) -> np.ndarray:
    """
    Фиксированная случайная проекция (одна и та же во всех прогонах).
    Сохраняет углы между векторами (лемма Джонсона–Линденштрауса), поэтому
    атакующий по проекции видит почти то же, что по полному обновлению.
    """
    key = (dim_in, dim_out)
    if key not in _PROJ_CACHE:
        g = np.random.default_rng(12345)
        _PROJ_CACHE[key] = (g.standard_normal((dim_out, dim_in)) /
                            np.sqrt(dim_out)).astype(np.float32)
    return _PROJ_CACHE[key]


def run_federated_continual(cfg, clients_data, verbose: bool = True,
                            return_clients: bool = False) -> dict:
    """Прогоняет полный эксперимент и возвращает словарь с результатами.
    return_clients=True — добавить обученные клиники (для сохранения модели, system.py)."""
    if getattr(cfg, "control_same_task", False) and cfg.protocol != "cil":
        raise ValueError("Контрольный режим (одна задача во всех фазах) "
                         "поддерживается только для --protocol cil")
    rng = np.random.default_rng(cfg.seed)
    T = cfg.n_tasks
    clients = build_clients(cfg, clients_data)
    agg = Aggregator(cfg, mode=cfg.crypto_mode)

    R = np.zeros((T, T), dtype=np.float64)
    norm_log = []   # служебная диагностика норм клиентских обновлений
    dp_rng = np.random.default_rng(cfg.seed + 999)   # шум режима dp
    rec = bool(getattr(cfg, "record_observations", False))
    obs_slot, obs_agg, obs_task = [], [], []         # что видит сервер
    obs_ins = []    # что видит клиника-участник: сумма обновлений ОСТАЛЬНЫХ
    obs_pslot, obs_pagg, obs_pins = [], [], []   # то же, полное обновление (проекция)
    Pm = projection_matrix(cfg.pool.total_floats) if rec else None
    M = cfg.pool.M

    if verbose:
        print(f"\n[режим] криптография = {agg.backend_name()}, "
              f"защита = {cfg.defense.mode}, протокол = {cfg.protocol}, "
              f"seed = {cfg.seed}")

    for task_id in range(T):
        if verbose:
            print(f"\n=== ЗАДАЧА {task_id} / {T-1} ===")
        for c, cd in zip(clients, clients_data):
            c.ensure_head(task_id, len(cd.tasks[task_id].global_classes))

        for rnd in range(cfg.train.rounds_per_task):
            before_k = [c.pool.keys.copy() for c in clients]
            before_p = [c.pool.prompts.copy() for c in clients]

            for c in clients:
                c.usage[:] = 0.0
            losses = local_training(cfg, clients, clients_data, task_id, rng)

            updates = []
            for ci, c in enumerate(clients):
                dK = (c.pool.keys - before_k[ci]).reshape(-1)
                dP = (c.pool.prompts - before_p[ci]).reshape(-1)
                nk_raw, np_raw = norms(dK, dP)
                dK, dP = apply_defense(dK, dP, cfg.defense, dp_rng)
                nk, npp = norms(dK, dP)
                norm_log.append({
                    "task": task_id, "round": rnd, "client": ci,
                    "key_norm_raw": nk_raw, "prompt_norm_raw": np_raw,
                    "key_norm": nk, "prompt_norm": npp, "sum_norm": nk + npp,
                })
                if getattr(cfg, "aggregation", "uniform") == "usage":
                    # (счётчик × обновление) по ячейкам + сами счётчики в конце
                    n_m = clients[ci].usage
                    dK = (dK.reshape(M, -1) * n_m[:, None]).reshape(-1)
                    dP = (dP.reshape(M, -1) * n_m[:, None]).reshape(-1)
                    updates.append(np.concatenate([dK, dP, n_m]))
                else:
                    updates.append(np.concatenate([dK, dP]))

            if rec:
                # профиль обновления по ячейкам пула: ||dK_m|| + ||dP_m||
                ksz0 = cfg.pool.keys_size
                base = cfg.pool.total_floats
                def _prof(u):
                    u = u[:base]
                    k = np.linalg.norm(u[:ksz0].reshape(M, -1), axis=1)
                    p_ = np.linalg.norm(u[ksz0:].reshape(M, -1), axis=1)
                    return k + p_
                obs_slot.append(np.stack([_prof(u) for u in updates]))
                obs_agg.append(_prof(np.mean(np.stack(updates), axis=0)))
                obs_pslot.append(np.stack([Pm @ u[:base].astype(np.float32) for u in updates]))
                obs_pagg.append(Pm @ np.mean(np.stack(updates), axis=0)[:base].astype(np.float32))
                obs_task.append(task_id)

            avg = agg.aggregate(updates)
            ksz = cfg.pool.keys_size
            if rec and getattr(cfg, "aggregation", "uniform") == "uniform":
                # Клиника расшифровывает среднее и знает своё обновление,
                # значит знает сумму обновлений остальных: n*среднее - своё.
                base = cfg.pool.total_floats
                tot = len(updates) * avg[:base]
                obs_ins.append(np.stack([_prof(tot - u[:base]) for u in updates]))
                obs_pins.append(np.stack([Pm @ (tot - u[:base]).astype(np.float32)
                                          for u in updates]))
            if getattr(cfg, "aggregation", "uniform") == "usage":
                # деление выполняют клиники ПОСЛЕ расшифровки
                base = cfg.pool.total_floats
                cnt = avg[base:base + M]
                safe = np.where(cnt > 1e-9, cnt, 1.0)
                keep = (cnt > 1e-9).astype(np.float64)[:, None]
                avg_k = avg[:ksz].reshape(M, -1) / safe[:, None] * keep
                avg_p = avg[ksz:base].reshape(M, -1) / safe[:, None] * keep
                avg_k = avg_k.reshape(cfg.pool.M, cfg.pool.D_k)
                avg_p = avg_p.reshape(cfg.pool.M, cfg.pool.L_p, cfg.pool.D)
            else:
                avg_k = avg[:ksz].reshape(cfg.pool.M, cfg.pool.D_k)
                avg_p = avg[ksz:].reshape(cfg.pool.M, cfg.pool.L_p, cfg.pool.D)
            for ci, c in enumerate(clients):
                c.pool.keys = before_k[ci] + avg_k
                c.pool.prompts = before_p[ci] + avg_p

            if verbose and (rnd + 1) % max(1, cfg.train.rounds_per_task // 4) == 0:
                ml = float(np.mean(losses)) if losses else float("nan")
                print(f"  раунд {rnd+1:>3}/{cfg.train.rounds_per_task}  loss={ml:.4f}")

        evaluate_all(cfg, clients, clients_data, task_id, R)
        if verbose:
            row = "  ".join(f"T{j}={R[task_id, j]:.3f}" for j in range(task_id + 1))
            print(f"  [оценка после задачи {task_id}] {row}")

    res = summarize(R)
    res["comm"] = agg.stats.summary()
    res["backend"] = agg.backend_name()
    res["server_has_secret_key"] = agg.server_has_secret_key() if cfg.crypto_mode == "ckks" else None
    if cfg.crypto_mode == "ckks" and agg.real_ckks:
        res["drift"] = agg.drift_summary()
    res["norm_log"] = norm_log
    if getattr(cfg, "dpsgd", None) is not None and cfg.dpsgd.z > 0:
        res["privacy"] = privacy_report(cfg, clients)
    if rec:
        res["observations"] = {
            "slot": np.array(obs_slot),          # [раунды, клиенты, M]
            "agg_slot": np.array(obs_agg),       # [раунды, M] — видно при SecAgg
            "task": np.array(obs_task),          # номер задачи в каждом раунде
            "cipher_hist": (np.array(agg.observed) if agg.observed else None),
            "insider": (np.array(obs_ins) if obs_ins else None),  # [раунды, клиники, M]
            # сильный атакующий: проекция ПОЛНОГО обновления (256 чисел)
            "p_slot": np.array(obs_pslot),
            "p_agg": np.array(obs_pagg),
            "p_insider": (np.array(obs_pins) if obs_pins else None),
            "p_cipher": (np.array(agg.observed_proj) if getattr(agg, "observed_proj", None) else None),
        }
    res["config"] = {
        "seed": cfg.seed, "n_tasks": cfg.n_tasks, "n_clients": len(clients),
        "crypto_mode": cfg.crypto_mode, "defense_mode": cfg.defense.mode,
        "protocol": cfg.protocol, "pool_floats": cfg.pool.total_floats,
        "aggregation": getattr(cfg, "aggregation", "uniform"),
        "rounds_per_task": cfg.train.rounds_per_task,
    }
    if return_clients:
        res["clients"] = clients
    return res
