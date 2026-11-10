#!/usr/bin/env python3
"""
run_experiment.py — запуск экспериментов.

Сценарии:
  single      — один прогон с заданными параметрами (удобно для отладки)
  crypto      — ГЛАВНЫЙ ЭКСПЕРИМЕНТ: открытая агрегация против CKKS.
                Влияет ли приближённая природа CKKS на точность и забывание
                при последовательном обучении нескольких задач?
  sensitivity — анализ чувствительности: реальная погрешность CKKS
                усиливается в k раз; при каком k эффект становится заметен?
                (доказывает, что нулевая разница в crypto — свойство схемы,
                а не слепота измерения)
  attack      — атака определения членства в духе PromptMIA при разных
                уровнях наблюдения сервера (что видит сервер) и режимах
                нормализации. Отвечает на вопрос, ЧТО на самом деле
                защищает от атаки.
  defense     — влияние нормализации обновлений на точность и забывание
                (+ служебная диагностика норм; это НЕ мера утечки)
  external    — воспроизведение выводов crypto/defense на НЕЗАВИСИМОЙ
                федерации (другие датасеты; модель обучается заново).
                Это воспроизводимость, а не «внешняя валидация» в
                клиническом смысле.

Примеры:
    python run_experiment.py --scenario crypto --seeds 1 2 3 4 5
    python run_experiment.py --scenario crypto --protocol cil --seeds 1 2 3
    python run_experiment.py --scenario attack --seeds 1 2 3 --targets 400
    python run_experiment.py --scenario sensitivity --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import ExperimentConfig, TrainConfig, DefenseConfig, DPSGDConfig
from core.data import build_federation
from core.federation import run_federated_continual
from core.attack import run_attack, OBSERVATIONS, DEFENSES


# ============================================================
#  Общие функции
# ============================================================
def make_cfg(args, seed: int, crypto: str, defense_mode: str) -> ExperimentConfig:
    cfg = ExperimentConfig(
        seed=seed,
        n_clients=args.clients,
        n_tasks=args.tasks,
        crypto_mode=crypto,
        protocol=args.protocol,
        max_train_per_client=args.max_train,
        max_test_per_client=args.max_test,
        features_path=args.features,
        out_dir=args.out,
    )
    cfg.train = TrainConfig(
        lr=args.lr,
        rounds_per_task=args.rounds,
        batches_per_round=args.batches,
        batch_size=args.batch,
    )
    cfg.defense = DefenseConfig(mode=defense_mode, C=args.C, C_K=args.C, C_P=args.C,
                                sigma=getattr(args, "sigma", 0.0))
    cfg.aggregation = getattr(args, "agg", "uniform")
    cfg.dpsgd = DPSGDConfig(z=getattr(args, "dpsgd_z", 0.0),
                            clip=getattr(args, "dpsgd_clip", 1.0),
                            delta=getattr(args, "dpsgd_delta", 1e-5),
                            clip_mode=getattr(args, "dpsgd_clip_mode", "global"),
                            clip_head=getattr(args, "dpsgd_clip_head", 1.0),
                            clip_prompt=getattr(args, "dpsgd_clip_prompt", 0.03),
                            clip_key=getattr(args, "dpsgd_clip_key", 1.0))
    return cfg


def one_run(cfg, verbose=True) -> dict:
    clients_data = build_federation(cfg, verbose=verbose)
    return run_federated_continual(cfg, clients_data, verbose=verbose)


def print_R(R):
    R = np.array(R)
    T = R.shape[0]
    print("\n    матрица R (строка = после задачи i, столбец = точность на задаче j)")
    print("         " + "".join(f"  задача{j}" for j in range(T)))
    for i in range(T):
        print(f"    после{i} " + "".join(f"   {R[i, j]:.3f}" for j in range(T)))


def summarize_runs(rs):
    faa = np.array([r["FAA"] for r in rs])
    fg = np.array([r["Forgetting"] for r in rs])
    bwt = np.array([r["BWT"] for r in rs])
    kb = float(np.mean([r["comm"]["kb_per_round_total"] for r in rs]))
    return {
        "FAA_mean": float(faa.mean()), "FAA_std": float(faa.std()),
        "Forgetting_mean": float(fg.mean()), "Forgetting_std": float(fg.std()),
        "BWT_mean": float(bwt.mean()), "BWT_std": float(bwt.std()),
        "kb_per_round": kb, "n_seeds": len(rs),
    }


def save(args, tag, results, summary=None):
    os.makedirs(args.out, exist_ok=True)
    slim = [{k: v for k, v in r.items() if k not in ("norm_log", "observations")}
            for r in results]
    path = os.path.join(args.out, f"{tag}_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": slim, "summary": summary,
                   "args": vars(args)}, f, indent=2, ensure_ascii=False)
    print(f"\n  результаты сохранены: {path}")


# ============================================================
#  Сценарии
# ============================================================
def scenario_single(args):
    cfg = make_cfg(args, args.seeds[0], args.crypto, args.defense)
    res = one_run(cfg)
    print_R(res["R"])
    print(f"\n  FAA         = {res['FAA']:.4f}")
    print(f"  Forgetting  = {res['Forgetting']:.4f}")
    print(f"  BWT         = {res['BWT']:.4f}")
    print(f"  protocol    = {cfg.protocol}")
    print(f"  backend     = {res['backend']}")
    if res.get("server_has_secret_key") is not None:
        print(f"  секретный ключ у сервера: {res['server_has_secret_key']}")
    c = res["comm"]
    print(f"  трафик      = {c['kb_per_round_total']:.1f} КБ/раунд, "
          f"всего {c['mb_total']:.2f} МБ")
    save(args, "single", [res])


def scenario_crypto(args):
    print("=" * 78)
    print(" ЭКСПЕРИМЕНТ: влияние приближённой природы CKKS на точность и забывание")
    print(f" протокол = {args.protocol}, задач = {args.tasks}")
    print("=" * 78)
    all_res = []
    for mode in ["plaintext", "ckks"]:
        for seed in args.seeds:
            cfg = make_cfg(args, seed, mode, args.defense)
            print(f"\n--- {mode}, seed={seed} ---")
            res = one_run(cfg, verbose=args.verbose)
            res["mode"], res["seed"] = mode, seed
            all_res.append(res)
            extra = ""
            if "drift" in res:
                extra = f"  погрешность агрегата ≤ {res['drift']['drift_abs_max']:.1e}"
            print(f"    FAA={res['FAA']:.4f}  Forgetting={res['Forgetting']:.4f}  "
                  f"BWT={res['BWT']:.4f}  ({res['backend']}){extra}")

    print("\n" + "=" * 78 + "\n СВОДКА\n" + "=" * 78)
    print(f"{'режим':<12} {'FAA':>16} {'Forgetting':>16} {'BWT':>16} {'КБ/раунд':>12}")
    summary = {}
    for mode in ["plaintext", "ckks"]:
        s = summarize_runs([r for r in all_res if r["mode"] == mode])
        summary[mode] = s
        print(f"{mode:<12} {s['FAA_mean']:>8.4f}±{s['FAA_std']:.4f} "
              f"{s['Forgetting_mean']:>8.4f}±{s['Forgetting_std']:.4f} "
              f"{s['BWT_mean']:>8.4f}±{s['BWT_std']:.4f} {s['kb_per_round']:>12.1f}")

    # попарное сравнение по сидам
    same = 0
    for seed in args.seeds:
        p = next(r for r in all_res if r["mode"] == "plaintext" and r["seed"] == seed)
        c = next(r for r in all_res if r["mode"] == "ckks" and r["seed"] == seed)
        same += int(abs(p["FAA"] - c["FAA"]) < 1e-9 and
                    abs(p["Forgetting"] - c["Forgetting"]) < 1e-9)
    d_faa = summary["ckks"]["FAA_mean"] - summary["plaintext"]["FAA_mean"]
    d_fg = summary["ckks"]["Forgetting_mean"] - summary["plaintext"]["Forgetting_mean"]
    print(f"\n  разница CKKS - plaintext:  ΔFAA = {d_faa:+.4f},  ΔForgetting = {d_fg:+.4f}")
    print(f"  совпадение по сидам: {same} из {len(args.seeds)}")
    print("  ВЫВОД: " + ("погрешность CKKS не влияет на точность и забывание."
                         if abs(d_fg) < 0.01 and abs(d_faa) < 0.01 else
                         "погрешность CKKS ЗАМЕТНО влияет — это отдельный результат."))
    print("  Замечание: при 40-битных внутренних модулях это ожидаемо;"
          " запас прочности покажет сценарий sensitivity.")
    summary["paired_identical"] = same
    save(args, "crypto", all_res, summary)


def scenario_sensitivity(args):
    print("=" * 78)
    print(" АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ: при какой погрешности CKKS ломается обучение")
    print("=" * 78)
    base = []
    for seed in args.seeds:
        r = one_run(make_cfg(args, seed, "plaintext", args.defense), verbose=args.verbose)
        base.append(r)
    b = summarize_runs(base)
    print(f"\n  эталон (открытый текст): FAA={b['FAA_mean']:.4f}  "
          f"Забывание={b['Forgetting_mean']:.4f}")

    rows = []
    for k in args.amplify:
        rs = []
        for seed in args.seeds:
            cfg = make_cfg(args, seed, "ckks", args.defense)
            cfg.ckks.error_amplify = float(k)
            rs.append(one_run(cfg, verbose=args.verbose))
        s = summarize_runs(rs)
        drift = float(np.mean([r["drift"]["drift_l2_per_round"] for r in rs if "drift" in r]))
        row = {"amplify": float(k), **s,
               "dFAA": s["FAA_mean"] - b["FAA_mean"],
               "dForgetting": s["Forgetting_mean"] - b["Forgetting_mean"],
               "raw_ckks_error_l2_per_round": drift}
        rows.append(row)
        print(f"  k={k:<10.0f} FAA={s['FAA_mean']:.4f}  Забывание={s['Forgetting_mean']:.4f}  "
              f"ΔFAA={row['dFAA']:+.4f}  ΔЗабывание={row['dForgetting']:+.4f}")

    detect = next((r["amplify"] for r in rows if r["amplify"] > 1 and
                   (abs(r["dFAA"]) > 1e-4 or abs(r["dForgetting"]) > 1e-4)), None)
    degrade = next((r["amplify"] for r in rows if abs(r["dFAA"]) > 0.01), None)
    print("\n" + "-" * 78)
    print(f"  реальная погрешность CKKS (L2 на раунд): "
          f"{rows[0]['raw_ckks_error_l2_per_round']:.2e}")
    print(f"  эффект становится различим при k = {detect}" if detect
          else "  эффект не обнаружен во всём диапазоне")
    print(f"  падение точности > 0.01 при k = {degrade}" if degrade
          else f"  падение точности > 0.01 не достигнуто до k = {args.amplify[-1]:.0f}")
    save(args, "sensitivity", rows, {"baseline": b, "detect_at": detect,
                                      "degrade_at": degrade})


def scenario_attack(args):
    print("=" * 78)
    print(" АТАКА ОПРЕДЕЛЕНИЯ ЧЛЕНСТВА (в духе PromptMIA): что на самом деле защищает")
    print("=" * 78)
    per_seed = []
    for seed in args.seeds:
        cfg = make_cfg(args, seed, "plaintext", "none")
        cd = build_federation(cfg, verbose=False)
        print(f"\n--- seed={seed}: целей {args.targets}, пакет {args.attack_batch}, "
              f"прогрев {args.warmup} раундов ---")
        r = run_attack(cfg, cd, n_targets=args.targets, attack_batch=args.attack_batch,
                       warmup_rounds=args.warmup, verbose=args.verbose)
        print(f"    ловушки в ячейках {r['trap_slots']}; цель выбирает все ловушки: "
              f"{r['target_selects_traps']:.2f}; чужие снимки задевают ловушку: "
              f"{r['false_trap_rate']:.2f}")
        per_seed.append(r)

    names = {
        "slot_plain": "обновление по ячейкам (FL без шифрования)",
        "slot_keys_only": "только ключи (подсказки зашифрованы)",
        "aggregate_slot": "сумма обновлений (безопасная агрегация)",
        "total_norm": "только суммарная норма обновления",
        "ciphertext": "только шифртексты CKKS (средний байт)",
        "ciphertext_len": "только длина шифртекста CKKS",
    }
    print("\n" + "=" * 78)
    print(" СВОДКА: AUC (среднее по сидам); 0.5 = угадывание монеткой")
    print("=" * 78)
    print(f"{'что видит сервер':<46}" + "".join(f"{d:>14}" for d in DEFENSES))
    summary = {}
    for o in OBSERVATIONS:
        cells = []
        for d in DEFENSES:
            key = f"{o}|{d}"
            vals = [r["table"][key] for r in per_seed if key in r["table"]]
            if not vals:
                cells.append("—")
                continue
            auc = np.array([v["AUC"] for v in vals])
            adv = np.array([v["Adv"] for v in vals])
            summary[key] = {"AUC_mean": float(auc.mean()), "AUC_std": float(auc.std()),
                            "Adv_mean": float(adv.mean()), "Adv_std": float(adv.std()),
                            "n_seeds": len(vals)}
            cells.append(f"{auc.mean():.3f}±{auc.std():.3f}")
        print(f"{names[o]:<46}" + "".join(f"{c:>14}" for c in cells))
    print("\n  Как читать: если в строке значения одинаковы во всех столбцах —")
    print("  нормализация на этот канал не влияет. Защищает тот уровень наблюдения,")
    print("  где AUC ≈ 0.5 при любом режиме нормализации.")
    save(args, "attack", per_seed, summary)


def scenario_defense(args):
    print("=" * 78)
    print(" НОРМАЛИЗАЦИЯ ОБНОВЛЕНИЙ: влияние на точность и забывание")
    print("=" * 78)
    all_res = []
    for mode in DEFENSES:
        for seed in args.seeds:
            cfg = make_cfg(args, seed, args.crypto, mode)
            print(f"\n--- защита={mode}, seed={seed} ---")
            res = one_run(cfg, verbose=args.verbose)
            res["defense"], res["seed"] = mode, seed
            sums = np.array([n["sum_norm"] for n in res["norm_log"]])
            res["sum_norm_mean"], res["sum_norm_std"] = float(sums.mean()), float(sums.std())
            all_res.append(res)
            print(f"    FAA={res['FAA']:.4f}  Forgetting={res['Forgetting']:.4f}  "
                  f"норма: {sums.mean():.4f}±{sums.std():.4f}  "
                  f"[{res['backend']}, {res['comm']['kb_per_round_total']:.0f} КБ/раунд]")

    print("\n" + "=" * 78 + "\n СВОДКА\n" + "=" * 78)
    print(f"{'защита':<16} {'FAA':>16} {'Forgetting':>16} {'СКО нормы':>14}")
    summary = {}
    for mode in DEFENSES:
        rs = [r for r in all_res if r["defense"] == mode]
        s = summarize_runs(rs)
        s["sum_norm_std"] = float(np.mean([r["sum_norm_std"] for r in rs]))
        summary[mode] = s
        print(f"{mode:<16} {s['FAA_mean']:>8.4f}±{s['FAA_std']:.4f} "
              f"{s['Forgetting_mean']:>8.4f}±{s['Forgetting_std']:.4f} "
              f"{s['sum_norm_std']:>14.6f}")
    print("\n  СКО нормы — служебная диагностика: при покомпонентной нормализации")
    print("  оно равно нулю ПО ПОСТРОЕНИЮ. Это не мера утечки приватности —")
    print("  утечку измеряет сценарий attack.")
    save(args, "defense", all_res, summary)


def scenario_external(args):
    print("=" * 78)
    print(" ВОСПРОИЗВОДИМОСТЬ НА НЕЗАВИСИМОЙ ФЕДЕРАЦИИ (модель обучается заново)")
    print("=" * 78)
    feds = {"primary": args.features, "external": args.features_external}
    for tag, path in feds.items():
        if not os.path.exists(path):
            raise SystemExit(
                f"\nНе найден файл признаков для федерации '{tag}': {path}\n"
                f"Выполните: python scripts/prepare_features.py --federation {tag} --mode real")
    all_res = []
    for fed_tag, path in feds.items():
        print(f"\n{'#' * 78}\n# ФЕДЕРАЦИЯ: {fed_tag}\n{'#' * 78}")
        for crypto in ["plaintext", "ckks"]:
            for seed in args.seeds:
                cfg = make_cfg(args, seed, crypto, "none")
                cfg.features_path = path
                print(f"\n--- {fed_tag} | {crypto} | seed={seed} ---")
                res = one_run(cfg, verbose=args.verbose)
                res.update(federation=fed_tag, mode=crypto, seed=seed)
                all_res.append(res)
                print(f"    FAA={res['FAA']:.4f}  Forgetting={res['Forgetting']:.4f}  "
                      f"клиентов={res['config']['n_clients']}")

    print("\n" + "=" * 78 + "\n СВОДКА ПО ФЕДЕРАЦИЯМ\n" + "=" * 78)
    summary, ok = {}, True
    for fed_tag in feds:
        for crypto in ["plaintext", "ckks"]:
            rs = [r for r in all_res if r["federation"] == fed_tag and r["mode"] == crypto]
            summary[f"{fed_tag}|{crypto}"] = summarize_runs(rs)
            s = summary[f"{fed_tag}|{crypto}"]
            print(f"{fed_tag:<12} {crypto:<11} FAA={s['FAA_mean']:.4f}±{s['FAA_std']:.4f}  "
                  f"Забывание={s['Forgetting_mean']:.4f}")
        d = (summary[f"{fed_tag}|ckks"]["Forgetting_mean"] -
             summary[f"{fed_tag}|plaintext"]["Forgetting_mean"])
        v = abs(d) < 0.01
        ok = ok and v
        print(f"  {fed_tag}: CKKS не усиливает забывание (Δ = {d:+.5f}): "
              f"{'ВОСПРОИЗВОДИТСЯ' if v else 'НЕ воспроизводится'}")
    print(f"\n  ИТОГ: вывод {'воспроизводится' if ok else 'НЕ воспроизводится'} "
          f"на обеих федерациях.")
    save(args, "external", all_res, summary)


# ============================================================
#  ПРОДОЛЬНАЯ УТЕЧКА (новый пробел, специфичный для FCL)
# ============================================================
def scenario_temporal(args):
    """
    Продольная утечка. Наблюдатели:
      сервер — видит то, что приходит по сети (открытые обновления, сумму
               при безопасной агрегации или только шифртексты при CKKS);
      клиника-участник — после расшифровки знает сумму обновлений остальных.
    Атакующие:
      слабый  — профиль обновления по ячейкам пула (10 чисел);
      сильный — всё обновление (случайная проекция 11 520 -> 256 чисел,
                сохраняющая углы); для шифртекстов — проекция сырых байтов.
    Для каждого варианта: превышение над уровнем угадывания с 95%-м
    доверительным интервалом и p-значение перестановочного теста.
    """
    from core.leakage import evaluate_sequences
    print("=" * 78)
    print(" ПРОДОЛЬНАЯ УТЕЧКА: что можно узнать из последовательности обновлений")
    if getattr(args, "control", False):
        print(" КОНТРОЛЬ: все фазы обучаются на ОДНОЙ задаче — утечки задачи нет.")
        print(" Превышение здесь = атакующий ловит стадию обучения (дрейф), а не задачу.")
        print(" Сравнение с основным прогоном: python compare_temporal.py")
    print("=" * 78)

    rows = {}

    def add(name, weak, strong, tasks, run):
        r = rows.setdefault(name, {"weak": [], "strong": [], "tasks": [], "runs": []})
        r["weak"].extend(weak); r["strong"].extend(strong)
        r["tasks"].extend(tasks); r["runs"].append(run)

    def per_client(a):
        return [a[:, c, :] for c in range(a.shape[1])]

    def run(seed, crypto, dmode, sigma):
        args.sigma = sigma
        cfg = make_cfg(args, seed, crypto, dmode)
        cfg.record_observations = True
        cfg.control_same_task = bool(getattr(args, "control", False))
        r = one_run(cfg, verbose=args.verbose)
        args.sigma = 0.0
        return r, r.pop("observations")

    for seed in args.seeds:
        r, o = run(seed, "plaintext", "none", 0.0)
        n = o["slot"].shape[1]
        add("открытый FL — сервер (обновления клиник)",
            per_client(o["slot"]), per_client(o["p_slot"]), [o["task"]] * n, r)
        add("безопасная агрегация — сервер (видна сумма)",
            [o["agg_slot"]], [o["p_agg"]], [o["task"]], r)
        add("открытый FL — клиника-участник",
            per_client(o["insider"]), per_client(o["p_insider"]), [o["task"]] * n, r)
        print(f"  seed={seed}  {'открытый':<28} FAA={r['FAA']:.4f}  Забывание={r['Forgetting']:.4f}")
        variants = [("CKKS", "none", 0.0, True),
                    ("CKKS + отсечение", "dp", 0.0, False)]
        variants += [(f"CKKS + шум σ={sg:g} (гибрид)", "dp", sg,
                      sg == args.insider_sigmas[-1]) for sg in args.insider_sigmas]
        for label, dmode, sg, server_row in variants:
            r, o = run(seed, "ckks", dmode, sg)
            if server_row and o["cipher_hist"] is not None:
                k = o["cipher_hist"].shape[1]
                add(f"{label} — сервер (только шифртексты)",
                    per_client(o["cipher_hist"]), per_client(o["p_cipher"]),
                    [o["task"]] * k, r)
            add(f"{label} — клиника-участник",
                per_client(o["insider"]), per_client(o["p_insider"]),
                [o["task"]] * o["insider"].shape[1], r)
            print(f"  seed={seed}  {label:<28} FAA={r['FAA']:.4f}  "
                  f"Забывание={r['Forgetting']:.4f}  [{r['backend']}]")

        for sg in args.sigmas:
            r, o = run(seed, "plaintext", "dp", sg)
            add(f"шум DP σ={sg:g} без шифрования — сервер",
                per_client(o["slot"]), per_client(o["p_slot"]),
                [o["task"]] * o["slot"].shape[1], r)
            print(f"  seed={seed}  {'DP σ=' + format(sg, 'g'):<28} FAA={r['FAA']:.4f}  "
                  f"Забывание={r['Forgetting']:.4f}")

    # --- сохранить наблюдения для повторного анализа без обучения ---
    os.makedirs(args.out, exist_ok=True)
    obs_path = os.path.join(args.out, "temporal_observations.npz")
    np.savez_compressed(obs_path, data=np.array(
        {k: {"weak": v["weak"], "strong": v["strong"], "tasks": v["tasks"],
             "FAA": [x["FAA"] for x in v["runs"]],
             "Forgetting": [x["Forgetting"] for x in v["runs"]]}
         for k, v in rows.items()}, dtype=object))

    summary = summarize_temporal(rows, args.n_perm, getattr(args, "smooth", 0))
    save(args, "temporal", [x for d in rows.values() for x in d["runs"]], summary)
    print(f"  наблюдения сохранены: {obs_path}")
    print("  пересчёт статистики без обучения: "
          f"python reanalyze_temporal.py --file {obs_path}")


def summarize_temporal(rows, n_perm, smooth=0):
    from core.leakage import evaluate_sequences
    print("\n" + "=" * 78)
    print(" СВОДКА: утечка (превышение над угадыванием, 95% ДИ, p) и цена защиты")
    if smooth and smooth > 1:
        print(f" Атакующие УСРЕДНЯЮТ наблюдения по {smooth} соседним раундам")
    print("=" * 78)
    hdr = (f"{'кто и что наблюдает':<58}  {'слабый атакующий':<30}"
           f"{'сильный атакующий':<30}{'FAA':>7}{'Забыв.':>8}")
    print(hdr)
    summary = {}

    def fmt(m):
        star = "*" if m["p_value"] < 0.05 else " "
        return (f"{m['excess']:+.3f} [{m['excess_ci_lo']:+.2f};{m['excess_ci_hi']:+.2f}]"
                f" p={m['p_value']:.3f}{star}")

    for name, d in rows.items():
        mw = evaluate_sequences(d["weak"], d["tasks"], seed=0, n_perm=n_perm, smooth=smooth)
        ms = evaluate_sequences(d["strong"], d["tasks"], seed=1, n_perm=n_perm, smooth=smooth)
        faa = d["FAA"] if "FAA" in d else [x["FAA"] for x in d["runs"]]
        fg = d["Forgetting"] if "Forgetting" in d else [x["Forgetting"] for x in d["runs"]]
        summary[name] = {"weak": mw, "strong": ms,
                         "FAA": float(np.mean(faa)), "Forgetting": float(np.mean(fg))}
        print(f"{name:<58}  {fmt(mw):<30}{fmt(ms):<30}"
              f"{np.mean(faa):>7.4f}{np.mean(fg):>8.4f}")
    print("\n  превышение — насколько наблюдатель восстанавливает текущую группу диагнозов")
    print("  клиники лучше случайного (0 = утечки нет); в скобках 95% доверительный")
    print("  интервал; p — перестановочный тест; * — утечка статистически значима (p<0.05).")
    print("  слабый атакующий — профиль по ячейкам (10 чисел); сильный — всё обновление.")
    print("  Клиника-участник знает сумму обновлений остальных (n*среднее − своё).")
    print("  Для формальной гарантии (eps=1, delta=1e-5) на ОДИН раунд нужен σ ≈ 4.84.")
    return summary


# ============================================================
#  АГРЕГАЦИЯ ПУЛА С УЧЁТОМ ИСПОЛЬЗОВАНИЯ ЯЧЕЕК
# ============================================================
def scenario_aggregation(args):
    """
    Сравнение правил агрегации пула:
      uniform — обычное среднее (FedAvg);
      usage   — вклад клиники в ячейку взвешен числом её снимков,
                выбравших ячейку. Совместимо с CKKS без нелинейных операций.
    Каждое правило — без отсечения и с отсечением нормы (C).
    Отдельно: usage под настоящим CKKS — совпадение чисел и трафик.
    """
    print("=" * 78)
    print(" АГРЕГАЦИЯ ПУЛА: обычное среднее против взвешивания по использованию")
    print("=" * 78)
    saved_agg, saved_sigma = args.agg, getattr(args, "sigma", 0.0)
    args.sigma = 0.0
    groups = {}
    for agg in ["uniform", "usage"]:
        args.agg = agg
        for dmode, dname in [("none", "без отсечения"), ("dp", "с отсечением")]:
            name = f"{agg:<8} {dname}"
            for seed in args.seeds:
                cfg = make_cfg(args, seed, "plaintext", dmode)
                r = one_run(cfg, verbose=args.verbose)
                groups.setdefault(name, []).append(r)
                print(f"  {name:<24} seed={seed}  FAA={r['FAA']:.4f}  "
                      f"Забывание={r['Forgetting']:.4f}")
    # usage под настоящим CKKS
    args.agg = "usage"
    for seed in args.seeds:
        cfg = make_cfg(args, seed, "ckks", "none")
        r = one_run(cfg, verbose=args.verbose)
        groups.setdefault("usage    без отсечения, CKKS", []).append(r)
        print(f"  {'usage CKKS':<24} seed={seed}  FAA={r['FAA']:.4f}  "
              f"Забывание={r['Forgetting']:.4f}  [{r['backend']}]")
    args.agg, args.sigma = saved_agg, saved_sigma

    print("\n" + "=" * 78)
    print(" СВОДКА")
    print("=" * 78)
    print(f"{'правило':<32}{'FAA':>18}{'Забывание':>18}{'КБ/раунд':>11}")
    summary = {}
    for name, rs in groups.items():
        sm = summarize_runs(rs)
        summary[name] = sm
        print(f"{name:<32}{sm['FAA_mean']:>10.4f}±{sm['FAA_std']:.4f}"
              f"{sm['Forgetting_mean']:>10.4f}±{sm['Forgetting_std']:.4f}"
              f"{sm['kb_per_round']:>11.1f}")
    # попарная разница usage - uniform по сидам
    for dname in ["без отсечения", "с отсечением"]:
        a = groups[f"uniform  {dname}"]; b = groups[f"usage    {dname}"]
        dF = [y["FAA"] - x["FAA"] for x, y in zip(a, b)]
        dG = [y["Forgetting"] - x["Forgetting"] for x, y in zip(a, b)]
        wins = sum(1 for d in dG if d < 0)
        print(f"\n  {dname}: usage − uniform  ΔFAA={np.mean(dF):+.4f}  "
              f"ΔЗабывание={np.mean(dG):+.4f}  (забывание ниже в {wins} из {len(dG)} сидов)")
    same = all(abs(x["FAA"] - y["FAA"]) < 1e-9 and abs(x["Forgetting"] - y["Forgetting"]) < 1e-9
               for x, y in zip(groups["usage    без отсечения"], groups["usage    без отсечения, CKKS"]))
    print(f"  usage: CKKS совпадает с открытым режимом во всех сидах: {'да' if same else 'нет'}")
    save(args, "aggregation", [r for rs in groups.values() for r in rs], summary)


# ============================================================
#  ФОРМАЛЬНАЯ ГАРАНТИЯ ДЛЯ ПАЦИЕНТА (DP-SGD)
# ============================================================
def scenario_dpsgd(args):
    """
    Паритет с работами, дающими формальную гарантию (MedHE, DP-FPL):
    DP-SGD над пулом и головой, eps считается для каждого пациента.
    Снимок участвует только в шагах своей задачи, поэтому eps пациента —
    это eps шагов его задачи (параллельная композиция, известное свойство).
    """
    from core.dp_accountant import epsilon, noise_for_epsilon
    print("=" * 78)
    print(" ФОРМАЛЬНАЯ ГАРАНТИЯ ДЛЯ ПАЦИЕНТА: DP-SGD (+ CKKS)")
    print("=" * 78)

    # оценка нужного шума до обучения
    cfg0 = make_cfg(args, args.seeds[0], "plaintext", "none")
    cd0 = build_federation(cfg0, verbose=False)
    n_min = min(len(t.X_train) for cd in cd0 for t in cd.tasks if len(t.X_train) > 0)
    q = min(1.0, args.batch / n_min)
    steps = args.rounds * args.batches
    print(f"  наименьшая задача: {n_min} снимков, q = {q:.4f}, шагов на задачу = {steps},"
          f" delta = {args.dpsgd_delta:g}")
    # множитель шума под каждую целевую гарантию (по наименьшей задаче —
    # худший случай; фактический eps каждой задачи затем считается точно)
    targets = list(args.dpsgd_eps)
    zs = [noise_for_epsilon(q, steps, args.dpsgd_delta, e) for e in targets]
    for e, z in zip(targets, zs):
        print(f"    цель eps <= {e:<4g} -> множитель шума z = {z:.2f}")
    print(f"  скорость обучения: {args.lr:g} (для DP рекомендуется 0.01, подобрана на синтетике)")

    groups = {}

    def run(label, crypto, z):
        args.dpsgd_z = z
        for seed in args.seeds:
            cfg = make_cfg(args, seed, crypto, "none")
            r = one_run(cfg, verbose=args.verbose)
            groups.setdefault(label, []).append(r)
            eps = r.get("privacy", {}).get("eps_per_patient", float("inf"))
            eps_s = "    нет" if eps > 1e6 else f"{eps:>7.2f}"
            print(f"  {label:<26} seed={seed}  eps={eps_s}  FAA={r['FAA']:.4f}  "
                  f"Забывание={r['Forgetting']:.4f}  [{r['backend']}]")
        args.dpsgd_z = 0.0

    run("без DP", "plaintext", 0.0)
    # Честная точка отсчёта: то же обучение DP-SGD (пуассоновская выборка,
    # ограничение вклада каждого снимка), но практически без шума. Разница с
    # ней — чистая цена шума; разница «без DP» -> «без шума» — эффект самого
    # ограничения вклада снимков.
    run("отсечение без шума", "plaintext", 1e-12)
    for e, z in zip(targets, zs):
        run(f"DP-SGD eps<={e:g}", "plaintext", z)
    ec, zc = targets[-1], zs[-1]
    run(f"DP-SGD eps<={ec:g} + CKKS", "ckks", zc)

    print("\n" + "=" * 78)
    print(" СВОДКА")
    print("=" * 78)
    print(f"{'вариант':<28}{'eps пациента':>14}{'eps без парал.':>16}"
          f"{'FAA':>18}{'Забывание':>18}")
    summary = {}
    for name, rs in groups.items():
        sm = summarize_runs(rs)
        pr = [r.get("privacy", {}) for r in rs]
        e1 = max((p.get("eps_per_patient", float("inf")) for p in pr), default=float("inf"))
        e2 = max((p.get("eps_if_sequential_all_tasks", float("inf")) for p in pr),
                 default=float("inf"))
        sm.update(eps_per_patient=e1, eps_if_sequential=e2)
        summary[name] = sm
        es1 = "нет" if e1 > 1e6 else f"{e1:.2f}"
        es2 = "нет" if e2 > 1e6 else f"{e2:.2f}"
        print(f"{name:<28}{es1:>14}{es2:>16}"
              f"{sm['FAA_mean']:>10.4f}±{sm['FAA_std']:.4f}"
              f"{sm['Forgetting_mean']:>10.4f}±{sm['Forgetting_std']:.4f}")
    ref = summary["отсечение без шума"]
    base = summary["без DP"]
    print("\n  Эффект ограничения вклада снимков (без шума против без DP):"
          f"  ΔFAA = {ref['FAA_mean'] - base['FAA_mean']:+.4f},"
          f"  ΔЗабывание = {ref['Forgetting_mean'] - base['Forgetting_mean']:+.4f}")
    print("  Чистая цена шума (DP-SGD против отсечения без шума):")
    for e in targets:
        sm = summary[f"DP-SGD eps<={e:g}"]
        print(f"    eps <= {e:<4g}  ΔFAA = {sm['FAA_mean'] - ref['FAA_mean']:+.4f}  "
              f"ΔЗабывание = {sm['Forgetting_mean'] - ref['Forgetting_mean']:+.4f}")
    a, b = groups[f"DP-SGD eps<={ec:g}"], groups[f"DP-SGD eps<={ec:g} + CKKS"]
    same = all(abs(x["FAA"] - y["FAA"]) < 1e-9 for x, y in zip(a, b))
    print(f"\n  DP-SGD под CKKS совпадает с открытым режимом во всех сидах: {'да' if same else 'нет'}")
    print("  eps пациента — гарантия (eps, delta)-DP для отдельного снимка: снимок")
    print("  участвует только в шагах своей задачи (параллельная композиция).")
    print("  eps без парал. — если бы снимок участвовал во всех шагах всех задач.")
    print("  Гарантия защищает пациентов; расписание задач клиники — отдельный")
    print("  вопрос (сценарий temporal).")
    save(args, "dpsgd", [r for rs in groups.values() for r in rs], summary)


# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Федеративное непрерывное обучение с гомоморфным шифрованием")
    ap.add_argument("--scenario", default="crypto",
                    choices=["single", "crypto", "sensitivity", "attack",
                             "defense", "external", "temporal", "aggregation", "dpsgd"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--protocol", choices=["til", "cil"], default="til",
                    help="til — номер задачи известен (как в прежних прогонах); "
                         "cil — номер задачи неизвестен (режим L2P)")
    ap.add_argument("--rounds", type=int, default=15,
                    help="раундов федеративного обучения на одну задачу")
    ap.add_argument("--batches", type=int, default=10,
                    help="локальных шагов (пакетов) на раунд")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crypto", choices=["plaintext", "ckks"], default="plaintext")
    ap.add_argument("--defense", choices=list(DEFENSES) + ["dp"], default="none")
    ap.add_argument("--dpsgd_z", type=float, default=0.0,
                    help="DP-SGD: множитель шума (0 = выключено)")
    ap.add_argument("--dpsgd_clip", type=float, default=1.0,
                    help="DP-SGD: порог отсечения градиента одного снимка")
    ap.add_argument("--dpsgd_delta", type=float, default=1e-5)
    ap.add_argument("--dpsgd_clip_mode", choices=["global", "group"], default="global",
                    help="DP-SGD: общий порог или свои пороги для головы/подсказок/ключей")
    ap.add_argument("--dpsgd_clip_head", type=float, default=1.0)
    ap.add_argument("--dpsgd_clip_prompt", type=float, default=0.03)
    ap.add_argument("--dpsgd_clip_key", type=float, default=1.0)
    ap.add_argument("--dpsgd_eps", type=float, nargs="+", default=[8.0, 4.0, 2.0, 1.0],
                    help="сценарий dpsgd: целевые гарантии eps (шум подбирается сам)")
    ap.add_argument("--control", action="store_true",
                    help="temporal: контроль — все фазы на одной задаче (только cil)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="temporal: атакующий усредняет наблюдения по N соседним раундам")
    ap.add_argument("--n_perm", type=int, default=30,
                    help="перестановок на последовательность для теста утечки (temporal)")
    ap.add_argument("--insider_sigmas", type=float, nargs="+", default=[0.03, 0.1],
                    help="шум гибрида CKKS + шум (защита от клиники-участника)")
    ap.add_argument("--agg", choices=["uniform", "usage"], default="uniform",
                    help="правило агрегации пула (usage — взвешивание по использованию ячеек)")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="множитель шума для --defense dp")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 5.0],
                    help="уровни шума DP для сценария temporal (0 = только отсечение)")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--max_train", type=int, default=2000, help="на одну задачу")
    ap.add_argument("--max_test", type=int, default=500, help="на одну задачу")
    ap.add_argument("--features", default="data/features_primary.npz")
    ap.add_argument("--features_external", default="data/features_external.npz")
    ap.add_argument("--amplify", type=float, nargs="+",
                    default=[1, 10, 100, 1000, 10000, 100000, 1000000],
                    help="коэффициенты усиления погрешности (sensitivity)")
    ap.add_argument("--targets", type=int, default=400, help="целей атаки (attack)")
    ap.add_argument("--attack_batch", type=int, default=8,
                    help="размер пакета клиники в раунде атаки")
    ap.add_argument("--warmup", type=int, default=10,
                    help="раундов прогрева перед атакой")
    ap.add_argument("--real", action="store_true", help="не используется (совместимость)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    {"single": scenario_single, "crypto": scenario_crypto,
     "sensitivity": scenario_sensitivity, "attack": scenario_attack,
     "defense": scenario_defense, "external": scenario_external,
     "temporal": scenario_temporal,
     "aggregation": scenario_aggregation,
     "dpsgd": scenario_dpsgd}[args.scenario](args)


if __name__ == "__main__":
    main()
