#!/usr/bin/env python3
"""
comm_compare.py — сколько стоит шифрование при разных подходах.

Измеряются РЕАЛЬНЫЕ размеры и время шифртекстов TenSEAL с теми же
параметрами, что в экспериментах (N=8192, модули 60-40-40-60, 128 бит).
Затем считается объём загрузки одного клиента за раунд для трёх подходов:

  pool       — шифруется пул подсказок L2P (наш подход): 11 520 чисел
  full_model — шифруется всё обновление ViT-Tiny (классическое HE-FL):
               5 524 416 параметров
  cls        — шифруются признаки КАЖДОГО снимка (как в работе
               «Privacy-Preserving Federated ViT ... Lightweight HE»):
               по одному шифртексту на снимок; дополнительно показан
               вариант с плотной упаковкой (21 вектор по 192 в шифртекст)

Все объёмы, кроме pool и full_model, растут с числом снимков клиники.
Запуск:
    python comm_compare.py
    python comm_compare.py --samples 500 1000 3000 10000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.config import ExperimentConfig

try:
    import tenseal as ts
except ImportError:
    raise SystemExit("Нужен TenSEAL:  pip install tenseal")

VIT_TINY_PARAMS = 5_524_416      # из check_setup.py (без классификационной головы)
CLS_DIM = 192


def measure(cfg, reps=20):
    c = cfg.ckks
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=c.poly_modulus_degree,
                     coeff_mod_bit_sizes=list(c.coeff_mod_bit_sizes))
    ctx.global_scale = 2 ** c.scale_bits
    pub = ctx.copy()
    pub.make_context_public()
    sctx = ts.context_from(pub.serialize())
    rng = np.random.default_rng(0)

    fresh, agg, t_enc, t_agg, t_dec = [], [], [], [], []
    for _ in range(reps):
        vs = [rng.standard_normal(c.slots) * 0.01 for _ in range(3)]
        t0 = time.perf_counter()
        blobs = [ts.ckks_vector(ctx, v.tolist()).serialize() for v in vs]
        t1 = time.perf_counter()
        acc = ts.ckks_vector_from(sctx, blobs[0])
        for b in blobs[1:]:
            acc = acc + ts.ckks_vector_from(sctx, b)
        out = (acc * (1.0 / 3)).serialize()
        t2 = time.perf_counter()
        ts.ckks_vector_from(ctx, out).decrypt()
        t3 = time.perf_counter()
        fresh.append(np.mean([len(b) for b in blobs]))
        agg.append(len(out))
        t_enc.append((t1 - t0) / 3 * 1000)
        t_agg.append((t2 - t1) * 1000)
        t_dec.append((t3 - t2) * 1000)
    return {
        "slots": c.slots,
        "fresh_ct_bytes": float(np.mean(fresh)),
        "aggregated_ct_bytes": float(np.mean(agg)),
        "encrypt_ms_per_ct": float(np.mean(t_enc)),
        "aggregate_3_ms_per_ct": float(np.mean(t_agg)),
        "decrypt_ms_per_ct": float(np.mean(t_dec)),
    }


def human(nbytes):
    for unit, div in [("ГБ", 1024 ** 3), ("МБ", 1024 ** 2), ("КБ", 1024)]:
        if nbytes >= div:
            return f"{nbytes / div:.2f} {unit}"
    return f"{nbytes:.0f} Б"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[500, 1000, 3000, 10000])
    ap.add_argument("--out", default="results/comm_compare.json")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    m = measure(cfg)
    S = m["slots"]
    pool_ct = math.ceil(cfg.pool.total_floats / S)
    full_ct = math.ceil(VIT_TINY_PARAMS / S)
    per_ct = S // CLS_DIM                       # плотная упаковка: 21 вектор

    print("=" * 78)
    print(" РЕАЛЬНЫЕ ИЗМЕРЕНИЯ TenSEAL (N=8192, 60-40-40-60, 128 бит)")
    print("=" * 78)
    print(f"  слотов в шифртексте:            {S}")
    print(f"  шифртекст после шифрования:     {human(m['fresh_ct_bytes'])}")
    print(f"  шифртекст после агрегации:      {human(m['aggregated_ct_bytes'])}")
    print(f"  шифрование одного шифртекста:   {m['encrypt_ms_per_ct']:.1f} мс")
    print(f"  расшифровка одного шифртекста:  {m['decrypt_ms_per_ct']:.1f} мс")

    rows = []

    def row(name, n_ct, note):
        up = n_ct * m["fresh_ct_bytes"]
        rows.append({"approach": name, "ciphertexts": n_ct, "upload_bytes": up,
                     "encrypt_s": n_ct * m["encrypt_ms_per_ct"] / 1000, "note": note})

    row("pool (наш подход)", pool_ct, "не зависит от числа снимков и задач")
    row("full_model (ViT-Tiny целиком)", full_ct, "не зависит от числа снимков")
    for n in args.samples:
        row(f"cls, 1 снимок = 1 шифртекст, N={n}", n, "растёт с числом снимков")
    for n in args.samples:
        row(f"cls, упаковка 21 в шифртекст, N={n}", math.ceil(n / per_ct),
            "растёт с числом снимков")

    base = rows[0]["upload_bytes"]
    print("\n" + "=" * 78)
    print(" ЗАГРУЗКА ОДНОГО КЛИЕНТА ЗА РАУНД")
    print("=" * 78)
    print(f"{'подход':<42}{'шифртекстов':>12}{'объём':>12}{'шифрование':>12}{'× пула':>9}")
    for r in rows:
        print(f"{r['approach']:<42}{r['ciphertexts']:>12}{human(r['upload_bytes']):>12}"
              f"{r['encrypt_s']:>10.1f} с{r['upload_bytes'] / base:>9.0f}")
    print("\n  Пул занимает фиксированные "
          f"{pool_ct} шифртекста при любом числе снимков и задач.")
    print("  Объём full_model посчитан по измеренному размеру одного шифртекста.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"measured": m, "rows": rows,
                   "pool_floats": cfg.pool.total_floats,
                   "vit_tiny_params": VIT_TINY_PARAMS}, f, indent=2, ensure_ascii=False)
    print(f"\n  сохранено: {args.out}")


if __name__ == "__main__":
    main()
