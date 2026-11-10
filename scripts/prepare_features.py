#!/usr/bin/env python3
"""
prepare_features.py — разовое извлечение признаков (CLS-токенов).

Два режима.

1) Настоящий (для диссертации):
       pip install torch timm medmnist
       python scripts/prepare_features.py --mode real
   Скачивает MedMNIST, прогоняет через ЗАМОРОЖЕННЫЙ ViT-Tiny (timm),
   сохраняет CLS-токены 192-D в data/features.npz.

2) Синтетический (чтобы конвейер запускался сразу, без тяжёлых зависимостей):
       python scripts/prepare_features.py --mode synthetic
   Генерирует разделимые 192-мерные признаки с той же структурой классов.
   Годится для проверки работоспособности кода, НЕ для научных выводов.

Формат выходного файла одинаков в обоих режимах.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ============================================================
#  ФЕДЕРАЦИИ
#
#  primary  — основная, на ней разрабатывался метод
#  external — ВНЕШНЯЯ ВАЛИДАЦИЯ: три другие модальности из трёх других
#             первоисточников, не использовавшихся при разработке
#  small    — облегчённая внешняя (быстрее качается)
#
#  Источники подписаны, чтобы можно было сослаться в диссертации.
# ============================================================
FEDERATIONS = {
    "primary": [
        # (имя клиента, флаг MedMNIST, классов, модальность, первоисточник)
        ("almaty_pathmnist", "pathmnist", 9,
         "гистология колоректального рака", "Kather et al."),
        ("astana_pneumoniamnist", "pneumoniamnist", 2,
         "рентген грудной клетки", "Kermany et al."),
        ("karaganda_bloodmnist", "bloodmnist", 8,
         "микроскопия клеток крови", "Acevedo et al."),
    ],
    "external": [
        ("clinic_derma", "dermamnist", 7,
         "дерматоскопия", "HAM10000, Tschandl et al."),
        ("clinic_organ", "organamnist", 11,
         "абдоминальная КТ", "LiTS, Bilic et al. / Xu et al."),
        ("clinic_tissue", "tissuemnist", 8,
         "микроскопия коры почки", "Broad Bioimage, Ljosa et al."),
    ],
    "small": [
        ("clinic_retina", "retinamnist", 5,
         "фундус-камера", "DeepDRiD Team"),
        ("clinic_breast", "breastmnist", 2,
         "УЗИ молочной железы", "Al-Dhabyani et al."),
        ("clinic_oct", "octmnist", 4,
         "оптическая когерентная томография", "Kermany et al."),
    ],
}

FEATURE_DIM = 192  # размерность CLS-токена ViT-Tiny


# ============================================================
#  Синтетический режим
# ============================================================
def make_synthetic(n_classes: int, n_train: int, n_test: int,
                   seed: int) -> tuple:
    """
    Синтетические признаки ТОЛЬКО для проверки работоспособности кода.

    Геометрия приближена к настоящим CLS-токенам ViT: они анизотропны —
    у всех снимков есть большая общая компонента (средний косинус между
    любыми снимками ~0.4), плюс компонента класса и шум. Это важно для
    сценария attack: в почти изотропной геометрии ключи пула почти не
    обучаются, и ловушки PromptMIA выбираются практически случайно.
    Точность линейного классификатора на этих данных ~0.75-0.85.

    Для научных выводов используйте --mode real.
    """
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(FEATURE_DIM)
    common /= np.linalg.norm(common)
    centers = rng.standard_normal((n_classes, FEATURE_DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    A, B, SIGMA = 3.0, 0.72, 0.25   # общая компонента, класс, шум

    def gen(n):
        y = rng.integers(0, n_classes, size=n)
        X = A * common + B * centers[y] + SIGMA * rng.standard_normal((n, FEATURE_DIM))
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X.astype(np.float32), y.astype(np.int64)

    Xtr, ytr = gen(n_train)
    Xte, yte = gen(n_test)
    return Xtr, ytr, Xte, yte


# ============================================================
#  Настоящий режим: ViT-Tiny + MedMNIST
# ============================================================
def extract_real(flag: str, n_train: int, n_test: int, batch: int = 128):
    try:
        import torch
        import timm
        import medmnist
        from medmnist import INFO
    except ImportError as e:
        print("[ОШИБКА] Для режима real нужны torch, timm, medmnist.")
        print("         pip install torch timm medmnist")
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    устройство: {device}")

    # замороженный ViT-Tiny
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                              num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    info = INFO[flag]
    DataClass = getattr(medmnist, info["python_class"])

    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3) if info["n_channels"] == 1 else T.Lambda(lambda x: x),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def run(split, limit):
        ds = DataClass(split=split, transform=tf, download=True)
        n = min(limit, len(ds))
        idx = np.random.default_rng(0).choice(len(ds), n, replace=False)
        feats, labels = [], []
        with torch.no_grad():
            for s in range(0, n, batch):
                chunk = idx[s:s + batch]
                imgs = torch.stack([ds[int(i)][0] for i in chunk]).to(device)
                f = model(imgs)                      # [B, 192] CLS-токен
                f = torch.nn.functional.normalize(f, dim=1)
                feats.append(f.cpu().numpy())
                labels.extend(int(ds[int(i)][1][0]) for i in chunk)
                print(f"      {split}: {min(s+batch, n)}/{n}", end="\r")
        print()
        return (np.concatenate(feats).astype(np.float32),
                np.array(labels, dtype=np.int64))

    Xtr, ytr = run("train", n_train)
    Xte, yte = run("test", n_test)
    return Xtr, ytr, Xte, yte


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--federation", choices=list(FEDERATIONS.keys()),
                    default="primary",
                    help="primary — основная, external — внешняя валидация")
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    clients = FEDERATIONS[args.federation]
    out_path = args.out or f"data/features_{args.federation}.npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Федерация: {args.federation}  (режим: {args.mode})")
    for _, _, _, modality, source in clients:
        print(f"    · {modality}  [{source}]")
    print()

    out = {}
    names = []
    for i, (name, flag, n_cls, modality, source) in enumerate(clients):
        print(f"[{i+1}/{len(clients)}] {name} ({flag}, классов: {n_cls})")
        if args.mode == "synthetic":
            Xtr, ytr, Xte, yte = make_synthetic(n_cls, args.n_train,
                                                args.n_test, seed=1000 + i)
        else:
            Xtr, ytr, Xte, yte = extract_real(flag, args.n_train, args.n_test)

        out[f"{name}_Xtr"] = Xtr
        out[f"{name}_ytr"] = ytr
        out[f"{name}_Xte"] = Xte
        out[f"{name}_yte"] = yte
        names.append(name)
        print(f"    train={Xtr.shape}, test={Xte.shape}, "
              f"классов={len(np.unique(ytr))}")

    out["client_names"] = np.array(names)
    out["mode"] = np.array([args.mode])
    out["federation"] = np.array([args.federation])
    np.savez_compressed(out_path, **out)
    print(f"\nСохранено: {out_path}")
    if args.mode == "synthetic":
        print("ВНИМАНИЕ: синтетические признаки годятся только для проверки "
              "работоспособности. Для диссертации запустите --mode real.")


if __name__ == "__main__":
    main()
