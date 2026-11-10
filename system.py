#!/usr/bin/env python3
"""
system.py — работа как настоящей системы: обучение, сохранение модели, диагноз по снимку.

    python system.py train                    # 3 клиники, CKKS + DP-SGD (ε ≤ 4) -> models/fcl_system.npz
    python system.py train --no-dp            # только CKKS, без формальной гарантии
    python system.py selftest                 # путь «снимок -> диагноз» = путь, на котором обучали?
    python system.py predict --clinic blood --image path/to/cell.jpg
    python system.py evaluate --clinic path --folder D:/data/crc_val_he_7k --limit 50
    python system.py internet                 # снимки из интернета (Wikimedia Commons)

Конвейер предсказания:
    снимок -> (уменьшение до 28×28, как в MedMNIST) -> замороженный ViT-Tiny -> CLS (192 числа)
           -> пул L2P (выбор 4 ячеек по ключам, подсказки) -> голова клиники -> диагноз

Обучение идёт только на числах (core/, NumPy + TenSEAL). torch/timm нужны лишь для
превращения нового снимка в признак (команды selftest/predict/evaluate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.config import ExperimentConfig, TrainConfig, DefenseConfig, DPSGDConfig
from core.data import build_federation
from core.federation import run_federated_continual
from core.dp_accountant import noise_for_epsilon
from core.pool import PromptPool

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = os.path.join(ROOT, "models", "fcl_system.npz")
RESULTS_DIR = os.path.join(ROOT, "results", "system")
# клиника -> (имя в файле признаков, флаг MedMNIST, каналов)
CLINICS = {
    "path": ("almaty_pathmnist", "pathmnist", 3),
    "pneumonia": ("astana_pneumoniamnist", "pneumoniamnist", 1),
    "blood": ("karaganda_bloodmnist", "bloodmnist", 3),
}
# синонимы имён подпапок (CRC-VAL-HE-7K, COVID-19 Radiography) -> номер класса MedMNIST
FOLDER_ALIASES = {
    "path": {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6,
             "STR": 7, "TUM": 8},
    "pneumonia": {"normal": 0, "Normal": 0, "NORMAL": 0, "pneumonia": 1, "Pneumonia": 1,
                  "PNEUMONIA": 1, "Viral Pneumonia": 1},
    "blood": {},
}


# Снимки из интернета (Wikimedia Commons, свободные лицензии) с известным диагнозом.
# файл -> (клиника, верный класс MedMNIST, что на снимке)
INTERNET_DIR = os.path.join(ROOT, "data", "internet_samples")
INTERNET = {
    "PBNeutrophil.jpg": ("blood", "neutrophil", "одиночный нейтрофил, мазок крови, 150×150"),
    "PBEosinophil.jpg": ("blood", "eosinophil", "одиночный эозинофил, мазок крови, 150×150"),
    "PBBasophil.jpg": ("blood", "basophil", "одиночный базофил, мазок крови, 150×150"),
    "Neutrophil_in_a_blood_smear.jpg": ("blood", "neutrophil", "нейтрофил среди эритроцитов, 540×428"),
    "X-ray_of_lobar_pneumonia.jpg": ("pneumonia", "pneumonia", "долевая пневмония, взрослый 67 лет"),
    "Boy_chest_x-ray_pic.jpg": ("pneumonia", "normal", "здоровые лёгкие, юноша 18 лет"),
    "Cecal_adenocarcinoma.jpg": ("path", "colorectal adenocarcinoma epithelium",
                                 "аденокарцинома слепой кишки, H&E, малое увеличение"),
}
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath/"


def class_names(flag):
    from medmnist import INFO
    lab = INFO[flag]["label"]
    return [lab[str(i)] for i in range(len(lab))]


# ------------------------------------------------------------------
#  Обучение
# ------------------------------------------------------------------
def cmd_train(a):
    cfg = ExperimentConfig(seed=a.seed, n_clients=3, n_tasks=3, crypto_mode="ckks",
                           protocol="cil", features_path=a.features)
    cfg.train = TrainConfig(lr=a.lr, rounds_per_task=a.rounds, batches_per_round=10, batch_size=64)
    cfg.defense = DefenseConfig(mode="none", C=0.3, C_K=0.3, C_P=0.3)
    cds = build_federation(cfg, verbose=True)
    z = 0.0
    if not a.no_dp:
        n_min = min(len(t.X_train) for cd in cds for t in cd.tasks if len(t.X_train) > 0)
        z = noise_for_epsilon(min(1.0, 64 / n_min), a.rounds * 10, 1e-5, a.eps)
    cfg.dpsgd = DPSGDConfig(z=z)
    print(f"\nОбучение: CKKS{' + DP-SGD (цель ε ≤ %g, z = %.4f)' % (a.eps, z) if z else ''}, "
          f"скорость {a.lr}, сид {a.seed}")
    t0 = time.time()
    res = run_federated_continual(cfg, cds, verbose=True, return_clients=True)
    clients = res.pop("clients")
    pools = [(c.pool.keys, c.pool.prompts) for c in clients]
    same_pool = all(np.array_equal(pools[0][0], k) and np.array_equal(pools[0][1], p) for k, p in pools)

    # точность итоговой модели на тестовых признаках каждой клиники (все изученные классы)
    test_acc = {}
    for c, cd in zip(clients, cds):
        X = np.concatenate([t.X_test for t in cd.tasks])
        y = np.concatenate([np.asarray(t.global_classes)[t.y_test] for t in cd.tasks])
        test_acc[cd.name] = c.evaluate(X, y, 0, None)

    out = {"keys": pools[0][0], "prompts": pools[0][1],
           "clinic_names": np.array([cd.name for cd in cds]),
           "meta": np.array(json.dumps({
               "seed": a.seed, "lr": a.lr, "rounds": a.rounds, "dpsgd_z": z,
               "eps_per_patient": res.get("privacy", {}).get("eps_per_patient"),
               "backend": res["backend"], "server_has_secret_key": res["server_has_secret_key"],
               "FAA": res["FAA"], "Forgetting": res["Forgetting"], "R": res["R"],
               "test_acc": test_acc, "pool_identical_across_clinics": same_pool,
               "features": a.features, "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
               "train_seconds": time.time() - t0}, ensure_ascii=False))}
    for c, cd in zip(clients, cds):
        W, b = c.heads["all"]
        out[f"{cd.name}_W"], out[f"{cd.name}_b"] = W, b
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    np.savez_compressed(MODEL, **out)
    m = json.loads(str(out["meta"]))
    print(f"\n  FAA = {m['FAA']:.4f}, забывание = {m['Forgetting']:.4f}, backend = {m['backend']}, "
          f"секретный ключ у сервера: {m['server_has_secret_key']}")
    if z:
        print(f"  гарантия для пациента: ε = {m['eps_per_patient']:.2f} (δ = 1e-5)")
    print(f"  пул одинаков у всех клиник после обучения: {same_pool}")
    for k, v in test_acc.items():
        print(f"  точность на тестовых признаках {k}: {v:.4f}")
    print(f"  модель сохранена: {os.path.relpath(MODEL, ROOT)}")


# ------------------------------------------------------------------
#  Модель для предсказаний
# ------------------------------------------------------------------
class Model:
    def __init__(self, path=None):
        path = path or MODEL
        if not os.path.exists(path):
            raise SystemExit(f"Нет модели {path}. Сначала: python system.py train")
        d = np.load(path, allow_pickle=True)
        self.meta = json.loads(str(d["meta"]))
        from core.config import PoolConfig
        self.pool = PromptPool(PoolConfig(), np.random.default_rng(0))
        self.pool.keys, self.pool.prompts = d["keys"], d["prompts"]
        self.heads = {str(n): (d[f"{n}_W"], d[f"{n}_b"]) for n in d["clinic_names"]}

    def proba(self, Q, clinic_name):
        """Q: [B, 192] нормированные CLS. Возвращает вероятности классов клиники."""
        idx = self.pool.select(Q)
        z = Q + self.pool.collapse(idx)
        W, b = self.heads[clinic_name]
        logits = z @ W.T + b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True), idx


class Featurizer:
    """Снимок -> CLS ViT-Tiny ровно так, как в scripts/prepare_features.py."""

    def __init__(self):
        import torch
        import timm
        import torchvision.transforms as T
        self.torch, self.T = torch, T
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model("vit_tiny_patch16_224", pretrained=True,
                                       num_classes=0).eval().to(self.device)

    def transform(self, channels):
        T = self.T
        return T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3) if channels == 1 else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @staticmethod
    def to_medmnist(img, channels, downscale=True):
        """Сырой снимок -> вид, на котором учили модель: 28×28, 1 или 3 канала."""
        from PIL import Image
        img = img.convert("L" if channels == 1 else "RGB")
        if downscale:
            img = img.resize((28, 28), Image.BICUBIC)
        return img

    def features(self, pil_images, channels, batch=64):
        tf = self.transform(channels)
        out = []
        with self.torch.no_grad():
            for s in range(0, len(pil_images), batch):
                x = self.torch.stack([tf(im) for im in pil_images[s:s + batch]]).to(self.device)
                f = self.torch.nn.functional.normalize(self.model(x), dim=1)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


# ------------------------------------------------------------------
#  Самопроверка: путь предсказания = путь обучения
# ------------------------------------------------------------------
def cmd_selftest(a):
    import medmnist
    from medmnist import INFO
    model, fz = Model(), Featurizer()
    feats = np.load(a.features, allow_pickle=True)
    report = {}
    print("Самопроверка: тестовые снимки MedMNIST -> признаки системы -> сравнение с файлом "
          "признаков, на котором обучали\n")
    for key, (name, flag, ch) in CLINICS.items():
        DataClass = getattr(medmnist, INFO[flag]["python_class"])
        ds = DataClass(split="test", download=True)
        n_file = len(feats[f"{name}_yte"])
        idx = np.random.default_rng(0).choice(len(ds), min(n_file, len(ds)), replace=False)
        take = idx[:a.n]
        imgs = [ds[int(i)][0] for i in take]               # PIL 28×28, как в prepare_features
        labels = np.array([int(ds[int(i)][1][0]) for i in take])
        F = fz.features([fz.to_medmnist(im, ch, downscale=False) for im in imgs], ch)
        Fref = feats[f"{name}_Xte"][:a.n]
        cos = (F * Fref).sum(1)
        lab_ok = bool(np.array_equal(labels, feats[f"{name}_yte"][:a.n]))
        P, _ = model.proba(F, name)
        Pref, _ = model.proba(Fref.astype(np.float64), name)
        agree = float(np.mean(P.argmax(1) == Pref.argmax(1)))
        acc = float(np.mean(P.argmax(1) == labels))
        report[name] = {"n": int(len(take)), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                        "max_abs_diff": float(np.abs(F - Fref).max()), "labels_equal": lab_ok,
                        "prediction_agreement": agree, "accuracy": acc}
        print(f"  {name:<24} снимков {len(take)}: косинус с признаками обучения min {cos.min():.6f}, "
              f"макс. разница {np.abs(F - Fref).max():.1e}, метки совпадают: {lab_ok}, "
              f"предсказания совпадают: {agree:.3f}, точность: {acc:.3f}")
    ok = all(r["cos_min"] > 0.999 and r["labels_equal"] and r["prediction_agreement"] == 1.0
             for r in report.values())
    print("\n  ИТОГ: путь «снимок -> диагноз» " + ("совпадает с путём обучения" if ok else "РАСХОДИТСЯ"))
    save_json(f"selftest", {"ok": ok, "clinics": report, "model_meta": model.meta})


# ------------------------------------------------------------------
#  Диагноз по одному снимку и оценка на папке
# ------------------------------------------------------------------
def predict_one(model, fz, clinic, image_path, expected=None, downscale=True, note=None):
    from PIL import Image
    name, flag, ch = CLINICS[clinic]
    names = class_names(flag)
    img = Image.open(image_path)
    F = fz.features([fz.to_medmnist(img, ch, downscale=downscale)], ch)
    P, idx = model.proba(F, name)
    order = np.argsort(-P[0])
    print(f"Снимок: {os.path.basename(image_path)}  ({img.size[0]}×{img.size[1]}, клиника {name})"
          + (f"  — {note}" if note else ""))
    print(f"  выбранные ячейки пула: {sorted(idx[0].tolist())}")
    for r, k in enumerate(order[:3]):
        print(f"  {r + 1}. {names[k]:<60} {P[0, k]:.3f}")
    res = {"image": image_path.replace("\\", "/"), "clinic": name, "size": list(img.size),
           "top3": [{"class": names[k], "index": int(k), "p": float(P[0, k])} for k in order[:3]],
           "expected": expected, "note": note}
    if expected is not None:
        ok1 = names[order[0]] == expected or str(order[0]) == str(expected)
        # при 3 и меньше классах «верно в top-3» выполняется всегда — метрика бессмысленна
        ok3 = (any(names[k] == expected or str(k) == str(expected) for k in order[:3])
               if len(names) > 3 else None)
        res.update(correct_top1=ok1, correct_top3=ok3)
        print(f"  верный диагноз: {expected} -> top-1 {'верно' if ok1 else 'неверно'}"
              + (f", top-3 {'верно' if ok3 else 'неверно'}" if ok3 is not None
                 else f" (классов {len(names)}: top-3 не считается)"))
    save_json(f"predict_{clinic}_{os.path.splitext(os.path.basename(image_path))[0]}", res)
    return res


def cmd_predict(a):
    predict_one(Model(), Featurizer(), a.clinic, a.image, a.expected, not a.no_downscale)


def cmd_internet(a):
    """Снимки из интернета: скачать недостающие (каждый в свою папку) и поставить диагноз."""
    import urllib.request
    model, fz = Model(), Featurizer()
    done = []
    for fname, (clinic, expected, note) in INTERNET.items():
        d = os.path.join(INTERNET_DIR, os.path.splitext(fname)[0])
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            os.makedirs(d, exist_ok=True)
            req = urllib.request.Request(COMMONS + fname, headers={"User-Agent": "fcl-thesis-check/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
            print(f"  скачано: {COMMONS + fname}")
        done.append(predict_one(model, fz, clinic, path, expected, True, note))
        print()
    t1 = sum(r["correct_top1"] for r in done)
    t3r = [r for r in done if r["correct_top3"] is not None]
    t3 = sum(r["correct_top3"] for r in t3r)
    print(f"ИТОГ: верно top-1 {t1} из {len(done)}; top-3 (клиники с >3 классами) {t3} из {len(t3r)} "
          f"(источник: Wikimedia Commons; снимки не входят в обучающие наборы)")


def cmd_evaluate(a):
    from PIL import Image
    name, flag, ch = CLINICS[a.clinic]
    names = class_names(flag)
    alias = dict(FOLDER_ALIASES[a.clinic])
    alias.update({n: i for i, n in enumerate(names)})
    alias.update({str(i): i for i in range(len(names))})
    rng = np.random.default_rng(a.seed)
    paths, labels = [], []
    for sub in sorted(os.listdir(a.folder)):
        if sub not in alias or not os.path.isdir(os.path.join(a.folder, sub)):
            continue
        files = sorted(f for f in os.listdir(os.path.join(a.folder, sub))
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")))
        if a.limit and len(files) > a.limit:
            files = list(rng.choice(files, a.limit, replace=False))
        paths += [os.path.join(a.folder, sub, f) for f in files]
        labels += [alias[sub]] * len(files)
    if not paths:
        raise SystemExit("В папке не найдено подпапок-классов со снимками")
    model, fz = Model(), Featurizer()
    labels = np.array(labels)
    F = fz.features([fz.to_medmnist(Image.open(p), ch, downscale=not a.no_downscale) for p in paths], ch)
    P, _ = model.proba(F, name)
    pred = P.argmax(1)
    acc = float(np.mean(pred == labels))
    present = sorted(set(labels.tolist()))
    per = {names[c]: float(np.mean(pred[labels == c] == c)) for c in present}
    cm = [[int(np.sum((labels == t) & (pred == p))) for p in range(len(names))] for t in present]
    chance = float(max(np.mean(labels == c) for c in present))
    print(f"Папка: {a.folder}  снимков: {len(paths)}  классов: {len(present)}  клиника: {name}")
    print(f"  точность: {acc:.4f}  (доля самого частого класса: {chance:.4f})")
    for k, v in per.items():
        print(f"    {k:<60} {v:.3f}")
    save_json(f"evaluate_{a.clinic}_{a.tag}", {
        "folder": a.folder, "clinic": name, "n": int(len(paths)), "accuracy": acc,
        "majority_rate": chance, "per_class": per, "confusion_rows_true_cols_pred": cm,
        "downscale_to_28": not a.no_downscale, "limit_per_class": a.limit,
        "model_meta": {k: model.meta[k] for k in ("FAA", "dpsgd_z", "eps_per_patient", "test_acc")}})


def save_json(tag, obj):
    out = RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  сохранено: {os.path.relpath(path, ROOT)}")


def main():
    global MODEL, RESULTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL, help="файл модели (по умолчанию models/fcl_system.npz)")
    ap.add_argument("--results_dir", default=RESULTS_DIR, help="куда писать результаты")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--rounds", type=int, default=15)
    t.add_argument("--eps", type=float, default=4.0)
    t.add_argument("--no-dp", action="store_true")
    t.add_argument("--features", default="data/features_primary.npz")
    s = sub.add_parser("selftest")
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--features", default="data/features_primary.npz")
    p = sub.add_parser("predict")
    p.add_argument("--clinic", choices=list(CLINICS), required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--expected", default=None, help="верный класс (для проверки)")
    p.add_argument("--no-downscale", action="store_true")
    sub.add_parser("internet")
    e = sub.add_parser("evaluate")
    e.add_argument("--clinic", choices=list(CLINICS), required=True)
    e.add_argument("--folder", required=True)
    e.add_argument("--limit", type=int, default=0, help="не больше N снимков на класс")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--tag", default="folder")
    e.add_argument("--no-downscale", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    MODEL, RESULTS_DIR = os.path.abspath(a.model), os.path.abspath(a.results_dir)
    {"train": cmd_train, "selftest": cmd_selftest, "predict": cmd_predict,
     "internet": cmd_internet, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
