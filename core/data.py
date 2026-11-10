"""
data.py — загрузка признаков и нарезка на ПОСЛЕДОВАТЕЛЬНЫЕ задачи.

Это ключевой файл для федеративного НЕПРЕРЫВНОГО обучения.
В отличие от обычного федеративного обучения, где клиент всё время учится на
одном и том же наборе, здесь каждый клиент проходит задачи по очереди:
    задача 0 -> задача 1 -> задача 2 ...
и после каждой задачи мы измеряем точность на ВСЕХ пройденных задачах.

Схема нарезки:
  - если классов у клиента не меньше, чем задач, — class-incremental
    (классы делятся на n_tasks непересекающихся групп);
  - иначе — domain-incremental (те же классы, разные порции данных).
Ограничения max_train/max_test действуют НА ОДНУ ЗАДАЧУ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class Task:
    """Одна задача одного клиента."""
    task_id: int
    X_train: np.ndarray   # [n, D] признаки (CLS-токены)
    y_train: np.ndarray   # [n] локальные метки 0..K_t-1
    X_test: np.ndarray
    y_test: np.ndarray
    global_classes: List[int]  # исходные номера классов


@dataclass
class ClientData:
    """Все задачи одного клиента, по порядку."""
    client_id: int
    name: str
    tasks: List[Task]
    n_classes_total: int   # макс. число классов в одной задаче (историческое имя)
    n_classes_all: int = 0 # всего различных классов у клиента (для протокола cil)


def _split_classes(classes: List[int], n_tasks: int) -> List[List[int]]:
    """Делит список классов на n_tasks непересекающихся групп."""
    classes = sorted(classes)
    groups: List[List[int]] = [[] for _ in range(n_tasks)]
    for i, c in enumerate(classes):
        groups[i % n_tasks].append(c)
    return groups


def _domain_incremental_split(y: np.ndarray, n_tasks: int,
                              rng: np.random.Generator) -> List[np.ndarray]:
    """
    Domain-incremental нарезка: пространство меток одно и то же,
    но данные делятся на n_tasks непересекающихся порций.

    Применяется для клиентов, у которых классов меньше, чем задач
    (например, PneumoniaMNIST — всего 2 класса).
    """
    idx = rng.permutation(len(y))
    return [np.sort(part) for part in np.array_split(idx, n_tasks)]


def build_client_tasks(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_tasks: int, client_id: int, name: str,
    max_train: int, max_test: int,
    rng: np.random.Generator,
) -> ClientData:
    """
    Нарезает данные одного клиента на n_tasks последовательных задач
    по принципу class-incremental.
    """
    all_classes = sorted(set(int(c) for c in np.unique(y_train)))

    # Выбор схемы нарезки:
    #   классов достаточно -> class-incremental (задачи = разные диагнозы)
    #   классов мало       -> domain-incremental (задачи = разные порции данных)
    class_incremental = len(all_classes) >= n_tasks
    if class_incremental:
        groups = _split_classes(all_classes, n_tasks)
        split_tr = split_te = None
    else:
        groups = [all_classes for _ in range(n_tasks)]
        split_tr = _domain_incremental_split(y_train, n_tasks, rng)
        split_te = _domain_incremental_split(y_test, n_tasks, rng)

    tasks: List[Task] = []
    n_classes_total = 0
    for t, group in enumerate(groups):
        if class_incremental:
            tr_mask = np.isin(y_train, group)
            te_mask = np.isin(y_test, group)
            Xtr, ytr = X_train[tr_mask], y_train[tr_mask]
            Xte, yte = X_test[te_mask], y_test[te_mask]
        else:
            Xtr, ytr = X_train[split_tr[t]], y_train[split_tr[t]]
            Xte, yte = X_test[split_te[t]], y_test[split_te[t]]

        # ограничение объёма
        if len(Xtr) > max_train:
            idx = rng.choice(len(Xtr), max_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        if len(Xte) > max_test:
            idx = rng.choice(len(Xte), max_test, replace=False)
            Xte, yte = Xte[idx], yte[idx]

        # перенумеровка меток внутри задачи: 0..K_t-1
        remap = {c: i for i, c in enumerate(group)}
        ytr_local = np.array([remap[int(c)] for c in ytr], dtype=np.int64)
        yte_local = np.array([remap[int(c)] for c in yte], dtype=np.int64)

        if len(Xtr) == 0 or len(Xte) == 0:
            print(f"    ВНИМАНИЕ: у клиента '{name}' задача {t} пуста "
                  f"(train={len(Xtr)}, test={len(Xte)}). Уменьшите --tasks.")

        tasks.append(Task(
            task_id=t,
            X_train=Xtr.astype(np.float32), y_train=ytr_local,
            X_test=Xte.astype(np.float32), y_test=yte_local,
            global_classes=group,
        ))
        n_classes_total = max(n_classes_total, len(group))

    n_classes_all = int(max(max(t.global_classes) for t in tasks)) + 1
    return ClientData(client_id=client_id, name=name,
                      tasks=tasks, n_classes_total=n_classes_total,
                      n_classes_all=n_classes_all)


def load_features(path: str) -> Dict[str, np.ndarray]:
    """Загружает заранее извлечённые признаки из npz."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл признаков не найден: {path}\n"
            f"Сначала выполните:  python scripts/prepare_features.py"
        )
    return dict(np.load(path, allow_pickle=True))


def build_federation(cfg, verbose: bool = True) -> List[ClientData]:
    """
    Собирает список клиентов с их последовательными задачами.

    Ожидаемая структура npz:
        <name>_Xtr, <name>_ytr, <name>_Xte, <name>_yte  для каждого клиента
        client_names — массив имён
    """
    data = load_features(cfg.features_path)
    names = [str(n) for n in data["client_names"]]
    rng = np.random.default_rng(cfg.seed)

    if len(names) < cfg.n_clients and verbose:
        print(f"  ВНИМАНИЕ: запрошено клиентов {cfg.n_clients}, а в файле "
              f"признаков только {len(names)}. Эксперимент пойдёт на "
              f"{len(names)} клиентах — это нужно указать в тексте.")

    clients: List[ClientData] = []
    for i, name in enumerate(names[: cfg.n_clients]):
        cd = build_client_tasks(
            X_train=data[f"{name}_Xtr"], y_train=data[f"{name}_ytr"],
            X_test=data[f"{name}_Xte"], y_test=data[f"{name}_yte"],
            n_tasks=cfg.n_tasks, client_id=i, name=name,
            max_train=cfg.max_train_per_client,
            max_test=cfg.max_test_per_client,
            rng=rng,
        )
        clients.append(cd)
        if verbose:
            sizes = [(len(t.X_train), len(t.X_test), len(t.global_classes))
                     for t in cd.tasks]
            print(f"  клиент {i} ({name}): задач={len(cd.tasks)}, "
                  f"(train,test,классов) по задачам: {sizes}")
    return clients
