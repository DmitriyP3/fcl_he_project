"""
config.py — все параметры эксперимента в одном месте.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Tuple
import json


@dataclass
class PoolConfig:
    """Параметры пула обучаемых подсказок (L2P)."""
    M: int = 10        # количество ячеек пула
    N: int = 4         # top-N выбираемых ячеек на один вход
    L_p: int = 5       # длина последовательности подсказок
    D_k: int = 192     # размерность ключа (= размерность CLS-токена ViT-Tiny)
    D: int = 192       # размерность вектора подсказки

    @property
    def keys_size(self) -> int:
        return self.M * self.D_k

    @property
    def prompts_size(self) -> int:
        return self.M * self.L_p * self.D

    @property
    def total_floats(self) -> int:
        return self.keys_size + self.prompts_size


@dataclass
class TrainConfig:
    """Параметры локального обучения."""
    lr: float = 1e-3           # скорость обучения (Adam)
    lam: float = 0.5           # вес вспомогательной потери выравнивания ключей
    batch_size: int = 64
    batches_per_round: int = 20
    rounds_per_task: int = 20  # раундов федеративного обучения на одну задачу
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class CKKSConfig:
    """
    Параметры схемы CKKS (TenSEAL).

    poly_modulus_degree = 8192 даёт 4096 вещественных слотов и 128-битную
    стойкость при цепочке модулей (60, 40, 40, 60). Пул из 11 520 чисел
    размещается в 3 шифртекстах. В тексте диссертации указывайте именно 8192.

    Внутренние модули по 40 бит выбраны с запасом: в литературе показано, что
    при внутренних модулях меньше 26 бит шум CKKS за много раундов начинает
    ухудшать сходимость (pFedCKKS, arXiv:2606.08521).
    """
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: Tuple[int, ...] = (60, 40, 40, 60)
    scale_bits: int = 40

    # Коэффициент усиления РЕАЛЬНОЙ погрешности CKKS (сценарий sensitivity).
    # 1.0 — штатный режим. k > 1 — та же по структуре погрешность, умноженная на k.
    error_amplify: float = 1.0

    @property
    def slots(self) -> int:
        """Количество вещественных слотов в одном шифртексте."""
        return self.poly_modulus_degree // 2


@dataclass
class DefenseConfig:
    """
    Нормализация обновлений перед отправкой.

    ВАЖНО: нормализация масштабирует вектор обновления целиком, но не меняет
    его структуру по ячейкам. Поэтому она скрывает только суммарный размер
    обновления и не защищает от наблюдателя, видящего обновление по ячейкам.
    Реальная оценка — сценарий attack.
    """
    mode: str = "none"      # none | global | percomponent | dp
    C: float = 0.3          # целевая норма (global) / порог отсечения (dp)
    # dp: множитель шума. К каждой координате обрезанного обновления
    # добавляется N(0, (sigma*C)^2). По классической оценке гауссова механизма
    # для гарантии (eps=1, delta=1e-5) на ОДИН раунд нужен sigma ≈ 4.84.
    sigma: float = 0.0
    C_K: float = 0.3        # целевая норма обновления ключей
    C_P: float = 0.3        # целевая норма обновления подсказок


@dataclass
class DPSGDConfig:
    """
    Формальная гарантия для каждого пациента (DP-SGD).
    z = 0 — выключено. Иначе каждый шаг: пуассоновская выборка снимков
    с вероятностью q = batch_size / N, отсечение градиента каждого снимка
    порогом clip, гауссов шум с множителем z, учёт (eps, delta) по
    фактически выполненным шагам на данных каждой задачи.
    """
    z: float = 0.0
    clip: float = 1.0
    delta: float = 1e-5
    # Отсечение по группам параметров (McMahan et al., ICLR 2018):
    #   global — один порог clip на весь градиент снимка;
    #   group  — свои пороги для головы, подсказок и ключей, шум в группе
    #            z*sqrt(G)*C_g. Гарантия та же (множитель шума z).
    # Пороги групп фиксированы заранее (не подбираются по приватным данным).
    clip_mode: str = "global"
    clip_head: float = 1.0
    clip_prompt: float = 0.03
    clip_key: float = 1.0


@dataclass
class ExperimentConfig:
    """Полная конфигурация прогона."""
    seed: int = 1
    n_clients: int = 3
    n_tasks: int = 3               # число последовательных задач
    crypto_mode: str = "plaintext" # plaintext | ckks

    # Протокол непрерывного обучения:
    #   til — task-incremental: у каждой задачи своя голова, номер задачи
    #         известен при тестировании (самый лёгкий режим; так было раньше)
    #   cil — class-incremental: одна голова на все классы клиента,
    #         номер задачи при тестировании НЕ известен (как задумано в L2P)
    protocol: str = "til"

    # Правило агрегации пула:
    #   uniform — обычное среднее обновлений клиник (FedAvg)
    #   usage   — вклад клиники в ячейку взвешивается числом её снимков,
    #             выбравших эту ячейку за раунд. Совместимо с шифрованием:
    #             клиника шифрует (счётчик × обновление) и сам счётчик, сервер
    #             складывает оба, клиники делят после расшифровки. Счётчики
    #             (M чисел) помещаются в те же шифртексты — трафик не растёт.
    aggregation: str = "uniform"

    # КОНТРОЛЬ достоверности сценария temporal: все фазы обучаются на данных
    # ОДНОЙ задачи (задачи 0). Если атакующий всё равно «различает фазы», он
    # ловит стадию обучения, а не смену диагнозов. Только для протокола cil.
    control_same_task: bool = False

    # Записывать то, что реально наблюдает сервер (сценарий temporal):
    # профили обновлений по ячейкам и статистику байтов шифртекстов.
    record_observations: bool = False

    # ограничение объёма данных НА ОДНУ ЗАДАЧУ клиента
    max_train_per_client: int = 2000
    max_test_per_client: int = 500

    pool: PoolConfig = field(default_factory=PoolConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ckks: CKKSConfig = field(default_factory=CKKSConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    dpsgd: DPSGDConfig = field(default_factory=DPSGDConfig)

    features_path: str = "data/features_primary.npz"
    out_dir: str = "results"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
