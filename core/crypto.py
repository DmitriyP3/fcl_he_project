"""
crypto.py — гомоморфное шифрование агрегации.

Два режима:
    plaintext — агрегация без шифрования (эталон)
    ckks      — реальный CKKS через TenSEAL

Если TenSEAL не установлен, режим ckks работает в режиме симуляции:
численная погрешность CKKS моделируется добавлением шума того же порядка,
а объём трафика считается аналитически. Это позволяет прогнать весь конвейер
без TenSEAL, но для диссертации нужен реальный режим (см. README).

Учёт трафика ведётся честно: измеряется размер сериализованного шифртекста.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import tenseal as ts
    HAS_TENSEAL = True
except Exception:
    HAS_TENSEAL = False


# Погрешность CKKS при scale = 2^40 имеет порядок 1e-7 относительно значения.
SIMULATED_CKKS_REL_ERROR = 1e-7


@dataclass
class CommStats:
    """Накопленная статистика трафика и времени."""
    bytes_uploaded: int = 0
    bytes_downloaded: int = 0
    encrypt_ms: float = 0.0
    aggregate_ms: float = 0.0
    decrypt_ms: float = 0.0
    n_rounds: int = 0
    ciphertexts_per_client_per_round: int = 0

    def add_round(self, up: int, down: int, enc: float, agg: float, dec: float):
        self.bytes_uploaded += up
        self.bytes_downloaded += down
        self.encrypt_ms += enc
        self.aggregate_ms += agg
        self.decrypt_ms += dec
        self.n_rounds += 1

    def summary(self) -> dict:
        r = max(1, self.n_rounds)
        total = self.bytes_uploaded + self.bytes_downloaded
        return {
            "n_rounds": self.n_rounds,
            "bytes_per_round_total": total / r,
            "bytes_per_round_uploaded": self.bytes_uploaded / r,
            "kb_per_round_total": total / r / 1024,
            "mb_total": total / (1024 ** 2),
            "encrypt_ms_per_round": self.encrypt_ms / r,
            "aggregate_ms_per_round": self.aggregate_ms / r,
            "decrypt_ms_per_round": self.decrypt_ms / r,
            "ciphertexts_per_client_per_round": self.ciphertexts_per_client_per_round,
        }


class Aggregator:
    """
    Агрегатор обновлений пула.

    В режиме ckks сервер складывает ШИФРТЕКСТЫ и не имеет секретного ключа.
    Расшифровку выполняет клиентская сторона.
    """

    def __init__(self, cfg, mode: str = "plaintext"):
        self.mode = mode
        self.ckks_cfg = cfg.ckks
        self.stats = CommStats()
        self.ctx = None          # контекст клиник: содержит СЕКРЕТНЫЙ ключ
        self.server_ctx = None   # контекст сервера: только открытый ключ
        self.real_ckks = False
        # усиление реальной погрешности CKKS (сценарий sensitivity)
        self.error_amplify = float(getattr(cfg.ckks, "error_amplify", 1.0))
        # учёт реальной погрешности агрегации относительно открытого эталона
        # запись наблюдений сервера (сценарий temporal)
        self.record = bool(getattr(cfg, "record_observations", False))
        self.observed = []     # по раундам: [n_clients, 16] гистограмма байтов
        self.observed_proj = []  # по раундам: [n_clients, 256] проекция сырых байтов
        self.drift_abs_max = 0.0
        self.drift_l2_sum = 0.0

        if mode == "ckks":
            if HAS_TENSEAL:
                self.ctx = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=self.ckks_cfg.poly_modulus_degree,
                    coeff_mod_bit_sizes=list(self.ckks_cfg.coeff_mod_bit_sizes),
                )
                self.ctx.global_scale = 2 ** self.ckks_cfg.scale_bits
                # Сервер получает копию контекста БЕЗ секретного ключа.
                # Для сложения и умножения на открытую константу ключи
                # вращения и релинеаризации не нужны.
                pub = self.ctx.copy()
                pub.make_context_public()
                self.server_ctx = ts.context_from(pub.serialize())
                self.real_ckks = True
            else:
                self.real_ckks = False

    def server_has_secret_key(self) -> bool:
        """Проверка разделения ключей: у сервера секретного ключа быть не должно."""
        if self.server_ctx is None:
            return False
        return bool(self.server_ctx.has_secret_key())

    # ---------------- служебное ----------------
    def _chunks(self, vec: np.ndarray) -> List[np.ndarray]:
        """Режет вектор на куски по числу слотов шифртекста."""
        s = self.ckks_cfg.slots
        return [vec[i:i + s] for i in range(0, len(vec), s)]

    def _plain_bytes(self, vec: np.ndarray) -> int:
        return vec.astype(np.float64).nbytes

    def _simulated_cipher_bytes(self, n_floats: int) -> int:
        """
        Аналитическая оценка размера шифртекста CKKS.
        Шифртекст состоит из 2 полиномов степени poly_modulus_degree,
        коэффициенты в RNS-представлении по числу простых модулей.
        """
        n = self.ckks_cfg.poly_modulus_degree
        n_primes = len(self.ckks_cfg.coeff_mod_bit_sizes)
        bytes_per_ct = 2 * n * n_primes * 8
        n_ct = int(np.ceil(n_floats / self.ckks_cfg.slots))
        return bytes_per_ct * n_ct

    # ---------------- основная операция ----------------
    def aggregate(self, updates: List[np.ndarray]) -> np.ndarray:
        """
        Усредняет обновления клиентов.
        updates: список векторов одинаковой длины.
        Возвращает усреднённый вектор (расшифрованный на стороне клиента).
        """
        n = len(updates)
        length = len(updates[0])

        if self.mode == "plaintext":
            t0 = time.perf_counter()
            agg = np.mean(np.stack(updates, axis=0), axis=0)
            t1 = time.perf_counter()
            up = sum(self._plain_bytes(u) for u in updates)
            down = self._plain_bytes(agg) * n
            self.stats.add_round(up, down, 0.0, (t1 - t0) * 1000, 0.0)
            self.stats.ciphertexts_per_client_per_round = 0
            return agg

        # ---- режим CKKS ----
        if self.real_ckks:
            return self._aggregate_real_ckks(updates)
        return self._aggregate_simulated(updates)

    def _aggregate_real_ckks(self, updates: List[np.ndarray]) -> np.ndarray:
        n = len(updates)
        length = len(updates[0])

        # --- 1. клиенты шифруют и отправляют БАЙТЫ ---
        t0 = time.perf_counter()
        sent = []            # что реально уходит по сети от каждого клиента
        up_bytes = 0
        n_ct = 0
        for u in updates:
            blobs = []
            for ch in self._chunks(u):
                b = ts.ckks_vector(self.ctx, ch.tolist()).serialize()
                blobs.append(b)
                up_bytes += len(b)
            sent.append(blobs)
            n_ct = len(blobs)
        t1 = time.perf_counter()

        # Сервер видит только байты. Записываем их статистику —
        # это всё, на чём наблюдатель может построить признаки.
        if self.record:
            hs = []
            for blobs in sent:
                raw = np.frombuffer(b"".join(blobs), dtype=np.uint8)
                h = np.bincount(raw, minlength=256).reshape(16, 16).sum(axis=1)
                hs.append(h / max(1, h.sum()))
            self.observed.append(np.array(hs, dtype=np.float64))
            # сильный атакующий по шифртексту: проекция первых 8192 байтов
            if not hasattr(self, "_byte_proj"):
                g = np.random.default_rng(54321)
                self._byte_proj = g.standard_normal((256, 8192)).astype(np.float32) / 16.0
            P = self._byte_proj
            ps = []
            for blobs in sent:
                raw = np.frombuffer(b"".join(blobs)[:8192], dtype=np.uint8)
                ps.append(P @ (raw.astype(np.float32) - 127.5))
            self.observed_proj.append(np.array(ps, dtype=np.float64))

        # --- 2. сервер: восстанавливает шифртексты в ПУБЛИЧНОМ контексте
        #        (секретного ключа нет) и складывает их ---
        acc = [ts.ckks_vector_from(self.server_ctx, b) for b in sent[0]]
        for blobs in sent[1:]:
            acc = [a + ts.ckks_vector_from(self.server_ctx, b)
                   for a, b in zip(acc, blobs)]
        acc = [a * (1.0 / n) for a in acc]
        back = [a.serialize() for a in acc]      # рассылка клиентам
        t2 = time.perf_counter()
        down_bytes = sum(len(b) for b in back) * n

        # --- 3. клиенты расшифровывают своим секретным ключом ---
        out = []
        for b in back:
            out.extend(ts.ckks_vector_from(self.ctx, b).decrypt())
        t3 = time.perf_counter()

        agg = np.array(out[:length], dtype=np.float64)

        # --- учёт реальной погрешности и (опционально) её усиление ---
        exact = np.mean(np.stack(updates, axis=0), axis=0)
        err = agg - exact
        self.drift_abs_max = max(self.drift_abs_max, float(np.abs(err).max()))
        self.drift_l2_sum += float(np.sqrt(np.sum(err * err)))
        if self.error_amplify != 1.0:
            agg = exact + err * self.error_amplify

        self.stats.add_round(up_bytes, down_bytes,
                             (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000)
        self.stats.ciphertexts_per_client_per_round = n_ct
        return agg

    def drift_summary(self) -> dict:
        r = max(1, self.stats.n_rounds)
        return {
            "drift_abs_max": self.drift_abs_max,
            "drift_l2_per_round": self.drift_l2_sum / r,
            "error_amplify": self.error_amplify,
        }

    def _aggregate_simulated(self, updates: List[np.ndarray]) -> np.ndarray:
        n = len(updates)
        length = len(updates[0])

        t0 = time.perf_counter()
        agg = np.mean(np.stack(updates, axis=0), axis=0)
        # моделируем приближённую природу CKKS
        scale = np.abs(agg).mean() + 1e-12
        noise = np.random.default_rng(0).standard_normal(agg.shape)
        agg = agg + noise * scale * SIMULATED_CKKS_REL_ERROR
        t1 = time.perf_counter()

        up = self._simulated_cipher_bytes(length) * n
        down = self._simulated_cipher_bytes(length) * n
        n_ct = int(np.ceil(length / self.ckks_cfg.slots))
        self.stats.add_round(up, down, 0.0, (t1 - t0) * 1000, 0.0)
        self.stats.ciphertexts_per_client_per_round = n_ct
        return agg

    def backend_name(self) -> str:
        if self.mode == "plaintext":
            return "plaintext"
        return "ckks-tenseal" if self.real_ckks else "ckks-simulated"
