"""
pool.py — обучаемый пул подсказок L2P и локальная модель клиента.

Реализация на чистом NumPy: градиенты выписаны вручную, autograd не нужен,
поскольку модель проста (замороженный фичеризатор + пул + линейная голова).

Устройство:
    Keys    [M, D_k]        ключи ячеек
    Prompts [M, L_p, D]     последовательности подсказок (prompt sequence)

Прямой проход:
    q = CLS-токен изображения (D_k)
    sims = cos(q, Keys[m]) для всех m
    top-N ячеек -> усреднение их подсказок по ячейкам и позициям -> p (D)
    z = q + p
    logits = W z + b
"""
from __future__ import annotations

from typing import Tuple
import numpy as np


def l2(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))


class PromptPool:
    """Пул обучаемых ключей и подсказок."""

    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.keys = (rng.standard_normal((cfg.M, cfg.D_k)) * 0.02).astype(np.float64)
        self.prompts = (rng.standard_normal((cfg.M, cfg.L_p, cfg.D)) * 0.02).astype(np.float64)

    # ---------- сериализация в плоский вектор (для шифрования) ----------
    def flatten(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.keys.reshape(-1).copy(), self.prompts.reshape(-1).copy()

    def load_flat(self, keys_flat: np.ndarray, prompts_flat: np.ndarray) -> None:
        self.keys = keys_flat.reshape(self.cfg.M, self.cfg.D_k).copy()
        self.prompts = prompts_flat.reshape(self.cfg.M, self.cfg.L_p, self.cfg.D).copy()

    def copy(self) -> "PromptPool":
        p = PromptPool.__new__(PromptPool)
        p.cfg = self.cfg
        p.keys = self.keys.copy()
        p.prompts = self.prompts.copy()
        return p

    # ---------- выбор ячеек ----------
    def select(self, Q: np.ndarray) -> np.ndarray:
        """
        Q: [B, D_k] батч CLS-токенов.
        Возвращает индексы top-N ячеек: [B, N].
        """
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        kn = self.keys / (np.linalg.norm(self.keys, axis=1, keepdims=True) + 1e-8)
        sims = qn @ kn.T                       # [B, M]
        idx = np.argsort(-sims, axis=1)[:, : self.cfg.N]
        return idx

    def collapse(self, idx: np.ndarray) -> np.ndarray:
        """
        Сворачивает подсказки выбранных ячеек в один вектор на пример.
        idx: [B, N] -> возвращает [B, D].
        """
        # prompts[idx]: [B, N, L_p, D] -> среднее по N и L_p
        sel = self.prompts[idx]                # [B, N, L_p, D]
        return sel.mean(axis=(1, 2))           # [B, D]


class AdamState:
    """Минимальная реализация Adam для произвольного массива."""

    def __init__(self, shape, cfg):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0
        self.cfg = cfg

    def step(self, param: np.ndarray, grad: np.ndarray, lr: float) -> None:
        c = self.cfg
        self.t += 1
        self.m = c.beta1 * self.m + (1 - c.beta1) * grad
        self.v = c.beta2 * self.v + (1 - c.beta2) * (grad * grad)
        mhat = self.m / (1 - c.beta1 ** self.t)
        vhat = self.v / (1 - c.beta2 ** self.t)
        param -= lr * mhat / (np.sqrt(vhat) + c.eps)


class Client:
    """
    Клиент федерации: общий пул + локальная классификационная голова.

    Головы НЕ передаются на сервер (у клиник разные наборы диагнозов).

    Протоколы непрерывного обучения:
      til — своя голова на каждую задачу; при тестировании задачи t
            используется голова задачи t (номер задачи известен);
      cil — одна голова на все классы клиента; при обучении задаче t
            логиты классов других задач маскируются; при тестировании
            предсказание делается среди ВСЕХ изученных классов
            (номер задачи неизвестен) — режим, для которого задуман L2P.
    """

    def __init__(self, client_id: int, name: str, pool_cfg, train_cfg,
                 rng: np.random.Generator, protocol: str = "til",
                 n_classes_all: int = 0):
        self.id = client_id
        self.name = name
        self.pool_cfg = pool_cfg
        self.train_cfg = train_cfg
        self.rng = rng
        self.protocol = protocol
        self.n_classes_all = n_classes_all
        self.pool = PromptPool(pool_cfg, rng)

        self.heads = {}        # ключ головы -> [W, b]
        self._head_opt = {}    # ключ головы -> (AdamState W, AdamState b)
        self._pool_opt = None
        # сколько раз за текущий раунд снимки этой клиники выбрали каждую ячейку
        self.usage = np.zeros(pool_cfg.M, dtype=np.float64)

    # ---------- голова ----------
    def _hkey(self, task_id: int):
        return "all" if self.protocol == "cil" else task_id

    def ensure_head(self, task_id: int, n_classes: int) -> None:
        key = self._hkey(task_id)
        if key in self.heads:
            return
        k = self.n_classes_all if self.protocol == "cil" else n_classes
        W = (self.rng.standard_normal((k, self.pool_cfg.D)) * 0.01)
        b = np.zeros(k, dtype=np.float64)
        self.heads[key] = [W, b]
        self._head_opt[key] = (
            AdamState(W.shape, self.train_cfg),
            AdamState(b.shape, self.train_cfg),
        )

    def _ensure_pool_opt(self) -> None:
        if self._pool_opt is None:
            self._pool_opt = (
                AdamState(self.pool.keys.shape, self.train_cfg),
                AdamState(self.pool.prompts.shape, self.train_cfg),
            )

    # ---------- прямой проход ----------
    def forward(self, Q: np.ndarray, task_id: int, class_mask=None):
        idx = self.pool.select(Q)              # [B, N]
        p = self.pool.collapse(idx)            # [B, D]
        z = Q + p                              # [B, D]
        W, b = self.heads[self._hkey(task_id)]
        logits = z @ W.T + b                   # [B, K]
        if class_mask is not None:
            logits = np.where(class_mask[None, :], logits, -1e9)
        return logits, z, p, idx

    # ---------- шаг обучения ----------
    def train_step(self, Q: np.ndarray, y: np.ndarray, task_id: int,
                   class_mask=None) -> float:
        cfg = self.train_cfg
        self._ensure_pool_opt()
        B = len(Q)

        logits, z, p, idx = self.forward(Q, task_id, class_mask)
        self.usage += np.bincount(idx.ravel(), minlength=self.pool_cfg.M)

        # softmax + кросс-энтропия (замаскированные классы получают вероятность 0)
        logits = logits - logits.max(axis=1, keepdims=True)
        expl = np.exp(logits)
        probs = expl / (expl.sum(axis=1, keepdims=True) + 1e-12)
        loss = float(-np.log(probs[np.arange(B), y] + 1e-12).mean())

        # dL/dlogits
        dlog = probs.copy()
        dlog[np.arange(B), y] -= 1.0
        dlog /= B                              # [B, K]

        # градиенты головы
        key = self._hkey(task_id)
        W, b = self.heads[key]
        gW = dlog.T @ z                        # [K, D]
        gb = dlog.sum(axis=0)                  # [K]
        dz = dlog @ W                          # [B, D]  (до обновления W)

        optW, optb = self._head_opt[key]
        optW.step(W, gW, cfg.lr)
        optb.step(b, gb, cfg.lr)

        # градиент по подсказкам: p = среднее по (N * L_p) выбранных векторов
        g_prompts = np.zeros_like(self.pool.prompts)
        denom = self.pool_cfg.N * self.pool_cfg.L_p
        for i in range(B):
            contrib = dz[i] / denom
            for m in idx[i]:
                g_prompts[m] += contrib        # broadcast по L_p

        # вспомогательная потеря выравнивания ключей:
        # L_align = lam * mean_i mean_{m in top-N} (1 - cos(q_i, k_m))
        g_keys = np.zeros_like(self.pool.keys)
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        knorm = np.linalg.norm(self.pool.keys, axis=1, keepdims=True) + 1e-8
        kn = self.pool.keys / knorm
        for i in range(B):
            for m in idx[i]:
                cos = float(np.dot(qn[i], kn[m]))
                gk = -(qn[i] - cos * kn[m]) / float(knorm[m, 0])
                g_keys[m] += cfg.lam * gk / (B * self.pool_cfg.N)

        optK, optP = self._pool_opt
        optK.step(self.pool.keys, g_keys, cfg.lr)
        optP.step(self.pool.prompts, g_prompts, cfg.lr)
        return loss

    # ---------- шаг DP-SGD ----------
    def _per_sample_grads(self, Q, y, task_id, class_mask):
        """
        Градиенты КАЖДОГО снимка отдельно (без деления на размер пакета).
        Сумма этих градиентов, делённая на B, в точности равна градиенту
        обычного шага train_step.
        Возвращает (dlog [B,K], zfeat [B,D], dz [B,D], idx [B,N],
                    gk [B,N,Dk] — градиенты ключей выбранных ячеек, loss).
        """
        cfg = self.train_cfg
        B = len(Q)
        logits, zfeat, p, idx = self.forward(Q, task_id, class_mask)
        logits = logits - logits.max(axis=1, keepdims=True)
        expl = np.exp(logits)
        probs = expl / (expl.sum(axis=1, keepdims=True) + 1e-12)
        loss = float(-np.log(probs[np.arange(B), y] + 1e-12).mean())
        dlog = probs.copy()
        dlog[np.arange(B), y] -= 1.0                       # [B, K]
        W, _ = self.heads[self._hkey(task_id)]
        dz = dlog @ W                                      # [B, D]
        qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
        knorm = np.linalg.norm(self.pool.keys, axis=1, keepdims=True) + 1e-8
        kn = self.pool.keys / knorm
        Nn = self.pool_cfg.N
        gk = np.zeros((B, Nn, self.pool_cfg.D_k))
        for i in range(B):
            for j, m in enumerate(idx[i]):
                cos = float(np.dot(qn[i], kn[m]))
                gk[i, j] = cfg.lam * (-(qn[i] - cos * kn[m]) / float(knorm[m, 0])) / Nn
        return dlog, zfeat, dz, idx, gk, loss

    def per_sample_norms(self, Q, y, task_id, class_mask=None) -> np.ndarray:
        """Норма полного градиента каждого снимка (для выбора порога отсечения)."""
        dlog, zfeat, dz, idx, gk, _ = self._per_sample_grads(Q, y, task_id, class_mask)
        denom = self.pool_cfg.N * self.pool_cfg.L_p
        sq = ((dlog ** 2).sum(1) * (zfeat ** 2).sum(1)          # голова W
              + (dlog ** 2).sum(1)                              # голова b
              + idx.shape[1] * self.pool_cfg.L_p * (dz ** 2).sum(1) / denom ** 2
              + (gk ** 2).sum(axis=(1, 2)))
        return np.sqrt(sq)

    def train_step_dpsgd(self, Q, y, task_id, class_mask, clip, z_noise,
                         divisor, noise_rng, group_clips=None) -> float:
        """
        Один шаг DP-SGD над ВСЕМИ обучаемыми параметрами (голова + пул).
        Голову тоже нужно учить приватно: она влияет на градиенты пула, и без
        этого гарантия для пула теряет смысл.

        1. градиент каждого снимка ограничивается по норме порогом clip;
        2. ограниченные градиенты складываются;
        3. добавляется гауссов шум N(0, (z_noise*clip)^2) — на весь пул и на
           строки головы текущих классов (строки замаскированных классов имеют
           нулевой градиент при любых данных, шум им не нужен и только портил бы
           старые классы);
        4. сумма делится на divisor (ожидаемый размер пакета q*N);
        5. шаг Adam — обработка уже зашумлённого градиента, гарантию не меняет.
        Пустой пакет (возможен при пуассоновской выборке) — шаг из одного шума.
        """
        cfg = self.train_cfg
        self._ensure_pool_opt()
        key = self._hkey(task_id)
        W, b = self.heads[key]
        M, Lp, D = self.pool_cfg.M, self.pool_cfg.L_p, self.pool_cfg.D
        denom = self.pool_cfg.N * Lp

        GW = np.zeros_like(W); Gb = np.zeros_like(b)
        GK = np.zeros_like(self.pool.keys); GP = np.zeros_like(self.pool.prompts)
        loss = float("nan")

        if len(Q) > 0:
            dlog, zfeat, dz, idx, gk, loss = self._per_sample_grads(Q, y, task_id, class_mask)
            self.usage += np.bincount(idx.ravel(), minlength=M)
            sqH = (dlog ** 2).sum(1) * (zfeat ** 2).sum(1) + (dlog ** 2).sum(1)
            sqP = idx.shape[1] * Lp * (dz ** 2).sum(1) / denom ** 2
            sqK = (gk ** 2).sum(axis=(1, 2))
            if group_clips is None:
                # общий порог на весь градиент снимка
                f = np.minimum(1.0, clip / (np.sqrt(sqH + sqP + sqK) + 1e-12))
                fH = fP = fK = f
            else:
                # свой порог для каждой группы параметров
                cH, cP, cK = group_clips
                fH = np.minimum(1.0, cH / (np.sqrt(sqH) + 1e-12))
                fP = np.minimum(1.0, cP / (np.sqrt(sqP) + 1e-12))
                fK = np.minimum(1.0, cK / (np.sqrt(sqK) + 1e-12))
            fd = dlog * fH[:, None]
            GW = fd.T @ zfeat
            Gb = fd.sum(axis=0)
            for i in range(len(Q)):
                c = fP[i] * dz[i] / denom
                for j, m in enumerate(idx[i]):
                    GP[m] += c                                    # broadcast по L_p
                    GK[m] += fK[i] * gk[i, j]

        if z_noise > 0:
            if group_clips is None:
                sH = sP = sK = z_noise * clip
            else:
                # шум группы g: z * sqrt(G) * C_g, G = 3 -> тот же множитель z
                g = np.sqrt(3.0)
                sH, sP, sK = (z_noise * g * group_clips[0], z_noise * g * group_clips[1],
                              z_noise * g * group_clips[2])
            rows = (np.ones(W.shape[0], dtype=bool) if class_mask is None
                    else np.asarray(class_mask, dtype=bool))
            GW[rows] += noise_rng.normal(0.0, sH, (int(rows.sum()), W.shape[1]))
            Gb[rows] += noise_rng.normal(0.0, sH, int(rows.sum()))
            GK += noise_rng.normal(0.0, sK, GK.shape)
            GP += noise_rng.normal(0.0, sP, GP.shape)

        GW /= divisor; Gb /= divisor; GK /= divisor; GP /= divisor
        optW, optb = self._head_opt[key]
        optW.step(W, GW, cfg.lr)
        optb.step(b, Gb, cfg.lr)
        optK, optP = self._pool_opt
        optK.step(self.pool.keys, GK, cfg.lr)
        optP.step(self.pool.prompts, GP, cfg.lr)
        return loss

    # ---------- оценка ----------
    def evaluate(self, X: np.ndarray, y: np.ndarray, task_id: int,
                 class_mask=None) -> float:
        if self._hkey(task_id) not in self.heads or len(X) == 0:
            return 0.0
        logits, _, _, _ = self.forward(X, task_id, class_mask)
        pred = np.argmax(logits, axis=1)
        return float((pred == y).mean())
