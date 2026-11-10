"""Юнит-тесты гомоморфной агрегации CKKS (сервер без секретного ключа)."""
import numpy as np
import pytest

pytest.importorskip("tenseal")
from core.config import ExperimentConfig  # noqa: E402
from core.crypto import Aggregator        # noqa: E402


@pytest.fixture(scope="module")
def ckks():
    return Aggregator(ExperimentConfig(), mode="ckks")


def test_encrypted_average_matches_plain_average(ckks):
    rng = np.random.default_rng(1)
    updates = [rng.normal(0, 0.01, ExperimentConfig().pool.total_floats) for _ in range(3)]
    result = ckks.aggregate(updates)
    assert ckks.backend_name() == "ckks-tenseal"
    np.testing.assert_allclose(result, np.mean(updates, axis=0), atol=1e-6)


def test_server_has_no_secret_key(ckks):
    assert ckks.server_has_secret_key() is False


def test_pool_fits_in_three_ciphertexts(ckks):
    # 11 520 чисел пула при 4096 слотах на шифртекст -> 3 шифртекста
    rng = np.random.default_rng(2)
    ckks.aggregate([rng.normal(size=ExperimentConfig().pool.total_floats) for _ in range(3)])
    assert ckks.stats.ciphertexts_per_client_per_round == 3
