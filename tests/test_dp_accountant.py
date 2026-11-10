"""Юнит-тесты счётчика приватности DP-SGD."""
import pytest

from core.dp_accountant import epsilon, noise_for_epsilon


@pytest.mark.parametrize("q, z, steps, expected", [
    (0.064, 1.0, 150, 6.1181),          # эталон: библиотека Google dp_accounting
    (256 / 60000, 1.1, 14062, 2.5970),  # классический пример DP-SGD на MNIST
])
def test_epsilon_matches_reference(q, z, steps, expected):
    eps, _ = epsilon(q, z, steps, 1e-5)
    assert eps == pytest.approx(expected, abs=1e-3)


def test_more_noise_gives_smaller_epsilon():
    eps_small_noise, _ = epsilon(0.064, 1.0, 150, 1e-5)
    eps_big_noise, _ = epsilon(0.064, 2.0, 150, 1e-5)
    assert eps_big_noise < eps_small_noise


def test_noise_for_epsilon_reaches_target():
    z = noise_for_epsilon(0.064, 150, 1e-5, 2.0)
    eps, _ = epsilon(0.064, z, 150, 1e-5)
    assert eps <= 2.0 + 1e-6
