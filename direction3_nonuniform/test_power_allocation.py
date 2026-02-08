"""
Unit tests for power_allocation.py.
Run: python test_power_allocation.py
"""

import numpy as np

from power_allocation import assign_power_heterogeneous, assign_power_centralized


def test_power_balance():
    for sigma_ratio in [0, 0.1, 0.3, 0.5, 0.8]:
        for side in ["gen", "con"]:
            p = assign_power_heterogeneous(25, 25, 1.0, sigma_ratio, side)
            assert abs(p.sum()) < 1e-10, (
                f"FAIL: side={side}, sigma_ratio={sigma_ratio}, sum={p.sum():.2e}"
            )
    for r in [0.04, 0.3, 0.7, 0.95]:
        p = assign_power_centralized(25, 25, 1.0, r)
        assert abs(p.sum()) < 1e-10, f"FAIL: r={r}, sum={p.sum():.2e}"
    print("PASS test_power_balance")


def test_sign_convention():
    for sigma_ratio in [0, 0.3, 0.8]:
        for side in ["gen", "con"]:
            p = assign_power_heterogeneous(25, 25, 1.0, sigma_ratio, side)
            assert np.all(p[:25] > 0), f"FAIL: gen not positive, side={side}, sigma_ratio={sigma_ratio}"
            assert np.all(p[25:] < 0), f"FAIL: con not negative, side={side}, sigma_ratio={sigma_ratio}"
    print("PASS test_sign_convention")


def test_total_power():
    for sigma_ratio in [0, 0.5]:
        for side in ["gen", "con"]:
            p = assign_power_heterogeneous(25, 25, 1.0, sigma_ratio, side)
            assert abs(p[:25].sum() - 1.0) < 1e-10, f"FAIL total gen: {p[:25].sum()}"
            assert abs(p[25:].sum() + 1.0) < 1e-10, f"FAIL total con: {p[25:].sum()}"
    print("PASS test_total_power")


def test_baseline_uniform():
    p = assign_power_heterogeneous(25, 25, 1.0, 0, "gen")
    assert np.allclose(p[:25], 0.04), "FAIL: sigma=0 gen not uniform"
    assert np.allclose(p[25:], -0.04), "FAIL: sigma=0 con not uniform"
    print("PASS test_baseline_uniform")


def test_centralized_baseline():
    p = assign_power_centralized(25, 25, 1.0, 1 / 25)
    assert np.allclose(p[:25], 0.04, atol=1e-10), "FAIL: r=1/n_plus not uniform"
    print("PASS test_centralized_baseline")


def test_centralized_extreme():
    p = assign_power_centralized(25, 25, 1.0, 0.95)
    assert abs(p[0] - 0.95) < 1e-12, f"FAIL: p_big={p[0]}, expected 0.95"
    expected_small = 0.05 / 24
    assert np.allclose(p[1:25], expected_small), "FAIL: small stations wrong"
    print("PASS test_centralized_extreme")


def test_heterogeneity_increases_variance():
    var_prev = 0.0
    for sigma_ratio in [0, 0.2, 0.5, 0.8]:
        variances = []
        for _ in range(50):
            p = assign_power_heterogeneous(25, 25, 1.0, sigma_ratio, "gen")
            variances.append(np.var(p[:25]))
        var_mean = float(np.mean(variances))
        assert var_mean >= var_prev - 1e-12, (
            f"FAIL: variance not monotone at sigma_ratio={sigma_ratio}"
        )
        var_prev = var_mean
    print("PASS test_heterogeneity_increases_variance")


def test_reproducibility():
    np.random.seed(42)
    p1 = assign_power_heterogeneous(25, 25, 1.0, 0.5, "gen")
    np.random.seed(42)
    p2 = assign_power_heterogeneous(25, 25, 1.0, 0.5, "gen")
    assert np.allclose(p1, p2), "FAIL: reproducibility"
    print("PASS test_reproducibility")


def test_vector_length():
    p = assign_power_heterogeneous(25, 25, 1.0, 0.5, "gen")
    assert len(p) == 50, f"FAIL: len(p)={len(p)}"
    p = assign_power_centralized(25, 25, 1.0, 0.5)
    assert len(p) == 50, f"FAIL: len(p)={len(p)}"
    print("PASS test_vector_length")


if __name__ == "__main__":
    print("=" * 50)
    print("Running unit tests for power_allocation.py")
    print("=" * 50)
    tests = [
        test_power_balance,
        test_sign_convention,
        test_total_power,
        test_baseline_uniform,
        test_centralized_baseline,
        test_centralized_extreme,
        test_heterogeneity_increases_variance,
        test_reproducibility,
        test_vector_length,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
