#!/usr/bin/env python3
import pytest
import numpy as np

from cqasim.cqa_gendata import gen_simplified_data, \
    draw_diameters, gaussian_1d, gen_p_data


def test_gen_p_data_shape_min():
    """Test that output shapes are correct with min correlation."""
    P, N, T, L = 2, 3, 100, 1000
    Zeta = 43
    diametro_m, diametro_delta = 1.0, 0.3
    height_m = 21
    height_delta = 0.43
    correlated_peaks = True
    gamma = 0.1

    data, diameter_per_nrn, heights_per_nrn, fields_per_nrn = gen_p_data(
        P, N, T, L, Zeta,
        diametro_m, diametro_delta,
        height_m, height_delta,
        correlated_peaks,
        gamma,
        correlated_dims="min",
        M_fixed=1,
        simplified=True,
        verbose=False,
        seed=42,
    )

    assert data.shape == (P, N, T)
    assert np.shape(diameter_per_nrn) == (P, N)
    assert np.shape(heights_per_nrn) == (P, N)
    assert np.shape(fields_per_nrn) == (P, N)
    for d in fields_per_nrn:
        assert np.all(d == 1)


def test_gen_p_data_shape_max():
    """Test that output shapes are correct with max correlation."""
    P, N, T, L = 2, 3, 100, 1000
    Zeta = 43
    diametro_m, diametro_delta = 1.0, 0.3
    height_m = 21
    height_delta = 0.43
    correlated_peaks = True
    gamma = 0.1
    data, diameter_per_nrn, heights_per_nrn, fields_per_nrn = gen_p_data(
        P, N, T, L, Zeta,
        diametro_m, diametro_delta,
        height_m, height_delta,
        correlated_peaks,
        gamma,
        correlated_dims="max",
        M_fixed=1,
        simplified=True,
        verbose=False,
        seed=42,
    )
    assert data.shape == (P, N, T)
    assert np.shape(diameter_per_nrn) == (P, N)
    assert np.shape(heights_per_nrn) == (P, N)
    assert np.shape(fields_per_nrn) == (P, N)


def test_data_shape():
    N, T, L = 5, 100, 10.0
    data, diameters, heights = gen_simplified_data(N, T, L, 0.1, 0.2, seed=42)
    assert data.shape == (N, T)
    assert len(diameters) == N
    assert len(heights) == N

def test_repeatability_with_seed():
    args = (5, 100, 10.0, 0.1, 0.2)
    data, diam, heights = gen_simplified_data(*args, seed=123)
    data2, diam2, heights2 = gen_simplified_data(*args, seed=123)
    np.testing.assert_array_almost_equal(data, data2)
    np.testing.assert_array_almost_equal(diam, diam2)
    np.testing.assert_array_almost_equal(heights, heights2)

def test_diameter_constraint():
    L = 10
    diameters = draw_diameters(0.1, 0.1, 3, L, seed=42)
    assert diameters is not None
    assert sum(diameters) <= L - 2 * len(diameters)

def test_gaussian_peak_location():
    N, T, L = 3, 90, 9.0
    data, *_ = gen_simplified_data(N, T, L, 0.1, 0.05, seed=1)
    peaks = np.argmax(data, axis=1)
    expected_centers = np.linspace(0, T, N, endpoint=False, dtype=int)

    for peak, center in zip(peaks, expected_centers):
        assert abs(peak - center) <= T // N // 2

def test_gaussian_1d_peak():
    x = np.linspace(-5, 5, 1000)
    y = gaussian_1d(x, radius=1.0)
    peak_index = np.argmax(y)
    assert abs(x[peak_index]) < 0.01
    assert np.isclose(y[peak_index], 1 / np.sqrt(2 * np.pi), atol=1e-4)
