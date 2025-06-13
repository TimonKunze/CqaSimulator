#!/usr/bin/env python3
import pytest

from pathlib import Path
from cqasim.cqa_defaults import CqaDefaults


class DummyWithDefaults(CqaDefaults):
    """Helper subclass to expose .par property like in real use."""
    @property
    def par(self):
        return self.default_par

    @par.setter
    def par(self, value):
        self.default_par = value


@pytest.fixture
def defaults_obj():
    return DummyWithDefaults()


def test_initialization(defaults_obj):
    """Test default parameter initialization."""
    assert isinstance(defaults_obj.default_par, dict)
    assert "P" in defaults_obj.default_par
    assert defaults_obj.default_par["g"] == 17.0


def test_build_fp(defaults_obj):
    """Test folder path construction from parameters."""
    base = Path("/base/path")

    result = defaults_obj.build_fp(base)

    fp = base / "reali_data"
    fp = fp / f"g{17.0}_kb{300}_step{0.08}"
    fp = fp / f"P{1}_N{999}"
    fp = fp / f"T{1000}_L{200}"
    fp = fp / f"zetaM{4.4}_fixedM{None}"
    fp = fp / f"corrP{1}_gamma{0.5}"
    fp = fp / f"heightM{1.549}_heightVar{0.0}"
    fp = fp / f"diaM{1.57}_diaVar{0.0}"
    expected = fp

    assert result == expected


def test_build_fn(defaults_obj):
    """Test file name string construction."""
    fn = defaults_obj.build_fn()
    expected_parts = [
        # "corrP0",
        "seed1"
    ]
    for part in expected_parts:
        assert part in fn
    assert fn.endswith("seed1")


def test_str_and_repr(defaults_obj):
    """Test string representations."""
    s = str(defaults_obj)
    r = repr(defaults_obj)
    assert "CqaModel(par=" in s
    assert "CqaModel(par=" in r
