#!/usr/bin/env python3
import pytest
from pathlib import Path
from src.cqa_defaults import CqaDefaults


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
    expected = base / "reali_data" / "g17.0_kb300_step0.08" / "P1_N999" / "T1000_L200"
    assert result == expected


def test_build_fn(defaults_obj):
    """Test file name string construction."""
    fn = defaults_obj.build_fn()
    expected_parts = [
        "fieldM1", "heightM1.549", "diaM1.57", "diaVar2.0",
        "corrP0", "gamma0.5", "seed1"
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
