#!/usr/bin/env python3
import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from cqasim.cqa_fileloader import CqaFileLoader, ObservableDict


@pytest.fixture
def mock_data_folder():
    """Creates a temporary folder with mock files."""
    temp_dir = tempfile.mkdtemp()

    # Simulate the nested folder structure
    nested_path = Path(temp_dir) / "reali_data" / "g17.0_kb300_step0.08" / "P1_N999" / "T1000_L200"
    nested_path.mkdir(parents=True)

    # Create a dummy .npy file
    dummy_array = np.array([1, 2, 3])
    np.save(nested_path / "data_fieldM1.npy", dummy_array)

    yield Path(temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir)


def test_initialization(mock_data_folder):
    fl = CqaFileLoader(paths=mock_data_folder)

    # Ensure defaults exist
    assert isinstance(fl.par, ObservableDict)
    assert "P" in fl.par
    print(fl.path)
    assert fl.path is not None


def test_path_construction(mock_data_folder):
    fl = CqaFileLoader(paths=mock_data_folder)

    fp = Path("reali_data")
    fp = fp / f"g{17.0}_kb{300}_step{0.08}"
    fp = fp / f"P{1}_N{999}"
    fp = fp / f"T{1000}_L{200}"
    fp = fp / f"zetaM{4.4}_fixedM{None}"
    fp = fp / f"corrP{int(1)}_gamma{0.5}"
    fp = fp / f"heightM{1.549}_heightVar{0.0}"
    fp = fp / f"diaM{1.57}_diaVar{0.0}"

    expected_path = mock_data_folder / fp

    assert fl.path == expected_path
    # assert expected_path.exists() # Why should I test this?


# def test_file_loading(mock_data_folder):
#     fl = CqaFileLoader(paths=mock_data_folder)
#     print(fl.files)
#     assert "data" in fl.files

#     np.testing.assert_array_equal(fl.files["data"], np.array([1, 2, 3]))
#     assert hasattr(fl, "data")  # Dynamic attribute


def test_parameter_update_triggers_path(mock_data_folder):
    fl = CqaFileLoader(paths=mock_data_folder)
    fl.par["simplified_data"] = True  # Should change the subfolder
    assert "simpl_data" in str(fl.path)


def test_str_repr(mock_data_folder):
    fl = CqaFileLoader(paths=mock_data_folder)
    str_out = str(fl)
    repr_out = repr(fl)
    assert "SimulationFileLoader" in repr_out
    assert fl.path.as_posix() in str_out
