import os

_THIS_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_THIS_DIR, "test_data")
_DATA_DIR = os.path.abspath(os.path.realpath(_DATA_DIR))  # normalised


def abs_data_path(filename: str) -> str:
    return os.path.abspath(os.path.join(_DATA_DIR, filename))


SAMPLE_FRAGSTORE_PATH = abs_data_path("sample_brics.pkl")
SAMPLE_SMILES_FILE = abs_data_path("sample.smi")
