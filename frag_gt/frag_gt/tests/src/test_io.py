from frag_gt.src.io import load_smiles_from_file, valid_mols_from_smiles
from frag_gt.tests.utils import SAMPLE_SMILES_FILE


def test_load_smiles_from_file():
    smiles = load_smiles_from_file(SAMPLE_SMILES_FILE)
    assert len(smiles) == 100


def test_valid_mols_from_smiles():
    smiles = load_smiles_from_file(SAMPLE_SMILES_FILE)
    valid_mols = valid_mols_from_smiles(smiles, n_jobs=1)
    assert len(valid_mols) == 100


def test_valid_mols_from_smiles_parallel():
    smiles = load_smiles_from_file(SAMPLE_SMILES_FILE)
    valid_mols = valid_mols_from_smiles(smiles, n_jobs=2)
    assert len(valid_mols) == 100


def test_valid_mols_from_smiles_invalid_mols():
    smiles = ['c1ccccc1', 'invalidmol', 'CCCCC']
    valid_mols = valid_mols_from_smiles(smiles)
    assert len(valid_mols) == 2
