from frag_gt.src.fragmentors import fragmentor_factory
from rdkit import Chem


def test_brics_fragmentation():
    # Given
    parent_smiles = ["CCSc1nnc(NC(=O)CCCOc2ccc(C)cc2)s1", "CCCC(=O)NNC(=O)Nc1ccccc1"]
    parent_smiles = [Chem.MolFromSmiles(parent) for parent in parent_smiles]

    # When
    fragmentor = fragmentor_factory("brics")
    brics_frags_mols = fragmentor.get_frags(parent_smiles[1])

    # Then
    assert fragmentor.name == "brics"
    expected_frags = ["[1*]C(=O)NNC(=O)CCC", "[5*]N[5*]", "[16*]c1ccccc1"]
    brics_frags_smiles = [Chem.MolToSmiles(f) for f in brics_frags_mols]
    assert brics_frags_smiles == expected_frags
