import random

import numpy as np
from frag_gt.src.fragmentors import fragmentor_factory
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.operators import substitute_node_mutation, add_node_mutation, delete_node_mutation, \
    single_point_crossover, connect_mol_from_frags
from frag_gt.src.query_builder import FragQueryBuilder
from frag_gt.tests.utils import SAMPLE_FRAGSTORE_PATH
from rdkit import Chem

# seed random functions as operators have stochastic behaviour
np.random.seed(1337)
random.seed(1337)


BRICS_FRAGMENTOR = fragmentor_factory("brics")
FRAGSTORE_DB = fragstore_factory("in_memory", SAMPLE_FRAGSTORE_PATH)
QUERY_BUILDER = FragQueryBuilder(FRAGSTORE_DB,
                                 scorer="counts",
                                 stochastic=True)
MOL1 = Chem.MolFromSmiles("CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)N3C=C(N=C3)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CC=C5")
MOL2 = Chem.MolFromSmiles("CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5")
MOL_NO_BRICS_FRAGS = Chem.MolFromSmiles("C/C(=N/NS(=O)(=O)c1ccc(Cl)cc1)c1ccc2c(c1)OCO2")


def test_substitute_node_mutation():
    # Given
    mol = Chem.Mol(MOL1)

    # When
    mutant = substitute_node_mutation(mol, BRICS_FRAGMENTOR, QUERY_BUILDER)

    # Then
    assert isinstance(Chem.MolToSmiles(mutant[0]), str)
    assert Chem.MolToSmiles(MOL1) == Chem.MolToSmiles(mol)  # original mol remains unchanged


def test_add_node_mutation():
    # Given
    mol = Chem.Mol(MOL1)

    # When
    mutant = add_node_mutation(mol, BRICS_FRAGMENTOR, QUERY_BUILDER)

    # Then
    assert isinstance(Chem.MolToSmiles(mutant[0]), str)
    assert Chem.MolToSmiles(MOL1) == Chem.MolToSmiles(mol)  # original mol remains unchanged


def test_add_node_mutation_no_brics_disconnections_afp_scorer():
    # Given
    mol = Chem.Mol(MOL_NO_BRICS_FRAGS)
    query_builder = FragQueryBuilder(FRAGSTORE_DB, scorer="afps", stochastic=True)

    # When
    mutant = add_node_mutation(mol, BRICS_FRAGMENTOR, query_builder)

    # Then
    assert isinstance(Chem.MolToSmiles(mutant[0]), str)


def test_delete_node_mutation():
    # Given
    mol = Chem.Mol(MOL1)

    # When
    mutant = delete_node_mutation(mol, BRICS_FRAGMENTOR, QUERY_BUILDER)

    # Then
    assert isinstance(Chem.MolToSmiles(mutant[0]), str)
    assert Chem.MolToSmiles(MOL1) == Chem.MolToSmiles(mol)  # original mol remains unchanged


def test_single_point_crossover():
    # Given
    mol1 = Chem.Mol(MOL1)
    mol2 = Chem.Mol(MOL2)

    # When
    new_mol1, new_mol2 = single_point_crossover(mol1, mol2, BRICS_FRAGMENTOR)

    # Then
    assert isinstance(Chem.MolToSmiles(new_mol1), str)
    assert isinstance(Chem.MolToSmiles(new_mol2), str)
    assert Chem.MolToSmiles(MOL1) == Chem.MolToSmiles(mol1)  # original mol remains unchanged
    assert Chem.MolToSmiles(MOL2) == Chem.MolToSmiles(mol2)  # original mol remains unchanged


def test_connect_mol_from_frags_brics():
    # Given
    mollist = [
        "c1ccccc1C=Cc2ccccc2",
        "CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)N3C=C(N=C3)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CC=C5"
    ]
    mollist = [Chem.MolFromSmiles(m, sanitize=True) for m in mollist]
    mollist_frags = [BRICS_FRAGMENTOR.get_frags(m) for m in mollist]

    # When
    reconstructed_mols = [connect_mol_from_frags(frags, fragmentor=BRICS_FRAGMENTOR) for frags in mollist_frags]

    # Then
    for x, y in zip([Chem.MolToSmiles(m) for m in mollist], [Chem.MolToSmiles(m) for m in reconstructed_mols]):
        assert x == y


def test_connect_mol_from_frags_atom_props_preserved():
    # Given
    mol = Chem.MolFromSmiles("c1ccccc1C=Cc2ccccc2", sanitize=True)
    frags = BRICS_FRAGMENTOR.get_frags(mol)
    atom1 = frags[0].GetAtomWithIdx(0)
    atom1.SetProp('test_prop', '123')

    # When
    reconstructed_mol = connect_mol_from_frags(frags, fragmentor=BRICS_FRAGMENTOR)

    # Then
    found = False
    for a in reconstructed_mol.GetAtoms():
        if a.HasProp('test_prop'):
            found = True
    assert found
