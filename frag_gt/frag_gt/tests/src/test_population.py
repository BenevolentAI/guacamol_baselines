import numpy as np
import random
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.population import MolecularPopulationGenerator, Molecule
from frag_gt.tests.utils import SAMPLE_FRAGSTORE_PATH, SAMPLE_SMILES_FILE
from rdkit import Chem

SAMPLE_FRAGSTORE = fragstore_factory("in_memory", SAMPLE_FRAGSTORE_PATH)
SAMPLE_FRAGSTORE.load()

# seed random functions as operators have stochastic behaviour
np.random.seed(1337)
random.seed(1337)


def _scored_population():
    """ read sample smiles and convert to mols as current_population """
    with open(SAMPLE_SMILES_FILE, 'r') as f:
        smiles = [x.strip() for x in f]
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    dummy_scores = list(range(len(molecules)))
    current_pool = [Molecule(*m) for m in zip(dummy_scores, molecules)]
    return current_pool


def test_population_generate():
    # Given
    n_molecules_to_generate = 10
    mol_generator = MolecularPopulationGenerator(fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                                 fragmentation_scheme="brics",
                                                 n_molecules=n_molecules_to_generate,
                                                 operators=None,
                                                 allow_unspecified_stereo=True,
                                                 selection_method="random")
    current_pool = _scored_population()

    # When
    new_pool = mol_generator.generate(current_pool)

    # Then

    # since crossover adds multiple, and stereo adds multiple, we do not guarantee that population size is exact
    # next tests shows a case where its possible to exactly generate a population by removing those factors
    assert len(new_pool) >= n_molecules_to_generate

    # this is now true, generate inputs population and outputs mol objects
    # assert isinstance(current_pool[0], type(new_pool[0])), "inputs and outputs have different types"


def test_population_generate_custom_operators():
    # Given
    n_molecules_to_generate = 10
    mol_generator = MolecularPopulationGenerator(fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                                 fragmentation_scheme="brics",
                                                 n_molecules=n_molecules_to_generate,
                                                 operators=[("substitute_node_mutation", 1.)],
                                                 allow_unspecified_stereo=True,
                                                 selection_method='random')
    current_pool = _scored_population()

    # When
    new_pool = mol_generator.generate(current_pool)

    # Then
    assert len(new_pool) == n_molecules_to_generate


def test_tournament_selection():
    # Given
    np.random.seed(1337)
    random.seed(1337)
    current_pool = _scored_population()

    # When
    fittest = MolecularPopulationGenerator.tournament_selection(current_pool, k=5)

    # Then
    assert int(fittest.score) == 92


def test_population_generate_tournament_selection():
    # Given
    n_molecules_to_generate = 10
    mol_generator = MolecularPopulationGenerator(fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                                 fragmentation_scheme="brics",
                                                 n_molecules=n_molecules_to_generate,
                                                 operators=None,
                                                 allow_unspecified_stereo=True,
                                                 selection_method="tournament")
    current_pool = _scored_population()

    # When
    new_pool = mol_generator.generate(current_pool)

    # Then
    assert len(new_pool) >= n_molecules_to_generate


def test_population_generate_fixed_substructure_pyrazole():
    # Given
    baricitinib = "CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3"
    pyrazole = "c1cn[nH]c1"
    current_pool = [Molecule(1., Chem.MolFromSmiles(baricitinib))]
    n_molecules_to_generate = 10

    # When
    mol_generator = MolecularPopulationGenerator(fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                                 fragmentation_scheme="brics",
                                                 n_molecules=n_molecules_to_generate,
                                                 operators=None,
                                                 allow_unspecified_stereo=True,
                                                 selection_method="tournament",
                                                 fixed_substructure_smarts=pyrazole)
    new_pool = mol_generator.generate(current_pool)

    # Then
    patt = Chem.MolFromSmarts(pyrazole)
    assert all([m.HasSubstructMatch(patt) for m in new_pool])


def test_population_generate_fixed_substructure_impossible_pattern():
    # Given
    baricitinib = "CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3"
    baricitinib_core_scaffold_smiles = "[N]1C=C(C=N1)C3=C2C=C[N]C2=NC=N3"
    current_pool = [Molecule(1., Chem.MolFromSmiles(baricitinib))]
    n_molecules_to_generate = 10

    # When
    mol_generator = MolecularPopulationGenerator(fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                                 fragmentation_scheme="brics",
                                                 n_molecules=n_molecules_to_generate,
                                                 operators=None,
                                                 allow_unspecified_stereo=True,
                                                 selection_method="tournament",
                                                 fixed_substructure_smarts=baricitinib_core_scaffold_smiles,
                                                 patience=100)
    new_pool = mol_generator.generate(current_pool)

    # Then

    # generator is unable to generate molecules for this fixed scaffold given the limited size of the sample fragstore
    # this checks that when no molecules can be generated, we dont fall into an infinite loop
    assert new_pool == []
