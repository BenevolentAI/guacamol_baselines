import os
import random

import numpy as np
from rdkit import Chem

from frag_gt.frag_gt import FragGTGenerator
from frag_gt.src.population import Molecule
from frag_gt.src.scorers import MolecularWeightScorer
from frag_gt.tests.utils import SAMPLE_SMILES_FILE, SAMPLE_FRAGSTORE_PATH

np.random.seed(1337)
random.seed(1337)


def test_fraggt_generator_e2e(tmpdir):
    # Given
    np.random.seed(1337)
    random.seed(1337)
    n_generations = 3
    number_of_requested_molecules = 10
    optimizer = FragGTGenerator(smi_file=SAMPLE_SMILES_FILE,
                                fragmentation_scheme="brics",
                                fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                allow_unspecified_stereo=False,
                                operators=None,  # use default operators
                                population_size=12,  # short run with small population
                                n_mutations=5,
                                generations=n_generations,
                                n_jobs=1,
                                random_start=True,
                                patience=5,
                                intermediate_results_dir=tmpdir)
    scoring_function = MolecularWeightScorer()
    job_name = "e2e_test"

    # When
    output_smis = optimizer.optimize(scoring_function=scoring_function,
                                     number_molecules=number_of_requested_molecules,
                                     starting_population=None,
                                     job_name=job_name)

    # Then
    assert len(output_smis) == number_of_requested_molecules

    intermediate_outfiles = set([str(os.path.basename(x)) for x in tmpdir.listdir()])
    expected_outfiles = set([f"{job_name}_{i}.csv" for i in range(n_generations + 1)])
    assert intermediate_outfiles == expected_outfiles


def test_fraggt_generator_custom_initial_population(tmpdir):
    # Given
    np.random.seed(1337)
    random.seed(1337)
    starting_population = ["c1ccccc1"]
    population_size = 10
    number_of_requested_molecules = 3
    optimizer = FragGTGenerator(smi_file=SAMPLE_SMILES_FILE,
                                fragmentation_scheme="brics",
                                fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                allow_unspecified_stereo=False,
                                operators=None,  # use default operators
                                population_size=population_size,  # short run with small population
                                n_mutations=5,
                                generations=2,
                                n_jobs=1,
                                random_start=True,
                                patience=5,
                                intermediate_results_dir=tmpdir)
    scoring_function = MolecularWeightScorer()
    job_name = "e2e_test"

    # When
    output_smis = optimizer.optimize(scoring_function=scoring_function,
                                     number_molecules=number_of_requested_molecules,
                                     starting_population=starting_population,
                                     job_name=job_name)

    # Then
    assert len(output_smis) == number_of_requested_molecules,\
        f"expected {number_of_requested_molecules} SMILES, got: {output_smis}"

    # first intermediate outfile should only have one molecule
    intermediate_outfiles = tmpdir.listdir()
    found = False
    for f in intermediate_outfiles:
        if str(f).endswith(f"{job_name}_0.csv"):
            smiles = f.readlines()
            assert len(smiles) == len(starting_population) + 1, "initial gen"  # +1 for title
            found = True
            break
    assert found

    # rest should have `population_size`
    # except this isn't true because there can be redundancy in the mols generated, and then duplicates are removed
    # found = False
    # for f in intermediate_outfiles:
    #     if str(f).endswith(f"{job_name}_1.csv"):
    #         smiles = f.readlines()
    #         assert len(smiles) == population_size + 1, 'first gen'  # +1 for title
    #         found = True
    #         break
    # assert found


def test_fraggt_generator_mapelites():
    # Given
    np.random.seed(1337)
    random.seed(1337)
    n_generations = 3
    number_of_requested_molecules = 10
    optimizer = FragGTGenerator(smi_file=SAMPLE_SMILES_FILE,
                                fragmentation_scheme="brics",
                                fragstore_path=SAMPLE_FRAGSTORE_PATH,
                                allow_unspecified_stereo=False,
                                operators=None,  # use default operators
                                population_size=12,  # short run with small population
                                n_mutations=5,
                                generations=n_generations,
                                map_elites="species",
                                n_jobs=1,
                                random_start=True,
                                patience=5)
    scoring_function = MolecularWeightScorer()

    # When
    output_smis = optimizer.optimize(scoring_function=scoring_function,
                                     number_molecules=number_of_requested_molecules,
                                     starting_population=None)

    # Then
    assert len(output_smis) == number_of_requested_molecules


def test_duplicate():
    smis = ['Clc1ccccc1', 'c1ccccc1Cl', 'c1cc(Cl)ccc1', 'CCCBr']
    mollist = [Molecule(0, Chem.MolFromSmiles(s)) for s in smis]
    deduped = FragGTGenerator.deduplicate(mollist)
    assert len(deduped) == 2
