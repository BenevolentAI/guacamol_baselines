from frag_gt.src.io import load_smiles_from_file, valid_mols_from_smiles
from frag_gt.src.mapelites import MWLogPMapElites, SpeciesMapElites
from frag_gt.src.population import Molecule
from frag_gt.src.scorers import MolecularWeightScorer
from frag_gt.tests.utils import SAMPLE_SMILES_FILE


def test_mapelites_species():
    smiles = load_smiles_from_file(SAMPLE_SMILES_FILE)
    valid_mols = valid_mols_from_smiles(smiles, n_jobs=1)
    scoring_function = MolecularWeightScorer()
    scores = scoring_function.score_list(smiles)
    population = [Molecule(*m) for m in zip(scores, valid_mols)]
    mapelites = SpeciesMapElites(fragmentor="brics")
    new_population = mapelites.place_in_map(population)
    assert len(new_population)
    assert len(new_population) < len(population)


def test_mapelites_mwlogp():
    smiles = load_smiles_from_file(SAMPLE_SMILES_FILE)
    valid_mols = valid_mols_from_smiles(smiles, n_jobs=1)
    scoring_function = MolecularWeightScorer()
    scores = scoring_function.score_list(smiles)
    population = [Molecule(*m) for m in zip(scores, valid_mols)]
    mapelites = MWLogPMapElites(mw_step_size=25, logp_step_size=0.25)
    new_population = mapelites.place_in_map(population)
    assert len(new_population)
    assert len(new_population) < len(population)
