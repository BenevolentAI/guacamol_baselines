from frag_gt.src.sampling import FragSampler
from rdkit import Chem


FRAG_COUNT_TUPLES = [("[2*]Cc1cc(O)cc(O[4*])c1", 2),
                     ("[2*]CC(=N[4*])C(C)(C)C", 8),
                     ("[2*]CC(N[4*])C(C)C", 1)]


def test_sampling():
    # Given
    fragment_sampler = FragSampler(scorer="random", stochastic=False)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    smiles, scores = fragment_sampler(FRAG_COUNT_TUPLES, -1)

    # Then
    assert len(smiles) == len(FRAG_COUNT_TUPLES)


def test_sampling_n_choices():
    # Given
    fragment_sampler = FragSampler(scorer="counts", stochastic=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    smiles, scores = fragment_sampler(FRAG_COUNT_TUPLES, 1)

    # Then
    assert len(smiles) == 1


def test_sampling_afp_scorer():
    # Given
    fragment_sampler = FragSampler(scorer="afps", stochastic=False)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    smiles, scores = fragment_sampler(FRAG_COUNT_TUPLES, 1, query_frag)

    # Then
    assert len(smiles) == 1


def test_sampling_ecfp4_scorer():
    # Given
    fragment_sampler = FragSampler(scorer="ecfp4", stochastic=False)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    smiles, scores = fragment_sampler(FRAG_COUNT_TUPLES, 1, query_frag)

    # Then
    assert len(smiles) == 1
