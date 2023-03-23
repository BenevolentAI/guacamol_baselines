from frag_gt.src.frag_scorers import FragScorer
from rdkit import Chem


FRAG_COUNT_TUPLES = [("[2*]Cc1cc(O)cc(O[4*])c1", 2),
                     ("[2*]CC(=N[4*])C(C)(C)C", 8),
                     ("[2*]CC(N[4*])C(C)C", 1)]


def test_sampling():
    # Given
    fragment_sampler = FragScorer(scorer="random", sort=False)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_sampler.score(gene_frag_list=FRAG_COUNT_TUPLES)

    # Then
    assert len(scored_frags) == len(FRAG_COUNT_TUPLES)


def test_sampling_counts_scorer_and_sort():
    # Given
    fragment_sampler = FragScorer(scorer="counts", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_sampler.score(gene_frag_list=FRAG_COUNT_TUPLES, query_frag=query_frag)

    # Then
    frag_smiles, scores = zip(*scored_frags)
    _, original_counts = zip(*FRAG_COUNT_TUPLES)
    assert scores == tuple(sorted(original_counts, reverse=True))


def test_sampling_afp_scorer():
    # Given
    fragment_sampler = FragScorer(scorer="afps", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_sampler.score(gene_frag_list=FRAG_COUNT_TUPLES[:1], query_frag=query_frag)

    # Then
    assert len(scored_frags) == 1


def test_sampling_ecfp4_scorer():
    # Given
    fragment_sampler = FragScorer(scorer="ecfp4", sort=True)
    query_frag = Chem.MolFromSmiles("[2*]CC(N[4*])C(C)C")

    # When
    scored_frags = fragment_sampler.score(gene_frag_list=FRAG_COUNT_TUPLES, query_frag=query_frag)

    # Then
    assert len(scored_frags) == 3
