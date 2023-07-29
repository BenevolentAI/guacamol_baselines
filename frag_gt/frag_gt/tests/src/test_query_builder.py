from rdkit import Chem

from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.query_builder import FragQueryBuilder
from frag_gt.tests.utils import SAMPLE_FRAGSTORE_PATH

FRAGSTORE_DB = fragstore_factory("in_memory", SAMPLE_FRAGSTORE_PATH)


def test_query_builder():
    # given
    fragstore_qb = FragQueryBuilder(FRAGSTORE_DB,
                                    scorer="counts",
                                    sort_by_score=False)
    query_frag = Chem.MolFromSmiles("[16*]c1ccccc1")
    gene_type = "16"

    # When
    frags, counts = fragstore_qb.query_frags(gene_type, query_frag)

    # Then
    assert len(frags)
    assert len(frags) == len(counts)


def test_query_builder_x_choices():
    # given
    fragstore_qb = FragQueryBuilder(FRAGSTORE_DB,
                                    scorer="counts",
                                    sort_by_score=False)
    query_frag = Chem.MolFromSmiles("[16*]c1ccccc1")
    gene_type = "16"
    test_cases = {
        'x_choice_input': [1, 0., 2, 0.5, 1000, 1.],
        'expected': [1, 1, 2, 28, 56, 56]
    }

    # When
    for inp, expected_n in zip(test_cases['x_choice_input'], test_cases['expected']):
        frags, _ = fragstore_qb.query_frags(gene_type, query_frag, x_choices=inp)
        assert len(frags) == expected_n


def test_query_builder_invalid_gene_type():
    # Given
    fragstore_qb = FragQueryBuilder(FRAGSTORE_DB,
                                    scorer="counts",
                                    sort_by_score=True)
    query_frag = Chem.MolFromSmiles("[16*]c1ccccc1")
    gene_type = "16#INVALID_GENE_TYPE"

    # When
    frags, counts = fragstore_qb.query_frags(gene_type, query_frag)

    # Then
    assert not len(frags)
