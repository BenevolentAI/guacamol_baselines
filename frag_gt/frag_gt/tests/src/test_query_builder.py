from rdkit import Chem

from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.query_builder import FragQueryBuilder
from frag_gt.tests.utils import SAMPLE_FRAGSTORE_PATH

FRAGSTORE_DB = fragstore_factory("in_memory", SAMPLE_FRAGSTORE_PATH)


def test_query_builder():
    # given
    fragstore_qb = FragQueryBuilder(FRAGSTORE_DB,
                                    scorer="counts",
                                    stochastic=True)
    query_frag = Chem.MolFromSmiles("[16*]c1ccccc1")
    gene_type = "16"

    # When
    frags, counts = fragstore_qb.query_frags(gene_type, query_frag)

    # Then
    assert len(frags)
    assert len(frags) == len(counts)


def test_query_builder_invalid_gene_type():
    # Given
    fragstore_qb = FragQueryBuilder(FRAGSTORE_DB,
                                    scorer="counts",
                                    stochastic=True)
    query_frag = Chem.MolFromSmiles("[16*]c1ccccc1")
    gene_type = "16#INVALID_GENE_TYPE"

    # When
    frags, counts = fragstore_qb.query_frags(gene_type, query_frag)

    # Then
    assert not len(frags)
