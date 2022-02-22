from frag_gt.fragstore_scripts.filter_fragstore import filter_fragstore
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.tests.utils import SAMPLE_FRAGSTORE_PATH

SAMPLE_FRAGSTORE = fragstore_factory("in_memory", SAMPLE_FRAGSTORE_PATH)
SAMPLE_FRAGSTORE.load()


def test_filter_fragstore():
    # Given
    old_fragstore = SAMPLE_FRAGSTORE.store

    # When
    new_fragstore = filter_fragstore(old_fragstore, count_limit=2)

    # Then
    assert len(old_fragstore["gene_types"]) > len(new_fragstore["gene_types"])


def test_null_filter():
    # Given
    old_fragstore = SAMPLE_FRAGSTORE.store

    # When
    new_fragstore = filter_fragstore(old_fragstore, count_limit=1)

    # Then
    # everything in the fragstore should have a count of at least 1 so nothing should be filtered
    assert len(old_fragstore["gene_types"]) == len(new_fragstore["gene_types"])
