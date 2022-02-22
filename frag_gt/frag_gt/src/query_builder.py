import logging

from rdkit import Chem
from typing import Optional, List, Tuple

from frag_gt.src.fragstore import FragStoreBase
from frag_gt.src.sampling import FragSampler

logger = logging.getLogger(__name__)


class FragQueryBuilder:
    """
    This class is used to communicate with a fragment store to retrieve fragments suitable to replace a query fragment.
    Candidate replacement fragments are identified by their gene type,
    and can be optionally sampled according to distribution of "scores": random | count | ecfp4 | afps
    """
    def __init__(self, frag_store: FragStoreBase, scorer: str = "random", stochastic: Optional[bool] = False):
        self.frag_sampler = FragSampler(scorer=scorer, stochastic=stochastic)
        self.db = frag_store
        self.db.load()

    def query_frags(self, gene_type: str, ref_frag: Optional[Chem.rdchem.Mol] = None) -> Tuple[List[str], List[float]]:
        """

        Args:
           gene_type: gene type to query fragstore with (e.g. "5#5")
           ref_frag: (optional) query mol to guide mol-dependent sampling methods (not used by "counts" or "random")

        Returns:
           A list of SMILES strings for the generated molecules.
        """

        if gene_type == "":
            logger.debug(f"empty gene_type: {Chem.MolToSmiles(ref_frag)} Skipping mutation")
            return [], []

        logger.debug(f"Finding genes with gene_type: {gene_type}")

        # get pool of haplotype fragment replacements
        # this returns either an empty list (if gene_type not in fragstore), or a nested iterable i.e. ([],)
        gene_type_frags = list(self.db.get_records("gene_types", {"gene_type": gene_type}))
        logger.debug(f"Possible genes (fragments) with gene_type {gene_type}: {len(gene_type_frags)}")

        if len(gene_type_frags) == 0:
            return [], []
        elif len(gene_type_frags) > 1:
            raise RuntimeError(f"More than one gene_type record in FragStore for {gene_type}, something is corrupted.")

        # filter for genes within desired property range
        # returns  tuples of (smiles, count in corpus)
        genes = []  # type: List[Tuple[str, int]]
        for hap, record in gene_type_frags[0]["haplotypes"].items():
            genes.extend([(s, atts["count"]) for s, atts in record["gene_frags"].items()])

        smiles, scores = self.frag_sampler(genes, n_choices=-1, query_frag=ref_frag)
        return smiles, scores
