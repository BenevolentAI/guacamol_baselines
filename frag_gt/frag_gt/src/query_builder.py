import logging
from typing import Optional, List, Tuple

import numpy as np
from rdkit import Chem

from frag_gt.src.frag_scorers import FragScorer
from frag_gt.src.fragstore import FragStoreBase

logger = logging.getLogger(__name__)


class FragQueryBuilder:
    """
    This class is used to communicate with a fragment store to retrieve fragments to replace a given reference fragment.
    Candidate replacement fragments are identified by their gene type,
    if desired a random sample is taken and fragments are scored by either random | count | ecfp4 | afps
    """
    def __init__(self, frag_store: FragStoreBase, scorer: str = "random", sort_by_score: bool = False):
        self.frag_sampler = FragScorer(scorer=scorer, sort=sort_by_score)
        self.db = frag_store
        self.db.load()

    def query_frags(self, gene_type: str,
                    ref_frag: Optional[Chem.rdchem.Mol] = None,
                    n_choices: int = -1,
                    ) -> Tuple[List[str], List[float]]:
        """

        Args:
           gene_type: gene type to query fragstore with (e.g. "5#5")
           ref_frag: (optional) query mol to guide mol-dependent sampling methods (not used by "counts" or "random")
           n_choices: (optional) number of random matched frags to return, default -1 returns all available

        Returns:
           A list of SMILES strings for the generated molecules
           A list of scores for those molecules (if sort_by_score=False, these are unsorted!)
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

        # unzip results to a list of tuples of (smiles, count in corpus)
        genes = []  # type: List[Tuple[str, int]]
        for hap, record in gene_type_frags[0]["haplotypes"].items():
            # if there are properties saved in the fragstore, we can filter on those here
            # e.g. mw range around `ref_frag` (if mw in fragstore)
            genes.extend([(s, atts["count"]) for s, atts in record["gene_frags"].items()])

        # determine how many molecules to sample
        if n_choices == -1 or n_choices > len(genes):
            n_choices = len(genes)

        # random sample of genes
        idx_lst = np.random.choice(len(genes), n_choices, replace=False)
        genes = [genes[i] for i in idx_lst]

        # score genes
        scored_genes = self.frag_sampler.score(genes, query_frag=ref_frag)

        smiles, scores = zip(*scored_genes)
        return list(smiles), list(scores)
