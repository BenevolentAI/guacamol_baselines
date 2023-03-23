import logging
from random import shuffle

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from typing import List, Tuple

from frag_gt.src.afp import calculate_alignment_similarity_scores

logger = logging.getLogger(__name__)


class FragScorer:
    """
    class to sample and score from fragment list

    Method by which to score fragments
    - "afps" uses the quality of alignment by afps to score fragments (slower than the others)
    - "counts" uses a prior based on the prevalence of fragments in the corpus used to generate the
       fragment database (can also be used to supply any external score)
    - "random" ignores the scoring aspect of this function and returns nan as the scores
    - "ecfp4" ranks candidate replacement fragments from the fragstore according to similarity to the query
    """
    def __init__(self, scorer: str = "random", sort: bool = True):
        self.scorer = scorer
        self.sort = sort
        logger.info(f"fragment sampler initialised: scorer={scorer} sort={sort}")

    def score(self,
              gene_frag_list: List[Tuple[str, int]],
              query_frag: Chem.rdchem.Mol = None,
              ) -> List[Tuple[str, float]]:
        """

        Args:
            gene_frag_list: [("[2*]Cc1cc(O)cc(O[4*])c1", 2), ("[2*]CC(=N[4*])C(C)(C)C", 8), ("[2*]CC(N[4*])C(C)C", 1)]
            query_frag: (optional) mol to guide scoring (not used by "counts" or "random")

        Returns:
            list of (smiles, score) tuples
        """
        # Unzip list of tuples retrieved from fragstore
        # this will include any precalculated or saved properties stored with each fragment
        # (e.g. count of how many times fragment occurred in corpus)
        smiles, counts = zip(*gene_frag_list)

        if self.scorer == "counts":
            # score frags using count in corpus
            scores = counts
        elif self.scorer == "ecfp4":
            # score frags using ecfp4 similarity to query frag
            scores = ecfp_fragment_scorer(query_frag, smiles)
        elif self.scorer == "afps":
            # score frags using the alignment score
            scores = afp_fragment_scorer(query_frag, smiles)
        elif self.scorer == "random":
            # random frag scorer
            scores = np.full(len(smiles), np.nan)
            smiles = list(smiles)
            shuffle(smiles)
        else:
            raise ValueError(f"requested scorer for sampling not recognised: {self.scorer}")

        if self.sort:
            # return smiles according to decreasing score (deterministically)
            sorted_tuples = sorted(zip(smiles, scores), key=lambda t: t[1], reverse=True)
            smiles = [s for s, sc in sorted_tuples]
            scores = [sc for s, sc in sorted_tuples]

        # zip back into same format as input
        scored_gene_frag_list = list(zip(smiles, scores))

        return scored_gene_frag_list


def afp_fragment_scorer(query_mol: Chem.rdchem.Mol, smiles_list: List[str]) -> List[float]:
    assert query_mol is not None, "Must specify `query_frag` argument if using the afp scorer to sample"
    try:
        scores = calculate_alignment_similarity_scores(query_mol, list(smiles_list))
    except AssertionError as e:
        if str(e) == "query must have attachments":
            # if query has no attachment points, score with fingerprints
            scores = ecfp_fragment_scorer(query_mol, smiles_list)
        else:
            raise AssertionError(e)
    return scores


def ecfp_fragment_scorer(query_mol: Chem.rdchem.Mol, smiles_list: List[str]) -> List[float]:
    scores = np.zeros(len(smiles_list))
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=512)
    for n, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            score = np.nan
        else:
            score = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=512))
        scores[n] = score
    return scores
