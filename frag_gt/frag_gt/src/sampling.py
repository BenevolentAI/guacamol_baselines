import logging
from random import shuffle

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from typing import Optional, List, Callable, Tuple

from frag_gt.src.afp import calculate_alignment_similarity_scores

logger = logging.getLogger(__name__)


class FragSampler:
    """
    This class provides functionality to reorder an input list of fragments according to a desired distribution.
    It has 3 primary stages: SCORE -> MODIFY -> CHOOSE

    SCORE
        Method by which to score the candidate fragments for replacement
        - "afps" uses the quality of alignment by afps to score fragments (slower than the others)
        - "counts" uses a prior based on the prevalence of fragments in the corpus used to generate the
           fragment database (can also be used to supply any external score)
        - "random" ignores the scoring aspect of this function and returns nan as the scores
        - "ecfp4" ranks candidate replacement fragments from the fragstore according to similarity to the query

    MODIFY
        - A modifier can be applied to the counts, valid modifiers can be found at the link below (e.g. np.log):
            - http://seismo.berkeley.edu/~kirchner/eps_120/Toolkits/Toolkit_03.pdf

    CHOOSE
        The desired number of fragments are returned (n_choices)
            - either deterministically using the highest score (stochastic=False)
            - Or stochastically using the scores to form probabilities and by choosing using np.random.choice
    """
    def __init__(self,
                 scorer: str = "random",
                 modifier: Optional[Callable[[List], List]] = None,
                 stochastic: Optional[bool] = False):

        self.scorer = scorer
        self.modifier = modifier
        self.stochastic = stochastic
        logger.info(f"fragment sampler initialised: scorer={scorer}, modifier={modifier}, stochastic={stochastic}")

    def __call__(self,
                 gene_frag_list: List[Tuple[str, int]],
                 n_choices: int = -1,
                 query_frag: Chem.rdchem.Mol = None,
                 ) -> Tuple[List[str], List[float]]:
        """

        Args:
            gene_frag_list: [("[2*]Cc1cc(O)cc(O[4*])c1", 2), ("[2*]CC(=N[4*])C(C)(C)C", 8), ("[2*]CC(N[4*])C(C)C", 1)]
            n_choices: number of fragments to return or -1 for entire list (with scores)
            query_frag: (optional) mol to guide scoring (not used by "counts" or "random")

        Returns:
            list of smiles, list of floating point "scores" whose interpretation depends on the type of scorer used
        """
        # Unzip list of tuples retrieved from fragstore
        smiles, counts = zip(*gene_frag_list)

        # Determine how many molecules to return
        n_smiles = len(smiles)
        if n_choices == -1:
            n_choices = n_smiles

        # (1) SCORE
        if self.scorer == "counts":
            # determine the probability of sampling molecule proportional to count in corpus
            scores = counts
        elif self.scorer == "ecfp4":
            # determine the probability of sampling molecule proportional to ecfp4 similarity to query frag
            assert query_frag is not None, "Must specify `query_frag` argument if using the ecfp4 scorer to sample"
            scores = score_with_fingerprints(query_frag, smiles)
        elif self.scorer == "afps":
            # use the alignment score to sort mols returned
            assert query_frag is not None, "Must specify `query_frag` argument if using the afp scorer to sample"
            try:
                scores = calculate_alignment_similarity_scores(query_frag, list(smiles))
            except AssertionError as e:
                if str(e) == "query must have attachments":
                    # if query has no attachment points, score with fingerprints
                    scores = score_with_fingerprints(query_frag, smiles)
                else:
                    raise AssertionError(e)

        elif self.scorer == "random":
            scores = np.full(len(smiles), np.nan)
            smiles = list(smiles)
            shuffle(smiles)
        else:
            raise ValueError(f"requested scorer for sampling not recognised: {self.scorer}")

        # (2) MODIFY
        # if modifier is provided, apply to scores
        if self.modifier is not None:
            scores = self.modifier(scores)

        # (3) CHOOSE
        # choose idxs
        if self.stochastic:

            if self.scorer == "random":
                idx_lst = range(n_choices)
            else:
                total = np.sum(scores)
                probabilities = np.array([float(score) / total for score in np.array(scores)])
                try:
                    idx_lst = np.random.choice(n_smiles, n_choices, p=probabilities, replace=False)
                except ValueError as e:
                    if str(e) == "Fewer non-zero entries in p than size":
                        # if n_choices > num non zero, pick non zeros first
                        num_non_zero = len(np.where(probabilities > 0)[0])
                        idx_lst = np.random.choice(n_smiles, num_non_zero, p=probabilities, replace=False)
                        n_remaining = n_choices - num_non_zero
                        remaining_to_choose_from = np.array(list(set(range(n_choices)) - set(idx_lst)))
                        idx_lst2 = np.random.choice(remaining_to_choose_from, n_remaining, replace=False)
                        idx_lst = np.concatenate([idx_lst, idx_lst2])
                    else:
                        raise ValueError(e)

            # get smiles and scores that have been sampled by np.random.choice
            # todo should this actually return genes in the tuple format to match the input?
            smiles = [smiles[i] for i in idx_lst]
            scores = [scores[i] for i in idx_lst]

        else:
            # return smiles according to decreasing score (deterministically)
            # todo dont we want to sort the stochastic samples too|?
            sorted_tuples = sorted(zip(smiles, scores), key=lambda t: t[1], reverse=True)[:n_choices]
            smiles = [s for s, sc in sorted_tuples]
            scores = [sc for s, sc in sorted_tuples]

        return smiles, scores


def score_with_fingerprints(query_mol, smiles_list):
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
