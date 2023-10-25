import logging
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from frag_gt.src.fragmentors import fragmentor_factory
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.operators import mc_operator_factory
from frag_gt.src.query_builder import FragQueryBuilder
from frag_gt.src.stereo import enumerate_unspecified_stereocenters

logger = logging.getLogger(__name__)
Molecule = namedtuple("Molecule", ["score", "mol"])


class MolecularPopulationGenerator:
    """ handles generation of a population of molecules, knows nothing of molecular fitness """
    def __init__(self,
                 fragstore_path: str,
                 fragmentation_scheme: str,
                 n_molecules: int,
                 operators: Optional[List[Tuple[str, float]]] = None,
                 allow_unspecified_stereo: bool = False,
                 selection_method: str = "tournament",
                 scorer: str = "counts",
                 fixed_substructure_smarts: Optional[str] = None,
                 patience: int = 1000):

        self.n_molecules = n_molecules
        self.fragmentor = fragmentor_factory(fragmentation_scheme)
        self._fragstore_path = fragstore_path
        self._fragstore = fragstore_factory("in_memory", fragstore_path)

        # the query builder supports the mutation operators and controls how we sample the fragstore
        self.fragstore_qb = FragQueryBuilder(self._fragstore,
                                             scorer=scorer,
                                             sort_by_score=False,
                                             skip_tournament_prob=0.02,
                                             sample_with_replacement=True)
        assert self._fragstore.scheme == self.fragmentor.name

        # tuple of (mutation/crossover operator, probability of applying)
        if operators is None:
            logger.debug("using default mutation and crossover operators")
            operators = [
                ("substitute_node_mutation", 0.4),
                ("add_node_mutation", 0.05),
                ("delete_node_mutation", 0.05),
                ("single_point_crossover", 0.45),
                ("substitute_edge_mutation", 0.05),
            ]
        logger.info(f"operator probabilities: {operators}")
        assert sum([tup[1] for tup in operators]) == 1.0, "operator probabilities must sum to one"
        self.mol_operators, self.mol_operator_probs = zip(*operators)
        self.allow_unspecified_stereo = allow_unspecified_stereo
        self.selection_method = selection_method

        # keep substructures fixed in the population
        self._fixed_substructure_smarts = fixed_substructure_smarts
        self.fixed_substructure = None
        if fixed_substructure_smarts is not None:
            self.fixed_substructure = Chem.MolFromSmarts(fixed_substructure_smarts)
            if self.fixed_substructure is None:
                raise ValueError(f"invalid smarts pattern returned None: {fixed_substructure_smarts}")
        self.patience = patience
        self.max_molecular_weight = 1500

    @staticmethod
    def tournament_selection(population: List[Molecule], k: int = 5) -> Molecule:
        """
        tournament selection randomly chooses k individuals from
        the population and returns the fittest one
        """
        entrant_idxs = np.random.choice(len(population), size=k, replace=True)
        fittest = sorted([population[idx] for idx in entrant_idxs], key=lambda x: x.score, reverse=True)[0]
        return fittest

    def generate(self, current_pool: List[Molecule]) -> List[Chem.rdchem.Mol]:
        """ generate a new pool of molecules """

        new_pool: List[Chem.rdchem.Mol] = []
        patience = 0
        while len(new_pool) < self.n_molecules:

            # select molecule(s) from the current pool
            if self.selection_method == "random":
                idxs = np.random.choice(len(current_pool), size=2)
                choices = [current_pool[i] for i in idxs]
            elif self.selection_method.startswith("tournament"):
                tournament_size = 3 if not "-" in self.selection_method else int(self.selection_method.split("-")[-1])
                choice_1 = self.tournament_selection(current_pool, k=tournament_size)
                choice_2 = self.tournament_selection(current_pool, k=tournament_size)
                choices = [choice_1, choice_2]
            else:
                raise ValueError(f"Unrecognised selection method: {self.selection_method}")

            # select a molecular transform operator (i.e mutation, crossover)
            j = np.random.choice(len(self.mol_operator_probs), 1, p=self.mol_operator_probs)[0]
            mc_op = self.mol_operators[j]

            # apply operators to generate children from parent compounds
            parent_mol = choices[0].mol
            if not mc_op.endswith("crossover"):
                # mutation
                children = mc_operator_factory(mc_op)(parent_mol, self.fragmentor, self.fragstore_qb)
            else:
                # crossover
                second_parent_mol = choices[1].mol
                children = mc_operator_factory(mc_op)(parent_mol, second_parent_mol, self.fragmentor)

            # handle unspecified stereocenters if necessary
            if not self.allow_unspecified_stereo:
                # todo check what happens if fragment has a terminal chiral group?
                explicit_stereo_children = []
                for child in children:
                    explicit_stereo_children.extend(enumerate_unspecified_stereocenters(child))
                children = explicit_stereo_children

            # filter molecules that do not maintain the fixed SMARTS
            # the advantage of filtering here instead of including a SMARTS scoring function
            # is that we dont waste compute on scoring functions
            if self.fixed_substructure is not None:
                filtered_children = []
                for mol in children:
                    if mol.HasSubstructMatch(self.fixed_substructure, useChirality=True):
                        filtered_children.append(mol)
                children = filtered_children

            # filter by mol weight (very large number, stops expensive calls to scoring fns)
            if self.max_molecular_weight > 0:
                filtered_children = []
                for i in range(len(children)):
                    if MolWt(children[i]) <= self.max_molecular_weight:
                        filtered_children.append(children[i])
                    else:
                        logger.info(f"Dropping molecule with MW exceeding max size ({self.max_molecular_weight})...")
                children = filtered_children

            # breaks infinite loop if no molecules can be generated
            if not len(children):
                patience += 1
                if not patience % 100:
                    logger.info(f"Failed to generate molecule: {patience}")
                if patience >= self.patience:
                    logger.info("Could not generate new pool of molecules, bailing...")
                    break
            else:
                patience = 0

            # extend the pool with new molecules
            new_pool.extend(children)

        return new_pool
