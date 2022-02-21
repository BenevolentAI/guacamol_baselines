import logging
from collections import namedtuple

import numpy as np
from rdkit import Chem
from typing import List, Optional, Tuple

from frag_gt.src.fragmentors import fragmentor_factory
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.operators import operator_factory
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
        # should probably assert fragmentor and fragstore have same scheme

        # the query builder supports the mutation operators and controls how we sample the fragstore
        self.fragstore_qb = FragQueryBuilder(self._fragstore,
                                             scorer=scorer,
                                             stochastic=True)
        self.selection_method = selection_method

        # tuple of (mutation/crossover operator, probability of applying)
        if operators is None:
            logger.debug("using default mutation and crossover operators")
            operators = [
                ("substitute_node_mutation", 0.4),
                ("add_node_mutation", 0.1),
                ("delete_node_mutation", 0.1),
                ("single_point_crossover", 0.4),
            ]
        logger.info(f"operator probabilities: {operators}")
        assert sum([tup[1] for tup in operators]) == 1.0, "operator probabilities must sum to one"
        self.mol_operators, self.mol_operator_probs = zip(*operators)
        self.allow_unspecified_stereo = allow_unspecified_stereo

        # keep substructures fixed in the population
        self._fixed_substructure_smarts = fixed_substructure_smarts
        self.fixed_substructure = None
        if fixed_substructure_smarts is not None:
            self.fixed_substructure = Chem.MolFromSmarts(fixed_substructure_smarts)
            if self.fixed_substructure is None:
                raise ValueError(f"invalid smarts pattern returned None: {fixed_substructure_smarts}")
        self.patience = patience

    @staticmethod
    def tournament_selection(population: List[Molecule], k: int = 5):
        """
        tournament selection randomly chooses five individuals from
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

            # select random molecule(s) from the current pool
            if self.selection_method == "random":
                idxs = np.random.choice(len(current_pool), size=2)
                choices = [current_pool[i] for i in idxs]
            elif self.selection_method == "tournament":
                choice_1 = self.tournament_selection(current_pool, k=5)
                choice_2 = self.tournament_selection(current_pool, k=5)
                choices = [choice_1, choice_2]
            else:
                raise ValueError(f"Unrecognised selection method: {self.selection_method}")

            # select a molecular transform operator (i.e mutation, crossover)
            j = np.random.choice(len(self.mol_operator_probs), 1, p=self.mol_operator_probs)[0]
            op = self.mol_operators[j]

            # apply operators to generate children from parent compounds
            parent_mol = choices[0].mol
            if not op.endswith("crossover"):
                # mutation
                children = operator_factory(op)(parent_mol, self.fragmentor, self.fragstore_qb)
            else:
                # crossover
                second_parent_mol = choices[1].mol
                children = operator_factory(op)(parent_mol, second_parent_mol, self.fragmentor)

            # handle unspecified stereocenters if necessary
            if not self.allow_unspecified_stereo:
                # check for unspecified stereocenters
                # todo what if fragment has a terminal chiral group?
                explicit_stereo_children = []
                for child in children:
                    explicit_stereo_children.extend(enumerate_unspecified_stereocenters(child))
                children = explicit_stereo_children

            # filter molecules that do not maintain the fixed SMARTS
            # the advantage of filtering here instead of having a SMARTS scoring function
            # is that we dont waste compute on other scoring functions
            if self.fixed_substructure is not None:
                contains_substructure = []
                for mol in children:
                    if mol.HasSubstructMatch(self.fixed_substructure, useChirality=True):
                        contains_substructure.append(mol)
                children = contains_substructure

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
