from abc import ABC, abstractmethod

from rdkit.Chem import Descriptors
from typing import List, Tuple, Union, Dict

from frag_gt.src.fragmentors import fragmentor_factory
from frag_gt.src.gene_type_utils import get_species
from frag_gt.src.population import Molecule


class MapElites(ABC):
    """
    Place molecules in discretized map of the feature space, where only the fittest `self.n_elites`
    molecules are kept per cell. This ensures diversity in the population.
    """
    def __init__(self, n_elites: int = 1):
        self.n_elites = n_elites

    def place_in_map(self, molecule_list: List[Molecule]) -> Tuple[List[Molecule], List[str]]:
        """
        1. Compute the feature descriptor of the solution to find the correct cell in the N-dimensional space
        2. Check if the cell is empty or if the previous performance is worse, place new solution in the cell

        Args:
            molecule_list: list of molecule objects with fitness scores

        Returns:

        """
        map: Dict[str, List[Molecule]] = {}
        for mol in molecule_list:

            # compute features and output a discrete cell id (str)
            f = self.compute_features(mol)

            # get existing molecule in that cell
            existing_m = map.get(f, [])

            # place the current mol in the map if its fitter than others in the cell
            if not len(existing_m) or (existing_m[-1].score < mol.score):
                existing_m.append(mol)
                existing_m = sorted(existing_m, key=lambda x: x.score, reverse=True)[:self.n_elites]
                map[f] = existing_m

        return [m for mollist in map.values() for m in mollist], list(map.keys())

    @abstractmethod
    def compute_features(self, m: Molecule):
        pass


class MWLogPMapElites(MapElites):
    """ map elites using two dimensions: molecular weight and log p """
    def __init__(self, mw_step_size: float = 25., logp_step_size: float = 0.25, n_elites: int = 1):
        self.mw_step_size = mw_step_size
        self.logp_step_size = logp_step_size
        super().__init__(n_elites)

    def compute_features(self, m: Molecule) -> str:
        mw = Descriptors.MolWt(m.mol)
        log_p = Descriptors.MolLogP(m.mol)

        # discretize
        mw_cell_midpoint = round(mw / self.mw_step_size) * self.mw_step_size
        log_p_cell_midpoint = round(log_p / self.logp_step_size) * self.logp_step_size

        return f"mw-midpoint={mw_cell_midpoint},logp-midpoint{log_p_cell_midpoint}"


class SpeciesMapElites(MapElites):
    """ map elites using a single dimension: species (constructed from the gene types of constituent fragment genes """
    def __init__(self, fragmentor: str, n_elites: int = 1):
        self.fragmentor = fragmentor_factory(fragmentor)
        super().__init__(n_elites)

    def compute_features(self, m: Molecule) -> str:
        frags = self.fragmentor.get_frags(m.mol)
        return get_species(frags)


def map_elites_factory(mapelites_str: str, fragmentation_scheme) -> Union[SpeciesMapElites, MWLogPMapElites]:
    if mapelites_str == "mwlogp":
        map_elites = MWLogPMapElites(mw_step_size=25, logp_step_size=0.5)
    elif mapelites_str == "species":
        map_elites = SpeciesMapElites(fragmentation_scheme)
    else:
        raise ValueError(f"unknown value for mapelites argument: {mapelites_str}")
    return map_elites