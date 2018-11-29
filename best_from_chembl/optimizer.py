import heapq
from typing import List, Optional, Tuple

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction

from .chembl_file_reader import ChemblFileReader


class BestFromChemblOptimizer(GoalDirectedGenerator):
    """
    Goal-directed molecule generator that will simply look for the most adequate molecules present in a file.
    """

    def __init__(self, smiles_reader: ChemblFileReader) -> None:
        self.smiles_reader = smiles_reader

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        """
        Will iterate through the reference set of SMILES strings and select the best molecules.

        It will create a heap and keep it to the required size so that minimal memory is used.
        """
        top_molecules: List[Tuple[float, str]] = []

        for smiles in self.smiles_reader:
            score = scoring_function.score(smiles)

            # Put molecule and corresponding score in a tuple that allows for appropriate comparison operator for the heap.
            item = (score, smiles)

            if len(top_molecules) < number_molecules:
                heapq.heappush(top_molecules, item)
            else:
                # Equivalent to a push, then a pop, but faster
                # NB: pop removes the smallest value, i.e. in this case the molecule with the lowest score.
                heapq.heappushpop(top_molecules, item)

        return [x[1] for x in top_molecules]
