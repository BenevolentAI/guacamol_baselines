from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List


class SmilesScorer(ABC):
    """
    This class is a simple abstraction for the scoring of molecules by FragGT.

    It is a minimal version of the `ScoringFunction` class provided with GuacaMol which may be more useful in practice.
    Scoring Functions from GuacaMol such as those in the benchmark suites can be used instead of FragGTScorer subclasses

    We provide this class to ensure that FragGT and GuacaMol are not coupled
    """
    @abstractmethod
    def score(self, smiles: str) -> float:
        """
        Score a single molecule as smiles
        """
        raise NotImplementedError

    def score_list(self, smiles_list: List[str]) -> List[float]:
        """
        Score a list of smiles.
        Override this function if there's a more efficient way to score batches of molecules.
        """
        return [self.score(smi) for smi in smiles_list]


class MolecularWeightScorer(SmilesScorer):
    """
    An example scorer that "scores" molecules according to molecular weight.
    FragGT is not provided with a library of scorers, these are left to the user to implement.
    """
    def score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid mol in scorer: {smiles}")

        return Descriptors.MolWt(mol)
