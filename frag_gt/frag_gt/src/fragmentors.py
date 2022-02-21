import logging
from abc import ABC, abstractmethod
from random import shuffle

from rdkit import Chem
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FragmentorBase(ABC):
    """ base class defining the minimal set of methods expected for fragmentating and reassembling molecules """

    @property
    @abstractmethod
    def name(self) -> str:
        """ returns a string with the name of the fragmentor """

    @property
    @abstractmethod
    def recombination_rules(self) -> Dict[Tuple[Any, Any], Any]:
        """
        returns a Python dictionary
            keys: two-membered tuple containing the "type" of atoms A and B to be connected. Sorted ascending.
            values: the type of bond that should be created to connect atoms A and B
        """

    @abstractmethod
    def get_frags(self, mol: Chem.rdchem.Mol) -> List[Chem.rdchem.Mol]:
        """
        (1) break bonds
        (2) add attachment points [*] to atoms that have been disconnected
        (3) assign the same unique attachment_idx to both [*] atoms
        """

    @abstractmethod
    def find_bonds(self, mol: Chem.rdchem.Mol):
        """ return a list of bonds that can be cut by the fragmentation rules """

    def get_cut_list(self, randomize_order: bool = True) -> list:
        """ Convenience function to get the list of atom cut types that can be recombined """
        types = list(set([x for y in self.recombination_rules.keys() for x in y]))
        if randomize_order:
            shuffle(types)
        return types


class BRICSFragmentor(FragmentorBase):
    """
    use the BRICS rules as implemented in RDKit for molecule fragmentation.
    based on retrosynthetic disconnections (see Degen et al. ChemMedChem, 3, 1503-7 (2008))
    """
    def __init__(self):
        from rdkit.Chem import BRICS
        self.BRICS = BRICS

    @property
    def name(self) -> str:
        return "brics"

    @property
    def recombination_rules(self) -> Dict[Tuple[Any, Any], Any]:
        recombination_rules = {}
        for bond_category in self.BRICS.reactionDefs:
            for start_type, end_type, bond_type in bond_category:

                # Not sure what the a and b are for since the smarts are identical...
                # This seems to fix things but haven't investigated beyond tests
                start_type = start_type.replace("a", "")
                end_type = end_type.replace("b", "")

                if bond_type == "-":
                    bond = Chem.rdchem.BondType.SINGLE
                elif bond_type == "=":
                    bond = Chem.rdchem.BondType.DOUBLE
                else:
                    raise ValueError(f"unrecognised bond type: {bond_type}")
                recombination_rules.update({(start_type, end_type): bond})
        return recombination_rules

    def get_frags(self, mol: Chem.rdchem.Mol) -> List[Chem.rdchem.Mol]:
        # Instead of fragmenting the molecule all at once, we loop over each bond to be cleaved
        # and add a unique attachment_idx so that molecules can be trivially reassembled
        idx = 0
        for brics_bond in self.find_bonds(mol):
            # BreakBRICSBonds returns a single mol object with [*] atoms with isotopes reflecting the cut type
            mol = self.BRICS.BreakBRICSBonds(mol, bonds=[brics_bond], sanitize=True, silent=True)

            # Assign attachment idx to the 2 new [*] atoms
            n_atoms = mol.GetNumAtoms()
            atom_1 = mol.GetAtomWithIdx(n_atoms - 1)
            assert atom_1.GetSymbol() == "*", "atom is not an attachment point"
            atom_2 = mol.GetAtomWithIdx(n_atoms - 2)
            assert atom_2.GetSymbol() == "*", "atom is not an attachment point"
            atom_1.SetProp("attachment_idx", str(idx))
            atom_2.SetProp("attachment_idx", str(idx))
            idx += 1
        return list(Chem.GetMolFrags(mol, asMols=True))

    def find_bonds(self, mol: Chem.rdchem.Mol) -> List[Tuple]:
        return list(self.BRICS.FindBRICSBonds(mol, randomizeOrder=False, silent=True))


def fragmentor_factory(fragment_scheme: str) -> FragmentorBase:
    """ retrieve fragmentor class by name """
    logger.info(f"Using fragmentation scheme: {fragment_scheme}")
    if fragment_scheme == "brics":
        return BRICSFragmentor()
    else:
        raise NotImplementedError(f"fragmentation scheme {fragment_scheme} not recognised")
