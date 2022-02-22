from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.rdchem import StereoSpecified
from typing import List

STEREO_OPTIONS = StereoEnumerationOptions(tryEmbedding=True, unique=True, maxIsomers=8, rand=None)

# todo explore fragment on chiral https://sourceforge.net/p/rdkit/mailman/message/35420297/


def mol_contains_unspecified_stereo(m: Chem.rdchem.Mol) -> bool:
    try:
        si = Chem.FindPotentialStereo(m)
    except ValueError as e:
        print(e)
        print(Chem.MolToSmiles(m))
        return False
    if any([element.specified == StereoSpecified.Unspecified for element in si]):
        return True
    else:
        return False


def enumerate_unspecified_stereocenters(m: Chem.rdchem.Mol) -> List[Chem.rdchem.Mol]:
    if mol_contains_unspecified_stereo(m):
        isomers = list(EnumerateStereoisomers(m, options=STEREO_OPTIONS))
    else:
        isomers = [m]
    return isomers
