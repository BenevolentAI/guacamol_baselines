import logging

from rdkit import Chem, RDLogger
from typing import List

logger = logging.getLogger(__name__)

RDLogger.DisableLog("rdApp.warning")
DUMMY_ATOM = Chem.MolFromSmarts("[#0]")
H_ATOM = Chem.MolFromSmiles("[H]")


def get_gene_type(frag: Chem.rdchem.Mol):
    """
    takes an rdkit mol object containing wildcard attachment points ([*]) and returns a canonical "type"
    "gene type" is type of attachment (encoded in atom isotopes) sorted and joined by "#"
    Usage:
    >>> m1 = Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2")
    >>> get_gene_type(m1)
    >>> "1#2#3#4"
    """
    cut_point_typelist = []
    for atom in frag.GetAtoms():
        if atom.GetSymbol() == "*":
            cut_point_typelist.append(atom.GetIsotope())
    gene_type = "#".join([str(x) for x in sorted(cut_point_typelist)])
    return gene_type


def get_haplotype_from_gene_frag(frag: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """ given a fragment with attachment points return the scaffold without attachments (haplotype frag) """
    haplotype_mol = Chem.ReplaceSubstructs(frag, DUMMY_ATOM, H_ATOM, replaceAll=True)[0]
    haplotype_mol = Chem.RemoveHs(haplotype_mol)
    return haplotype_mol


def get_species(frags: List[Chem.rdchem.Mol]):
    """
    This function tries to assign a fragmented molecule (chromosome) to a canonical species using frag gene types
    A species is a string of the fragment gene types joined by "." where gene_types appear in a canonical order
    """
    gene_types = [get_gene_type(x) for x in frags]
    if len(gene_types) == 1:
        return ""

    # Sort to generate a canonical "species"
    # - first by number of attachments (i.e. scaffolds first, pendant last)
    # - then by attachment type
    sorted_gts = sorted(gene_types, key=lambda x: (len(x.split("#")), -int("".join(x.split("#")))), reverse=True)
    return ".".join(sorted_gts)
