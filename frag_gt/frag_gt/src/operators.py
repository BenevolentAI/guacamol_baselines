import logging
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BreakBRICSBonds
from typing import Dict, List, Tuple

from frag_gt.src.afp import renumber_frag_attachment_idxs, match_fragment_attachment_points
from frag_gt.src.fragmentors import FragmentorBase
from frag_gt.src.gene_type_utils import get_gene_type, get_haplotype_from_gene_frag
from frag_gt.src.query_builder import FragQueryBuilder

logger = logging.getLogger(__name__)

DUMMY_SMARTS = Chem.MolFromSmarts("[#0]")
DUMMY_PLUS_ANCHOR_SMARTS = Chem.MolFromSmarts("[#0]~*")


def connect_mol_from_frags(frags: List[Chem.rdchem.Mol], fragmentor: FragmentorBase) -> Chem.rdchem.Mol:
    """
    Given a list of fragments (RDKit mol objects) with attachment points [*] marked by integer pairs (attachment_idx)
    Return a new mol object

    Atom properties are maintained from input frags

    Note: BRICSBuild function in rdkit enumerates results, is slow, and only considers cut_type (not attachment_idx)
    """

    # Combine mols to single object
    composite_frags = frags[0]
    for i in range(len(frags) - 1):
        composite_frags = Chem.rdmolops.CombineMols(composite_frags, frags[i + 1])

    # get pairs to be bonded together
    # dict keys are attachment_idxs. A pair of * atoms should both have this attachment_idx in the input frags
    # dict values are two-value tuple of lists. ([], [])
    # the first list contains mol object atom idxs to be bonded
    # the second list describes the type of re-connection that should be made (used to select bond type)
    # e.g. pairs = {3: [[0, 15], [3, 3]], 1: [[8, 17], [1, 1]], 2: [[21, 30], [7, 7]]}
    pairs = {}  # type: Dict[int, Tuple[List[int], List[str]]]
    for match in composite_frags.GetSubstructMatches(DUMMY_PLUS_ANCHOR_SMARTS):
        dummy_atom = composite_frags.GetAtomWithIdx(match[0])
        anchor_atom = composite_frags.GetAtomWithIdx(match[1])

        attachment_idx = dummy_atom.GetProp("attachment_idx")
        new_bond_info = pairs.get(attachment_idx, ([], []))
        new_bond_info[0].append(anchor_atom.GetIdx())
        new_bond_info[1].append(dummy_atom.GetIsotope())
        new_bond_info[1].sort()  # sort isotopes ascending to match keys in fragmentor.recombination_rules
        pairs[attachment_idx] = new_bond_info

    # get dict describing {cut_type: bond type} relationships
    recombination_dict = fragmentor.recombination_rules

    # make new bonds and delete bonds to [*]
    new_mol = Chem.RWMol(composite_frags)
    for i in pairs:
        new_mol.AddBond(pairs[i][0][0], pairs[i][0][1], recombination_dict[(str(pairs[i][1][0]), str(pairs[i][1][1]))])
    new_mol = new_mol.GetMol()
    new_mol = AllChem.DeleteSubstructs(new_mol, DUMMY_SMARTS)

    Chem.SanitizeMol(new_mol)
    return new_mol


def delete_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                         frag_db: FragQueryBuilder) -> List[Chem.rdchem.Mol]:
    """
    Delete a random pendant node
    Check that now severed node exists in db
    Else, mutate
    # TODO strict and relaxed flag where relaxed doesnt care if fragment exists or not
    """
    # fragment parent
    frags = fragmentor.get_frags(parent_mol)

    # if mol could not be fragmented, return empty list
    if len(frags) <= 1:
        return []

    # Identify pendant frags (frags with only one attachment)
    pendant_frag_idxs = [n for n, f in enumerate(frags)
                         if len(get_gene_type(f).split("#")) == 1]

    # If there are no pendant frags, return None
    if not len(pendant_frag_idxs):
        return []

    # Choose a pendant fragment to remove
    i = np.random.randint(0, len(pendant_frag_idxs))
    deleted_frag = frags.pop(pendant_frag_idxs[i])

    # Identify remaining fragment with hanging edge
    for a in deleted_frag.GetAtoms():
        if a.GetSymbol() == "*":
            hanging_idx = a.GetProp("attachment_idx")
            break

    # Identify frag and get atom idx of hanging edge with the same attachment_idx
    # TODO: this could be simplified by only breaking one bond in the first place!
    remaining_frag_idx = -1
    for n, frag in enumerate(frags):
        for a in frag.GetAtoms():
            if a.GetSymbol() == "*":
                if a.GetProp("attachment_idx") == hanging_idx:
                    remaining_frag_idx = n
                    hanging_edge_to_remove = a.GetIdx()
                    break
        if remaining_frag_idx != -1:
            break

    # Modify remaining frag to remove hanging attachment point
    remaining_frag = frags.pop(remaining_frag_idx)
    remaining_frag = Chem.RWMol(remaining_frag)
    remaining_frag.ReplaceAtom(hanging_edge_to_remove, Chem.Atom(1))
    remaining_frag = remaining_frag.GetMol()
    remaining_frag = Chem.RemoveHs(remaining_frag)
    Chem.SanitizeMol(remaining_frag)

    # Check if frag exists in db (1) Get gene_type from DB
    gene_type = get_gene_type(remaining_frag)
    gene_type_results = list(frag_db.db.get_records("gene_types", {"gene_type": gene_type}))

    # Check if frag exists in db (2) Check for specific gene among results
    if len(gene_type_results) > 1:
        logger.warning(f"DB is corrupted, multiple results for gene_type {gene_type}")
    elif gene_type == "":
        # If there are only two frags in original molecule and we are deleting one of them.
        # Gene type is "" as remaining frag is the whole molecule
        frag_exists = True
    elif len(gene_type_results) == 0:
        # This gene_type is not in the database
        frag_exists = False
    else:
        # Match on identical SMILES
        frag_exists = False
        rf_haplotype_smi = Chem.MolToSmiles(get_haplotype_from_gene_frag(remaining_frag))
        rf_gene_smi = Chem.MolToSmiles(remaining_frag)
        for hap, hap_record in gene_type_results[0]["haplotypes"].items():
            if hap == rf_haplotype_smi:
                if rf_gene_smi in hap_record["gene_frags"]:
                    frag_exists = True
                break

    # if mutated_frag does not exist, mutate to something that does
    if frag_exists:
        mutant_frag = remaining_frag
        accept_mutant = True
    else:
        query_frag = remaining_frag

        gene_type = get_gene_type(query_frag)  # variable change
        mutant_smiles, _ = frag_db.query_frags(gene_type, query_frag)

        # TODO: Could try to rematch alignment instead of moving to next smiles?
        accept_mutant = False
        while (accept_mutant is False) and (len(mutant_smiles) > 0):
            mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

            # align attachment idxs of mutant frag to original reference
            mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

            # Check reaction types agree
            accept_mutant = True
            idx_type_list = [(a.GetIsotope(), int(a.GetProp("attachment_idx"))) for a in query_frag.GetAtoms() if
                             a.GetSymbol() == "*"]
            for a in mutant_frag.GetAtoms():
                if a.GetSymbol() == "*":
                    mutant_idx_type = (a.GetIsotope(), int(a.GetProp("attachment_idx")))
                    if mutant_idx_type not in idx_type_list:
                        logger.debug("Not matching types")
                        logger.debug(idx_type_list)
                        logger.debug(mutant_idx_type)
                        accept_mutant = False
                        break

    if accept_mutant:
        # Reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # Reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"MutationDelNode: could not delete node from {Chem.MolToSmiles(parent_mol)}")
        return []


def substitute_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                             frag_db: FragQueryBuilder) -> List[Chem.rdchem.Mol]:
    """
    Choose a fragment from the parent molecule (fragmented using fragmentor)
    Retrieve fragments from the fragment store that have the same connectivity (gene_type)
    Align the new mutant fragment to the original fragment using the alignment fingerprint (afp)
    Reconnect the molecule incorporating the mutant fragment in place of the query fragment
    """

    # Fragment parent molecule
    frags = fragmentor.get_frags(parent_mol)

    # choose a fragment to mutate
    i = np.random.randint(0, len(frags))
    query_frag = frags.pop(i)

    # retrieve pool
    gene_type = get_gene_type(query_frag)
    mutant_smiles, _ = frag_db.query_frags(gene_type, query_frag)

    # TODO: Could try to rematch alignment instead of moving to next smiles? rematch by zscores? overcomplicated?
    accept_mutant = False
    while (accept_mutant is False) and (len(mutant_smiles) > 0):
        mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

        # align attachment idxs of mutant frag to original reference
        mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

        # Check reaction types agree
        # TODO: this can be simplified as a set comparison while ignoring negatives
        accept_mutant = True
        idx_type_list = [(a.GetIsotope(), int(a.GetProp("attachment_idx"))) for a in query_frag.GetAtoms() if
                         a.GetSymbol() == "*"]
        for a in mutant_frag.GetAtoms():
            if a.GetSymbol() == "*":
                mutant_idx_type = (a.GetIsotope(), int(a.GetProp("attachment_idx")))
                if mutant_idx_type not in idx_type_list:
                    logger.debug(f"MutationSubNode: Not matching types: {idx_type_list} {mutant_idx_type}")
                    accept_mutant = False
                    break

    if accept_mutant:
        # reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"substitute_node_mutation: no substitutions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def add_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                      frag_db: FragQueryBuilder) -> List[Chem.rdchem.Mol]:
    """
    Choose a random node to mutate
    Mutate to node with one too many attachment points
    Add a small attachment to the extra limb
    """

    # Fragment parent
    frags = fragmentor.get_frags(parent_mol)

    # Choose a random fragment to mutate
    i = np.random.randint(0, len(frags))
    query_frag = frags.pop(i)

    # Get query fragment gene_type
    gene_type = get_gene_type(query_frag)

    # Create list of all possible cut types to add (in a random order)
    # n_cut_types = len(fragmentor.get_cut_list()) - 1
    # cut_types = list(np.random.choice(n_cut_types, n_cut_types, replace=False))
    cut_types = fragmentor.get_cut_list(randomize_order=True)

    # Try to mutate gene_type until a suitable mutation is found
    gene_type_list = gene_type.split("#")
    accept_mutant = False
    while (accept_mutant is False) and (len(cut_types) > 0):
        # Try to add new node with cut_type: added_cut_type
        added_cut_type = str(cut_types.pop(0))

        # Retrieve small node for new pendant attachment
        # If no suitable nodes exist, continue to next cut type
        pendant_smiles_list, _ = frag_db.query_frags(added_cut_type, Chem.MolFromSmiles(f"[{added_cut_type}*]C"))
        if not len(pendant_smiles_list):
            logger.debug(f"add_node_mutation: No small fragments in DB with cut_type: {added_cut_type}")
            continue

        # TODO: BRICS has multiple options here so another nested while loop?!
        # Mutate existing fragment to contain an extra attachment point
        # Identify suitable cut type to add
        possible_joins = [rule for rule in fragmentor.recombination_rules if added_cut_type in rule]
        random.shuffle(possible_joins)
        picked_join = set(possible_joins.pop(0))
        if len(picked_join) == 1:  # since its a set {'5', '5'} becomes {'5'}
            added_cut_type2 = picked_join.pop()
        else:
            picked_join.remove(added_cut_type)
            added_cut_type2 = picked_join.pop()

        # Given new cut type, try to mutate existing fragment
        mutant_gene_type_list = sorted([int(x) for x in gene_type_list if x is not ''] + [int(added_cut_type2)])
        mutant_gene_type = "#".join([str(x) for x in mutant_gene_type_list])
        logger.debug(f"add_node_mutation: Attempting to mutate existing gene {gene_type} -> {mutant_gene_type}")

        # Query database to retrieve frags belonging to mutant_gene_type
        mutant_gene_list, _ = frag_db.query_frags(mutant_gene_type, query_frag)

        while (accept_mutant is False) and (len(mutant_gene_list) > 0):
            # Pop candidate mutant smiles and add "attachment_idx" atom prop to [*] atoms
            mutant_frag = Chem.MolFromSmiles(mutant_gene_list.pop(0))

            # align attachment idxs of mutant frag to original reference
            mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

            # Check aligned query and mutant cut-types agree, else continue to next candidate gene
            accept_mutant = True

            # Query frag aligned type details
            idx_type_list = [(a.GetIsotope(), int(a.GetProp("attachment_idx")))
                             for a in query_frag.GetAtoms() if a.GetSymbol() == "*"]

            # Mutant frag aligned type details, ignore att id -1 since this is the newly added edge
            # TODO: this can be simplified as a set comparison while ignoring negatives
            for a in mutant_frag.GetAtoms():
                if a.GetSymbol() == "*":
                    mutant_idx_type = (a.GetIsotope(), int(a.GetProp("attachment_idx")))
                    if mutant_idx_type[1] < 0:
                        continue
                    if mutant_idx_type not in idx_type_list:
                        logger.debug(f"add_node_mutation: Not matching types: {idx_type_list} {mutant_idx_type}")
                        accept_mutant = False
                        break

    if accept_mutant:

        # Format added frag
        new_smiles = pendant_smiles_list.pop(0)
        new_frag = Chem.MolFromSmiles(new_smiles)
        for a in new_frag.GetAtoms():
            if a.GetSymbol() == "*":
                a.SetProp("attachment_idx", str(-1))

        # Reconstruct list of frags for new molecule
        frags.append(new_frag)
        frags.append(mutant_frag)

        # Reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"add_node_mutation: no suitable additions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def single_point_crossover(m1: Chem.rdchem.Mol, m2: Chem.rdchem.Mol,
                           fragmentor: FragmentorBase) -> List[Chem.rdchem.Mol]:

    # Get list of possible cuts for each molecules
    # Tuple([pair of atom idxs], [pair of cut type str])
    # e.g. [([17, 16], ('5', '5')), ([5, 6], ('9', '9')), ([8, 7], ('9', '9'))]
    m1_possible_cuts = fragmentor.find_bonds(m1)
    m2_possible_cuts = fragmentor.find_bonds(m2)

    # Get types in common
    m1_cut_types = [ct for _, ct in m1_possible_cuts]
    m2_cut_types = [ct for _, ct in m2_possible_cuts]
    common_cts = list(set(m1_cut_types) & set(m2_cut_types))

    # If no applicable cut types, return None
    if not len(common_cts):
        logger.debug(f"single_point_crossover: no common cut types: {Chem.MolToSmiles(m1)}, {Chem.MolToSmiles(m2)}")
        return []

    # Filter to cuts in common
    m1_possible_cuts = [cut for cut in m1_possible_cuts if cut[1] in common_cts]
    m2_possible_cuts = [cut for cut in m2_possible_cuts if cut[1] in common_cts]

    # Shuffle both lists to avoid systematic bias
    random.shuffle(m1_possible_cuts)
    random.shuffle(m2_possible_cuts)

    # Get first mol cut
    picked_cut_1 = m1_possible_cuts.pop(0)

    # Get second mol cut
    for n, cut in enumerate(m2_possible_cuts):
        if cut[1] == picked_cut_1[1]:
            picked_cut_2 = m2_possible_cuts.pop(n)

    # Cut both molecules using BRICS function (turns out this func is quite general!)
    # Syntax here is break bond, and assign isotopes as cut types
    m1_pieces = BreakBRICSBonds(m1, [picked_cut_1])
    m2_pieces = BreakBRICSBonds(m2, [picked_cut_2])

    # Separate into different mol objects
    m1_pieces = list(Chem.GetMolFrags(m1_pieces, asMols=True))
    m2_pieces = list(Chem.GetMolFrags(m2_pieces, asMols=True))

    # Add attachment_idx so that frags can be reconnected (each should only have one)
    m1_pieces = [renumber_frag_attachment_idxs(frag) for frag in m1_pieces]
    m2_pieces = [renumber_frag_attachment_idxs(frag) for frag in m2_pieces]

    # Crossover
    # There are good and bad combinations...
    # Here we lose two possibilities that are presumed not sensible
    # TODO: generate all four molecules and filter?
    m1_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    m2_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    new_m1 = [m1_pieces[0], m2_pieces[1]]
    new_m2 = [m1_pieces[1], m2_pieces[0]]

    try:
        return [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)]
    except KeyError:
        # TODO: bug with asymmetric bond cuts when wrong partners are re-combined
        new_m1 = [m1_pieces[1], m2_pieces[1]]
        new_m2 = [m1_pieces[0], m2_pieces[0]]
        return [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)]


def operator_factory(operator_name: str):
    if operator_name.lower() == "substitute_node_mutation":
        return substitute_node_mutation
    elif operator_name.lower() == "add_node_mutation":
        return add_node_mutation
    elif operator_name.lower() == "delete_node_mutation":
        return delete_node_mutation
    elif operator_name.lower() == "single_point_crossover":
        return single_point_crossover
    else:
        raise(Exception(f"operator {operator_name} not recognised"))
