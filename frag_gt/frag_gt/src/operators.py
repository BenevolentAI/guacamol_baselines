import logging
import random
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BreakBRICSBonds

from frag_gt.src.afp import renumber_frag_attachment_idxs, match_fragment_attachment_points
from frag_gt.src.fragmentors import FragmentorBase
from frag_gt.src.gene_type_utils import get_gene_type, get_haplotype_from_gene_frag, get_attachment_type_idx_pairs
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
    # combine mols to single object
    composite_frags = frags[0]
    for i in range(len(frags) - 1):
        composite_frags = Chem.CombineMols(composite_frags, frags[i + 1])

    # get pairs to be bonded together
    # dict keys are attachment_idxs. A pair of * atoms should both have this attachment_idx in the input frags
    # dict values are two-value tuple of lists. ([], [])
    # - the first list contains mol object atom idxs to be bonded
    # - the second list describes the type of re-connection that should be made (used to select bond type)
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


def _find_partner_frag_and_atom_idx(frag_list: List[Chem.rdchem.Mol], query_attachment_idx: int):
    """ Given a query_attachment_idx, find the corresponding fragment and atom that partners it in a list of frags """
    for n, frag in enumerate(frag_list):
        for m in frag.GetSubstructMatches(DUMMY_SMARTS):
            a = frag.GetAtomWithIdx(m[0])
            if a.GetProp("attachment_idx") == query_attachment_idx:
                partner_frag_idx = n
                partner_atom_idx = a.GetIdx()
                return partner_frag_idx, partner_atom_idx


def _find_pendant_frag_idxs(frags: List[Chem.rdchem.Mol]) -> List[int]:
    """ identify pendant frags (frags with only one attachment) """
    return [n for n, f in enumerate(frags) if len(get_gene_type(f).split("#")) == 1]


def _get_partner_cut(added_cut_type_new_frag: str, fragmentor: FragmentorBase) -> str:
    """
    given a cut type, find corresponding partner to join
    handles asymmetric cut types e.g. [*5]-[7*])
    returns a random join picked from available options
    """
    possible_joins = [rule for rule in fragmentor.recombination_rules if added_cut_type_new_frag in rule]
    random.shuffle(possible_joins)
    picked_join = possible_joins.pop(0)
    if picked_join[0] == added_cut_type_new_frag:
        return picked_join[1]
    else:
        return picked_join[0]


def delete_node_mutation(parent_mol: Chem.rdchem.Mol,
                         fragmentor: FragmentorBase,
                         frag_db: FragQueryBuilder,
                         strict: bool = True) -> List[Chem.rdchem.Mol]:
    """
    Delete a random pendant node
    (if strict=True) Check that severed "hanging" node exists in fragstore, else mutate to one that does
    (if strict=False) patch the hanging edge to hydrogen
    """
    # fragment parent
    frags = fragmentor.get_frags(parent_mol)

    # if mol could not be fragmented, return empty list
    if len(frags) <= 1:
        logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} could not be fragmented")
        return []

    # identify pendant frags
    pendant_frag_idxs = _find_pendant_frag_idxs(frags)
    if not len(pendant_frag_idxs):
        logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} has no pendant frags ")
        return []

    # choose a pendant fragment to remove
    deleted_frag = frags.pop(pendant_frag_idxs[np.random.randint(len(pendant_frag_idxs))])

    # get attachment_idx of pendant frag and use to locate remaining hanging edge
    a = deleted_frag.GetAtomWithIdx(deleted_frag.GetSubstructMatch(DUMMY_SMARTS)[0])
    hanging_idx = a.GetProp("attachment_idx")
    remaining_frag_idx, hanging_edge_idx = _find_partner_frag_and_atom_idx(frags, hanging_idx)

    # modify remaining frag to remove hanging attachment point
    remaining_frag = frags.pop(remaining_frag_idx)
    remaining_frag = Chem.RWMol(remaining_frag)
    remaining_frag.ReplaceAtom(hanging_edge_idx, Chem.Atom(1))
    remaining_frag = remaining_frag.GetMol()
    remaining_frag = Chem.RemoveHs(remaining_frag)
    Chem.SanitizeMol(remaining_frag)

    if not strict:
        # todo neaten
        accept_mutant = True
        mutant_frag = remaining_frag
    else:
        # check if frag exists in db
        # todo change (1) and (2) into functions
        # (1) get gene_type results from DB
        gene_type = get_gene_type(remaining_frag)
        gene_type_results = list(frag_db.db.get_records("gene_types", {"gene_type": gene_type}))

        # (2) check for specific gene among results
        if len(gene_type_results) > 1:
            raise RuntimeError(f"MutationDelNode: DB is corrupted, multiple results for gene_type {gene_type}")
        elif gene_type == "":
            # if there are only two frags in original mol and we are deleting one of them gene type is ""
            # hard to check so false here (would be okay in strict=False)
            logger.debug(f"MutationDelNode: deleting frag from {Chem.MolToSmiles(parent_mol)} "
                         f"only leaves one (strict behaviour)")
            frag_exists = False
        elif len(gene_type_results) == 0:
            # this gene_type is not in the database
            logger.debug(f"MutationDelNode: hanging frag gene type ({gene_type}) not in DB (strict behaviour)")
            frag_exists = False
        else:
            # check if SMILES exist in frag store
            rf_haplotype_smi = Chem.MolToSmiles(get_haplotype_from_gene_frag(remaining_frag))
            rf_gene_smi = Chem.MolToSmiles(remaining_frag)

            frag_exists = False
            if rf_haplotype_smi in gene_type_results[0]["haplotypes"]:
                if rf_gene_smi in gene_type_results[0]["haplotypes"][rf_haplotype_smi]["gene_frags"]:
                    frag_exists = True
            if not frag_exists:
                logger.debug(f"MutationDelNode: mutated frag {rf_gene_smi} NOT found in DB (strict behaviour)")

        # (3) if mutated_frag does not exist, mutate to something that does
        if frag_exists:
            mutant_frag = remaining_frag
            accept_mutant = True
        else:
            query_frag = remaining_frag

            gene_type = get_gene_type(query_frag)  # todo we already have this above...
            mutant_smiles, mutant_scores = frag_db.query_frags(gene_type, query_frag, x_choices=0.3)
            mutant_smiles = [x for _, x in sorted(zip(mutant_scores, mutant_smiles), key=lambda pair: pair[0])]
            if not len(mutant_smiles):
                logger.debug(f"MutationDelNode: no frags with {gene_type} to mutate hanging frag to")
                return []

            accept_mutant = False
            while (accept_mutant is False) and (len(mutant_smiles) > 0):
                mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

                # align attachment_idxs of mutant frag to original reference
                mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

                # check reaction types agree
                query_idx_types = get_attachment_type_idx_pairs(query_frag)
                mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
                if query_idx_types != mutant_idx_types:
                    accept_mutant = False
                    logger.debug(f"MutationDelNode: Not matching types: {query_idx_types} {mutant_idx_types}")
                    logger.debug(f"MutationDelNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                                 f"{Chem.MolToSmiles(mutant_frag)}")
                else:
                    accept_mutant = True

    if accept_mutant:
        # reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"MutationDelNode: could not delete node from {Chem.MolToSmiles(parent_mol)}")
        return []


def substitute_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                             frag_db: FragQueryBuilder, strict: bool = True) -> List[Chem.rdchem.Mol]:
    """
    Choose a fragment from the parent molecule (fragmented using fragmentor)
    Retrieve fragments from the fragment store that have the same connectivity (gene_type)
    Align the new mutant fragment to the original fragment using the alignment fingerprint (afp)
    Reconnect the molecule incorporating the mutant fragment in place of the query fragment
    """

    # fragment parent molecule
    frags = fragmentor.get_frags(parent_mol)

    # choose a fragment to mutate
    query_frag = frags.pop(np.random.randint(len(frags)))
    gene_type = get_gene_type(query_frag)

    attempt = -1
    accept_mutant = False
    while (attempt < 5) and (accept_mutant is False):
        attempt += 1

        # retrieve new fragment from fragment store via tournament selection
        mutant_smiles, mutant_scores = frag_db.query_frags(gene_type, query_frag, x_choices=0.3)
        if not len(mutant_scores):
            logger.debug(f'MutationSubNode: nothing retrieved from db for {gene_type}')
            continue
        winning_mutant_smiles = mutant_smiles[np.argmax(mutant_scores)]
        mutant_frag = Chem.MolFromSmiles(winning_mutant_smiles)

        # mutant frag must be different from query
        if Chem.MolToSmiles(mutant_frag) == Chem.MolToSmiles(query_frag):
            logger.debug(f'MutationSubNode: skipping frag as its same as query')
            continue

        # align attachment_idxs of mutant frag to original reference
        mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

        # check reaction types agree
        query_idx_types = get_attachment_type_idx_pairs(query_frag)
        mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
        if query_idx_types != mutant_idx_types:
            accept_mutant = False
            logger.debug(f"MutationSubNode: Not matching types: {query_idx_types} {mutant_idx_types}")
            logger.debug(f"MutationSubNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                         f"{Chem.MolToSmiles(mutant_frag)}")
        else:
            accept_mutant = True

    if accept_mutant:
        # reconstruct list of frags for new molecule
        frags.append(mutant_frag)

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"MutationSubNode: no substitutions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def add_node_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                      frag_db: FragQueryBuilder, strict: bool = True) -> List[Chem.rdchem.Mol]:
    """
    Choose a random node to mutate
    Mutate to node with one too many attachment points
    Add a small attachment to the extra limb
    """

    # fragment parent
    frags = fragmentor.get_frags(parent_mol)

    # choose a random fragment to mutate
    query_frag = frags.pop(np.random.randint(len(frags)))

    # Get query fragment gene_type
    gene_type = get_gene_type(query_frag)

    # create list of all possible cut types to add (in a random order)
    cut_types = fragmentor.get_cut_list(randomize_order=True)

    # try to mutate gene_type until a suitable mutation is found
    gene_type_list = gene_type.split("#")
    accept_mutant = False
    while (accept_mutant is False) and (len(cut_types) > 0):
        # try to add new node with cut_type: added_cut_type_new_frag
        added_cut_type_new_frag = str(cut_types.pop(0))

        # retrieve small node for new pendant attachment
        # if no suitable nodes exist, continue to next cut type
        pendant_smiles_list, _ = frag_db.query_frags(added_cut_type_new_frag,
                                                     Chem.MolFromSmiles(f"[{added_cut_type_new_frag}*]C"),
                                                     x_choices=0.3)
        if len(pendant_smiles_list):
            # tournament selection for shortest smiles seq
            new_pfrag_smiles = min(pendant_smiles_list, key=len)
        else:
            logger.debug(f"MutationAddNode: No small fragments in DB with cut_type: {added_cut_type_new_frag}")
            continue

        # mutate existing fragment to contain an extra attachment point
        added_cut_type_existing_frag = _get_partner_cut(added_cut_type_new_frag, fragmentor)

        # given new cut type, try to mutate existing fragment
        mutant_gene_type_list = sorted([int(x) for x in gene_type_list if x is not ''] +
                                       [int(added_cut_type_existing_frag)])
        mutant_gene_type = "#".join([str(x) for x in mutant_gene_type_list])
        # logger.debug(f"MutationAddNode: Attempting to mutate existing gene {gene_type} -> {mutant_gene_type}")

        # query database to retrieve frags belonging to mutant_gene_type
        mutant_smiles, mutant_scores = frag_db.query_frags(mutant_gene_type, query_frag, x_choices=0.3)
        mutant_smiles = [x for _, x in sorted(zip(mutant_scores, mutant_smiles), key=lambda pair: pair[0])]

        while (accept_mutant is False) and (len(mutant_smiles) > 0):
            # try best first
            mutant_frag = Chem.MolFromSmiles(mutant_smiles.pop(0))

            # align attachment_idxs of mutant frag to original reference
            mutant_frag = match_fragment_attachment_points(mutant_frag, query_frag)

            # check aligned query and mutant cut-types agree, else continue to next candidate gene
            query_idx_types = get_attachment_type_idx_pairs(query_frag)
            mutant_idx_types = get_attachment_type_idx_pairs(mutant_frag)
            difference = mutant_idx_types - query_idx_types
            if len(difference) == 1 and int(difference.pop()[1]) == -1:
                # ignore attachment id -1 since this is the newly added edge
                accept_mutant = True
            else:
                accept_mutant = False
                logger.debug(f"MutationAddNode: Not matching types: {query_idx_types} {mutant_idx_types}")
                logger.debug(f"MutationAddNode: {Chem.MolToSmiles(parent_mol)} {Chem.MolToSmiles(query_frag)} "
                             f"{Chem.MolToSmiles(mutant_frag)}")

    if accept_mutant:
        # format added pendant frag
        new_pfrag = Chem.MolFromSmiles(new_pfrag_smiles)
        a = new_pfrag.GetAtomWithIdx(new_pfrag.GetSubstructMatch(DUMMY_SMARTS)[0])
        a.SetProp("attachment_idx", str(-1))

        # reconstruct list of frags for new molecule
        frags.extend([new_pfrag, mutant_frag])

        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        return [new_mol]
    else:
        logger.debug(f"MutationAddNode: no suitable additions were found for {Chem.MolToSmiles(query_frag)}")
        return []


def substitute_edge_mutation(parent_mol: Chem.rdchem.Mol, fragmentor: FragmentorBase,
                             frag_db: FragQueryBuilder, strict: bool = True) -> List[Chem.rdchem.Mol]:
    """
    Select pendant fragment and mutate to a different gene type
    Also mutate the gene type of the existing fragment to which the above was attached (the hanging frag)
    """

    # fragment parent molecule
    frags = fragmentor.get_frags(parent_mol)
    if len(frags) == 1:
        return []

    # identify pendant frags
    pendant_frag_idxs = _find_pendant_frag_idxs(frags)
    if not len(pendant_frag_idxs):
        return []

    # choose a pendant frag to mutate gene type
    pfrag = frags.pop(pendant_frag_idxs[np.random.randint(len(pendant_frag_idxs))])
    pfrag_gene_type = get_gene_type(pfrag)

    # get attachment_idx of pendant frag and use to locate remaining hanging edge
    a = pfrag.GetAtomWithIdx(pfrag.GetSubstructMatch(DUMMY_SMARTS)[0])
    hanging_idx = a.GetProp("attachment_idx")
    remaining_frag_idx, hanging_edge_idx = _find_partner_frag_and_atom_idx(frags, hanging_idx)

    # get hanging frag
    hfrag = frags.pop(remaining_frag_idx)
    # hfrag_gene_type = get_gene_type(hfrag)
    # todo used in strict setup to check and if not exist, mutate hfrag to real frag, same as delete

    # mutate hanging and pendant
    # could try multiple mutations here in a WHILE loop, but instead we simplify by randomizing and try first
    # todo should restrict to gene types with the same bond connection (possible joins), instead try except for now
    cut_types = fragmentor.get_cut_list(randomize_order=True)
    cut_types.remove(pfrag_gene_type)
    new_pcut_type = cut_types.pop(0)

    # mutate pfrag to a new gene type
    pmutant_smiles, pmutant_scores = frag_db.query_frags(new_pcut_type, pfrag, x_choices=0.3)
    if not len(pmutant_smiles):
        logger.debug(f"MutationSubEdge: no pmutants found for {new_pcut_type}")
        # continue to next cut_type?
        return []
    winning_pmutant_smiles = pmutant_smiles[np.argmax(pmutant_scores)]

    # find corresponding pair types associated with `new_pcut_type`
    # mutate hanging fragment to substitute corresponding pair type in
    added_hcut_type = _get_partner_cut(new_pcut_type, fragmentor)

    # mutate old to new on both hfrag and pfrag
    a = hfrag.GetAtomWithIdx(hanging_edge_idx)
    att_idx = a.GetProp('attachment_idx')
    # logger.debug(f"MutationSubEdge: mutating hanging frag {a.GetIsotope()}->{added_hcut_type}")
    a.SetIsotope(int(added_hcut_type))
    # could mutate to an actually existing frag here? Strict flag? todo

    # annotate new pfrag with attachment_idx
    pmutant_frag = Chem.MolFromSmiles(winning_pmutant_smiles)
    a = pmutant_frag.GetAtomWithIdx(pmutant_frag.GetSubstructMatch(DUMMY_SMARTS)[0])
    a.SetProp('attachment_idx', att_idx)
    # logger.debug(f"MutationSubEdge: mutating pendant frag {pfrag_gene_type}->{new_pcut_type}")

    # reconstruct list of frags for new molecule
    frags.extend([hfrag, pmutant_frag])

    try:  # restrict gene types to same bond to remove this
        # reconnect mol frags
        new_mol = connect_mol_from_frags(frags, fragmentor=fragmentor)

        # does it need the accept malarky
        return [new_mol]
    except Exception as e:
        logger.debug(f"MutationSubEdge: error {e}")
        logger.debug(f"MutationSubEdge: {Chem.MolToSmiles(parent_mol)} {winning_pmutant_smiles} {new_pcut_type}")
        return []


def single_point_crossover(m1: Chem.rdchem.Mol, m2: Chem.rdchem.Mol,
                           fragmentor: FragmentorBase) -> List[Chem.rdchem.Mol]:

    # get list of possible cuts for each molecules
    # tuple([pair of atom idxs], [pair of cut type str])
    # e.g. [([17, 16], ('5', '5')), ([5, 6], ('9', '9')), ([8, 7], ('9', '9'))]
    m1_possible_cuts = fragmentor.find_bonds(m1)
    m2_possible_cuts = fragmentor.find_bonds(m2)

    # get types in common
    m1_cut_types = [ct for _, ct in m1_possible_cuts]
    m2_cut_types = [ct for _, ct in m2_possible_cuts]
    common_cts = list(set(m1_cut_types) & set(m2_cut_types))

    # if no applicable cut types, return None
    if not len(common_cts):
        logger.debug(f"CrossoverSP: no common cut types: {Chem.MolToSmiles(m1)}, {Chem.MolToSmiles(m2)}")
        return []

    # filter to cuts in common
    m1_possible_cuts = [cut for cut in m1_possible_cuts if cut[1] in common_cts]
    m2_possible_cuts = [cut for cut in m2_possible_cuts if cut[1] in common_cts]

    # shuffle both lists to avoid systematic bias
    random.shuffle(m1_possible_cuts)
    random.shuffle(m2_possible_cuts)

    # get first mol cut
    picked_cut_1 = m1_possible_cuts.pop(0)

    # get second mol cut
    for n, cut in enumerate(m2_possible_cuts):
        if cut[1] == picked_cut_1[1]:
            picked_cut_2 = m2_possible_cuts.pop(n)
            break

    # cut both molecules using BRICS function (turns out this func is quite general!)
    # syntax here is break bond, and assign isotopes as cut types
    m1_pieces = BreakBRICSBonds(m1, [picked_cut_1])
    m2_pieces = BreakBRICSBonds(m2, [picked_cut_2])

    # separate into different mol objects
    m1_pieces = list(Chem.GetMolFrags(m1_pieces, asMols=True))
    m2_pieces = list(Chem.GetMolFrags(m2_pieces, asMols=True))

    # add attachment_idx so that frags can be reconnected (each should only have one)
    m1_pieces = [renumber_frag_attachment_idxs(frag) for frag in m1_pieces]
    m2_pieces = [renumber_frag_attachment_idxs(frag) for frag in m2_pieces]

    # crossover
    # combine 0 with 1, if bond type is wrong, try 1 and 1
    # this could be improved by working out which halves to recombine by afp
    m1_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    m2_pieces.sort(key=lambda m: m.GetNumHeavyAtoms())
    new_m1 = [m1_pieces[0], m2_pieces[1]]
    new_m2 = [m1_pieces[1], m2_pieces[0]]

    try:
        return [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)]
    except KeyError:
        logger.debug(f"CrossoverSP: asymmetric bond error: {Chem.MolToSmiles(m1)}, {Chem.MolToSmiles(m2)}")
        # todo does this introduce weirdness to nonasymmetrics? do all four?
        new_m1 = [m1_pieces[1], m2_pieces[1]]
        new_m2 = [m1_pieces[0], m2_pieces[0]]
        return [connect_mol_from_frags(new_m1, fragmentor=fragmentor),
                connect_mol_from_frags(new_m2, fragmentor=fragmentor)]


def mc_operator_factory(operator_name: str):
    """
    Mutation and crossover operator factory

    (`operator_name` must end with '_mutation' or '_crossover' since these have different inputs)
    """
    if operator_name.lower() == "substitute_node_mutation":
        return substitute_node_mutation
    elif operator_name.lower() == "add_node_mutation":
        return add_node_mutation
    elif operator_name.lower() == "delete_node_mutation":
        return delete_node_mutation
    elif operator_name.lower() == "substitute_edge_mutation":
        return substitute_edge_mutation
    elif operator_name.lower() == "single_point_crossover":
        return single_point_crossover
    else:
        raise(Exception(f"operator {operator_name} not recognised"))
