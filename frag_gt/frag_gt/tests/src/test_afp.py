import numpy as np
from frag_gt.src.afp import compare_alignment_fps, create_alignment_fp, renumber_frag_attachment_idxs, \
    match_fragment_attachment_points, calculate_alignment_similarity_scores
from rdkit import Chem


def test_create_alignment_fp():
    # Given
    frags = ["[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2", "c1cc([1*])ccc1"]
    prepared_frags = []
    for f in frags:
        mol_frag = renumber_frag_attachment_idxs(Chem.MolFromSmiles(f))
        prepared_frags.append(mol_frag)

    # When
    afp_list = [create_alignment_fp(f) for f in prepared_frags]

    # Then
    expected_afp_lens = [4, 1]
    assert [len(a) for a in afp_list] == expected_afp_lens


def test_compare_alignment_fps():
    # Given
    m1 = renumber_frag_attachment_idxs(Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2"))
    afp1 = create_alignment_fp(m1)
    m2 = renumber_frag_attachment_idxs(Chem.MolFromSmiles("[1*]c1cc(N[3*])c2c(n1)C([2*])CCCC2"))
    afp2 = create_alignment_fp(m2)

    # When
    alignment, score = compare_alignment_fps(afp1, afp2)

    # Then
    expected = {0: 0, 1: 1, 2: 2, 3: -1}
    assert alignment == expected
    assert score > 0


def test_match_fragment_attachment_points():
    # Given
    reference_frag_with_attachment_idxs = renumber_frag_attachment_idxs(
        Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2"))
    new_fragment_to_align = Chem.MolFromSmiles("[1*]c1cc(N[3*])c2c(n1)C([2*])CCCC2")

    # When
    aligned_frag = match_fragment_attachment_points(new_fragment_to_align, reference_frag_with_attachment_idxs)

    # Then
    aligned_attachment_ids = [a.GetProp("attachment_idx") for a in aligned_frag.GetAtoms() if a.GetSymbol() == "*"]
    original_attachment_ids = [a.GetProp("attachment_idx") for a in reference_frag_with_attachment_idxs.GetAtoms()
                               if a.GetSymbol() == "*"]
    assert aligned_attachment_ids != original_attachment_ids


def test_calculate_alignment_similarity_scores():
    # Given
    query_frag_mol = Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2")
    frag_smiles = ["[1*]c1cc(N[3*])c2c(n1)C([2*])CCCC2", "[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2"]

    # When
    scores = calculate_alignment_similarity_scores(query_frag_mol, frag_smiles)

    # Then
    assert len(scores) == len(frag_smiles)
    assert np.equal(scores[1], 1.0 * frag_smiles[1].count('*'))
