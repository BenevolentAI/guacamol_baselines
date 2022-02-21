from frag_gt.src.gene_type_utils import get_species, get_gene_type, get_haplotype_from_gene_frag
from rdkit import Chem


def test_get_species():
    # Given
    # parent: "CCCC(=O)NNC(=O)Nc1ccccc1"
    mol_frags = [Chem.MolFromSmiles(x) for x in ["[1*]C(=O)NNC(=O)CCC", "[5*]N[5*]", "[16*]c1ccccc1"]]

    # When
    species = get_species(mol_frags)

    # Then
    assert species == "5#5.1.16"


def test_get_gene_type():
    # Given
    frag1 = Chem.MolFromSmiles("[2*]c1cc(N[1*])c2c(n1)C([3*])CNC([4*])C2")
    frag2 = Chem.MolFromSmiles("[6*]NC(=O)CCC[7*]")

    # When
    gene_type1 = get_gene_type(frag1)
    gene_type2 = get_gene_type(frag2)

    # Then
    assert gene_type1 == "1#2#3#4"
    assert gene_type2 == "6#7"


def test_get_haplotype_from_gene_frag():
    # Given
    gene_frag = Chem.MolFromSmiles("[1*]C(=O)NNC(=O)CCC")

    # When
    haplotype_frag = get_haplotype_from_gene_frag(gene_frag)

    # Then
    assert any([a.GetSymbol() == "*" for a in gene_frag.GetAtoms()])
    assert all([a.GetSymbol() != "*" for a in haplotype_frag.GetAtoms()])
