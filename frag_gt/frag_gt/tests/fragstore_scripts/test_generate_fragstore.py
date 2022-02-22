import os

from rdkit import Chem

from frag_gt.fragstore_scripts.generate_fragstore import FragmentStoreCreator
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.tests.utils import SAMPLE_SMILES_FILE


def test_create_gene_table(tmp_path):
    # Given
    sample_smiles_file = SAMPLE_SMILES_FILE
    fragstore_output_dir = tmp_path / "output_dir"
    fragstore_output_dir.mkdir()
    fragstore_path = fragstore_output_dir / "temp_fragstore.pkl"

    # When
    db_creator = FragmentStoreCreator(frag_scheme="brics")
    db_creator.create_gene_table(smiles_file=str(sample_smiles_file))
    db_creator.create_gene_type_table()
    db_creator.save_fragstore_to_disc(str(fragstore_path))

    reloaded_db = fragstore_factory("in_memory", str(fragstore_path))
    reloaded_db.load()

    # Then
    num_genes = db_creator.frag_db.get_records(query={}, collection='genes', return_count=True)
    assert num_genes == 516
    assert os.path.exists(fragstore_path)
    assert len(reloaded_db.store["gene_types"])


def test_genes_from_parent_mol():
    # Given
    parent_mol = Chem.MolFromSmiles("CCSc1nnc(NC(=O)CCCOc2ccc(C)cc2)s1")
    db_generator = FragmentStoreCreator(frag_scheme="brics")

    # When
    mol_genes = db_generator.genes_from_parent_mol(parent_mol, fragmentor=db_generator.fragmentor)

    # Then
    assert len(mol_genes) == 7
    assert mol_genes[0] == {
        "gene_frag_smiles": "[4*]CC",
        "hap_frag_smiles": "CC",
        "parent_smiles": "CCSc1nnc(NC(=O)CCCOc2ccc(C)cc2)s1",
        "gene_type": "4"
    }
    assert len(set([x["parent_smiles"] for x in mol_genes])) == 1


def test_genes_from_parent_mol_multi():
    # Given
    parent_smiles = ["CCSc1nnc(NC(=O)CCCOc2ccc(C)cc2)s1", "CCCC(=O)NNC(=O)Nc1ccccc1"]
    parent_mols = [Chem.MolFromSmiles(x) for x in parent_smiles]
    db_generator = FragmentStoreCreator(frag_scheme="brics")

    # When
    all_genes = []
    for mol in parent_mols:
        mol_genes = db_generator.genes_from_parent_mol(mol, fragmentor=db_generator.fragmentor)
        all_genes.extend(mol_genes)

    # Then
    assert len(all_genes) == 10
    assert all_genes[0] == {
        "gene_frag_smiles": "[4*]CC",
        "hap_frag_smiles": "CC",
        "parent_smiles": "CCSc1nnc(NC(=O)CCCOc2ccc(C)cc2)s1",
        "gene_type": "4"
    }
    assert len(set([x["parent_smiles"] for x in all_genes])) == 2
