import argparse
import os

from rdkit import Chem
from tqdm import tqdm
from typing import List, Optional

from frag_gt.src.fragmentors import fragmentor_factory, FragmentorBase
from frag_gt.src.fragstore import fragstore_factory
from frag_gt.src.gene_type_utils import get_gene_type, get_haplotype_from_gene_frag
from frag_gt.src.io import valid_mols_from_smiles, load_smiles_from_file


class FragmentStoreCreator:
    """ class to orchestrate creation of fragment stores from input smiles file """
    def __init__(self, frag_scheme: str):

        # retrieve fragmentor from available e.g. brics fragmentor
        self.fragmentor = fragmentor_factory(frag_scheme)

        # We only support an "in_memory" fragment store in the code release.
        # If a DB is desired these arguments allow easy extension
        # i.e. fragstore_type="mongodb", fragstore_path="db_name"
        self.fragstore_type = "in_memory"

        # retrieve fragstore object, this knows how to read and write from fragstore
        self.frag_db = fragstore_factory(self.fragstore_type, "no path needed since this is a blank slate")

        # single processor takes 2.1 hours on chemblv24
        self.n_jobs = 1

    @staticmethod
    def smi2mol(smi: str) -> Optional[Chem.rdchem.Mol]:
        return Chem.MolFromSmiles(smi)

    def create_gene_table(self, smiles_file: str):
        """ creates gene database from input smiles file. """

        # read smiles file
        smiles_list = load_smiles_from_file(smiles_file)
        print(f"{len(smiles_list)} smiles read from file")

        # parse mols
        print(f"parsing mols using {self.n_jobs} threads")
        mol_list = valid_mols_from_smiles(smiles_list, self.n_jobs)

        # create fragment (gene) records
        print(f"converting mols to genes and storing in fragstore")
        if self.n_jobs == 1:
            for mol in tqdm(mol_list):
                records_for_single_smiles = self.genes_from_parent_mol(mol, self.fragmentor)

                if len(records_for_single_smiles):
                    self.frag_db.add_records("genes", records_for_single_smiles)

        total_gene_count = self.frag_db.get_records("genes", {}, return_count=True)
        print(f"Gene (fragment) database finished loading (n={total_gene_count})")

    @staticmethod
    def genes_from_parent_mol(mol: Chem.rdchem.Mol, fragmentor: FragmentorBase) -> List[dict]:

        # get genes (frags) for smiles
        frags = fragmentor.get_frags(mol)

        # construct a json-like object for each frag to be stored in fragstore
        records_for_single_mol = []
        if len(frags) > 1:
            for frag in frags:
                gene_type = get_gene_type(frag)
                record = {
                    "gene_frag_smiles": Chem.MolToSmiles(frag),
                    "hap_frag_smiles": Chem.MolToSmiles(get_haplotype_from_gene_frag(frag)),
                    "parent_smiles": Chem.MolToSmiles(mol),
                    "gene_type": gene_type
                }
                records_for_single_mol.append(record)
        else:
            # if parent has zero cut points, an empty list will be returned
            pass

        return records_for_single_mol

    def create_gene_type_table(self):
        """ Group genes by gene type since this is how they are accessed at runtime. """

        # retrieve genes from gene table
        genes = self.frag_db.get_records("genes", {})

        gene_types = {}
        for n, gene in enumerate(genes):
            # gene is a json-like object e.g.
            # {"_id": ObjectId("5d108874fddeccd17661992c"), "gene_frag_smiles": "[4*]NC(=O)CCC",
            #  "hap_frag_smiles": "CCCC(N)=O", "parent_smiles": "CCCC(=O)NNC(=O)NC1=CC=CC=C1", "gene_type": "4"

            # Get dict so far for this gene type
            gt = gene_types.get(gene["gene_type"], {})

            # within the gene type, genes are grouped by their haplotype (frag without attachments)
            hap = gt.get(gene["hap_frag_smiles"], {'gene_frags': {}})

            # within each haplotype, gene_frags contains the specific genes along with their occurrence frequency
            g = hap["gene_frags"].get(gene["gene_frag_smiles"], {"count": 0})
            count = int(g["count"]) + 1
            hap["gene_frags"][gene["gene_frag_smiles"]] = {"count": count}

            # update gene_types table
            gt[gene["hap_frag_smiles"]] = hap
            gene_types[gene["gene_type"]] = gt

        records = [{"gene_type": gt, "haplotypes": haps} for gt, haps in gene_types.items()]
        self.frag_db.add_records("gene_types", records)

        print(f"Gene type database finished loading (n gene_types={len(gene_types)})")

    def save_fragstore_to_disc(self, path: str):
        self.frag_db.save(path)
        print(f"saved fragstore of type: {self.fragstore_type}, name: {path}")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/smiles_files/chembl_29_chemreps_std.smiles")
    parser.add_argument("--output_dir", type=str, help="directory to output .pkl for 'in_memory' fragment store")
    parser.add_argument("--frag_scheme", type=str, default="brics")
    return parser


def main():
    args = get_arg_parser().parse_args()

    db_generator = FragmentStoreCreator(frag_scheme=args.frag_scheme)

    # create gene (fragment) database from smiles file
    print("Starting to load gene (fragment) database")
    db_generator.create_gene_table(smiles_file=args.smiles_file)

    # group genes by gene_type
    print("Starting to generate gene_type table, grouping each frag by its gene type")
    db_generator.create_gene_type_table()

    # save to disc
    output_name = os.path.basename(args.smiles_file).split('.')[0] + f"_fragstore_{args.frag_scheme}.pkl"
    output_path = os.path.join(args.output_dir, output_name)
    db_generator.save_fragstore_to_disc(output_path)

    print(f"Finished. Writing {output_path}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
