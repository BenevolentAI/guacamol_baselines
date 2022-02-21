import argparse
import os.path

import gzip
import numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm
from typing import List

from guacamol.data.get_data import extract_chembl, AllowedSmilesCharDictionary, get_raw_smiles, write_smiles
from guacamol.utils.data import download_if_not_present

RDLogger.DisableLog("rdApp.info")


FRAG_GT_ALLOWED_SYMBOLS = {
    "As", "Cs", "Rb", "Se", "se", "Si", "Sr", "Zn",
    "Ag", "Al", "Am", "Ar", "At", "Au", "D", "E", "Fe", "G", "K", "L", "M", "Ra", "Re",
    "Rf", "Rg", "Rh", "Ru", "T", "U", "V", "W", "Xe",
    "Y", "Zr", "a", "d", "f", "g", "h", "k", "m", "si", "t", "te", "u", "v", "y"
}


def standardize_smiles(raw_smiles: List[str]) -> List[str]:
    """
    Use the rdkit chembl_structure_pipeline to standardize the smiles strings
    Note: this fn is not necessary for chembl inputs
    """
    from chembl_structure_pipeline import standardizer
    std_smiles = []
    for smi in tqdm(raw_smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        standardized_m = standardizer.standardize_mol(m)
        standardized_smi = Chem.MolToSmiles(standardized_m)
        std_smiles.append(standardized_smi)
    return std_smiles


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--chembl_version", default="chembl_29", help="chembl database version (default: 29)")
    parser.add_argument("-d", "--data_root", default=".", help="root to dir containing raw chembl smiles download")
    parser.add_argument("-s", "--standardize_mols", action="store_true",
                        help="(bool) whether to use chembl pipeline to standardise mols")
    return parser


def download_chembl_smiles():
    """
    Example smiles preprocessing pipeline for chembl intended as input for FragGT fragment store creation.
    This script serves as a base for preprocessing smiles datasets with RDKit
    and can be modified to operate over other datasources of molecular structures (possibly with different formats)

    This script is adapted from GuacaMol `get_data.py` and uses a number of simple functions from that codebase
    """
    np.random.seed(1337)

    # parse args
    argparser = get_argparser()
    args = argparser.parse_args()

    # chembl variables
    print(f"Downloading {args.chembl_version}")
    chembl_chemreps_filename = f"{args.chembl_version}_chemreps.txt.gz"
    chembl_chemreps_url = f"ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/" \
                          f"{args.chembl_version}/{chembl_chemreps_filename}"
    chembl_chemreps_local = os.path.join(args.data_root, chembl_chemreps_filename)
    os.makedirs(args.data_root, exist_ok=True)

    # Download raw gzipped chemical representations file if needed
    download_if_not_present(chembl_chemreps_local, uri=chembl_chemreps_url)

    # Extract smiles from file (only simple rules to avoid overhead of parsing mols)
    print("Extracting molecules and filtering with simple string-based rules")
    raw_smiles = get_raw_smiles(chembl_chemreps_local,
                                smiles_char_dict=AllowedSmilesCharDictionary(forbidden_symbols=FRAG_GT_ALLOWED_SYMBOLS),
                                open_fn=gzip.open,
                                extract_fn=extract_chembl)

    # Standardize if requested (this fn is used by chembl, so not necessary here but option left for other sources)
    if args.standardize_mols:
        print("Standardizing molecules using the rdkit chembl structure pipeline")
        std_smiles = standardize_smiles(raw_smiles)
        print(f"Processed {len(raw_smiles)} raw smiles to produce {len(std_smiles)} standardized outputs")
    else:
        print("Skipping standardization")
        std_smiles = raw_smiles

    # Sort, shuffle and remove duplicates
    final_smiles = sorted(list(set(std_smiles)))
    np.random.shuffle(final_smiles)

    # write output smiles
    output_filename = os.path.join(args.data_root, chembl_chemreps_filename.replace(".txt.gz", "_std.smiles"))
    write_smiles(final_smiles, output_filename)
    print(f"Smiles dataset generation successful. Written to {output_filename}.")


if __name__ == "__main__":
    download_chembl_smiles()
