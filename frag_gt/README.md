# FragGT


FragGT is a fragment-based evolutionary algorithm for generating molecules.


#### Install

```bash
conda create -y --name fraggt python=3.7
pip install .
```

To verify installation was successful, run the unit testing suite using `$ pytest frag_gt`

Optional dependencies required to create custom fragment stores:
```bash
conda install -c conda-forge chembl_structure_pipeline
pip install guacamol
```

#### Fragstore download

FragGT includes a small set of precomputed fragments for testing, for real world applications most people will want to use a larger fragment store.
We provide precomputed fragment stores for convenience, along with code for generating custom fragment stores (see below).
Precomputed fragstores can be downloaded from Zenodo (by default FragGT expects `data/` in the top-level of the frag-gt directory, alongside this README):
```
wget https://zenodo.org/record/6038464/files/frag_gt.zip?download=1 -O frag_gt.zip
unzip frag_gt.zip
rm frag_gt.zip
```

#### Generate molecules

Given an objective scoring function, FragGT can generate molecules with just a few lines of code!

```python
from frag_gt.frag_gt import FragGTGenerator
from frag_gt.src.scorers import MolecularWeightScorer

# lightweight generator for prototyping (else just use defaults: `generator = FragGTGenerator()`)
generator = FragGTGenerator(generations=5,
                            population_size=20,
                            n_mutations=20,
                            allow_unspecified_stereo=True,
                            smi_file='frag_gt/tests/test_data/sample.smi')

scoring_function = MolecularWeightScorer()

output_smis = generator.optimize(scoring_function=scoring_function,
                                 number_molecules=10,
                                 starting_population=None)
```

#### Defining scoring functions

FragGT is compatible with GuacaMol scoring functions however we provide a minimal scorer class to decouple the guacamol and frag-gt libraries.
In practice the guacamol scoring functions are more powerful. Unfortunately we cannot support an extensive library of scoring functions at this time.

The `MolecularWeightScorer()` class is provided as an example scorer.
An additional custom scorer that can be used optimize molecules towards high cLogP is shown below:

```python
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from frag_gt.src.scorers import SmilesScorer


class MyCustomScorer(SmilesScorer):
    def score(self, smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Invalid mol in scorer: {smiles}")

        return MolLogP(mol)
```

#### Create a new fragment store

The fragment store is the set of fragments used to generate molecules. 
We provide a default fragment store based on chembl v29.
We also provide a fragstore generated from the guacamol training set which was used for guacamol experiments.

If you would like to generate a new fragment store there are three stages: 
- a. Download a set of SMILES that will be the parents of the fragments
- b. Generate a fragment store from the smiles file
- c. (optional) Filter fragments in the store based on the frequency of occurrence in the corpus

##### a. Download SMILES

We start from a smiles file, this can come from any source. We provide code for downloading all SMILES from ChEMBL.
The script uses the [ChEMBL structure pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline) (version 1.0.0)
to standardize SMILES so if required you should install using `conda`, then from the root of this directory:

```bash
python -m frag_gt.fragstore_scripts.download_chembl_smiles -v chembl_29 -d data/smiles_files -s  # ~45 mins
```
This will write to `data/smiles_files/chembl_29_chemreps_std.smiles`

##### b. Create fragstore

Now we can create a new fragment store from the `.smi` file using the smiles file from (a)

```bash
python -m frag_gt.fragstore_scripts.generate_fragstore --smiles_file data/smiles_files/chembl_29_chemreps_std.smiles --output_dir data/fragment_libraries  # ~2 hrs
```

##### c. Filter fragstore

We provide code for filtering fragstores based on the frequency of fragment occurence in the corpus.
To remove all fragments with fewer than two occurences in the fragstore:

```bash
python -m frag_gt.fragstore_scripts.filter_fragstore --fragstore_path data/fragment_libraries/chembl_29_chemreps_std_fragstore_brics.pkl --frequency_cutoff 2
```

#### GuacaMol benchmarks

To reproduce the guacamol goal directed benchmark, run from `guacamol_baselines` root:

```bash
python frag_gt/goal_directed_generation.py --fragstore_path frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl --smiles_file data/guacamol_v1_all.smiles
```

#### Analyse molecule generation

To analyse the trajectory of molecule generation, you will need to direct FragGT to save molecules from all generations.
This can be activated by providing:
- an `intermediate_results_dir` directory in the initialization of the `FragGTGenerator`
- `job_name` the `generate_optimized_molecules` method

```bash
optimizer = FragGTGenerator(intermediate_results_dir=tmpdir)
output_smis = optimizer.generate_optimized_molecules(scoring_function=scoring_function,
                                                     number_molecules=number_of_requested_molecules,
                                                     starting_population=None,
                                                     job_name=job_name)
```

This will produce outputs per generation in the `intermediate_results_dir`.

#### Component diagram

The FragGT contains a number of modules. An overview of the components can be seen below:

![DeeplyTough overview figure](frag-gt-component-diagram.png?raw=true "FragGT component diagram.")

#### Licence

This project is licensed under the terms of the MIT license.