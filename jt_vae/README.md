# Junction Tree Variational Autoencoder for Molecular Graph Generation

## Dendencies
To install the dependencies run:
```commandline
pip install -r requirements.txt
```

## Benchmarking
A model trained on the `guacamol` dataset is provided in this repo in the directory: `jtnn_model` .  

### Distribution
```commandline

```

### Goal-Directed
```commandline
python guacamol_run_bo.py --latent_model_path jtnn_model/latent_model --vocab data/chembl_vocab.txt --smiles_train_path TODO
```




## Model Training
If you'd like to train it from scratch follow these instructions:

### Vocab Building
```
python -m jtnn.jtnn.mol_tree < TODO > data/vocab.txt
```

### Pre-Training
This pre-trains a model ... TODO
```commandline

``` 
### Fine-Tuning
This fine-tune a model ... TODO
```commandline

```
