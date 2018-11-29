import argparse
import gzip
import os.path
import os.path
import pathlib
import pickle
import time
from typing import List, Optional

import joblib
import networkx as nx
import numpy as np
import rdkit
import scipy.stats as sps
import torch
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.helpers import setup_default_logger
from joblib import delayed
from jtnn.bo import sascorer
from jtnn.bo.sparse_gp import SparseGP
from jtnn.jtnn import JTNNVAE, Vocab, create_var
from rdkit.Chem import Descriptors, MolFromSmiles, rdmolops

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)


# We define the functions used to load and save objects
def save_object(obj, filename):
    filename = pathlib.Path(filename)
    ppath = filename.parents[0]

    if not os.path.exists(ppath):
        os.makedirs(ppath)

    result = pickle.dumps(obj)

    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret


def _score_mols_jt_vae_(valid_smiles, SA_scores_mu=None, SA_scores_std=None,
                        logP_values_mu=None, logP_values_std=None,
                        cycle_scores_mu=None, cycle_scores_std=None):

    scores = []
    SA_scores = []
    logP_values = []
    cycle_scores = []
    for i in range(len(valid_smiles)):
        current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[i]))
        current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[i]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[i]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        current_cycle_score = -cycle_length

        SA_scores.append(current_SA_score)
        logP_values.append(current_log_P_value)
        cycle_scores.append(current_cycle_score)

    if SA_scores_mu is None:
        SA_scores_mu = np.mean(SA_scores)
        SA_scores_std = 1. if sum(SA_scores) == 0 else np.std(SA_scores)
    if logP_values_mu is None:
        logP_values_mu = np.mean(logP_values)
        logP_values_std = 1. if sum(logP_values) == 0 else np.std(logP_values)
    if cycle_scores_mu is None:
        cycle_scores_mu = np.mean(cycle_scores)
        cycle_scores_std = 1. if sum(cycle_scores) == 0 else np.std(cycle_scores)

    current_SA_score_normalized = (np.array(SA_scores) - SA_scores_mu) / SA_scores_std
    current_log_P_value_normalized = (np.array(logP_values) - logP_values_mu) / logP_values_std
    current_cycle_score_normalized = (np.array(cycle_scores) - cycle_scores_mu) / cycle_scores_std

    score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized

    # target is always minued
    scores.append(-score)

    return np.array(scores)


class JT_Generator(GoalDirectedGenerator):
    def __init__(self, smiles_train_path,
                 vocab_path,
                 latent_model_path,
                 latent_size,
                 hidden_size,
                 depth,
                 bo_max_iterations,
                 bo_learning_rate=0.001,
                 ei_sample_size=60,
                 ei_keep_size=50,
                 n_inducing_points=500,
                 n_generator_iterations=1,
                 random_seed=1,
                 output_path=None,
                 use_cuda=False,
                 n_jobs=-1,
                 init_size=1000,
                 random_start=False):

        self.smiles_train_path = smiles_train_path
        self.vocab_path = vocab_path

        self.latent_model_path = latent_model_path
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth

        self.bo_max_iterations = bo_max_iterations
        self.bo_learning_rate = bo_learning_rate
        self.ei_sample_size = ei_sample_size
        self.ei_keep_size = ei_keep_size
        self.n_inducing_points = n_inducing_points
        self.n_generator_iterations = n_generator_iterations

        self.output_path = output_path
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.use_cuda = use_cuda
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.init_size = init_size
        self.random_start = random_start

        vocab = [x.strip("\r\n ") for x in open(vocab_path)]
        vocab = Vocab(vocab)

        np.random.seed(self.random_seed)

        if self.output_path is not None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

        self.latent_model = None

        if smiles_train_path:
            with open(smiles_train_path) as f:
                self.all_smiles = [s.strip() for s in f.readlines()]

        self.latent_model = JTNNVAE(vocab, hidden_size, latent_size, depth, use_cuda=use_cuda)
        device = 'cuda' if use_cuda else 'cpu'
        self.latent_model.load_state_dict(torch.load(self.latent_model_path, map_location=device))
        if use_cuda: self.latent_model = self.latent_model.cuda()
        print(f'Model is on: {device}')

    def make_latent(self, starting_population_smiles):

        start_t = time.time()

        batch_size = min(1024, len(starting_population_smiles))

        batches = [starting_population_smiles[i:i + batch_size]
                   for i in range(0, len(starting_population_smiles), batch_size)]

        latent_batches = [self._encode_latent_smiles(batch) for batch in batches]

        latent_batches = np.vstack(latent_batches)
        print(f'latent_batches: {latent_batches.shape} elapsed: {time.time() - start_t:2f}')
        return latent_batches

    def _encode_latent_smiles(self, batch) -> np.array:

        mol_vec = self.latent_model.encode_latent_mean(batch)
        latents = mol_vec.data.cpu().numpy()
        return latents

    def _create_fit_sgp_(self, X_train, y_train, n_inducing_points, X_test=None, y_test=None,
                         minibatch_size=None, max_iterations=100, learning_rate=0.001):

        sgp = SparseGP(input_means=X_train, input_vars=0 * X_train, training_targets=y_train,
                       n_inducing_points=n_inducing_points)

        sgp.train_via_ADAM(input_means=X_train,
                           input_vars=0 * X_train,
                           training_targets=y_train,
                           input_means_test=X_test,
                           input_vars_test=X_test * 0,
                           test_targets=y_test,
                           minibatch_size=n_inducing_points * 10 if minibatch_size is None else minibatch_size,
                           max_iterations=max_iterations,
                           learning_rate=learning_rate)

        return sgp

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        """
        Given an objective function, generate molecules that score as high as possible.

        Args:
            scoring_function: scoring function
            number_molecules: number of molecules to generate
            starting_population: molecules to start the optimization from (optional)

        Returns:
            A list of SMILES strings for the generated molecules.
        """
        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = np.random.choice(self.all_smiles, self.init_size)
            else:
                starting_population = self.top_k(self.all_smiles, scoring_function, self.init_size)

        X_latent = self.make_latent(starting_population)
        X_smiles = starting_population

        init_scores = self.pool((delayed(scoring_function.score)(s) for s in X_smiles))
        init_scores = -np.vstack(init_scores).reshape((-1, 1)) # (init_scores is minued!)
        print(f'init_scores: {init_scores.shape}')

        n = X_latent.shape[0]
        permutation = np.random.choice(n, n, replace=False)

        X_train = X_latent[permutation, :][0: np.int(np.round(0.9 * n)), :]
        X_test = X_latent[permutation, :][np.int(np.round(0.9 * n)):, :]

        y_train = init_scores[permutation][0: np.int(np.round(0.9 * n))]
        y_test = init_scores[permutation][np.int(np.round(0.9 * n)):]

        all_sampled_smiles = []
        all_sampled_scores = []
        for iteration in range(self.n_generator_iterations):
            # We fit the GP
            np.random.seed(iteration * self.random_seed)
            n_inducing_points = min(self.n_inducing_points, len(X_train))
            minibatch_size = min(n_inducing_points * 10, len(X_train))

            print(f'n_inducing_points {n_inducing_points}, minibatch_size {minibatch_size}, X_train {X_train.shape}')
            print(f'Train SGP')
            sgp = self._create_fit_sgp_(X_train=X_train,
                                        y_train=y_train,
                                        n_inducing_points=n_inducing_points,
                                        X_test=X_test,
                                        y_test=y_test,
                                        minibatch_size=minibatch_size,
                                        max_iterations=self.bo_max_iterations,
                                        learning_rate=self.bo_learning_rate)

            pred, uncert = sgp.predict(X_train, 0 * X_train)
            error = np.sqrt(np.mean((pred - y_train) ** 2))
            trainll = np.mean(sps.norm.logpdf(pred - y_train, scale=np.sqrt(uncert)))
            print(('Train RMSE: ', error))
            print(('Train ll: ', trainll))

            if X_test is not None:
                pred, uncert = sgp.predict(X_test, 0 * X_test)
                error = np.sqrt(np.mean((pred - y_test) ** 2))
                testll = np.mean(sps.norm.logpdf(pred - y_test, scale=np.sqrt(uncert)))
                print(('Test RMSE: ', error))
                print(('Test ll: ', testll))

            print(f'Sample {self.ei_sample_size} molecules')
            next_inputs = sgp.batched_greedy_ei(self.ei_sample_size, np.min(X_train, 0), np.max(X_train, 0))
            new_valid_smiles = []
            sampled_features = []

            print(f'Decode smiles')
            for i in range(self.ei_sample_size):
                all_vec = next_inputs[i].reshape((1, -1))
                tree_vec, mol_vec = np.hsplit(all_vec, 2)
                tree_vec = create_var(torch.from_numpy(tree_vec).float(), use_cuda=self.use_cuda)
                mol_vec = create_var(torch.from_numpy(mol_vec).float(), use_cuda=self.use_cuda)
                s = self.latent_model.decode(tree_vec, mol_vec, prob_decode=False)
                if s is not None:
                    new_valid_smiles.append(s)
                    sampled_features.append(all_vec)

            print(f'# molecules found: {len(new_valid_smiles)}')
            new_valid_smiles = new_valid_smiles[:self.ei_keep_size]
            all_sampled_smiles.append(new_valid_smiles)
            sampled_features = sampled_features[:self.ei_keep_size]
            sampled_features = np.vstack(sampled_features)

            print(f'# Score samples')
            sampled_scores = scoring_function.score_list(new_valid_smiles)
            new_y = np.array(sampled_scores)[:, None]
            all_sampled_scores.append(sampled_scores)

            if self.output_path is not None:
                print(f'# save ')
                p1 = pathlib.Path(f'{self.output_path}').joinpath('valid_smiles_{iteration}.dat')
                p2 = pathlib.Path(f'{self.output_path}').joinpath('scores_{iteration}.dat')

                save_object(new_valid_smiles, p1)
                save_object(all_sampled_scores, p2)

            if len(sampled_features) > 0:
                X_train = np.concatenate([X_train, sampled_features], 0)
                y_train = np.concatenate([y_train, new_y], 0)

        # flatten list of lists and keep best
        flat_smiles = [item for sublist in all_sampled_smiles for item in sublist]
        flat_scores = [-item for sublist in all_sampled_scores for item in sublist]
        sorted_samples = sorted(zip(flat_smiles, flat_scores), key=lambda x: x[1], reverse=False)
        best_smiles = [smiles for smiles, score in sorted_samples][:number_molecules]
        return best_smiles


if __name__ == '__main__':
    setup_default_logger()
    # code adapted from run_bo.py

    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_train_path", default=None, help="data_path")  # TODO
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--hidden", help="hidden_size", default=450, type=int)
    parser.add_argument("--latent", help="latent_size", default=56, type=int)
    parser.add_argument("--depth", help="depth", default=3, type=int)
    parser.add_argument("--seed", help="random_seed", default=1, type=int)
    parser.add_argument("--bo_max_iterations", help="bo_max_iterations", default=100, type=int)
    parser.add_argument("--bo_learning_rate", help="bo_learning_rate", default=0.001, type=float)
    parser.add_argument("--ei_sample_size", help="ei_sample_size", default=100, type=int)
    parser.add_argument("--ei_keep_size", help="ei_keep_size", default=100, type=int)
    parser.add_argument("--n_inducing_points", help="n_inducing_points", default=500, type=int)
    parser.add_argument("--n_generator_iterations", help="n_generator_iterations", default=3, type=int)
    parser.add_argument("--n_jobs", help="n_jobs", default=-1, type=int)
    parser.add_argument("--init_size", default=1000, type=int)
    parser.add_argument('--random_start', action='store_true')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.smiles_train_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.smiles_train_path = os.path.join(dir_path, 'data', 'guacamol_v1_all.smiles.1000')

    if args.vocab_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.vocab_path = os.path.join(dir_path, 'data', 'chembl_vocab.txt')

    if args.model_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.model_path = os.path.join(dir_path, 'pretrained_model', 'model.iter-6')

    args.cuda = torch.cuda.is_available()

    mol_generator = JT_Generator(smiles_train_path=args.smiles_train_path,
                                 vocab_path=args.vocab_path,
                                 latent_model_path=args.model_path,
                                 latent_size=args.latent,
                                 hidden_size=args.hidden,
                                 depth=args.depth,
                                 bo_max_iterations=args.bo_max_iterations,
                                 bo_learning_rate=args.bo_learning_rate,
                                 ei_sample_size=args.ei_sample_size,
                                 ei_keep_size=args.ei_keep_size,
                                 n_inducing_points=args.n_inducing_points,
                                 n_generator_iterations=args.n_generator_iterations,
                                 random_seed=args.seed,
                                 output_path=args.output_dir,
                                 use_cuda=args.cuda,
                                 n_jobs=args.n_jobs,
                                 random_start=args.random_start,
                                 init_size=args.init_size)

    json_file_path = os.path.join(args.output_dir, 'goal_directed_results.json')
    assess_goal_directed_generation(mol_generator, json_output_file=json_file_path)
