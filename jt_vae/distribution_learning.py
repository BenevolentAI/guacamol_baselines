import argparse
import os
from typing import List

import rdkit
import torch
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator
from guacamol.utils.helpers import setup_default_logger
from joblib import Parallel, delayed, parallel_backend
from jtnn.jtnn import JTNNVAE, Vocab

rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL)


class JtVaeSampler(DistributionMatchingGenerator):
    def __init__(self, model: JTNNVAE, n_jobs: int):
        self.model = model
        self.n_jobs = n_jobs

    def generate(self, number_samples: int) -> List[str]:
        with parallel_backend('threading', n_jobs=self.n_jobs):
            samples = Parallel()(delayed(model.sample_prior)() for _ in range(number_samples))
        return samples


if __name__ == '__main__':
    setup_default_logger()

    # Mainly imported from sample.py
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_path", default=None)
    parser.add_argument("-m", "--model_path", default=None)
    parser.add_argument("-o", "--output_dir", default=None)
    parser.add_argument("-w", "--hidden_size", default=450, type=int)
    parser.add_argument("-l", "--latent_size", default=56, type=int)
    parser.add_argument("-d", "--depth", default=3, type=int)
    parser.add_argument("--n_jobs", default=-1, type=int)
    parser.add_argument('--dist_file', default='data/guacamol_v1_all.smiles')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if args.vocab_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.vocab_path = os.path.join(dir_path, 'data', 'chembl_vocab.txt')

    if args.model_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        args.model_path = os.path.join(dir_path, 'pretrained_model', 'model.iter-6')

    args.cuda = torch.cuda.is_available()

    vocab = [x.strip() for x in open(args.vocab_path)]
    vocab = Vocab(vocab)

    model = JTNNVAE(vocab, args.hidden_size, args.latent_size, args.depth, use_cuda=args.cuda)

    device = 'cuda' if args.cuda else 'cpu'
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.cuda:
        model = model.cuda()

    torch.manual_seed(42)

    sampler = JtVaeSampler(model=model, n_jobs=args.n_jobs)

    json_file_path = os.path.join(args.output_dir, 'distribution_learning_results.json')

    assess_distribution_learning(sampler,
                                 chembl_training_file=args.dist_file,
                                 json_output_file=json_file_path)
