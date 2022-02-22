from __future__ import print_function

import argparse
import ast
import json
import logging
import numpy as np
import os
import random
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.helpers import setup_default_logger
from typing import Optional, List

from frag_gt.frag_gt import FragGTGenerator

logger = logging.getLogger(__name__)
setup_default_logger()


class FragGTGoalDirectedGenerator(FragGTGenerator, GoalDirectedGenerator):
    """ wrapper class intended to keep FragGT and GuacaMol independent """

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        return self.optimize(scoring_function=scoring_function,  # type: ignore
                             number_molecules=number_molecules,
                             starting_population=starting_population,
                             fixed_substructure_smarts=None,
                             job_name=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_file", default="data/guacamol_v1_all.smiles", help="smiles file for initial population")
    parser.add_argument("--fragstore_path", type=str, default="frag_gt/data/fragment_libraries/guacamol_v1_all_fragstore_brics.pkl")
    parser.add_argument("--allow_unspecified_stereocenters", type=bool, default=True,
                        help="if false, unspecified stereocenters will be enumerated to specific stereoisomers")
    parser.add_argument("--scorer", type=str, default="counts", help="random|counts|ecfp4|afps")
    parser.add_argument("--operators", type=ast.literal_eval, default=None,
                        help="List of tuples of (operator, prob of applying) where probabilities must add to 1")
    parser.add_argument("--population_size", type=int, default=500)
    parser.add_argument("--n_mutations", type=int, default=500)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--mapelites", type=str, default=None, help="keep elites in discretized space for diversity: species|mwlogp")
    # parser.add_argument("--write_all_generations", type=bool, default=False,
    #                     help="if true, all intermediate generations will be written to the output directory")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--random_start", action="store_true", help="sample initial population instead of scoring and taking top k")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--suite", default="v2", help="version of the guacamol benchmark suite to run")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # by default, write to same directory as this file
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    # # setup writing directory for intermediate results (requires modification of guacamol, coming soon)
    # intermediate_results_dir: Optional[str] = None
    # if args.write_all_generations:
    #     intermediate_results_dir = os.path.join(args.output_dir, "generations/")
    #     os.makedirs(intermediate_results_dir, exist_ok=True)
    #     logger.info(f"writing intermediate generation molecules to {intermediate_results_dir}")

    # save command line args
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "goal_directed_params.json"), "w") as jf:
        json.dump(vars(args), jf, sort_keys=True, indent=4)

    optimizer = FragGTGoalDirectedGenerator(smi_file=args.smiles_file,
                                            fragmentation_scheme="brics",
                                            fragstore_path=args.fragstore_path,
                                            allow_unspecified_stereo=args.allow_unspecified_stereocenters,
                                            scorer=args.scorer,
                                            operators=args.operators,
                                            population_size=args.population_size,
                                            n_mutations=args.n_mutations,
                                            generations=args.generations,
                                            map_elites=args.mapelites,
                                            random_start=args.random_start,
                                            patience=args.patience,
                                            n_jobs=args.n_jobs,
                                            intermediate_results_dir=None)

    json_file_path = os.path.join(args.output_dir, "goal_directed_results.json")
    assess_goal_directed_generation(optimizer, json_output_file=json_file_path, benchmark_version=args.suite)


if __name__ == "__main__":
    main()
