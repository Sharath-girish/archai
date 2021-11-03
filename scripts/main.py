# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from typing import Dict, Type

from archai.common import utils
from archai.nas.exp_runner import ExperimentRunner
from archai.algos.darts.darts_exp_runner import DartsExperimentRunner
from archai.algos.petridish.petridish_exp_runner import PetridishExperimentRunner
from archai.algos.random.random_exp_runner import RandomExperimentRunner
from archai.algos.manual.manual_exp_runner import ManualExperimentRunner
from archai.algos.xnas.xnas_exp_runner import XnasExperimentRunner
from archai.algos.gumbelsoftmax.gs_exp_runner import GsExperimentRunner
from archai.algos.divnas.divnas_exp_runner import DivnasExperimentRunner
from archai.algos.didarts.didarts_exp_runner import DiDartsExperimentRunner
from archai.algos.proxynas.freeze_darts_space_experiment_runner import FreezeDartsSpaceExperimentRunner
from archai.algos.proxynas.freeze_natsbench_experiment_runner import FreezeNatsbenchExperimentRunner
from archai.algos.proxynas.freeze_natsbench_sss_experiment_runner import FreezeNatsbenchSSSExperimentRunner
from archai.algos.proxynas.freeze_nasbench101_experiment_runner import FreezeNasbench101ExperimentRunner
from archai.algos.proxynas.freeze_manual_experiment_runner import ManualFreezeExperimentRunner
from archai.algos.naswotrain.naswotrain_natsbench_conditional_experiment_runner import NaswotConditionalNatsbenchExperimentRunner
from archai.algos.natsbench.natsbench_regular_experiment_runner import NatsbenchRegularExperimentRunner
from archai.algos.natsbench.natsbench_sss_regular_experiment_runner import NatsbenchSSSRegularExperimentRunner
from archai.algos.nasbench101.nasbench101_exp_runner import Nb101RegularExperimentRunner
from archai.algos.proxynas.phased_freeze_natsbench_experiment_runner import PhasedFreezeNatsbenchExperimentRunner
from archai.algos.proxynas.freezeaddon_nasbench101_experiment_runner import FreezeAddonNasbench101ExperimentRunner
from archai.algos.zero_cost_measures.zero_cost_natsbench_experiment_runner import ZeroCostNatsbenchExperimentRunner
from archai.algos.zero_cost_measures.zero_cost_natsbench_conditional_experiment_runner import ZeroCostConditionalNatsbenchExperimentRunner
from archai.algos.zero_cost_measures.zero_cost_natsbench_epochs_experiment_runner import ZeroCostNatsbenchEpochsExperimentRunner
from archai.algos.random_natsbench.random_natsbench_tss_far_exp_runner import RandomNatsbenchTssFarExpRunner
from archai.algos.random_natsbench.random_natsbench_tss_far_post_exp_runner import RandomNatsbenchTssFarPostExpRunner
from archai.algos.random_natsbench.random_natsbench_tss_reg_exp_runner import RandomNatsbenchTssRegExpRunner
from archai.algos.random_darts.random_dartsspace_reg_exp_runner import RandomDartsSpaceRegExpRunner
from archai.algos.random_darts.random_dartsspace_far_exp_runner import RandomDartsSpaceFarExpRunner
from archai.algos.local_search_natsbench.local_natsbench_tss_far_exp_runner import LocalNatsbenchTssFarExpRunner
from archai.algos.local_search_natsbench.local_search_natsbench_tss_fear_exp_runner import LocalSearchNatsbenchTSSFearExpRunner
from archai.algos.local_search_natsbench.local_search_natsbench_tss_reg_exp_runner import LocalSearchNatsbenchTSSRegExpRunner
from archai.algos.local_search_darts.local_search_darts_reg_exp_runner import LocalSearchDartsRegExpRunner

def main():
    runner_types:Dict[str, Type[ExperimentRunner]] = {
        'darts': DartsExperimentRunner,
        'petridish': PetridishExperimentRunner,
        'xnas': XnasExperimentRunner,
        'random': RandomExperimentRunner,
        'manual': ManualExperimentRunner,
        'gs': GsExperimentRunner,
        'divnas': DivnasExperimentRunner,
        'didarts': DiDartsExperimentRunner,
        'proxynas_darts_space': FreezeDartsSpaceExperimentRunner,
        'proxynas_natsbench_space': FreezeNatsbenchExperimentRunner,
        'proxynas_natsbench_sss_space': FreezeNatsbenchSSSExperimentRunner,
        'proxynas_nasbench101_space': FreezeNasbench101ExperimentRunner,
        'proxynas_manual': ManualFreezeExperimentRunner,
        'naswot_conditional_natsbench_space': NaswotConditionalNatsbenchExperimentRunner,
        'zerocost_natsbench_space': ZeroCostNatsbenchExperimentRunner,
        'zerocost_conditional_natsbench_space': ZeroCostConditionalNatsbenchExperimentRunner,
        'zerocost_natsbench_epochs_space': ZeroCostNatsbenchEpochsExperimentRunner,
        'natsbench_regular_eval': NatsbenchRegularExperimentRunner,
        'natsbench_sss_regular_eval': NatsbenchSSSRegularExperimentRunner,
        'nb101_regular_eval': Nb101RegularExperimentRunner,
        'phased_freezetrain_natsbench_space': PhasedFreezeNatsbenchExperimentRunner,
        'freezeaddon_nasbench101_space': FreezeAddonNasbench101ExperimentRunner,
        'random_natsbench_tss_far': RandomNatsbenchTssFarExpRunner,
        'random_natsbench_tss_far_post': RandomNatsbenchTssFarPostExpRunner,
        'random_natsbench_tss_reg': RandomNatsbenchTssRegExpRunner,
        'random_dartsspace_reg': RandomDartsSpaceRegExpRunner,
        'random_dartsspace_far': RandomDartsSpaceFarExpRunner,
        'local_natsbench_tss_far': LocalNatsbenchTssFarExpRunner,
        'local_search_natsbench_tss_reg': LocalSearchNatsbenchTSSRegExpRunner,
        'local_search_natsbench_tss_fear': LocalSearchNatsbenchTSSFearExpRunner,
        'local_search_darts_reg': LocalSearchDartsRegExpRunner
    }

    parser = argparse.ArgumentParser(description='NAS E2E Runs')
    parser.add_argument('--algos', type=str, default='''darts,
                                                        xnas,
                                                        random,
                                                        didarts,
                                                        petridish,
                                                        gs,
                                                        manual,
                                                        divnas,
                                                        proxynas_manual,
                                                        proxynas_darts_space,
                                                        proxynas_natsbench_space,
                                                        proxynas_natsbench_sss_space,
                                                        proxynas_nasbench101_space,
                                                        freezeaddon_nasbench101_space,
                                                        naswot_conditional_natsbench_space,
                                                        zerocost_natsbench_space,
                                                        zerocost_conditional_natsbench_space,
                                                        zerocost_natsbench_epochs_space,
                                                        natsbench_regular_eval,
                                                        natsbench_sss_regular_eval,
                                                        nb101_regular_eval,
                                                        phased_freezetrain_natsbench_space,
                                                        random_natsbench_tss_far,
                                                        random_natsbench_tss_far_post,
                                                        random_natsbench_tss_reg,
                                                        random_dartsspace_reg,
                                                        random_dartsspace_far,
                                                        local_natsbench_tss_far''',
                        help='NAS algos to run, separated by comma')
    parser.add_argument('--datasets', type=str, default='cifar10',
                        help='datasets to use, separated by comma')
    parser.add_argument('--full', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Run in full or toy mode just to check for compile errors')
    parser.add_argument('--no-search', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Do not run search')
    parser.add_argument('--no-eval', type=lambda x:x.lower()=='true',
                        nargs='?', const=True, default=False,
                        help='Do not run eval')
    parser.add_argument('--exp-prefix', type=str, default='throwaway',
                        help='Experiment prefix is used for directory names')
    args, extra_args = parser.parse_known_args()

    if '--common.experiment_name' in extra_args:
        raise RuntimeError('Please use --exp-prefix instead of --common.experiment_name so that main.py can generate experiment directories with search and eval suffix')

    for dataset in args.datasets.split(','):
        for algo in args.algos.split(','):
            algo = algo.strip()
            print('Running (algo, dataset): ', (algo, dataset))
            runner_type:Type[ExperimentRunner] = runner_types[algo]

            # get the conf files for algo and dataset
            algo_conf_filepath = f'confs/algos/{algo}.yaml' if args.full \
                                               else f'confs/algos/{algo}_toy.yaml'
            dataset_conf_filepath = f'confs/datasets/{dataset}.yaml'
            conf_filepaths = ';'.join((algo_conf_filepath, dataset_conf_filepath))

            runner = runner_type(conf_filepaths,
                                base_name=f'{algo}_{dataset}_{args.exp_prefix}',
                                # for toy and debug runs, clean exp dirs
                                clean_expdir=utils.is_debugging() or not args.full)

            runner.run(search=not args.no_search, eval=not args.no_eval)


if __name__ == '__main__':
    main()
