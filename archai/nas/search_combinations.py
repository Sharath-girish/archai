# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Iterator, Mapping, Type, Optional, Tuple, List
import math
import copy
import random
import os

import torch
import tensorwatch as tw
from torch.utils.data.dataloader import DataLoader
import yaml

from archai.common.common import logger
from archai.common.checkpoint import CheckPoint
from archai.common.config import Config
from archai.nas.model_desc_builder import ModelDescBuilder
from archai.nas.arch_trainer import TArchTrainer
from archai.nas import nas_utils
from archai.nas.model_desc import CellType, ModelDesc
from archai.common.trainer import Trainer
from archai.datasets import data
from archai.nas.model import Model
from archai.common.metrics import EpochMetrics, Metrics
from archai.common import utils
from archai.nas.finalizers import Finalizers
from archai.nas.search import ModelMetrics, Search, SearchResult



class SearchCombinations(Search):
    def search(self, conf_search:Config, model_desc_builder:ModelDescBuilder,
               trainer_class:TArchTrainer, finalizers:Finalizers)->SearchResult:

        # region config vars
        conf_model_desc = conf_search['model_desc']
        final_desc_filename = conf_search['final_desc_filename']
        conf_postsearch_train = conf_search['post_train']

        conf_pareto = conf_search['pareto']
        pareto_enabled = conf_pareto['enabled']
        pareto_summary_filename = conf_pareto['summary_filename']

        min_cells = conf_model_desc['n_cells']
        min_reductions = conf_model_desc['n_reductions']
        min_nodes = conf_model_desc['cell']['n_nodes']
        max_cells = conf_pareto['max_cells']
        max_reductions = conf_pareto['max_reductions']
        max_nodes = conf_pareto['max_nodes']

        conf_checkpoint = conf_search['checkpoint']
        resume = conf_search['resume']

        self._metrics_dir = conf_search['metrics_dir']
        # endregion


        self._summary_filepath = utils.full_path(pareto_summary_filename)
        self._checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)

        logger.info({'pareto_enabled': pareto_enabled,
                     'min_reductions': min_reductions,
                     'min_cells': min_cells,
                     'min_nodes': min_nodes,
                     'max_reductions': max_reductions,
                     'max_cells': max_cells,
                     'max_nodes': max_nodes
                     })

        macro_combinations = list(self.get_combinations())
        start_macro_i, best_search_result = self.restore_checkpoint(macro_combinations)
        best_macro_comb = -1,-1,-1 # reductions, cells, nodes

        for macro_comb_i in range(start_macro_i, len(macro_combinations)):
            reductions, cells, nodes = macro_combinations[macro_comb_i]
            logger.pushd(f'r{reductions}.c{cells}.n{nodes}')

            # build model description that we will search on
            model_desc = self.build_model_desc(model_desc_builder, conf_model_desc,
                                               reductions, cells, nodes)

            # perform search on model description
            model_desc, search_metrics = self.search_model_desc(conf_search,
                model_desc, trainer_class, finalizers)

            # train searched model for few epochs to get some perf metrics
            model_metrics = self.train_model_desc(model_desc,
                                                  conf_postsearch_train)

            assert model_metrics is not None, "'post_train' section in yaml should have non-zero epochs if running combinations search"

            # save result
            self.save_trained(reductions, cells, nodes, model_metrics)

            # update the best result so far
            if self.is_better_metrics(best_search_result.search_metrics,
                                      model_metrics.metrics):
                best_search_result = SearchResult(model_desc, search_metrics,
                                                  model_metrics.metrics)
                best_macro_comb = reductions, cells, nodes

            # checkpoint
            assert best_search_result is not None
            self.record_checkpoint(macro_comb_i, best_search_result)
            logger.popd() # reductions, cells, nodes

        assert best_search_result is not None
        self.clean_log_result(best_search_result, final_desc_filename)
        logger.info({'best_macro_comb':best_macro_comb})

        return best_search_result

    def is_better_metrics(self, metrics1:Optional[Metrics],
                          metrics2:Optional[Metrics])->bool:
        if metrics1 is None or metrics2 is None:
            return True
        return metrics2.best_val_top1() >= metrics1.best_val_top1()

    def restore_checkpoint(self, macro_combinations)\
            ->Tuple[int, Optional[SearchResult]]:
        # if checkpoint is available then restart from last combination we were running
        checkpoint_avail = self._checkpoint is not None
        resumed, state = False, None
        start_macro_i, best_result = 0, None
        if checkpoint_avail:
            state = self._checkpoint.get('search', None)
            if state is not None:
                start_macro_i = state['start_macro_i']
                assert start_macro_i >= 0 and start_macro_i < len(macro_combinations)

                best_result = yaml.load(state['best_result'], Loader=yaml.Loader)

                start_macro_i += 1 # resume after the last checkpoint
                resumed = True

        if not resumed:
            # erase previous file left over from run
            utils.zero_file(self._summary_filepath)

        logger.warn({'resumed': resumed, 'checkpoint_avail': checkpoint_avail,
                     'checkpoint_val': state is not None,
                     'start_macro_i': start_macro_i,
                     'total_macro_combinations': len(macro_combinations)})
        return start_macro_i, best_result

    def record_checkpoint(self, macro_comb_i:int, best_result:SearchResult)->None:
        if self._checkpoint is not None:
            state = {'start_macro_i': macro_comb_i,
                     'best_result': yaml.dump(best_result)}
            self._checkpoint.new()
            self._checkpoint['search'] = state
            self._checkpoint.commit()

    def get_combinations(self)->Iterator[Tuple[int, int, int]]:
        if not self.pareto_enabled:
            yield self.min_reductions, self.min_cells, self.min_nodes
        else:
            # TODO: what happens when reductions is 3 but cells is 2? have to step
            # through code and check
            for reductions in range(self.min_reductions, self.max_reductions+1):
                for cells in range(self.min_cells, self.max_cells+1):
                    for nodes in range(self.min_nodes, self.max_nodes+1):
                        yield reductions, cells, nodes

    def save_trained(self, reductions:int, cells:int, nodes:int,
                      model_metrics:ModelMetrics)->None:
        """Save the model and metric info into a log file"""

        # construct path where we will save
        subdir = utils.full_path(self._metrics_dir.format(**vars()), create=True)

        model_stats = self.get_model_stats(model_metrics.model)

        # save model_stats in its own file
        model_stats_filepath = os.path.join(subdir, 'model_stats.yaml')
        if model_stats_filepath:
            with open(model_stats_filepath, 'w') as f:
                yaml.dump(model_stats, f)

        # save just metrics separately for convinience
        metrics_filepath = os.path.join(subdir, 'metrics.yaml')
        if metrics_filepath:
            with open(metrics_filepath, 'w') as f:
                yaml.dump(model_stats.metrics, f)

        logger.info({'model_stats_filepath': model_stats_filepath,
                     'metrics_filepath': metrics_filepath})

        # append key info in root pareto data
        if self._summary_filepath:
            train_top1 = val_top1 = train_epoch = val_epoch = math.nan
            # extract metrics
            if model_metrics.metrics:
                best_metrics = model_metrics.metrics.run_metrics.best_epoch()
                train_top1 = best_metrics[0].top1.avg
                train_epoch = best_metrics[0].index
                if best_metrics[1]:
                    val_top1 = best_metrics[1].top1.avg if len(best_metrics)>1 else math.nan
                    val_epoch = best_metrics[1].index if len(best_metrics)>1 else math.nan

            # extract model stats
            flops = model_stats.Flops
            parameters = model_stats.parameters
            inference_memory = model_stats.inference_memory
            inference_duration = model_stats.duration

            utils.append_csv_file(self._summary_filepath, [
                ('reductions', reductions),
                ('cells', cells),
                ('nodes', nodes),
                ('train_top1', train_top1),
                ('train_epoch', train_epoch),
                ('val_top1', val_top1),
                ('val_epoch', val_epoch),
                ('flops', flops),
                ('params', parameters),
                ('inference_memory', inference_memory),
                ('inference_duration', inference_duration)
                ])



