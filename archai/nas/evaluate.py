# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
import importlib
import sys
import string
import os

# only works on linux
import ray

import torch
from torch import nn

from archai.common.trainer import Trainer
from archai.common.config import Config
from archai.common.common import logger
from archai.datasets import data
from archai.nas.model_desc import ModelDesc
from archai.nas.cell_builder import CellBuilder
from archai.nas import nas_utils
from archai.common import common
from archai.common import ml_utils, utils


def eval_arch(conf_eval:Config, cell_builder:Optional[CellBuilder]):
    logger.pushd('eval_arch')

    # region conf vars
    conf_loader       = conf_eval['loader']
    model_filename    = conf_eval['model_filename']
    metric_filename    = conf_eval['metric_filename']
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    # endregion

    if cell_builder:
        cell_builder.register_ops()

    model = create_model(conf_eval)

    # get data
    train_dl, _, test_dl = data.get_data(conf_loader)
    assert train_dl is not None and test_dl is not None

    checkpoint = nas_utils.create_checkpoint(conf_checkpoint, resume)
    trainer = Trainer(conf_train, model, checkpoint)
    train_metrics = trainer.fit(train_dl, test_dl)
    train_metrics.save(metric_filename)

    # save model
    if model_filename:
        model_filename = utils.full_path(model_filename)
        ml_utils.save_model(model, model_filename)

    logger.info({'model_save_path': model_filename})

    logger.popd()

@ray.remote(num_gpus=1)
def helper_train_desc(model, conf_train, train_dl, test_dl, metric_filename, model_filename, common_state):
    ''' Train given a model '''
    common.init_from(common_state)
    # TODO: checkpointing for distributed training
    checkpoint = None   
    trainer = Trainer(conf_train, model, checkpoint)
    train_metrics = trainer.fit(train_dl, test_dl)
    train_metrics.save(metric_filename)

    # save model
    if model_filename:
        model_filename = utils.full_path(model_filename)
        ml_utils.save_model(model, model_filename)
        logger.info({'model_save_path': model_filename})


def eval_archs(conf_eval:Config, cell_builder:Optional[CellBuilder]):
    ''' Takes a folder of model descriptions output by search process and 
    trains them in a distributed manner using ray with 1 gpu '''
    logger.pushd('eval_arch')

    logger.info('----------starting to final train all models found by search------------------')

    # region conf vars
    conf_loader       = conf_eval['loader']
    final_desc_foldername = conf_eval['final_desc_foldername']
    final_desc_folderpath = utils.full_path(final_desc_foldername)
    conf_checkpoint = conf_eval['checkpoint']
    resume = conf_eval['resume']
    conf_train = conf_eval['trainer']
    # endregion

    if cell_builder:
        cell_builder.register_ops()

    # get list of model descs in the gallery folder
    files = [os.path.join(final_desc_folderpath, f) for f in os.listdir(final_desc_folderpath) if os.path.isfile(os.path.join(final_desc_folderpath, f))]
    logger.info(f'found {len(files)} models to final train')

    remote_ids = []
    for model_desc_filename in files:
        full_desc_filename = model_desc_filename.split('.')[0] + '_full.yaml'
        metric_filename = model_desc_filename.split('.')[0] + '_metrics.yaml'
        model_filename = model_desc_filename.split('.')[0] + '_model.pt'
        model = create_model(conf_eval, final_desc_filename=model_desc_filename, full_desc_filename=full_desc_filename)

        # get data
        train_dl, _, test_dl = data.get_data(conf_loader)
        assert train_dl is not None and test_dl is not None

        this_id = helper_train_desc.remote(model, conf_train, train_dl, test_dl, metric_filename, model_filename, common.get_state())
        remote_ids.append(this_id)

    # wait for all eval jobs to be finished
    ray.wait(remote_ids, num_returns=len(remote_ids))
    logger.info('---------finished training all models on the pareto frontier-----------------')

    logger.popd()

    


def _default_module_name(dataset_name:str, function_name:str)->str:
    module_name = ''
    # TODO: below detection code is too week, need to improve, possibly encode image size in yaml and use that instead
    if dataset_name.startswith('cifar'):
        if function_name.startswith('res'): # support resnext as well
            module_name = 'archai.cifar10_models.resnet'
        elif function_name.startswith('dense'):
            module_name = 'archai.cifar10_models.densenet'
    elif dataset_name.startswith('imagenet') or dataset_name.startswith('sport8'):
        module_name = 'torchvision.models'
    if not module_name:
        raise NotImplementedError(f'Cannot get default module for {function_name} and dataset {dataset_name} because it is not supported yet')
    return module_name

def create_model(conf_eval:Config, final_desc_filename=None, full_desc_filename=None)->nn.Module:
    # region conf vars
    dataset_name = conf_eval['loader']['dataset']['name']
    # if explicitly passed in then don't get from conf
    if not final_desc_filename:
        final_desc_filename = conf_eval['final_desc_filename']
        full_desc_filename = conf_eval['full_desc_filename']
    final_model_factory = conf_eval['final_model_factory']
    conf_model_desc   = conf_eval['model_desc']
    # endregion

    if final_model_factory:
        splitted = final_model_factory.rsplit('.', 1)
        function_name = splitted[-1]

        if len(splitted) > 1:
            module_name = splitted[0]
        else:
            module_name = _default_module_name(dataset_name, function_name)

        module = importlib.import_module(module_name) if module_name else sys.modules[__name__]
        function = getattr(module, function_name)
        model = function()

        logger.info({'model_factory':True,
                    'module_name': module_name,
                    'function_name': function_name,
                    'params': ml_utils.param_size(model)})
    else:
        # load model desc file to get template model
        template_model_desc = ModelDesc.load(final_desc_filename)

        model = nas_utils.model_from_conf(full_desc_filename,
                                    conf_model_desc, affine=True, droppath=True,
                                    template_model_desc=template_model_desc)

        logger.info({'model_factory':False,
                    'cells_len':len(model.desc.cell_descs()),
                    'init_node_ch': conf_model_desc['init_node_ch'],
                    'n_cells': conf_model_desc['n_cells'],
                    'n_reductions': conf_model_desc['n_reductions'],
                    'n_nodes': conf_model_desc['n_nodes']})

    return model



