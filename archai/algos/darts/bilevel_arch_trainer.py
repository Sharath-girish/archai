# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Mapping, Optional, Union
import copy

import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn, autograd
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from overrides import overrides

from archai.common.config import Config
from archai.nas.arch_trainer import ArchTrainer
from archai.common import utils, ml_utils
from archai.nas.model import Model
from archai.common.checkpoint import CheckPoint
from archai.common.common import logger
from archai.algos.darts.bilevel_optimizer import BilevelOptimizer

class BilevelArchTrainer(ArchTrainer):
    def __init__(self, conf_train: Config, model: Model,
                 checkpoint:Optional[CheckPoint]) -> None:
        super().__init__(conf_train, model, checkpoint)

        self._conf_w_optim = conf_train['optimizer']
        self._conf_w_lossfn = conf_train['lossfn']
        self._conf_alpha_optim = conf_train['alpha_optimizer']

    @overrides
    def pre_fit(self, train_dl: DataLoader, val_dl: Optional[DataLoader])->None:
        super().pre_fit(train_dl, val_dl)

        # optimizers, schedulers needs to be recreated for each fit call
        # as they have state
        assert val_dl is not None
        w_momentum = self._conf_w_optim['momentum']
        w_decay = self._conf_w_optim['decay']
        lossfn = ml_utils.get_lossfn(self._conf_w_lossfn).to(self.get_device())

        self._bilevel_optim = BilevelOptimizer(self._conf_alpha_optim, w_momentum,
                                                w_decay, self.model, lossfn,
                                                self.get_device(), self.batch_chunks)

    @overrides
    def post_fit(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        # delete state we created in pre_fit
        del self._bilevel_optim
        return super().post_fit(train_dl, val_dl)

    @overrides
    def pre_epoch(self, train_dl: DataLoader, val_dl: Optional[DataLoader])->None:
        super().pre_epoch(train_dl, val_dl)

        # prep val set to train alphas
        self._valid_iter = iter(val_dl)  # type: ignore

    @overrides
    def post_epoch(self, train_dl:DataLoader, val_dl:Optional[DataLoader])->None:
        del self._valid_iter # clean up
        super().post_epoch(train_dl, val_dl)

    @overrides
    def pre_step(self, x: Tensor, y: Tensor) -> None:
        super().pre_step(x, y)

        # reset val loader if we exausted it
        try:
            x_val, y_val = next(self._valid_iter)
        except StopIteration:
            # reinit iterator
            self._valid_iter = iter(self._val_dl)
            x_val, y_val = next(self._valid_iter)

        # update alphas
        self._bilevel_optim.step(x, y, x_val, y_val, super().get_optimizer())

    @overrides
    def update_checkpoint(self, check_point:CheckPoint)->None:
        super().update_checkpoint(check_point)
        check_point['bilevel_optim'] = self._bilevel_optim.state_dict()

    @overrides
    def restore_checkpoint(self)->None:
        super().restore_checkpoint()
        self._bilevel_optim.load_state_dict(self.check_point['bilevel_optim'])

