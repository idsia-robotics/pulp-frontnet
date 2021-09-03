# ##############################################################################
#  Copyright (c) 2018 Bjarte Mehus Sunde                                       #
#                                                                              #
#  Permission is hereby granted, free of charge, to any person obtaining a     #
#  copy of this software and associated documentation files (the "Software"),  #
#  to deal in the Software without restriction, including without limitation   #
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
#  and/or sell copies of the Software, and to permit persons to whom the       #
#  Software is furnished to do so, subject to the following conditions:        #
#                                                                              #
#  The above copyright notice and this permission notice shall be included in  #
#  all copies or substantial portions of the Software.                         #
#                                                                              #
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
#  DEALINGS IN THE SOFTWARE.                                                   #
#                                                                              #
#  Copyright (c) 2019–2021 IDSIA, USI and SUPSI, Switzerland                   #
#                2019-2021 University of Bologna, Italy                        #
#                2019-2021 ETH Zürich, Switzerland                             #
#  All rights reserved.                                                        #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#      http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
#                                                                              #
#  File: EarlyStopping.py                                                      #
#  Source: https://github.com/Bjarten/early-stopping-pytorch                   #
# ##############################################################################

import logging
import os

from .Utils import ModelManager


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, cleanup=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            cleanup (bool): Delete old checkpoints when saving a new one.
                            Default: True
        """
        self.patience = patience
        self.verbose = verbose
        self.cleanup = cleanup
        self.counter = 0
        self.best_loss = None
        self.best_path = None
        self.early_stop = False

    def __call__(self, loss, model, epoch, file_name='checkpoint.pt'):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model, epoch, file_name)
        elif loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                logging.info("[EarlyStopping] Counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logging.info("[EarlyStopping] Validation loss decreased {} --> {}. Saving model as {}".format(self.best_loss, loss, file_name))
            self.best_loss = loss
            self.save_checkpoint(model, epoch, file_name)
            self.counter = 0

    def save_checkpoint(self, model, epoch, file_name):
        """Saves model when validation loss decrease."""
        ModelManager.Write(model, epoch, file_name)

        if self.cleanup and self.best_path is not None:
            if self.verbose:
                logging.info("[EarlyStopping] Removing old checkpoint {}".format(self.best_path))
            try:
                os.remove(self.best_path)
            except OSError:
                # If the file does not exist or cannot be removed, ignore.
                pass

        self.best_path = file_name
