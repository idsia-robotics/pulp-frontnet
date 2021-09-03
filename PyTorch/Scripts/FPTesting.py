# ##############################################################################
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
#                                                     9                         #
#  File: FPTesting.py                                                          #
# ##############################################################################

import logging
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils import data

from Frontnet.DataProcessor import DataProcessor
from Frontnet.ModelTrainer import ModelTrainer
from Frontnet.Dataset import Dataset
from Frontnet.ModelManager import ModelManager
from Frontnet.Frontnet import FrontnetModel
from Frontnet import Utils


def main():
    args = Utils.ParseArgs()

    model_path = args.load_model
    testset_path = args.load_testset
    data_params = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    out_folder = "Results/{}x{}".format(w, c)
    os.makedirs(out_folder, exist_ok=True)

    Utils.Logger(logfile=os.path.join(out_folder, "FPTesting.log"))

    torch.manual_seed(args.seed)

    # Load the test data
    [x_test, y_test] = DataProcessor.ProcessTestData(testset_path)

    # Create the PyTorch data loaders
    test_set = Dataset(x_test, y_test)
    test_loader = data.DataLoader(test_set, **data_params)

    # Choose your model
    model = FrontnetModel(**args.model_params)

    if model_path is None:
        model_path = os.path.join(out_folder, "{}.pt".format(model.name))

    logging.info("[FPTesting] Loading model checkpoint {}".format(model_path))
    ModelManager.Read(model_path, model)

    # Evaluate model performance
    trainer = ModelTrainer(model)
    trainer.folderPath = out_folder
    trainer.Test(test_loader)


if __name__ == '__main__':
    main()
