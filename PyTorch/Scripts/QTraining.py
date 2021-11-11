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
#                                                                              #
#  File: QTraining.py                                                          #
# ##############################################################################

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import logging

import torch
from torch.utils import data

from Frontnet.DataProcessor import DataProcessor
from Frontnet.ModelTrainer import ModelTrainer
from Frontnet.Dataset import Dataset
from Frontnet.Frontnet import FrontnetModel
from Frontnet import Utils
from Frontnet.Utils import ModelManager


def LoadData(args):
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(args.load_trainset)
    training_set = Dataset(x_train, y_train, True)
    validation_set = Dataset(x_validation, y_validation)

    # Parameters
    # num_workers - 0 for debug in Mac+PyCharm, 6 for everything else
    num_workers = 6
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': num_workers}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)

    return train_loader, validation_loader


def main():
    args = Utils.ParseArgs(quantize=True)

    model_path = args.load_model
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    out_folder = "Results/{}x{}".format(w, c)
    os.makedirs(out_folder, exist_ok=True)

    Utils.Logger(logfile=os.path.join(out_folder, "QTraining.log"))

    torch.manual_seed(args.seed)

    # Load data
    train_loader, validation_loader = LoadData(args)

    model = FrontnetModel(**args.model_params)

    if model_path is None:
        model_path = os.path.join(out_folder, "{}.pt".format(model.name))

    logging.info("[QTraining] Loading model checkpoint {}".format(model_path))
    ModelManager.Read(model_path, model)

    # [NEMO] Load the JSON regime file.
    regime = {}
    if args.regime is None:
        print("ERROR! Missing regime JSON.")
        raise Exception
    else:
        with open(args.regime, "r") as f:
            rr = json.load(f)
        for k in rr.keys():
            try:
                regime[int(k)] = rr[k]
            except ValueError:
                regime[k] = rr[k]

    trainer = ModelTrainer(model, args, regime)
    trainer.folderPath = out_folder
    trainer.TrainQuantized(train_loader, validation_loader, h, w, args.epochs, args.save_model)

    logging.info("[QTraining] Model: %s", model)


if __name__ == '__main__':
    main()
