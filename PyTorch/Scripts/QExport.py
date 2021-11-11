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
#  File: QExport.py                                                            #
# ##############################################################################

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import logging

import torch
from torch.utils import data
import nemo

from Frontnet.DataProcessor import DataProcessor
from Frontnet.ModelTrainer import ModelTrainer
from Frontnet.Dataset import Dataset
from Frontnet.Frontnet import FrontnetModel
from Frontnet import Utils
from Frontnet.Utils import ModelManager


def main():
    args = Utils.ParseArgs(quantize=True)

    model_path = args.load_model
    data_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    out_folder = "Results/{}x{}".format(w, c)
    os.makedirs(out_folder, exist_ok=True)

    Utils.Logger(logfile=os.path.join(out_folder, "QExport.log"))

    torch.manual_seed(args.seed)

    # Load data
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(args.load_trainset)
    validation_set = Dataset(x_validation, y_validation)
    validation_loader = data.DataLoader(validation_set, **data_params)

    model = FrontnetModel(**args.model_params)
    model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, h, w)).to("cpu"))

    if model_path is None:
        model_path = os.path.join(out_folder, "{}.Q.pt".format(model.name))

    logging.info("[QExport] Loading model checkpoint {}".format(model_path))
    epoch, prec_dict = ModelManager.ReadQ(model_path, model)

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
    trainer.ExportQuantized(validation_loader, h, w, prec_dict)

    logging.info("[QExport] Model: %s", model)


if __name__ == '__main__':
    main()
