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
#  File: FPTraining.py                                                         #
# ##############################################################################

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils import data

from Frontnet.DataProcessor import DataProcessor
from Frontnet.ModelTrainer import ModelTrainer
from Frontnet.Dataset import Dataset
from Frontnet.Frontnet import FrontnetModel
from Frontnet import Utils


def main():
    args = Utils.ParseArgs()

    trainset_path = args.load_trainset
    data_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    out_folder = "Results/{}x{}".format(w, c)
    os.makedirs(out_folder, exist_ok=True)

    Utils.Logger(logfile=os.path.join(out_folder, "FPTraining.log"))

    torch.manual_seed(args.seed)

    # Load the training data (which will be split to validation and train)
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(trainset_path)

    # Create the PyTorch data loaders
    training_set = Dataset(x_train, y_train, train=True)
    training_generator = data.DataLoader(training_set, **data_params)
    validation_set = Dataset(x_validation, y_validation, train=False)
    validation_generator = data.DataLoader(validation_set, **data_params)

    # Choose your model
    model = FrontnetModel(**args.model_params)

    # Run the training loop
    trainer = ModelTrainer(model)
    trainer.folderPath = out_folder
    trainer.Train(training_generator, validation_generator, args.save_model)


if __name__ == '__main__':
    main()
