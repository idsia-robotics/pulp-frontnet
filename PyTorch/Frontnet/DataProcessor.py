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
#  File: DataProcessor.py                                                      #
# ##############################################################################

import logging
import random
import pandas as pd
import numpy as np


class DataProcessor:
    @staticmethod
    def GetSizeDataFromDataFrame(dataset):
        h = int(dataset['h'].values[0])
        w = int(dataset['w'].values[0])
        c = int(dataset['c'].values[0])

        return h, w, c

    @staticmethod
    def ProcessTrainData(trainPath):
        """Reads the .pickle file and converts it into a format suitable fot training

            Parameters
            ----------
            trainPath : str
                The file location of the .pickl

            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """
        train_set = pd.read_pickle(trainPath)

        logging.info('[DataProcessor] train shape: ' + str(train_set.shape))
        size = train_set.shape[0]
        n_val = int(float(size) * 0.2)

        h, w, c = DataProcessor.GetSizeDataFromDataFrame(train_set)

        np.random.seed(1749)
        random.seed(1749)

        # split between train and test sets:
        x_train = train_set['x'].values
        x_train = np.vstack(x_train[:]).astype(np.float32)
        x_train = np.reshape(x_train, (-1, h, w, c))

        x_train = np.swapaxes(x_train, 1, 3)
        x_train = np.swapaxes(x_train, 2, 3)

        y_train = train_set['y'].values
        y_train = np.vstack(y_train[:]).astype(np.float32)

        ix_val, ix_tr = np.split(np.random.permutation(train_set.shape[0]), [n_val])
        x_validation = x_train[ix_val, :]
        x_train = x_train[ix_tr, :]
        y_validation = y_train[ix_val, :]
        y_train = y_train[ix_tr, :]

        shape_ = len(x_train)

        sel_idx = random.sample(range(0, shape_), k=(size-n_val))
        x_train = x_train[sel_idx, :]
        y_train = y_train[sel_idx, :]

        return [x_train, x_validation, y_train, y_validation]

    @staticmethod
    def ProcessTestData(testPath):
        """Reads the .pickle file and converts it into a format suitable fot testing

            Parameters
            ----------
            testPath : str
                The file location of the .pickle

            Returns
            -------
            list
                list of video frames and list of labels (poses)
            """

        test_set = pd.read_pickle(testPath)
        logging.info('[DataProcessor] test shape: ' + str(test_set.shape))
        h, w, c = DataProcessor.GetSizeDataFromDataFrame(test_set)

        x_test = test_set['x'].values
        x_test = np.vstack(x_test[:]).astype(np.float32)
        x_test = np.reshape(x_test, (-1, h, w, c))

        x_test = np.swapaxes(x_test, 1, 3)
        x_test = np.swapaxes(x_test, 2, 3)
        y_test = test_set['y'].values
        y_test = np.vstack(y_test[:]).astype(np.float32)

        return [x_test, y_test]
