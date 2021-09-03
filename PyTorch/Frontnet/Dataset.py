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
#  File: Dataset.py                                                            #
# ##############################################################################

import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, labels, train=False):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        length = len(self.data)
        self.list_IDs = range(0, length)
        self.train = train

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def toNumpy(self, X):
        X = X.cpu().numpy()
        h, w = X.shape[1:3]
        X = np.reshape(X, (h, w)).astype("uint8")
        return X

    def toTensor(self, X):
        h, w = X.shape
        X = np.reshape(X, (1, h, w))
        X = torch.from_numpy(X).float()
        return X

    def augmentDR(self, X):
        # # dynamic range augmentation
        dr = np.random.uniform(0.4, 0.8)  # dynamic range
        lo = np.random.uniform(0, 0.3)
        hi = min(1.0, lo + dr)
        X = np.interp(X / 255.0, [0, lo, hi, 1], [0, 0, 1, 1])
        X = 255 * X

        return X

    def __getitem__(self, index):
        """Generates one sample of data"""
        ID = index

        X = self.data[ID]
        y = self.labels[ID]

        if self.train == True:
            if np.random.choice([True, False]):
                X = torch.flip(X, [2])
                y[1] = -y[1]  # Y
                y[3] = -y[3]  # Relative YAW

        return X, y
