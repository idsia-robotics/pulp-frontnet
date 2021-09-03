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
#  File: Utils.py                                                              #
# ##############################################################################

import argparse
import logging
import os

import shutil

import pandas as pd
import sklearn.metrics
import torch

from .Frontnet import FrontnetModel


################
# Script utils #
################

def Logger(logfile="log.txt"):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=logfile,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def ParseArgs(quantize=False):
    parser = argparse.ArgumentParser(description='PyTorch Frontnet')
    parser.add_argument('model_config', type=str, default='160x32', choices=FrontnetModel.configs.keys(),
                        help='network configuration to be used (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # [NEMO] Model saving/loading improved for convenience
    parser.add_argument('--save-model', default=None, type=str,
                        help='for saving the model')
    parser.add_argument('--load-model', default=None, type=str,
                        help='for loading the model')

    parser.add_argument('--load-trainset', default=None, type=str,
                        help='for loading the train dataset')
    parser.add_argument('--load-testset', default=None, type=str,
                        help='for loading the test dataset')

    # [NEMO] If `quantize` is False, the script operates like the original PyTorch example
    parser.add_argument('--quantize', default=quantize, action="store_true",
                        help='for loading the model')
    # [NEMO] The training regime (in JSON) used to store all NEMO configuration.
    parser.add_argument('--regime', default='Scripts/regime.json', type=str,
                        help='for loading the model')

    args = parser.parse_args()

    args.model_params = FrontnetModel.configs[args.model_config]
    w, h, c = args.model_params['w'], args.model_params['h'], args.model_params['c']

    if args.load_trainset is None:
        args.load_trainset = "Data/{}x{}OthersTrainsetAug.pickle".format(w, h)

    if args.load_testset is None:
        args.load_testset = "Data/{}x{}StrangersTestset.pickle".format(w, h)

    return args


def Confirm(prompt, default=None):
    if default is None:
        prompt = '{} y|n: '.format(prompt)
    elif default is True:
        prompt = '{} [y]|n: '.format(prompt)
    else:
        prompt = '{} y|[n]: '.format(prompt)

    while True:
        answer = input(prompt).lower()

        if not answer and default is not None:
            return default
        elif answer == 'y':
            return True
        elif answer == 'n':
            return False
        else:
            print('Please enter y or n.')


def RemoveChildren(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


######################
# Validation metrics #
######################

class AverageBase(object):
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value


class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """

    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count

    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """

    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


class Metrics:
    def __init__(self):
        self.MSE = []
        self.MAE = []
        self.r2_score = []

        self.train_losses_x = []
        self.train_losses_y = []
        self.train_losses_z = []
        self.train_losses_phi = []
        self.valid_losses_x = []
        self.valid_losses_y = []
        self.valid_losses_z = []
        self.valid_losses_phi = []
        self.gt_labels = []
        self.y_pred = []

    def Update(self, y_pred, gt_labels, train_loss, valid_loss):
        self.train_losses_x.append(train_loss[0])
        self.train_losses_y.append(train_loss[1])
        self.train_losses_z.append(train_loss[2])
        self.train_losses_phi.append(train_loss[3])

        self.valid_losses_x.append(valid_loss[0])
        self.valid_losses_y.append(valid_loss[1])
        self.valid_losses_z.append(valid_loss[2])
        self.valid_losses_phi.append(valid_loss[3])

        self.y_pred.append(y_pred)
        self.gt_labels.append(gt_labels)

        MSE = torch.mean((y_pred - gt_labels).pow(2), 0)
        MAE = torch.mean(torch.abs(y_pred - gt_labels), 0)

        x = y_pred[:, 0]
        x = x.cpu().numpy()
        x_gt = gt_labels[:, 0]
        x_gt = x_gt.cpu().numpy()

        y = y_pred[:, 1]
        y = y.cpu().numpy()
        y_gt = gt_labels[:, 1]
        y_gt = y_gt.cpu().numpy()

        z = y_pred[:, 2]
        z = z.cpu().numpy()
        z_gt = gt_labels[:, 2]
        z_gt = z_gt.cpu().numpy()

        phi = y_pred[:, 3]
        phi = phi.cpu().numpy()
        phi_gt = gt_labels[:, 3]
        phi_gt = phi_gt.cpu().numpy()

        x_r2 = sklearn.metrics.r2_score(x_gt, x)
        y_r2 = sklearn.metrics.r2_score(y_gt, y)
        z_r2 = sklearn.metrics.r2_score(z_gt, z)
        phi_r2 = sklearn.metrics.r2_score(phi_gt, phi)
        r2_score = torch.FloatTensor([x_r2, y_r2, z_r2, phi_r2])

        self.MSE.append(MSE)
        self.MAE.append(MAE)
        self.r2_score.append(r2_score)

        return MSE, MAE, r2_score

    def Reset(self):
        self.MSE.clear()
        self.MAE.clear()
        self.r2_score.clear()

    def GetPred(self):
        return self.y_pred

    def GetLabels(self):
        return self.gt_labels

    def GetLosses(self):
        return self.train_losses_x, self.train_losses_y, self.train_losses_z, self.train_losses_phi , \
               self.valid_losses_x, self.valid_losses_y, self.valid_losses_z, self.valid_losses_phi

    def GetMSE(self):
        return self.MSE

    def GetMAE(self):
        return self.MAE

    def GetR2(self):
        return self.r2_score


def SaveModelResultsToCSV(MSE, MAE, r2_score, labels, predictions, name):
    x_gt = []
    y_gt = []
    z_gt = []
    phi_gt = []
    x_pr = []
    y_pr = []
    z_pr = []
    phi_pr = []

    r2_score = torch.stack(r2_score, 0)
    x = r2_score[:, 0]
    r2_score_x = x.cpu().numpy()
    y = r2_score[:, 1]
    r2_score_y = y.cpu().numpy()
    z = r2_score[:, 2]
    r2_score_z = z.cpu().numpy()
    phi = r2_score[:, 3]
    r2_score_phi = phi.cpu().numpy()

    MSE = torch.stack(MSE, 0)
    x = MSE[:, 0]
    MSE_x = x.cpu().numpy()
    y = MSE[:, 1]
    MSE_y = y.cpu().numpy()
    z = MSE[:, 2]
    MSE_z = z.cpu().numpy()
    phi = MSE[:, 3]
    MSE_phi = phi.cpu().numpy()

    MAE = torch.stack(MAE, 0)
    x = MAE[:, 0]
    MAE_x = x.cpu().numpy()
    y = MAE[:, 1]
    MAE_y = y.cpu().numpy()
    z = MAE[:, 2]
    MAE_z = z.cpu().numpy()
    phi = MAE[:, 3]
    MAE_phi = phi.cpu().numpy()

    df = pd.DataFrame(
        data={ 'MSE_x': MSE_x, 'MSE_y': MSE_y, 'MSE_z': MSE_z, 'MSE_phi': MSE_phi,
              'MAE_x': MAE_x, 'MAE_y': MAE_y, 'MAE_z': MAE_z, 'MAE_phi': MAE_phi,
               'r2_score_x': r2_score_x, 'r2_score_y': r2_score_y, 'r2_score_z': r2_score_z, 'r2_score_phi': r2_score_phi})
    df.index.name = "epochs"

    df.to_csv(name + ".csv", header=True)


############################
# Model checkpoint manager #
############################

class ModelManager:
    @staticmethod
    def Read(filename, model):
        """Reads model file

            Parameters
            ----------
            filename : str
                location of the model file
            model : NN class
                A PyTorch NN object

            Returns
            -------
            int
                the number of epochs this model was trained
            """

        state_dict = torch.load(filename, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        epoch = state_dict['epoch']

        return epoch

    @staticmethod
    def ReadQ(filename, model):
        """Reads model file

            Parameters
            ----------
            filename : str
                location of the model file
            model : NN class
                A PyTorch NN object

            Returns
            -------
            int
                the number of epochs this model was trained
            """

        state_dict = torch.load(filename, map_location='cpu')
        epoch = state_dict['epoch']
        try:
            prec_dict = state_dict['precision']
        except KeyError:
            prec_dict = None

        try:
            model.load_state_dict(state_dict['state_dict'], strict=True)
        except KeyError:
            model.load_state_dict(state_dict, strict=False)

        return epoch, prec_dict

    @staticmethod
    def Write(model, epoch, filename):
        """writes model to file

        Parameters
        ----------
        model : NN class
                A PyTorch NN object
        epoch: int
            the number of epochs this model was trained
        filename : str
            name and location of the model to be saved
        """

        checkpoint_dict = {
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)

    @staticmethod
    def WriteQ(net, epoch, filename, optimizer, acc=0.0):
        # Based on nemo.utils.save_checkpoint(...), but take an arbitrary output
        # filename as parameter.
        try:
            optimizer_state = optimizer.state_dict()
        except AttributeError:
            optimizer_state = None

        try:
            precision = net.export_precision()
        except AttributeError:
            precision = None

        state = {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'precision': precision,
            'acc': acc,
            'optimizer': optimizer_state,
        }

        torch.save(state, filename)
