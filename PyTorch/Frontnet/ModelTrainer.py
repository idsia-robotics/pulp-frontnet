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
#  File: ModelTrainer.py                                                       #
# ##############################################################################

import logging
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import nemo

from .Utils import RunningAverage, MovingAverage, Metrics, SaveModelResultsToCSV, ModelManager
from .DataVisualization import DataVisualization
from .EarlyStopping import EarlyStopping
from . import Utils


class ModelTrainer:
    def __init__(self, model, args=None, regime=None):
        self.model = model

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logging.info("[ModelTrainer] " + device)
        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if args is not None:
            self.args = args
            self.regime = regime
            self.num_epochs = args.epochs

            if self.args.quantize:
                param = dict(self.model.named_parameters())
                fp_params = list({k: v for k, v in param.items() if k[:3] == "fc."}.values())
                qnt_params = list({k: v for k, v in param.items() if k[:3] != "fc."}.values())
                self.optimizer = torch.optim.Adam((
                    {'params': qnt_params, 'lr': float(regime['lr']), 'weight_decay': float(regime['weight_decay'])},
                    {'params': fp_params, 'lr': float(regime['lr']), 'weight_decay': float(regime['weight_decay'])}
                ))
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            self.num_epochs = 100

        self.folderPath = "Models/"

    def GetModel(self):
        return self.model

    def TrainSingleEpoch(self, training_generator):
        self.model.train()
        train_loss_x = MovingAverage()
        train_loss_y = MovingAverage()
        train_loss_z = MovingAverage()
        train_loss_phi = MovingAverage()

        i = 0
        for batch_samples, batch_targets in training_generator:

            batch_targets = batch_targets.to(self.device)
            batch_samples = batch_samples.to(self.device)
            outputs = self.model(batch_samples)

            loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
            loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
            loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
            loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
            loss = loss_x + loss_y + loss_z + loss_phi
            # loss = self.criterion(outputs, batch_targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss_x.update(loss_x)
            train_loss_y.update(loss_y)
            train_loss_z.update(loss_z)
            train_loss_phi.update(loss_phi)

            if (i + 1) % 100 == 0:
                logging.info("[ModelTrainer] Step [{}]: Average train loss {}, {}, {}, {}".format(
                    i+1, train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value
                ))
            i += 1

        return train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value

    def ValidateSingleEpoch(self, validation_generator, integer=False):
        self.model.eval()

        valid_loss = RunningAverage()
        valid_loss_x = RunningAverage()
        valid_loss_y = RunningAverage()
        valid_loss_z = RunningAverage()
        valid_loss_phi = RunningAverage()

        y_pred = []
        gt_labels = []
        with torch.no_grad():
            for batch_samples, batch_targets in validation_generator:
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)
                if integer:
                    eps_fcin = (self.model.layer3.relu2.alpha / (2 ** self.model.layer3.relu2.precision.get_bits() - 1))
                    eps_fcout = self.model.fc.get_output_eps(eps_fcin)
                    # workaround because PACT_Linear is not properly quantizing biases!
                    outputs = [(o - self.model.fc.bias[i]) * eps_fcout + self.model.fc.bias[i] for i, o in
                               enumerate(outputs)]
                    batch_targets = torch.round(batch_targets / eps_fcout) * eps_fcout

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

                valid_loss.update(loss)
                valid_loss_x.update(loss_x)
                valid_loss_y.update(loss_y)
                valid_loss_z.update(loss_z)
                valid_loss_phi.update(loss_phi)

                outputs = torch.stack(outputs, 0)
                outputs = torch.squeeze(outputs)
                outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

        logging.info("[ModelTrainer] Average validation loss {}, {}, {}, {}".format(
            valid_loss_x.value, valid_loss_y.value,
            valid_loss_z.value,
            valid_loss_phi.value
        ))

        return valid_loss_x.value, valid_loss_y.value, valid_loss_z.value, valid_loss_phi.value, y_pred, gt_labels

    def Train(self, training_generator, validation_generator, model_filename=None):
        metrics = Metrics()
        early_stopping = EarlyStopping(verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
                                                               patience=5, verbose=False,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0.1e-6, eps=1e-08)

        checkpoints_folder = os.path.join(self.folderPath, "Checkpoints")
        os.makedirs(checkpoints_folder, exist_ok=True)

        metrics_folder = os.path.join(self.folderPath, "Metrics")
        os.makedirs(metrics_folder, exist_ok=True)

        if len(os.listdir(checkpoints_folder)) != 0:
            confirmed = Utils.Confirm(
                "[ALERT] Previous checkpoints exist for the current experiment and will be deleted, continue?",
                default=False
            )

            if not confirmed:
                return

            # Clear old checkpoints
            Utils.RemoveChildren(checkpoints_folder)

        for epoch in range(1, self.num_epochs + 1):
            logging.info("[ModelTrainer] Starting Epoch {}".format(epoch))

            train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(training_generator)

            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_generator)

            valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
            scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                                [train_loss_x, train_loss_y, train_loss_z, train_loss_phi],
                                                [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

            logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Validation r2_score: {}'.format(r2_score))

            checkpoint_filename = os.path.join(checkpoints_folder, '{}-{:03d}.pt'.format(self.model.name, epoch))
            early_stopping(valid_loss, self.model, epoch, checkpoint_filename)
            if early_stopping.early_stop:
                logging.info("[ModelTrainer] Early stopping")
                break

        if model_filename is None:
            model_filename = os.path.join(self.folderPath, '{}.pt'.format(self.model.name))

        logging.info('[ModelTrainer] Saving trained model as {}'.format(model_filename))
        shutil.copy2(early_stopping.best_path, model_filename)

        MSEs = metrics.GetMSE()
        MAEs = metrics.GetMAE()
        r2_scores = metrics.GetR2()
        y_pred_viz = metrics.GetPred()
        gt_labels_viz = metrics.GetLabels()
        train_losses_x, train_losses_y, train_losses_z, train_losses_phi, valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi = metrics.GetLosses()

        csv_path = os.path.join(metrics_folder, 'FPTraining')
        SaveModelResultsToCSV(MSEs, MAEs, r2_scores, gt_labels_viz, y_pred_viz, csv_path)

        DataVisualization.folderPath = metrics_folder
        DataVisualization.desc = "FPTraining_"
        DataVisualization.PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi)
        DataVisualization.PlotMSE(MSEs)
        DataVisualization.PlotMAE(MAEs)
        DataVisualization.PlotR2Score(r2_scores)
        DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)

        DataVisualization.DisplayPlots()

    def Test(self, test_generator):
        metrics = Metrics()

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            test_generator)

        outputs = y_pred
        outputs = np.reshape(outputs, (-1, 4))
        labels = gt_labels
        y_pred = np.reshape(y_pred, (-1, 4))
        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                            [0, 0, 0, 0],
                                            [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

        logging.info('[ModelTrainer] Test MSE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MSE[0], MSE[1], MSE[2], MSE[3]))
        logging.info('[ModelTrainer] Test MAE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MAE[0], MAE[1], MAE[2], MAE[3]))
        logging.info('[ModelTrainer] Test r2_score: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(r2_score[0], r2_score[1], r2_score[2], r2_score[3]))

        return MSE, MAE, r2_score, outputs, labels

    def TrainQuantized(self, train_loader, validation_loader, h, w, epochs=100, model_filename=None, relaxation=False):
        checkpoints_folder = os.path.join(self.folderPath, "CheckpointsQ")
        os.makedirs(checkpoints_folder, exist_ok=True)

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: Before quantization process: %f" % acc)

        # [NEMO] This call "transforms" the model into a quantization-aware one,
        # which is printed immediately afterwards.
        self.model = nemo.transform.quantize_pact(self.model,
                                                  dummy_input=torch.ones((1, 1, h, w)).to(self.device))  # .cuda()
        logging.info("[ModelTrainer] Model: %s", self.model)
        self.model.change_precision(bits=20)

        # calibration
        self.model.reset_alpha_weights()
        self.model.set_statistics_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        _ = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        self.model.unset_statistics_act()
        self.model.reset_alpha_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: After calibration process: %f" % acc)

        precision_rule = self.regime['relaxation']

        # [NEMO] The relaxation engine can be stepped to automatically change the
        # DNN precisions and end training if the final target has been achieved.
        if relaxation:
            relax = nemo.relaxation.RelaxationEngine(self.model, self.optimizer, criterion=None, trainloader=None,
                                                     precision_rule=precision_rule, reset_alpha_weights=False,
                                                     min_prec_dict=None, evaluator=None)

        prec_dict = {}
        if relaxation:
            self.model.change_precision(bits=12, min_prec_dict=prec_dict)
            self.model.change_precision(bits=12, scale_activations=False, min_prec_dict=prec_dict)
        else:
            self.model.change_precision(bits=8, min_prec_dict=prec_dict)
            self.model.change_precision(bits=7, scale_activations=False, min_prec_dict=prec_dict)

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: Before fine-tuning: %f" % acc)

        loss_epoch_m1 = 1e3
        best = 0.

        for epoch in range(1, epochs):
            change_prec = False
            ended = False

            if relaxation:
                change_prec, ended = relax.step(loss_epoch_m1, epoch, checkpoint_name=self.model.name)
                # If I try to run with Relaxation = True, I get exception here. This is because
                # loss_epoch_m1 > self.precision_rule['divergence_abs_threshold'] and the code tries to
                # load a checkpoint that does not exist..... Setting loss_epoch_m1 = acc solves it,
                # but who knows if it's correct.
            else:
                self.optimizer.param_groups[0]['lr'] *= float(self.regime['lr_decay'])
                self.optimizer.param_groups[1]['lr'] *= float(self.regime['lr_decay'])

            if ended:
                break

            train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(train_loader)
            loss_epoch_m1 = train_loss_x + train_loss_y + train_loss_z + train_loss_phi

            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_loader)
            acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)

            logging.info(
                "[ModelTrainer] Epoch: %d Train loss: %.2f Accuracy: %.2f" % (epoch, loss_epoch_m1, acc))

            if acc > best:
                checkpoint_path = os.path.join(checkpoints_folder, "{}.Q.pt".format(self.model.name))
                ModelManager.WriteQ(self.model, epoch, checkpoint_path, self.optimizer, acc)

                best = acc
                best_path = checkpoint_path

        if model_filename is None:
            model_filename = os.path.join(self.folderPath, '{}.Q.pt'.format(self.model.name))

        logging.info('[ModelTrainer] Saving trained model as {}'.format(model_filename))
        shutil.copy2(best_path, model_filename)

    def ExportQuantized(self, validation_loader, h, w, prec_dict=None):
        export_folder = os.path.join(self.folderPath, "Export")
        os.makedirs(export_folder, exist_ok=True)

        logging.info("[ModelTrainer] Model: %s", self.model)
        self.model.change_precision(bits=1, reset_alpha=False, min_prec_dict=prec_dict)

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: FakeQuantized network: %f" % acc)

        # qd_stage requires NEMO>=0.0.3
        # input is in [0,255], so eps_in=1 (smallest representable amount in the input) and there is no input bias
        self.model.qd_stage(eps_in=1.0)
        bin_qd, bout_qd, (valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred,
                          gt_labels) = nemo.utils.get_intermediate_activations(self.model, self.ValidateSingleEpoch,
                                                                               validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: QuantizedDeployable network: %f" % acc)

        # id_stage requires NEMO>=0.0.3
        self.model.id_stage()
        bin_id, bout_id, (valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred,
                          gt_labels) = nemo.utils.get_intermediate_activations(self.model, self.ValidateSingleEpoch,
                                                                               validation_loader, integer=True)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: IntegerDeployable: %f" % acc)
        eps_fcin = (self.model.layer3.relu2.alpha / (2 ** self.model.layer3.relu2.precision.get_bits() - 1))
        eps_fcout = self.model.fc.get_output_eps(eps_fcin)
        logging.info("[ModelTrainer]: output quantum is eps_out=%f" % eps_fcout)
        logging.info("[ModelTrainer]: fc biases are {}".format(self.model.fc.bias))

        with open(os.path.join(export_folder, "output_dequant.txt"), "w") as f:
            f.write("FLOAT_X   = INT_X   * %.5e + %.5e\n" % (eps_fcout, self.model.fc.bias[0]))
            f.write("FLOAT_Y   = INT_Y   * %.5e + %.5e\n" % (eps_fcout, self.model.fc.bias[1]))
            f.write("FLOAT_Z   = INT_Z   * %.5e + %.5e\n" % (eps_fcout, self.model.fc.bias[2]))
            f.write("FLOAT_PHI = INT_PHI * %.5e + %.5e\n" % (eps_fcout, self.model.fc.bias[3]))

        # export model
        model_path = os.path.join(export_folder, "{}.onnx".format(self.model.name))
        nemo.utils.export_onnx(model_path, self.model, self.model, (1, h, w), perm=None)

        # workaround because PACT_Linear is not properly quantizing biases!
        self.model.fc.bias.data[:] = 0

        # export golden outputs
        b_in = bin_id
        b_out = bout_id

        bidx = 0

        # Network input: use the '' key to obtain the input to the overall model
        name = ''
        try:
            actbuf = b_in[name][0][bidx].permute((1, 2, 0))
        except RuntimeError:
            actbuf = b_in[name][0][bidx]
        np.savetxt(os.path.join(export_folder, "input.txt"), actbuf.cpu().detach().numpy().flatten(),
                   header="input (shape %s)" % (list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')

        # Network activations
        patterns = ['relu', 'maxpool', 'fc']
        counter = 0
        for n, m in self.model.named_modules():
            # Ignore any layer whose name doesn't match one of the provided patterns
            if not any(p in n for p in patterns):
                continue

            try:
                actbuf = b_out[n][bidx].permute((1, 2, 0))
            except RuntimeError:
                actbuf = b_out[n][bidx]

            np.savetxt(os.path.join(export_folder, "out_layer%d.txt" % counter),
                       actbuf.cpu().detach().numpy().flatten(),
                       header="%s (shape %s)" % (n, list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')
            counter += 1
