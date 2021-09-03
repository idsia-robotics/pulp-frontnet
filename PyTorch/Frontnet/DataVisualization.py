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
#  File: DataVisualization.py                                                  #
# ##############################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import torch


class DataVisualization:
    figure_counter = 0
    folderPath = "Results"
    desc = ""

    @staticmethod
    def PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi):
        epochs = range(1, len(train_losses_x) + 1)

        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')

        plt.plot(epochs, train_losses_x, color='green', label='Training loss')
        plt.plot(epochs, valid_losses_x, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')

        plt.plot(epochs, train_losses_y, color='blue', label='Training loss')
        plt.plot(epochs, valid_losses_y, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')

        plt.plot(epochs, train_losses_z, color='r', label='Training loss')
        plt.plot(epochs, valid_losses_z, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')

        plt.plot(epochs, train_losses_phi, color='m', label='Training loss')
        plt.plot(epochs, valid_losses_phi, color='black', label='Validation loss')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Learning curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        fig_path = os.path.join(DataVisualization.folderPath, '{}LearningCurves.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def PlotMSE(MSE):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

        epochs = range(1, len(MSE) + 1)
        MSE = torch.stack(MSE, 0)
        x = MSE[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = MSE[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = MSE[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r', label='z')
        phi = MSE[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', label='phi')
        plt.legend()
        plt.title('Pose Variables MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.xticks(np.arange(0, len(MSE), step=5))

        fig_path = os.path.join(DataVisualization.folderPath, '{}MSE.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def PlotMAE(MAE):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

        epochs = range(1, len(MAE) + 1)
        MAE = torch.stack(MAE, 0)
        x = MAE[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = MAE[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = MAE[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r', label='z')
        phi = MAE[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', label='phi')
        plt.legend()
        plt.title('Pose Variables MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.xticks(np.arange(0, len(MAE), step=5))

        fig_path = os.path.join(DataVisualization.folderPath, '{}MAE.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def PlotR2Score(r2_score):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

        epochs = range(1, len(r2_score) + 1)
        r2_score = torch.stack(r2_score, 0)
        x = r2_score[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = r2_score[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = r2_score[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r',  label='z')
        phi = r2_score[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', label='phi')
        plt.legend()
        plt.title('Pose Variables ')
        plt.xlabel('Epoch')
        plt.ylabel('')
        plt.xticks(np.arange(0, len(r2_score), step=5))

        fig_path = os.path.join(DataVisualization.folderPath, '{}Rsq.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def PlotGTandEstimationVsTime(gt_labels, predictions):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gt_labels = torch.stack(gt_labels, 0)
        predictions = torch.stack(predictions, 0)
        gt_labels = gt_labels.cpu().numpy()
        gt_labels = np.reshape(gt_labels, (-1, 4))
        predictions = predictions.cpu().numpy()
        predictions = np.reshape(predictions, (-1, 4))
        samples = len(gt_labels[:, 0])
        samples = range(1, samples+1)

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')
        x_gt = gt_labels[:, 0]
        x_pred = predictions[:, 0]
        plt.plot(samples, x_gt, color='green', label='GT')
        plt.plot(samples, x_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')
        y_gt = gt_labels[:, 1]
        y_pred = predictions[:, 1]
        plt.plot(samples, y_gt, color='blue', label='GT')
        plt.plot(samples, y_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')
        z_gt = gt_labels[:, 2]
        z_pred = predictions[:, 2]
        plt.plot(samples, z_gt, color='r', label='GT')
        plt.plot(samples, z_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')
        phi_gt = gt_labels[:, 3]
        phi_pred = predictions[:, 3]
        plt.plot(samples, phi_gt, color='m', label='GT')
        plt.plot(samples, phi_pred, color='black', label='Prediction')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground Truth and Predictions vs time')

        fig_path = os.path.join(DataVisualization.folderPath, '{}GTandPredVsTime.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def DisplayPlots():
        plt.show()

    @staticmethod
    def PlotGTVsEstimation(gt_labels, predictions):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))

        gt_labels = torch.stack(gt_labels, 0)
        predictions = torch.stack(predictions, 0)
        gt_labels = gt_labels.cpu().numpy()
        gt_labels = np.reshape(gt_labels, (-1, 4))
        predictions = predictions.cpu().numpy()
        predictions = np.reshape(predictions, (-1, 4))

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')
        ax.set_xmargin(0.2)
        x_gt = gt_labels[:, 0]
        x_pred = predictions[:, 0]
        plt.scatter(x_gt, x_pred, color='green', marker='o')
        plt.plot(x_gt, x_gt, color='black', linestyle='--')

        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')
        ax.set_xmargin(0.2)
        y_gt = gt_labels[:, 1]
        y_pred = predictions[:, 1]
        plt.scatter(y_gt, y_pred, color='blue', marker='o')
        plt.plot(y_gt, y_gt, color='black', linestyle='--')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')
        ax.set_xmargin(0.2)
        z_gt = gt_labels[:, 2]
        z_pred = predictions[:, 2]
        plt.scatter(z_gt, z_pred, color='r', marker='o')
        plt.plot(z_gt, z_gt, color='black', linestyle='--')

        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')
        ax.set_xmargin(0.2)
        phi_gt = gt_labels[:, 3]
        phi_pred = predictions[:, 3]
        plt.scatter(phi_gt, phi_pred, color='m', marker='o')
        plt.plot(phi_gt, phi_gt, color='black', linestyle='--')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground Truth vs Predictions')

        fig_path = os.path.join(DataVisualization.folderPath, '{}GTvsPred.png'.format(DataVisualization.desc))
        plt.savefig(fig_path)

    @staticmethod
    def DisplayFrameAndPose(frame, gt_labels, predictions):
        #DataVisualization.figure_counter += 1
        fig = plt.figure(666, figsize=(10, 6))

        w = 20
        h = 12
        bar_length = h - 2
        offset_x = int((w-bar_length)/2)
        ax1 = plt.subplot2grid((h, w), (0, offset_x), colspan=bar_length)
        ax1.set_title('x')
        ax1.xaxis.tick_top()
        x_gt = gt_labels[0]
        x_pred = predictions[0]
        ax1.set_xlim([0, 4])
        ax1.set_ylim([-0.5, 0.5])
        ax1.set_yticklabels([])
        plt.scatter(x_gt, -0.05,  color='green', label='GT', s=100)
        plt.scatter(x_pred, 0.05, color='blue', label='Prediction', s=100)

        ax2 = plt.subplot2grid((h, w), (1, 0), rowspan=bar_length)
        ax2.set_title('y')
        y_gt = gt_labels[1]
        y_pred = predictions[1]
        ax2.set_ylim([-1, 1])
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_xticklabels([])
        plt.scatter(-0.05, y_gt, color='green', label='GT', s=100)
        plt.scatter(0.05, y_pred, color='blue', label='Prediction', s=100)

        ax3 = plt.subplot2grid((h, w), (1, 1), rowspan=bar_length, colspan=(w-2))
        ax3.axis('off')
        frame = frame.transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        plt.imshow(frame)

        ax4 = plt.subplot2grid((h, w), (1, w-1), rowspan=bar_length)
        ax4.set_title('z')
        z_gt = gt_labels[2]
        z_pred = predictions[2]
        ax4.yaxis.tick_right()
        ax4.set_ylim([-1, 1])
        ax4.set_xlim([-0.5, 0.5])
        ax4.set_xticklabels([])
        plt.scatter(-0.05, z_gt, color='green', label='GT', s=100)
        plt.scatter(0.05, z_pred, color='blue', label='Prediction', s=100)

        ax5 = plt.subplot2grid((h, w), (h-1, offset_x), colspan=bar_length)
        ax5.set_title('phi')
        phi_gt = gt_labels[3]
        phi_pred = predictions[3]
        ax5.set_xlim([-2, 2])
        ax5.set_ylim([-0.5, 0.5])
        ax5.set_yticklabels([])
        plt.scatter(phi_gt, -0.05, color='green', label='GT', s=100)
        plt.scatter(phi_pred, 0.05,  color='blue', label='Prediction', s=100)

        plt.subplots_adjust(hspace=1.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        return fig

        # fig_path = os.path.join(DataVisualization.folderPath, '{}GTandPredandPose.png'.format(DataVisualization.desc))
        # plt.savefig(fig_path)

    @staticmethod
    def CoolDroneStuff(frame, gt_labels, predictions):
        fig = plt.figure(888, figsize=(15, 5))

        img = mpimg.imread('minidrone.jpg')
        frame = frame.transpose(1, 2, 0)
        frame = frame.astype(np.uint8)

        h = 5
        w = 15

        x_gt = gt_labels[0]
        x_pred = predictions[0]
        y_gt = gt_labels[1]
        y_pred = predictions[1]
        z_gt = gt_labels[2]
        z_pred = predictions[2]
        phi_gt = gt_labels[3] - np.pi/2
        phi_pred = predictions[3] - np.pi/2

        str1 = "x_gt={:05.3f}, y_gt={:05.3f}, z_gt={:05.3f}, phi_gt={:05.3f}".format(x_gt, y_gt, z_gt, phi_gt)
        str2 = "x_pr={:05.3f}, y_pr={:05.3f}, z_pr={:05.3f}, phi_pr={:05.3f}".format(x_pred, y_pred, z_pred, phi_pred)

        ax0 = plt.subplot2grid((h, w), (0, 0), colspan=6)
        ax0.axis('off')
        ax0.text(0, 1.5, str1, fontsize=10)
        ax0.text(0, 1, str2, fontsize=10)

        ax1 = plt.subplot2grid((h, w), (1, 0), colspan=7, rowspan=4)
        ax1.set_title('Relative Pose (x,y)')
        ax1.set_xlim([-3, 3])
        ax1.set_ylim([0, 3])
        ax1.yaxis.set_ticks([0, 1.5, 3])  # set y-ticks
        ax1.xaxis.set_ticks([-3.0, -1.5, 0, 1.5, 3.0])  # set y-ticks
        ax1.xaxis.tick_top()  # and move the X-Axis
        ax1.yaxis.tick_left()  # remove right y-Ticks
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        trianglex = [3, 0, -3, 3]
        triangley = [3, 0, 3, 3]
        plt.fill(trianglex, triangley, facecolor='lightskyblue')

        plt.plot(y_gt, x_gt, color='green', label='GT', linestyle='None', marker='o', markersize=10)
        plt.plot(y_pred, x_pred, color='blue', label='Prediction', linestyle='None', marker='o', markersize=10)
        ax1.arrow(y_gt, x_gt, np.cos(phi_gt), np.sin(phi_gt), head_width=0.05, head_length=0.05, color='green')
        ax1.arrow(y_pred, x_pred, np.cos(phi_pred), np.sin(phi_pred), head_width=0.05, head_length=0.05, color='blue')
        plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

        ax2 = plt.subplot2grid((h, w), (1, 7), rowspan=4)
        ax2.set_title('Relative z', pad=20)
        ax2.yaxis.tick_right()
        ax2.set_ylim([-1, 1])
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_xticklabels([])
        ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
        ax2.xaxis.set_ticks_position('none')
        plt.scatter(-0.05, z_gt, color='green', label='GT', s=100)
        plt.scatter(0.05, z_pred, color='blue', label='Prediction', s=100)

        ax3 = plt.subplot2grid((h, w), (1, 8), rowspan=4, colspan=7)
        ax3.set_title('Frame', pad=25)
        ax3.axis('off')

        plt.imshow(frame)
        plt.subplots_adjust(wspace=1.5)

        newax = fig.add_axes([0.248, 0.0, 0.1, 0.1], anchor='S')
        newax.imshow(img)
        newax.axis('off')
