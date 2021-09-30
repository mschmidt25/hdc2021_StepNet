"""
Concatenation of 20 U-Nets for deblurring. Each U-Net is supposed to 
reduce the blur level by 1.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import numpy as np
from dival.reconstructors.networks.unet import get_unet_model

from hdc2021_challenge.utils.ocr import evaluateImage
from hdc2021_challenge.forward_model.bokeh_blur_rfft_train import BokehBlur


RADIUS_DICT = {
    0 : 1.0*8., 
    1 : 1.2*8., 
    2 : 1.3*8.,
    3 : 1.4*8., 
    4 : 2.2*8.,
    5 : 3.75*8.,
    6 : 4.5*8.,
    7 : 5.25*8., 
    8 : 6.75*8.,
    9 : 8.2*8.,
    10 : 8.8*8.,
    11 : 9.4*8.,
    12 : 10.3*8.,
    13 : 10.8*8.,
    14 : 11.5*8.,
    15 : 12.1*8.,
    16 : 13.5*8.,
    17 : 16.0*8., 
    18 : 17.8*8., 
    19 : 19.4*8.
}

DOWN_SHAPES = {
    1 : (1460, 2360),
    2 : (730, 1180),
    3 : (365, 590)
}


class StepNetDeblurrer(pl.LightningModule):
    def __init__(self, lr:float = 1e-5, downsampling:int = 1, step:int = 0, scales:int = 4,
                 skip_channels:int = 4, channels:tuple = None, use_sigmoid:bool = False, batch_norm:bool = True,
                 reuse_input:bool = False, which_loss:str = 'mse', jittering_std:float = 0.0):
        """
        A deblurrer which consists of 20 U-Nets. The U-Nets are connect in a row. The task of U-Net i is to
        deblur an image from blurring step i to i-1, where i=-1 is the final output (reconstruction).

        For training and inference, the architecture of the network is dynamic. An input with blurring level
        i will pass trough the first i U-Nets in descending order, i.e. i -> i-1 -> i-2 -> ... -> 0 -> -1.

        Args:
            lr (float, optional): Learning rate for the Adam optimizer. Defaults to 1e-5.
            downsampling (int, optional): Power 2^(x-1) of the average pooling downsampling, e.g. 
                                          downsampling=3 -> 2²=4 times spatial downsampling for
                                          the input. The output will be automatically upsampled
                                          by nearest interpolation to match the ground truth size.
                                          Defaults to 1.
            step (int, optional): Current blur step for which the model is used. Defaults to 0.
            scales (int, optional): Number of scales in each U-Net. Defaults to 4.
            skip_channels (int, optional): Number of skip channels in each U-Net. Defaults to 4.
            channels (tuple, optional): Number of channels for each scale in the U-Net. Defaults to None.
            use_sigmoid (bool, optional): Sigmoid activation on each U-Net output. Defaults to False.
            batch_norm (bool, optional): Use batch normalization. Defaults to True.
            reuse_input (bool, optional): Add the blurry measurement as an additional input to all U-Nets. Defaults to False.
            which_loss (str, optional): Choose the loss function from "l1", "mse" and "both". Defaults to "mse".
            jittering_std (float, optional): Add small Gaussian noise with mean 0 and this std to the measurements. Defaults to 0.0
        """
        super().__init__()

        self.lr = lr
        self.downsampling = downsampling
        self.step = step
        self.reuse_input = reuse_input
        self.which_loss = which_loss
        self.jittering_std = jittering_std

        self.blur = BokehBlur(r=RADIUS_DICT[step] / (2**(self.downsampling-1)),
                              shape=DOWN_SHAPES[self.downsampling])

        if channels is None:
            channels = (32, 32, 64, 64, 64, 64)

        save_hparams = {
            'lr': lr,
            'downsampling': downsampling,
            'scales': scales,
            'skip_channels': skip_channels,
            'channels': channels,
            'use_sigmoid': use_sigmoid,
            'batch_norm': batch_norm,
            'reuse_input': reuse_input,
            'which_loss': which_loss,
            'jittering_std': jittering_std
        }
        self.save_hyperparameters(save_hparams)

        if self.reuse_input:
            in_ch = 2
        else:
            in_ch = 1

        self.net = nn.ModuleList([get_unet_model(in_ch=in_ch, out_ch=1, scales=scales, skip=skip_channels,
                                                 channels=channels, use_sigmoid=use_sigmoid, use_norm=batch_norm)
                                  for i in range(20)])

        if self.downsampling > 1:
            self.down = [nn.AvgPool2d(kernel_size=3, stride=2, padding=1) for i in range(self.downsampling-1)]
            self.down = nn.Sequential(*self.down)
            self.up = nn.Upsample(size=DOWN_SHAPES[1], mode='nearest')

        self.prediction_mode = False

    def forward(self, y):
        if self.prediction_mode and self.downsampling > 1:
            y = self.down(y)

        if self.reuse_input:
            y_init = y

        for i in range(self.step, -1, -1):
            if self.reuse_input:
                y = torch.cat([y, y_init], dim=1)
            y = self.net[i](y)

        if self.prediction_mode and self.downsampling > 1:
            y = self.up(y)

        return y

    def set_step_train(self, step):
        self.step = step
        self.blur = BokehBlur(r=RADIUS_DICT[step] / (2**(self.downsampling-1)),
                              shape=DOWN_SHAPES[self.downsampling])
        for i in range(20):
            for param in self.net[i].parameters():
                param.requires_grad = i == self.step

    def set_step(self, step):
        self.step = step

    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], list):
            batch = [batch]

        x, y, _ = batch[0]

        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)

        if self.jittering_std > 0:
            y = y + torch.randn(y.shape,device=self.device)*self.jittering_std

        x_hat = self.forward(y) 

        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x)
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)
           
        for i in range(1, len(batch)):
            x, _ = batch[i]
            x = x[0:2, ...]

            y = self.blur(x)

            if self.jittering_std > 0:
                y = y + torch.randn(y.shape,device=self.device)*self.jittering_std
                x = x + torch.randn(x.shape,device=self.device)*self.jittering_std

            x_hat = self.forward(y) 
        
            loss = loss +  0.1 * F.mse_loss(x_hat, x)

        self.log('train_loss', loss)

        if batch_idx == 3:
            self.last_batch = batch

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, text = batch

        if batch_idx == 0:
            self.first_batch_val = batch

        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)
        
        x_hat = self.forward(y) 
        
        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x)
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)

        # preprocess image
        if self.downsampling > 1:
            x_hat = self.up(x_hat)
        x_hat = x_hat.cpu().numpy()
        x_hat = np.clip(x_hat, 0, 1)

        ocr_acc = []
        for i in range(len(text)):
            ocr_acc.append(evaluateImage(x_hat[i, 0, :, :], text[i]))

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_ocr_acc', np.mean(ocr_acc))
        
        return loss 
    
    def training_epoch_end(self, result):
        x, y, _ = self.last_batch[0]

        if self.downsampling:
            x = self.down(x)
            y = self.down(y)

        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.forward(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
            for idx in range(1, len(self.last_batch)):
                x, _ = self.last_batch[idx]
                with torch.no_grad():
                    y = self.blur(x)
                    x_hat = self.forward(y) 

                    reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "deblurred set " + str(idx) , reco_grid, global_step=self.current_epoch)

                    gt_grid = torchvision.utils.make_grid(x, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "ground set " + str(idx), gt_grid, global_step=self.current_epoch)

                    blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "blurred set " + str(idx), blurred_grid, global_step=self.current_epoch)

    def validation_epoch_end(self, result):
        x, y, _ = self.first_batch_val
        if self.downsampling > 1:
            x = self.down(x)
            y = self.down(y)

        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.forward(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
          
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
