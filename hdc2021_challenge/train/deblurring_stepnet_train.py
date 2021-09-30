"""
Training script for the StepNet model. It will be trained on a mixture of the
blurred text images and simulated blurry images of the STL10 dataset.
Notice: This model can use a lot of GPU memory during training for higher steps.
Use higher downsampling levels, smaller models or 16Bit precision if necessary.
"""

import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from hdc2021_challenge.utils.blurred_dataset import MultipleBlurredDataModule
from hdc2021_challenge.deblurrer.StepNet_deblurrer import StepNetDeblurrer


DOWN_SHAPES = {
    1 : (1460, 2360),
    2 : (730, 1180),
    3 : (365, 590)
}

# Basic train setting
start_step = 0
batch_size = 3
epochs = 50
downsampling = 3
path_to_ckpt = None

# Create or load a reconstructor
if path_to_ckpt is None:
    reconstructor = StepNetDeblurrer(lr=1e-4,
                                    downsampling=downsampling,
                                    step=start_step,
                                    scales=5,
                                    skip_channels=4,
                                    channels=(32, 64, 128, 256, 256),
                                    use_sigmoid=True,
                                    batch_norm=True,
                                    reuse_input=False,
                                    which_loss='mse',
                                    jittering_std=0.0005)
else:
    reconstructor = StepNetDeblurrer.load_from_checkpoint(path_to_ckpt, strict=False)

# Train each separate U-Net of the whole StepNet model one after another
# During training for blurring level i, the parameters of all U-Nets, except for U-Net i, are frozen.
# The training routine starts with the training on level 0 for a fixed number of epochs. Afterwards,
# the best parameters for U-Net 0 are loaded and frozen. This continues until the end of level 19 
# is reached.
for step in range(start_step, 20):

    # Prepare dataset
    dataset = MultipleBlurredDataModule(batch_size=batch_size, blurring_step=step,
                                        img_size=DOWN_SHAPES[downsampling],
                                        num_data_loader_workers=0)
    dataset.prepare_data()
    dataset.setup()

    # Save model with best OCR accuracy on the validation set
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        verbose=True,
        monitor='val_ocr_acc',
        mode='max'
    )

    # Path for storing weights and the tensorboard log
    base_path = 'deblurring_experiments'
    experiment_name = 'stepnet_deblurring_robust'
    blurring_step = "step_" + str(step)
    path_parts = [base_path, experiment_name, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

    # Arguments for the Pytorch Lightning trainer
    trainer_args = {'plugins': DDPPlugin(find_unused_parameters=False),
                    'gpus': -1,
                    'default_root_dir': log_dir,
                    'callbacks': [checkpoint_callback],
                    'benchmark': False,
                    'fast_dev_run': False,
                    'gradient_clip_val': 1.0,
                    'logger': tb_logger,
                    'log_every_n_steps': 20,
                    'auto_scale_batch_size': 'binsearch',
                    'multiple_trainloader_mode': 'min_size'}

    # Train model
    trainer = pl.Trainer(max_epochs=epochs, **trainer_args)

    # Freeze all U-Nets, except the i-th
    reconstructor.set_step_train(step=step)
    trainer.fit(reconstructor, datamodule=dataset)

    # Load the best weights for U-Net i from the current training to use this (fixed) weights 
    # in the training of U-Net i+1
    reconstructor = reconstructor.load_from_checkpoint(checkpoint_callback.best_model_path)
