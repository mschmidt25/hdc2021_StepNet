import os

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from hdc2021_challenge.utils.blurred_dataset import MultipleBlurredDataModule
from hdc2021_challenge.deblurrer.StepNet_deblurrer import StepNetDeblurrer


start_step = 0
batch_size = 1
epochs = 50
downsampling = 3
path_to_ckpt = None


if path_to_ckpt is None:
    reconstructor = StepNetDeblurrer(lr=1e-4,
                                    downsampling=downsampling,
                                    step=start_step,
                                    scales=5,
                                    skip_channels=4,
                                    channels=(32, 64, 128, 256, 256),
                                    use_sigmoid=True,
                                    batch_norm=True,
                                    reuse_input=False)
else:
    reconstructor = StepNetDeblurrer.load_from_checkpoint(path_to_ckpt, strict=False)

# Train each separate U-Net of the whole StepNet model one after another
# During training for blurring level i, the parameters of all U-Nets, except for U-Net i, are frozen.
# The training routine starts with the training on level 0 for a fixed number of epochs. Afterwards,
# the best parameters for U-Net 0 are loaded and frozen. This continues until the end of level 19 
# is reached.
for i in range(start_step, 20):

    blurring_step = i
    dataset = MultipleBlurredDataModule(batch_size=batch_size, blurring_step=blurring_step,
                                        img_size=(int(1460 // (2**downsampling)),
                                                  int(2360 // (2**downsampling) - 1)),
                                        num_data_loader_workers=0)
    dataset.prepare_data()
    dataset.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    base_path = 'deblurring'
    experiment_name = 'stepnet_deblurring_robust'
    blurring_step = "step_" + str(blurring_step)
    path_parts = [base_path, experiment_name, blurring_step]
    log_dir = os.path.join(*path_parts)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

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

    trainer = pl.Trainer(max_epochs=epochs, **trainer_args)

    # Freeze all U-Nets, except the i-th
    reconstructor.set_step_train(i)
    trainer.fit(reconstructor, datamodule=dataset)

    # Load the best weights for U-Net i from the current training to use this (fixed) weights 
    # in the training of U-Net i+1
    reconstructor = reconstructor.load_from_checkpoint(checkpoint_callback.best_model_path)
