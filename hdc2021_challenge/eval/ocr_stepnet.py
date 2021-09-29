import os
from pathlib import Path

import torch
import yaml
from dival.measure import PSNR, SSIM
from tqdm import tqdm
import numpy as np 
from skimage.transform import resize

from hdc2021_challenge.utils.blurred_dataset import BlurredDataModule
from hdc2021_challenge.deblurrer.StepNet_deblurrer import StepNetDeblurrer
from hdc2021_challenge.utils.ocr import evaluateImage


base_path = "../train/deblurring_experiments/stepnet_deblurring_r3"
experiment_name = 'step_19'
version = 'version_0'
chkp_name = 'epoch=22-step=3679'
path_parts = [base_path, experiment_name, 'default',
            version, 'checkpoints', chkp_name + '.ckpt']

chkp_path = os.path.join(*path_parts)
if not os.path.exists(chkp_path):
    print("File not found: ", chkp_path)

reconstructor = StepNetDeblurrer.load_from_checkpoint(chkp_path, strict=False)
reconstructor.to("cuda")
reconstructor.prediction_mode = True


for step in range(20):
    print("Eval OCR for step ", step)
    print("--------------------------------\n")
    save_report = True 
    plot_examples = False 

    reconstructor.set_step(step=step)

    dataset = BlurredDataModule(batch_size=1, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()

    num_test_images = len(dataset.test_dataloader().dataset)

    if save_report:
        report_name = version + '_' + chkp_name + '_step=' + str(step) + '_images=' + str(num_test_images) + "_ocr"
        report_path = path_parts[:-2]
        report_path.append(report_name)
        report_path = os.path.join(*report_path)
        Path(report_path).mkdir(parents=True, exist_ok=True)

    psnrs = []
    ssims = []
    ocr_acc = []
    with torch.no_grad():
        for i, batch in tqdm(zip(range(num_test_images), dataset.test_dataloader()),
                             total=num_test_images):
            gt, obs, text = batch
            obs = obs.to('cuda')

            # create reconstruction from observation
            reco = reconstructor.forward(obs)
            reco = reco.cpu().numpy()
            reco = np.clip(reco, 0, 1)
            
            # calculate quality metrics
            psnrs.append(PSNR(reco[0][0], gt.numpy()[0][0]))
            ssims.append(SSIM(reco[0][0], gt.numpy()[0][0]))
            ocr_acc.append(evaluateImage(reco[0][0], text[0]))

    mean_psnr = np.mean(psnrs)
    std_psnr = np.std(psnrs)
    mean_ssim = np.mean(ssims)
    std_ssim = np.std(ssims)

    print('---')
    print('Results:')
    print('mean psnr: {:f}'.format(mean_psnr))
    print('std psnr: {:f}'.format(std_psnr))
    print('mean ssim: {:f}'.format(mean_ssim))
    print('std ssim: {:f}'.format(std_ssim))
    print('mean ocr acc: ', np.mean(ocr_acc))

    if save_report:
        report_dict = {'settings': {'num_test_images': num_test_images},
                    'results': {'mean_psnr': float(np.mean(psnrs)) , 
                                'std_psnr': float(np.std(psnrs)),
                                'mean_ssim': float(np.mean(ssims)) ,
                                'std_ssim': float(np.std(ssims)), 
                                'mean_ocr_acc': float(np.mean(ocr_acc)) }}
        report_file_path =  os.path.join(report_path, 'report.yaml')
        with open(report_file_path, 'w') as file:
            documents = yaml.dump(report_dict, file)
