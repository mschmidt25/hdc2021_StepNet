import os
import argparse
from PIL import Image

import torch 
import matplotlib.pyplot as plt 
import numpy as np 

from hdc2021_challenge.deblurrer.StepNet_deblurrer import StepNetDeblurrer


parser = argparse.ArgumentParser(description='Apply Deblurrer to every image in a directory.')
parser.add_argument('input_files')
parser.add_argument('output_files')
parser.add_argument('step')


def main(input_files, output_files, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model 
    base_path = os.path.join(os.path.dirname(__file__), 'weights')
    experiment_name = 'deblurring' 
    version = 'stepnet_deblurring_robust'
    chkp_name = 'all_steps'
    path_parts = [base_path, experiment_name, version, chkp_name + '.ckpt']
    chkp_path = os.path.join(*path_parts)

    reconstructor = StepNetDeblurrer.load_from_checkpoint(chkp_path)
    reconstructor.to(device)
    
    # Enable reconstruction mode and set correct step
    reconstructor.prediction_mode = True
    reconstructor.set_step(step=step)

    for f in os.listdir(input_files):
        if f.endswith("tif"):
            y = np.array(Image.open(os.path.join(input_files, f))) # not blurry
            print(y.shape)
            y = torch.from_numpy(y/65535.).float()
            y = y.unsqueeze(0).unsqueeze(0)
            y = y.to(device)
            with torch.no_grad():
                x_hat = reconstructor.forward(y)
                x_hat = x_hat.cpu().numpy()

            im = Image.fromarray(x_hat[0][0]*255.).convert("L")
            print(im)
            os.makedirs(output_files, exist_ok=True)
            im.save(os.path.join(output_files,f.split(".")[0] + ".PNG"))

    return 0


if __name__ == "__main__":

    args = parser.parse_args()
    main(args.input_files, args.output_files, args.step)