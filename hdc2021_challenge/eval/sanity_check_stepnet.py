import os

import torch 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from hdc2021_challenge.deblurrer.StepNet_deblurrer import StepNetDeblurrer
from hdc2021_challenge.forward_model.bokeh_blur_rfft_train import BokehBlur


RADIUS_DICT = {
    0 : 1.0*8., 1 : 1.2*8., 2 : 1.3*8., 3 : 1.4*8., 4 : 2.2*8.,
    5 : 3.75*8., 6 : 4.5*8., 7 : 5.25*8., 8 : 6.75*8., 9 : 8.2*8.,
    10 : 8.8*8., 11 : 9.4*8., 12 : 10.3*8., 13 : 10.8*8., 14 : 11.5*8.,
    15 : 12.1*8., 16 : 13.5*8., 17 : 16.0*8., 18 : 17.8*8., 19 : 19.4*8.
}

base_path = os.path.join(os.path.dirname(__file__), '..')
experiment_name = 'deblurring' 
version = 'stepnet_deblurring_robust'
chkp_name = 'all_steps'
path_parts = [base_path, 'weights', experiment_name, version, chkp_name + '.ckpt']
chkp_path = os.path.join(*path_parts)

reconstructor = StepNetDeblurrer.load_from_checkpoint(chkp_path)
reconstructor.to("cuda")
reconstructor.prediction_mode = True

transform = transforms.Compose(
    [transforms.Grayscale(), 
    transforms.ToTensor(), 
    transforms.Resize(size=(1460, 2360))])

trainset = torchvision.datasets.STL10(root="/localdata/STL10", split='train', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

images, _ = next(iter(trainloader))

for step in range(20):
    print('Sanity check for step: ' + str(step))
    reconstructor.set_step(step)

    with torch.no_grad():
        blur = BokehBlur(r=RADIUS_DICT[step], shape=(1460, 2360))

        image_blur = blur(images.to("cuda"))
        image_reco = reconstructor.forward(image_blur)
        image_blur = image_blur.cpu()
        image_reco = image_reco.cpu()

        for i in range(image_reco.shape[0]):
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            fig.set_size_inches(32, 24)

            p1 = ax1.imshow(images[i,0,:,:], cmap="gray")
            ax1.set_title("gt x")
            p2 = ax2.imshow(image_blur[i,0,:,:], cmap="gray")
            ax2.set_title("blurred y")
            p3 = ax3.imshow(image_reco[i,0,:,:], cmap="gray")
            ax3.set_title("reconstruction")
            fig.tight_layout()

            plt.show()
            plt.savefig('images/step_' + str(step) + '_image_' + str(i), dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', format=None, transparent=False, bbox_inches=None,
            pad_inches=0.1, metadata=None)
