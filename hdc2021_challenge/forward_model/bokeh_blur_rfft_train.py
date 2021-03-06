import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 


class BokehBlur(nn.Module):
    def __init__(self, r=500., shape=(1460, 2360)):
        super(BokehBlur, self).__init__()

        self.r = torch.nn.parameter.Parameter(torch.tensor(r).float(), requires_grad=True)
        self.shape = shape

        kx, ky = torch.meshgrid(torch.arange(-self.shape[0]//2, self.shape[0]//2),
                              torch.arange(-self.shape[1]//2, self.shape[1]//2))
        kx = kx.float()
        ky = ky.float()                              
        self.KR = torch.sqrt(kx**2 + ky**2)
        self.scale = torch.tensor(200.).float()
    
    def get_filter(self):
        D_numeric =  1.0/(np.pi*self.r.to(self.r.device)**2)*torch.sigmoid(self.scale.to(self.r.device)*(self.r - self.KR.to(self.r.device)))

        D_numeric = torch.fft.fftshift(D_numeric)
        D_filter = torch.fft.rfft2(D_numeric)
        return D_filter

    def forward(self, x):
        D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        D_numeric = torch.fft.fftshift(D_numeric)
        D_filter = torch.fft.rfft2(D_numeric)

        x_fft = torch.fft.rfft2(x)
        x_processed = torch.multiply(D_filter,x_fft)

        x_ifft = torch.fft.irfft2(x_processed)
        return x_ifft

    def adjoint(self, x):
        D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        D_numeric = torch.fft.fftshift(D_numeric)
        D_filter = torch.fft.rfft2(D_numeric)
        D_filter_adjoint = torch.conj(D_filter)

        x_fft = torch.fft.rfft2(x)
        x_processed = torch.multiply(D_filter_adjoint.to(x.device),x_fft)

        x_ifft = torch.fft.irfft2(x_processed)
        return x_ifft

    def grad(self, x, y):
        
        x_fft = torch.fft.rfft2(x)
        y_fft = torch.fft.rfft2(y)

        D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        D_filter = torch.fft.rfft2(D_numeric)

        f_grad = torch.multiply(D_filter.to(x.device)**2, x_fft) - torch.multiply(D_filter.to(x.device), y_fft)
        out = torch.fft.irfft2(f_grad)
    
        return out


if __name__ == "__main__":
    from hdc2021_challenge.utils.data_util import load_data
    import matplotlib.pyplot as plt 

    forward_model = BokehBlur(r=69.)

    step=9
    x, y = load_data('Verdana',step)
    x = x[0,:,:]/65535.
    y = y[0,:,:]/65535.
    
    x_torch = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y_torch = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)

    print(x_torch.shape, y_torch.shape)
    
    with torch.no_grad():
        y_pred = forward_model(x_torch)

    print(y_pred.shape)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)

    im = ax1.imshow(x_torch[0,0,:,:], cmap="gray")
    ax1.set_title("Groundtruth")
    fig.colorbar(im, ax=ax1)
    im = ax2.imshow(y_torch[0,0,:,:], cmap="gray")
    ax2.set_title("Blurred Image")
    fig.colorbar(im, ax=ax2)

    im = ax3.imshow(y_pred[0,0,:,:], cmap="gray")
    ax3.set_title("Conv with Disk")
    fig.colorbar(im, ax=ax3)

    fig.suptitle("Blurring Step: " + str(step))
    plt.show()

    lamb = 1/(torch.max(torch.abs(forward_model.get_filter()))**2*6)
    xk = torch.zeros_like(x_torch)
    print(lamb)
    with torch.no_grad():
        for i in range(501):
            xk = xk - lamb*forward_model.grad(xk, y_torch) #+ 0.05*lamb*pm
            if i % 50 == 0:
                print(i)
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True)

                ax1.imshow(x_torch[0,0,:,:], cmap="gray")
                ax1.set_title("Groundtruth")

                ax3.imshow(torch.abs(xk[0,0,:,:]), cmap="gray")
                ax3.set_title("Reconstruction at iter " + str(i))

                fig.suptitle("Blurring Step: " + str(step))
                plt.savefig("GD_{}.png".format(i))
                plt.close()
