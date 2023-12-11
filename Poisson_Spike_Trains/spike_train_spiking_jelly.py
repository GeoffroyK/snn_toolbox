import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, encoding, learning
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

'''
PoC Script using spikingjelly module to create a spike train using an input image following a Poisson law depending on the pixel intensity.
'''

class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(28 * 28, 100, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)

'''
Pre-processing of the input image following the strategy described in : https://github.com/QVPR/VPRSNN
as we want to use this method for VPR.
'''
def get_patches2D(image, patch_size):

    if patch_size[0] % 2 == 0: 
        nrows = image.shape[0] - patch_size[0] + 2
        ncols = image.shape[1] - patch_size[1] + 2
    else:
        nrows = image.shape[0] - patch_size[0] + 1
        ncols = image.shape[1] - patch_size[1] + 1
    return np.lib.stride_tricks.as_strided(image, patch_size + (nrows, ncols), image.strides + image.strides).reshape(patch_size[0]*patch_size[1],-1)

def patch_normalise_pad(image, patch_size):

    patch_size = (patch_size, patch_size)
    patch_half_size = [int((p-1)/2) for p in patch_size ]

    image_pad = np.pad(np.float64(image), patch_half_size, 'constant', constant_values=np.nan)

    nrows = image.shape[0]
    ncols = image.shape[1]
    patches = get_patches2D(image_pad, patch_size)
    mus = np.nanmean(patches, 0)
    stds = np.nanstd(patches, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        out = (image - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

    out[np.isnan(out)] = 0.0
    out[out < -1.0] = -1.0
    out[out > 1.0] = 1.0
    return out

def loadImg(imgPath):
    # Read and convert image from BGR to RGB
    img = cv.imread(imgPath)[:,:,::-1]
    return img

def process_Nordland_Image(imgPath, imWidth, imHeight, numPatches):

    img = loadImg(imgPath)

    img = cv.resize(img, (imWidth, imHeight))
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    img_norm = patch_normalise_pad(img, numPatches)

    # Scale elements to be between 0 and 255
    img = np.uint8(255. * (1 + img_norm) / 2.0)
    # Scale elements to be between 0 and 1 for Pytorch implementation
    img = img / 255.

    return img

'''
    Test for STDP learner declaration
'''
def f_weight(x):
    return torch.clamp(x -1, 1)

if __name__ == "__main__":
    vpr = VPR(tau=2.0)
    vpr.to('cpu')

    img = process_Nordland_Image("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-00001.png", 28, 28, 7)
    
    img = torch.tensor(img)
    encoder = encoding.PoissonEncoder() 
    encoded_img = encoder(img)

    out = vpr(encoded_img.float())

    l_spikes = []
    ll_spikes = []

    # Sample 1
    for i in range(100):
        encoded_img = encoder(img)
        spikes = encoded_img.numpy()
        l_spikes.append(spikes[0][0])

    # Sample 2
    for i in range(100):
        encoded_img = encoder(img)
        spikes = encoded_img.numpy()
        ll_spikes.append(spikes[0][0])

# Plotting stuff
plt.plot(np.arange(len(l_spikes)), l_spikes, label="Sampling 1")
plt.plot(np.arange(len(ll_spikes)), ll_spikes, label="Sampling 2")
plt.title("Spike-train using Poisson Law with pixel intensity")
plt.tight_layout()
plt.legend()
plt.show()

plt.imshow(img.numpy(), cmap="gray")
plt.show()