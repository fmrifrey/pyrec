# import packages
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from rec.TV_FISTA import tvdeblur
from skimage.data import shepp_logan_phantom

filterwarnings("ignore") # ignore floor divide warnings
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# create a simple shepp logan phantom and plot it
image = utils.phantomNd()
im_size = image.shape

# convert the phantom to a tensor and unsqueeze coil and batch dimension
image = torch.tensor(image, device=device).unsqueeze(0).unsqueeze(0)

# create a k-space trajectory and plot it
spokelength = image.shape[-1] * 2
grid_size = (spokelength, spokelength)
nspokes = 64

ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
kx = np.zeros(shape=(spokelength, nspokes))
ky = np.zeros(shape=(spokelength, nspokes))
ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
for i in range(1, nspokes):
    kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
    ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
    
ky = np.transpose(ky)
kx = np.transpose(kx)

ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

# convert k-space trajectory to a tensor
ktraj = torch.tensor(ktraj, device=device)

# build nufft operators
nufft_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size).to(image)
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size).to(image)

# calculate k-space data
kdata = nufft_ob(image, ktraj, smaps=None)

# add some noise (robustness test)
siglevel = torch.abs(kdata).mean()
kdata = kdata + (siglevel/5) * torch.randn(kdata.shape).to(kdata)

# convert kdata to numpy
kdata_numpy = np.reshape(kdata.cpu().numpy(), (1, nspokes, spokelength))

# adjnufft back
# method 1: no density compensation (blurry image)
image_blurry = adjnufft_ob(kdata, ktraj, smaps=None)

# show the images, some phase errors may occur due to smap
image_blurry_numpy = np.squeeze(image_blurry.cpu().numpy())
plt.figure(0)
plt.imshow(np.absolute(image_blurry_numpy))
plt.gray()
plt.title('blurry image')
plt.show()

# define fwd and adj operators for iterative recon:
def A_fwd(x):
    return nufft_ob(x, ktraj)
def A_adj(b):
    return adjnufft_ob(b, ktraj, smaps=None)

image_ir = tvdeblur(A_fwd,A_adj,kdata)