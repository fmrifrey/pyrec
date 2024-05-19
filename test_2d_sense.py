# import packages
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchkbnufft as tkbn
from skimage.data import shepp_logan_phantom
import utils.opt
from recon.TV_FISTA import *

filterwarnings("ignore") # ignore floor divide warnings
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# create a simple shepp logan phantom
image = shepp_logan_phantom().astype(complex)
im_size = image.shape

image = torch.tensor(image).to(device).unsqueeze(0).unsqueeze(0)

# create a k-space trajectory and plot it
spokelength = image.shape[-1] * 2
grid_size = (spokelength, spokelength)
nspokes = 17

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
ktraj = torch.tensor(ktraj).to(device)

# create NUFFT objects, use 'ortho' for orthogonal FFTs
nufft_ob = tkbn.KbNufft(
    im_size=im_size,
    grid_size=grid_size,
).to(image)
adjnufft_ob = tkbn.KbNufftAdjoint(
    im_size=im_size,
    grid_size=grid_size,
).to(image)

# calculate k-space data
kdata = nufft_ob(image, ktraj)

# add some noise (robustness test)
siglevel = torch.abs(kdata).mean()
kdata = kdata + (siglevel/5) * torch.randn(kdata.shape).to(kdata)

# method 1: no density compensation (blurry image)
image_blurry = adjnufft_ob(kdata, ktraj)
image_blurry_numpy = np.squeeze(image_blurry.cpu().numpy())

smaps = torch.rand(1, 8, 400, 400) + 1j * torch.rand(1, 8, 400, 400)
def A_fwd(x):
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
    return nufft_ob(x,ktraj,smaps=smaps.to(x))
def A_adj(b):
    return torch.tensor(adjnufft_ob(b,ktraj,smaps=smaps.to(image_blurry))).squeeze()

L = utils.opt.pwritr(A_fwd,A_adj,image_blurry.squeeze())
image_tvfista, cost, x_set = tvdeblur(A_fwd, A_adj, kdata, niter=2, L=L)
plt.figure(0)
image_tvfista_numpy = np.squeeze(image_tvfista.cpu().numpy())
plt.imshow(np.absolute(image_tvfista_numpy))
plt.show()
