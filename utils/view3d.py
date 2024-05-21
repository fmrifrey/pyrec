import torch
import numpy as np
import matplotlib.pyplot as plt

def ortho(x,offset=[0,0,0]):

    X, Y, Z = x.shape

    # compute the center indices
    x0 = X // 2
    y0 = Y // 2
    z0 = Z // 2

    # get cross-sections through the center
    xy = x[x0+offset[0], :, :].squeeze()  # depth slice
    xz = x[:, y0+offset[1], :].squeeze()  # height slice
    yz = x[:, :, z0+offset[2]].squeeze()  # width slice

    ov = torch.concatenate((xy,xz,yz),1)
    #plt.figure(0)
    #plt.imshow(np.abs(ov.cpu().numpy()))
    #plt.show()

    return ov

def lightbox(x):
    a = 0