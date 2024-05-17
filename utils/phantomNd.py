import torch
import numpy as np

def phantomNd(N, E=None):

    # set default phantom
    if E is None:
        E = 'Modified Shepp-Logan'

    # load default head phantom
    if isinstance(E, str):
        if E.lower() == 'shepp-logan':
            E = shepp_logan()
        elif E.lower() == 'modified shepp-logan':
            E = modified_shepp_logan()
        elif E.lower() == 'yu-ye-wang':
            E = yu_ye_wang()
        else:
            raise ValueError(f'Invalid default phantom: {E}')

    # simulate random noise along time
    N = np.pad(N, (0, max(0, 4 - len(N))), mode='constant', constant_values=1)
    if N[3] > 1 and E.shape[2] == 1:
        E = E.repeat(N[3], axis=2)
        E[:, 4:, 1:] += 0.1 * E[:, 4:, 1:] * (2 * np.random.rand(E.shape[0], 6, N[3] - 1) - 1)
    elif N[3] != E.shape[2]:
        raise ValueError('size(E,3) must be == Nt, or 1 to simulate random noise')

    # make image grid points
    x = [np.linspace(-1, 1, n) if n > 1 else np.zeros(n) for n in N[:3]]
    X, Y, Z = np.meshgrid(*x, indexing='ij')
    r = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

    # initialize vectorized image array
    P = torch.zeros((r.shape[1], N[3]))

    for nt in range(N[3]):  # loop through time points
        for ne in range(E.shape[0]):  # loop through ellipsoids
            # get properties of current ellipse
            rho = E[ne, 0, nt]  # intensity
            D = np.diag(E[ne, 1:4, nt])  # stretch
            rc = E[ne, 4:7, nt].reshape(-1, 1)  # center
            R = eul2rotm(np.pi / 180 * E[ne, 7:10, nt][::-1])  # rotation

            # determine ellipsoid ROI and add amplitude
            ROI = torch.norm(torch.tensor(np.linalg.inv(D) @ R.T @ (r - rc)), dim=0) <= 1
            P[ROI, nt] += rho

    # reshape the image
    P = P.reshape(*N)
    return P

def eul2rotm(eul):

    phi, theta, psi = eul
    R_z1 = np.array([ # in-plane rotation
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    R_x = np.array([ # azimuthal rotation
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    R_z2 = np.array([ # polar rotation
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])

    return R_z2 @ R_x @ R_z1

def shepp_logan():
    e = modified_shepp_logan()
    e[:, 0] = np.array([1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return e

def modified_shepp_logan():
    return np.array([
        [1, 0.69, 0.92, 0.81, 0, 0, 0, 0, 0, 0],
        [-0.8, 0.6624, 0.874, 0.78, 0, -0.0184, 0, 0, 0, 0],
        [-0.2, 0.11, 0.31, 0.22, 0.22, 0, 0, -18, 0, 10],
        [-0.2, 0.16, 0.41, 0.28, -0.22, 0, 0, 18, 0, 10],
        [0.1, 0.21, 0.25, 0.41, 0, 0.35, -0.15, 0, 0, 0],
        [0.1, 0.046, 0.046, 0.05, 0, 0.1, 0.25, 0, 0, 0],
        [0.1, 0.046, 0.046, 0.05, 0, -0.1, 0.25, 0, 0, 0],
        [0.1, 0.046, 0.023, 0.05, -0.08, -0.605, 0, 0, 0, 0],
        [0.1, 0.023, 0.023, 0.02, 0, -0.606, 0, 0, 0, 0],
        [0.1, 0.023, 0.046, 0.02, 0.06, -0.605, 0, 0, 0, 0]
    ])

def yu_ye_wang():
    return np.array([
        [1, 0.69, 0.92, 0.9, 0, 0, 0, 0, 0, 0],
        [-0.8, 0.6624, 0.874, 0.88, 0, 0, 0, 0, 0, 0],
        [-0.2, 0.41, 0.16, 0.21, -0.22, 0, -0.25, 108, 0, 0],
        [-0.2, 0.31, 0.11, 0.22, 0.22, 0, -0.25, 72, 0, 0],
        [0.2, 0.21, 0.25, 0.5, 0, 0.35, -0.25, 0, 0, 0],
        [0.2, 0.046, 0.046, 0.046, 0, 0.1, -0.25, 0, 0, 0],
        [0.1, 0.046, 0.023, 0.02, -0.08, -0.65, -0.25, 0, 0, 0],
        [0.1, 0.046, 0.023, 0.02, 0.06, -0.65, -0.25, 90, 0, 0],
        [0.2, 0.056, 0.04, 0.1, 0.06, -0.105, 0.625, 90, 0, 0],
        [-0.2, 0.056, 0.056, 0.1, 0, 0.1, 0.625, 0, 0, 0]
    ])
