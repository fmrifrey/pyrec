import torch
import numpy as np

def phantom3d(N, E=None):

    # define default phantoms:
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
    def shepp_logan():
        e = modified_shepp_logan()
        e[:, 0] = np.array([1, -0.98, -0.02, -0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        return e
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

    # set default phantom
    if E is None:
        E = 'Modified Shepp-Logan'

    # convert 2d to 3d
    if len(N) == 2:
        np.append(N,1)

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

    # make image grid points
    x = [np.linspace(-1, 1, n) if n > 1 else np.zeros(n) for n in N[:3]]
    X, Y, Z = np.meshgrid(*x, indexing='ij')
    r = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=0)

    # initialize vectorized image array
    P = torch.zeros(r.shape[1], dtype=torch.complex128)

    for ne in range(E.shape[0]):  # loop through ellipsoids
        # get properties of current ellipse
        rho = E[ne, 0]  # intensity
        D = np.diag(E[ne, 1:4])  # stretch
        rc = E[ne, 4:7].reshape(-1, 1)  # center
        R = eul2rotm(np.pi / 180 * E[ne, 7:10][::-1])  # rotation

        # determine ellipsoid ROI and add amplitude
        ROI = torch.norm(torch.tensor(np.linalg.inv(D) @ R.T @ (r - rc)), dim=0) <= 1
        P[ROI] += rho

    # reshape the image
    P = P.reshape(*N)
    return P

def simsmaps(N,ncoils=4,sigma=0.8):

    # create grid
    x = np.linspace(-1,1,N[0])
    y = np.linspace(-1,1,N[1])
    z = np.linspace(-1,1,N[2])
    xv,yv,zv = np.meshgrid(x,y,z)

    # define coil sensitvity function
    def coil_center(n):
        return (np.cos(2 * np.pi * n / ncoils), np.sin(2 * np.pi * n/ncoils), 0)
    def coil_sensitivity(center):
        return np.exp(-((xv - center[0])**2 + (yv - center[1])**2 + (zv - center[2])**2) / (2 * sigma**2))

    # make sensitivity maps
    smaps = [coil_sensitivity(coil_center(n)) for n in range(0,ncoils)]

    return torch.tensor(smaps,dtype=torch.complex128)

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

