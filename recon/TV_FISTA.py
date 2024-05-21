import torch
import numpy as np

def tvdeblur(A_fwd, A_adj, b, tvtype='L1', niter=100, lam=0.1, L=1, chat=1):

    # initialize variables
    P = None
    x = A_adj(b)
    res = A_fwd(x) - b
    if tvtype == 'iso':
        cost = [1/2*torch.norm(res)**2 + lam * tvnorm_iso(x)]
    elif tvtype == 'L1':
        cost = [1/2*torch.norm(res)**2 + lam * tvnorm_L1(x)]
    else:
        raise print("error: invalid tvtype")
    x_set = [x]
    Y = x
    t = 1.0

    # loop through FISTA iterations
    for i in range(niter):

        # store old values
        x_old = x
        t_old = t

        # calculate the gradient
        grad = A_adj(A_fwd(Y) - b)
        x = Y - grad/L

        # total variation denoising
        if abs(lam) > 0:
            x, P = tvdenoise(x, P, tvtype=tvtype, niter=niter, lam=lam/L)

        # update step size
        t = (1 + np.sqrt(1 + 4 * t_old**2)) / 2

        # update Y
        Y = x + (t_old-1)/t * (x - x_old)

        # calculate residual
        res = A_fwd(x) - b

        # calculate cost
        if tvtype == 'iso':
            cost.append(1/2*torch.norm(res)**2 + lam * tvnorm_iso(x))
        elif tvtype == 'L1':
            cost.append(1/2*torch.norm(res)**2 + lam * tvnorm_L1(x))
        else:
            raise print("error: invalid tvtype")

        # save the image
        x_set.append(x)

        if chat:
            print("TV_FISTA.tvdeblur(): iter %d, cost = %f" % (i, cost[i]))

    return x, cost, x_set

def tvdenoise(v, P=None, tvtype='L1', niter=100, lam=0.1, tol=1e-5):
    
    # get the shape of the input tensor v
    shape = v.shape
    ndim = len(v.squeeze().shape)

    # initialize variables
    if P is None:
        P = [torch.zeros_like(P_tensor) for P_tensor in L_adj(v)]
    t = 1.0
    R = P
    D = torch.zeros_like(v)

    # loop through FISTA iterations
    for i in range(niter):

        # store old values
        D_old = D
        P_old = P
        t_old = t
       
        # compute gradient of objective function
        D = v - lam * L_fwd(R)
        Q = L_adj(D)

        # gradient descent step
        for d in range(ndim):
            P[d] = R[d] + 1/(4*ndim*lam) * Q[d]

        # proximal operator for TV regularization
        if tvtype == 'iso':
            P = tvprox_iso(P)
        elif tvtype == 'L1':
            P = tvprox_L1(P)
        else:
            raise print("error: invalid tvtype")

        # update step size
        t = (1 + np.sqrt(1 + 4 * t**2)) / 2

        # update R
        for d in range(ndim):
            R[d] = P[d] + (t_old - 1)/t * (P[d] - P_old[d])

        # calculate residual
        res = torch.norm(D - D_old) / torch.norm(D)
        if res < tol:
            break

    # calculate Y
    Y = v - lam * L_fwd(P)

    return Y, P

def tvprox_iso(P):

    P_prox = []

    # proximal mapping divides each P by rsos
    rsos = torch.sqrt(torch.sum([P_tensor**2 for P_tensor in P])) # not yet tested (05/19/2024)
    for P_tensor in P:
        P_temp = P_tensor / torch.maximum(rsos,torch.ones_like(torch.abs(P_tensor)))
        P_prox = P_prox.append(P_temp)
    
    return P_prox

def tvprox_L1(P):

    P_prox = []

    # proximal mapping divides each P by absolute max
    for P_tensor in P:
        P_tensor_abs = torch.abs(P_tensor)
        P_temp = P_tensor / torch.maximum(P_tensor_abs,torch.ones_like(P_tensor_abs))
        P_prox.append(P_temp)
    
    return P_prox

def tvnorm_iso(x):

    # calculate difference matrices along each dimension
    P = L_adj(x)

    # calculate root sum of square differences along each dimension
    tv_norm = sum(torch.sqrt(torch.sum(torch.pow(P_tensor,2))) for P_tensor in P)

    return tv_norm

def tvnorm_L1(x):

    # calculate difference matrices along each dimension
    P = L_adj(x)

    # sum the abs differences along each dimension
    tv_norm = sum(torch.sum(torch.abs(P_tensor)) for P_tensor in P)

    return tv_norm

def L_fwd(P):

    # get size
    sz = list(P[0].shape)
    sz[0] += 1
    ndim = len(P[0].shape)

    # initialize image
    x = torch.zeros(sz, dtype=P[0].dtype, device=P[0].device)

    # loop through dimensions
    for dim in range(ndim):
        # create slices for addition and subtraction
        slices1 = [slice(None)] * ndim
        slices2 = [slice(None)] * ndim
        slices1[dim] = slice(0, sz[dim] - 1)
        slices2[dim] = slice(1, sz[dim])

        # update the tensor x
        x[tuple(slices1)] += P[dim]
        x[tuple(slices2)] -= P[dim]

    return x

def L_adj(x):
    
    # get the shape of the input tensor x
    sz = list(x.shape)
    ndim = len(sz)
    
    # initialize list of difference matrices P
    P = []
    
    # calculate diffs along each dimension
    for dim in range(ndim):
        # create slices for current and previous elements
        slices1 = [slice(None)] * ndim
        slices2 = [slice(None)] * ndim
        slices1[dim] = slice(1, sz[dim])
        slices2[dim] = slice(0, sz[dim] - 1)
        
        # calculate the differences along the current dimension
        P_tensor = x[tuple(slices2)] - x[tuple(slices1)]
        
        # append the diff tensor to the list
        P.append(P_tensor)
    
    return P
