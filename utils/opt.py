import torch

def pwritr(A_fwd, A_adj, x_tmp, niter=100, tol=1e-2):

    # generate random input b_k
    b_k = torch.randn_like(x_tmp)
    nrm = 1

    # iterate
    for _ in range(niter):

        # compute A^*Ab_k
        AaAb_k = A_adj(A_fwd(b_k))

        # compute the norm
        nrm_old = nrm
        nrm = torch.norm(AaAb_k).item()

        # check for convergence
        if abs(nrm - nrm_old) / nrm_old < tol:
            break

        # update b_k
        b_k = AaAb_k / nrm

    L = torch.real(torch.inner(b_k.view(-1),AaAb_k.view(-1))/torch.inner(b_k.view(-1),b_k.view(-1)))

    return L