import torch

def pwritr(A_fwd, A_adj, x_tmp, niter=100, tol=1e-2):

    # generate random input b_k
    b_k = torch.randn_like(x_tmp)

    # iterate
    for _ in range(niter):

        # compute Ab_k
        Ab_k = A_fwd(b_k)

        # compute A^*Ab_k
        AaAb_k = A_adj(Ab_k)

        # compute the norm
        nrm = torch.norm(AaAb_k).item()

        # update b_k
        b_k_old = b_k.clone()
        nrm_old = torch.norm(b_k_old).item()
        b_k = AaAb_k / nrm

        # check for convergence
        if (nrm - nrm_old) / nrm_old < tol:
            break

    L = torch.real(torch.dot(b_k.view(-1),AaAb_k.view(-1))/torch.dot(b_k.view(-1),b_k.view(-1)))

    return L