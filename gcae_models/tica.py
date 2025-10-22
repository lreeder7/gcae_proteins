import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def centering(X):
    # minus the mean in batch dimension
    return X - torch.mean(X, 1, keepdim = True)

def whitening(X, debug_mode = False):
    X = centering(X) # apply centering
    cov_X = torch.cov(X, correction= 0)
    if debug_mode:
        print(f'covariance is {cov_X}')
        print(f'outer product mean is {1/X.shape[1]*torch.matmul(X, X.T)}')
        print(f'covariance shape is {cov_X.shape}')
    U, S, Vh = torch.linalg.svd(cov_X, full_matrices=False) 
    cov_X_sqrt_inv = U @ torch.diag(torch.sqrt(1/S)) @ Vh # square root of matrix
    if debug_mode:
        print(f'cov_X_sqrt_inv shape is {cov_X_sqrt_inv.shape}')
        print(f'Sample covariance post postprocessing is {torch.cov(cov_X_sqrt_inv@X, correction = 0)}.')
    
    return cov_X_sqrt_inv@X

def rotate(X, R, debug_mode = False):
    if debug_mode:
        if torch.dist(R.T@R, torch.eye(3)) > 1e-3:
            raise ValueError('R is not orthogonal')
            
    Y = X.view(-1, 3, X.shape[1])
    if debug_mode:
        print(f'size of Y is {Y.shape}')
        print(f'size of R is {R.shape}')

    Y = torch.einsum('ij, kjl -> kil', R, Y)
    if debug_mode:
        print(f'size of Y is {Y.shape}')
        
    X = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
    return X
    
def time_lag_data(Z, tau, debug_mode = False):
#     process Z into a matrix X = [Z_{1}, Z_{2}, ..., Z_{T-tau}]
    T = Z.shape[1] # trajectory length
    X_list = [Z[:, i].view(-1, 1) for i in range(0, T - tau)]
    Y_list = [Z[:, i + tau].view(-1, 1) for i in range(0, T - tau)]

    if debug_mode:
        assert (len(X_list)) == T - tau
        assert torch.dist(Y_list[-1], Z[:, -1].view(-1, 1)) < 1e-5

    
    return torch.hstack(X_list), torch.hstack(Y_list)

def get_linear_TCCA(Z, tau, d, debug_mode = False):
    # get the linear encoder & decoder for TCCA
    Z = whitening(Z)
    X, Y = time_lag_data(Z, tau)
    K_full = Y@X.T@torch.linalg.inv(X@X.T)
    U, S, Vh = torch.linalg.svd(K_full, full_matrices = False)
    D = (Vh.T)[:, 0:d] # decoder matrix
    E = torch.diag(S[0:d])@(U[:, 0:d]).T # encoder matrix
    return D, E
    
def get_linear_TICA(Z, tau, d, debug_mode = False):
    # get the linear encoder & decoder for TICA. This case considers a 
    Z_enlarged = torch.hstack([Z, torch.flip(Z, dims = (1,))]) # data augmentation with reversible trajectory
    D, E = get_linear_TCCA(Z_enlarged, tau, d, debug_mode = False)
    return D, E
    
def random_unitary():
    A = torch.rand([3, 3])
    R, _, _ = torch.svd(A.T + A)
    return R