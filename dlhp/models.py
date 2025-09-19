import math 
import time
from typing import Optional,Tuple,List,Dict

import torch
import torch.nn as nn 
import torch.nn.functional as F

def save_matrix_inverse(A:torch.Tensor,eps:float=1e-6)->torch.Tensor:
    I=torch.eye(A.shape[-1],device=A.device,dtype=A.dtype)
    try:
        return torch.linalg.inv(A)
    except Exception:
        return torch.linalg.inv(A+eps*I)

def block_diag_from_diagpair(real:torch.Tensor,imag:torch.Tensor)->torch.Tensor:
    '''
    Input:real,imag:(P,)
    Output:(2P,2P)
    '''
    P=real.shape[0]
    M=torch.zeros(2*P,2*P,device=real.device,dtype=real.dtype)
    for i in range(P):
        a=real[i]
        b=imag[i]
        M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[a, -b], [b, a]], device=real.device, dtype=real.dtype)
    return  M

def complex_pairs_to_real_blockdiag(real:torch.Tensor,imag:torch.Tensor)->torch.Tensor:
    return block_diag_from_diagpair(real,imag)

class LinearDynamicsExact(nn.Module):
    def __init__(self,P:int,H:int,K:int,diag_param:bool=True,input_dependent:bool=True):
        super().__init__(*args, **kwargs)