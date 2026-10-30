import torch
import torch.nn as nn
import triton 
import triton.language as tl
from .matmul import GEMM

@triton.jit
def fwdkrnl(block_x,block_y,stride_x,stride_y,in_ptr,out_ptr):
    pid_x=tl.program_id(0)
    pid_y=tl.program_id(1)
    
    ofs_x=pid_x*block_x+tl.arange(0,block_x)
    ofs_y=pid_y*block_y+tl.arange(0,block_y)
    
    ofs=ofs_x[None,:]*stride_x+ofs_y[:,None]*stride_y
    cordinate_u=in_ptr+ofs
    cordinate_g=out_ptr+ofs
    U=tl.load(cordinate_u)
    G=tl.load(cordinate_g)
    swish=U*tl.sigmoid(U)

    swiGLU=swish*G
    tl.store(out_ptr+ofs,swiGLU)
@triton.jit
def bwdkrnl(pid_x,pid_y,block_x,block_y,stride_x,stride_y,in_ptr,out_ptr):
    pid_x=tl.program_id(0)
    pid_y=tl.program_id(1)

    off_x=pid_x*block_x+tl.arange(0,block_x)
    off_y=pid_y*block_y+tl.arange(0,block_y)

    off=off_x[None,:]*stride_x+off_y[:,None]*stride_y

    dSWG=tl.load(in_ptr+off)



class swiGLU():
    @staticmethod
    def forwad():
        fwdkrnl[]()

    @staticmethod
    def backward():
        bwdkrnl[]()
