import torch
import torch.nn as nn
import triton 
import triton.language as tl

from transf.krnl.swiglu import bwdkrnl, fwdkrnl

@triton.jit
def fwdkrnl(block_x,block_y,stride_x,stride_y,in_ptr,out_ptr):
    pid_x=tl.program_id(0)
    pid_y=tl.program_id(1)
    offs_x=pid_x*block_x+tl.arange(0,block_x)
    offs_y=pid_y*block_y+tl.arange(0,block_y)
    offs=offs_x[None,:]*stride_x+offs_y[:,None]*stride_y
    
    x=tl.load(in_ptr+offs)
    rms=tl.sum(x*x,axis=1)
    gamma=tl.load(in_ptr+offs)
    rmsNorm=x*gamma/tl.sqrt(rms[:,None]+1e-8)
    tl.store(out_ptr+offs,rmsNorm)
@triton.jit
def bwdkrnl(block_x,block_y,stride_x,stride_y,in_ptr,out_ptr):
    pass

class rmsNorm():
    @staticmethod()
    def forward():
        fwdkrnl[]()
        pass
    @staticmethod()
    def backward():
        bwdkrnl[]()

