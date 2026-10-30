import torch

class Config:

    vocab_size = 32000
    d_model_Teacher = 712
    d_model_Student = 256
    mx_sq=512
    batch_size=32
    temperature=2
    alpha=0.5

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")