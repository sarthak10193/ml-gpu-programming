import torch
import torch.nn as nn
from typing import Tuple
import cuda_machinary


class LinearOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
        assert input.is_cuda, "Input must be a CUDA tensor"
        assert input.dim() == 2, f"Expected 2D input tensor, got {input.dim()}D"
            
        output = cuda_machinary.linear_forward(input, scale, bias)
        ctx.save_for_backward(input)
        ctx.scale = scale
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        input, = ctx.saved_tensors
        grad_input = grad_output * ctx.scale
        return grad_input, None, None


class Linear(nn.Module):
    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.scale = scale
        self.bias = bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearOp.apply(x, self.scale, self.bias)


def linear_op(input: torch.Tensor, scale: float = 1.0, bias: float = 0.0) -> torch.Tensor:
    return LinearOp.apply(input, scale, bias)