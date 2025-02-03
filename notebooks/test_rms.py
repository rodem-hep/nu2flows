from time import process_time

import torch as T
from torch import nn
from torch.nn import functional as F


def timeit(func, *args, **kwargs):
    for _ in range(10):
        _ = func(*args, **kwargs)
    start = process_time()
    for _ in range(1000):
        _ = func(*args, **kwargs)
    return process_time() - start


class RMSNorm1(nn.Module):
    """Root Mean Square Normalisation layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(T.ones(dim))
        self.const = dim ** (-0.5)

    def forward(self, x: T.Tensor) -> T.Tensor:
        norm = T.linalg.norm(x.float(), dim=-1, keepdim=True).to(x.dtype)
        return x * self.scale / (norm * self.const)


class RMSNorm2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(T.ones(dim))

    def forward(self, x: T):
        x_dtype = x.dtype
        x = x.float()
        rrms = T.rsqrt(T.mean(x**2, dim=-1, keepdim=True))
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(T.ones(2, dim))
        self.qk_dim = 2 * dim
        self.const = dim ** (-0.5)

    def forward(self, x: T.Tensor):
        B, S, _ = x.shape
        qk, v = x[..., : self.qk_dim], x[..., self.qk_dim :]
        qk = qk.view(B, S, 2, -1)
        norm = T.linalg.norm(qk.float(), dim=-1, keepdim=True).to(qk.dtype)
        qk = qk * self.scale / (norm * self.const)
        qk = qk.view(B, S, -1)
        return T.cat([qk, v], dim=-1)


def fn1(x, weight, bias):
    return F.layer_norm(x, (x.shape[-1],), None, None, 1e-5) * weight + bias


def fn2(x, weight, bias):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5)


def fn3(x, weight, bias):
    a = F.layer_norm(x, (x.shape[-1],), None, None, 1e-5)
    return T.addcmul(bias, weight, a)


dim = 1024
x = T.randn(256, 100, dim).to("cuda")
w = T.randn(dim).to("cuda")
b = T.randn(dim).to("cuda")

out1 = fn1(x, w, b)
out2 = fn2(x, w, b)
out3 = fn3(x, w, b)
print(T.max(T.abs(out1 - out2)))
print(T.max(T.abs(out1 - out3)))
print(T.max(T.abs(out2 - out3)))

# qk_rms = QKNorm(dim)
# q_rms = RMSNorm1(dim)
# k_rms = RMSNorm1(dim)

# qkv1 = qk_rms(x)

# q, k, v = x.chunk(3, dim=-1)
# q = q_rms(q)
# k = k_rms(k)
# qkv2 = T.cat([q, k, v], dim=-1)

print(f"Dim: {dim}")
print("1", timeit(fn1, x, w, b))
print("2", timeit(fn2, x, w, b))
print("3", timeit(fn3, x, w, b))
print()
