from time import process_time

import torch as T


def timeit(func, *args, **kwargs):
    for _ in range(10):
        _ = func(*args, **kwargs)
    start = process_time()
    for _ in range(100):
        func(*args, **kwargs)
    return process_time() - start


def precompute_trig_tables(x: T.Tensor, theta: float = 10000.0) -> T.Tensor:
    _B, S, D = x.shape
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = 1.0 / (theta ** (T.arange(0, D, 2).float() / D))
    freqs = T.outer(t, freqs)
    emb = T.repeat_interleave(freqs, 2, dim=-1)
    cos = emb.cos().to(x.dtype).unsqueeze(0)
    sin = emb.sin().to(x.dtype).unsqueeze(0)
    return cos, sin


def precompute_freqs_cis(x=T.Tensor, theta: float = 10000.0):
    _B, S, D = x.shape
    t = T.arange(S, device=x.device, dtype=T.float32)
    freqs = 1.0 / (theta ** (T.arange(0, D, 2).float() / D))
    freqs = T.outer(t, freqs)
    return T.polar(T.ones_like(freqs), freqs)


def rotate_half(x: T.Tensor) -> T.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return T.cat((-x2, x1), dim=-1)


def rope1(x: T.Tensor, cos: T.Tensor, sin: T.Tensor) -> T.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def rope(x: T.Tensor, freqs_cis: T.Tensor) -> T.Tensor:
    B, S, D = x.shape
    q = T.view_as_complex(x.float().reshape(B, S, D // 2, 2))
    q = T.view_as_real(q * freqs_cis)
    return q.view_as(x).type_as(x)


x = T.randn(128, 64, 256, requires_grad=True)
a = T.empty(0)
cos, sin = precompute_trig_tables(x)
r1 = rope1(x, cos, sin)

freqs_cis = precompute_freqs_cis(x)
# r2 = rope2(x, freqs_cis)

# print(T.max(T.abs(r1 - r2)))
# print("Time rope1:", timeit(rope1, x, cos, sin))
# print("Time rope2:", timeit(rope2, x, freqs_cis))
