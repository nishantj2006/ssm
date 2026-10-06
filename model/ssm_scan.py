"""Shared selective SSM scan for the pure and hybrid language models.

The dense implementation is retained as a CPU fallback and a benchmark baseline.
Only the scan is custom: layer normalization and learned projections stay in PyTorch.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def dense_reference_scan(u, c, dt, log_a, residual):
    """The original quadratic implementation, including its rounding behavior."""
    _, length, _ = u.shape
    decay_log = dt * -torch.exp(log_a)
    prefix = torch.cumsum(decay_log, dim=1)
    log_matrix = prefix.unsqueeze(2) - prefix.unsqueeze(1)
    indices = torch.arange(length, device=u.device)
    mask = indices[:, None] >= indices[None, :]
    log_matrix = log_matrix.masked_fill(~mask[None, :, :, None], float("-inf"))
    matrix = torch.exp(torch.clamp(log_matrix, max=20.0))
    state = torch.einsum("btjd,bjd->btd", matrix, u)
    return residual + state * c


if triton is not None:
    @triton.jit
    def _compose(a_left, u_left, a_right, u_right):
        return a_right * a_left, u_right + a_right * u_left


    @triton.jit
    def _scan_forward(U, C, DT, LOG_A, RESIDUAL, STATES, OUTPUT,
                      T: tl.constexpr, D: tl.constexpr,
                      CHUNK: tl.constexpr, CHANNELS: tl.constexpr):
        batch = tl.program_id(0)
        channel = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
        row = tl.arange(0, CHUNK)
        good_channel = channel < D
        a_cont = -tl.exp(tl.load(LOG_A + channel, good_channel, other=0).to(tl.float32))
        carry = tl.full((CHANNELS,), 0, tl.float32)

        for chunk in range(tl.cdiv(T, CHUNK)):
            time = chunk * CHUNK + row
            good_time = time < T
            good = good_time[:, None] & good_channel[None, :]
            offset = (batch * T + time[:, None]) * D + channel[None, :]
            delta = tl.load(DT + batch * T + time, good_time, other=0).to(tl.float32)
            u = tl.load(U + offset, good, other=0).to(tl.float32)
            a = tl.exp(delta[:, None] * a_cont[None, :])
            a = tl.where(good_time[:, None], a, 1.0)
            prefix_a, prefix_u = tl.associative_scan((a, u), axis=0, combine_fn=_compose)
            state = prefix_u + prefix_a * carry[None, :]
            carry = tl.sum(tl.where(row[:, None] == CHUNK - 1, state, 0.0), axis=0)
            gate = tl.load(C + offset, good, other=0).to(tl.float32)
            residual = tl.load(RESIDUAL + offset, good, other=0).to(tl.float32)
            tl.store(STATES + offset, state, good)
            tl.store(OUTPUT + offset, residual + gate * state, good)


    @triton.jit
    def _scan_backward(C, DT, LOG_A, STATES, GRAD_OUTPUT,
                       GRAD_U, GRAD_C, GRAD_DT_ELEMENT, GRAD_LOG_A_ELEMENT,
                       T: tl.constexpr, D: tl.constexpr,
                       CHUNK: tl.constexpr, CHANNELS: tl.constexpr):
        batch = tl.program_id(0)
        channel = tl.program_id(1) * CHANNELS + tl.arange(0, CHANNELS)
        row = tl.arange(0, CHUNK)
        good_channel = channel < D
        a_cont = -tl.exp(tl.load(LOG_A + channel, good_channel, other=0).to(tl.float32))
        carry = tl.full((CHANNELS,), 0, tl.float32)

        for chunk in range(tl.cdiv(T, CHUNK)):
            time = T - 1 - chunk * CHUNK - row
            good_time = time >= 0
            good = good_time[:, None] & good_channel[None, :]
            offset = (batch * T + time[:, None]) * D + channel[None, :]
            next_time = time + 1
            next_good_time = (next_time >= 0) & (next_time < T)
            next_delta = tl.load(DT + batch * T + next_time, next_good_time, other=0).to(tl.float32)
            next_a = tl.exp(next_delta[:, None] * a_cont[None, :])
            next_a = tl.where(next_good_time[:, None], next_a, 0.0)
            gate = tl.load(C + offset, good, other=0).to(tl.float32)
            grad = tl.load(GRAD_OUTPUT + offset, good, other=0).to(tl.float32)
            source = grad * gate
            prefix_a, prefix_u = tl.associative_scan((next_a, source), axis=0, combine_fn=_compose)
            grad_state = prefix_u + prefix_a * carry[None, :]
            carry = tl.sum(tl.where(row[:, None] == CHUNK - 1, grad_state, 0.0), axis=0)

            state = tl.load(STATES + offset, good, other=0).to(tl.float32)
            prev_time = time - 1
            prev_good = (prev_time >= 0)[:, None] & good_channel[None, :]
            prev_offset = (batch * T + prev_time[:, None]) * D + channel[None, :]
            prev_state = tl.load(STATES + prev_offset, prev_good, other=0).to(tl.float32)
            delta = tl.load(DT + batch * T + time, good_time, other=0).to(tl.float32)
            a = tl.exp(delta[:, None] * a_cont[None, :])
            grad_decay = grad_state * prev_state * a * a_cont[None, :]
            tl.store(GRAD_U + offset, grad_state, good)
            tl.store(GRAD_C + offset, grad * state, good)
            tl.store(GRAD_DT_ELEMENT + offset, grad_decay, good)
            tl.store(GRAD_LOG_A_ELEMENT + offset, grad_decay * delta[:, None], good)


class _TritonScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, c, dt, log_a, residual):
        batch, length, dim = u.shape
        u, c, dt, residual = (v.contiguous() for v in (u, c, dt, residual))
        log_a = log_a.contiguous()
        state = torch.empty((batch, length, dim), device=u.device, dtype=torch.float32)
        output = torch.empty_like(residual)
        _scan_forward[(batch, triton.cdiv(dim, 16))](
            u, c, dt, log_a, residual, state, output,
            length, dim, 64, 16, num_warps=4,
        )
        ctx.u_dtype = u.dtype
        ctx.save_for_backward(c, dt, log_a, state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        c, dt, log_a, state = ctx.saved_tensors
        batch, length, dim = state.shape
        grad_output = grad_output.contiguous()
        grad_u = torch.empty_like(state)
        grad_c = torch.empty_like(state)
        grad_dt_element = torch.empty_like(state)
        grad_log_a_element = torch.empty_like(state)
        _scan_backward[(batch, triton.cdiv(dim, 16))](
            c, dt, log_a, state, grad_output,
            grad_u, grad_c, grad_dt_element, grad_log_a_element,
            length, dim, 64, 16, num_warps=4,
        )
        grad_dt = grad_dt_element.sum(dim=-1, keepdim=True).to(dt.dtype)
        grad_log_a = grad_log_a_element.sum(dim=(0, 1)).to(log_a.dtype)
        return grad_u.to(ctx.u_dtype), grad_c.to(c.dtype), grad_dt, grad_log_a, grad_output


@torch.compiler.disable
def _run_triton_scan(u, c, dt, log_a, residual):
    # PyTorch 2.8 cannot safely trace these direct Triton launches with Triton 3.5.
    # Keep the kernels eager while torch.compile optimizes the surrounding model.
    return _TritonScan.apply(u, c, dt, log_a, residual)


def selective_scan(u, c, dt, log_a, residual, backend="auto"):
    """Compute the gated scan, selecting Triton on CUDA when available.

    backend='dense' runs the previous implementation for comparison.
    backend='triton' requires CUDA and a Triton installation.
    """
    if backend not in ("auto", "dense", "triton"):
        raise ValueError(f"Unknown SSM backend: {backend}")
    if backend == "dense" or (backend == "auto" and (triton is None or not u.is_cuda)):
        return dense_reference_scan(u, c, dt, log_a, residual)
    if triton is None or not u.is_cuda:
        raise RuntimeError("The Triton SSM backend requires Triton and a CUDA tensor")
    return _run_triton_scan(u, c, dt, log_a, residual)
