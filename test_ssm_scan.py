"""Correctness checks for the shared dense and Triton SSM paths."""

import sys
import unittest
import copy
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model" / "pure"))
sys.path.insert(0, str(ROOT / "model" / "hybrid"))
from single_ssm import PureSSMLanguageModel, SingleHeadSSMLayer
from ssm_full import HybridLanguageModel
from ssm_scan import dense_reference_scan, selective_scan, triton


def recurrent_reference(u, c, dt, log_a, residual):
    state = torch.zeros_like(u[:, 0], dtype=torch.float32)
    output = []
    for time in range(u.shape[1]):
        decay = torch.exp(dt[:, time].float() * -torch.exp(log_a.float()))
        state = u[:, time].float() + decay * state
        output.append(residual[:, time].float() + c[:, time].float() * state)
    return torch.stack(output, dim=1)


def inputs(device, length, dim, dtype=torch.float32, batch=2):
    torch.manual_seed(17)
    u = torch.randn(batch, length, dim, device=device, dtype=dtype).requires_grad_()
    c = torch.randn(batch, length, dim, device=device, dtype=dtype).requires_grad_()
    dt = F.softplus(torch.randn(batch, length, 1, device=device, dtype=dtype)).requires_grad_()
    log_a = (torch.randn(dim, device=device) * 0.02).requires_grad_()
    residual = torch.randn(batch, length, dim, device=device).requires_grad_()
    return u, c, dt, log_a, residual


class SSMScanTests(unittest.TestCase):
    def compare(self, device, length, dim, dtype, backend, reference):
        values = inputs(device, length, dim, dtype)
        output = selective_scan(*values, backend=backend)
        autocast = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" and dtype == torch.bfloat16 else nullcontext())
        with autocast:
            expected = reference(*values)
        atol, rtol = (4e-2, 4e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)
        self.assertTrue(torch.allclose(output.float(), expected.float(), atol=atol, rtol=rtol),
                        f"output max difference: {(output - expected).abs().max().item()}")
        probe = torch.randn_like(output)
        grads = torch.autograd.grad(output, values, probe, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, values, probe)
        for name, got, want in zip(("u", "c", "dt", "log_a", "residual"), grads, expected_grads):
            self.assertTrue(torch.allclose(got.float(), want.float(), atol=atol, rtol=rtol),
                            f"{name} max difference: {(got - want).abs().max().item()}")

    def test_dense_matches_recurrence_on_cpu(self):
        for length in (1, 7, 65):
            with self.subTest(length=length):
                self.compare("cpu", length, 17, torch.float32, "dense", recurrent_reference)

    def test_small_model_training_steps_on_cpu(self):
        for model in (PureSSMLanguageModel(64, 32, 2),
                      HybridLanguageModel(64, 32, 3, attn_layers=(1,))):
            with self.subTest(model=type(model).__name__):
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                tokens = torch.randint(64, (2, 8))
                logits = model(tokens)
                loss = F.cross_entropy(logits.reshape(-1, 64), tokens.reshape(-1))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                opt.step()

    def test_block_checkpointing_preserves_gradients(self):
        for original in (PureSSMLanguageModel(64, 32, 2),
                         HybridLanguageModel(64, 32, 3, attn_layers=(1,))):
            with self.subTest(model=type(original).__name__):
                checkpointed = copy.deepcopy(original)
                checkpointed.gradient_checkpointing = True
                tokens = torch.randint(64, (2, 8))
                original(tokens).sum().backward()
                checkpointed(tokens).sum().backward()
                for a, b in zip(original.parameters(), checkpointed.parameters()):
                    self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton unavailable")
    def test_triton_matches_dense(self):
        for length, dim, dtype in ((1, 17, torch.float32),
                                   (7, 32, torch.float32),
                                   (65, 17, torch.float32),
                                   (129, 32, torch.bfloat16),
                                   (1025, 17, torch.float32)):
            with self.subTest(length=length, dim=dim, dtype=dtype):
                self.compare("cuda", length, dim, dtype, "triton", dense_reference_scan)

    @unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton unavailable")
    def test_triton_training_shape(self):
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 12 * 2**30:
            self.skipTest("Dense 4x512x512 reference needs at least 12 GiB free")
        values = inputs("cuda", 512, 512, torch.bfloat16, batch=4)
        output = selective_scan(*values, backend="triton")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            expected = dense_reference_scan(*values)
        self.assertTrue(torch.allclose(output.float(), expected.float(), atol=0.08, rtol=0.05))
        probe = torch.randn_like(output)
        grads = torch.autograd.grad(output, values, probe, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, values, probe)
        for name, got, want in zip(("u", "c", "dt", "log_a", "residual"), grads, expected_grads):
            difference = got.float() - want.float()
            rms_relative = (difference.square().mean().sqrt() /
                            want.float().square().mean().sqrt().clamp_min(1e-6))
            max_difference = difference.abs().max()
            max_allowed = 0.01 + 0.01 * want.float().abs().max()
            self.assertLess(rms_relative.item(), 0.01, name)
            self.assertLess(max_difference.item(), max_allowed.item(), name)

    @unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton unavailable")
    def test_small_model_training_steps_on_cuda(self):
        for model in (PureSSMLanguageModel(64, 32, 2),
                      HybridLanguageModel(64, 32, 3, attn_layers=(1,))):
            model = model.cuda()
            for layer in model.modules():
                if hasattr(layer, "scan_backend"):
                    layer.scan_backend = "triton"
            with self.subTest(model=type(model).__name__):
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
                tokens = torch.randint(64, (2, 8), device="cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(tokens)
                    loss = F.cross_entropy(logits.reshape(-1, 64), tokens.reshape(-1))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))
                opt.step()

    @unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton unavailable")
    def test_compiled_triton_matches_eager(self):
        torch.manual_seed(7)
        eager = SingleHeadSSMLayer(32).cuda()
        eager.scan_backend = "triton"
        compiled = torch.compile(copy.deepcopy(eager), mode="default")
        x = torch.randn(2, 65, 32, device="cuda")
        probe = torch.randn_like(x)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            eager_output = eager(x)
            compiled_output = compiled(x)
        self.assertTrue(torch.allclose(eager_output, compiled_output, atol=0.01, rtol=0.01))
        (eager_output * probe).sum().backward()
        (compiled_output * probe).sum().backward()
        for (name, eager_param), (_, compiled_param) in zip(
                eager.named_parameters(), compiled._orig_mod.named_parameters()):
            difference = eager_param.grad - compiled_param.grad
            relative_rms = (difference.square().mean().sqrt() /
                            eager_param.grad.square().mean().sqrt().clamp_min(1e-6))
            self.assertLess(relative_rms.item(), 0.01, name)


if __name__ == "__main__":
    unittest.main()
