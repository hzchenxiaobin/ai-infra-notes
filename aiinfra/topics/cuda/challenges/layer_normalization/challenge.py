import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Layer Normalization"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(
        self,
        input: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        output: torch.Tensor,
        N: int,
        D: int,
        eps: float,
    ):
        assert input.shape == output.shape == (N, D)
        assert gamma.shape == beta.shape == (D,)
        assert input.dtype == gamma.dtype == beta.dtype == output.dtype
        assert input.device == gamma.device == beta.device == output.device

        # Per-sample statistics along the feature dimension (dim=1, the last dim)
        mean = torch.mean(input, dim=1)  # Shape: [N]
        variance = torch.var(input, dim=1, unbiased=False)  # Shape: [N]

        # Normalize, scale and shift
        normalized = (input - mean[:, None]) / torch.sqrt(variance[:, None] + eps)
        output.copy_(gamma * normalized + beta)

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "gamma": (ctypes.POINTER(ctypes.c_float), "in"),
            "beta": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "N": (ctypes.c_int, "in"),
            "D": (ctypes.c_int, "in"),
            "eps": (ctypes.c_float, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, D = 3, 2
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype)
        gamma = torch.tensor([1.0, 1.0], device=self.device, dtype=dtype)
        beta = torch.tensor([0.0, 0.0], device=self.device, dtype=dtype)
        output = torch.empty((N, D), device=self.device, dtype=dtype)
        eps = 1e-5
        return {
            "input": input,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": N,
            "D": D,
            "eps": eps,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        tests = []

        # basic_small
        N, D = 3, 2
        tests.append(
            {
                "input": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=self.device, dtype=dtype
                ),
                "gamma": torch.tensor([1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # single_sample
        N, D = 1, 4
        tests.append(
            {
                "input": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=dtype),
                "gamma": torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # all_zeros
        N, D = 4, 3
        tests.append(
            {
                "input": torch.zeros((N, D), device=self.device, dtype=dtype),
                "gamma": torch.ones(D, device=self.device, dtype=dtype),
                "beta": torch.zeros(D, device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # negative_numbers
        N, D = 2, 3
        tests.append(
            {
                "input": torch.tensor(
                    [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], device=self.device, dtype=dtype
                ),
                "gamma": torch.tensor([1.0, 1.0, 1.0], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.0, 0.0, 0.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # different_gamma_beta
        N, D = 2, 2
        tests.append(
            {
                "input": torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=self.device, dtype=dtype),
                "gamma": torch.tensor([2.0, 0.5], device=self.device, dtype=dtype),
                "beta": torch.tensor([1.0, -1.0], device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # large_values
        N, D = 5, 3
        tests.append(
            {
                "input": torch.empty((N, D), device=self.device, dtype=dtype).uniform_(-50.0, 50.0),
                "gamma": torch.empty(D, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(D, device=self.device, dtype=dtype).uniform_(-5.0, 5.0),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # medium_size
        N, D = 64, 32
        tests.append(
            {
                "input": torch.empty((N, D), device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
                "gamma": torch.empty(D, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
                "beta": torch.empty(D, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # single_feature (var == 0, output == beta)
        N, D = 100, 1
        tests.append(
            {
                "input": torch.empty((N, D), device=self.device, dtype=dtype).uniform_(-1.0, 1.0),
                "gamma": torch.tensor([1.5], device=self.device, dtype=dtype),
                "beta": torch.tensor([0.5], device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # high_variance
        N, D = 10, 5
        input_data = torch.empty((N, D), device=self.device, dtype=dtype)
        for n in range(N):
            input_data[n, :] = torch.linspace(
                -100 + n * 10, 100 - n * 10, D, device=self.device, dtype=dtype
            )
        tests.append(
            {
                "input": input_data,
                "gamma": torch.ones(D, device=self.device, dtype=dtype),
                "beta": torch.zeros(D, device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        # large_hidden_dim (transformer-typical)
        N, D = 256, 768
        tests.append(
            {
                "input": torch.empty((N, D), device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
                "gamma": torch.ones(D, device=self.device, dtype=dtype),
                "beta": torch.zeros(D, device=self.device, dtype=dtype),
                "output": torch.empty((N, D), device=self.device, dtype=dtype),
                "N": N,
                "D": D,
                "eps": 1e-5,
            }
        )

        return tests

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        N, D = 5000, 512
        return {
            "input": torch.empty((N, D), device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
            "gamma": torch.empty(D, device=self.device, dtype=dtype).uniform_(0.5, 2.0),
            "beta": torch.empty(D, device=self.device, dtype=dtype).uniform_(-2.0, 2.0),
            "output": torch.empty((N, D), device=self.device, dtype=dtype),
            "N": N,
            "D": D,
            "eps": 1e-5,
        }
