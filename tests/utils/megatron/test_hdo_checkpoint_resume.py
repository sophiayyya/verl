# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU regression for loading HDO after verl offloads optimizer main shards.

Use the real HDO, verl offload helpers, and engine checkpoint entry point.
The small checkpoint manager substitutes local tensor IO for distributed shards.
"""

# ruff: noqa: E402

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")
pytest.importorskip("transformer_engine.pytorch")

from megatron.core import parallel_state
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

from verl.utils.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer
from verl.workers.engine.megatron.transformer_impl import MegatronEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="HDO resume requires a GPU")


class CPUAdamW(torch.optim.AdamW):
    """Distinct CPU type: HDO dispatches state placement via isinstance."""


@pytest.fixture(scope="module", autouse=True)
def distributed(tmp_path_factory):
    rendezvous = tmp_path_factory.mktemp("hdo_pg") / "init"
    dist.init_process_group("nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def make_optimizer(offload_fraction, overlap):
    # DistributedOptimizer supplies FP32 main shards for BF16 model weights,
    # plus native FP32 parameters such as DSv4's routers.
    params = [torch.linspace(-0.5, 0.5, 64, device="cuda") + i for i in range(2)]
    native_fp32 = torch.linspace(0.1, 0.9, 64, device="cuda")
    hdo = HybridDeviceOptimizer(
        params + [native_fp32],
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=CPUAdamW,
        # Use per-parameter AdamW state on both devices. TE FusedAdam's group
        # step counters need additional DistributedOptimizer checkpoint IO,
        # which this focused HDO/engine regression does not instantiate.
        gpu_optimizer_cls=torch.optim.AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=0.001,
        weight_decay=0.01,
    )
    outer = SimpleNamespace(optimizer=hdo, shard_fp32_from_float16_groups=[params])
    return outer, params + [native_fp32]


def step(outer, params, index):
    for i, param in enumerate(params):
        param.grad = torch.linspace(-0.2, 0.3, param.numel(), device="cuda") + 0.01 * (i + index)
    outer.optimizer.step()
    outer.optimizer.zero_grad()
    torch.cuda.synchronize()


def save(path, outer, params):
    torch.save({"params": [p.detach().cpu() for p in params], "optimizer": outer.optimizer.state_dict()}, path)


class TensorCheckpointManager:
    def __init__(self, outer, params):
        self.outer = outer
        self.params = params

    def load_checkpoint(self, local_path, **kwargs):
        checkpoint = torch.load(local_path, map_location="cpu", weights_only=False)
        for param, saved in zip(self.params, checkpoint["params"], strict=True):
            param.copy_(saved)
        self.outer.optimizer.load_state_dict(checkpoint["optimizer"])


def engine(outer, params, optimizer_offload=True):
    # No model buffers are needed to exercise the independent optimizer shards.
    return SimpleNamespace(
        _is_offload_param=False,
        _is_offload_optimizer=optimizer_offload,
        optimizer=outer,
        checkpoint_mananager=TensorCheckpointManager(outer, params),
    )


def assert_matches(outer, params, expected):
    expected_params, expected_state = expected
    for param, value in zip(params, expected_params, strict=True):
        torch.testing.assert_close(param.cpu(), value, rtol=0, atol=0)
    for actual, saved in zip(outer.optimizer.state_dict()["state"].values(), expected_state.values(), strict=True):
        for key in ("exp_avg", "exp_avg_sq", "master_param"):
            torch.testing.assert_close(actual[key].cpu(), saved[key].cpu(), rtol=0, atol=0)
        assert actual["step"] == saved["step"]


def test_old_load_order_reproduces_missing_cpu_copy(tmp_path):
    outer, params = make_optimizer(1.0, True)
    step(outer, params, 1)
    save(tmp_path / "checkpoint.pt", outer, params)
    offload_megatron_optimizer(outer)
    # Previous engine behavior: load checkpoint before restoring main shards.
    TensorCheckpointManager(outer, params).load_checkpoint(tmp_path / "checkpoint.pt")
    load_megatron_optimizer(outer)
    with pytest.raises(KeyError):
        step(outer, params, 2)


@pytest.mark.parametrize("offload_fraction,overlap", [(1.0, True), (0.5, True), (1.0, False)])
@pytest.mark.parametrize("optimizer_offload", [True, False])
def test_engine_resume_matches_uninterrupted_training(tmp_path, offload_fraction, overlap, optimizer_offload):
    reference, ref_params = make_optimizer(offload_fraction, overlap)
    step(reference, ref_params, 1)
    step(reference, ref_params, 2)
    checkpoint_path = tmp_path / "step2.pt"
    save(checkpoint_path, reference, ref_params)
    expected_steps = []
    for index in (3, 4):
        step(reference, ref_params, index)
        expected_steps.append(
            ([p.detach().cpu().clone() for p in ref_params], deepcopy(reference.optimizer.state_dict()["state"]))
        )

    # Resume twice, including a save after the first restored step.
    for index, expected in zip((3, 4), expected_steps, strict=True):
        restored, params = make_optimizer(offload_fraction, overlap)
        if optimizer_offload:
            offload_megatron_optimizer(restored)
        MegatronEngine.load_checkpoint(engine(restored, params, optimizer_offload), str(checkpoint_path))
        load_megatron_optimizer(restored)
        step(restored, params, index)
        assert_matches(restored, params, expected)
        checkpoint_path = tmp_path / f"step{index}.pt"
        save(checkpoint_path, restored, params)
