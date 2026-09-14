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

"""GPU regressions for MXFP4 QAT with native Megatron/TE checkpoint IO.

Run with the training stack (ModelOpt 0.44, TE >= 2, Megatron-Core) on one GPU:
    pytest -q tests/utils/modelopt/test_mxfp4_checkpoint.py
"""

# ruff: noqa: E402

import pickle
from contextlib import nullcontext
from importlib.metadata import version

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")
mtq = pytest.importorskip("modelopt.torch.quantization")
te = pytest.importorskip("transformer_engine.pytorch")

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TEColumnParallelLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, Format

from verl.utils.modelopt.checkpoint import preserve_mxfp4_checkpoint_methods
from verl.utils.modelopt.quantize import apply_qat, build_quantize_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron TE checkpoint regression needs a GPU")


@pytest.fixture(scope="module", autouse=True)
def distributed(tmp_path_factory):
    rendezvous = tmp_path_factory.mktemp("mxfp4_pg") / "init"
    dist.init_process_group("nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(1234)
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def enable_te_fp8(monkeypatch):
    import modelopt.torch.quantization.plugins.transformer_engine as modelopt_te

    monkeypatch.setattr(modelopt_te, "_assert_te_fp8_enabled", lambda: None)


def make_model(recipe="blockwise"):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_moe_experts=2,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp8="hybrid" if recipe else None,
        fp8_recipe=recipe or "delayed",
    )
    model = MegatronModule(config)
    model.layer = MegatronModule(config)
    model.layer.mlp = MegatronModule(config)
    model.layer.mlp.experts = TEGroupedMLP(
        num_local_experts=2,
        config=config,
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear,
            linear_fc2=TERowParallelGroupedLinear,
        ),
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    # ModelOpt also converts non-expert linears, even though their quantizers
    # are disabled. Their TE state and ordinary weights must survive too.
    model.layer.dense = TEColumnParallelLinear(
        256,
        256,
        config=config,
        init_method=config.init_method,
        gather_output=False,
        bias=False,
        skip_bias_add=True,
        is_expert=False,
    )
    model.layer.plain = torch.nn.Linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    return model


def train_step(model, optimizer, recipe):
    experts = model.layer.mlp.experts
    for linear in (experts.linear_fc1, experts.linear_fc2, model.layer.dense):
        linear.is_first_microbatch = True
    fp8_recipe = (
        Float8BlockScaling(fp8_format=Format.HYBRID)
        if recipe == "blockwise"
        else DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=4)
    )
    context = te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe) if recipe else nullcontext()
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with context:
        output, _ = experts(inputs, torch.tensor([128, 128]), torch.ones(256, device="cuda"))
        output, _ = model.layer.dense(output)
        output = model.layer.plain(output)
        # Delayed scaling starts with unit scales. A mean over 65,536 values
        # underflows its first FP8 backward; a sum keeps this IO test in range.
        loss = output.float().square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())
    optimizer.step()


def sharded(model):
    return {"model": model.sharded_state_dict(metadata={"dp_cp_group": parallel_state.get_data_parallel_group()})}


def test_modelopt_044_reproduces_fp8_save_error():
    if not version("nvidia-modelopt").startswith("0.44."):
        pytest.skip("Upstream failure is specific to ModelOpt 0.44")
    model = make_model()
    mtq.quantize(model, build_quantize_config("mxfp4_experts"))
    train_step(model, torch.optim.Adam(model.parameters(), lr=1e-3), "blockwise")
    with pytest.raises(pickle.UnpicklingError, match="persistent"):
        sharded(model)


@pytest.mark.parametrize("recipe", ["blockwise", "delayed", None])
def test_mxfp4_save_load_and_continue(tmp_path, recipe):
    model = make_model(recipe)
    original_parameter_names = list(dict(model.named_parameters()))
    original_shard_keys = set(sharded(model)["model"])
    original_heterogeneous = model.config.hetereogenous_dist_checkpoint
    apply_qat(model, "mxfp4_experts")
    assert list(dict(model.named_parameters())) == original_parameter_names
    assert model.config.hetereogenous_dist_checkpoint == original_heterogeneous
    assert set(sharded(model)["model"]) == original_shard_keys
    assert not model.layer.dense.weight_quantizer.is_enabled
    assert not model.layer.plain.weight_quantizer.is_enabled
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(2):
        train_step(model, optimizer, recipe)
        save_dir = tmp_path / f"step_{step}"
        save_dir.mkdir()
        dist_checkpointing.save(sharded(model), str(save_dir))
    expected_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer_path = tmp_path / "optimizer.pt"
    torch.save(optimizer.state_dict(), optimizer_path)

    restored = make_model(recipe)
    apply_qat(restored, "mxfp4_experts")
    loaded = dist_checkpointing.load(sharded(restored), str(save_dir))
    restored.load_state_dict(loaded["model"], strict=True)
    for name, p in restored.named_parameters():
        torch.testing.assert_close(p, expected_weights[name], rtol=0, atol=0)
    for source, target in zip(
        (model.layer.mlp.experts.linear_fc1, model.layer.mlp.experts.linear_fc2, model.layer.dense),
        (restored.layer.mlp.experts.linear_fc1, restored.layer.mlp.experts.linear_fc2, restored.layer.dense),
        strict=True,
    ):
        assert target.weight_quantizer.is_enabled == source.weight_quantizer.is_enabled
        assert not target.input_quantizer.is_enabled
        source_extra = source.get_extra_state()
        target_extra = target.get_extra_state()
        source_state = pickle.loads(source_extra.cpu().numpy().tobytes()) if source_extra.numel() else None
        target_state = pickle.loads(target_extra.cpu().numpy().tobytes()) if target_extra.numel() else None
        if recipe:
            if hasattr(target, "num_gemms"):
                assert target_state["extra_fp8_variables"]["num_gemms"] == 2
            assert type(target_state["recipe"]) is type(source_state["recipe"])
            if recipe == "delayed":
                for name in ("scale_fwd", "scale_bwd", "amax_history_fwd", "amax_history_bwd"):
                    torch.testing.assert_close(target_state[name], source_state[name], rtol=0, atol=0)
        else:
            assert target_state is None
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    train_step(restored, restored_optimizer, recipe)
    assert any(not torch.equal(p, expected_weights[name]) for name, p in restored.named_parameters())


def test_persistent_quantizer_state_is_rejected():
    model = torch.nn.Module()
    with pytest.raises(RuntimeError, match="requires stateless quantizers"):
        with preserve_mxfp4_checkpoint_methods(model):
            model.weight_quantizer = TensorQuantizer()
            model.weight_quantizer.register_buffer("_amax", torch.tensor(1.0))


def test_other_qat_modes_keep_modelopt_checkpoint_methods(monkeypatch):
    model = torch.nn.Linear(32, 32)
    modelopt_method = lambda: "modelopt state"

    def quantize(module, config):
        module.get_extra_state = modelopt_method

    monkeypatch.setattr(mtq, "quantize", quantize)
    apply_qat(model, "w4a16")
    assert model.get_extra_state is modelopt_method
