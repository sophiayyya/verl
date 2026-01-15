#!/usr/bin/env bash
set -xeuo pipefail

# need cuda12.9 or higher
# use docker://verlai/verl:dev.vllm_nightly-243ed7d32e94f00a9a32fbbc51be932f6277a55d or self build


# this env var is required for TE fp8 training
# if you are running multiple nodes, you need to set this env var in RUNTIME_ENV
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export PYTHONPATH=/lustre/fsw/general_sa/sopyang/rl/verl:/lustre/fsw/general_sa/sopyang/rl/Megatron-LM
################################################### quick config ###################################################


rollout_mode="async"
rollout_name="vllm" # sglang or vllm
return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi
dtype="bfloat16" # ["bfloat16", "float16"]

project_name='DAPO_fp8_qwen3_8B'
exp_name='fp8train_fp8rollout_async'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Rollout Correction parameters for FP8 rollout
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null
rollout_rs_threshold_lower=null
rollout_token_veto_threshold=null
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

max_prompt_length=$((1024))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32


PROFILE=${PROFILE:-0}
if [ $PROFILE -eq 1 ]; then
    echo "Profiling is enabled"
    PROFILE_STEPS="[2]"

    DISCRETE=True
    PROFILE_RANKS_ALL=False
    PROFILE_RANKS=[0,1,2,3,4,5,6,7]
else
    PROFILE_STEPS=null
    TOTAL_TRAINING_STEPS=null
    DISCRETE=True
    PROFILE_RANKS_ALL=False
    PROFILE_RANKS=null
fi


# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-4}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/lustre/fsw/general_sa/sopyang/"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/hf_models/Qwen3-8B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpt/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/dapo_data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/datasets/dapo_data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
offload=true
gen_tp=1
gen_pp=1
train_tp=2
train_pp=1
CP=1
ETP=${ETP:-1}


# Set Flash-RL environment variables
# export VERL_LOGGING_LEVEL=DEBUG
# export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export VLLM_USE_DEEP_GEMM=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_CUDA_ARCH_LIST="9.0"  
################################################### start of config ###################################################

FP8=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="e4m3" # e4m3 or hybrid
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.megatron.override_ddp_config.fp8_param_gather=True
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
)

DATA=(
    #  dddd
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=$return_raw_chat
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.filter_overlong_prompts=True
)

REWARD_MODEL=(
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
    reward_model.reward_manager=dapo
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.use_mbridge=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.dtype=${dtype}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 32 ))
    actor_rollout_ref.rollout.max_num_seqs=256 
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    +actor_rollout_ref.rollout.quantization=fp8
)

TRAINER=(
    trainer.logger=['console','wandb']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.save_freq=20
    trainer.total_training_steps=500 
    trainer.total_epochs=10
    trainer.max_actor_ckpt_to_keep=5 
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=10
    +trainer.dump_high_diff_tokens=False  
    +trainer.dump_high_diff_dir="${CKPTS_DIR}/8B_logprob_diff_dumps"
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    # actor_rollout_ref.model.trust_remote_code=True
    # data.trust_remote_code=True
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    algorithm.rollout_correction.rollout_rs_threshold_lower=${rollout_rs_threshold_lower} \
    algorithm.rollout_correction.rollout_token_veto_threshold=${rollout_token_veto_threshold} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
)
RECOMPUTE=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ADVANCE=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
)
if [ $PROFILE -eq 1 ]; then
    NSYS=(
        trainer.profile_steps=$PROFILE_STEPS \
        actor_rollout_ref.profiler.ranks=$PROFILE_RANKS \
        actor_rollout_ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
        actor_rollout_ref.profiler.discrete=$DISCRETE \
    )
fi
################################################### start script ###################################################
wandb login WANDB_KEY
# python3 -m recipe.dapo.main_dapo \
DATETIME=$(date +%Y%m%d%H%M%S)
ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REWARD_MODEL[@]}" \
    "${PERF_OPT[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${RECOMPUTE[@]}" \
    "${FP8[@]}" \
    # "${ADVANCE[@]}" 
    # > ${CKPTS_DIR}/train_fp8train_bf16rollout_async_${DATETIME}.log 2>&1 &

    # ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     -- python3 -m verl.trainer.main_ppo \
#     --config-path=config \
#     --config-name='ppo_megatron_trainer.yaml' \