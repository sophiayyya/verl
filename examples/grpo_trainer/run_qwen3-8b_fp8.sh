set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
wandb login WANDB_KEY

gsm8k_train_path=/lustre/fsw/general_sa/sopyang/datasets/gsm8k/train.parquet
gsm8k_test_path=/lustre/fsw/general_sa/sopyang/datasets/gsm8k/test.parquet
math_train_path=/lustre/fsw/general_sa/sopyang/datasets/math/train.parquet
math_test_path=/lustre/fsw/general_sa/sopyang/datasets/math/test.parquet
OUTPUT_DIR=/lustre/fsw/general_sa/sopyang/code/verl_logs/nsight_files
train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# Nsight profiling configuration
# PROFILE_STEPS="[2,3,4]" # or [] or null
# PROFILE_RANKS_ALL=True # or True
# DISCRETE=True  # or True

PROFILE=${PROFILE:-0}
if [ $PROFILE -eq 1 ]; then
    echo "Profiling is enabled"
    PROFILE_STEPS="[2,6,10]"
    TOTAL_TRAINING_STEPS=2
    DISCRETE=True
    PROFILE_RANKS_ALL=False
    PROFILE_RANKS=[0,4,8]
else
    PROFILE_STEPS=null
    TOTAL_TRAINING_STEPS=null
    DISCRETE=True
    PROFILE_RANKS_ALL=False
    PROFILE_RANKS=null
fi

TOTAL_TRAINING_STEPS=2


use_dynamic_bsz=True
max_prompt_length=2048
max_response_length=8192
offload=True


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=/lustre/fsw/general_sa/sopyang/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) )) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 10)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 10)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.enforce_eager=True  \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_gsm8k_math_exp' \
    trainer.experiment_name='i512o1024_bs512n16_tp4pp2_lbmbs64_cudagraph_2xppomaxtoken' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    2>&1 | tee /lustre/fsw/general_sa/sopyang/code/verl_logs/i512o1024_bs512n16_tp2pp1_lbmbs64_gentp1_nocudagraph_fp8.log 2>&1 &

    # trainer.profile_steps=$PROFILE_STEPS \
    # trainer.total_epochs=1 \
    # actor_rollout_ref.profiler.ranks=$PROFILE_RANKS \
    # actor_rollout_ref.profiler.all_ranks=$PROFILE_RANKS_ALL \
    # actor_rollout_ref.profiler.discrete=$DISCRETE \

if [ $PROFILE -eq 1 ]; then
    sleep 60s
    echo "Copy nsys-rep to ${OUTPUT_DIR} ..."
    cp /tmp/ray/session_latest/logs/nsight/*.nsys-rep ${OUTPUT_DIR}/
fi
