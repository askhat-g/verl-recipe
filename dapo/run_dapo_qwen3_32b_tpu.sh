#!/usr/bin/env bash
# DAPO | Qwen3-32B | DAPO-Math-17k / AIME-2024 | TorchTitan Training (SimpleFSDP + Ulysses CP, DEFER_NEVER) & vLLM Rollout (8x TP=4 Replicas, Raiden) | TPU v6e
# V1 PPOTrainer (Separate Async Overlap), Raiden P2P weight sync.

set -xeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set +x
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
    set -x
fi
if [[ "${WANDB_BASE_URL:-}" == *"forge.coreweave.com"* ]]; then
    unset WANDB_BASE_URL
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl-torchtpu-grpo-test}"
export WANDB_MODE="${WANDB_MODE:-online}"

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

# JAX/XLA Launch Barrier & Single-Bucket Compilation Configuration
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-2048}"

# PyTorch Dynamo / Compile Logging and Recompile Limits
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
exp_name="${EXPERIMENT_NAME:-DAPO-Qwen3-32B-TPU}"

# DAPO Algorithm Parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Sequence length & DAPO Overlong Reward Shaping
# Capped to 2048 total tokens (512 prompt + 1536 response) so 32B FSDP32 fits in 31.24 GiB/chip TPU v6e HBM
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1536}
enable_overlong_buffer=True
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-256}
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

# DAPO Dynamic Sampling (filter_groups) & Batching
enable_filter_groups=${ENABLE_FILTER_GROUPS:-True}
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-$((train_prompt_bsz * 3))}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-8}
parameter_sync_step=${PARAMETER_SYNC_STEP:-$((train_prompt_bsz / train_prompt_mini_bsz))}

# Paths (prefer /dev/shm or /tmp local cache when staged to avoid multi-worker gcsfuse contention)
RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/jialei"}
DEFAULT_MODEL_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-32B"
if [[ -d "/dev/shm/Qwen3-32B" ]]; then
    DEFAULT_MODEL_PATH="/dev/shm/Qwen3-32B"
elif [[ -d "/tmp/verl-model-cache/Qwen3-32B" ]]; then
    DEFAULT_MODEL_PATH="/tmp/verl-model-cache/Qwen3-32B"
fi
MODEL_PATH=${MODEL_PATH:-"${DEFAULT_MODEL_PATH}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

DEFAULT_TRAIN_FILE="${RAY_DATA_HOME}/data/dapo_math/dapo-math-17k.parquet"
if [[ -f "/tmp/data/dapo-math-17k.parquet" ]]; then
    DEFAULT_TRAIN_FILE="/tmp/data/dapo-math-17k.parquet"
fi
TRAIN_FILE=${TRAIN_FILE:-"${DEFAULT_TRAIN_FILE}"}

DEFAULT_TEST_FILE="${RAY_DATA_HOME}/data/aime_2024/aime-2024.parquet"
if [[ -f "/tmp/data/aime-2024.parquet" ]]; then
    DEFAULT_TEST_FILE="/tmp/data/aime-2024.parquet"
fi
TEST_FILE=${TEST_FILE:-"${DEFAULT_TEST_FILE}"}

# Rollout Sampling Parameters
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# TPU Slice Topology (default: 32 trainer chips [8 hosts x 4 chips] + 32 rollout chips [8 hosts x 4 chips -> 8x TP=4 vLLM replicas])
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
ROLLOUT_TP="${ROLLOUT_TP:-4}"
ROLLOUT_DP="${ROLLOUT_DP:-$((TOTAL_ROLLOUT_CHIPS / ROLLOUT_TP))}"
ROLLOUT_SLICES="${ROLLOUT_SLICES:-}"
TRAINER_SLICES="${TRAINER_SLICES:-}"
TPU_SLICE_ARGS=()
if [[ -n "${ROLLOUT_SLICES}" ]]; then
    TPU_SLICE_ARGS+=("actor_rollout_ref.rollout.tpu_slices=[${ROLLOUT_SLICES}]")
fi
if [[ -n "${TRAINER_SLICES}" ]]; then
    TPU_SLICE_ARGS+=("+trainer.tpu_slices=[${TRAINER_SLICES}]")
fi

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-${ULYSSES_CP:-4}}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

# Performance & TPU TorchTitan Parameters
sp_size=${CONTEXT_PARALLEL_SIZE}
use_dynamic_bsz=False
max_model_len=$((max_prompt_length + max_response_length))
actor_ppo_max_token_len=${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length) / sp_size))}
infer_ppo_max_token_len=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$(((max_prompt_length + max_response_length) / sp_size))}
torchtitan_max_seq_len=${TORCHTITAN_MAX_SEQ_LEN:-${max_model_len}}
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
VERIFY_PARITY="${VERIFY_PARITY:-off}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"

# Off-policy correction (truncated importance sampling for separate_async overlap)
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

python3 -m recipe.dapo.main_dapo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=${parameter_sync_step} \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +data.pad_mode=no_padding \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.max_length=${max_model_len} \
    +data.max_token_len_per_gpu=${actor_ppo_max_token_len} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
    actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp=True \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-${max_model_len}} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity="${VERIFY_PARITY}" \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger="${TRAINER_LOGGER:-$(if [[ -n "${WANDB_API_KEY:-}" ]]; then echo "['console','wandb','tensorboard']"; else echo "['console','tensorboard']"; fi)}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS:-20}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-/tmp/verl_dump/${exp_name}/rollout}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR:-/tmp/verl_dump/${exp_name}/validation}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    "${TPU_SLICE_ARGS[@]}" "$@"
