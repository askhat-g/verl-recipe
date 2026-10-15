#!/usr/bin/env bash
# ReTool Stage 1 (Cold-Start Multi-Turn SFT) | Qwen3-32B | ReTool-SFT | TorchTitan Engine (SimpleFSDP, DEFER_NEVER) | TPU v6e-32 (4x8)

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-32}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-2048}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load local WandB credentials (.env is gitignored; disable xtrace to avoid echoing WANDB_API_KEY)
set +x
if [[ -f "${VERL_ROOT}/.env" ]]; then
    source "${VERL_ROOT}/.env"
fi
set -x
# Ensure WANDB_BASE_URL is not set to the HTML frontend URL
if [[ "${WANDB_BASE_URL:-}" == *"forge.coreweave.com"* ]]; then
    unset WANDB_BASE_URL
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl-torchtpu-grpo-test}"
export WANDB_MODE="${WANDB_MODE:-online}"

project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
experiment_name="${EXPERIMENT_NAME:-multiturn-sft-qwen3-32b-tpu}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
TRAIN_DATA="${TRAIN_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
EVAL_DATA="${EVAL_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -f "/dev/shm/Qwen3-32B/model.safetensors.index.json" ]]; then
        MODEL_PATH="/dev/shm/Qwen3-32B"
    elif [[ -f "/tmp/Qwen3-32B/model.safetensors.index.json" ]]; then
        MODEL_PATH="/tmp/Qwen3-32B"
    else
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-32B"
    fi
fi
if [[ -z "${SAVE_PATH:-}" ]]; then
    if [[ -d "${RAY_DATA_HOME}" ]]; then
        SAVE_PATH="${RAY_DATA_HOME}/checkpoint/${experiment_name}"
    elif [[ -d "/dev/shm" ]]; then
        SAVE_PATH="/dev/shm/checkpoint/${experiment_name}"
    else
        SAVE_PATH="${RAY_DATA_HOME}/checkpoint/${experiment_name}"
    fi
fi

# TPU v6e-32 (4x8) Topology: 8 physical VM hosts x 4 TPU chips per host = 32 trainer chips (FSDP32)
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))}"

# Train at 16384 max_length with Ulysses Context Parallelism (CP=4, FSDP=8) on 32 TPU v6e chips
# (6 epochs x 62 steps/epoch at train_batch_size=32 = 372 total steps)
MAX_LENGTH="${MAX_LENGTH:-16384}"
TRUNCATION="${TRUNCATION:-right}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU=$((MICRO_BATCH_SIZE_PER_GPU * MAX_LENGTH / CONTEXT_PARALLEL_SIZE))

TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
SAVE_FREQ="${SAVE_FREQ:-186}"
TEST_FREQ="${TEST_FREQ:-62}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"

python3 -m verl.trainer.sft_trainer_ray \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${EVAL_DATA}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.pad_mode=no_padding \
    data.truncation="${TRUNCATION}" \
    data.use_dynamic_bsz=False \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=torchtitan \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    engine.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    engine.use_torch_compile=True \
    engine.use_splash_attention=True \
    engine.use_simple_fsdp=True \
    engine.tpu_eager_mode=DEFER_NEVER \
    engine.activation_checkpoint=full \
    engine.attn_type=varlen \
    engine.max_seq_len="${MAX_LENGTH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    "checkpoint.save_contents=['hf_model']" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger="['console','wandb','tensorboard','file']" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "$@"
