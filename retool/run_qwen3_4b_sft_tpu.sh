#!/usr/bin/env bash
# ReTool Stage 1 (Cold-Start Multi-Turn SFT) | Qwen3-4B | ReTool-SFT | TorchTitan Engine | TPU v6e-8

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-1024}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

project_name="${PROJECT_NAME:-retool_sft}"
experiment_name="${EXPERIMENT_NAME:-multiturn-sft-qwen3-4b-tpu}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
TRAIN_DATA="${TRAIN_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
EVAL_DATA="${EVAL_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-4B}"
SAVE_PATH="${SAVE_PATH:-${RAY_DATA_HOME}/checkpoint/${experiment_name}}"

# TPU Topology (default: 1 host x 8 TPU v6e chips = 8 trainer chips; override NNODES_TRAINER=2 N_CHIPS_TRAINER=4 for 2-host v6e-8)
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${TOTAL_TRAINER_CHIPS}}"

MAX_LENGTH="${MAX_LENGTH:-16384}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU=$((MICRO_BATCH_SIZE_PER_GPU * MAX_LENGTH))

TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
SAVE_FREQ="${SAVE_FREQ:-62}"
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
    data.truncation=error \
    data.use_dynamic_bsz=False \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=torchtitan \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size=1 \
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
    trainer.logger="['console','tensorboard']" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "$@"
