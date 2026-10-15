#!/usr/bin/env bash
# ReTool Stage 2 (Multi-Turn DAPO RL with Code Interpreter) | Qwen3-4B | TorchTitan & vLLM Rollout | TPU v6e-8 x2 Slices

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"
export VERL_TPU_SEQ_BUCKET_SIZE="${VERL_TPU_SEQ_BUCKET_SIZE:-1024}"
export TORCH_LOGS="${TORCH_LOGS:-recompiles}"
export TORCH_DYNAMO_RECOMPILE_LIMIT="${TORCH_DYNAMO_RECOMPILE_LIMIT:-64}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure local SandboxFusion FastAPI code execution server is running (default port 8080; fallback 18080 if 8080 is in use by another process)
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-8080}"
if ! python3 "${SCRIPT_DIR}/sandbox_fusion_server.py" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
    nohup python3 "${SCRIPT_DIR}/sandbox_fusion_server.py" --host 0.0.0.0 --port "${SANDBOX_FUSION_PORT}" > /tmp/sandbox_fusion_server.log 2>&1 &
    sleep 2
    if ! python3 "${SCRIPT_DIR}/sandbox_fusion_server.py" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
        SANDBOX_FUSION_PORT=18080
        if ! python3 "${SCRIPT_DIR}/sandbox_fusion_server.py" --port "${SANDBOX_FUSION_PORT}" --check >/dev/null 2>&1; then
            nohup python3 "${SCRIPT_DIR}/sandbox_fusion_server.py" --host 0.0.0.0 --port "${SANDBOX_FUSION_PORT}" > /tmp/sandbox_fusion_server.log 2>&1 &
            sleep 2
        fi
    fi
fi

# Load local WandB credentials (.env is gitignored; disable xtrace to avoid echoing WANDB_API_KEY)
set +x
if [[ -f "${VERL_ROOT}/.env" ]]; then
    source "${VERL_ROOT}/.env"
fi
set -x
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"

# ================= data/model/tool =================
HDFS_ROOT="${HDFS_ROOT:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-/data/jialei}"

dapo_math_17k="${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k"
aime_2024="${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024"
aime_2025="${DATA_ROOT}/dataset/yentinglin/aime_2025"
sft_experiment_name="${SFT_EXPERIMENT_NAME:-multiturn-sft-qwen3-4b-tpu}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -f "/tmp/multiturn-sft-qwen3-4b-tpu-hf/model.safetensors.index.json" ]]; then
        model_path="/tmp/multiturn-sft-qwen3-4b-tpu-hf"
    else
        model_path="${HDFS_ROOT}/checkpoint/${sft_experiment_name}/global_step_372/huggingface"
    fi
else
    model_path="${MODEL_PATH}"
fi

train_files="['${dapo_math_17k}']"
test_files="['${aime_2025}', '${aime_2024}']"

# tool & custom dataset/reward
tool_config_path="${VERL_ROOT}/recipe/retool/sandbox_fusion_tool_config.yaml"
if [[ "${SANDBOX_FUSION_PORT}" != "8080" ]]; then
    sed "s|http://localhost:8080/run_code|http://127.0.0.1:${SANDBOX_FUSION_PORT}/run_code|g" "${tool_config_path}" > "/tmp/sandbox_fusion_tool_config_${SANDBOX_FUSION_PORT}.yaml"
    tool_config_path="/tmp/sandbox_fusion_tool_config_${SANDBOX_FUSION_PORT}.yaml"
fi
retool_py_path="${VERL_ROOT}/recipe/retool/retool.py"

# logging / checkpoint
project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
experiment_name="${EXPERIMENT_NAME:-qwen3-4b_retool_dapo_tpu}"
default_local_dir="${CKPTS_DIR:-${DATA_ROOT}/checkpoint/${experiment_name}}"

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns="${MAX_TURNS:-16}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
max_model_len=$((max_prompt_length + max_response_length))
actor_lr="${ACTOR_LR:-1e-6}"

train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-30}"

# ================= TPU topology & performance =================
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))
ROLLOUT_TP="${ROLLOUT_TP:-${TOTAL_ROLLOUT_CHIPS}}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

actor_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
RELEASE_SYNC_BUFFERS="${RELEASE_SYNC_BUFFERS:-True}"
VERIFY_PARITY="${VERIFY_PARITY:-True}"

# Off-policy correction for separate_async overlap
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=True \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    +data.max_length="${max_model_len}" \
    +data.max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    data.custom_cls.path="${retool_py_path}" \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path="${retool_py_path}" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
    actor_rollout_ref.actor.torchtitan.tpu_eager_mode="${TPU_EAGER_MODE}" \
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp=True \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${max_model_len}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.release_buffers_after_sync="${RELEASE_SYNC_BUFFERS}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.raiden.verify_parity="${VERIFY_PARITY}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${tool_config_path}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt_val}" \
    trainer.logger="['console','wandb','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations=20 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
