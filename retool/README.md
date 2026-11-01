# Retool

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

## Overview
- Base model: [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) / [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (MoE on TPU)
- SFT dataset: [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
- RL dataset: [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Val dataset: [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025), [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)

## How it works

 Retool's workflow is divided into two key phases:

1.  Cold Start and Supervised Fine Tuning (SFT)
    
     The data generation pipeline builds a high-quality dataset containing code-enhanced inference trajectories, and supervised fine-tuning enables the model to master basic Tool call (e.g., code execution) and analysis of the execution results.
    
2.  Dynamic Interaction and Policy Optimization (RL).
    
     With the verl Reinforcement Learning framework, the model dynamically inserts code blocks during inference and interacts with the sandbox environment in real-time, generating a hybrid trajectory of natural language thinking and code snippets, sending the code to the sandbox for asynchronous execution when code termination markers are detected, and the execution results (success outputs/errors) are fed back to the model for guiding the subsequent inference. This "think-execute-feedback" cycle, together with the design of rewards based on the accuracy of the final answer, enables the model to independently optimize the Tool call strategy, and improves the reasoning efficiency and computational accuracy.

## Installation

```bash
pip install verl==0.6.1
```

## SFT
1. Data preparation
```bash
python3 recipe/retool/retool_sft_preprocess.py
```

2. Training
```bash
bash recipe/retool/run_qwen2-32b_sft.sh
```

After 6 epoches, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.24
- val-aux/num_turns/mean: 7.2

## RL

### GRPO
```bash
bash recipe/retool/run_qwen2-32b_dapo.sh
```

After 150 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.6
- val-aux/num_turns/mean: 10

### PPO

```bash
bash recipe/retool/run_qwen2-32b_ppo.sh
```

After 250 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.55
- val-aux/num_turns/mean: 8.3

## TPU Recipe (`Qwen3-30B-A3B` MoE on TPU v7x / v6e)

Self-contained TPU recipe for `Qwen/Qwen3-30B-A3B` (`30.5B` total parameters, `3.3B` active per token, `128` routed experts, `top_k=8`) using the **TorchTitan** trainer (`SimpleFSDP=16`, `CP=1`, `EP=1`, `TPU_EAGER_MODE=DEFER_NEVER`, `SplashAttention`, MegaBlox Pallas `grouped_mm`) and **vLLM** multi-turn async rollout (`TP=16`, `EP=16`) with **Raiden** direct P2P weight synchronization.

### Hardware Topology (`2 x TPU v7x 2x2x4` Slices)

| Role | Default Slice (`2 hosts x 8 TensorCores = 16 TensorCores`) | Parallelism | Memory & Throughput Notes |
| :--- | :--- | :--- | :--- |
| **Trainer** (`16` TensorCores) | `tpu-group-0` (`2 hosts x 8 TensorCores`, `2x2x4`) | `SimpleFSDP=16`, `CP=1`, `TP=1`, `EP=1` | Full `18,432`-token sequence fits on one `96 GB` HBM TensorCore (`VERL_TPU_SEQ_BUCKET_SIZE=4608`, `TPU_EAGER_MODE=DEFER_NEVER`). |
| **Rollout** (`16` TensorCores) | `tpu-group-1` (`2 hosts x 8 TensorCores`, `2x2x4`) | `1` server of `TP=16`, `EP=16` | `ROLLOUT_EP == ROLLOUT_TP` (`8` whole experts/TensorCore), `block_size=256`, `gpu_memory_utilization=0.80`, `max_num_seqs=512`, `max_num_batched_tokens=8192`. |
| **Code Sandbox** | CPU head node (`127.0.0.1:18080`) | `16` `uvicorn` workers | Loopback FastAPI Python sandbox (`8s` default timeout, `1 GiB` `RLIMIT_AS`); `ExecutionWorker` is created lazily and pinned to the CPU head node via `ray_actor_options: {resources: {"node:__internal_head__": 0.001}}` in `sandbox_fusion_tool_config_tpu.yaml`. |

### 1. Stage 1: Multi-Turn Cold-Start SFT (`372` Steps)

```bash
bash recipe/retool/run_qwen3_30b_a3b_sft_tpu.sh
```

### 2. Stage 2: Multi-Turn DAPO RL with Code Interpreter (`150` Steps, `16K` Native Thinking)

Default schedule on `2 x TPU v7x 2x2x4` (`TRAIN_BATCH_SIZE=64`, `PPO_MINI_BATCH_SIZE=64`, `N_RESP_PER_PROMPT=16` (`1,024` rollouts/step), `MAX_PROMPT_LENGTH=2048`, `MAX_RESPONSE_LENGTH=16384`, `MAX_TURNS=8`, `ENABLE_THINKING=True`, `ACTOR_LR=2e-6`, `CLIP_GRAD=0.0625`, `bypass_mode=True`, `TOTAL_TRAINING_STEPS=150`):

```bash
bash recipe/retool/run_qwen3_30b_a3b_dapo_tpu.sh
```

On `2 x TPU v6e-32` slices (`8 hosts x 4 chips = 32 chips` per slice):

```bash
TPU_ACCELERATOR_TYPE=v6e NNODES_TRAINER=8 N_CHIPS_TRAINER=4 NNODES_ROLLOUT=8 N_CHIPS_ROLLOUT=4 \
    CONTEXT_PARALLEL_SIZE=4 ACTOR_MAX_TOKEN_LEN_PER_GPU=4608 LOG_PROB_MAX_TOKEN_LEN_PER_GPU=4608 \
    VERL_TPU_SEQ_BUCKET_SIZE=2048 ROLLOUT_GPU_MEMORY_UTILIZATION=0.62 ROLLOUT_MAX_NUM_SEQS=256 \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=4096 bash recipe/retool/run_qwen3_30b_a3b_dapo_tpu.sh
```
