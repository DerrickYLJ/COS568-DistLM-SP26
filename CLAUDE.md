# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **COS 568 (Princeton, Spring 2026) Assignment 2** on distributed training of language models. The assignment fine-tunes BERT on GLUE benchmark tasks and progressively implements distributed data parallelism.

## Assignment Structure

Four tasks of increasing complexity:
- **Task 1** — Single-node BERT fine-tuning on the RTE dataset
- **Task 2a** — Distributed training using gather/scatter communication primitives
- **Task 2b** — Distributed training using all_reduce
- **Task 3** — Distributed training using PyTorch `DistributedDataParallel`
- **Task 4** — Profiling and benchmarking

Deliverables are organized into subdirectories: `task1/`, `task2a/`, `task2b/`, `task3/`, `task4/`.

## Setup

```bash
# Python dependencies
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn tqdm pytorch_transformers apex

# Download GLUE data
mkdir glue_data
python3 download_glue_data.py --data_dir glue_data
```

## Running Training

### Task 1 — Single-node on CloudLab

**1. SSH into node-0**
```bash
ssh -i ~/path/to/id_cloudlab ly3223@clnode087.clemson.cloudlab.us
```

**2. Install dependencies (run once per experiment)**
```bash
git clone https://github.com/DerrickYLJ/COS568-DistLM-SP26.git & sudo apt-get update && sudo apt-get install -y htop dstat python3-pip & echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc && source ~/.bashrc & pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu & pip install numpy scipy scikit-learn tqdm pytorch_transformers apex
```

**3. Clone repo and download data**
```bash
cd COS568-DistLM-SP26 & mkdir ~/glue_data & python3 download_glue_data.py --data_dir ~/glue_data
```

**4. Run training inside tmux (persists on disconnect)**
```bash
tmux new -s task1

export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python3 run_glue_skeleton.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir 2>&1 | tee task1_output.log
```

Reattach after disconnect: `tmux attach -s task1`

### Task 2a — Distributed gather/scatter on CloudLab (4 nodes)

**1. Find the experimental network IP on each node**
```bash
ifconfig | grep "10.10.1"
```
Note node-0's `10.10.1.*` address — this is `MASTER_IP`.

**2. On each node: clone repo + install deps + download data** (same as Task 1 setup)

**3. Launch on all 4 nodes simultaneously** (open tmux on each before starting)

```bash
# node-0 (rank 0) — master
export GLUE_DIR=$HOME/glue_data TASK_NAME=RTE
cd ~/COS568-DistLM-SP26/task2a
tmux new -s task2a
python3 run_glue.py \
  --model_type bert --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME --do_train --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 --per_device_train_batch_size 16 \
  --learning_rate 2e-5 --num_train_epochs 1 \
  --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir \
  --master_ip <MASTER_IP> --master_port 12345 \
  --world_size 4 --local_rank 0 2>&1 | tee task2a_rank0.log

# node-1 (rank 1): same command with --local_rank 1
# node-2 (rank 2): same command with --local_rank 2
# node-3 (rank 3): same command with --local_rank 3
```

Key parameters:
- `--per_device_train_batch_size 16` × 4 workers = 64 total (same as Task 1)
- `--num_train_epochs 1` (required for Tasks 2/3 per README)
- All 4 nodes must start within ~30s of each other or init times out
- On Clemson CloudLab (no 10.10.1.* interface): use the public IP from `ifconfig` (e.g. `128.110.219.149`)

Implementation in `task2a/run_glue.py`:
- `sync_gradients_gather_scatter()` — rank 0 gathers all `.grad` tensors, averages them, scatters mean back to every rank
- `DistributedSampler` replaces `RandomSampler` so each rank sees a disjoint data shard
- Evaluation runs on rank 0 only

### Task 2b — Distributed all_reduce on CloudLab (4 nodes)

Same 4-node setup as Task 2a. Run from `task2b/` dir using a different port:

```bash
# node-0 (rank 0) — master
export GLUE_DIR=$HOME/glue_data TASK_NAME=RTE
cd ~/COS568-DistLM-SP26/task2b
tmux new -s task2b
python3 run_glue.py \
  --model_type bert --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME --do_train --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 --per_device_train_batch_size 16 \
  --learning_rate 2e-5 --num_train_epochs 1 \
  --output_dir /tmp/${TASK_NAME}_2b/ --overwrite_output_dir \
  --master_ip <MASTER_IP> --master_port 9877 \
  --world_size 4 --local_rank 0 2>&1 | tee task2b_rank0.log

# node-1 (rank 1): same command with --local_rank 1
# node-2 (rank 2): same command with --local_rank 2
# node-3 (rank 3): same command with --local_rank 3
```

Implementation in `task2b/run_glue.py`:
- `sync_gradients_all_reduce()` — in-place SUM all_reduce on every `.grad` tensor, then divides by `world_size`
- Simpler and more scalable than gather/scatter; loss values should be identical to Task 2a

## Code Architecture

### `run_glue_skeleton.py` — Main training script
- `main()` — Parses args, initializes model/tokenizer/optimizer, calls `train()` and `evaluate()`
- `train()` — Training loop: forward pass → backward pass → gradient sync → optimizer step. Contains `# TODO` markers where students implement distributed gradient aggregation
- `evaluate()` — Runs dev set evaluation and computes task-specific metrics
- `load_and_cache_examples()` — Loads and tokenizes dataset, with disk caching

The script supports four model architectures via `pytorch_transformers`: BERT, RoBERTa, XLNet, XLM. Model/tokenizer/config classes are selected via `MODEL_CLASSES` dict keyed by `--model_type`.

### `utils_glue.py` — GLUE dataset utilities
- `DataProcessor` base class + one subclass per GLUE task (MRPC, MNLI, CoLA, SST-2, STS-B, QQP, QNLI, RTE, WNLI)
- `convert_examples_to_features()` — Tokenizes examples and pads/truncates to `max_seq_length`
- Metric functions: accuracy, F1, Matthews correlation, Pearson/Spearman correlation

### Key implementation locations for distributed training
- Gradient synchronization goes inside `train()` after `loss.backward()` and before `optimizer.step()`
- Use `torch.distributed` for point-to-point and collective communication
- `DistributedSampler` ensures each rank sees a disjoint data partition
- PyTorch distributed init: `torch.distributed.init_process_group(backend='gloo', init_method='tcp://<master_ip>:<port>', world_size=N, rank=r)`


### Task 1
Output: 
