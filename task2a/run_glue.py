# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Task 2a: Distributed fine-tuning with gradient synchronization via gather/scatter. """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.distributed
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert':    (BertConfig,   BertForSequenceClassification,   BertTokenizer),
    'xlnet':   (XLNetConfig,  XLNetForSequenceClassification,  XLNetTokenizer),
    'xlm':     (XLMConfig,    XLMForSequenceClassification,    XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)


def sync_gradients_gather_scatter(model, args):
    """Synchronize gradients across all workers using gather + scatter.

    Rank 0 gathers all gradients, averages them, then scatters the mean
    back to every worker.  After this call every rank holds identical
    averaged gradients, so a subsequent optimizer.step() is equivalent to
    the single-node update on the full (world_size × local_batch) batch.
    """
    world_size = args.world_size
    rank = args.local_rank

    for param in model.parameters():
        if param.grad is None:
            continue

        grad = param.grad.data

        # ── gather ────────────────────────────────────────────────────────────
        # Rank 0 allocates a list to receive one tensor per worker.
        # Other ranks pass gather_list=None (ignored by the collective).
        if rank == 0:
            gather_list = [torch.zeros_like(grad) for _ in range(world_size)]
        else:
            gather_list = None

        torch.distributed.gather(grad, gather_list=gather_list, dst=0)

        # ── average + scatter ─────────────────────────────────────────────────
        # Rank 0 computes the element-wise mean and builds scatter_list.
        # scatter() sends scatter_list[i] to rank i.
        if rank == 0:
            avg = torch.mean(torch.stack(gather_list), dim=0)
            scatter_list = [avg.clone() for _ in range(world_size)]
        else:
            scatter_list = None

        torch.distributed.scatter(grad, scatter_list=scatter_list, src=0)

        # Write the received averaged gradient back into the parameter.
        param.grad.data.copy_(grad)


def train(args, train_dataset, model, tokenizer):
    """Train the model (distributed: gather/scatter gradient sync)."""

    args.train_batch_size = args.per_device_train_batch_size

    # Use DistributedSampler so each rank sees a disjoint data partition.
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=args.world_size,
                                           rank=args.local_rank)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training (Task 2a: gather/scatter) *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per worker = %d", args.per_device_train_batch_size)
    logger.info("  Total batch size (all workers) = %d",
                args.train_batch_size * args.gradient_accumulation_steps *
                (args.world_size if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        iter_times = []

        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                'labels':         batch[3],
            }

            t0 = time.time()

            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Synchronize gradients across all workers via gather/scatter.
            if args.local_rank != -1:
                sync_gradients_gather_scatter(model, args)

            tr_loss += loss.item()

            # Log loss for first 5 iterations on rank 0
            if global_step < 5 and args.local_rank in [-1, 0]:
                logger.info("  Step %d loss = %.6f", global_step, loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                iter_time = time.time() - t0
                iter_times.append(iter_time)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # Report average iteration time (skip first iteration to exclude warmup)
        if len(iter_times) > 1 and args.local_rank in [-1, 0]:
            avg_time = sum(iter_times[1:]) / len(iter_times[1:])
            logger.info("  Avg time per iteration (excl. first): %.4f s", avg_time)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # Evaluate at end of each epoch (rank 0 only)
        if args.local_rank in [-1, 0]:
            evaluate(args, model, tokenizer, prefix="")

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.eval_batch_size)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'labels':         batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    # Barriers only needed during training so all ranks share the same cache.
    # During evaluation only rank 0 runs, so skip barriers to avoid hanging.
    if not evaluate and args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if not evaluate and args.local_rank == 0:
        torch.distributed.barrier()

    all_input_ids  = torch.tensor([f.input_ids   for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask  for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_type", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    # Other parameters
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--per_device_train_batch_size", default=8, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')

    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Rank of this process. -1 = single-node (no distribution).")
    parser.add_argument("--master_ip", type=str, default="10.10.1.1",
                        help="IP address of rank-0 node (use 10.10.1.* on CloudLab).")
    parser.add_argument("--master_port", type=str, default="12345",
                        help="Free port on the master node (>1023).")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of participating nodes.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Initialize distributed process group
    if args.local_rank != -1:
        torch.distributed.init_process_group(
            backend='gloo',
            init_method=f"tcp://{args.master_ip}:{args.master_port}",
            world_size=args.world_size,
            rank=args.local_rank,
        )
        logger.info("Distributed process group initialized: rank %d / %d",
                    args.local_rank, args.world_size)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, world_size: %s, 16-bits: %s",
                   args.local_rank, args.device, args.world_size, args.fp16)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Only rank 0 downloads model weights; others wait at barrier.
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.local_rank in [-1, 0]:
        evaluate(args, model, tokenizer, prefix="")


if __name__ == "__main__":
    main()
