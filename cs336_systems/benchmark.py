import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import os
import json, time
from timeit import default_timer
import argparse

from cs336_basics.tool import data_loader, save_checkpoint, load_checkpoint, load_tokens
from cs336_basics.tool import resolve_device, make_fixed_batches,log_jsonl
from cs336_basics.transformer import transformer_lm
from cs336_basics.optimizer import AdamW, cross_entropy, cosine_learning_rate, gradient_clipping

from dataclasses import asdict, dataclass

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B":{
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    }
}
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
def run_step(model, inputs, targets, mode, optimizer):
    if mode not in {"forward", "forward_backward",'full_step'}:
        raise ValueError(f"Unknown mode: {mode}")
    if mode in {'forward_backward','full_step'}:
        model.zero_grad(set_to_none = True)
    with torch.cuda.nvtx.range("forward"):
        logits = model(inputs)
    if mode == 'forward':
        return 
    with torch.cuda.nvtx.range("loss"):
        loss = cross_entropy(logits, targets)
    with torch.cuda.nvtx.range("backward"):
        loss.backward()

    if mode == 'full_step':
        with torch.cuda.nvtx.range("optimizer"):
            optimizer.step()

def synchronize(device):
    #避免在没有cuda的时候报错
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def parse_args():
    parser = argparse.ArgumentParser(description = 
                                     'Benchmark Transformer forward and backward')
    parser.add_argument(
        '--mode',
        choices = ['forward','forward_backward','full_step'],
        default = 'forward_backward',
        help = 'Which model operations to benchmark'
    )
    parser.add_argument(
        '--warmup-steps',
        type = int,
        default = 5,
        help = "Number of warmup steps"
    )
    parser.add_argument(
        '--measurement-steps',
        type = int,
        default = 10,
        help = 'Number of measured steps'
    )
    parser.add_argument(
        '--device',
        choices = ['cpu', 'cuda'],
        default = 'cpu',
        help = 'Device used for benchmarking'
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Floating-point dtype used by the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences in one batch",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Number of tokens in each sequence",
    )
    parser.add_argument(
        '--model-size',
        choices = ['small','medium','large','xl','10B'],
        default = 'medium',
        help = 'Transformer model configuration'
    )
    args = parser.parse_args()  #不是循环嵌套

    if args.device == "cpu" and args.dtype != "float32":
        parser.error("CPU benchmarks currently only support --dtype float32")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be at least 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.context_length < 1:
        parser.error("--context-length must be at least 1")
    if args.measurement_steps < 1:
        parser.error("--measurement-steps must be at least 1")

    return args
def main():
    args = parse_args()

    dtype = DTYPE_MAP[args.dtype]
    device = torch.device(args.device)
    mode = args.mode
    batch_size = args.batch_size
    context_length = args.context_length
    model_size = args.model_size
    model_config = MODEL_CONFIGS[model_size]
    vocab_size = 10000

    model = transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=model_config['num_layers'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=10000,
        device=device,
        dtype=dtype,
    )

    optimizer = AdamW(model.parameters(),lr = 1e-3)

    #生成随机token
    inputs = torch.randint(low = 0, high = vocab_size, size = (batch_size,context_length), device=device,dtype = torch.long)
    targets = torch.randint(low = 0, high = vocab_size, size = (batch_size,context_length),device = device, dtype = torch.long)
    warmup_steps = args.warmup_steps
    #warm-up 的作用是让首次执行涉及的初始化、内存分配和缓存建立完成，避免它们污染正式计时结果。
    for _ in range(warmup_steps):
        run_step(model,inputs,targets,mode,optimizer)
        synchronize(device)
    print('warmup finished')

    #正式测量
    measurement_steps = args.measurement_steps
    elapsed_times = []
    with torch.cuda.nvtx.range("measured_region"):
        for _ in range(measurement_steps):
            synchronize(device)
            start = default_timer()

            run_step(model, inputs, targets,mode,optimizer)

            synchronize(device)
            end = default_timer()
            elapsed_times.append(end - start)
    elapsed_times_ms = np.array(elapsed_times) * 1000  #转化为毫秒
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f'mode: {mode}')
    print(f"batch size: {batch_size}")
    print(f"context length: {context_length}")
    print(f"model size: {model_size}")
    print(f"model config: {model_config}")
    print(f'mean: {elapsed_times_ms.mean():.3f} ms')
    print(f'std: {elapsed_times_ms.std():.3f} ms')
if __name__ == '__main__':
    main()