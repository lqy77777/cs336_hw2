import torch
import numpy as np
from numpy.typing import NDArray
import torch.nn as nn
from math import sqrt,cos,pi
from einops import einsum
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional
import os
from typing import BinaryIO,IO
import json

def load_tokens(
        path: str, #token文件路径
        dtype: str, #如unit16
        context_length: int,  #每条训练样本所含token数量
) -> np.memmap:
    """以内存映射方式打开.bin token 文件，并进行长度检查。
    """
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")

    if len(data) < context_length + 1:
        raise ValueError(f" 数据只有 {len(data)} 个 token,装不下 context_length={context_length}")

    return data

def data_loader(
        x: Int[NDArray,''],
        batch_size,
        context_length,
        device = None,
        dtype = torch.int64
) -> tuple[Tensor,Tensor]:
    '''从一整条 token 序列中随机截取 batch_size 个连续片段，
    并构造语言模型训练需要的输入 inputs 和右移一位的标签 targets
    轻量的随机 batch 采样函数'''
    B, n, m = batch_size, len(x), context_length
    batch_start = np.random.randint(0,n-m,size = B)
    index = batch_start[:,None] +np.arange(m)
    inputs = torch.tensor(x[index], dtype= dtype,device = device)
    targets = torch.tensor(x[index+1], dtype= dtype,device = device)
    return (inputs, targets)

def make_fixed_batches(
        data: NDArray, #通常是验证集memmap
        batch_size: int,
        context_length: int,
        device: torch.device,
        num_batches: int,
        seed: int,
) -> list[tuple[Tensor, Tensor]]:
    """提前从验证集随机采样固定的一组 batch。
    之后每次验证都使用同样的数据，避免验证曲线因为每次随机样本不同而抖动。
    """
    state = np.random.get_state()
    np.random.seed(seed)
    batches = [data_loader(data, batch_size, context_length, device) for _ in range(num_batches)]
    np.random.set_state(state)
    return batches

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    state_model = model.state_dict()
    state_opt = optimizer.state_dict()
    obj = {'model': state_model,'opt': state_opt,'iteration': iteration}
    torch.save(obj,out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['opt'])
    return obj['iteration']

def resolve_device(name: str) -> torch.device:
    '''把配置字符串转换成 torch.device
        根据当前机器环境选择或检查可用设备'''
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但这台机器没有可用的 CUDA")
    return torch.device(name)

def log_jsonl(path: str, record: dict) -> None:
    """每条日志追加一行 JSON。Section 6/7 画曲线时直接读这个文件，不必去 grep stdout。"""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
