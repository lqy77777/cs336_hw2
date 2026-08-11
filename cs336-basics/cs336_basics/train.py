import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import os
import json, time

from cs336_basics.tool import data_loader, save_checkpoint, load_checkpoint, load_tokens
from cs336_basics.tool import resolve_device, make_fixed_batches,log_jsonl
from cs336_basics.transformer import transformer_lm
from cs336_basics.optimizer import AdamW, cross_entropy, cosine_learning_rate, gradient_clipping

from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path

@dataclass
class TrainConfig:
    # 必填路径。无默认值字段必须写在最前面。
    train_data: str
    val_data: str
    out_dir: str

    # 数据
    data_dtype: str = "uint16"

    # 模型
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10_000.0

    # 优化器与学习率调度
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    alpha_max: float = 3e-4
    alpha_min: float = 3e-5
    T_w: int = 0
    T_c: int | None = None
    total_steps: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # 训练过程
    batch_size: int = 32
    device: Literal["auto", "cpu", "cuda", "mps"] = "cpu"
    seed: int = 0
    log_interval: int = 20
    eval_interval: int = 200
    eval_batches: int = 10
    checkpoint_interval: int = 500
    milestone_interval: int = 0
    #是否从一个已有 checkpoint 恢复训练，以及要从哪个 checkpoint 文件恢复。
    resume_from: str | None = None 

    def __post_init__(self) -> None:
        # 普通 dataclass 不会自动检查类型，因此仍需显式校验关键约束。
        if self.vocab_size <= 0:
            raise ValueError("vocab_size 必须为正")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("d_model 和 d_ff 必须为正")
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("num_layers 和 num_heads 必须为正")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} 不能被 num_heads={self.num_heads} 整除"
            )

        if self.T_c is None:
            self.T_c = self.total_steps

        # 上面已经将 None 替换为 int；assert 也能帮助类型检查器理解。
        assert self.T_c is not None

        if self.T_w < 0:
            raise ValueError("T_w 不能为负数")
        if self.T_w >= self.T_c:
            raise ValueError(f"T_w={self.T_w} 必须小于 T_c={self.T_c}")

        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正")
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches 必须为正")
        if self.context_length <= 0:
            raise ValueError("context_length 必须为正")

        if len(self.betas) != 2:
            raise ValueError("betas 必须正好包含 beta_1 和 beta_2 两个数")
        beta_1, beta_2 = self.betas
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"beta_1={beta_1} 必须位于 [0, 1)")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"beta_2={beta_2} 必须位于 [0, 1)")

        if self.eps < 0:
            raise ValueError("eps 不能为负数")
        if self.weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip 必须为正")
        if self.alpha_min < 0 or self.alpha_max < 0:
            raise ValueError("学习率不能为负数")
        if self.alpha_min > self.alpha_max:
            raise ValueError("alpha_min 不能大于 alpha_max")

        valid_devices = {"auto", "cpu", "cuda", "mps"}
        if self.device not in valid_devices:
            raise ValueError(f"device={self.device!r} 不在 {valid_devices} 中")

@torch.no_grad()
def evaluate(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """在固定验证 batch 上计算平均交叉熵，并禁止构建反向传播计算图"""
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        total += cross_entropy(model(inputs), targets).item()
    model.train()
    return total / len(batches)

def initialize() -> TrainConfig:
    return TrainConfig(
        train_data="tokens_id/tinystories_train_tokens.bin",
        val_data="tokens_id/tinystories_valid_tokens.bin",
        out_dir="runs/experiment01",
        device="cpu",
        T_w=100,
    )

def main(config: TrainConfig) -> None:

    # 1️⃣ 设置输出目录与配置
    out_dir = Path(config.out_dir)  #把字符串转换成path对象
    out_dir.mkdir(parents=True,exist_ok=True)
    config_dict = asdict(config)  #把配置保存为dict
    config_path = out_dir / "config.json"    #配置文件  
    metrics_path = out_dir / "metrics.jsonl"   #训练数据文件
    last_checkpoint_path = out_dir / "ckpt_last.pt"
    final_checkpoint_path = out_dir / "ckpt_final.pt"
    #保存到config.json
    with open(config_path,"w",encoding="utf-8",) as f:
        json.dump(config_dict,f,indent=2,ensure_ascii=False,default=str)

    #2️⃣ 创建数据、模型和优化器等
    # 1.设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # 2.设置设备
    device = resolve_device(config.device)
    print(f"[device] {device}")
    # 3.读取数据(memmap)
    train_data = load_tokens(config.train_data, config.data_dtype, config.context_length)
    val_data = load_tokens(config.val_data, config.data_dtype, config.context_length)
    # 4.创建模型
    model = transformer_lm(config.vocab_size,
                          config.context_length,
                          config.num_layers,
                          config.d_model,
                          config.num_heads,
                          config.d_ff,
                          config.rope_theta,
                          device)
    model.train()  #进入train模式
    n_params = sum(p.numel() for p in model.parameters()) #统计总参数数量
    print(f"[model] {n_params:,} parameters")
    # 5.创建优化器
    optimizer = AdamW(model.parameters(),
                      config.alpha_max,
                      config.betas,
                      config.eps,
                      config.weight_decay)
    # 6.固定验证集
    val_batches = make_fixed_batches(val_data, config.batch_size,
            config.context_length,device, config.eval_batches, config.seed + 1)

    #3️⃣ 恢复checkpoint
    #默认无需恢复checkpoint，从0开始训练，但如果需要恢复，则从恢复点开始训练
    start_step = 0
    if config.resume_from is not None:
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        print(f"[resume] 从 {config.resume_from} 恢复,从第 {start_step} 步继续")


    #4️⃣ 训练主循环
    t0 = time.perf_counter()
    tokens_per_step = config.batch_size * config.context_length
    loss_value = float("nan")
    for step in range(start_step, config.total_steps):
        # 1. 计算当前学习率，写进 param_groups
        lr = cosine_learning_rate(step, config.alpha_max, config.alpha_min, config.T_w, config.T_c)
        for group in optimizer.param_groups:
            #大部分情形只有一个参数组
            group["lr"] = lr
        # 2.清除旧梯度
        optimizer.zero_grad(set_to_none=True)
        # 3. 采样一个batch
        inputs, targets = data_loader(train_data, config.batch_size, config.context_length, device)
        # 4. 前向传播 + 计算loss + 反向传播
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        # 5. 梯度裁剪
        pre_clip_norm = gradient_clipping(model.parameters(), config.grad_clip)
        # 6. AdamW更新参数
        optimizer.step()
        # 7. 周期性完成日志、验证、checkpoint等记录任务
        #need_log:当前训练 step 是否需要打印并保存训练日志
        need_log = None
        if config.log_interval > 0:
            need_log = (step % config.log_interval == 0) or (step == config.total_steps - 1)
        if need_log:
            loss_value = loss.item()  #储存loss数值
            elapsed = time.perf_counter() - t0
            done = step - start_step + 1  #已进行的训练次数
            print(f"step {step:6d} | loss {loss_value:8.4f} | lr {lr:.3e} | gnorm {pre_clip_norm:8.3f}"
                  f" | {elapsed:7.1f}s | {done * tokens_per_step / max(elapsed, 1e-9):9,.0f} tok/s")
            log_jsonl(metrics_path, {"step": step, "train_loss": loss_value, "lr": lr,
                                     "grad_norm": pre_clip_norm, "elapsed": elapsed})
        #验证集评估
        if config.eval_interval > 0 and ((step + 1) % config.eval_interval == 0 or step + 1 == config.total_steps):
            val_loss = evaluate(model, val_batches)
            print(f"step {step:6d} | val loss {val_loss:8.4f}")
            log_jsonl(metrics_path, {"step": step, "val_loss": val_loss, "elapsed": time.perf_counter() - t0})
        #记录checkpoint
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            # 定期覆盖同一个文件（崩溃恢复用）
            save_checkpoint(model, optimizer, step + 1, last_checkpoint_path)
            if config.milestone_interval > 0 and (step + 1) % config.milestone_interval == 0:
                # 关键节点另存（事后分析用）
                save_checkpoint(model, optimizer, step + 1, os.path.join(config.out_dir, f"ckpt_{step + 1}.pt"))

    #5️⃣ 最终收尾
    save_checkpoint(model, optimizer, config.total_steps, final_checkpoint_path)
    final_val = evaluate(model, val_batches)
    total_time = time.perf_counter() - t0
    print(f"[done] {config.total_steps - start_step} steps in {total_time:.1f}s"
          f" | final train loss {loss_value:.4f} | final val loss {final_val:.4f}")
    log_jsonl(metrics_path, {"step": config.total_steps, "final_val_loss": final_val, "elapsed": total_time})


if __name__ == "__main__":
    config = initialize()
    main(config)
