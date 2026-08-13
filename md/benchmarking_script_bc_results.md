# `benchmarking_script` Part (b) 与 Part (c) 实验报告

实验日期：2026-08-13

本报告使用现有的 `cs336_systems/benchmark.py` 完成实验，没有修改该脚本。

## 1. 实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU 数量 | 1 |
| 可用显存 | 31.36 GiB（`nvidia-smi` 标称 32,607 MiB） |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| `uv` | 0.12.3 |
| dtype | `torch.float32` |
| vocabulary size | 10,000 |
| batch size | 4 |
| context length | 512 |
| measurement steps | 10 |

每个模型和 mode 都在独立进程中初始化，然后顺序执行，未在同一张 GPU 上并发运行实验。计时单位均为毫秒；脚本使用 `numpy.std()`，因此表中的 std 是 population standard deviation，即 `ddof=0`。

当前 `forward` 模式没有使用 `torch.no_grad()`，所以测得的是启用 autograd 的 training-style forward，而不是 inference-mode forward。

## 2. 计时含义与限制

现有脚本直接测量以下三种累计时间：

$$
T_F
$$

$$
T_{F+B}
$$

$$
T_{F+B+O}
$$

其中 $F$、$B$、$O$ 分别表示 forward、backward 和 optimizer step。因此，本报告中的单独 backward 和 optimizer 时间只能通过独立 benchmark run 的均值作近似差分：

$$
T_B \approx T_{F+B}-T_F
$$

$$
T_O \approx T_{F+B+O}-T_{F+B}
$$

差分结果不是脚本直接测得的 isolated phase latency，且不同 run 之间的 GPU 时钟和系统状态可能不同，所以应视为近似值。累计时间及其 std 才是脚本直接产生的原始统计结果。

## 3. Part (b)：5 次 warm-up 的结果

实验配置为 5 次 warm-up 和 10 次正式 measurements。

### 3.1 脚本直接测得的累计时间

表格格式为 `mean ± std`，单位为 ms。

| model | forward $T_F$ | forward + backward $T_{F+B}$ | full step $T_{F+B+O}$ |
| --- | ---: | ---: | ---: |
| small | 16.292 ± 0.030 | 56.257 ± 2.220 | 69.323 ± 4.909 |
| medium | 47.073 ± 0.352 | 152.292 ± 0.129 | 162.910 ± 0.419 |
| large | 109.847 ± 0.172 | 337.127 ± 0.724 | 362.778 ± 0.479 |
| xl | OOM | OOM | OOM |
| 10B | OOM | OOM | OOM |

### 3.2 由均值差分得到的阶段时间

| model | forward | backward（近似） | optimizer（近似） |
| --- | ---: | ---: | ---: |
| small | 16.292 | 39.965 | 13.066 |
| medium | 47.073 | 105.219 | 10.618 |
| large | 109.847 | 227.280 | 25.651 |
| xl | OOM | OOM | OOM |
| 10B | OOM | OOM | OOM |

### 3.3 OOM 情况

- `xl` 可以完成模型初始化，但三种 mode 都在第一次 warm-up forward 中 OOM。报错时进程约使用 31.33 GiB GPU memory，只剩约 17.69 MiB，下一次 20 MiB allocation 失败。
- `10B` 在模型初始化期间即 OOM，因此与 mode 无关。报错时进程约使用 31.28 GiB GPU memory，下一次 82 MiB parameter allocation 失败。
- `xl` 的 forward 之所以也会 OOM，是因为当前 `forward` 模式启用了 autograd，需要为 backward 保存 residuals；它不是 `torch.no_grad()` 下的 inference forward。

### 3.4 Part (b) 结果分析

在 RTX 5090 上，small、medium、large 的 forward 平均耗时分别为 16.292、47.073、109.847 ms；由累计时间差分估计，backward 分别约为 39.965、105.219、227.280 ms。充分 warm-up 后大多数配置的 std 相对 mean 很小，说明测量较稳定，但 small 的 forward+backward 和 full-step 分别出现 2.220 ms 和 4.909 ms 的较明显波动；xl 和 10B 在当前 FP32、batch size 4、context length 512 设置下 OOM。

## 4. Part (c)：不同 warm-up 次数的对照

除 warm-up steps 外，其余设置保持不变。对可以运行的 small、medium 和 large 模型，分别测试了 0、1、2、5 次 warm-up；`xl` 与 `10B` 的 OOM 不会因减少 warm-up 次数而消失，因此没有重复无意义的完整 sweep。

### 4.1 Small

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 43.274 ± 80.897 | 86.214 ± 100.098 | 97.448 ± 105.763 |
| 1 | 16.398 ± 0.040 | 53.292 ± 1.696 | 55.187 ± 0.194 |
| 2 | 16.371 ± 0.024 | 54.668 ± 2.414 | 65.424 ± 7.944 |
| 5 | 16.292 ± 0.030 | 56.257 ± 2.220 | 69.323 ± 4.909 |

### 4.2 Medium

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 72.196 ± 76.533 | 183.317 ± 94.105 | 196.996 ± 100.924 |
| 1 | 46.819 ± 0.415 | 152.195 ± 0.912 | 163.000 ± 0.198 |
| 2 | 46.854 ± 0.370 | 152.722 ± 0.722 | 163.002 ± 0.169 |
| 5 | 47.073 ± 0.352 | 152.292 ± 0.129 | 162.910 ± 0.419 |

### 4.3 Large

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 133.985 ± 72.362 | 364.738 ± 87.297 | 392.344 ± 89.637 |
| 1 | 110.262 ± 1.706 | 336.272 ± 1.312 | 363.430 ± 0.300 |
| 2 | 109.943 ± 0.334 | 336.895 ± 0.514 | 363.856 ± 0.276 |
| 5 | 109.847 ± 0.172 | 337.127 ± 0.724 | 362.778 ± 0.479 |

### 4.4 Part (c) 结果分析

不做 warm-up 时，首次 model step 包含尚未被模型初始化过程触发的 CUDA library/kernel lazy loading、首次中间 tensor 显存分配等一次性成本；full-step 还包含 AdamW optimizer state 的首次初始化，因此三个可运行模型的 mean 都被抬高，std 更达到约 72–106 ms。做 1 次 warm-up 后，本次 RTX 5090 实验中的多数配置已经接近 steady state，2–5 次主要进一步减小部分配置的波动；不过少量 warm-up 在其他 GPU、shape、dtype、`torch.compile` 或 Triton JIT 环境中仍可能不足，因为不同 lazy initialization 和 cache 建立过程不一定能在同一步内全部完成。

small 的 full-step 在不同独立进程之间仍有一定波动，且 1、2、5 次 warm-up 的均值不单调。这说明在已经去除明显冷启动后，独立 run 之间的 GPU 动态频率、温度、系统噪声和 allocator 状态仍可能影响较短的 workload；不能把这些小范围差异解释成“更多 warm-up 会让计算本身变慢”。

## 5. 可直接用于 writeup 的精简回答

### Part (b)

在 RTX 5090 上，small、medium、large 的 forward pass 分别耗时 16.292、47.073、109.847 ms，由累计模式差分估计 backward 分别约耗时 39.965、105.219、227.280 ms；大多数充分预热后的结果标准差较小，但 small 的两个训练模式波动稍明显。xl 和 10B 在 FP32、batch size 4、context length 512 下超出 31.36 GiB 显存，其中 xl 在第一次 forward OOM，10B 在模型初始化时 OOM。

### Part (c)

没有 warm-up 时，首次 step 尚未触发的底层库/kernel lazy loading、首次中间 tensor 显存分配以及 optimizer state 初始化进入正式计时，使 mean 明显升高，并让 std 达到约 72–106 ms。1 次 warm-up 在本机多数配置上已基本消除冷启动，2–5 次可进一步确保进入 steady state；但初始化可能分散在多个代码路径中，所以 1–2 次 warm-up 在不同模型、硬件或启用编译时仍未必足够。

## 6. 复现实验命令

Part (b) 的单次配置示例：

```bash
uv run python -m cs336_systems.benchmark \
  --device cuda \
  --dtype float32 \
  --model-size small \
  --mode forward \
  --batch-size 4 \
  --context-length 512 \
  --warmup-steps 5 \
  --measurement-steps 10
```

Part (c) 只需保持其他参数不变，将 `--warmup-steps` 分别设为 `0`、`1`、`2` 和 `5`。
