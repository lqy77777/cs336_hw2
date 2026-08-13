# CS336 Assignment 2：`benchmarking_script` 完整实验总结

## 1. 问题目标：这道题要我们做什么

`benchmarking_script` 是 Assignment 2 的第一个问题。它的目标不是训练出一个有效的语言模型，而是建立一个之后可以复用的端到端性能测试工具，并借此理解如何正确测量 GPU 程序。

整个问题分为三部分。

### 1.1 Part (a)：实现 benchmark 脚本

脚本需要完成以下功能：

1. 根据给定的模型超参数初始化 Transformer；
2. 生成随机 input tokens 和 targets；
3. 正式计时前运行 $w$ 个 warm-up steps；
4. 正式测量 $n$ 个 steps；
5. 支持三种运行模式：
   - 只运行 forward；
   - 运行 forward 和 backward；
   - 运行 forward、backward 和 optimizer step；
6. 使用高精度 wall-clock timer，例如 `timeit.default_timer()`；
7. 每个 GPU step 后调用 `torch.cuda.synchronize()`，确保测到 GPU 真正完成计算的时间。

Part (a) 的交付物是一个可以通过命令行配置并重复运行的 benchmark harness。

### 1.2 Part (b)：测试不同规模的模型

题目给出 small、medium、large、xl、10B 五种模型配置，要求在统一条件下测试 forward、backward 和 optimizer step：

- vocabulary size 为 10,000；
- batch size 为 4；
- context length 为 512；
- 5 次 warm-up；
- 10 次正式 measurements；
- 计算平均时间和标准差；
- 分析不同规模模型的耗时与测量稳定性。

### 1.3 Part (c)：研究 warm-up 的作用

题目要求重新执行实验，但不做 warm-up，并与使用 1、2、5 次 warm-up 的结果比较。需要解释：

- 为什么没有 warm-up 时结果会不同；
- 为什么前几个 step 往往更慢；
- 为什么 1–2 个 warm-up steps 有时仍不足以得到稳定结果。

## 2. 我的实现

实现文件是 `cs336_systems/benchmark.py`。

### 2.1 模型配置

代码中的 `MODEL_CONFIGS` 定义了题目指定的五种模型：

| model | `d_model` | `d_ff` | `num_layers` | `num_heads` |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

脚本通过 `--model-size` 选择一组配置，并用 `transformer_lm` 初始化模型。模型直接在目标 device 和 dtype 上创建，避免先在 CPU 构造大型模型再复制到 GPU。

### 2.2 随机数据

脚本在计时前生成形状为 `[batch_size, context_length]` 的随机 input tokens 和 targets：

```python
inputs = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, context_length),
    device=device,
    dtype=torch.long,
)
```

数据生成不在正式计时区间内，因此结果反映的是模型 step，而不是随机数生成或数据加载时间。

### 2.3 三种运行模式

`run_step` 支持：

```text
forward:
    logits = model(inputs)

forward_backward:
    清除 gradient
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()

full_step:
    清除 gradient
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
```

需要 backward 的模式在每一步开始前调用：

```python
model.zero_grad(set_to_none=True)
```

这是必要的，因为 PyTorch 默认会把新 gradient 累加到已有 `.grad` 中。`set_to_none=True` 让下一次 backward 重新创建 gradient，而不是先将旧 gradient tensor 全部写零。

### 2.4 Warm-up 和正式计时

Warm-up 执行与正式 measurement 完全相同的 mode：

```python
for _ in range(warmup_steps):
    run_step(model, inputs, targets, mode, optimizer)
    synchronize(device)
```

正式测量的边界为：

```python
synchronize(device)
start = default_timer()

run_step(model, inputs, targets, mode, optimizer)

synchronize(device)
end = default_timer()
```

计时前同步可以防止之前未完成的 CUDA 工作混入当前 sample；计时后同步可以保证本次 GPU 工作已经完成，CPU 才停止计时。

### 2.5 统计量

脚本记录 10 个 wall-clock samples，并计算平均值：

$$
\bar{t}=\frac{1}{n}\sum_{i=1}^{n}t_i
$$

代码使用 `numpy.std()` 的默认设置，即 `ddof=0`，所以报告的是 population standard deviation：

$$
\sigma=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(t_i-\bar{t})^2}
$$

## 3. 实验环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU 数量 | 1 |
| GPU 显存 | 31.36 GiB 可寻址容量；`nvidia-smi` 标称 32,607 MiB |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| dtype | `torch.float32` |
| vocabulary size | 10,000 |
| batch size | 4 |
| context length | 512 |
| measurement steps | 10 |

每个 `model × mode` 组合都在独立进程中重新初始化模型和 optimizer，并在同一张 GPU 上顺序运行，没有并发 benchmark 干扰。

## 4. 实验结果：Part (b)

Part (b) 使用 5 次 warm-up 和 10 次 measurements。所有时间单位均为 ms，表格格式为 `mean ± std`。

### 4.1 脚本直接测得的累计时间

| model | forward $T_F$ | forward + backward $T_{F+B}$ | full step $T_{F+B+O}$ |
| --- | ---: | ---: | ---: |
| small | 16.292 ± 0.030 | 56.257 ± 2.220 | 69.323 ± 4.909 |
| medium | 47.073 ± 0.352 | 152.292 ± 0.129 | 162.910 ± 0.419 |
| large | 109.847 ± 0.172 | 337.127 ± 0.724 | 362.778 ± 0.479 |
| xl | OOM | OOM | OOM |
| 10B | OOM | OOM | OOM |

这里的 $O$ 表示 optimizer step。当前脚本直接测量的是三个端到端累计区间，而不是在同一个 training step 内独立测量 forward、backward 和 optimizer。

### 4.2 阶段时间的近似估计

可以用不同 mode 的平均值之差近似估计 backward 和 optimizer 时间：

$$
T_B \approx T_{F+B}-T_F
$$

$$
T_O \approx T_{F+B+O}-T_{F+B}
$$

结果为：

| model | forward | backward（近似） | optimizer（近似） |
| --- | ---: | ---: | ---: |
| small | 16.292 | 39.965 | 13.066 |
| medium | 47.073 | 105.219 | 10.618 |
| large | 109.847 | 227.280 | 25.651 |
| xl | OOM | OOM | OOM |
| 10B | OOM | OOM | OOM |

这些差分值来自不同进程中的独立 benchmark run，因此不是严格的 isolated phase measurements。不同 run 之间的 GPU 动态频率、温度和 allocator 状态都可能影响差分结果；脚本直接输出的累计时间更加可信。

### 4.3 模型规模与运行时间

从 small 扩展到 medium 和 large，forward latency 从 16.292 ms 增长到 47.073 ms 和 109.847 ms，full-step latency 从 69.323 ms 增长到 162.910 ms 和 362.778 ms。模型变宽、FFN 维度增大且层数增加后，矩阵乘法数量与规模都在增长，所以运行时间随模型规模明显增加。

由差分估计，backward 大约是 forward 的：

| model | 近似 $T_B/T_F$ |
| --- | ---: |
| small | 2.45 |
| medium | 2.24 |
| large | 2.07 |

Backward 比 forward 慢是合理的：forward 主要计算 activation，而 backward 既要计算对输入的 gradient，也要计算对参数的 gradient，通常涉及更多矩阵乘法和内存访问。不过这里的比例只是基于独立 run 差分得到的观察值，不应视为普遍不变的理论常数。

### 4.4 测量稳定性

用变异系数衡量标准差相对 mean 的大小：

$$
\mathrm{CV}=\frac{\sigma}{\bar{t}}\times100\%
$$

Part (b) 的 CV 为：

| model | forward CV | forward + backward CV | full-step CV |
| --- | ---: | ---: | ---: |
| small | 0.18% | 3.95% | 7.08% |
| medium | 0.75% | 0.08% | 0.26% |
| large | 0.16% | 0.21% | 0.13% |

Medium 和 large 的结果非常稳定。Small 的 forward 同样稳定，但两个训练模式的相对波动更明显。短 workload 更容易受到 GPU 动态频率、系统调度和固定开销影响，因此同样大小的绝对扰动会表现为更大的相对误差。

### 4.5 OOM 分析

`xl` 可以完成模型初始化，但在第一次 training-style forward 中 OOM。失败时进程已使用约 31.33 GiB GPU memory，只剩约 17.69 MiB，接下来申请 20 MiB 失败。

`10B` 在模型初始化阶段 OOM。失败时进程已使用约 31.28 GiB GPU memory，接下来创建一个 Linear parameter 所需的 82 MiB allocation 失败。

二者的 OOM 原因不同：

- `xl`：模型参数可以放入显存，但参数加上 forward 中 autograd 保存的 activation、attention 中间量和临时 buffer 超过显存；
- `10B`：FP32 模型参数本身就接近或超过单卡容量，尚未开始 forward 就无法完成初始化。

一个 FP32 参数占 4 bytes。若模型约有 10 billion parameters，仅参数理论上就需要：

$$
10\times10^9\times4\text{ bytes}
=40\times10^9\text{ bytes}
\approx37.25\text{ GiB}
$$

这已经超过当前 GPU 的 31.36 GiB。若进行 AdamW 训练，还需要为每个参数保存 gradient、一阶矩和二阶矩。忽略 activation 和临时 buffer，仅这些持久数据大致需要：

$$
4+4+4+4=16\text{ bytes/parameter}
$$

对 10B 参数即约 149 GiB，因此普通单卡 FP32 AdamW 训练远超本次硬件容量。

需要特别注意：当前 `forward` mode 没有使用 `torch.no_grad()`，所以即使选择 `--mode forward`，PyTorch 仍会构建 autograd graph 并保存 backward 所需的 residuals。这里测的是 training-style forward，而不是 inference forward，这也是 `xl` 的 forward-only mode 仍然 OOM 的关键原因。

## 5. 实验结果：Part (c)

Part (c) 保持模型、dtype、batch size、context length、mode 和 measurement steps 不变，只将 warm-up steps 设置为 0、1、2、5。由于 `xl` 和 `10B` 的显存需求不会因减少 warm-up 次数而降低，warm-up 对照只测试能够运行的 small、medium 和 large。

### 5.1 Small

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 43.274 ± 80.897 | 86.214 ± 100.098 | 97.448 ± 105.763 |
| 1 | 16.398 ± 0.040 | 53.292 ± 1.696 | 55.187 ± 0.194 |
| 2 | 16.371 ± 0.024 | 54.668 ± 2.414 | 65.424 ± 7.944 |
| 5 | 16.292 ± 0.030 | 56.257 ± 2.220 | 69.323 ± 4.909 |

### 5.2 Medium

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 72.196 ± 76.533 | 183.317 ± 94.105 | 196.996 ± 100.924 |
| 1 | 46.819 ± 0.415 | 152.195 ± 0.912 | 163.000 ± 0.198 |
| 2 | 46.854 ± 0.370 | 152.722 ± 0.722 | 163.002 ± 0.169 |
| 5 | 47.073 ± 0.352 | 152.292 ± 0.129 | 162.910 ± 0.419 |

### 5.3 Large

| warm-up steps | forward | forward + backward | full step |
| ---: | ---: | ---: | ---: |
| 0 | 133.985 ± 72.362 | 364.738 ± 87.297 | 392.344 ± 89.637 |
| 1 | 110.262 ± 1.706 | 336.272 ± 1.312 | 363.430 ± 0.300 |
| 2 | 109.943 ± 0.334 | 336.895 ± 0.514 | 363.856 ± 0.276 |
| 5 | 109.847 ± 0.172 | 337.127 ± 0.724 | 362.778 ± 0.479 |

### 5.4 没有 warm-up 时发生了什么

0 次 warm-up 时，所有可运行配置的 mean 都明显高于 steady-state latency，std 更达到约 72–106 ms。例如 small forward 从 5 次 warm-up 时的 16.292 ± 0.030 ms 变成 43.274 ± 80.897 ms。

原因是 10 个正式 samples 中的第一个 sample 包含一次性或冷启动成本，例如：

- 尚未由模型构造过程触发的 CUDA library 初始化；
- kernel/module lazy loading；
- cuBLAS 等底层库为当前 shape 建立执行路径；
- PyTorch caching allocator 首次为中间 tensor 申请显存；
- cache 尚未进入稳定状态；
- full-step 中 AdamW 首次创建一阶矩和二阶矩状态。

一个极慢的首个 sample 会同时抬高平均值和标准差。由于一共只有 10 个 samples，单个 outlier 的影响尤其明显。

### 5.5 1、2、5 次 warm-up 的比较

在本次 RTX 5090 实验中，1 次 warm-up 后多数配置已经接近 steady state，medium 和 large 的结果尤其稳定。2–5 次 warm-up 可以进一步确保冷启动工作被排除，并降低部分配置的波动。

1–2 次 warm-up 并不在所有情况下都必然足够，因为初始化可能分布在不同代码路径中。例如：

- forward warm-up 不会初始化 AdamW state；
- 更换 shape 或 dtype 可能触发新的底层执行路径；
- `torch.compile` 可能需要图捕获和编译；
- Triton kernel 可能在首次遇到某个配置时进行 JIT compilation；
- 某些 library cache 或内存池需要多个 iteration 才进入稳定状态。

Small full-step 在 1、2、5 次 warm-up 的均值并不单调，而且部分 run 的 std 仍偏大。这不意味着“更多 warm-up 会让模型变慢”，而是说明不同独立进程的 GPU 动态频率、温度、系统噪声和 allocator 状态仍会影响较短 workload。Warm-up 的目标是排除系统性冷启动成本，而不是保证每次独立运行得到完全相同的数字。

## 6. 这道题涉及的核心知识点

### 6.1 GPU 操作的异步执行

调用 CUDA 版 PyTorch operation 时，CPU 通常只是把工作提交到 GPU stream，然后立即继续执行 Python。若直接这样计时：

```python
start = default_timer()
logits = model(inputs)
end = default_timer()
```

测到的主要是 CPU 发射 kernel 的时间，而不是 GPU 完成计算的时间。

因此 wall-clock benchmark 必须在边界同步：

```text
synchronize -> start -> GPU work -> synchronize -> end
```

这是本题最重要的知识点之一：GPU benchmark 的计时方法必须符合 GPU 的异步执行模型。

### 6.2 `torch.cuda.synchronize()` 的位置

计时前同步的作用是确保之前的 GPU 工作已完成；计时后同步的作用是等待本次工作完成。Warm-up 每一步后也同步，可以保证所有 warm-up 工作真正完成后才开始正式计时。

如果测量完整 training step，只应在整个 step 的外部同步。若在 forward、backward、optimizer 之间反复同步，会引入额外开销，并改变它们自然的异步调度行为。

只有当实验目标确实是测量各阶段 latency 时，才应在阶段边界同步；此时得到的阶段时间之和可能不等于端到端 full-step 时间。

### 6.3 Warm-up 与 steady state

Benchmark 应尽量测量 steady-state performance，而不是程序第一次执行的冷启动性能。Warm-up 的本质是：在正式计时之前，使用与 measurement 相同的 workload 先运行若干次，让只在首次或前几次执行中出现的额外工作提前完成。

可以把第 $i$ 次 step 的时间粗略表示为：

$$
T_i=T_{\text{steady}}+T_{\text{cold},i}+\epsilon_i
$$

其中：

- $T_{\text{steady}}$ 是进入稳定状态后每个 step 的正常执行时间；
- $T_{\text{cold},i}$ 是第 $i$ 次执行中特有的初始化、分配或缓存建立成本，通常只在前几次非零；
- $\epsilon_i$ 是 GPU 动态频率、系统调度和温度等造成的随机噪声。

Warm-up 不会改变模型的理论 FLOPs，也不会从根本上让矩阵乘法变快。它只是让正式 measurement 尽量满足：

$$
T_{\text{cold},i}\approx0
$$

从而使测量值主要反映 $T_{\text{steady}}$。

#### 6.3.1 CUDA runtime 和 context 初始化

一个进程第一次真正使用 CUDA 时，可能需要建立 CUDA context、加载 driver/runtime 组件，并初始化与 GPU 的通信状态。这些工作通常只发生一次，可能远慢于普通 kernel launch。

在本脚本中，模型和随机 tensor 直接在 CUDA device 上创建，所以部分 CUDA 初始化可能已经发生在 benchmark loop 之前；但模型第一次执行仍可能触发尚未访问的 CUDA 或数学库路径。因此，不能简单认为“模型已经在 GPU 上创建”就等于所有冷启动工作都已完成。

#### 6.3.2 CUDA library 和 kernel 的 lazy loading

PyTorch 的许多 operation 会调用 cuBLAS、cuDNN 或其他 CUDA library。这些库可能在第一次遇到某个 operation、shape、layout 或 dtype 时才完成：

- 动态库或 kernel module 加载；
- handle 和内部状态初始化；
- 为特定 tensor shape 选择算法；
- 创建临时 workspace；
- 建立内部 cache。

因此，即使第二个 step 与第一个 step 执行相同的 Python 代码，第一个 step 也可能额外承担初始化成本。

#### 6.3.3 PyTorch caching allocator

Forward 和 backward 会创建许多临时 tensor。第一次运行时，PyTorch CUDA allocator 可能需要向 CUDA driver 申请新的显存块。Tensor 被释放后，PyTorch 通常不会立刻把所有显存还给 driver，而是将它保留在 caching allocator 中，供后续相同或相近大小的 allocation 复用。

于是常见情况是：

```text
第一个 step：向 driver 申请显存块 + 正常计算
后续 step：复用缓存的显存块 + 正常计算
```

Warm-up 可以把首次 allocation 移到正式计时之外。这里的 cache 是显存分配器缓存，不应与 GPU 的 L1/L2 hardware cache 混为一谈。

#### 6.3.4 Autograd 的首次执行路径

`forward_backward` 和 `full_step` 会运行 autograd engine。第一次 backward 可能触发之前没有执行过的 backward kernels、内部数据结构或 library 路径。

因此只运行 forward warm-up，不能充分预热 backward benchmark。Warm-up 的 mode 必须和正式 measurement 一致：

```text
测 forward              -> warm-up forward
测 forward + backward   -> warm-up forward + backward
测 full step            -> warm-up完整 full step
```

#### 6.3.5 AdamW optimizer state 的惰性初始化

AdamW 通常不会在 optimizer 构造时立刻为全部参数创建一阶矩和二阶矩，而是在某个参数第一次具有 gradient 并执行 `optimizer.step()` 时创建：

$$
m_0=0,
\qquad
v_0=0
$$

这意味着第一个 full-step 还可能包含：

- 为 $m$ 和 $v$ 分配显存；
- 将状态 tensor 初始化为零；
- 初始化 per-parameter step state；
- 第一次执行 optimizer 的逐元素 CUDA kernels。

如果 full-step 不做 warm-up，这些一次性成本会被错误地当成每个正常 optimizer step 都需要支付的时间。

#### 6.3.6 编译和 autotuning

当前脚本没有使用 `torch.compile` 或 Triton JIT，但后续实验会遇到更明显的冷启动：

- `torch.compile` 需要 graph capture、lowering、code generation 和 compilation；
- Triton 需要为新的 shape、dtype 和 meta-parameters 编译 kernel；
- autotuning 可能试运行多个 kernel configuration，再缓存最快方案。

这类编译成本可能远大于普通 step，所以 1–2 次 warm-up 未必足够。还必须确认正式实验是否应该报告“包含编译的首次运行时间”还是“编译完成后的 steady-state 时间”，二者代表不同使用场景。

#### 6.3.7 GPU 动态频率和温度状态

空闲 GPU 可能处于较低功耗或较低时钟状态，负载开始后再提升频率。连续运行一段时间后，GPU 的频率、功耗和温度趋于新的工作状态。

Warm-up 能让 GPU 更接近实际持续负载下的状态，但它无法完全消除：

- 动态频率变化；
- 温度波动；
- 其他进程争抢 GPU；
- 操作系统调度噪声。

因此做了 warm-up 后仍然需要多次 measurement，并同时报告 mean 和 std。

#### 6.3.8 为什么 warm-up 后必须同步

CUDA kernel 默认异步执行。如果只从 Python 循环中发射若干 warm-up steps，却不在进入正式计时前等待 GPU 完成，那么 CPU 可能已经开始 measurement，而 GPU 仍在处理 warm-up workload。此时 warm-up 工作会混入第一个正式 sample。

本脚本在每个 warm-up step 后执行：

```python
synchronize(device)
```

从而保证该 warm-up step 已经在 GPU 上真正完成。正式 measurement 开始前再次同步，则进一步保证计时起点没有遗留 CUDA 工作。

#### 6.3.9 Warm-up 必须匹配正式 workload

Warm-up 必须与正式测量使用相同的：

- mode；
- tensor shape；
- dtype；
- device；
- 代码路径。

还应尽量保持相同的：

- causal mask 或其他模型分支；
- 是否启用 autocast；
- 是否启用 `torch.compile`；
- optimizer 类型；
- requires-grad 状态。

原因是许多 cache、kernel 和编译结果都与 shape、dtype 或代码分支相关。用 FP32、context length 512 完成 warm-up，不能保证 BF16、context length 2048 的执行路径也已经预热。

#### 6.3.10 应该做多少次 warm-up

不存在对所有模型和硬件都最优的固定次数。题目规定 5 次，是一个简单且通常足够的实验协议。更一般的做法是观察逐 step latency：当前几个 sample 明显偏高、之后进入稳定平台时，说明冷启动成本已经基本消失。

可以用以下原则选择：

1. 至少覆盖每一种正式 measurement mode；
2. 覆盖 optimizer state 初始化；
3. 若使用 compile/JIT，等待编译与 autotuning 完成；
4. 连续若干 step 的 latency 和显存状态基本稳定；
5. 固定次数后，对所有对照组使用相同实验协议。

Warm-up 次数也不是越多越好。过多 warm-up 会浪费实验时间，还可能改变 GPU 温度和频率状态。目标是进入有代表性的 steady state，而不是无限运行。

#### 6.3.11 Warm-up 不应该做什么

Warm-up samples 不应加入正式 mean/std，否则冷启动数据仍会污染统计结果。也不应在 warm-up 和 measurement 之间改变模型、shape、dtype 或 mode，否则之前的预热可能失效。

对于训练 benchmark，warm-up 中的 `optimizer.step()` 会真实更新模型参数。本题只测速度、使用随机权重和随机数据，因此参数变化不影响任务目标；但在研究 loss、收敛或数值正确性时，必须明确区分“性能 warm-up”和“真实训练 step”，必要时保存并恢复模型与 optimizer state。

### 6.4 Wall-clock latency 与 GPU kernel time

本题使用同步后的 wall-clock latency，它表示用户真正等待一个 step 完成的时间。它可能包含：

- Python dispatch；
- CUDA kernel launch；
- GPU kernel execution；
- 必要的同步；
- allocator 和 optimizer 工作。

它不能告诉我们时间具体花在哪个 kernel 上。要分析 kernel 分布，需要后续的 Nsight Systems profiling。

### 6.5 Benchmark measurement scope

任何 benchmark 都必须先明确“计时区间包含什么”。当前三个 mode 分别测量：

- model forward；
- model forward、loss、backward；
- gradient clearing、model forward、loss、backward、optimizer step。

只有清楚 measurement scope，不同数字之间的比较才有意义。尤其不能把累计时间直接称为 isolated backward latency。

### 6.6 Gradient accumulation 和 `zero_grad`

PyTorch 默认执行：

$$
g_{\text{stored}}\leftarrow g_{\text{stored}}+g_{\text{new}}
$$

如果 benchmark 中不清 gradient，每个 step 的训练语义和内存状态都会改变。使用 `zero_grad(set_to_none=True)` 可以保证每次 measurement 都近似代表一个独立 training step。

### 6.7 AdamW state 的惰性初始化

AdamW 通常在第一次 `optimizer.step()` 时为每个参数创建一阶矩和二阶矩。这会产生大量显存 allocation 和初始化操作，所以 full-step benchmark 尤其需要同 mode 的 warm-up。

### 6.8 Training-style forward 与 inference forward

是否调用 backward 不是决定 forward 显存占用的唯一因素。只要 autograd 开启且参数需要 gradient，普通 forward 就会构建 computation graph 并保存 residuals。

```python
logits = model(inputs)
```

是 training-style forward；而：

```python
with torch.no_grad():
    logits = model(inputs)
```

才是不保存 backward graph 的 inference-style forward。两者的时间和显存含义不同，不能不加说明地混在同一张表中比较。

### 6.9 模型参数、activation、gradient 和 optimizer state

训练显存不只由模型参数决定。峰值显存通常包括：

- parameters；
- forward 保存的 activations/residuals；
- gradients；
- optimizer states；
- temporary workspaces 和 allocator reserve。

FP32 AdamW 中，仅持久的 parameter-related 数据就可能达到约 16 bytes/parameter。Activation 则随 batch size、context length、hidden size 和 layer 数增长；朴素 attention 的某些中间 tensor 还会随 context length 的平方增长。

### 6.10 OOM 与计算速度是两个不同维度

一个 GPU 可以拥有很高的计算吞吐量，但显存容量仍不足以容纳某个模型。当前 RTX 5090 能较快完成 large full-step，但 `xl` 和 `10B` 仍然 OOM，说明：

- 速度取决于计算吞吐量、显存带宽和 kernel 效率；
- 能否运行取决于峰值显存需求是否小于可用显存。

“算得快”不等于“装得下”。

### 6.11 Mean、standard deviation 和离群值

只报告一次运行时间无法判断结果是否可靠。Mean 表示中心水平，std 表示样本波动；CV 可以比较不同量级实验的相对稳定性。

当 std 很大时，应检查：

- 是否遗漏 warm-up；
- 是否遗漏 synchronize；
- GPU 是否有其他任务；
- 是否存在首次 allocation；
- workload 是否太短；
- GPU 是否发生动态频率或温度变化。

### 6.12 控制变量和可复现性

比较模型或 warm-up 次数时，只能改变目标变量，其余条件应保持一致。实验至少应记录：

- GPU 型号和显存；
- PyTorch/CUDA 版本；
- dtype；
- batch size 和 context length；
- model config；
- warm-up 和 measurement 次数；
- measurement scope；
- OOM 或其他失败状态。

否则 timing 数字脱离环境后几乎无法解释或复现。

### 6.13 FP32、BF16 与 mixed precision 的区别

当前实验使用 FP32。脚本的 `--dtype bfloat16` 会直接以 BF16 创建模型参数，这不等同于后续题目要求的 `torch.autocast` mixed precision。Mixed precision 通常保留 FP32 master parameters 或在数值敏感 operation 中使用较高精度，而将适合 Tensor Core 的计算转换到 BF16/FP16。

因此，当前的 dtype 参数虽然方便做其他实验，但不能直接当作 mixed-precision implementation。

## 7. 我应该从这道题学到什么

### 7.1 正确测量比得到一个数字更重要

如果不知道 CUDA 是异步的，没有同步就会得到看似漂亮但没有意义的结果。一个可信 benchmark 首先要有正确的 measurement boundary，其次才是最终数值。

### 7.2 性能实验必须先定义口径

应该能够明确回答：

- 测的是 inference forward 还是 training forward？
- loss 是否包含在 backward mode 中？
- gradient clearing 是否包含在 full-step 中？
- 报告的是累计 latency 还是 isolated phase latency？

如果这些问题没有答案，就不能可靠地解释表格。

### 7.3 Warm-up 是实验设计的一部分

Warm-up 不是为了让 GPU “计算得更快”，而是为了把一次性冷启动成本与 steady-state execution 分开。本次实验中 0 次 warm-up 导致巨大的 std，直观验证了这一点。

### 7.4 平均值必须配合方差理解

只看 mean 会隐藏不稳定性。Small full-step 的 CV 明显高于 medium 和 large，说明即使平均值看起来合理，也应继续关注单次 samples 和系统噪声。

### 7.5 OOM 也是有价值的实验结果

OOM 不是应当隐藏的“失败数据”。它揭示了当前硬件、precision、模型结构和 autograd 策略下的可运行边界：

- `xl` 的瓶颈是 training activation 加参数后的峰值显存；
- `10B` 的瓶颈已经是 FP32 参数本身。

这为后续 mixed precision、FlashAttention、activation checkpointing、optimizer state sharding 和 FSDP 提供了明确动机。

### 7.6 Latency 不能直接证明实现已经高效

Large full-step 约为 362.778 ms，表面上很快，但仅凭 latency 不能判断 GPU 利用率。还需要结合理论 FLOPs、achieved FLOP/s、显存带宽和 profiler kernel breakdown，才能判断实现是否接近硬件能力上限。

### 7.7 Benchmark 应自动化并保存原始数据

手动运行大量配置容易遗漏或抄错。一个成熟的 benchmark harness 应该自动遍历配置，并保存：

- 每个原始 sample；
- mean/std；
- 环境信息；
- 成功或 OOM 状态。

当前脚本已经具备核心功能，但以后可以进一步增加 JSONL/CSV 输出和自动 sweep，使后续 mixed precision、`torch.compile` 与 profiling 实验更容易复用。

## 8. 对当前实现的评价与后续改进方向

### 8.1 已完成的部分

- 根据题目配置初始化模型；
- 在 GPU 上生成随机数据；
- 支持三种 mode；
- 支持可配置 warm-up 和 measurement 次数；
- 使用 `default_timer()`；
- 在正确的计时边界调用 CUDA synchronize；
- 正确清除 gradient；
- 输出 mean 和 std；
- 完成 Part (b) 和 Part (c) 的真实 GPU 实验。

### 8.2 可以继续改进的部分

1. 保存每个原始 measurement sample，而不是只打印 mean/std；
2. 自动遍历所有 model 和 mode，并将结果写入 JSONL 或 CSV；
3. 如果题目需要严格的阶段时间，在同一个 step 内单独测量 forward、backward 和 optimizer；
4. 明确提供 training forward 与 inference forward 两种模式；
5. 输出 GPU 型号、PyTorch/CUDA 版本和随机种子；
6. 捕获 OOM 并结构化记录，而不是只留下 traceback；
7. 删除与 benchmark 无关的 unused imports，使脚本职责更清楚；
8. 后续 mixed precision 实验应使用 `torch.autocast`，而不是仅改变模型参数 dtype。

## 9. 可用于作业 writeup 的精简答案

### Part (b)

在 RTX 5090 上，small、medium、large 的 forward pass 分别耗时 16.292、47.073、109.847 ms；由累计模式差分估计，backward 分别约耗时 39.965、105.219、227.280 ms。充分 warm-up 后大多数配置的标准差相对 mean 很小，但 small 的训练模式波动稍明显；xl 在第一次 training-style forward 时 OOM，10B 在 FP32 模型初始化时 OOM。

### Part (c)

没有 warm-up 时，首次 step 的底层库/kernel lazy loading、首次中间 tensor allocation 以及 optimizer state 初始化被计入正式测量，使 mean 明显升高，并让 std 达到约 72–106 ms。1 次 warm-up 在本机多数配置上已接近 steady state，2–5 次可进一步排除初始化和缓存建立的影响；但不同 shape、dtype、编译或 JIT 路径可能需要更多 warm-up，因此 1–2 次并不总是足够。

## 10. 复现实验命令

Part (b) 的单个配置示例：

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

Part (c) 保持其余参数不变，将 `--warmup-steps` 分别设置为 `0`、`1`、`2` 和 `5`。
