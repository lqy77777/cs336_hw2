# `benchmarking_script` 实现手册

> 对应 CS336 Assignment 2 的第一个任务：`Problem (benchmarking_script): Benchmarking Script (4 points)`。
>
> 适合读者：完成了作业 1，具备 Python/PyTorch 基础，但刚开始接触 GPU benchmark。
>
> 本手册解释任务、工具、知识点、脚本结构和实验方法，不提供可以直接提交的完整实现。你应根据自己的作业 1 模型接口完成代码。

## 1. 这个任务到底要解决什么问题

这个任务不是让你训练出一个好模型，而是建立一个之后可以反复复用的性能测量工具。

你需要回答三类问题：

1. Transformer 的 forward、backward 和 optimizer step 分别需要多长时间？
2. 不同模型规模下，时间如何变化，测量是否稳定？
3. 如果不做 warm-up，为什么结果会变化？

后面的 mixed precision、`torch.compile`、Nsight、DDP 等实验都会继续使用或扩展这个脚本。因此，与其把它写成只能运行一次的临时代码，更合理的目标是：

```text
一个命令行可配置、输出可保存、测量边界清楚的 benchmark harness
```

其中 benchmark harness 可以翻译为“基准测试框架”或“基准测试脚手架”。

---

## 2. 原题背景完整翻译

### 2.1 Profiling and Benchmarking

在作业的第一部分，我们将探索如何优化 Transformer 模型的性能，从而尽可能高效地使用 GPU。我们会分析模型，以了解它在 forward 和 backward 过程中把时间与内存花在了哪里；随后，我们会使用自定义 GPU kernel 优化 self-attention，使它比普通 PyTorch 实现更快。在作业后面的部分，我们将使用多张 GPU，并理解如何在一个 cluster 上训练模型。

### 2.2 Profiling

在实现任何优化之前，最好先分析程序，理解它把资源（例如时间和内存）花在哪里。否则，我们可能会优化那些并未占用大量时间或内存的部分，最终看不到可测量的端到端性能提升。

本作业将实现三条性能评估路径：

1. 使用 Python 标准库进行简单的端到端 benchmark，对 forward 和 backward 计时；
2. 使用 NVIDIA Nsight Systems 进行计算分析，理解时间如何分布在 CPU 和 GPU 的不同操作上；
3. 进行内存分析。

### 2.3 Setup - 导入作业 1 的 Transformer

首先确认能够加载上一次作业中的模型。作业 1 已经把模型组织成 Python package，因此可以在后续作业中方便地导入。

作业 2 提供了 `./cs336-basics`，其中包含作业 1 的参考实现，外层 `pyproject.toml` 已指向这个本地 package。照常运行 `uv run [command]` 时，`uv` 会自动找到本地的 `cs336-basics`。

如果希望使用自己的作业 1 实现，可以修改作业 2 根目录的 `pyproject.toml`，让它指向自己的 package。

可以用下面的方式检查导入：

```bash
uv run python
```

然后在 Python 中：

```python
import cs336_basics
```

作业 1 中的相关模块应该可以被正常导入。

### 2.4 Model Sizing

整个作业会对不同规模的模型进行 benchmark 和 profiling，以理解模型扩展后性能如何变化。

除 leaderboard 外，统一使用：

- vocabulary size：10,000；
- batch size：4；
- context length：除非另行指定，否则为 512。

模型配置如下：

| Size | `d_model` | `d_ff` | `num_layers` | `num_heads` |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

讲义还建议用代码自动生成 writeup 中的表格，而不是手动在 LaTeX 或 Typst 中抄写结果。

### 2.5 End-to-End Benchmarking

现在要实现一个简单的性能评估脚本。后续会测试模型的许多变体，例如改变精度或替换 layer，因此最好通过 command-line arguments 控制这些变体，方便重复运行。

首先进行最简单的 profiling：测量模型的 forward pass、backward pass 和 optimizer step。由于当前只测量速度与内存，可以使用随机权重和随机数据。

性能测量存在一些容易踩中的陷阱。GPU benchmark 的一个关键问题是 CUDA 调用是异步的。当调用 `torch.matmul` 等 CUDA kernel 时，PyTorch 函数会在矩阵乘法真正完成之前把控制权返回给 Python。这样 CPU 可以继续调度新操作，而 GPU 同时完成矩阵乘法，这对性能非常重要。

但这也意味着：如果只测量 `torch.matmul` 这个 Python 调用多久返回，测到的并不是 GPU 真正执行矩阵乘法的时间。

PyTorch 提供 `torch.cuda.synchronize()`，用于等待所有已经调度的 GPU kernel 完成，从而得到更准确的 CUDA kernel runtime。这里的 synchronize 指让 CPU runtime 与 GPU runtime 同步。

---

## 3. `benchmarking_script` 原题完整翻译

### Problem: Benchmarking Script（4 分）

#### (a) 编写基准测试脚本

编写一个脚本，对模型的 forward pass、backward pass 和 optimizer step 进行基本的端到端 benchmark。

脚本需要支持：

1. 根据给定的 hyperparameters（例如 layer 数量）初始化模型；
2. 生成一批随机数据；
3. 首先运行 $w$ 个 warm-up steps，这些 step 不计入正式测量；随后测量 $n$ 个 steps 的执行时间；
4. 根据命令行参数，能够测量以下三种模式：
   - 只执行 forward；
   - 执行 forward 和 backward；
   - 执行 forward、backward 和 optimizer step；
5. 可以使用 Python 的 `timeit` 模块计时，例如 `timeit.timeit()` 或 `timeit.default_timer()`；
6. `timeit.default_timer()` 使用系统可用的最高分辨率计时器，因此通常比 `time.time()` 更适合 benchmark；
7. 每个 step 后调用 `torch.cuda.synchronize()`。

交付内容：一个脚本，它能够：

- 使用给定 hyperparameters 初始化 basics Transformer；
- 创建随机 batch；
- 测量 forward-only；
- 测量 forward-and-backward；
- 测量包含 optimizer step 的完整 training step。

#### (b) 测量不同模型规模

对第 2.1.2 节中给出的模型规模进行 forward、backward 和 optimizer step 计时。

实验要求：

- 使用 5 个 warm-up steps；
- 使用 10 个 measurement steps；
- 计算测量时间的平均值；
- 计算测量时间的标准差；
- 回答 forward pass 需要多长时间；
- 回答 backward pass 需要多长时间；
- 判断不同测量之间的波动是否很大，或者标准差是否较小。

交付内容：用 1-2 句话报告时间结果。

#### (c) 研究 warm-up 的影响

Benchmark 的一个常见问题是不执行 warm-up。

需要：

1. 在完全不做 warm-up 的情况下重复前面的分析；
2. 说明结果发生了什么变化；
3. 解释为什么会发生这种变化；
4. 分别尝试只做 1 个和 2 个 warm-up steps；
5. 解释为什么 1-2 个 warm-up 后的结果仍可能与充分 warm-up 的结果不同。

交付内容：用 2-3 句话回答。

---

## 4. 把题目转换成工程需求

题目文字可以转换为以下脚本需求。

### 4.1 输入

脚本至少需要接收：

- 模型 hyperparameters；
- benchmark mode；
- warm-up step 数 $w$；
- measurement step 数 $n$；
- device；
- batch size；
- context length；
- vocabulary size。

建议额外预留：

- dtype；
- random seed；
- 是否启用 `torch.compile`；
- 输出文件路径；
- 模型规模名称。

这些额外参数不是第一题的硬性要求，但后续 mixed precision 和 compile 实验会用到。现在设计好，可以避免反复重写脚本。

### 4.2 输出

脚本应输出或保存：

- 完整实验配置；
- 每一个 measurement step 的原始时间；
- 平均时间；
- 标准差；
- 成功或 OOM 状态；
- GPU 名称；
- PyTorch/CUDA 版本；
- 时间单位。

不要只保留平均值。原始样本可以用于重新计算统计量和排查异常值。

### 4.3 三种执行模式

建议使用清晰且互斥的名字：

```text
forward
forward_backward
full_step
```

它们对应：

```text
forward:
    forward

forward_backward:
    forward -> scalar objective -> backward

full_step:
    clear gradients -> forward -> scalar objective -> backward -> optimizer step
```

### 4.4 默认配置

第一题的正式 sweep 应使用：

```text
vocab_size = 10_000
batch_size = 4
context_length = 512
warmup_steps = 5
measurement_steps = 10
```

模型规模来自 Table 1。

---

## 5. 推荐使用的工具

### 5.1 `uv`

用途：

- 根据 `pyproject.toml` 建立运行环境；
- 安装本地 `cs336-basics` package；
- 保证脚本在作业环境中运行。

典型运行形式：

```bash
uv run python -m cs336_systems.benchmarking --help
```

如果把脚本放在 `cs336_systems/benchmarking.py`，使用 `python -m` 运行通常比依赖当前工作目录的相对 import 更稳健。

### 5.2 `argparse`

用途：定义 command-line arguments。

第一版至少需要这些参数：

```text
--mode
--model-size
--warmup-steps
--measurement-steps
--batch-size
--context-length
--vocab-size
--device
--seed
--output
```

对于 `--mode`，应该限制合法值，避免用户输入拼写错误后悄悄走错分支。

### 5.3 `dataclasses.dataclass`

用途：把模型 hyperparameters 和 benchmark settings 组织成结构化对象。

建议区分两类配置：

```text
ModelConfig
BenchmarkConfig
```

这样模型结构与测量方式不会混在同一个巨大参数列表中。

### 5.4 PyTorch

需要使用：

- `torch.manual_seed`：设置随机种子；
- `torch.randint`：生成随机 token IDs 和 targets；
- `torch.cuda.synchronize`：同步 CPU/GPU；
- `torch.cuda.get_device_name`：记录 GPU；
- `model.parameters()`：构造 optimizer；
- `loss.backward()`：执行 backward；
- `optimizer.step()`：执行参数更新；
- `optimizer.zero_grad(set_to_none=True)`：清除旧 gradient。

### 5.5 `timeit.default_timer`

用途：获得高分辨率 wall-clock timer。

推荐测量结构：

```python
torch.cuda.synchronize()
start = timeit.default_timer()

# 执行需要测量的 step

torch.cuda.synchronize()
elapsed = timeit.default_timer() - start
```

题目明确要求每个 step 后同步。测量开始前也同步，可以保证计时区间内没有混入前一个 step 尚未完成的 GPU 工作。

### 5.6 `statistics` 或 NumPy

用途：计算平均值和标准差。

Python 标准库已经足够：

```python
statistics.fmean(samples)
statistics.stdev(samples)
```

`statistics.stdev` 计算 sample standard deviation；`statistics.pstdev` 计算 population standard deviation。题目没有明确指定哪一种，因此选择一种、在 writeup 中保持一致即可。更推荐保留原始数据并注明使用哪种定义。

### 5.7 JSON、JSONL 或 CSV

用途：保存实验结果。

推荐两种形式：

- JSONL：每完成一个配置就追加一条结构化记录，长 sweep 中途失败也不会丢掉之前结果；
- CSV：方便用 pandas 直接生成表格。

不要把唯一结果只打印到 terminal。

### 5.8 pandas

用途：整理不同模型和模式的结果，生成 writeup 表格。

推荐数据结构：每个 measurement sample 一行，例如：

| model | mode | warmup | sample | time_ms | status |
| --- | --- | ---: | ---: | ---: | --- |
| small | forward | 5 | 0 | ... | ok |
| small | forward | 5 | 1 | ... | ok |

然后用 `groupby` 计算 mean/std，再导出到 Markdown、LaTeX 或 Typst。

### 5.9 `nvidia-smi`

用途：确认 GPU 型号、数量、显存和是否有其他进程占用 GPU。

它不是脚本计时器，但正式实验前应检查：

```bash
nvidia-smi
```

如果 GPU 与其他用户共享，时间方差可能明显增大。

### 5.10 第一题暂时不需要的工具

第一题不要求使用：

- Nsight Systems：这是紧接着的下一项任务，用于解释时间花在哪里；
- Python `cProfile`：它无法准确表示异步 CUDA kernel 的真实执行时间；
- CUDA Event：可以用于更细粒度的 GPU timing，但第一题已经明确建议 `timeit` 和 `torch.cuda.synchronize()`；
- `torch.utils.benchmark`：是有用的通用工具，但不是完成当前题目的必要条件。

第一版最好先严格完成题目要求，再在后续任务中扩展 profiler 或其他计时后端。

---

## 6. 当前工作区中的模型接口

讲义举例使用：

```python
import cs336_basics.model
```

但你的当前作业一实现位于：

```text
cs336_basics/transformer.py
```

模型类名是：

```text
transformer_lm
```

当前构造参数包括：

```text
vocab_size
context_length
num_layers
d_model
num_heads
d_ff
rope_theta
device
dtype
```

因此，实现前应做的第一件事不是照抄讲义 import，而是确认：

```python
from cs336_basics.transformer import transformer_lm
```

能否在作业 2 环境中正常工作。

还要注意：

- `d_model` 必须能被 `num_heads` 整除；
- input token IDs 应为整数类型；
- 模型输出 shape 应类似 `[batch_size, context_length, vocab_size]`；
- `rope_theta` 虽未出现在 Table 1 中，但你的构造函数要求提供；应使用作业 1 已采用的配置，并在实验记录中固定它；
- 最好直接用 `device="cuda"` 初始化模型，避免先在 CPU 构造巨大模型再复制到 GPU。

如果以后切换回 staff implementation，类名和构造函数可能变化。因此可以把“根据 config 构造模型”的逻辑集中放在一个 factory function 中。

---

## 7. 必须掌握的知识点

### 7.1 Forward pass

Forward pass（前向传播）把 token IDs 输入模型并产生 logits：

```text
token IDs -> embedding -> Transformer blocks -> final norm -> LM head -> logits
```

Forward-only benchmark 要明确测量的是：

- training-style forward：启用 autograd，会保存 backward 所需的 activation；
- inference-style forward：使用 `torch.no_grad()` 或 inference mode，不保存 backward graph。

第一题随后要比较 forward 与 forward+backward。为了让二者语义更一致，建议第一题默认使用 training-style forward，不要在 forward-only 模式中悄悄加 `torch.no_grad()`。如果确实要测 inference，应增加独立参数并明确标记。

在三种 mode 中显式使用一致的 `model.train()` 状态。当前模型可能没有 dropout，但后续替换 layer 后，`train()` 与 `eval()` 可能改变计算行为。

### 7.2 Backward pass

PyTorch 只能从 scalar objective 方便地开始 backward。模型输出 logits 不是标量，因此需要构造一个 loss 或简单标量。

有两种常见选择：

1. 使用作业 1 的 cross-entropy 和随机 targets，更接近真实 training step；
2. 使用 logits 的 sum/mean，逻辑更简单，更接近“只测模型 backward”。

题目没有明确规定 loss，因此关键是：

- 选择一种语义；
- 所有模型配置保持一致；
- 在脚本输出和 writeup 中说明；
- 不要让某种模式包含 loss，而另一种模式不包含，却仍直接比较。

每次 backward 前都需要重新执行 forward。不能对同一个 computation graph 连续调用普通 `backward()`，因为第一次 backward 后中间 graph 通常会被释放。

### 7.3 Gradient accumulation

PyTorch 默认把新 gradient 累加到已有 `.grad`：

$$
g_{stored} \leftarrow g_{stored} + g_{new}
$$

如果 benchmark 循环中不清除 gradient：

- 每一步的状态不一致；
- 可能改变性能和显存；
- full step 的 optimizer 会使用累积 gradient；
- 实验不再代表独立 training steps。

因此每个需要 backward 的 step 前都要清理 gradient。

推荐理解：

```python
optimizer.zero_grad(set_to_none=True)
```

`set_to_none=True` 让 `.grad` 变回 `None`，通常比把整个 gradient tensor 填成 0 更节省写内存。

### 7.4 Optimizer state 的惰性初始化

AdamW 通常在第一次 `optimizer.step()` 时才为 parameter 创建一阶矩和二阶矩。

因此 full-step 模式的第一次执行可能额外包含：

- optimizer state allocation；
- state tensor 初始化；
- CUDA allocator 工作。

这正是 warm-up 非常重要的原因之一。如果 full-step 模式做了足够 warm-up，正式测量通常不会再包含首次 optimizer state 初始化。

不同 benchmark mode 要分别 warm-up。只 warm-up forward 后直接测 full step，不能消除 optimizer 的冷启动成本。

### 7.5 CUDA asynchronous execution

CUDA 操作默认异步排队。下面的代码通常只测到了 enqueue 时间：

```python
start = timeit.default_timer()
logits = model(inputs)
elapsed = timeit.default_timer() - start
```

正确的 wall-clock 测量必须让 CPU 在结束边界等待 GPU：

```python
torch.cuda.synchronize()
start = timeit.default_timer()
logits = model(inputs)
torch.cuda.synchronize()
elapsed = timeit.default_timer() - start
```

### 7.6 Warm-up

Warm-up 用于消除或减小只在前几步出现的成本，例如：

- CUDA context 初始化；
- CUDA library 动态加载；
- cuBLAS 内部初始化；
- PyTorch caching allocator 首次申请显存；
- kernel/module lazy initialization；
- optimizer state 第一次创建；
- cache 冷启动；
- 如果后续启用 `torch.compile`，还包括图捕获与编译；
- 如果后续使用 Triton，还包括 JIT compilation。

Warm-up step 必须执行与正式测量相同的 mode、shape、dtype 和代码路径。

### 7.7 Wall-clock time

Wall-clock time 是现实中从开始到结束经过的时间。它包括计时区间内所有阻塞在 critical path 上的工作。

在这个任务中，应记录每个 step 的 wall-clock time，而不是使用 Python CPU time。

### 7.8 Mean 和 standard deviation

对测量样本 $t_1,\ldots,t_n$，平均时间是：

$$
\bar{t}=\frac{1}{n}\sum_{i=1}^{n}t_i
$$

Sample standard deviation 是：

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(t_i-\bar{t})^2}
$$

Standard deviation 较小，说明多次测量比较稳定；较大时应检查：

- GPU 是否被共享；
- 是否遗漏 warm-up；
- 是否遗漏 synchronize；
- 是否有首次 allocation；
- 是否存在 thermal throttling；
- 是否混入数据生成、文件 I/O 或日志输出；
- 某个 configuration 是否接近显存上限。

### 7.9 OOM

OOM 是 out of memory。大模型可能无法在当前 GPU 上完成某些 mode，尤其 full step 需要同时存放：

- parameters；
- activations saved for backward；
- gradients；
- optimizer states；
- temporary buffers。

不要把 OOM 当成脚本失败后直接丢弃。应把它记录成实验结果：

```text
status = OOM
```

但也不要在每一个 measurement step 中调用 `torch.cuda.empty_cache()`，这会改变 allocator 行为和计时结果。

### 7.10 Tensor 生命周期

不要把每一步的 logits、loss 或其他 CUDA tensor追加到 Python list 中。只要 Python 仍然持有这些 tensor，它们对应的 computation graph 和 activation 就可能无法释放，表现为显存逐步上涨。

应该保存的是普通 Python 时间数字，而不是模型输出：

```text
保存 elapsed_ms
不保存 logits tensor
```

如果 step function 返回 tensor，调用者也必须确保不会跨 measurement steps 长期持有它。

---

## 8. 推荐的脚本模块设计

建议把脚本拆成职责清楚的小函数，而不是全部写在 `main()` 中。

### 8.1 配置表

维护一个模型规模表：

```text
MODEL_CONFIGS
  small  -> d_model, d_ff, num_layers, num_heads
  medium -> ...
  large  -> ...
  xl     -> ...
  10B    -> ...
```

固定的 `vocab_size`、`batch_size` 和 `context_length` 可以放在 benchmark config 中。

### 8.2 Model factory

职责：

- 接收 model config；
- 调用你实际的 `transformer_lm` 构造函数；
- 设置 device/dtype；
- 返回模型。

单独封装后，staff model 和自己的 model 可以只改这里。

### 8.3 Random batch factory

职责：

- 生成 input token IDs；
- 如果需要 loss，生成 target token IDs；
- 直接在目标 device 上创建；
- 使用正确 shape 与 dtype。

典型 shape：

```text
inputs:  [batch_size, context_length]
targets: [batch_size, context_length]
```

token 值域应满足：

$$
0 \le token\_id < vocab\_size
$$

token IDs 通常使用 `torch.long`，不要把它们改成模型 parameter 的浮点 dtype。

### 8.4 Step function

职责：根据 mode 执行恰好需要的阶段。

伪代码：

```text
run_step(mode):
    if mode needs backward:
        clear old gradients

    logits = model(inputs)

    if mode needs backward:
        objective = make_scalar_objective(logits, targets)
        backward(objective)

    if mode is full_step:
        optimizer step
```

关键要求：不要在 `run_step` 中生成随机 batch、写文件或打印大量日志，否则测量包含的就不只是模型 step。

### 8.5 Warm-up function

伪代码：

```text
repeat warmup_steps times:
    run_step(mode)
    synchronize GPU
```

Warm-up 结果不应加入 measurement samples。

### 8.6 Measurement function

伪代码：

```text
samples = []

repeat measurement_steps times:
    synchronize GPU
    start timer
    run_step(mode)
    synchronize GPU
    stop timer
    append elapsed time
```

计时区间外可以：

- 把秒转换成毫秒；
- 保存 sample；
- 打印简短进度。

不要在 GPU 尚未同步完成时读取结果并假设 step 已结束。

### 8.7 Summary function

职责：

- 计算 mean；
- 计算 standard deviation；
- 返回结构化结果；
- 保留原始 samples。

建议统一使用毫秒：

$$
t_{ms}=1000t_s
$$

### 8.8 Result writer

职责：

- 写 JSONL/CSV；
- 每个 configuration 完成后立即保存；
- 写明配置和环境；
- 对 OOM 写 `status`，而不是伪造时间数字。

---

## 9. “Forward、backward、optimizer 分别多长”应该怎么解释

题目 (a) 明确要求三种累计模式：

```text
F    = forward-only
FB   = forward + backward
FBO  = forward + backward + optimizer
```

但题目 (b) 的文字又要求讨论 forward、backward 和 optimizer step。这里有两种设计方式。

### 9.1 方式一：报告累计模式时间

直接报告：

- $T_F$；
- $T_{FB}$；
- $T_{FBO}$。

然后用差值近似单独阶段：

$$
T_B \approx T_{FB}-T_F
$$

$$
T_O \approx T_{FBO}-T_{FB}
$$

优点：

- 完全符合题目要求的三种 mode；
- 每个 mode 只有 step 末尾一次同步；
- 对端到端性能更自然。

缺点：

- 两个独立实验的噪声会进入差值；
- forward 在不同 mode 中保存的 graph 生命周期可能略有差异；
- optimizer 第一次 state 初始化必须被充分 warm-up。

### 9.2 方式二：在完整 step 中分别计时

在一个 full step 内分别给 forward、backward 和 optimizer 加计时边界。

优点：

- 直接得到各阶段时间；
- 不需要用两个总时间相减。

缺点：

- 每个阶段之间需要 synchronize；
- 额外同步会改变自然的异步执行和潜在 overlap；
- 阶段时间相加不一定完全等于正常 full-step 时间。

### 9.3 推荐做法

第一版优先实现题目明确要求的三个累计 mode，并保存原始样本。如果 writeup 需要明确给出 isolated backward/optimizer 时间，再额外增加一个 phase-timing 模式，但不要混淆两种结果。

无论选择哪种方式，都要在 writeup 中说明测量定义。

---

## 10. 计时边界应该包含什么

这是 benchmark 最容易出现争议的部分。

### 10.1 随机数据生成

建议放在计时区间外。

理由：题目主要测模型 forward/backward/optimizer，不是在测 data loader 或随机数生成器。

### 10.2 模型初始化

必须放在计时区间外。

题目要求根据 hyperparameters 初始化模型，但不要求测量 initialization time。

### 10.3 Gradient clearing

它必须在每个 backward step 前发生，但是否计入 full-step 时间需要明确定义。

建议：

- phase timing 中，把 `zero_grad` 单独视作准备工作；
- end-to-end training step 如果希望模拟真实训练，可以把它包含进去；
- 最重要的是所有 configuration 使用一致规则。

### 10.4 Loss/objective

Backward 必须有 scalar objective。应明确它是否属于 forward/backward 测量区间。

建议在结果 metadata 中写：

```text
objective = cross_entropy
loss_included = true
```

或者：

```text
objective = logits_sum
```

### 10.5 日志与文件写入

放在计时区间外。Terminal I/O 可能远比 Python 算术慢，而且与 GPU 性能无关。

---

## 11. Command-line interface 设计

一个可扩展的调用方式可以是：

```bash
uv run python -m cs336_systems.benchmarking \
  --model-size small \
  --mode forward_backward \
  --warmup-steps 5 \
  --measurement-steps 10 \
  --batch-size 4 \
  --context-length 512 \
  --vocab-size 10000 \
  --device cuda \
  --output results/benchmark.jsonl
```

不要一开始就运行 Table 1 的最大模型。先使用一个极小 debug config 检查控制流，例如：

```text
d_model = 64
d_ff = 256
num_layers = 2
num_heads = 4
batch_size = 2
context_length = 32
```

这个 debug config 不是正式实验结果，只用于快速发现 shape、gradient 和 CLI 错误。

---

## 12. 推荐实现顺序

### Step 1：验证 package import

目标：确认作业 2 可以导入你的作业 1 模型。

检查：

```bash
uv run python -c "import cs336_basics"
```

再验证真正的模型模块和类。

### Step 2：只实现 CPU 小模型 forward

目标：先验证配置、随机 batch、output shape 和 command-line parsing。

此时不关心 GPU 时间。

### Step 3：迁移到 CUDA

检查：

- model 是否在 CUDA；
- inputs/targets 是否在同一 device；
- forward 是否成功；
- GPU 型号是否被正确记录。

### Step 4：实现 forward mode

先完成：

```text
warm-up -> measurement -> samples -> mean/std
```

检查 measurement sample 数是否恰好等于 $n$。

### Step 5：实现 backward mode

增加：

- scalar objective；
- gradient clearing；
- backward；
- 检查至少一个 parameter 的 `.grad` 不为 `None`。

### Step 6：实现 full-step mode

增加 optimizer，并检查 parameter 在 step 后发生变化。

不要在正式 benchmark 中每步复制完整 parameter 来比较；正确性检查只在 debug 阶段做一次。

### Step 7：增加结构化输出

先保存单个 configuration，再实现 Table 1 sweep。

### Step 8：运行正式实验

对每个 model size 和 mode：

```text
warmup = 5
measurements = 10
```

如果 OOM，记录 GPU、mode 和 configuration。

### Step 9：做 warm-up 对照

至少运行：

```text
warmup = 0
warmup = 1
warmup = 2
warmup = 5
```

其余配置应完全相同，才能把差异归因于 warm-up。

---

## 13. Part (b) 的实验表格设计

推荐保留三种累计 mode：

| Model | Mode | Mean (ms) | Std (ms) | Status |
| --- | --- | ---: | ---: | --- |
| small | forward | ... | ... | ok |
| small | forward_backward | ... | ... | ok |
| small | full_step | ... | ... | ok |

如果额外计算 isolated phase estimates，可以单独生成另一张表：

| Model | Forward (ms) | Backward estimate (ms) | Optimizer estimate (ms) |
| --- | ---: | ---: | ---: |
| small | ... | ... | ... |

不要在同一个列名 `Backward` 中混用“backward-only”和“forward+backward”。

报告标准差时应带单位，并与 mean 使用相同小数位。

---

## 14. Part (c) 应该观察什么

不要预先假定无 warm-up 一定慢多少。你要用结果回答。

重点观察：

1. warm-up 为 0 时，第一个 measurement 是否明显更慢；
2. mean 是否被前几个慢样本拉高；
3. standard deviation 是否变大；
4. warm-up 为 1 或 2 时，是否仍有残余初始化成本；
5. forward、backward 和 full-step 的 warm-up 效果是否不同；
6. full-step 的第一次 optimizer state allocation 是否特别明显。

分析时可以画出 sample index 与 time 的关系：

```text
x-axis: measurement sample index
y-axis: time in ms
```

如果最前面的点很高，之后趋于稳定，能够直观说明 cold-start effect。

---

## 15. 没有本地 NVIDIA GPU 时怎么做

本地 CPU 或 Mac 可以完成：

- CLI 设计；
- config 表；
- model factory；
- random batch；
- forward/backward/full-step 控制流；
- mean/std 计算；
- JSONL/CSV 输出；
- pandas 表格生成；
- CPU 小模型正确性检查。

必须在 NVIDIA GPU 上完成：

- 正式 CUDA timing；
- `torch.cuda.synchronize()` 行为验证；
- Table 1 正式结果；
- warm-up 影响；
- OOM 边界；
- 后续 Nsight profile。

Apple MPS 不能替代题目指定的 CUDA/NVIDIA 环境。可以用它测试部分 PyTorch 逻辑，但不能把 MPS timing 当作作业所需 GPU timing。

---

## 16. 常见错误

### 16.1 忘记 synchronize

症状：时间异常地小，且看起来不随模型规模合理增长。

原因：测到的是 CUDA work enqueue，而不是 GPU completion。

### 16.2 只在最后同步一次

如果一次启动 10 个 steps，最后只同步一次，再把总时间简单当成逐 step 样本，就无法得到题目要求的 10 个独立 measurement samples 和 standard deviation。

题目明确要求每个 step 后同步。

### 16.3 把 warm-up 计入 samples

Warm-up 是正式测量前的准备步骤，不应参与 mean/std。

### 16.4 不同 mode 复用错误状态

例如先运行 full step 改变了 weights，再直接把同一 model 用于另一个 mode。虽然随机权重下性能通常近似，但更干净的实验应对每个独立 configuration 明确初始化状态和随机种子。

### 16.5 忘记清 gradient

症状：gradient 不断累加，训练语义错误，显存或 optimizer 行为发生变化。

### 16.6 在计时区间内生成数据

这样测到的包含 RNG 和可能的数据传输，不再是纯模型 step。

### 16.7 在计时区间内打印

Terminal I/O 会引入与 GPU 无关的噪声。

### 16.8 Forward-only 使用 `no_grad`，其他模式使用 grad

这样得到的是 inference forward 与 training forward 的混合比较。除非这是明确实验目标，否则保持一致。

### 16.9 对同一个 graph 重复 backward

第一次 backward 后 graph 通常已释放。每个 measurement step 都要重新 forward。

### 16.10 把秒和毫秒混淆

建议内部统一用秒，输出时一次性转换为毫秒，并在列名中写 `(ms)`。

### 16.11 每步调用 `empty_cache`

这会破坏 caching allocator 的稳态行为，不应放进 benchmark loop。

### 16.12 第一次 optimizer step 被正式计时

AdamW state 常在第一次 step 懒初始化。正确的 full-step warm-up 应覆盖它。

### 16.13 只报告成功配置

如果某个模型 OOM，应记录而不是从表格中悄悄删掉。

### 16.14 模型、input 和 target 不在同一 device

这会直接报 device mismatch，或者引入不希望出现的数据传输。

---

## 17. 正确性检查清单

在正式跑大模型前，逐项确认：

- [ ] `uv run` 能导入 `cs336_basics`；
- [ ] 实际模型模块和类名正确；
- [ ] Table 1 配置无抄写错误；
- [ ] `d_model % num_heads == 0`；
- [ ] input shape 为 `[batch_size, context_length]`；
- [ ] token IDs 是整数 dtype；
- [ ] token 值小于 `vocab_size`；
- [ ] logits shape 符合预期；
- [ ] backward 后 parameter gradient 存在；
- [ ] full step 后 parameter 能更新；
- [ ] 每个 backward step 前清除旧 gradient；
- [ ] warm-up 与 measurement 执行完全相同的 mode；
- [ ] warm-up samples 没有进入统计；
- [ ] measurement samples 数量恰好为 $n$；
- [ ] 每个 step 后调用 synchronize；
- [ ] timer 单位清楚；
- [ ] mean/std 能由原始 samples 重新计算；
- [ ] OOM 被结构化记录；
- [ ] 输出包含 GPU 和软件环境；
- [ ] 数据生成、日志和文件写入不在计时区间内。

---

## 18. 实验质量检查清单

正式结果还应满足：

- [ ] 使用题目指定的 batch size 4；
- [ ] 使用题目指定的 vocabulary size 10,000；
- [ ] 未另行指定时 context length 为 512；
- [ ] Part (b) 使用 5 次 warm-up；
- [ ] Part (b) 使用 10 次 measurements；
- [ ] Part (c) 至少比较 0、1、2、5 次 warm-up；
- [ ] 不同 warm-up 实验除 warm-up 数量外配置相同；
- [ ] GPU 没有明显的其他任务干扰；
- [ ] 没有把 compilation、model initialization 或 data generation 混入正式计时；
- [ ] 结果表明确区分 cumulative mode 与 isolated phase；
- [ ] writeup 说明 scalar objective 和 standard deviation 定义。

---

## 19. 最终交付物建议

### 19.1 代码

建议位置：

```text
cs336_systems/benchmarking.py
```

脚本至少支持：

- 通过 CLI 选择模型 hyperparameters；
- 生成随机 batch；
- 设置 warm-up 和 measurement 数量；
- 选择三种 benchmark mode；
- 每个 step 后同步；
- 输出每次时间与 mean/std。

### 19.2 原始结果

建议保留：

```text
results/benchmark_samples.jsonl
```

是否提交原始结果取决于课程要求，但本地应保留，便于复查 writeup。

### 19.3 Writeup 的 Part (b)

准备：

- 一张自动生成的结果表；
- 1-2 句话总结 forward/backward 时间及 variability。

不要在拿到真实测量前预写“标准差很小”或“backward 是 forward 的两倍”。这些都是需要实验验证的结论。

### 19.4 Writeup 的 Part (c)

准备：

- 0、1、2、5 次 warm-up 的对照结果；
- 2-3 句话解释 cold-start 和初始化成本。

解释必须结合你的真实样本，而不是只复述理论。

---

## 20. 一页式实现路线图

```text
1. 确认 cs336_basics import
2. 建立 Table 1 model config 映射
3. 用 argparse 解析 mode/warmup/steps 等参数
4. 根据 config 在目标 device 上创建模型
5. 创建固定随机 inputs/targets
6. 创建 optimizer
7. 实现 forward / forward_backward / full_step
8. 对同一 mode 执行 w 个 warm-up steps
9. 每个 measurement step：同步 -> 开始计时 -> step -> 同步 -> 结束计时
10. 保存 n 个原始 samples
11. 计算 mean/std
12. 写 JSONL/CSV
13. 先用 tiny config 验证，再跑 Table 1
14. 用 warm-up 0/1/2/5 做对照
15. pandas 自动生成表格，最后写结论
```

如果在任何一步无法解释“当前计时区间到底包含什么”，先停止跑大实验，把测量定义写清楚。可信的 benchmark 首先是一个定义清楚的实验，其次才是一组数字。
