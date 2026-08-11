# CS336 作业 2：GPU 与系统知识入门指南

> 适合读者：有 Python 和 PyTorch 基础，已经完成 CS336 作业 1，但没有 GPU 编程、操作系统或分布式系统基础。
>
> 本文依据 `cs336_assignment2_systems.pdf` 的全部 48 页整理。目标是补齐完成作业所需的知识框架和调试方法，而不是直接给出各题答案。

## 0. 先建立整份作业的地图

作业 1 的核心问题是“Transformer 在数学上如何实现”；作业 2 的核心问题则变成：

1. 同一个 Transformer 为什么有时快、有时慢？
2. 它的时间和显存具体花在哪里？
3. 怎样重新组织计算，让 GPU 少等待、少访问显存？
4. 一张 GPU 放不下或算不快时，怎样让多张 GPU 协作？
5. 并行之后，通信会不会比计算更慢？

讲义中的任务可以分成六条主线：

| 作业部分 | 你要做什么 | 必须理解的知识 |
| --- | --- | --- |
| Benchmarking / Profiling（基准测试/性能分析） | 测量 forward、backward、optimizer | CUDA 异步执行、同步、warm-up、统计方法、Nsight、显存分析 |
| Mixed Precision / Checkpointing（混合精度/检查点） | 降低时间或显存 | FP32/FP16/BF16、autocast、数值稳定性、autograd saved tensors、重计算 |
| FlashAttention-2 / Triton | 写自定义 GPU kernel | GPU 执行模型、内存层次、tiling、stride、kernel fusion、online softmax |
| DDP（Distributed Data Parallel，分布式数据并行） | 多 GPU 数据并行 | 进程、rank、process group、collective、all-reduce、异步通信与重叠 |
| Optimizer State Sharding / FSDP（优化器状态分片/完全分片数据并行） | 对状态、梯度和参数分片 | 参数生命周期、all-gather、reduce-scatter、prefetch、通信与显存权衡 |
| Parallelism Analysis（并行策略分析） | 分析 DP/FSDP/TP/2D 并行 | FLOPs、带宽、通信量、critical path、compute-bound 与 communication-bound |

最重要的总框架是：

$$
\text{训练性能}
=
\text{计算成本}
+
\text{内存访问成本}
+
\text{通信成本}
+
\text{调度与启动成本}
$$

这里的四类成本分别是：

- 计算成本（compute cost）：GPU 真正执行乘法、加法、指数、归约等运算所花的时间。
- 内存访问成本（memory-access cost）：数据在同一张 GPU 内部的寄存器、SRAM/cache 和 HBM 之间移动的成本。
- 通信成本（communication cost）：数据在不同 GPU、不同进程或不同机器之间，通过 NVLink、PCIe 或网络移动的成本。
- 调度与启动成本（scheduling and kernel-launch overhead）：CPU/Python 调度工作、发起 CUDA API 和启动 kernel 的固定开销。

内存访问和通信的共同点都是“搬数据”，区别是移动边界不同：前者主要发生在一张 GPU 内部，后者跨越 GPU、进程或机器。一次 FSDP all-gather 会同时涉及两者：GPU 之间收集权重属于通信，收集后的权重从 HBM 送入计算单元又属于内存访问。

这些成本不一定简单相加。如果计算和通信能够并行，它们对关键路径（critical path）的贡献更接近两者的最大值，而不是总和。

---

## 1. 从 Python 代码到 GPU 内核（GPU kernel）：程序究竟在哪里运行

### 1.1 CPU 和 GPU 的分工

运行一段 PyTorch CUDA 程序时，通常有两个参与者：

- CPU，也称主机端（host）：运行 Python、调度 PyTorch 操作、准备数据、发起 CUDA 调用。
- GPU，也称设备端（device）：执行矩阵乘法、softmax、逐元素运算等 kernel。

例如：

```python
y = torch.matmul(x, w)
```

从 Python 的视角看只有一行，但大致会经历：

```text
Python
  -> PyTorch dispatcher / ATen
  -> CUDA 或 cuBLAS API 调用
  -> 把 kernel 排入 CUDA stream
  -> GPU 在稍后真正执行 kernel
```

CPU 通常只负责“把任务排进队列”，不会在每次调用后等待 GPU 做完。这是理解本作业基准测试、Nsight trace、异步 collective 和通信重叠的共同起点。

### 1.2 GPU 内核（kernel）是什么

kernel 是在 GPU 上执行的函数。一个 PyTorch 表达式可能对应：

- 一个 kernel，例如某个逐元素操作；
- 多个 kernel，例如由若干基础操作组成的 LayerNorm；
- 一个高度优化的库 kernel，例如 cuBLAS 提供的 GEMM；
- 一个编译器融合出来的 kernel，例如 `torch.compile` 生成的 Triton kernel。

kernel 启动（kernel launch）不是免费的。即使一次计算很小，CPU 发起调用、CUDA 调度、GPU 接收任务也有固定延迟。因此：

- 很多小 kernel 往往比少量大 kernel 更慢；
- 把多个操作融合成一个 kernel 可以减少启动开销；
- DDP 中很多小 all-reduce 也会遭遇类似的固定调用开销。

### 1.3 CUDA 流（CUDA stream）与异步执行（asynchronous execution）

CUDA stream 可以理解为 GPU 的有序任务队列。同一个 stream 内的工作按顺序执行，但 CPU 可以在 GPU 工作时继续提交后续任务。

因此下面的计时代码通常是错的：

```python
start = time.perf_counter()
y = torch.matmul(x, w)
elapsed = time.perf_counter() - start
```

这里测到的可能主要是“把 matmul 排入队列”的时间，而不是 GPU 完成 matmul 的时间。正确思路是在测量边界同步：

```python
torch.cuda.synchronize()
start = time.perf_counter()
y = torch.matmul(x, w)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

在作业的逐 step 基准中，讲义要求每一步后调用 `torch.cuda.synchronize()`。如果要精确隔离单个 GPU 区间，也可以了解 CUDA Event，但应以题目要求的测量方式为准。

### 1.4 “同步调用”（synchronous call）不一定表示 GPU 已经完成

这是分布式部分很容易混淆的地方：

- `dist.all_reduce(..., async_op=False)` 通常保证通信操作已经被排入 GPU，而不一定保证 GPU 已经做完。
- `async_op=True` 会更早返回一个 handle；调用 `handle.wait()` 后，结果才能安全地被后续依赖操作使用。
- 做墙钟时间基准时，仍需要在正确的边界调用 `torch.cuda.synchronize()`。

要区分三个时刻：

1. Python API 返回；
2. 操作成功排入 GPU stream；
3. GPU 真正完成操作。

---

## 2. GPU 执行模型（GPU execution model）

### 2.1 为什么 GPU 适合训练神经网络

CPU 擅长复杂控制流和低延迟任务；GPU 擅长对大量数据执行相似操作。矩阵乘法中存在大量相互独立的乘加，所以非常适合并行。

可以先采用一个简化模型：

- GPU 包含许多 Streaming Multiprocessor，简称 SM；
- kernel 被拆成许多线程块（thread block）；
- block 被调度到 SM；
- block 中的线程以线程束（warp）为基本执行组，在 NVIDIA GPU 上通常一个 warp 有 32 个线程；
- 同一 warp 的线程以 SIMT（Single Instruction, Multiple Threads）方式执行相同指令，但处理不同数据。

Triton 隐藏了大量原生 CUDA 线程细节。你主要编写的是“一个 program instance 如何处理一个 tile”，再通过 launch grid 启动许多 program instance。

### 2.2 启动网格（grid）、程序实例（program instance）和分块（tile）

在 Triton 中，常见写法是：

```python
kernel[(grid_0, grid_1)](...)
```

kernel 内部可以用：

```python
pid_0 = tl.program_id(0)
pid_1 = tl.program_id(1)
```

取得当前 program instance 在 launch grid 中的位置。

完成本作业时可以把三者理解为：

- grid：总共启动多少份并行工作；
- program instance：其中一份工作；
- tile：这份工作负责处理的数据块。

例如 FlashAttention forward 中，一个 program instance 可以负责一个 batch 中的一个 query tile，然后在 kernel 内循环遍历所有 key/value tile。

### 2.3 线程束分歧（warp divergence）

如果同一个 warp 内的线程走不同分支，GPU 往往需要分别执行不同路径，部分线程在每条路径上处于空闲状态。这称为 divergence。

本作业中 causal mask、边界判断都可能引入条件逻辑。不过不要看到 `if` 就立即认为很慢：

- 编译期常量分支通常可在编译时消除；
- tile 级别跳过完全被 mask 的区域可能节省大量工作；
- 真正应关注的是同一执行组内部是否频繁走不同路径。

### 2.4 占用率（occupancy）的直觉

occupancy 描述 SM 上可同时驻留的活跃 warp 相对于硬件上限的程度。影响它的因素包括：

- 每个 program/block 使用多少寄存器；
- 使用多少 shared memory；
- tile 是否太大；
- 每个 block 的线程与 warp 数量。

更大的 tile 可能增加数据复用、减少 HBM 访问，却也可能占用过多寄存器，使并发 program 数下降。因此 tile size 需要 benchmark，而不是只靠直觉决定。

---

## 3. GPU 内存层次（memory hierarchy）与性能瓶颈

### 3.1 从快到慢的内存层次（memory hierarchy）

可以用下面的简化层次理解 GPU：

```text
寄存器（每个线程/执行单元附近，最快、最少）
  -> shared memory / SRAM（每个 SM 附近，较快、较少）
  -> L2 cache（全 GPU 共享）
  -> HBM / global memory（容量大、延迟高、带宽有限）
  -> CPU 内存或其他 GPU（还要经过互连）
```

名称会随资料和架构略有差异，但作业中最重要的对比是：

- on-chip SRAM 很快但容量小；
- HBM 容量大但搬运成本高；
- FlashAttention 的关键目标是避免把巨大的 attention matrix 反复写入和读取 HBM。

### 3.2 算术强度（arithmetic intensity）

算术强度是计算量与数据搬运量的比值：

$$
I = \frac{\text{FLOPs}}{\text{bytes transferred}}
$$

如果算术强度低，GPU 大量时间在等待数据，程序更可能是内存带宽受限（memory-bound）；如果算术强度高，计算单元更容易被充分利用，程序更可能是计算受限（compute-bound）。

粗略的 Roofline 判断是：设硬件峰值计算吞吐为 $C$ FLOP/s，内存带宽为 $W$ byte/s，则转折点为：

$$
I^* = \frac{C}{W}
$$

- 当 $I < I^*$ 时，更可能受内存带宽限制；
- 当 $I > I^*$ 时，更可能受计算吞吐限制。

这不是精确运行时间预测，但它能解释为什么：

- 大矩阵乘法通常能有效利用 Tensor Core；
- softmax、normalization、optimizer update 等逐元素或 reduction 操作可能 FLOPs 不多，却占用显著时间；
- fusion 能通过减少中间张量的 HBM 往返改善性能。

### 3.3 FLOP 的计算

矩阵乘法：

$$
(A,B) \times (B,C) \rightarrow (A,C)
$$

通常计为：

$$
2ABC \text{ FLOPs}
$$

原因是输出中有 $AC$ 个元素，每个元素约做 $B$ 次乘法和 $B$ 次加法。

理论计算时间可以粗略写为：

$$
T_{\text{compute}} \approx \frac{F}{C}
$$

其中 $F$ 是总 FLOPs，$C$ 是设备的有效计算吞吐。注意实际吞吐通常低于规格表峰值，因此理论值主要用于比较趋势。

### 3.4 字节数和张量大小

一个张量的存储量是：

$$
M = \text{numel} \times \text{element\_size}
$$

常见数据类型：

| dtype | 每元素字节数 |
| --- | ---: |
| FP32 | 4 |
| FP16 | 2 |
| BF16 | 2 |
| INT64 / `torch.long` | 8 |

换算 MiB 时除以 $1024^2$，换算 GiB 时除以 $1024^3$。作业中的 activation、attention matrix、参数、梯度和 Adam 状态都应使用这种方式进行显存核算。

### 3.5 张量核心（Tensor Core）和低精度矩阵乘法

现代 NVIDIA GPU 有专门加速低精度矩阵乘法的 Tensor Core。FP16/BF16 matmul 往往比普通 FP32 快得多，但前提包括：

- 输入形状适合硬件 tile；
- 维度通常最好是某些倍数；
- 操作确实进入 Tensor Core 路径；
- 数据供给没有成为新的瓶颈。

因此“换成 BF16 就一定按峰值倍数加速”是错误预期。小矩阵、频繁启动、reduction 或 memory-bound 操作的收益可能有限。

---

## 4. 如何做可信的 GPU 基准测试（benchmark）

### 4.1 预热（warm-up）为什么必要

第一次执行往往包含稳态运行没有的成本：

- CUDA context 初始化；
- 动态加载 CUDA 库；
- cuBLAS/cuDNN 的算法选择与内部初始化；
- `torch.compile` 图捕获和编译；
- Triton JIT 编译；
- allocator 首次申请显存；
- cache 处于冷状态；
- NCCL communicator 初始化。

因此应先执行若干 warm-up，再开始采样。讲义在多个问题中指定 5 次 warm-up；应遵循题目配置，不能为了让数字好看而随意改变。

### 4.2 测量区间（measurement region）必须明确

“训练一步”可能包含：

```text
zero_grad
-> forward
-> loss
-> backward
-> gradient synchronization
-> optimizer step
```

作业会分别要求 forward-only、forward-backward、完整 training step。脚本中最好把阶段明确拆开，并保证每种模式真正只包含题目要求的工作。

要特别小心：

- backward 会累积到已有 `.grad`；
- 多次测 backward 时必须重新建立 computation graph；
- optimizer step 会改变权重和初始化 optimizer state；
- `zero_grad(set_to_none=True)` 与把梯度填零的行为和显存不同；
- 随机数据生成是否在计时区间内会改变结果。

### 4.3 平均值（mean）、标准差（standard deviation）和异常值（outlier）

只报告一次测量通常不可信。若采样为 $t_1,\ldots,t_n$，平均值为：

$$
\bar{t}=\frac{1}{n}\sum_{i=1}^{n}t_i
$$

样本标准差为：

$$
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(t_i-\bar{t})^2}
$$

标准差较大时应检查：

- 是否漏了同步；
- 是否混入编译和初始化；
- GPU 是否被其他任务共享；
- 是否发生 thermal throttling；
- 不同 rank 是否负载不均；
- 输入形状是否触发不同编译路径。

### 4.4 可复现实验记录

每组结果至少记录：

- GPU 型号与数量；
- PyTorch、CUDA、Triton、Nsight 版本；
- dtype；
- batch size、sequence length、模型配置；
- warm-up 和 measurement 次数；
- 是否使用 `torch.compile`、autocast、causal mask；
- 单位以及 OOM 情况；
- git commit。

不同 GPU 的结果不可直接横向当作算法优劣证据。作业指定 B200 的实验，应在指定硬件上跑最终数字。

---

## 5. Nsight Systems 性能分析器（profiler）：读懂 CPU 与 GPU 时间线

### 5.1 性能分析（profiling）和基准测试（benchmarking）的区别

- benchmark 回答“整体花了多久”；
- profiler 回答“时间花在哪里，以及操作之间如何排列”。

Profiler 会引入额外开销，所以 profile 得到的总时间不一定等于无 profiler 时的 benchmark。它更适合分析占比、kernel 次数、依赖关系和重叠情况。

### 5.2 Nsight trace 中常见轨道

应学会关联：

- Python / CPU thread：你的 Python 调用；
- CUDA API：CPU 发出的 CUDA 调用；
- CUDA GPU kernel：GPU 真正执行的 kernel；
- NVTX range：你手动标记的 forward、backward、attention 等逻辑区间；
- NCCL：多 GPU collective；
- GPU metrics：利用率、带宽等硬件指标。

CPU 上的一次 CUDA API 调用和 GPU 上的 kernel 不一定在时间轴上紧挨着，因为中间存在排队。

### 5.3 NVTX 的用途

NVTX 相当于在 profiler 时间线上加标签：

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("forward"):
    logits = model(x)
```

它不会改变计算语义，但可以：

- 排除 warm-up；
- 只查看 forward 或 backward；
- 分离 attention、softmax、matmul；
- 判断 NCCL 通信是否与 backward 重叠。

### 5.4 看什么，而不只是看颜色

分析 trace 时依次问：

1. GPU 时间线上是否有明显空洞？
2. CPU 是否来不及提交工作？
3. 最耗累计时间的 kernel 是什么？调用多少次？
4. 单个 kernel 很慢，还是大量小 kernel 累积很慢？
5. 通信是否与计算并行，还是完全落在关键路径上？
6. 不同 rank 的时间线是否对齐？是否有某个 rank 更慢？

---

## 6. 混合精度（mixed precision）与数值稳定性（numerical stability）

### 6.1 FP32、FP16 和 BF16 的差别

浮点数由符号位、指数位和尾数组成。直觉上：

- 指数位决定可表示数值范围；
- 尾数位决定有效精度。

| dtype | 总位数 | 指数位 | 尾数位 | 主要特点 |
| --- | ---: | ---: | ---: | --- |
| FP32 | 32 | 8 | 23 | 范围与精度都较高 |
| FP16 | 16 | 5 | 10 | 精度较低且动态范围小，容易 overflow/underflow |
| BF16 | 16 | 8 | 7 | 与 FP32 接近的动态范围，但有效精度较低 |

BF16 通常比 FP16 更不容易因为范围不足而溢出，但它并不“和 FP32 一样准确”。大量小量累加、均值、方差、softmax 等 reduction 仍常常需要 FP32 accumulation。

### 6.2 为什么低精度累加会越来越不准

浮点数只能表示离散值。当累计和变大时，相邻可表示数之间的距离也会增大。一个很小的增量可能小于当前数附近的最小间隔，于是加法结果不再改变。

这解释了讲义中的 accumulation 实验，也解释了常见策略：

- 输入和输出可以是 BF16/FP16；
- 中间乘法可以低精度；
- reduction accumulator 保持 FP32；
- 最终再 cast 回低精度。

### 6.3 autocast 到底做了什么

`torch.autocast` 不是把整个模型永久改成低精度。它会按操作类别决定计算 dtype：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    y = model(x)
```

重要区别：

- 模型的 master parameters 通常仍是 FP32；
- 某些 matmul/linear 输入会临时 cast 为低精度；
- 某些 normalization、reduction 或 loss 可能保留较高精度；
- gradients 的存储 dtype 通常跟参数相关，而不是简单等于 forward 输出 dtype。

判断某层 dtype 时，不要靠猜；用小实验打印参数、activation、loss 和 `.grad.dtype`。

### 6.4 FP16 损失缩放（loss scaling）

FP16 动态范围较小，微小 gradient 可能下溢为 0。Loss scaling 将 loss 乘以一个尺度 $s$：

$$
\tilde{L}=sL
$$

这样 gradient 也先放大 $s$ 倍，optimizer 前再除回去。BF16 因指数范围更大，通常不需要同样的 loss scaling，但仍应监控 NaN、Inf 和收敛质量。

---

## 7. PyTorch 自动微分（Autograd）、显存构成与激活检查点（activation checkpointing）

### 7.1 计算图（computation graph）

当 tensor 的 `requires_grad=True` 时，PyTorch 会记录相关操作，形成动态图。输出 tensor 的 `grad_fn` 指向反向传播所需的节点。

Backward 并不只需要最终输出。为了计算局部梯度，很多操作必须保存 forward 的输入或中间结果，这些称为：

- 保存张量（saved tensors）；
- 残留量（residuals）；
- 为反向传播保存的激活（activations saved for backward）。

它们往往是训练显存的大头。

### 7.2 一次 AdamW 训练的主要显存类别

应区分：

1. parameters；
2. gradients；
3. optimizer states，例如 Adam 的一阶矩和二阶矩；
4. forward activations / saved tensors；
5. 临时 workspace 和 kernel buffer；
6. CUDA context、library workspace；
7. PyTorch caching allocator 已保留但当前未使用的内存。

若有 $P$ 个 FP32 参数，一个粗略基础核算是：

- 参数：约 $4P$ bytes；
- 梯度：约 $4P$ bytes；
- Adam 两份 FP32 moment：约 $8P$ bytes。

这还没有算 activation、临时张量和 allocator 碎片。混合精度策略也可能额外保存低精度计算副本，所以必须结合具体实现核算。

### 7.3 已分配显存（allocated）、保留显存（reserved）和峰值（peak）

PyTorch caching allocator 会保留已经向 CUDA 申请的显存，供未来重复使用。因此：

- allocated：当前 tensor 真正在占用的内存；
- reserved：PyTorch 从 CUDA 保留的内存，通常不小于 allocated；
- `nvidia-smi`：看到的是进程总体占用，包含更多运行时开销。

比较实验时要使用一致的指标。要测某一区间 peak，通常先 reset peak stats，再运行目标区间并同步，然后读取 peak allocated/reserved。

### 7.4 保存张量钩子（saved tensor hooks）

`torch.autograd.graph.saved_tensors_hooks` 可以观察 autograd 保存和取回哪些 tensor。使用它时要记录：

- shape；
- dtype；
- `numel() * element_size()`；
- 是否是 parameter，避免重复核算；
- stack trace 或对应算子。

仅统计 forward 后仍存活的 saved tensors，不等于进程完整 peak memory，但能回答“为了 backward 保存了什么”。

### 7.5 内核融合（kernel fusion）为什么也能省显存

如果 RMSNorm 被拆成许多 PyTorch 操作，autograd 可能为每个基础操作保存输入。融合后，编译器或自定义 autograd function 可以把整个 RMSNorm 视为一个单元，只保存真正需要的少量数据。

Fusion 的两类收益：

- 减少 kernel launch 和 HBM 中间读写，提高速度；
- 减少 autograd residual，降低显存。

### 7.6 激活检查点（activation checkpointing）

检查点技术（checkpointing）用计算换显存：

- forward：只保存 checkpoint 区间的输入，丢弃区间内部 residual；
- backward：从保存的输入重算 forward，再立即执行该区间 backward。

如果每个 block 保存的 activation 很大，普通训练的峰值 activation memory 随层数 $N$ 近似线性增长。Checkpointing 将长期保存的 checkpoint 和短期重算 residual 进行权衡。

选择 checkpoint 粒度时考虑：

- 区间太大：重算时会同时物化较多 residual；
- 区间太小：checkpoint 输入本身数量增多；
- 嵌套 checkpoint：可进一步省显存，但可能多次重算；
- 随机操作：要关注 RNG state；
- mutation 和 side effect：重算必须与原 forward 语义一致。

作业要求研究“显存最优”与“只允许一层重计算”两种约束。建议先画出 activation 的生命周期，再推导策略，最后用 profiler 验证。

---

## 8. Triton GPU 内核编程所需的最小系统知识

### 8.1 为什么不是继续写普通 PyTorch

普通 PyTorch 表达能力高，但你不总能控制：

- 中间张量是否写入 HBM；
- 操作是否融合；
- tile 如何划分；
- 数据以何种顺序加载；
- 每个 program 负责什么。

Triton 让你在比 CUDA C 更高的抽象层上控制这些细节。本作业的目标不是学习全部 GPU 编程，而是理解“tile + 显式 load/store + fusion”。

### 8.2 指针（pointer）、形状（shape）、步幅（stride）和连续布局（contiguous layout）

tensor 在内存中是一段线性数据。多维索引通过 stride 映射到线性地址。

对于二维 tensor $X$，元素地址可以理解为：

$$
\operatorname{offset}(i,j)
=
i\cdot \operatorname{stride}_0
+
j\cdot \operatorname{stride}_1
$$

shape 说明合法索引范围，stride 说明沿某一维前进一步要跨过多少元素。

`contiguous()` 通常表示数据按预期的连续布局存储。`transpose`、切片等操作可能只改变 view 和 stride，而不重排底层数据。写 Triton kernel 时如果错误假设 contiguous，结果可能悄悄算错。

### 8.3 块指针（block pointer）

`tl.make_block_ptr` 需要理解以下字段：

- base pointer；
- 整体 shape；
- 每一维 stride；
- 当前 tile 的 offsets；
- block shape；
- memory order。

block pointer 的价值是把复杂的逐元素地址计算组织成对一个 N 维 tile 的加载。每轮循环后用 `.advance(...)` 移到下一个 tile。

最常见错误包括：

- offset 忘记乘 tile size；
- Q、K、V 的 stride 维度顺序写反；
- advance 了错误的轴；
- batch offset 漏乘 batch stride；
- 输出 pointer 与输入 pointer 使用了不一致的布局。

### 8.4 边界检查（boundary check）和填充值（padding）

当维度不是 tile size 的整数倍时，最后一个 tile 会越界。加载时需要 mask 或 block pointer 的 boundary check，并为越界值选择正确 padding。

不同 reduction 对 padding 值的需求不同：

- 求和通常用 0；
- 最大值通常用 $-\infty$；
- softmax mask 通常将非法 score 设为很大的负数。

本作业部分测试保证维度为至少 16 的 2 的幂，但 causal mask 和泛化实现仍要求你理解 mask 的含义。

### 8.5 `tl.constexpr`

标记为 `tl.constexpr` 的参数在编译时已知，例如 tile size、hidden dimension 或 causal flag。编译器可以据此：

- 展开或优化循环；
- 删除不会执行的分支；
- 确定 on-chip buffer 形状。

代价是不同 constexpr 组合可能触发不同 kernel 编译，benchmark 时要避免把 JIT 编译混入稳态计时。

### 8.6 归约（reduction）与累加数据类型（accumulation dtype）

FlashAttention 的 running maximum、normalizer 和 output accumulator 应使用 FP32，以减少在线 softmax 累积误差。`tl.dot` 的输入可以是 BF16/FP16，但 accumulator 常应保持 FP32，写回 HBM 前再转换为输出 dtype。

### 8.7 自定义 `torch.autograd.Function`

自定义 kernel 要接入 PyTorch autograd，通常需要：

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ...):
        ctx.save_for_backward(...)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ...
        return grad_inputs
```

注意：

- `forward` 的 tensor 运算默认不会像普通外部 PyTorch 代码那样被 autograd 自动记录；
- backward 返回值顺序必须对应 forward 输入；
- 非 tensor 参数通常返回 `None`；
- 只保存 backward 真正需要的 tensor；
- 保存过多 tensor 会抵消 FlashAttention 的显存优势。

### 8.8 Triton kernel 的正确开发顺序

推荐顺序：

1. 写一个清晰但慢的纯 PyTorch reference；
2. 用极小 shape 手算或检查中间量；
3. 一次只实现一个 tile；
4. 比较每个阶段，而不只比较最终输出；
5. 覆盖多种 dtype、shape、causal/non-causal；
6. 检查 forward；
7. 用 autograd reference 或 gradcheck 思路检查 backward；
8. 正确后才 benchmark 和调 tile。

性能优化不能建立在“多数输入看起来差不多”之上。

---

## 9. FlashAttention-2 必须理解的数学与系统直觉

### 9.1 普通 attention 的问题

忽略 mask，attention 为：

$$
S=\frac{QK^\top}{\sqrt{d}},\qquad
P=\operatorname{softmax}(S),\qquad
O=PV
$$

若 sequence length 为 $N$，每个 batch/head 的 $S$ 和 $P$ 都是 $N\times N$。它们的大小随 $N^2$ 增长。

问题不仅是“放不下”：

- forward 要把大矩阵写入 HBM；
- softmax 再读写它；
- 后续 matmul 再读取；
- backward 还可能再次读取或保存它。

因此普通 attention 同时受到 peak memory 和 memory IO 的影响。

### 9.2 三个核心技巧：分块、融合和重计算

FlashAttention 依靠：

1. 分块（tiling）：每次只在 SRAM 中处理 attention score 的一小块；
2. 融合（fusion）：在一个 kernel 中完成 score、mask、softmax 更新和乘 $V$；
3. 重计算（recomputation）：不保存完整 $P$，backward 时从 $Q,K,V$ 和较小的统计量重算。

它仍然是精确 attention，不是近似 attention。主要改变的是运算顺序和内存访问方式。

### 9.3 为什么普通 softmax 不能直接分块

一行 softmax 为：

$$
p_j=\frac{e^{s_j}}{\sum_k e^{s_k}}
$$

分母依赖整行，看到第一个 key tile 时还不知道后续 tile 的值。因此需要 online softmax：每处理一块，就更新 running maximum 和 running normalizer。

### 9.4 数值稳定 softmax

为了防止指数溢出，softmax 通常减去行最大值：

$$
\operatorname{softmax}(s)_j
=
\frac{e^{s_j-m}}{\sum_k e^{s_k-m}},
\qquad m=\max_k s_k
$$

### 9.5 在线 softmax（online softmax）的更新直觉

假设旧 tile 已有 running maximum $m_{old}$、normalizer $l_{old}$，新 tile 的最大值是 $m_{tile}$，则：

$$
m_{new}=\max(m_{old},m_{tile})
$$

旧的指数和原本以 $m_{old}$ 为基准。最大值更新后，必须缩放到新基准：

$$
l_{new}
=
e^{m_{old}-m_{new}}l_{old}
+
\sum_j e^{s_j-m_{new}}
$$

输出 accumulator 也必须用相同因子重标定：

$$
O_{acc,new}
=
e^{m_{old}-m_{new}}O_{acc,old}
+
\tilde{P}_{tile}V_{tile}
$$

遍历所有 key tile 后，再除以最终 $l$。理解“旧统计量换了指数基准，所以必须 rescale”比死记公式更重要。

### 9.6 对数求和指数（logsumexp）在 backward 中的作用

forward 保存每行：

$$
L_i=\log\sum_j e^{S_{ij}}
$$

backward 时可以重建：

$$
P_{ij}=e^{S_{ij}-L_i}
$$

于是无需保存完整 $P$。这就是 checkpoint/recomputation 思想在单个算子内部的应用。

### 9.7 causal mask

自回归模型中，第 $i$ 个 token 只能关注不晚于自己的 key：

$$
S_{ij}=-\infty \quad \text{when } j>i
$$

tile kernel 中必须比较 query 和 key 的全局索引，而不是只比较 tile 内局部索引。常见 bug 是忘记加 tile 起始 offset。

### 9.8 FlashAttention 的正确性检查

至少检查：

- 输出 shape/dtype/device；
- 与 PyTorch reference 的 forward 误差；
- $L$ 是否正确；
- causal mask 的边界；
- $dQ,dK,dV$ 与 reference 的误差；
- FP32 与 BF16 的合理 tolerance；
- 极大/极小 score 下是否出现 NaN；
- 多个 batch 和不同 $N_q,N_k,d$。

---

## 10. 没有操作系统（operating system）基础时，需要掌握哪些概念

### 10.1 程序（program）、进程（process）和线程（thread）

- 程序（program）：磁盘上的代码和可执行内容；
- 进程（process）：正在运行的程序实例，有自己的地址空间和资源；
- 线程（thread）：process 内的执行流，通常共享该 process 的内存。

PyTorch 分布式训练通常采用“一张 GPU 对应一个 worker process”。不同 process 默认不共享 Python 对象或普通内存，各自有独立的模型对象和 optimizer。

### 10.2 为什么使用多个 process

主要原因包括：

- 每个 process 清晰绑定一张 GPU；
- 避免 Python GIL 对线程级 Python 执行的影响；
- 分布式通信库以 process/rank 为基本参与者；
- 单机和多机可以使用相似编程模型。

### 10.3 `spawn` 的含义

`torch.multiprocessing.spawn` 会启动多个新 process，并把 rank 作为第一个参数传给 worker function。

要使用：

```python
if __name__ == "__main__":
    ...
```

这是因为 spawn 出来的子 process 会重新导入主模块。如果在模块顶层直接再次 spawn，就可能递归创建 process。

### 10.4 地址（address）、端口（port）和会合（rendezvous）

多个 process 在组成 process group 前需要找到彼此：

- `MASTER_ADDR`：协调者所在机器地址；
- `MASTER_PORT`：该机器上用于协调的端口；
- `world_size`：参与 process 数；
- `rank`：每个 process 的唯一编号。

端口可以理解为一台机器上某个网络服务的编号。常见失败原因：

- 端口已被占用；
- 某个 process 没启动；
- world size 写错；
- 不同 process 使用了不同地址/端口；
- 防火墙阻止多机连接。

### 10.5 节点（node）、全局编号（global rank）和本地编号（local rank）

- 节点（node）：一台机器；
- 全局规模（world size）：整个 job 的 worker 总数；
- 全局编号（global rank）：整个 job 内唯一编号；
- 本地规模（local world size）：当前 node 上的 worker 数；
- 本地编号（local rank）：当前 node 内的编号。

单机时 local rank 常等于 global rank；多机时二者不同。GPU 绑定通常使用 local rank：

```python
torch.cuda.set_device(local_rank)
```

### 10.6 进程组（process group）

process group 定义了哪些 rank 一起通信。Collective 操作要求 group 中所有参与 rank 以兼容的顺序调用。

如果 rank 0 调用 `all_reduce`，rank 1 却进入 `broadcast` 或直接跳过通信，程序很可能永久等待。这就是分布式 deadlock 的典型来源。

### 10.7 Gloo 和 NCCL

- Gloo：支持 CPU tensor，适合没有 GPU 时在本地开发控制逻辑；
- NCCL：针对 NVIDIA GPU collective 优化，是作业 GPU benchmark 应使用的 backend。

Gloo 测试通过不意味着 NCCL 性能正确，也不保证 CUDA stream 相关 race 不存在。但它非常适合验证 rank、collective 顺序、参数同步等基本逻辑。

---

## 11. 集合通信（collective communication）

集合操作（collective operation）是一组 rank 共同参与的通信操作。

### 11.1 广播（broadcast）

一个 source rank 持有输入，结束后所有 rank 都得到相同 tensor。

典型用途：DDP 初始化时由 rank 0 向其他 rank 同步模型参数。

### 11.2 归约（reduce）

所有 rank 提供 tensor，执行 sum/mean/max 等 reduction，结果只放到目标 rank。

### 11.3 全归约（all-reduce）

所有 rank 提供 tensor，reduction 后每个 rank 都得到完整结果。

典型用途：DDP 对各 rank 的 parameter gradient 求和或平均。

如果每个 rank 的 loss 是本地 batch mean，要认真推导最终应 sum 还是 average，避免把 learning rate 等效缩放错。

### 11.4 全收集（all-gather）

每个 rank 持有一个 shard，结束后每个 rank 都得到所有 shard 拼成的完整 tensor。

典型用途：FSDP 在某一层计算前临时收集完整权重。

### 11.5 归约分散（reduce-scatter）

每个 rank 提供完整 tensor，先 reduction，再把结果切成 shard 分给各 rank。

典型用途：FSDP 将完整的局部 weight gradient 求和，同时让每个 rank 只保留自己的 gradient shard。

### 11.6 屏障（barrier）

所有 rank 到达 barrier 后才能继续。它适合调试和建立明确实验边界，但滥用会破坏本来可实现的重叠。

### 11.7 collective 必须满足的契约

参与 rank 通常需要保证：

- collective 调用顺序一致；
- tensor shape、dtype 和设备兼容；
- process group 一致；
- root/source rank 一致；
- 没有某个 rank 因异常提前退出。

某个 rank 报错，其他 rank 常表现为“卡住”，所以调试时必须查看全部 rank 日志，而不只看 rank 0。

---

## 12. 通信性能模型（communication performance model）

### 12.1 延迟（latency）与带宽（bandwidth）

通信时间可以粗略分为：

$$
T_{comm}\approx \alpha \cdot n_{steps}+\frac{V}{W}
$$

其中：

- $\alpha$ 是每个通信步骤或调用的固定延迟；
- $n_{steps}$ 是通信步骤数；
- $V$ 是实际传输字节数；
- $W$ 是有效带宽。

这解释了两种不同优化：

- 把许多小 gradient flatten/bucket，减少 $\alpha$ 成本；
- 使用更好的拓扑、dtype 或算法，减少 $V/W$ 成本。

### 12.2 ring all-gather

设最终完整 tensor 为 $S$ bytes，共 $N$ 个设备，每个设备开始持有 $S/N$ bytes。Ring 中每轮把一个 shard 传给邻居，共 $N-1$ 轮。理想化的 bandwidth 项为：

$$
T_{allgather}\approx \frac{N-1}{N}\frac{S}{W}
$$

### 12.3 ring reduce-scatter

Reduce-scatter 同样分成 $N-1$ 轮，每轮传输并累加一个 shard，其理想 bandwidth 项也近似：

$$
T_{reducescatter}\approx \frac{N-1}{N}\frac{S}{W}
$$

### 12.4 ring all-reduce

常见实现可看作 reduce-scatter 加 all-gather，因此理想 bandwidth 项近似：

$$
T_{allreduce}\approx 2\frac{N-1}{N}\frac{S}{W}
$$

讲义的分析题会在忽略 latency 的简化模型中使用这些公式。真实系统还受 NVLink/PCIe/网络拓扑、拥塞、协议和 message size 影响。

### 12.5 通信受限（communication-bound）

若计算与通信不能重叠：

$$
T_{step}\approx T_{compute}+T_{comm}
$$

若它们可以完全重叠：

$$
T_{step}\approx \max(T_{compute},T_{comm})
$$

当 $T_{comm}>T_{compute}$ 时，即使再增加计算设备，也很难获得线性加速，因为通信已经在关键路径上。

### 12.6 强扩展（strong scaling）与弱扩展（weak scaling）

- 强扩展（strong scaling）：固定总问题规模，增加设备；每张设备的计算越来越少，通信更容易占主导。
- 弱扩展（weak scaling）：每张设备的问题规模近似不变，设备增加时总问题规模也增加；较容易保持效率。

本作业的并行分析本质上是在研究 strong scaling 能扩展到多少设备。

---

## 13. 分布式数据并行（Distributed Data Parallel，DDP）

### 13.1 DDP 的不变量（invariant）

DDP 中每个 rank：

- 持有完整且相同的模型参数；
- 持有完整 optimizer state；
- 处理不同的数据 shard；
- 得到不同的本地 gradient；
- gradient averaging 后得到相同 gradient；
- 执行相同 optimizer step 后，参数继续保持一致。

这个“所有 rank 参数始终一致”是不变量。实现和测试都应围绕它设计。

### 13.2 一步 DDP

```text
rank 0 参数 broadcast 到所有 rank

每一步：
  每个 rank 读取不同 mini-batch
  -> local forward
  -> local backward
  -> all-reduce / average gradients
  -> 每个 rank 独立执行同样的 optimizer step
```

如果每张 GPU 的 local batch 为 $b$，world size 为 $N$，则 effective global batch 通常为 $bN$。

### 13.3 naïve DDP 的两个性能问题

1. 每个 parameter 单独 all-reduce：调用次数多，固定 latency 大。
2. 等整个 backward 完成才通信：计算和通信完全串行。

### 13.4 展平（flatten）与梯度桶（gradient bucket）

把多个 gradient 展平并合成较大 buffer 后再通信，可减少 collective 次数。但 bucket 也有权衡：

- bucket 太小：调用开销仍大；
- bucket 太大：必须等更多 gradient ready，延迟通信开始；
- 单一大 flat buffer：通信次数最少，但几乎无法与早期 backward 重叠。

成熟 DDP 通常使用多个 bucket，在调用数与 overlap 之间折中。

### 13.5 反向钩子（backward hook）与计算通信重叠（overlap）

Backward 从靠近 loss 的层逐步向输入传播。某个 parameter 的 gradient ready 后，就可以用 post-accumulate hook 发起异步 all-reduce，不必等其他 parameter。

```text
计算后层 gradient -> 发起其 all-reduce
同时计算前一层 gradient
-> 发起前一层 all-reduce
...
-> optimizer 前等待所有 handle
```

真正的 overlap 必须在 Nsight 中看到 NCCL kernel 与 backward compute 同时出现。仅仅设置 `async_op=True` 并不能自动保证有有效重叠。

### 13.6 DDP 常见正确性问题

- 初始化参数未 broadcast；
- all-reduce 后忘记除以 world size；
- hook 在 gradient 尚未完整累积时触发；
- optimizer step 前没有等待通信；
- 某些 parameter 没参与 forward，导致不同 rank collective 数量不一致；
- 多次 backward 后 handle 列表未清理；
- 所有 rank 使用了同一 GPU；
- 每个 rank 读到了相同数据，而不是不同 shard。

---

## 14. 优化器状态分片（optimizer state sharding）

### 14.1 为什么 optimizer state 很贵

AdamW 通常为每个 parameter 保存：

- 一阶矩 $m$；
- 二阶矩 $v$；
- 可能还有 step counter。

如果 $m,v$ 都是 FP32，仅两份 moment 就需要约两倍 FP32 parameter 大小。普通 DDP 每个 rank 都完整复制这些状态，造成巨大冗余。

### 14.2 作业中的简化 sharding 思路

把 parameter 分配给不同 rank：

- 每个 rank 的 optimizer 只管理自己的 parameter shard；
- 每个 rank 只保存自己 shard 的 optimizer state；
- step 后，parameter 的 owner 把更新结果 broadcast 给其他 rank；
- 所有 rank 重新拥有一致的完整模型参数。

这减少 optimizer state，但没有分片完整模型参数本身。

### 14.3 参数组（parameter group）

PyTorch optimizer 的输入不一定只是 parameter 列表，还可能是 parameter groups：

```python
[
    {"params": group_a, "lr": 1e-3},
    {"params": group_b, "lr": 1e-4, "weight_decay": 0.0},
]
```

实现 wrapper 时必须保留各 group 的 hyperparameters，并支持后续 `add_param_group`。不能只按 `model.parameters()` 的最简单情况设计。

### 14.4 分片策略

需要决定哪个 parameter 归哪个 rank。目标通常是让各 rank 的 parameter numel 大致平衡，而不是简单让每个 rank 获得相同数量的 tensor，因为 tensor 大小可能差异很大。

所有 rank 必须独立得出相同 ownership 映射，否则 broadcast source 会不一致并导致错误或 deadlock。

### 14.5 与 ZeRO 的关系

ZeRO 的核心思想是逐步消除 data parallel rank 间的状态冗余：

- Stage 1：分片 optimizer states；
- Stage 2：进一步分片 gradients；
- Stage 3：进一步分片 parameters。

作业中的实现用于理解基本思想，通信安排和内存行为不一定与工业级 ZeRO 完全相同。分析题要求你从“每一步传多少、每张卡长期存什么”两个角度比较，而不是只比较名称。

---

## 15. 完全分片数据并行（Fully-Sharded Data Parallel，FSDP）

### 15.1 DDP 与 FSDP 的核心差别

普通 DDP：

- parameter：每个 rank 完整复制；
- gradient：每个 rank 最终完整持有；
- optimizer state：每个 rank 完整复制。

FSDP：

- parameter：长期只保留 shard；
- gradient：最终只保留 shard；
- optimizer state：只对应本 rank shard；
- 计算某层前临时 all-gather 完整权重；
- gradient ready 后用 reduce-scatter 聚合并分片。

### 15.2 forward 的权重生命周期（weight lifecycle）

对一个被 sharding 的 Linear/Embedding 层：

```text
长期保存 master weight shard
-> all-gather 形成临时完整 weight
-> 用完整 weight 做该层 forward
-> 尽快释放完整 weight
```

如果收集太晚，GPU compute 会等待通信；如果收集太早或释放太晚，多个完整 weight 同时存在，peak memory 会升高。

### 15.3 backward

Backward 计算 weight gradient 通常仍需要完整 weight，因此可能需要再次 all-gather。局部 backward 产生的完整 gradient contribution 经 reduce-scatter 后，每个 rank 只保留自己的 gradient shard。

### 15.4 预取（prefetch）

为了隐藏 all-gather 延迟，可以在当前层计算时预取未来层权重：

```text
compute layer k
同时 all-gather layer k+1 或 k+2
```

预取距离需要折中：

- 太短：通信来不及完成，compute 等待；
- 太长：过多完整权重同时驻留，显存上升。

讲义对预取时机给出具体约束，实现时应严格对应，而不是任意提前全部 gather。

### 15.5 哪些层值得 shard

很小的层，例如某些 normalization，参数很少、计算很快。对它们发起 all-gather 的固定 latency 可能比节省的显存更不划算。因此作业主要要求对 Linear 和 Embedding 做 sharding。

### 15.6 mixed-precision FSDP

常见策略：

- master weight shard 保持 FP32，供 optimizer 反复更新；
- all-gather 前转换为 BF16/FP16；
- 通信和计算使用低精度副本；
- optimizer 仍更新 FP32 master weight。

这样既降低通信字节数，也保持长期累积更新的精度。

### 15.7 FSDP 最容易出现的 bug

- 临时完整 weight 释放过早，backward 访问无效数据；
- 释放过晚，显存没有真正下降；
- parameter object 被替换后 optimizer 仍引用旧对象；
- shard shape/offset 在不同 rank 不一致；
- 参数元素数不能整除 world size 时处理错误；
- all-gather/reduce-scatter 顺序在 rank 间不一致；
- forward prefetch 与 backward gather 共用 buffer 时发生 race；
- mixed precision 把 master weights 也永久转成低精度。

讲义建议多次运行测试，是因为 race condition 可能并非每次触发。

---

## 16. 其他并行策略（parallelism strategies）及其定位

### 16.1 数据并行（Data Parallelism，DP）

- shard：batch；
- replicate：model weights；
- communication：gradient all-reduce；
- 优点：概念简单，计算效率高；
- 限制：模型和 optimizer state 仍需放入每张 GPU。

### 16.2 完全分片数据并行（Fully-Sharded Data Parallelism，FSDP）

- shard：batch、weights、gradients、optimizer states；
- communication：weight all-gather、gradient reduce-scatter；
- 优点：大幅降低每张 GPU 的模型状态显存；
- 限制：通信更多，调度和生命周期复杂。

### 16.3 张量并行（Tensor Parallelism，TP）

- shard：单个 weight matrix 的输入或输出维；
- communication：activation all-reduce/all-gather 等；
- 优点：单层计算本身跨设备，可处理单层也放不下的模型；
- 限制：几乎每层都可能通信，对低延迟高带宽互连要求高。

术语：

- 列并行（column parallel）：按 weight 输出维分片；
- 行并行（row parallel）：按 weight 输入维分片。

合理组合 row/column parallel，可以让中间 activation 保持分片，避免不必要的 all-gather。

### 16.4 流水线并行（Pipeline Parallelism，PP）

- shard：模型层；
- 每个设备负责连续的一段层；
- activation 在 stage 间传递；
- 需要 microbatch 来减少 pipeline bubble。

### 16.5 专家并行（Expert Parallelism，EP）

- shard：Mixture-of-Experts 中的 experts；
- token 根据 router 被发送到不同设备；
- 常见通信是 all-to-all；
- 负载均衡非常关键。

### 16.6 二维并行（2D parallelism）

把设备排列成两个维度，例如：

```text
FSDP axis × TP axis
```

一个 axis 负责 batch 与 model-state sharding，另一个 axis 负责 layer 内 tensor sharding。分析时要分别计算两条 axis 的通信：

- 若网络资源独立且可完全重叠，关键成本更接近 `max`；
- 若共享同一互连而不能重叠，成本更接近求和。

---

## 17. 如何做作业中的并行性能推导（parallel performance analysis）

不要直接套公式。每一道分析题都按以下顺序做。

### 17.1 第一步：标 shape

对每个 tensor 写 shape。例如 FFN：

$$
x:(B,D),\quad
W_1,W_2:(D,D_{FF}),\quad
W_3:(D_{FF},D)
$$

然后根据 DP/FSDP/TP 修改 local shape。

### 17.2 第二步：数 matmul

对每个 matmul 使用 $2ABC$，区分 forward、activation gradient 和 weight gradient。不要把 backward 简单说成“约两倍”，分析题通常要求精确到给定简化模型。

### 17.3 第三步：列出 collective

逐个写明：

- collective 类型；
- 输入完整大小还是 shard 大小；
- 数据类型与每元素字节数；
- 在 forward 还是 backward；
- 是否能与其他工作重叠。

### 17.4 第四步：换成时间

$$
T_{compute}=\frac{F}{C}
$$

$$
T_{comm}=\frac{\text{communicated bytes}}{W}
$$

如果题目忽略 latency，就不要自行加入 $\alpha$；如果做真实 benchmark 解释，则应考虑 latency。

### 17.5 第五步：找瓶颈条件

令：

$$
T_{comm}\le T_{compute}
$$

再整理出设备数的上限。最后做 sanity check：

- batch 或 hidden size 增大时，可扩展设备数是否应该增大？
- bandwidth 增大时，通信瓶颈是否应该推迟？
- accelerator compute 更快但带宽不变时，是否反而更容易通信瓶颈？

这些单调性检查很容易发现代数错误。

---

## 18. 没有本地 GPU，应该怎样完成这份作业

### 18.1 先明确：Mac 的 GPU 不能替代作业要求的 NVIDIA GPU

如果你使用 Apple Silicon：

- PyTorch MPS 可以运行部分通用 tensor 代码；
- 但 CUDA、NCCL、Nsight Systems 的 CUDA trace、Triton CUDA kernel 都依赖 NVIDIA 环境；
- 作业指定的 B200 性能数据必须在相应 NVIDIA 机器上取得。

不要花大量时间尝试让 Nsight/NCCL/Triton CUDA 在 Mac 本地“等价运行”。正确策略是本地完成控制逻辑、数学和 CPU 测试，远程 GPU 做集成、调试和测量。

### 18.2 本地 CPU 可以完成的工作

- 阅读讲义和完成并行计算推导；
- 设计 benchmark CLI、参数表、结果保存格式；
- 测试模型配置和随机数据生成；
- 编写纯 PyTorch reference attention；
- 使用小 shape 验证 online softmax 算法；
- 使用 Gloo 测试 process group、rank、broadcast、all-reduce；
- 用小 CPU model 测 DDP 参数是否保持一致；
- 测试 optimizer parameter group 和 ownership 逻辑；
- 编写绘图、表格和 writeup 自动化代码；
- 做静态检查和不依赖 CUDA 的单元测试。

### 18.3 必须在 NVIDIA GPU 上完成的工作

- CUDA timing 和 `torch.cuda.synchronize()` 实验；
- Nsight Systems CUDA/NCCL profile；
- CUDA memory snapshot 和 OOM 边界；
- autocast 的 CUDA 性能比较；
- Triton FlashAttention kernel 编译、调试和 benchmark；
- NCCL collective benchmark；
- DDP communication overlap；
- optimizer sharding 的真实 GPU memory/time；
- FSDP all-gather/reduce-scatter、prefetch 和 race 调试；
- B200 上指定配置的最终实验与 leaderboard。

### 18.4 GPU 数量需求

按讲义要求规划资源：

| 工作 | 资源 |
| --- | --- |
| 单模型 benchmark、memory profile、FlashAttention | 1 张 NVIDIA GPU，最终部分实验指定 B200 |
| DDP、optimizer sharding、FSDP 标准配置 | 单机 2 张 GPU |
| single-node all-reduce sweep | 单机 2、4、6 张 GPU，最多 6 张 |
| leaderboard | 2 张 B200 |

### 18.5 远程机器上的最小检查

登录后先检查：

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
uv run python -c "import triton; print(triton.__version__)"
nsys --version
```

多 GPU 时再检查：

```bash
nvidia-smi -L
```

确认 job 实际分配到几张 GPU，不要假定设备编号与物理机器全局编号一致；调度系统常通过 `CUDA_VISIBLE_DEVICES` 重映射。

### 18.6 本地与远程的工作循环

```text
本地：写清楚 reference 和小测试
  -> git commit
远程：拉取同一 commit
  -> 小 shape / 单 step 验证
  -> 完整测试
  -> profiler
  -> benchmark
  -> 保存原始结果与环境信息
本地：整理表格和 writeup
```

不要一上 GPU 就跑最大配置。建议按：

```text
最小 shape
-> 中等 shape
-> 单次执行
-> 少量 warm-up/measurement
-> 题目完整配置
```

逐级放大，可以减少昂贵 GPU 时间被 OOM、编译错误或 deadlock 浪费。

---

## 19. 推荐的完成顺序

### 阶段 A：建立测量基础

1. 修好作业 1 package 导入。
2. 写统一 benchmark 参数配置。
3. 区分 forward、backward、optimizer step。
4. 理解同步和 warm-up。
5. 自动输出均值、标准差、OOM 和环境信息。

这部分代码会被后面的 mixed precision、memory、DDP benchmark 反复使用，值得先设计好。

### 阶段 B：单 GPU 系统知识

1. 用 Nsight 建立 CPU API 与 GPU kernel 的对应关系。
2. 做 mixed precision 小实验。
3. 学会 memory snapshot 和 saved tensor hooks。
4. 理解 activation checkpointing 的时间/显存交换。

### 阶段 C：FlashAttention

1. 普通 attention benchmark；
2. `torch.compile` 对照；
3. Triton weighted-sum 示例；
4. 纯 PyTorch tiled online softmax；
5. Triton forward；
6. causal mask；
7. backward recomputation；
8. correctness 完整后再 benchmark。

### 阶段 D：分布式基础

1. 本地 Gloo 两个 process；
2. broadcast/all-reduce 小 tensor；
3. 参数同步不变量；
4. GPU NCCL collective benchmark；
5. naïve DDP；
6. flatten gradients；
7. hook + async overlap；
8. Nsight 验证重叠。

### 阶段 E：状态与参数分片

1. optimizer state memory accounting；
2. parameter ownership；
3. sharded optimizer；
4. FSDP weight lifecycle；
5. all-gather/reduce-scatter；
6. prefetch、释放和 mixed precision；
7. 多次测试排查 race。

### 阶段 F：并行策略分析

最后做 DP、FSDP、TP 和 2D parallelism 的公式推导。此时你已经在实验中见过所有 collective，公式不会只是抽象符号。

---

## 20. 常见故障的系统化排查

### 20.1 时间异常地小

检查：

- 是否在结束计时前同步；
- 是否真的执行了 backward；
- 结果是否被 lazy/compile 路径影响；
- 计时单位是否弄错；
- 是否只测到 API enqueue。

### 20.2 第一次特别慢

通常来自初始化或 JIT。确认 warm-up 是否覆盖相同 shape、dtype、causal flag 和代码路径。

### 20.3 CUDA OOM

按顺序检查：

1. 报错发生在 forward、backward 还是 optimizer step；
2. peak allocated 与 reserved；
3. 是否还引用旧 graph/output/loss；
4. gradient 是否未清空；
5. benchmark 是否在循环中不断保存 tensor；
6. attention $N^2$ tensor 是否被物化；
7. optimizer state 是否在第一次 step 才初始化；
8. profiler 是否增加了额外开销。

### 20.4 mixed precision 出现 NaN

检查：

- FP16 是否需要 loss scaling；
- softmax 是否减最大值；
- accumulator 是否使用 FP32；
- mask 后是否出现整行全为 $-\infty$；
- normalization 的 epsilon；
- learning rate 与输入幅值。

### 20.5 Triton 输出错误

优先缩小到一个 tile，检查：

- strides；
- batch offset；
- program ID 与 tile offset；
- `.advance` 的轴和步长；
- boundary mask；
- causal 全局索引；
- accumulator dtype；
- 写回 output 的 cast；
- backward 返回顺序。

### 20.6 分布式程序卡住

这通常不是“运行得慢”，而是 collective 不匹配。检查：

- 每个 rank 是否到达相同 collective；
- 调用顺序是否一致；
- tensor shape/dtype/device 是否一致；
- world size 是否与实际 process 数一致；
- 某个 rank 是否先 OOM 或异常退出；
- master port 是否冲突；
- 每个 rank 是否绑定不同 GPU。

为每条日志加 rank：

```python
print(f"[rank={rank}] before all_reduce", flush=True)
```

### 20.7 DDP 参数逐步漂移

检查：

- 初始化 broadcast；
- gradient average；
- optimizer hyperparameters 和初始 state；
- optimizer 前等待通信；
- 数据不同是正常的，但同步后 gradient 应一致；
- 是否存在未同步 parameter。

### 20.8 FSDP 偶发失败

偶发通常提示 race 或生命周期错误：

- async handle 是否等待；
- buffer 是否在通信完成前复用；
- gathered weight 是否过早释放；
- hook 顺序是否依赖不稳定的遍历；
- 所有 rank 是否建立同样的 layer 顺序。

---

## 21. 关键术语速查

| 中文与英文术语 | 简短解释 |
| --- | --- |
| 主机端/设备端（host/device） | CPU 侧 / GPU 侧 |
| 计算成本（compute cost） | 执行算术运算所花的时间或资源 |
| 内存访问成本（memory-access cost） | 同一 GPU 内部在寄存器、cache/SRAM 和 HBM 之间搬数据的成本 |
| 通信成本（communication cost） | 在 GPU、进程或机器之间搬数据的成本 |
| 调度与内核启动开销（scheduling and kernel-launch overhead） | CPU 调度、CUDA API 调用和启动 GPU kernel 的固定成本 |
| GPU 内核（kernel） | 在 GPU 上执行的函数 |
| CUDA 流（CUDA stream） | GPU 工作的有序队列 |
| 流式多处理器（Streaming Multiprocessor，SM） | GPU 上调度和执行 block 的处理单元 |
| 线程束（warp） | NVIDIA GPU 的基本线程执行组 |
| 高带宽显存（High Bandwidth Memory，HBM） | GPU 的大容量高带宽显存 |
| 片上静态存储（Static Random-Access Memory，SRAM） | GPU 芯片上的快速小容量存储 |
| 内存层次（memory hierarchy） | 寄存器、SRAM/cache、HBM 等不同容量与速度的存储层级 |
| 分块（tile） | 一个 program/block 处理的数据块 |
| 步幅（stride） | 多维 tensor 沿某轴移动一步跨过的元素数 |
| 浮点运算次数（FLOP） | 浮点计算量的计量单位 |
| 算术强度（arithmetic intensity） | 每搬运一个 byte 做多少 FLOPs |
| 计算受限（compute-bound） | 性能主要受计算吞吐限制 |
| 内存受限（memory-bound） | 性能主要受内存带宽限制 |
| 通信受限（communication-bound） | 性能主要受设备间通信限制 |
| 关键路径（critical path） | 决定一步总墙钟时间的依赖链 |
| 延迟（latency） | 发起并完成一次操作的固定或基础时间成本 |
| 带宽（bandwidth） | 单位时间能够传输的数据量 |
| 预热（warm-up） | 正式计时前消除初始化/JIT 等冷启动影响的运行 |
| 性能分析器（profiler） | 记录操作耗时、调用关系和时间线的工具 |
| NVTX 范围（NVTX range） | 给 profiler 时间线添加的逻辑标签 |
| 混合精度（mixed precision） | 按操作需要组合使用低精度与高精度计算 |
| 保存张量（saved tensor） | backward 所需、由 forward 保存的 tensor |
| 激活检查点（activation checkpointing） | 丢弃中间 activation，backward 时重算 |
| 内核融合（kernel fusion） | 把多个操作合成更少的 kernel |
| 程序/进程/线程（program/process/thread） | 磁盘代码、运行实例和进程内执行流 |
| 进程组（process group） | 共同参与分布式通信的一组 worker process |
| 编号（rank） | process group 中 worker 的唯一编号 |
| 全局规模（world size） | process group 的 worker 总数 |
| 集合通信（collective communication） | 一组 rank 共同参与的通信操作 |
| 广播（broadcast） | 从一个 source rank 向所有 rank 复制数据 |
| 全归约（all-reduce） | 聚合后把完整结果给每个 rank |
| 全收集（all-gather） | 收集所有 shard，让每个 rank 得到完整结果 |
| 归约分散（reduce-scatter） | 聚合完整输入后，把结果 shard 分给各 rank |
| 屏障（barrier） | 等所有参与 rank 到达后再一起继续 |
| 计算通信重叠（overlap） | 让通信与计算在时间上并行发生 |
| 分布式数据并行（Distributed Data Parallel，DDP） | batch 分片，模型复制，gradient all-reduce |
| 优化器状态分片（optimizer state sharding） | 每个 rank 只长期保存部分 optimizer state |
| 完全分片数据并行（Fully-Sharded Data Parallel，FSDP） | 参数、梯度和 optimizer state 都分片 |
| 张量并行（Tensor Parallelism，TP） | 单层 weight matrix 跨设备分片 |
| 流水线并行（Pipeline Parallelism，PP） | 按模型层把计算划分为多个 stage |
| 专家并行（Expert Parallelism，EP） | 将 Mixture-of-Experts 中的 experts 分配到不同设备 |

---

## 22. 开始编码前的自检清单

如果下面多数问题都能回答，就具备完成作业的知识基础：

- [ ] 为什么 GPU 计时前后需要同步？
- [ ] warm-up 消除了哪些成本？
- [ ] 为什么 FLOPs 很少的 softmax 仍可能占显著时间？
- [ ] FP16 和 BF16 的动态范围为什么不同？
- [ ] autocast 为什么不等于把所有 parameter 改成低精度？
- [ ] autograd 为什么需要保存 activation？
- [ ] checkpointing 用什么换取显存？
- [ ] shape 和 stride 如何决定 tensor 地址？
- [ ] 一个 Triton program instance 应该负责哪个 tile？
- [ ] online softmax 更新最大值后为什么要重标定旧 accumulator？
- [ ] FlashAttention 为什么避免 $N^2$ 的 HBM 中间张量？
- [ ] process、rank、world size 和 local rank 分别是什么？
- [ ] all-reduce、all-gather、reduce-scatter 的输入输出有什么区别？
- [ ] `async_op=True` 为什么不自动等于有效 overlap？
- [ ] DDP 如何保证所有 rank 的参数始终一致？
- [ ] flatten gradients 为什么降低 latency，却可能减少 overlap？
- [ ] optimizer state sharding 省掉了哪部分显存？
- [ ] FSDP forward 为什么需要临时 all-gather 权重？
- [ ] 为什么 FSDP 要及时释放 gathered weights？
- [ ] 如何比较 $T_{compute}$ 与 $T_{comm}$？
- [ ] 为什么 accelerator 越快，反而可能越容易 communication-bound？

---

## 23. 建议阅读材料

讲义末尾给出的材料与本作业高度相关，建议按顺序阅读：

1. Horace He，*Making Deep Learning Go Brrrr From First Principles*：建立 GPU 性能、fusion 和内存访问直觉。
2. Milakov 与 Gimelshein，*Online Normalizer Calculation for Softmax*：理解 online softmax。
3. FlashAttention 原论文：理解 IO-aware attention 的核心动机。
4. FlashAttention-2：理解并行划分和 work partitioning 的改进。
5. ZeRO 论文：理解 optimizer、gradient、parameter 三阶段 sharding。
6. TPU Scaling Book：系统理解并行策略、拓扑和扩展分析。
7. Ultra-Scale Playbook：补充大规模训练和 pipeline parallelism。

阅读论文时不必一开始追求每个证明都掌握。先回答三类问题：

- 它减少了哪种资源：计算、HBM IO、peak memory 还是网络通信？
- 它引入了什么新成本？
- 它依赖什么硬件或执行假设？

能稳定回答这三问，就已经具备系统方向最重要的性能分析习惯。
