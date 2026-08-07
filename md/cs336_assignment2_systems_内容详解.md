# CS336 Assignment 2（Systems and Parallelism）内容详解

> 原文：`cs336_assignment2_systems.pdf`  
> 版本：26.1.3  
> 课程：Stanford CS336，Spring 2026  
> 文档长度：48 页

## 1. 这份作业在讲什么

这份作业的主题是“大语言模型训练系统”。它不是主要考察 Transformer 数学本身，而是要求从系统角度回答三个问题：

1. 单张 GPU 为什么没有跑满，时间和显存究竟花在哪里？
2. 如何用定制 GPU kernel、混合精度和重计算提高单卡效率？
3. 如何把训练扩展到多张 GPU，同时控制通信开销和每卡显存？

整份作业形成一条完整的优化链路：

```text
建立可靠基准
  → 用 Nsight/PyTorch profiler 找瓶颈
  → 混合精度与 torch.compile
  → activation checkpointing 降低单卡显存
  → Triton FlashAttention-2 改善 attention 的 IO 与显存复杂度
  → DDP 扩展到多卡
  → 优化通信粒度并让通信和计算重叠
  → optimizer state sharding
  → FSDP 分片参数、梯度和优化器状态
  → 用理论模型分析 DP/FSDP/TP/二维并行的扩展上限
```

因此，这份作业真正训练的是一种系统优化方法论：**先测量，再解释，再优化，最后验证优化是否真的改善了端到端性能。**

## 2. 作业结构、提交物和环境

### 2.1 要实现的六类核心内容

PDF 开头列出的实现任务是：

1. benchmark 与 profiling 基础设施；
2. activation checkpointing；
3. FlashAttention-2 Triton kernel；
4. distributed data parallel（DDP）；
5. optimizer state sharding；
6. fully sharded data parallel（FSDP）。

### 2.2 仓库结构

- `cs336-basics/`：Assignment 1 的参考实现，当前作业会对其中的 Transformer 进行测量和优化；也可以在 `pyproject.toml` 中改为使用自己的 Assignment 1 实现。
- `cs336_systems/`：本作业的空模块，主要实现可以自行组织在这里。
- `tests/*.py`：评分测试。测试通过 `tests/adapters.py` 中的适配器调用学生实现。
- `README.md`：环境和目录结构的补充说明。

文档强调：可以为调试增加或临时修改测试，但最终代码应通过原始测试套件。

### 2.3 最终提交物

- `writeup.pdf`：所有书面回答、表格、图、profiling 截图和分析，要求排版。
- `code.zip`：自己编写的代码。

仓库提供 `test_and_make_submission.sh` 用于运行测试并生成提交压缩包。

### 2.4 统一模型规格

除 leaderboard 外，默认词表大小是 10,000，batch size 是 4，默认 context length 是 512（题目另有指定时除外）。

| 规模 | `d_model` | `d_ff` | 层数 | 头数 |
|---|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

这些配置大多参考 GPT-2。它们贯穿 benchmark、memory profiling、DDP 和 FSDP 实验，因此最好把模型配置集中管理，不要散落在不同脚本中。

## 3. 第 2 章：Profiling and Benchmarking

这一章的目标是建立可信的性能测量方法。文档反复强调，如果没有先定位真正的时间和显存热点，很容易优化一个对端到端运行几乎没有影响的局部。

### 3.1 端到端 benchmark

第一项任务是写一个可配置脚本，能：

- 根据超参数创建 Transformer；
- 随机生成一批输入数据；
- 运行指定数量的 warm-up；
- 分别测量 forward-only、forward + backward、完整训练步（再加 optimizer step）；
- 每个测量步之后调用 `torch.cuda.synchronize()`；
- 输出多次测量的平均值和标准差。

这里最重要的系统知识是：CUDA kernel launch 通常是异步的。Python 函数返回只说明操作已经被排入 GPU 队列，并不表示 GPU 已完成计算。若不显式同步，测到的很可能主要是 CPU 发射 kernel 的时间，而不是 GPU 实际执行时间。

Warm-up 也不是装饰。首次运行可能包含 CUDA context 初始化、kernel 加载、缓存建立、JIT/compile 等一次性成本；没有预热，或者只预热一两步，结果可能仍未进入稳定状态。

对应问题：`benchmarking_script`（4 分）。实验要求对表中的模型使用 5 个 warm-up step、10 个测量 step，报告平均值、标准差以及无预热时的差异。

### 3.2 Nsight Systems

端到端时间只能告诉你“慢”，不能告诉你“慢在哪里”。因此下一步用 NVIDIA Nsight Systems 同时观察：

- CPU 端的 CUDA API 调用；
- GPU 端实际执行的 kernel；
- cuBLAS、cuDNN、CUDA runtime；
- PyTorch module/autograd 范围；
- GPU utilization；
- 必要时的 CUDA/Python backtrace。

文档给出的基础调用形式是：

```bash
uv run nsys profile -- python benchmark.py
```

更完整的配置会加入 `--trace`、`--pytorch`、backtrace、GPU metrics 和 NVTX capture。文档特别建议用 NVTX range 标出测量区间、forward、backward，以及 attention 内部的 score matmul、softmax、最终 matmul，从而把源代码阶段与具体 CUDA kernel 对齐。

Profiling 本身会产生开销，尤其是完整 backtrace，因此应根据问题选择功能，而不是每次打开所有选项。`torch.compile` 也会融合或重写计算图，导致 kernel 很难重新归因到某一行 Python 代码；做归因分析时可能需要调整 compile 和 NVTX 的边界。

对应问题：`nsys_profile`（5 分）。需要选择两个模型规模和三个大于 128 的 2 的幂次 context length，分析：

- Nsight 的 forward 总时间是否与 Python benchmark 一致；
- forward 中累计 GPU 时间最长的 kernel、调用次数，以及加入 backward 后热点是否变化；
- matmul 之外哪些 kernel 占用显著时间；
- 完整 AdamW 训练步中，matmul 与其他 kernel 的时间占比如何改变；
- attention 内 softmax 和 matmul 的运行时间差异是否与 FLOP 差异相符。

这里隐含的一项重要认识是：**FLOP 多不等于一定最慢。** 某些逐元素、归约或数据搬运操作 FLOP 很少，却可能受限于内存带宽、同步或 kernel launch 开销。

### 3.3 混合精度

文档对比了 FP32、FP16 和 BF16：

- 现代 NVIDIA Tensor Core 对 FP16/BF16 matmul 的吞吐量远高于 FP32；
- FP16 动态范围小，微小梯度可能下溢为 0，较大数值也容易溢出，因此训练时常配合 loss scaling；
- BF16 的尾数精度较低，但指数范围与 FP32 相同，通常比 FP16 更稳定；
- 纯低精度训练仍可能影响最终模型质量，因此实际常用 mixed precision。

PyTorch 的 `torch.autocast` 会按算子选择计算 dtype：适合 Tensor Core 的 matmul/linear 可用低精度，而对数值范围和累加精度敏感的 reduction、normalization 等操作可能保留 FP32。

文档安排了两个问题来建立直觉：

- `mixed_precision_accumulation`（1 分）：重复累加小数，比较 FP32 累加、FP16 累加和类型转换后的结果，理解舍入误差为何会逐步积累。
- `benchmarking_mixed_precision`（2 分）：分析 toy model 中参数、linear 输出、LayerNorm 输出、logits、loss 和 gradient 的 dtype；解释 LayerNorm 哪些部分敏感；再给 benchmark 脚本加入 BF16 autocast，比较所有模型规模的 forward/backward 时间。

核心点是：mixed precision 并不是简单地把整个模型执行一次 `.half()`。参数的存储 dtype、算子的实际计算 dtype、累加 dtype和输出 dtype可能不同。

### 3.4 显存 profiling

文档使用 PyTorch CUDA memory history：

```python
torch.cuda.memory._record_memory_history(max_entries=1000000)
# 被分析的代码
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

生成的 pickle 可载入 `pytorch.org/memory_viz`，查看 active memory timeline、分配大小和分配点堆栈。与只查看 `max_memory_allocated` 相比，这种时间线能够把 forward 保存 activation、backward 释放 residual 并产生 gradient、optimizer 初始化或更新状态等阶段区分开。

`memory_profiling`（4 分）聚焦 xl 模型、context length 128 与 2048，要求：

- 对比仅 forward 与完整训练步的 active-memory timeline；
- 报告不同 context length 下的峰值；
- 比较 mixed precision 是否显著影响峰值；
- 计算 residual stream 中一个 FP32 activation tensor 的大小；
- 从分配堆栈定位最大的 allocation；
- 用 Nsight/NVTX 分析单个 TransformerBlock 保存了哪些 backward residual，列出最大的五个来源，并根据 forward/backward 显存变化核对每层 gradient tensor 的大小。

这一部分为后面的 checkpointing 和 FSDP 提供基线：只有知道显存由参数、梯度、优化器状态、activation 和临时 buffer 中的哪一项主导，才能选择正确的分片或重计算策略。

## 4. 第 3 章：Single-GPU Memory

### 4.1 Autograd residuals

PyTorch backward 需要 forward 中的中间值。Autograd 保存的这些张量称为 saved tensors 或 residuals。文档用手写 RMSNorm 和 `torch.autograd.graph.saved_tensors_hooks` 展示：一个看起来简单的表达式可能让 autograd 保存多份完整大小 activation 和多个归约中间量。

这说明代码中“数学表达式很短”并不代表 backward 的内存占用也小。算子边界和 autograd graph 的粒度会直接决定要保存什么。

### 4.2 Operator fusion

将 RMSNorm 交给 `torch.compile` 融合后，autograd 把它更接近视为一个整体，保存的完整 activation 数量明显减少。Operator fusion 的收益有两类：

- 避免将多个中间结果反复写入/读出 HBM；
- 减少 backward 所需保存的中间张量。

但 fusion 不能消除所有 activation。文档中的 xl TransformerBlock 即使经过尽量融合，单层仍保存约 3651.31 MiB residual；32 层合计约 114 GiB，明显超过普通单卡显存。

### 4.3 Activation checkpointing / recomputation

`torch.utils.checkpoint.checkpoint` 的逻辑是：

- forward 时只保存 checkpoint 区域入口的输入，不保存区域内部的中间张量；
- backward 到达该区域时重新跑一次 forward，临时重建所需 residual，然后立即执行该区域的 backward 并释放它们。

这会把显存分成两类：长期存在的 checkpoint 输入，以及某个正在重计算区域内短期物化的 residual。Checkpoint 太少时，单个重计算区域很大；checkpoint 太多时，保存的边界输入变多。峰值显存由两者的平衡决定。

文档示例中，四个 block 不 checkpoint 时保存约 14605.25 MiB；将每两个 block 包成一个 checkpoint 后，forward 长期保存的张量约为 160 MiB。但这并不表示 backward 峰值也只有 160 MiB，因为重计算时仍会临时物化区域内部 residual。

`gradient_checkpointing`（4 分）分两部分：

- 在允许任意嵌套 checkpoint、忽略计算成本时，设计使 N 个顺序 block 的峰值 activation memory 最小的策略，并分析渐近显存与计算量；
- 对 xl、batch 4、sequence 2048，在只允许一层重计算（不可嵌套）的限制下，用实测比较相邻 checkpoint block size，寻找峰值最低的划分。

## 5. 第 4 章：GPU Kernels 与 FlashAttention-2

这是整份作业中最集中的 GPU kernel 编程部分。

### 5.1 为什么普通 attention 有问题

标准 attention 为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\operatorname{mask}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right)V.
$$

普通实现会显式物化每个 batch/head 的 $N\times N$ attention score/probability 矩阵。它的计算量随序列长度平方增长，保存给 backward 的中间张量和 HBM IO 也随 $N^2$ 增长。长序列首先经常不是算不动，而是因为 attention matrix 和其 backward residual 放不下。

`pytorch_attention`（2 分）要求 batch 固定为 8、去掉 head 维度，遍历：

- head embedding dimension：16、32、64、128；
- sequence length：256、1024、4096、8192、16384；

测量 100 次 forward 和 100 次 backward、backward 前显存，并报告 OOM 边界及显存核算。

### 5.2 `torch.compile` 基线

PyTorch 2 的编译器能够分析计算图、融合算子并生成 Triton kernel。`torch_compile`（2 分）要求：

- 比较编译前后的普通 attention forward/backward；
- 比较整个 Transformer 编译前后，在 forward、forward-backward 和完整训练步上的表现。

这个实验用于区分“自动编译器已经能做的优化”和“必须通过算法重排、显式控制数据搬运才能做到的优化”。即使 `torch.compile` 能减少 launch 和中间张量，它也未必能从根本上消除长序列 attention 的 $N^2$ HBM 中间矩阵。

### 5.3 Triton 入门示例：weighted sum

PDF 用一个逐列加权再按行求和的算子介绍 Triton。主要概念包括：

- Triton program instance 相当于一个可并行执行的线程块；
- kernel 接收指向内存首元素的 pointer 和各维 stride，而不是高层 Tensor；
- `tl.program_id` 选择当前 program 负责的 tile；
- `tl.make_block_ptr` 描述全局 shape、stride、offset、block shape 和内存顺序；
- `tl.load`/`tl.store` 显式完成内存读写；
- `boundary_check` 与 zero padding 处理 tile 越界；
- `advance` 沿某一维移动 block pointer；
- launch grid 决定启动多少 program instance。

示例还展示了如何用 `torch.autograd.Function` 包装 Triton kernel。Forward 保存 backward 所需输入，backward 根据链式法则分别计算输入梯度和权重梯度。由于权重梯度需要跨 row tile 归约，各 program 先写 partial gradient，最后再由 PyTorch 做一次求和。

这一小节的目的不只是教语法，而是在正式写 FlashAttention 前建立三种意识：tile 如何分工、数据何时进入/离开片上存储、跨 program 的 reduction 如何处理。

### 5.4 FlashAttention-2 的核心思想

文档把普通 attention 的问题总结为：forward/backward 会把巨大的 $P$ 矩阵反复在片上 SRAM 与 HBM 之间搬运。FlashAttention-2 用三种技术消除这一瓶颈：

1. **Tiling**：沿 query 和 key 维度分块，不对隐藏维度分块；
2. **Recomputation**：不保存完整 $P$，backward 时利用 $Q,K,V$ 和辅助统计量重建局部概率；
3. **Operator fusion**：在一个 kernel 内完成 score、mask、online softmax 和乘 V，避免中间矩阵落到 HBM。

它不是近似 attention；目标仍然是计算数学上等价的精确 attention，只是改变计算顺序和内存访问方式。

### 5.5 Online softmax

对每个 query tile，算法维护三个片上状态：

- $m$：迄今为止每一行的最大 score，用于数值稳定；
- $l$：经过最大值修正后的 softmax 分母累计量；
- $O$：尚未最终归一化的输出累计量。

每读入一个新的 key/value tile，先计算局部 $S=QK^T/\sqrt d$，再更新行最大值。由于最大值可能变化，旧的 $l$ 和 $O$ 都必须乘以相应的指数缩放因子，再与当前 tile 的贡献合并。处理完全部 key tile 后，用最终 $l$ 归一化 $O$，并保存：

$$
L_i=\log\sum_j e^{S_{ij}}=m_i+\log l_i.
$$

$L$ 只随 query 数量线性增长，却足以让 backward 重建每个局部 tile 的概率：

$$
P_{ij}=\exp(S_{ij}-L_i).
$$

这就是避免保存完整 $P$ 的关键。

### 5.6 Forward 实现任务

`flash_forward`（15 分）是本章最大题之一，分三步：

1. 先用纯 PyTorch 在 `torch.autograd.Function` 中按 tile 实现 Algorithm 1，暂不做 backward，用它作为易调试的参考；
2. 写单个 Triton fused forward kernel，launch grid 为 `(T_q, batch_size)`，每个 program 负责一个 batch 的一个 query tile，并只在内部循环 key tile；
3. 增加可选 causal mask，Triton 参数必须是 `is_causal: tl.constexpr`，根据 query/key 绝对下标构造 mask。

数值精度要求也很明确：片上 $O,l,m$ buffer 使用 FP32；矩阵乘法前按需要转到 V 的 dtype；写回输出时再转换到目标 dtype。测试维度均为不小于 16 的 2 的幂，因此基础测试不要求处理任意形状。

对应适配器与测试：

- `adapters.get_flashattention_autograd_function_pytorch`；
- `uv run pytest -k test_flash_forward_pass_pytorch`；
- `adapters.get_flash_autograd_function_triton`；
- `uv run pytest -k test_flash_forward_pass_triton`。

### 5.7 Backward 的重计算公式

普通 backward 需要 $P$。FlashAttention backward 先计算一个行向量：

$$
D=\operatorname{rowsum}(O\circ dO),
$$

然后对 score/probability tile 进行重计算，并利用：

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i)
$$

继续得到 $dQ,dK,dV$。`flash_backward`（5 分）要求先不用 Triton，而是用 PyTorch 写公式并通过 `torch.compile` 编译，输入包括 $Q,K,V,O,dO,L$。

### 5.8 FlashAttention benchmark 与可选 Triton backward

`flash_benchmarking`（5 分）使用 `triton.testing.do_bench`，在单张 B200 上比较普通 PyTorch attention 与该作业的 FlashAttention：

- batch size = 1；
- causal masking；
- sequence length 从 128 到 65536 的若干 2 的幂；
- embedding dimension 从 16 到 128 的若干 2 的幂；
- BF16 和 FP32；
- 分别报告 forward、backward、端到端 forward-backward latency。

第 4.2.3 节是可选题，给出了 tiled Triton backward 算法。它将 $dQ$ 与 $dK/dV$ 分两遍计算，从而避免不同 program block 之间的同步或昂贵 atomic。它对 leaderboard 很有价值，但不是基础必做问题。

## 6. 第 5 章：Distributed Data Parallel Training

### 6.1 PyTorch 分布式通信基础

文档从四个进程各自产生 tensor、然后 `all_reduce` 求和的示例开始。关键 API 和概念是：

- `torch.multiprocessing.spawn` 启动多个 worker process；
- 每个 worker 得到唯一的 `rank`；
- `dist.init_process_group` 初始化进程组；
- `world_size` 是进程组内 worker 总数；
- rank 0 作为 master，通过 `MASTER_ADDR` 和 `MASTER_PORT` 协调；
- `all_reduce` 原地将每个 rank 的 tensor 替换为规约后的相同结果。

单节点多卡时 global rank 与 local rank 通常相同，但多节点时必须区分：global rank 在整个进程组唯一，local rank 只表示当前机器上的设备序号。每个 rank 必须绑定到不同 GPU，可调用 `torch.cuda.set_device(rank)` 或显式使用 `cuda:{rank}`。

后端选择：

- Gloo：可在 CPU 上调试；
- NCCL：GPU collective 的正常选择，正式 GPU benchmark 应使用它。

### 6.2 分布式 benchmark 的测量规则

文档建议：

- 对比实验尽量在同一机器上进行；
- NCCL 至少预热约 5 次；
- 即便 `async_op=False`，GPU benchmark 仍要 `torch.cuda.synchronize()`，因为调用返回只保证操作已排队；
- 不同 rank 时间略有差异，应考虑用 `all_gather_object` 收集各 rank 结果后统一统计；
- 本地先用 CPU/Gloo 调通正确性，再切换到 GPU/NCCL。

`distributed_communication_single_node`（5 分）要求测量 float32 all-reduce：数据大小 1 MB、10 MB、100 MB、1 GB；GPU/process 数量 2、4、6；用表或图展示规模、设备数和通信时间的关系。

### 6.3 Naïve DDP

最基础的数据并行流程是：

1. 各 rank 建立模型，rank 0 将初始参数 broadcast 给其他 rank；
2. 全局 batch 沿 batch 维均匀分给各 rank；
3. 每个 rank 对本地子 batch 做 forward/backward；
4. 对每个参数的 gradient 做 all-reduce 并除以 world size，得到全局平均梯度；
5. 每个 rank 独立运行相同 optimizer step。

因为各 rank 起始参数、optimizer state 和每一步使用的平均 gradient 都相同，更新后模型仍保持一致。

`naive_ddp`（5 分）要求实现逐参数 gradient all-reduce，并通过 `tests/test_ddp.py`。`naive_ddp_benchmarking`（3 分）在 1 节点、2 GPU、xl 模型上测每训练步总时间以及 gradient 通信占比。

### 6.4 降低 collective 调用次数

逐参数 all-reduce 的问题是小 collective 太多，每次调用都有固定 latency。第一种改进是将全部 parameter gradients flatten/concatenate 成一个大 tensor，只做一次 all-reduce，再 unflatten 回原形状。

`minimal_ddp_flat_benchmarking`（2 分）要求在同样的 1 节点、2 GPU、xl 条件下比较单次 batched all-reduce 和逐参数 all-reduce。

这种方法减少了调用次数，却仍要等整个 backward 结束才开始通信。因此它改善的是 collective latency，未解决通信位于关键路径的问题。

### 6.5 让 backward 与通信重叠

Backward 从输出层向输入层逐层产生 gradient。只要某个参数的 gradient 已完整累积，就可以立刻异步 all-reduce，而无需等待其他层。

文档建议用：

- `register_post_accumulate_grad_hook`：在参数 gradient 完成累积后触发通信；
- `dist.all_reduce(..., async_op=True)`：立即获得 request handle；
- `finish_gradient_synchronization()`：optimizer step 前逐一 `handle.wait()`，确保依赖 gradient 的更新可以安全排队。

`ddp_overlap_individual_parameters`（5 分）要求实现一个包装任意 `nn.Module` 的 DDP container，包括初始参数 broadcast、forward 代理、异步 gradient averaging 和完成同步接口。测试应重复运行，以发现偶发 race condition。

`ddp_overlap_individual_parameters_benchmarking`（1 分）要求：

- 与逐参数同步 all-reduce、单个 flatten all-reduce 比较训练步时间；
- 用 Nsight 截图明确展示初版没有 overlap，而 hook + async 版本确实将通信嵌入 backward compute 时间线。

这一节揭示了一个重要取舍：flatten 减少调用次数但延迟通信开始；逐参数异步能尽早通信但产生更多小 collective。生产级 DDP 通常用 bucket，在二者之间折中。

## 7. 第 6 章：Optimizer State Sharding

普通 DDP 在每张卡上复制完整参数、梯度和 optimizer state。AdamW 对每个参数通常保存一阶矩和二阶矩两个浮点状态，因此 optimizer state 可能是参数权重显存的两倍。

本章实现简化的 optimizer state sharding：

- 参数集合在各 rank 之间分配；
- 每个 rank 的本地 optimizer 只持有并更新自己负责参数的状态；
- 本地 optimizer step 后，由负责该参数的 rank broadcast 更新后的参数；
- 最终每个 rank 仍拥有完整且一致的模型参数，但只保存约 $1/\text{world\_size}$ 的 optimizer state。

`optimizer_state_sharding`（15 分）要求包装任意 `torch.optim.Optimizer`，建议接口包括：

- 构造函数接收参数或 parameter groups、optimizer class 与其关键字参数；
- `step` 调用内部 optimizer，再同步更新后的参数；
- `add_param_group` 支持构造期间以及训练途中动态加入参数组；
- 正确调用 `torch.optim.Optimizer` 父类构造函数。

测试入口是 `adapters.get_sharded_optimizer` 和 `tests/test_sharded_optimizer.py`，同样建议多跑几次以检查分布式稳定性。

`optimizer_state_sharding_accounting`（5 分）要求在 1 节点、2 GPU、xl 上：

- 比较普通与 sharded optimizer 在模型初始化后、optimizer step 前、step 后的显存；
- 将显存拆成参数、梯度、optimizer state 等来源；
- 比较每 iteration 时间；
- 解释这个简化实现与 ZeRO stage 1 在内存和通信量上的差异。

一个容易忽视的现象是：Adam 的状态常在第一次 `step()` 时惰性创建，所以“optimizer 构造后”和“第一次 step 后”的显存不能混为一谈。

## 8. 第 7 章：Fully-Sharded Data Parallel

Optimizer state sharding 仍在每卡保存完整参数。FSDP 进一步将参数、gradient 和 optimizer state 都沿 data-parallel rank 分片。

### 8.1 FSDP 的张量生命周期

对一个被分片的 Linear/Embedding 层，典型生命周期是：

```text
长期驻留：本 rank 的 FP32 master-weight shard
  → forward 前 all-gather 得到完整计算权重
  → 执行该层 forward
  → 释放完整权重
  → backward 前再次 all-gather 完整权重
  → 执行该层 backward
  → 对完整/局部贡献的梯度做 reduce-scatter
  → 只留下本 rank 的 gradient shard
  → 本地 AdamW 更新 master-weight shard
```

若等到层真正需要权重时才 all-gather，GPU compute 会空等通信。作业要求预取，但为了限制峰值显存，forward 中只能在“当前层之前两层已经完成”之后开始 gather。换言之，要在足够提前与不过度提前之间控制 in-flight 的完整权重数量。

LayerNorm/RMSNorm 等小层计算与参数都较小，通信固定开销可能不划算，所以文档建议主要分片 Linear 和 Embedding。

### 8.2 Mixed precision 与 master weights

反复由 optimizer 累积更新的 master weights 应保留 FP32；但用于 forward/backward matmul 的完整权重可在 all-gather 前转为 BF16/FP16。这样不仅计算使用低精度，通信字节数也减半。

`fsdp`（15 分）要求包装任意完整模型，并完成：

- Linear/Embedding 参数分片；
- forward/backward 所需权重的异步 all-gather 与预取；
- 用后及时释放 gathered weights；
- gradient reduce-scatter；
- `finish_gradient_synchronization()`；
- 可选 `compute_dtype`，同时保持 FP32 master weights 和 optimizer update；
- 与 Assignment 1 的标准 AdamW 兼容。

测试入口是 `adapters.get_fsdp` 和 `tests/test_fsdp.py`，需要多次执行以检查竞态。

`fsdp_accounting`（5 分）要求：

- 基于第 6 章的显存核算，预测 FSDP 能进一步节省多少峰值显存（可忽略预分配 all-gather buffer）；
- 在两张 GPU 上 profile xl 模型，观察 weight all-gather 是否能在 forward 使用前完成，并用 Nsight 截图和时间数据支持结论。

## 9. 第 8 章：并行策略的理论分析

前几章是实现和实测，这一章建立简化的计算—通信模型，用它判断增加 GPU 后何时会从 compute-bound 变成 communication-bound。

文档列出五种常见并行轴：

- DP：切 batch，聚合 weight gradient；
- FSDP：在 DP 基础上再切参数、梯度和 optimizer state；
- TP：切 weight matrix 的输入或输出维度，并通信 activation；
- PP：按层将模型切成 pipeline stage；
- EP：在 MoE 中将 experts 分布到不同设备。

本章重点分析 DP、FSDP、TP 和 FSDP + TP。

### 9.1 通信原语与 ring 模型

假设有 $N$ 个设备，每个设备的出口带宽为 $W$ bytes/s，总 tensor 大小为 $S$ bytes：

- ring all-gather 有 $N-1$ 轮，每轮每卡发送 $S/N$；
- ring reduce-scatter 也有相同的轮数和每轮流量；
- ring all-reduce 可由 reduce-scatter + all-gather 构成。

因此文档给出的理想时间为：

$$
T_{\text{all-gather}}=T_{\text{reduce-scatter}}
=\frac{N-1}{N}\frac{S}{W},
$$

$$
T_{\text{all-reduce}}
=2\frac{N-1}{N}\frac{S}{W}.
$$

`alternate_ring_all_reduce`（1 分）给出一种每轮转发完整 $S$ 大小 tensor 的替代算法，要求分析它的通信时间。这道题意在说明：算法同样正确，并不意味着通信量同样好。

### 9.2 统一 FFN 模型

后续计算都围绕 gated FFN：

$$
x_1=xW_1,\qquad x_2=xW_2,
$$

$$
z=f(x_1)\circ x_2,\qquad y=zW_3,
$$

其中 $x$ 的形状为 $(B,D)$，$W_1,W_2$ 为 $(D,D_{FF})$，$W_3$ 为 $(D_{FF},D)$。分析忽略非 matmul FLOP，并使用一次 $(A,B)\times(B,C)$ matmul 需要 $2ABC$ FLOP 的约定。设备计算能力记为 $C$ FLOP/s，出口带宽记为 $W$ bytes/s，权重与 activation 均按 FP16 的 2 bytes 计算。

通用分析步骤是：

1. 写出每卡 forward/backward 中各次 matmul 的 shape；
2. 求每卡总 FLOP，再除以 $C$ 得计算时间；
3. 列出 collective 的 tensor 大小和次数，用 ring 公式求通信时间；
4. 若通信与计算可以重叠，则比较二者最大值；
5. 令通信时间不大于计算时间，解出并行度上限。

### 9.3 Data Parallel 分析

DP 将 $B$ 切成 $N_{DP}$ 份。Forward 不需要 collective；backward 每卡只计算本地 batch 对完整 weight gradient 的部分和，随后对三个 weight gradient 做 all-reduce。

`data_parallel_calcs`（3 分）要求推导：

- 每卡 backward FLOP；
- backward 通信时间；
- 在通信与 backward compute 可重叠时，$N_{DP}$ 增长到多大开始通信受限。

这里的直觉是：DP 越大，每卡 batch 越小，所以每卡计算量下降；但要 all-reduce 的完整参数梯度大小不随本地 batch 缩小，因此最终通信会主导。

### 9.4 FSDP 分析

FSDP 同样切 batch，同时把每个 weight 切成 $N_{FSDP}$ 份：

- forward 前 all-gather weights，然后做 batch-sharded forward；
- backward 前再次 all-gather weights；
- backward 后不用 all-reduce 完整 gradient，而是 reduce-scatter，只留下本 rank 的 shard。

`fsdp_calcs`（3 分）分别要求 forward/backward 的 FLOP、通信时间以及各自不被通信限制的最大 $N_{FSDP}$。

与 DP 相比，FSDP 用更多 weight 通信换取参数/梯度/optimizer state 的显存分片。尤其 forward 在普通 DP 中没有 collective，而 FSDP forward 必须 all-gather 权重。

### 9.5 Tensor Parallel 分析

TP 有两种基本切法：

- column parallel：沿 weight 输出维切分，每卡产生一部分输出，必要时 all-gather；
- row parallel：沿 weight 输入维切分，每卡计算部分和，最后 all-reduce。

作业指定 $W_1,W_2$ 使用 column parallel，$W_3$ 使用 row parallel。这样 $x_1,x_2,z$ 始终保持 $D_{FF}$ 分片，不需要在中间 all-gather；直到 $W_3$ 输出处才对 activation 部分和做一次 all-reduce。

`tp_calcs`（4 分）要求：

- 写完整 TP backward 方程，包括每卡 weight gradient 和最终 $dx$；
- 计算 forward/backward FLOP；
- 计算 forward/backward activation 通信；
- 推导 $N_{TP}$ 的 compute-bound 上限。

TP 与 DP/FSDP 的关键区别是通信对象：DP 主要通信与模型大小相关的 parameter gradient，TP 主要通信与 batch/sequence activation 大小相关的 tensor。

### 9.6 二维并行：FSDP + TP

设备排成 $N_{TP}\times N_{FSDP}$ 二维网格，每卡有 TP rank 和 FSDP rank：

- TP 沿一个 weight 维度分片；
- FSDP 再沿另一个 weight 维度分片；
- batch 沿 FSDP 轴分片；
- FSDP 轴对 weight 做 all-gather；
- TP 轴对 activation 做 all-reduce；
- 总设备数 $N=N_{TP}N_{FSDP}$。

`fsdp_tp_calcs`（6 分）要求：

- 求二维并行 forward 每卡 FLOP；
- 若两个轴的网络通信可以重叠，将通信时间写成 FSDP 成本与 TP 成本的 `max`；
- 选择最优 $N_{TP}$、$N_{FSDP}$，求可保持 compute-bound 的最大总设备数；
- 再分析两个轴共享网络、通信不能重叠时的最优配置与扩展上限。

这一节的主要思想是：不同并行方式受不同维度限制。把设备只堆在一个并行轴上会较早遇到通信瓶颈；二维并行通过平衡两种通信成本获得更大的总扩展规模。

## 10. 第 9 章：Leaderboard

Leaderboard 测试一个约 8B 的模型，在两张 B200 上完成完整训练步：forward、loss、backward 和 AdamW update。

配置为：

| 参数 | 值 |
|---|---:|
| batch size | 2 |
| context length | 32768 |
| vocabulary size | 151936 |
| `d_model` | 4096 |
| `d_ff` | 11008 |
| layers | 34 |
| heads | 32 |
| dtype | BF16 |
| mask | causal |

限制和目标：

- 输入输出行为必须与 `cs336_basics` 模型一致；
- 实现必须是自己的，不能直接使用或复制已有实现；
- 从空 PyTorch/Triton cache 开始，整次 benchmark 必须在 10 分钟内完成，因此过度 `torch.compile` 或 autotune 也会失败；
- 期望击败 10 秒的 naïve baseline；
- 最快的 5–10 个提交会额外验证正确性与性能。

PDF 给出的优化方向包括：

- 调整 Triton tile size，谨慎使用 autotune；
- 调整 Triton/`torch.compile` 配置；
- fused AdamW；
- 融合 LM head 与 cross-entropy，避免物化巨大的 `[batch, seq_len, vocab_size]` logits；
- 用 Triton 实现 FlashAttention backward；
- $dQ$ 与 $dK/dV$ 分遍计算，避免 atomic/synchronization；
- causal attention 中提前跳过完全被 mask 的 tile；
- 将非对角、无需 mask 判断的 tile 与对角 tile 分开处理；
- 在 Hopper 之后的架构利用 TMA；
- 只有在显存确实需要时才用 activation checkpointing，因为它会增加计算。

`leaderboard`（10 分）的交付物是最佳完整训练步 wall-clock time。

## 11. 全部计分问题与交付物索引

| 章节 | 问题 | 分值 | 主要类型 | 核心交付物 |
|---|---|---:|---|---|
| 2 | `benchmarking_script` | 4 | 实现 + 实验 | 三种训练阶段的 benchmark 脚本、均值/标准差、warm-up 分析 |
| 2 | `nsys_profile` | 5 | Profiling | 多组模型/上下文的 kernel、时间占比和 NVTX 分析 |
| 2 | `mixed_precision_accumulation` | 1 | 问答 | 不同 dtype 累加精度分析 |
| 2 | `benchmarking_mixed_precision` | 2 | 问答 + 实验 | autocast dtype 分析与 BF16 benchmark |
| 2 | `memory_profiling` | 4 | Profiling | memory timeline、峰值表格、allocation/residual 分析 |
| 3 | `gradient_checkpointing` | 4 | 理论 + 实验 | 最优渐近策略和 xl 实测最优 checkpoint 粒度 |
| 4 | `pytorch_attention` | 2 | Benchmark | attention 时延/OOM/显存表与分析 |
| 4 | `torch_compile` | 2 | Benchmark | attention 与完整 Transformer 编译前后对比 |
| 4 | `flash_forward` | 15 | 实现 | PyTorch tiled reference、Triton forward、causal mask |
| 4 | `flash_backward` | 5 | 实现 | PyTorch + `torch.compile` 的重计算 backward |
| 4 | `flash_benchmarking` | 5 | Benchmark | PyTorch 与 FlashAttention 的多维度 latency 表 |
| 5 | `distributed_communication_single_node` | 5 | 分布式实验 | 不同数据量和 GPU 数的 all-reduce 图表 |
| 5 | `naive_ddp` | 5 | 实现 | 逐参数 gradient all-reduce DDP |
| 5 | `naive_ddp_benchmarking` | 3 | Benchmark | xl 两卡训练步与通信占比 |
| 5 | `minimal_ddp_flat_benchmarking` | 2 | 实现 + 实验 | flatten 后单 collective 的性能 |
| 5 | `ddp_overlap_individual_parameters` | 5 | 实现 | hook + async all-reduce DDP container |
| 5 | `ddp_overlap_individual_parameters_benchmarking` | 1 | Profiling | 三种 DDP 对比及 overlap Nsight 截图 |
| 6 | `optimizer_state_sharding` | 15 | 实现 | 通用 sharded optimizer wrapper |
| 6 | `optimizer_state_sharding_accounting` | 5 | 实验 + 问答 | 显存、速度与 ZeRO stage 1 对比 |
| 7 | `fsdp` | 15 | 实现 | 权重 all-gather、梯度 reduce-scatter、预取与混合精度 |
| 7 | `fsdp_accounting` | 5 | 实验 | 峰值显存预测与 all-gather overlap 证据 |
| 8 | `alternate_ring_all_reduce` | 1 | 理论 | 替代 ring 算法时间复杂度 |
| 8 | `data_parallel_calcs` | 3 | 理论 | DP FLOP、通信时间、扩展上限 |
| 8 | `fsdp_calcs` | 3 | 理论 | FSDP forward/backward 计算与通信上限 |
| 8 | `tp_calcs` | 4 | 理论 | TP backward 方程、计算/通信与扩展上限 |
| 8 | `fsdp_tp_calcs` | 6 | 理论 | 二维并行最优配比和扩展上限 |
| 9 | `leaderboard` | 10 | 综合优化 | 两张 B200 上 8B 模型完整训练步最佳时间 |
|  | **合计** | **137** |  |  |

此外，第 4.2.3 节的 Triton FlashAttention backward 是明确标注的 OPTIONAL 内容，不另列基础分值。

## 12. 关键测试与适配器

| 功能 | 适配器 | 文档给出的测试方式 |
|---|---|---|
| PyTorch tiled FlashAttention forward | `get_flashattention_autograd_function_pytorch` | `uv run pytest -k test_flash_forward_pass_pytorch` |
| Triton FlashAttention forward | `get_flash_autograd_function_triton` | `uv run pytest -k test_flash_forward_pass_triton` |
| FlashAttention backward | 对应 FlashAttention adapter | `uv run pytest -k test_flash_backward` |
| DDP | `get_ddp`、可选 `ddp_on_after_backward` | `uv run pytest tests/test_ddp.py` |
| Sharded optimizer | `get_sharded_optimizer` | `uv run pytest tests/test_sharded_optimizer.py` |
| FSDP | `get_fsdp` | `uv run pytest tests/test_fsdp.py` |

分布式测试建议重复约 5 次，因为异步通信和 hook 顺序可能让竞态只在少数运行中出现。

## 13. 贯穿全文的关键概念

### 13.1 FLOP、带宽和延迟是不同瓶颈

- 大 matmul 常受计算吞吐限制；
- elementwise/reduction 可能受 HBM 带宽限制；
- 大量小 kernel 或小 collective 可能受 launch latency 限制；
- 只看 FLOP 数无法预测真实运行时间。

### 13.2 峰值显存不是各项静态大小的简单相加

训练中 tensor 的生命周期不同：forward residual 逐层积累；backward 一边释放 residual，一边产生 gradient；Adam state 可能在第一次 step 惰性创建；all-gather 还会产生短期完整权重和通信 buffer。因此既要做理论 accounting，也要看 timeline。

### 13.3 降显存通常要付出计算或通信

- Checkpointing：少存 activation，付出 forward 重计算；
- FlashAttention：不存 $N^2$ attention matrix，付出局部重计算，但通常因 IO 大幅下降而更快；
- Optimizer sharding：少存 optimizer state，付出参数同步；
- FSDP：少存参数和 gradient，付出 forward/backward weight all-gather 与 gradient reduce-scatter。

### 13.4 Overlap 只在通信及时完成时才有效

把 collective 标记为 async 并不自动等于性能提升。真正的问题是：通信是否足够早地发出，是否与独立计算并行，以及依赖该结果的操作开始前它是否已经完成。Nsight 时间线是验证这些条件的直接证据。

### 13.5 正确性、稳定性和性能必须分开验证

建议的顺序是：

1. 用小 shape、FP32、CPU/Gloo 或纯 PyTorch reference 验证数学正确性；
2. 用测试检查 dtype、shape、causal mask、gradient 和多 rank 一致性；
3. 重复分布式测试排查 race；
4. 再在目标 GPU/NCCL/BF16 环境测性能；
5. 最后用 profiler 证明优化确实改变了预期的 kernel、IO、显存或通信时间线。

## 14. 推荐完成顺序

结合题目依赖关系，一个自然的推进顺序是：

1. 先完成可复用的模型配置、随机数据和 benchmark harness；
2. 加入 NVTX、mixed precision、compile 和 memory snapshot 开关；
3. 用现有 attention 做规模扫描，建立普通实现的性能/显存基线；
4. 先写 PyTorch tiled FlashAttention reference，再写 Triton forward，最后接 backward；
5. 单独把 distributed process-group、rank-device mapping 和 collective benchmark 调通；
6. 依次实现 naïve DDP、flatten DDP、overlapped DDP；
7. 在 DDP 正确的基础上实现 optimizer state sharding；
8. 最后实现生命周期和异步依赖最复杂的 FSDP；
9. 用前面的实测数据帮助理解第 8 章公式，并完成理论分析；
10. 只有在所有正确性测试稳定通过后，再针对 leaderboard 做融合和 autotuning。

## 15. 总结

这份作业把单卡 kernel 优化和多卡并行放在同一套“计算、显存、通信”框架中：

- Profiling 告诉你瓶颈在哪里；
- mixed precision、fusion、FlashAttention 改善单卡吞吐和 IO；
- checkpointing 用计算换 activation 显存；
- DDP 用更多设备分摊 batch compute，但引入 gradient communication；
- optimizer sharding 和 FSDP 逐步去除数据并行中的冗余状态；
- TP 与二维并行通过改变分片维度，让更大规模训练仍能维持 compute-bound。

完成这份作业后，应该不仅能使用 PyTorch 的现成分布式接口，还能解释这些系统为什么这样设计、性能瓶颈如何形成，以及如何用 benchmark、profile、理论 accounting 三种证据共同验证一个优化。
