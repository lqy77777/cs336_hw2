# Memory-bound、Compute-bound 与计算强度

## 1. 先建立一个直觉

执行一个 GPU kernel 时，通常需要做两类工作：

1. **搬运数据**：从 HBM（显存）读取输入、读取参数，并把结果写回 HBM。
2. **执行计算**：例如加法、乘法、矩阵乘法和指数运算。

如果数据搬运所需的时间更长，kernel 就是 **memory-bound**；如果计算所需的时间更长，kernel 就是 **compute-bound**。

可以把 GPU 想成一家餐厅：

- 厨师做菜的速度对应计算吞吐量；
- 仓库向厨房送食材的速度对应内存带宽；
- 菜需要大量食材、但加工很少时，送货速度决定出餐速度；
- 同一批食材能被反复加工很多次时，厨师的速度更可能成为瓶颈。

这里的关键不是一个算子“看起来是否复杂”，而是：**每搬运一个 byte 的数据，能够完成多少次有用计算。**

---

## 2. 两种主要性能瓶颈

设一个 kernel：

- 总计算量为 $F$，单位 FLOP；
- HBM 数据传输量为 $Q$，单位 byte；
- 加速器计算吞吐上限为 $C$，单位 FLOP/s；
- HBM 带宽上限为 $B$，单位 byte/s。

只考虑计算时，至少需要：

$$
T_{\text{compute}} = \frac{F}{C}
$$

只考虑数据搬运时，至少需要：

$$
T_{\text{memory}} = \frac{Q}{B}
$$

在理想情况下，计算和访存可以部分重叠，因此常用下面的近似下界：

$$
T \gtrsim \max\left(T_{\text{compute}}, T_{\text{memory}}\right)
$$

### Memory-bound（内存带宽受限）

当

$$
T_{\text{memory}} > T_{\text{compute}}
$$

时，计算单元经常在等待数据，增加更多计算单元或提高峰值 FLOPS 通常帮助不大。逐元素操作、向量加法、许多归一化操作以及 batch 很小的矩阵—向量乘法往往属于这一类。

常见优化方向是减少 HBM 流量：

- 融合多个 kernel，避免中间结果反复写回、读出 HBM；
- 使用 tiling，让数据在寄存器、shared memory 或 cache 中被重复利用；
- 使用更低精度的数据类型，减少每个元素的字节数；
- 改善连续、合并的内存访问；
- 适当增大 batch，让权重被更多样本复用；
- 避免不必要的 tensor materialization、复制和 layout 转换。

### Compute-bound（计算吞吐受限）

当

$$
T_{\text{compute}} > T_{\text{memory}}
$$

时，数据供应已经足够快，主要时间花在数学运算上。规模足够大的矩阵—矩阵乘法通常属于这一类。

常见优化方向是提高有效计算吞吐或减少计算量：

- 使用 Tensor Cores 支持的精度和矩阵形状；
- 采用高效的 GEMM、attention 等 kernel；
- 减少不必要的 FLOP，选择更高效的算法；
- 改善并行度、occupancy 和指令调度；
- 在数值精度允许时采用 FP16、BF16、FP8 等低精度计算。

> Compute-bound 不等于“程序已经足够快”，memory-bound 也不等于“代码一定很差”。它们只说明在当前实现和硬件上，哪一类资源构成主要上限。

---

## 3. Arithmetic intensity

**Arithmetic intensity（算术强度，简称 AI）**定义为计算量与数据传输量之比：

$$
I_{\text{arith}} = \frac{F}{Q}
$$

单位是：

$$
\frac{\text{FLOP}}{\text{byte}}
$$

它回答的问题是：**每从目标内存层级搬运 1 byte 数据，kernel 做了多少 FLOP？**

- AI 低：搬很多数据，只做少量计算，更可能 memory-bound；
- AI 高：数据被反复利用，做大量计算，更可能 compute-bound。

### FLOP 应该怎样计数？

在深度学习的常见约定中：

- 一次加法算 1 FLOP；
- 一次乘法算 1 FLOP；
- 一次 fused multiply-add（FMA）通常算 2 FLOP。

因此，矩阵乘法 $A_{M\times K}B_{K\times N}$ 的计算量通常近似为：

$$
F \approx 2MKN
$$

分析时应明确自己的计数约定。不同资料可能对比较、指数、除法等操作采用不同的等价 FLOP 数，因此数字不一定能直接横向比较。

### 数据传输量应该怎样计数？

必须先明确所分析的**内存边界**：

- HBM arithmetic intensity：分母是 HBM 与芯片之间的流量；
- L2 arithmetic intensity：分母是 L2 与更靠近计算单元的存储层之间的流量；
- shared-memory intensity：还可以继续分析更近的存储层级。

同一个 kernel 在不同边界上会有不同的 AI。Roofline 分析中通常最先考察 HBM 流量，但不能把“tensor 的逻辑大小”无条件当作实际流量：cache 命中、重复加载、写回、非合并访问和临时 tensor 都可能改变真实传输量。

理论上根据算法最少需要的数据流量得到的值，常称为 **algorithmic arithmetic intensity**；根据 profiler 测得的实际流量计算出的值，也常称为 **operational intensity**。实际讨论中这两个术语有时会被混用。

---

## 4. Accelerator intensity

在 CS336 的语境中，**accelerator intensity** 指硬件峰值计算吞吐与峰值内存带宽之比：

$$
I_{\text{acc}} = \frac{C}{B}
$$

单位也是 FLOP/byte。它表示：**为了让这块加速器的计算单元以峰值速度持续工作，每搬运 1 byte 数据，工作负载至少要提供多少 FLOP。**

这个量也常被称为：

- machine balance；
- hardware balance；
- machine/processor ops:byte ratio；
- Roofline ridge point（Roofline 的转折点）。

“Accelerator intensity” 并不是所有文献都统一采用的标准名称，但其含义就是上述硬件比值。

### 为什么比较两个 intensity 就能判断瓶颈？

从

$$
\frac{F}{C} \quad \text{和} \quad \frac{Q}{B}
$$

比较两种时间。两边相除或移项可得：

$$
\frac{F}{Q} \quad \text{和} \quad \frac{C}{B}
$$

也就是比较 arithmetic intensity 和 accelerator intensity：

$$
\boxed{
\begin{aligned}
I_{\text{arith}} < I_{\text{acc}} &\Rightarrow \text{memory-bound} \\
I_{\text{arith}} > I_{\text{acc}} &\Rightarrow \text{compute-bound} \\
I_{\text{arith}} = I_{\text{acc}} &\Rightarrow \text{位于理想转折点附近}
\end{aligned}}
$$

注意：accelerator intensity 取决于使用的计算精度和硬件执行路径。某块 GPU 的 FP32、Tensor Core BF16、FP16 和 FP8 峰值吞吐不同，因此对应的 $I_{\text{acc}}$ 也不同。

---

## 5. Roofline model

Roofline 模型把上述关系写成：

$$
P_{\text{attainable}} \leq \min\left(C,\;B I_{\text{arith}}\right)
$$

其中 $P_{\text{attainable}}$ 是可达到的计算性能，单位 FLOP/s。

- $BI_{\text{arith}}$ 是斜线部分，即内存带宽所允许的性能上限；
- $C$ 是水平线部分，即计算单元所允许的峰值性能；
- 两条线在 $I_{\text{arith}}=C/B=I_{\text{acc}}$ 处相交。

```text
性能 (FLOP/s)
^
|                         ───────────── C：计算上限
|                       /
|                     /
|                   /
|                 /   斜率为 B：带宽上限
|               /
|_____________/________________________________> AI (FLOP/byte)
              I_acc = C / B
  memory-bound             compute-bound
```

### 一个假想硬件例子

假设某加速器具有：

- 100 TFLOP/s 的峰值计算吞吐；
- 1 TB/s 的 HBM 带宽。

则：

$$
I_{\text{acc}} = \frac{100\times10^{12}}{1\times10^{12}}
=100\ \text{FLOP/byte}
$$

对 arithmetic intensity 为 20 FLOP/byte 的 kernel：

$$
P \leq \min(100,1\times20)=20\ \text{TFLOP/s}
$$

它是 memory-bound。即使计算单元理论上能达到 100 TFLOP/s，内存最多只供得上 20 TFLOP/s 所需的数据。

对 arithmetic intensity 为 250 FLOP/byte 的 kernel：

$$
P \leq \min(100,1\times250)=100\ \text{TFLOP/s}
$$

此时 Roofline 的限制来自计算峰值，因此它是 compute-bound。

---

## 6. 常见算子的例子

以下均采用简化模型，只统计理想的 HBM 流量；实际实现还会受到 cache、融合方式和数据布局的影响。

### 6.1 FP16 ReLU：低 AI

对每个元素 $y=\max(0,x)$：

- 读一个 FP16 输入：2 byte；
- 写一个 FP16 输出：2 byte；
- 近似计一次比较操作：1 FLOP。

因此：

$$
I_{\text{arith}} \approx \frac{1}{2+2}
=0.25\ \text{FLOP/byte}
$$

这个值远低于现代 GPU 常见的硬件平衡点，所以 ReLU 通常是 memory-bound。它的关键优化通常不是让比较运算更快，而是把它和相邻算子融合，减少中间 tensor 的 HBM 读写。

### 6.2 FP16 向量加法：低 AI

对 $z=x+y$，每个元素：

- 读取 $x$ 和 $y$：4 byte；
- 写入 $z$：2 byte；
- 做一次加法：1 FLOP。

因此：

$$
I_{\text{arith}} \approx \frac{1}{6}
\approx0.167\ \text{FLOP/byte}
$$

它同样典型地是 memory-bound。

### 6.3 FP16 矩阵—向量乘法：权重只复用一次

对 $y=Ax$，其中 $A\in\mathbb{R}^{M\times K}$：

$$
F \approx 2MK
$$

理想数据流量约为：

$$
Q \approx 2(MK+K+M)\ \text{byte}
$$

当 $M,K$ 很大时，矩阵 $A$ 的读取占主导：

$$
I_{\text{arith}} \approx \frac{2MK}{2MK}
=1\ \text{FLOP/byte}
$$

每个权重从 HBM 读入后只服务于一个向量，数据复用很少，所以大模型的 batch-1 自回归解码经常受到权重读取带宽限制。

### 6.4 FP16 矩阵—矩阵乘法：高数据复用

对 $C=AB$，其中：

- $A\in\mathbb{R}^{M\times K}$；
- $B\in\mathbb{R}^{K\times N}$；
- $C\in\mathbb{R}^{M\times N}$。

计算量为：

$$
F\approx2MKN
$$

假设每个矩阵只从 HBM 读取或写入一次，FP16 的流量近似为：

$$
Q\approx2(MK+KN+MN)\ \text{byte}
$$

所以：

$$
I_{\text{arith}}
\approx
\frac{2MKN}{2(MK+KN+MN)}
=\frac{MKN}{MK+KN+MN}
$$

若 $M=N=K=n$，则：

$$
I_{\text{arith}}\approx\frac{n}{3}\ \text{FLOP/byte}
$$

矩阵越大，数据复用越多，AI 越高，因此规模足够大的 GEMM 通常会进入 compute-bound 区域。这里“每个矩阵只访问一次”依赖合理的 tiling；朴素实现若不断从 HBM 重复加载数据，实际 AI 会低得多。

---

## 7. 在 Transformer 中的典型情况

### 训练与 prefill

训练以及较大 batch/sequence 的 prompt prefill 会把许多 token 合并成较大的矩阵—矩阵乘法。权重可以被一批 token 反复使用，arithmetic intensity 较高，大型 GEMM 往往更接近 compute-bound。

### 自回归 decode

每一步通常只为每条序列生成一个 token。当 batch 较小时，许多线性层更像矩阵—向量乘法：必须读取大量模型权重，却只对每个权重完成很少的计算，因此往往 memory-bound。

增大 decode batch 可以让同一份权重同时服务更多 token，从而提高 AI；但 batch 还会增加 KV cache 流量、显存占用和单请求延迟，需要结合服务目标权衡。

### Elementwise、normalization 与 softmax

这些算子通常每个元素只执行少量运算，同时至少要读取并写回一次数据，所以常常是 memory-bound。Kernel fusion（例如 fused normalization 或 fused optimizer）对它们尤其重要。

### FlashAttention 的系统直觉

标准 attention 若显式生成并写回 $N\times N$ attention 矩阵，会产生大量 HBM 流量。FlashAttention 通过 tiling 和在线 softmax，让中间数据主要停留在片上 SRAM 中，减少 HBM 读写。它的核心收益可以理解为：减少通信量 $Q$，从而提高相对于 HBM 的 operational intensity。

---

## 8. 容易混淆的地方

### 8.1 AI 高不代表一定跑得快

AI 只描述计算与数据移动的比例，不描述这些计算是否高效执行。即使理论上 compute-bound，也可能因为以下问题远低于 Roofline：

- kernel 太小，无法占满 GPU；
- occupancy 低或寄存器压力过大；
- 指令依赖导致流水线停顿；
- 数据访问不合并；
- 频繁的 kernel launch；
- 分支发散；
- 没有使用适合的 Tensor Core 路径。

Roofline 给出的是性能上限和瓶颈分类，不是精确运行时间预测器。

### 8.2 增加无用计算虽然提高 AI，却不会让任务更高效

如果只是人为加入无意义的 FLOP，$F/Q$ 确实会上升，但完成原任务的时间不会因此缩短。真正有价值的是通过数据复用、融合、tiling 或算法调整，让每次数据移动对应更多**有用工作**。

### 8.3 Memory-bound 不等于 latency-bound

Roofline 主要讨论吞吐量上限。对于很小的 kernel，内存访问延迟、kernel launch 延迟或同步开销可能占主导；此时甚至没有足够的工作量来饱和峰值带宽，不能简单归类为“达到了 memory-bandwidth roof”。

### 8.4 单卡 Roofline 没有覆盖所有分布式瓶颈

多 GPU 训练还可能受 NVLink、PCIe 或网络通信限制。此时可以对网络边界建立类似模型，例如用“每个网络 byte 对应多少 FLOP”衡量通信强度，但不能只看 HBM arithmetic intensity。

### 8.5 理论峰值不等于可持续峰值

产品规格中的峰值 FLOPS 和带宽通常需要特定数据类型、形状、稀疏性或访问模式。更严谨的分析可以使用实测的 GEMM 吞吐和实测 HBM 带宽作为 $C$ 与 $B$，得到更符合当前环境的 Roofline。

---

## 9. 实际分析一个 kernel 的步骤

1. **明确分析边界**：HBM、L2、shared memory，还是跨 GPU 网络？
2. **计算有用 FLOP**：写出 tensor shape，并明确 FLOP 计数约定。
3. **估算最小数据流量**：统计所有输入读取和输出写入的 byte 数。
4. **计算 arithmetic intensity**：$I_{\text{arith}}=F/Q$。
5. **取得匹配精度的硬件指标**：计算 $I_{\text{acc}}=C/B$。
6. **比较两者**：初步判断 memory-bound 或 compute-bound。
7. **用 profiler 验证**：观察实际 HBM 吞吐、计算单元利用率、cache 命中和 stall 原因。
8. **针对瓶颈优化后重新测量**：不要只凭理论 AI 判断优化是否有效。

一个很实用的自检是单位分析：

$$
(\text{byte/s})\times(\text{FLOP/byte})=\text{FLOP/s}
$$

如果单位无法消去，通常说明公式或计数出了问题。

---

## 10. 一页总结

| 概念 | 定义 | 主要由谁决定 |
|---|---|---|
| Arithmetic intensity | $F/Q$，完成的 FLOP 除以传输的 byte | 算法、实现和所选内存边界 |
| Accelerator intensity | $C/B$，峰值 FLOP/s 除以峰值 byte/s | 硬件、数据精度和执行路径 |
| Memory-bound | $I_{\text{arith}}<I_{\text{acc}}$ | 数据供应跟不上计算能力 |
| Compute-bound | $I_{\text{arith}}>I_{\text{acc}}$ | 计算吞吐成为主要上限 |
| Roofline | $P\leq\min(C,BI_{\text{arith}})$ | 同时表达计算与带宽上限 |

最值得记住的关系是：

$$
\boxed{
P_{\text{attainable}}
\leq
\min\left(
\text{peak compute},
\text{memory bandwidth}\times\text{arithmetic intensity}
\right)}
$$

以及：

$$
\boxed{
\text{比较 workload 的 }\frac{\text{FLOP}}{\text{byte}}
\text{ 与 hardware 的 }\frac{\text{FLOP/s}}{\text{byte/s}}
}
$$

---

## 参考资料

- [Stanford CS336 Lecture 02：resource accounting 与 arithmetic intensity](https://cs336.stanford.edu/lectures/?trace=lecture_02)
- [NVIDIA Nsight Compute：Roofline Charts](https://docs.nvidia.com/nsight-compute/2021.3/ProfilingGuide/index.html#roofline-charts)
- [NVIDIA：GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [Google Cloud：AI accelerator performance and Roofline analysis](https://docs.cloud.google.com/docs/ai-ml/accelerator-performance-benchmarking)

