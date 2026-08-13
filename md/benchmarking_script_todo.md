# `benchmarking_script` 任务清单

本文档对应 `cs336_assignment2_systems.pdf` 中的：

```text
Problem (benchmarking_script): Benchmarking Script (4 points)
```

目标是完成一个可靠的端到端 benchmark，并用它回答不同模型规模和不同 warm-up 次数下的运行时间问题。

## 1. 题目规定的统一实验配置

除非题目另有说明，使用：

| 配置项 | 取值 |
| --- | ---: |
| `vocab_size` | 10,000 |
| `batch_size` | 4 |
| `context_length` | 512 |
| 正式实验设备 | CUDA GPU |
| Part (b) warm-up 次数 | 5 |
| Part (b) measurement 次数 | 10 |

需要测试的模型配置：

| 模型 | `d_model` | `d_ff` | `num_layers` | `num_heads` |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

如果某个配置因 GPU 显存不足而 OOM，应如实记录 GPU 型号、运行模式和 OOM，而不是虚构 timing。

## 2. Part (a)：实现 benchmark 脚本

脚本必须完成以下事情：

- [x] 根据模型超参数初始化 Transformer。
- [x] 在模型所在 device 上生成随机 input tokens 和 targets。
- [x] 支持设置 warm-up 次数 $w$。
- [x] 支持设置正式测量次数 $n$。
- [x] 支持 `forward` 模式。
- [x] 支持 `forward_backward` 模式。
- [x] 支持包含 optimizer step 的 `full_step` 模式。
- [x] 使用 `timeit.default_timer()` 或同等的高精度 wall-clock timer。
- [x] warm-up 不计入正式 measurement samples。
- [x] 每个 CUDA step 后调用 `torch.cuda.synchronize()`。
- [x] 计算 $n$ 个 samples 的平均值和标准差。

当前实现位于 `cs336_systems/benchmark.py`，Part (a) 的主体已经完成。

### 2.1 必须保证的计时结构

每个正式 measurement 应遵循：

```text
CUDA synchronize
开始计时
执行选定的 step
CUDA synchronize
结束计时
```

随机数据生成、模型初始化和 terminal 输出不能放进正式计时区间。

### 2.2 三种模式的准确含义

```text
forward:
    model(inputs)

forward_backward:
    清除旧 gradient
    model(inputs)
    计算 loss
    loss.backward()

full_step:
    清除旧 gradient
    model(inputs)
    计算 loss
    loss.backward()
    optimizer.step()
```

每次 backward 前都必须清除旧 gradient，否则 PyTorch 会累积 gradient，使不同 step 的状态不一致。

### 2.3 当前实现还应确认或完善的地方

- [ ] 明确报告的是三种模式的累计时间，还是三个阶段各自的独立时间。
- [ ] 若题目答案要直接给出 backward 和 optimizer 时间，最好在同一个 training step 内分别计时。
- [ ] 至少保存或打印每次 measurement 的原始 sample，便于检查异常值。
- [ ] 记录正式实验使用的 GPU 型号和 dtype。
- [ ] 考虑自动保存 JSONL/CSV，避免长实验结束后只剩 terminal 输出。
- [ ] 正式运行前确认所有模式使用相同的模型配置和 loss 定义。

当前三种模式测到的是：

$$
T_{F}
$$

$$
T_{F+B}
$$

$$
T_{F+B+O}
$$

如果不修改脚本，也可以用不同模式的均值之差粗略估算：

$$
T_B \approx T_{F+B}-T_F
$$

$$
T_O \approx T_{F+B+O}-T_{F+B}
$$

但这种方法会混入不同 benchmark run 之间的波动，因此不如在同一个 step 内分别设置同步和计时边界准确。

## 3. Part (b)：测量所有模型规模

需要对每个模型执行正式实验：

- [ ] `small`
- [ ] `medium`
- [ ] `large`
- [ ] `xl`
- [ ] `10B`

每个模型至少需要测量：

- [ ] forward；
- [ ] forward + backward；
- [ ] forward + backward + optimizer step；
- [ ] 5 次 warm-up；
- [ ] 10 次正式 measurements；
- [ ] mean；
- [ ] standard deviation。

当前脚本的一次运行示例：

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

需要分别改变 `--model-size` 和 `--mode`。为了减少手工抄写错误，建议另外写一个 sweep 入口自动遍历配置并保存结果，但这不是题目硬性要求。

### 3.1 建议结果表

| model | forward mean/std | backward mean/std | optimizer mean/std | full step mean/std | status |
| --- | ---: | ---: | ---: | ---: | --- |
| small | 待测 | 待测 | 待测 | 待测 | 待测 |
| medium | 待测 | 待测 | 待测 | 待测 | 待测 |
| large | 待测 | 待测 | 待测 | 待测 | 待测 |
| xl | 待测 | 待测 | 待测 | 待测 | 待测 |
| 10B | 待测 | 待测 | 待测 | 待测 | 待测 |

Part (b) 最终还需要写 1–2 句话，回答：

1. forward pass 多久；
2. backward pass 多久；
3. measurement 之间的波动是否明显；
4. standard deviation 相对 mean 是大还是小。

不能在实际测量前预设“backward 一定是 forward 的两倍”或“标准差一定很小”。结论必须以实际 GPU 结果为依据。

## 4. Part (c)：研究 warm-up 的影响

保持模型、mode、batch size、context length、dtype 和 GPU 不变，只改变 warm-up 次数：

- [ ] `--warmup-steps 0`
- [ ] `--warmup-steps 1`
- [ ] `--warmup-steps 2`
- [ ] `--warmup-steps 5`

建议至少选择一个能稳定运行的模型，把四组结果并排比较。若时间允许，可以对 Part (b) 的全部配置重复实验。

示例：

```bash
uv run python -m cs336_systems.benchmark \
  --device cuda \
  --dtype float32 \
  --model-size small \
  --mode full_step \
  --warmup-steps 0 \
  --measurement-steps 10
```

Part (c) 最终需要用 2–3 句话回答：

1. 不做 warm-up 后，mean 和 standard deviation 如何变化；
2. 第一个或前几个 measurement 是否明显更慢；
3. 1–2 次 warm-up 为什么仍可能不够；
4. 5 次 warm-up 后结果是否进入相对稳定的 steady state。

可能涉及的冷启动成本包括：

- CUDA context 和动态库初始化；
- cuBLAS 等底层库初始化；
- CUDA caching allocator 的首次显存分配；
- kernel/module lazy loading；
- cache 尚未进入稳定状态；
- AdamW 第一次 `step()` 时惰性创建 optimizer states。

应结合实际样本判断哪些因素最可能影响了本次实验。

## 5. 最终交付检查

### 代码

- [x] 有一个可通过命令行配置的 benchmark 脚本。
- [x] 支持三种规定的运行模式。
- [x] 正确处理 warm-up、measurement 和 CUDA 同步。
- [ ] 复核 backward 与 optimizer 的时间定义。
- [ ] 在实际 CUDA 环境中完成所有正式运行。

### Writeup Part (b)

- [ ] 包含所有实际 timing。
- [ ] 包含 mean 和 standard deviation。
- [ ] 用 1–2 句话回答 forward、backward 和 variability。

### Writeup Part (c)

- [ ] 比较 0、1、2 和 5 次 warm-up。
- [ ] 用 2–3 句话解释观察到的差异。

## 6. 当前进度结论

目前 `benchmarking_script` 的 Part (a) 已基本完成；Part (b) 和 Part (c) 所需的命令行能力已经具备，但正式 CUDA 实验、结果汇总和 writeup 回答尚未完成。

`--dtype bfloat16` 当前表示直接用 BF16 参数构造模型，不等同于后续 `benchmarking_mixed_precision` 所要求的 `torch.autocast` 混合精度。该区别不妨碍当前题目的 FP32 benchmark，但之后实现 mixed precision 时需要单独处理。
