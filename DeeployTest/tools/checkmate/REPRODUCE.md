# 复现一条重算曲线

三步：**求解/枚举 → 部署量内存 → gvsoc 量延迟**。下面每一步都给出可直接运行的命令，
以及本仓库已测到的参考值，好让你在偏离时立刻发现，而不是跑完一整轮才发现。

以 CCT 为例；MobileNetV1 与 ResNet8 的差别只在第 1 步用哪个工具，见 README 的粒度表。

---

## 第 0 步：拿到部署器真正会调度的那张图

**必须先部署一次。** 求解要对着 `backend_post_parsing.onnx`，不是导出的模型：
解析阶段会融合和重命名节点，对着错误节点集求出的调度无法回放。

```bash
deeployTrainingRunner_tiled_gap9.py \
    -t Tests/Models/Training/CCT/cct_train -p GAP9 -s gvsoc --toolchain LLVM \
    --l1 122000 --defaultMemLevel L3 --cores 8 --n-steps 1 --n-accum 1 \
    --num-data-inputs 1
# -> <build>/Tests/Models/Training/CCT/cct_train/deeployStates/backend_post_parsing.onnx
```

> **这张图必须来自零重算的基线部署。**
> 用一次已经带重算的部署产物会安静地给出一整套错数：MobileNetV1 的基线在正确的
> 281 节点图上是 **1848.4 KB**，在"已重算 1 个块"的 287 节点图上枚举出来是
> **1704.2 KB**，不报任何错。先确认节点数与基线一致。

---

## 第 1 步：求出调度

### CCT / MobileNetV1 —— 组级枚举

```bash
export PYTHONPATH=<repo>            # 需要 testUtils.trainingUtils
tools/checkmate/enumerate_groups.py <graph.onnx> units  recompute_cct    # CCT,2^7
tools/checkmate/enumerate_groups.py <graph.onnx> blocks recompute_mnv1   # MNv1,2^13
```

参考值（replay KB，本仓库实测；数字对不上就说明图不对）：

```
CCT          962.5  738.5  705.5  704.5  672.5  672.5  768.5
             ^0组   ^1     ^2     ^3     ^4     ^6     ^7 ← 反弹
MobileNetV1 1848.4 1560.2 1343.7 1199.3 1090.3 1017.6  961.6  905.6  896.6
             ^0     ^1     ^2     ^3     ^4     ^5     ^6     ^7     ^8 ← 触底
```

CCT 第 7 组会**反弹**：加入 stem&head 后 replay 从 672.5 升到 768.5。这不是 bug——
冻结的 tokenizer 链只有末端一个 32 KB 张量被反向需要，重算它要把整条链约 512 KB 的
中间量重新物化。**留在枚举里比手工排除更有用**，它是一个可解释的结果而不是一个坑。

### ResNet8 —— 节点级 ILP

```bash
SEQ_PATH=recompute_rn8.json \
tools/checkmate/solve.py <graph.onnx> 400 mem_min_schedule ConvGradW 240
```

第二个参数是**激活预算 KB**。扫 689/650/600/550/500/450/400/350，低于 300 不可行。

---

## 第 2 步：部署，量峰值内存

```bash
deeployTrainingRunner_tiled_gap9.py ... --recomputeSchedule recompute_cct_3.json
```

**先看这一行**，没有它就不要看任何数字：

```
[Recompute] 358 scheduled executions, 114 recompute clones
```

- **克隆数为 0 或缺失 = 该点作废。** 重算在数值上是中性的，一个没生效的调度照样给出
  `Errors: 0` 和逐位相同的 loss，周期数也完全合理。
- **请求数 ≠ 存活数。** 一个请求 106 次重算的 CCT 调度实际只有 44 个克隆存活，其余 62 个
  被放在其消费者之前线性化的阶段、没人读，由 `_pruneDeadRecomputes` 剪掉。
  **报告存活数。**

峰值从部署自己的 `deeployStates/memory_alloc.html` 读（按地址去重的活块字节和）。
拆 persistent / activation 用 `AI_AGENT .../tools/measure/memsplit.py`。

参考值（CCT，无 `--convChannelsFirst`）：

| 组数 | 存活克隆 | activation | peak |
|---|---|---|---|
| 0 | 0 | 834.5 KB | 2899 KB |
| 1 | 53 | 674.5 | 2739 |
| 2 | 106 | 577.5 | 2642 |
| 3 | 114 | **545.5** | **2610** |
| 4 | 126 | 545.5 | 2610 ← 被 3 组支配 |
| 6 | 158 | 578.5 | 2643 ← 比 3 组**更差** |

persistent 在每个点上都是 **2064.8 KB**，逐字节相同——那是重算碰不到的部分。
它若在不同点之间变化，说明拆分算错了。

第 6 组比第 3 组差 33 KB，而 replay 认为它们并列 672.5 KB：**每个存活克隆自己占的内存
replay 看不见**。这就是"必须部署实测、不能只信 replay"的具体代价。

---

## 第 3 步：gvsoc，量延迟

同一条命令即可（`-s gvsoc`）。参考值（CCT，370 MHz）：

| 组数 | cycles | vs 基线 |
|---|---|---|
| 0 | 71.17 M | — |
| 1 | 74.13 M | +4.17% |
| 2 | 76.90 M | +8.05% |
| 3 | 77.09 M | +8.32% |

**噪声底约 0.3%。** 两次完全相同的配置跑出 78.84 M 和 78.62 M：codegen 不确定，两次
identity 调度生成的 C 差 3685 行声明顺序（固定 `PYTHONHASHSEED` 只降到 744 行），
改变缓冲区布局进而改变 DMA 行为。内存峰值则稳定到 KB。
**任何小于约 0.5% 的周期差不可解读。**

---

## 两个会安静给出错数的开关

**`--convChannelsFirst` 是按网络定的，不是全局偏好。**

| | CCT | MobileNetV1 |
|---|---|---|
| 内存 | 2899 KB，带与不带**相同** | — |
| 周期 | 71.17 M → 78.89 M，**+9.8%** | — |
| 结论 | **不要开** | **必须开**，否则装不进 L1 |

**调度与图版本绑定。** `recomputeScheduler` 按节点名回放，名字覆盖率低于 90% 会
**直接 raise**（这是好事，否则会静默按默认顺序跑）。换了拓扑优化 pass、换了分支，
就要重新求解，不能复用旧 JSON。
