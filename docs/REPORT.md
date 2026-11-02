# π0.5 在 Jetson Orin 上的部署与实时性解剖

> **呈现主页面**。方法与可复现细节见 [METHODOLOGY.md](METHODOLOGY.md)，全流程踩坑见 [PROBLEMS.md](PROBLEMS.md)。
>
> **状态**：G1 / G2 已完成并有实测数据；**G3 桥接闭环冒烟通过（2/2）+ Top2 async prefetch 墙已实测**；G3 全量成功率评测、G4 / G5 待做。本页只写**已实测**或**已从源码核实**的内容；推导值与未测项均显式标注。

## 摘要

本项目把 Physical Intelligence 的 **π0.5**（[openpi](https://github.com/Physical-Intelligence/openpi)，`pi05_libero` 配置）用 jetson-containers 在 **Jetson AGX Orin**（JetPack 6.2 / CUDA 12.6 / sm_87）上**从源码现场构建**成自包含容器并跑通推理，回答一个问题：**一个主流 VLA 在边缘芯片上到底能不能进实时控制环，瓶颈在哪、能靠什么手段挪动。**

头条结果（实测）：π0.5 在 Orin 上的纯推理延迟 **L1 = 487.7 ms（2.05 Hz）**，输出 action chunk 形状 `(10, 7)`。结合从 openpi / LIBERO 源码核实的官方评测协议——**控制频率 20 Hz、每 5 步重规划（`replan_steps=5`）**——可算出每次策略查询只有 **250 ms** 的机器人时间预算，而实测推理要 ~490 ms，**超支 1.96×**，真机上只能产出 ~10.2 Hz 的动作流。

但 `replan_steps` 这个**纯编排旋钮**（零模型改动、零重训）调到 10 后，G3 闭环实测却推翻了「实时可行」的乐观推算：**L1 不是常数**——由 GPU DVFS 主导，同步闭环里 GPU 在两次推理间空转掉频，L1 从背靠背 ~504 ms 升到 ~587 ms ⇒ replan=10 同步占空比 **117% 超支**；只有 async prefetch 让 GPU 持续忙才拉回 ~101%（贴边）。**而 prefetch 又撞上第二堵墙**：推理 ~504 ms ≈ 一个 chunk 执行时间 500 ms，prefetch 必须提前 1 整 chunk 用旧 obs 推理 ⇒ obs 陈旧毁掉成功率（实测 sync replan=10 = 6/6 vs prefetch replan=10 = 0/6）。**⇒ Orin 上 π0.5「20 Hz 实时」与「准确」不可兼得**，出路只有更快推理（G4/G5）或更低控制频率。同时，LIBERO 官方 eval 循环**没有任何实时 pacing**，因此 **benchmark 成功率与推理延迟是两条独立的轴**——「复现官方成功率」与「实时可行性」必须分开报告，不能混成一个 Hz 数字。

**诚实定位**：不训练、不微调（微调在后期 Kaggle 阶段，且必须服务部署主线）；纯仿真、无真机；主交付物 = **Orin 性能解剖 + action-chunk 调度层**，TRT / 量化是加分项。

## 实验逻辑链

| 轮次 | 问题 | 实验 | 结论 | 逼出的下一个问题 |
|---|---|---|---|---|
| P0 | Orin 上有现成 openpi 镜像吗？ | jetson-containers 构建 | 无预构建镜像，须现场 build 29 包链 | 构建能否跑通 |
| G1 | LIBERO 仿真端能否在 8 GB 消费卡上跑？ | 仿真机建 env + 离屏渲染 | ✅ 跑通（osmesa、pin 死老依赖栈） | 策略端能否上 Orin |
| G2 | π0.5 能否在 Orin 容器内推理？多快？ | 自包含容器 + L1 测量（N=20） | ✅ **L1 = 487.7 ms / 2.05 Hz** | 这个速度够不够进控制环 |
| 协议核实 | 官方 eval 究竟怎么调用策略？ | 读 openpi / LIBERO 源码 | **20 Hz 控制、`replan_steps=5`、同步阻塞、无实时 pacing** | 250 ms 预算 vs 490 ms 推理怎么办 |
| G3 | 跨机桥接闭环能否跑通？ | 仿真机 client + Orin server，1 task×2 ep | ✅ **冒烟 2/2 成功**、L2−L1 网络开销仅 ~7 ms | 全量 500 ep 成功率 vs 官方 98.8 |
| DVFS 核实 | 闭环 L1（587ms）为何高于 G2 in-process（490ms）？ | 变调用间隔复测 + 直接采 GPU 频率 | **GPU DVFS**：空闲 306MHz↔boost 1300MHz；L1 随间隔 504→569ms | 同步闭环该用 throttled L1 |
| Top2 prefetch | async prefetch 能救回实时性吗？ | sync vs prefetch 各 6 ep 对照（同 task/seed） | **prefetch 墙**：obs 陈旧 1 chunk ⇒ 0/6 失败（sync 6/6）；实时与准确不可兼得 | 出路=更快推理（G4/G5） |
| G4 | 延迟花在哪？ | 模块级 profile（VLM backbone vs flow-matching expert、去噪步数） | 🔜 待做（**主交付物**） | 哪一步值得优化 |
| G5 | 能否 TRT / 量化再提速？ | 导出尝试 | 🔜 加分项；不通则写成墙结论 | — |

> **地图 1（结构图 + 模块级资源账）** 随 G4 性能解剖产出后补入本节之前——现在还没有模块级实测数据，不预先画。

## 核心结果

### 表 1 · G2 L1 纯推理延迟（π0.5 `pi05_libero`，Jetson AGX Orin，自包含容器）

口径：**L1 = Orin 本地单次策略推理墙钟，in-process，不含任何网络往返**。N=20，3 次 warmup 吸收 JIT。

| 指标 | run #1 | run #2 | 首轮基线（11:08） |
|---|---|---|---|
| mean (ms) | **487.7** | 494.3 | 500.3 |
| median (ms) | 487.7 | 494.3 | 500.7 |
| min (ms) | 482.3 | 489.1 | 489.0 |
| max (ms) | 494.9 | 498.2 | 504.9 |
| std (ms) | 4.2 | 2.3 | 3.4 |
| 吞吐 (Hz) | **2.05** | 2.02 | 2.00 |
| per-action = mean/10 (ms) | 48.8 | 49.4 | 50.0 |
| 模型加载 (s) | 9.5 | 8.4 | 28.9 |
| warmup 0（含 JIT）(s) | ~9.3 | ~9.2 | 10.1 |
| action chunk | (10, 7) float64 | (10, 7) float64 | (10, 7) |
| jax 设备 | CudaDevice(id=0) | CudaDevice(id=0) | CudaDevice(id=0) |

**结果**：三轮（含重启前基线）mean 落在 487.7–500.3 ms 区间，两轮重启后复跑彼此只差 6.6 ms（1.3%），std 2.3–4.2 ms。
**分析**：软件栈完全一致（同一镜像、同一 checkpoint），差异属运行间噪声；重启后略快 ~2% 与「重启清掉了此前连串 GPU core dump 累积的 nvgpu 驱动劣化」一致（推断，非证明）。加载时间 28.9 s → 8.4–9.5 s 是因为首轮付了冷读/首次初始化成本，后两轮 checkpoint 已常驻 NVMe 缓存（`download+config time = 0.0 s`）——**不影响延迟这个核心指标**。

### 表 2 · 占空比算术（**推导值**，输入 = 实测 L1 + 源码核实的协议参数）

机器人时间预算 = `replan_steps × (1 / control_freq)`；`control_freq = 20 Hz`（源码核实）。**🔴 L1 不是常数**：G3 闭环实测发现它由 GPU DVFS 主导，分两个口径（见表 3）——**boosted L1 ≈ 504 ms**（背靠背 / async prefetch，GPU 持续忙、维持 ~1300 MHz）vs **throttled L1 ≈ 587 ms**（同步闭环，GPU 在两次推理间空转、时钟掉到 306 MHz 地板）。

| 配置 | 预算 | 用哪个 L1 | 占空比 | 20 Hz 实时可行？ |
|---|---|---|---|---|
| **官方 `replan_steps=5`（同步）** | 250 ms | throttled 587 | **235%** | ❌ 远超 |
| `replan_steps=10`（同步） | 500 ms | throttled 587 | **117%** | ❌ 超支 |
| `replan_steps=10` + **async prefetch** | 500 ms | boosted 504 | **~101%** | ⚠️ 贴边、勉强 |
| `replan_steps=5` + async prefetch | 250 ms | boosted 504 | 202% | ❌ 仍远超 |

**结果**：官方默认（replan=5）无论同步异步都远超预算（200%+）。`replan_steps=10` 把预算抬到 500 ms，但**同步闭环下 DVFS 把 L1 拖到 587 ms ⇒ 117% 仍不可行**；只有叠加 **async prefetch** 让 GPU 持续忙、维持 boost（L1 回到 ~504 ms）才拉回 ~**101%（贴边勉强）**。
**分析**：这给 async prefetch 一个**双重收益**——不只是「把推理藏进执行」，更是「让 GPU 不空转、避开 ~16% 的 DVFS 降频罚」。`replan_steps` 与 prefetch 都是**纯编排旋钮**（零模型改动），却直接决定 Orin 上能否贴住 20 Hz。**但 prefetch 有代价——obs 陈旧会毁掉成功率，见表 3。**

### 表 3 · G3 闭环实测：桥接、DVFS、prefetch 墙（2026-11-02）

**桥接形态**：仿真机（LIBERO sim client）→ websocket → Orin（`serve_policy.py --env LIBERO` server）。冒烟 `libero_spatial` task0 × 2 ep：**2/2 成功**、action_chunk (10,7)、**L2 − L1 ≈ 7 ms**（网络+序列化开销可忽略）⇒ 跨机桥接成立、不污染延迟口径。

**DVFS 实测**（`probe_server_latency.py`，换独立变量「调用间隔」+ 直接采 GPU 频率）：

| 调用间隔 gap | server L1 | GPU 行为 |
|---|---|---|
| 0（背靠背） | **504 ms** | 维持 boost ~1300 MHz |
| 0.3 s | 527 ms | 间歇掉频 |
| 0.6 s | 558 ms | ↓ |
| 1.0 s | **569 ms** | 趋平台 |
| 闭环（osmesa 慢渲染，间隔更大） | 587 ms | 平台 + 噪声 |

gap=1.0 探针期间并发采 GPU `cur_freq`（80 样本）：**17 @ 306 MHz（空闲地板）+ 24 @ 1300.5 MHz（boost）+ 中间爬坡**；devfreq 范围 306↔1300.5 MHz（**4.25×**）⇒ **直接观测到时钟双峰振荡**，坐实 L1 差异根因是 GPU DVFS（非 warmup、非网络、非挂死）。

**🔴 prefetch 墙**（同 task / seed / 初始状态，6 ep 对照，唯一变量 = sync vs prefetch）：

| 配置 | 成功率 | 步数 |
|---|---|---|
| sync replan=10（新鲜 obs） | **6/6** | 84-116 步完成 |
| prefetch replan=10（陈旧 obs） | **0/6** | 全跑满 230 步失败 |

**机制**：π0.5 在 Orin 上**推理 ~504 ms ≈ replan=10 一个 chunk 的执行时间 500 ms**（10×50ms@20Hz），这是 async prefetch 的最坏情形——为让下一 chunk 及时就绪，必须在当前 chunk 一开始就用「那时的 obs」推理 ⇒ 每个 chunk 陈旧整整 1 chunk（头两 chunk 还因机器人未动而重复）⇒ 机器人重复/错位执行、过冲、失败。**⇒ 延迟-新鲜度根本权衡：Orin 上 π0.5 要么 sync（新鲜 obs、准确，但机器人等待 + DVFS 降频、有效 ~9-10 Hz 非实时）要么 prefetch（实时可行但 obs 陈旧、成功率崩）——「20 Hz 实时」与「准确」二者不可兼得。** 出路只有**更快推理**（G4/G5 把 504 ms 压到 <250 ms，则 replan=5+prefetch 陈旧 <0.5 chunk 或可兼得）或**更低控制频率**。

## 方法

### I/O 三层次（分清这三层是讲清项目的前提）

1. **运行时系统**（部署后每个请求）：
   - **入**：`observation/image` (224,224,3) uint8 + `observation/wrist_image` (224,224,3) uint8 + `observation/state` (8,) float（`eef_pos` 3 + `axisangle(eef_quat)` 3 + `gripper_qpos` 2）+ `prompt` str
   - **出**：`actions` (10, 7) float64 —— 10 步 × 7 维（6 DoF 位姿增量 + 夹爪）
   - 原始渲染是 256×256，客户端 `resize_with_pad` 到 224，并**旋转 180°**（`[::-1, ::-1]`）以匹配训练预处理
2. **构建流水线**（我们做的工程）：openpi 源码 @ `215abfb` + jetson-containers 29 包链 → 自包含镜像 `openpi:r36.4.tegra-aarch64-cu126-22.04-openpi`（33.9 GB，jax 0.5.3 + `XLA_FLAGS` 已烘焙）
3. **评测**：LIBERO `libero_spatial`（10 任务 × 50 trial = 500 episode，固定初始状态）→ 成功率

**不做什么**：不训练、不微调（当前阶段）；不改模型结构；不输出轨迹/控制量以外的东西；无真机。

### 三个预注册延迟口径

任何 Hz 数字必须标明属于哪一口径（制度化修「纯推理 Hz 与控制周期 Hz 混淆」这个坑）：

- **L1 纯推理** = Orin 本地单次策略推理墙钟，不含网络 → 「 Orin 能跑多快」。**已测**（表 1）
- **L2 控制周期** = obs → Orin 推理 → action 回仿真机的端到端往返 → 「闭环实际控制频率」。**未测**（G3）
- **L3 任务要求** = LIBERO 期望控制频率 → 参照系。**已核实 = 20 Hz**

「边缘实时性」的判据是 **L2 vs L3**，不是 L1。L1 只是引擎地板。

### 官方 eval 协议（源码核实，行号可查）

| 事实 | 来源 |
|---|---|
| `control_freq = 20` | `third_party/libero/libero/libero/envs/bddl_base_domain.py:63`、`envs/env_wrapper.py:27` |
| `replan_steps = 5`（默认） | `examples/libero/main.py:29` |
| 只取 chunk 前 5 步 | `examples/libero/main.py:148` `action_plan.extend(action_chunk[: args.replan_steps])` |
| assert chunk ≥ replan_steps | `examples/libero/main.py:145-147` |
| 同步阻塞、无 async prefetch | `examples/libero/main.py:127-153`（deque 空 → 阻塞 `client.infer()` → 取 5 → pop 1 → `env.step()`） |
| **循环内无 sleep / 无时钟对齐** | `examples/libero/main.py:104-158` 全循环 ⇒ 成功率与延迟无关 |
| `num_trials_per_task = 50`、`seed = 7` | `examples/libero/main.py:38,45` |
| `max_steps`：spatial 220 / object 280 / goal 300 / libero_10 520 / libero_90 400，另加 `num_steps_wait=10` | `examples/libero/main.py:60-71` |
| 官方架构本就是 client/server 分离 | `examples/libero/compose.yml`（`runtime` + `openpi_server`，均 `network_mode: host`） |
| 官方报告成功率：Spatial 98.8 / Object 98.2 / Goal 98.0 / Libero-10 92.4 / **Average 96.85** | `examples/libero/README.md` Results 表 |

⚠️ **96.85 是 4 个 suite 的平均，不是单 suite 数**。默认 `task_suite_name="libero_spatial"` ⇒ 单套件对标值是 **98.8**；`libero_90` 不在官方表内。

### 任务数与初始状态（实测）

| suite | 任务数 | `.pruned_init` 数 |
|---|---|---|
| libero_spatial / object / goal / 10 | 各 **10** | 各 10 |
| libero_90 | **90** | 90 |
| 合计 | 130 | 130 |

每个 `.pruned_init` 内含 **50 个初始状态**（`ndarray len 50`，元素 `(92,) float64` = MuJoCo qpos+qvel），与 `num_trials_per_task=50` 一一对应——**一 trial 一个固定初始状态，这就是官方成功率可复现的机制**。任务顺序由 `task_orders[0] = [0..9]` 固定。

## 讨论

### 项目故事被这组数字改写了（两次）

旧口径（沿用同硬件上另一个 VLA 项目的结论）是「Orin VLA ~2 Hz 进不了控制环」，听起来像**跑不动**。源码核实 + G3 闭环实测后，新口径是三段论：

1. π0.5 在 Orin 上 L1 ≈ 490-504 ms；LIBERO 是 20 Hz 控制、官方 eval 每 5 步重规划 ⇒ 预算 250 ms，**超支 ~2×**，真机只能出 ~10 Hz。
2. **`replan_steps` 是纯编排旋钮**——调到 10 把预算抬到 500 ms。但 G3 实测发现 **L1 不是常数**：同步闭环里 GPU 空转掉频（DVFS），L1 升到 587 ms ⇒ 117% 仍不可行；只有 async prefetch 让 GPU 持续忙（L1 回 504 ms）才拉回 ~101% 贴边。
3. **而 prefetch 又撞上第二堵墙**：推理 ≈ chunk 执行时间 ⇒ prefetch 必须提前 1 整 chunk 用旧 obs ⇒ obs 陈旧毁掉成功率（实测 sync 6/6 vs prefetch 0/6）。**⇒「20 Hz 实时」与「准确」不可兼得**，出路只有更快推理（G4/G5）或更低控制频率。

外加一条口径纪律：**LIBERO eval 没有实时 pacing，所以 benchmark 成功率与延迟是两条独立的轴**——复现 98.8 在 Orin 上 2 Hz 也能做到且**有效**（仿真按离散步推进、不等墙钟）；延迟只在**真机实时控制**里才是判据。把这两件事混成一个数字，正是同类项目最容易被打穿的地方。

### 由此浮出的、没人在 Orin 上做过的实验

**`replan_steps` × 占空比 × 成功率 三方 Pareto**。重规划变少 → 反应性下降 → 成功率可能掉，这个 trade-off 的方向和幅度**未知、可测、且直接决定边缘上 VLA 该怎么调度**。openpi 官方无边缘支持，没人在 sm_87 上扫过这条曲线。这是 G4 / Top2 主交付物的内核，也是本项目区别于「跑通别人的 README」的地方。

### 诚实边界

- **无真机、纯仿真**：所有实时性结论都是「机器人时间预算 vs 实测墙钟」的算术，不是真机控制验证。
- **模型层零贡献**：用官方 fine-tuned 权重。复现 98.8 / 96.85 只是**校准步骤**（证明 harness 正确），不是成果；报告时必须标「官方报告值 vs 我复现值」。
- **「端侧运行时」不能过度声称**：运行时应含调度/资源管理/编排。当前只有 L1 测量；调度层是 Top2，做完才配这个词。
- **表 2 全部是推导值**，输入是实测 L1 + 源码协议参数；L2 尚未测量，跨机网络往返会吃掉本就只有 ~2% 的余量。
- **构建可复现性缺口**：jetson-containers 构建链未 pin 上游 commit（除 openpi 本身 pin 在 `215abfb`），镜像 33.9 GB 未归档，只保留构建配方。

### 待办

- **G3 全量**：`libero_spatial` 500 episode 复现官方 98.8（≈3.5–5 h Orin 时间），记差值、分歧定位到层。（桥接冒烟 + L2 已完成：L2−L1 ≈ 7 ms。）
- **G4（主交付物）**：性能解剖——VLM backbone vs flow-matching action expert 延迟分解、去噪步数 → 延迟 Pareto、统一内存 roofline。**被 prefetch 墙直接驱动**：要打破「实时 vs 准确」权衡，必须把 504 ms 压下来，先要知道时间花在哪。
- **Top2 续**：prefetch 墙已实测（sync 6/6 vs prefetch 0/6）；可再探更聪明方案（temporal ensembling / obs 预测）能否缓解陈旧，但机制上「推理≈chunk 时长 ⇒ 必陈旧 1 chunk」是硬约束。
- **G5**：TRT / 量化（加分项；不通就写成「flow-matching VLA 在 TRT 10.7 / sm_87 的导出障碍是 X」这类墙结论）。
- **demo 页**：G4 + G3 全量出结果后做（需 rollout 视频 + 性能解剖 + 成功率）。
- **后期 Kaggle 微调**：少步推理微调（用 G4 的步数 Pareto 把前沿前推），闭合「训练→量化→部署→评测」全环。
