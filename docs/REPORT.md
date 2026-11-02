# π0.5 在 Jetson Orin 上的部署与实时性解剖

> **呈现主页面**。方法与可复现细节见 [METHODOLOGY.md](METHODOLOGY.md)，全流程踩坑见 [PROBLEMS.md](PROBLEMS.md)。
>
> **状态**：G1 / G2 已完成并有实测数据；G3 进行中；G4 / Top2 / G5 待做。本页只写**已实测**或**已从源码核实**的内容；推导值与未测项均显式标注。

## 摘要

本项目把 Physical Intelligence 的 **π0.5**（[openpi](https://github.com/Physical-Intelligence/openpi)，`pi05_libero` 配置）用 jetson-containers 在 **Jetson AGX Orin**（JetPack 6.2 / CUDA 12.6 / sm_87）上**从源码现场构建**成自包含容器并跑通推理，回答一个问题：**一个主流 VLA 在边缘芯片上到底能不能进实时控制环，瓶颈在哪、能靠什么手段挪动。**

头条结果（实测）：π0.5 在 Orin 上的纯推理延迟 **L1 = 487.7 ms（2.05 Hz）**，输出 action chunk 形状 `(10, 7)`。结合从 openpi / LIBERO 源码核实的官方评测协议——**控制频率 20 Hz、每 5 步重规划（`replan_steps=5`）**——可算出每次策略查询只有 **250 ms** 的机器人时间预算，而实测推理要 ~490 ms，**超支 1.96×**，真机上只能产出 ~10.2 Hz 的动作流。

但 `replan_steps` 是**纯编排旋钮**（零模型改动、零重训）：调到 10（用满模型本来就预测出的 chunk）即把占空比降到 **98%**、动作产出率升到 ~20.4 Hz，实时可行；再叠加 async prefetch 可把推理延迟隐藏在执行之后。同时，LIBERO 官方 eval 循环**没有任何实时 pacing**，因此 **benchmark 成功率与推理延迟是两条独立的轴**——这决定了「复现官方成功率」与「实时可行性」必须分开报告，不能混成一个 Hz 数字。

**诚实定位**：不训练、不微调（微调在后期 Kaggle 阶段，且必须服务部署主线）；纯仿真、无真机；主交付物 = **Orin 性能解剖 + action-chunk 调度层**，TRT / 量化是加分项。

## 实验逻辑链

| 轮次 | 问题 | 实验 | 结论 | 逼出的下一个问题 |
|---|---|---|---|---|
| P0 | Orin 上有现成 openpi 镜像吗？ | jetson-containers 构建 | 无预构建镜像，须现场 build 29 包链 | 构建能否跑通 |
| G1 | LIBERO 仿真端能否在 8 GB 消费卡上跑？ | 仿真机建 env + 离屏渲染 | ✅ 跑通（osmesa、pin 死老依赖栈） | 策略端能否上 Orin |
| G2 | π0.5 能否在 Orin 容器内推理？多快？ | 自包含容器 + L1 测量（N=20） | ✅ **L1 = 487.7 ms / 2.05 Hz** | 这个速度够不够进控制环 |
| 协议核实 | 官方 eval 究竟怎么调用策略？ | 读 openpi / LIBERO 源码 | **20 Hz 控制、`replan_steps=5`、同步阻塞、无实时 pacing** | 250 ms 预算 vs 490 ms 推理怎么办 |
| G3 | 跨机桥接闭环能否复现官方成功率？ | 仿真机 client + Orin server | ⏳ 进行中 | 成功率与 `replan_steps` 的关系 |
| G4 | 延迟花在哪？ | 模块级 profile（VLM backbone vs flow-matching expert、去噪步数） | 🔜 待做（**主交付物**） | 哪一步值得优化 |
| Top2 | 调度层能把实时性拉回来吗？代价是什么？ | `replan_steps` × 占空比 × 成功率 三方 Pareto | 🔜 待做 | — |
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

### 表 2 · 占空比算术（**推导值**，输入 = 表 1 实测 + 源码核实的协议参数）

机器人时间预算 = `replan_steps × (1 / control_freq)`；`control_freq = 20 Hz`（源码核实），L1 取实测 ~490 ms。

| 配置 | 每次查询的机器人时间预算 | vs L1 ≈ 490 ms | 实际动作产出率 | 20 Hz 实时可行？ |
|---|---|---|---|---|
| **官方 `replan_steps=5`** | 5 × 50 ms = **250 ms** | **超支 1.96×** | 5 / 0.490 ≈ **10.2 Hz** | ❌ 只有一半速度 |
| `replan_steps=10`（用满 chunk） | 10 × 50 ms = **500 ms** | **98% 占空比** | 10 / 0.490 ≈ **20.4 Hz** | ✅ 勉强，~2% 余量 |
| `replan_steps=5` + async prefetch | 250 ms | 490 > 250，队列必然抽干 | — | ❌ 仍不可行 |
| `replan_steps=10` + async prefetch | 500 ms | 推理被藏在执行之后 | 20 Hz | ✅ 有余量 |

**结果**：官方默认参数下 Orin 超支近一倍；把 `replan_steps` 从 5 调到 10 就能把占空比从 196% 拉到 98%。
**分析**：模型本来就预测 10 步，官方 eval 却只取前 5 步、丢掉后 5 步——**在算力充裕的机器上这是用重规划频率换反应性；在 Orin 上它直接决定了实时可行性**。所以 `replan_steps` 不是超参微调，而是边缘部署的一等公民旋钮。代价是重规划频率减半 ⇒ 反应性下降 ⇒ **成功率可能掉**，方向与幅度未知，这正是 Top2 要测的三方 Pareto。

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

### 项目故事被这组数字改写了

旧口径（沿用同硬件上另一个 VLA 项目的结论）是「 Orin VLA ~2 Hz 进不了控制环」，听起来像**跑不动**。有源码支撑的新口径是三段论：

1. π0.5 在 Orin 上 L1 = 490 ms；LIBERO 是 20 Hz 控制、官方 eval 每 5 步重规划 ⇒ 预算 250 ms，**超支 1.96×**，真机只能出 10.2 Hz。
2. **但 `replan_steps` 是纯编排旋钮**——调到 10 ⇒ 98% 占空比、20.4 Hz，实时可行；再加 async prefetch 把延迟藏掉。
3. **而 LIBERO eval 没有实时 pacing，所以 benchmark 成功率与延迟是两条独立的轴**，必须分开报。

第 3 条是口径纪律：复现 98.8 在 Orin 上 2 Hz 也能做到且**有效**，因为仿真按离散步推进、不等墙钟；延迟只在**真机实时控制**里才是判据。把这两件事混成一个数字，正是同类项目最容易被打穿的地方。

### 由此浮出的、没人在 Orin 上做过的实验

**`replan_steps` × 占空比 × 成功率 三方 Pareto**。重规划变少 → 反应性下降 → 成功率可能掉，这个 trade-off 的方向和幅度**未知、可测、且直接决定边缘上 VLA 该怎么调度**。openpi 官方无边缘支持，没人在 sm_87 上扫过这条曲线。这是 G4 / Top2 主交付物的内核，也是本项目区别于「跑通别人的 README」的地方。

### 诚实边界

- **无真机、纯仿真**：所有实时性结论都是「机器人时间预算 vs 实测墙钟」的算术，不是真机控制验证。
- **模型层零贡献**：用官方 fine-tuned 权重。复现 98.8 / 96.85 只是**校准步骤**（证明 harness 正确），不是成果；报告时必须标「官方报告值 vs 我复现值」。
- **「端侧运行时」不能过度声称**：运行时应含调度/资源管理/编排。当前只有 L1 测量；调度层是 Top2，做完才配这个词。
- **表 2 全部是推导值**，输入是实测 L1 + 源码协议参数；L2 尚未测量，跨机网络往返会吃掉本就只有 ~2% 的余量。
- **构建可复现性缺口**：jetson-containers 构建链未 pin 上游 commit（除 openpi 本身 pin 在 `215abfb`），镜像 33.9 GB 未归档，只保留构建配方。

### 待办

- **G3**：桥接闭环 + 复现 `libero_spatial` 官方 98.8（500 episode，≈3.5–5 h Orin 时间），记差值、分歧定位到层；同时测出 **L2**。
- **G4**：性能解剖（VLM backbone vs flow-matching action expert 延迟分解、去噪步数 → 延迟/成功率 Pareto、统一内存 roofline）。
- **Top2**：调度层（`replan_steps` 扫描 + async prefetch），实测端到端控制周期、抖动、成功率。
- **G5**：TRT / 量化（加分项；不通就写成「flow-matching VLA 在 TRT 10.7 / sm_87 的导出障碍是 X」这类墙结论）。
- **后期 Kaggle 微调**：少步推理微调（用 G4 的步数 Pareto 把前沿前推），闭合「训练→量化→部署→评测」全环。
