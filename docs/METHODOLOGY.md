# METHODOLOGY · 来源、口径与复现

> 本页只放**方法与可复现**信息。结果与讨论见 [REPORT.md](REPORT.md)，踩坑见 [PROBLEMS.md](PROBLEMS.md)。
> 所有版本/型号值均**实测取自环境**（`git rev-parse` / `nvcc --version` / `head -1 /etc/nv_tegra_release` / `import x; x.__version__`），不靠记忆。

## 1. 来源与版本表

### 1.1 设备

| 项 | 值 | 取法 |
|---|---|---|
| 型号 | **NVIDIA Jetson AGX Orin Developer Kit** | `cat /proc/device-tree/model` |
| 统一内存 | 30696 MB（32 GB，CPU/GPU 共享） | `free -m` |
| CPU 核数 | 12 | `nproc` |
| L4T | **R36.4.4**（`# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025`） | `head -1 /etc/nv_tegra_release` |
| JetPack | 6.2 | 由 L4T R36.4.4 对应 |
| CUDA | **12.6**（`V12.6.85`） | `nvcc --version` |
| GPU 计算能力 | sm_87（Ampere，Tegra） | 由 Orin 型号确定 |

⚠️ **功耗模式未固定**（未执行 `nvpmodel` 查询/锁定）。延迟数字对应测量当时的默认模式，这是可复现性缺口之一。

### 1.2 容器与软件栈

| 项 | 值 |
|---|---|
| 镜像 tag | `openpi:r36.4.tegra-aarch64-cu126-22.04-openpi` |
| 镜像 ID / 体积 | `8684a2f513ba` / **33.9 GB** |
| 构建方式 | [jetson-containers](https://github.com/dusty-nv/jetson-containers) 现场构建，29 包链，`--skip-tests=all` |
| 直接基础镜像 | `openpi:r36.4.tegra-aarch64-cu126-22.04-lerobot`（`2c862fb79fd5`，30.4 GB） |
| Python | 3.10.22 |
| jax / jaxlib | **0.5.3 / 0.5.3** |
| jax-cuda12-plugin / pjrt | **0.5.3 / 0.5.3** |
| flax | 0.10.2 |
| numpy | 2.2.6 |
| `XLA_FLAGS`（已烘焙进镜像 ENV） | `--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0` |

> **为什么是 jax 0.5.3 + 这两个 flag**：Tegra sm_87 上通用 jax-cuda12-plugin 的 `GemmFusionAutotuner` redzone 路径会 `LOG(FATAL)` 崩溃（gdb 实锤，与 jax/cuDNN 版本无关）。0.5.3 是 openpi 自己的 pin，其 plugin 元数据要求 `cudnn>=9.1,<10`，恰好被 JetPack 6.2 自带的 Tegra cuDNN 9.3 满足；`autotune_level=0` 是绕开崩溃的核心。详见 [PROBLEMS.md](PROBLEMS.md) 坑 10。

### 1.3 模型与权重

| 项 | 值 |
|---|---|
| 上游仓库 | <https://github.com/Physical-Intelligence/openpi> |
| **pin 的 commit** | `215abfb217dbac7d5f1273282331b9b1866c0479`（commit date **2026-08-24**） |
| 配置名 | `pi05_libero` |
| checkpoint | `gs://openpi-assets/checkpoints/pi05_libero`（**官方 fine-tuned 权重，非我们训练**） |
| checkpoint 规模 | 16 文件 / **12,439,134,633 字节**（6 个大文件在 `params/ocdbt.process_0/d/`） |
| 本地缓存 | Orin:`<orin-workdir>/openpi_cache` → 容器内 `/openpi_cache`（经 `OPENPI_DATA_HOME`） |
| 官方报告推理显存需求 | > 8 GB |

### 1.4 仿真端（G1 / G3 client）

| 项 | 值 |
|---|---|
| GPU | RTX 3070 8 GB |
| OS / Python | Ubuntu 24.04 / Python 3.10 |
| venv | `<sim-workdir>/pi05_orin/venv` |
| 关键 pin | `numpy==1.22.4` `mujoco==2.3.2` `robosuite==1.4.0` `opencv-python==4.6.0.66` `numba==0.56.4` `gym==0.25.2` `bddl==1.0.1` `scipy==1.9.3` `hydra-core==1.2.0` |
| 渲染 | **osmesa**（`MUJOCO_GL=osmesa`，需 `sudo apt install libosmesa6`） |
| 🔴 铁律 | **绝不装 torch / thop**——现代 torch 会把 numpy 顶到 2.x，连带搞崩 opencv-python 4.6 与 numba 0.56.4 |

### 1.5 LIBERO

| 项 | 值 |
|---|---|
| 来源 | openpi 的 git submodule `third_party/libero`（上游 <https://github.com/Lifelong-Robot-Learning/LIBERO>） |
| `control_freq` | **20**（`envs/bddl_base_domain.py:63`、`envs/env_wrapper.py:27`） |
| bddl / init 文件 | 随 submodule 提供，**无需下载 datasets** |
| `.pruned_init` 总数 | 130（spatial/object/goal/10 各 10，libero_90 90） |

## 2. 三个延迟口径（预注册）

任何 Hz / ms 数字必须标明属于哪一口径。

| 口径 | 定义 | 测量方式 | 回答什么 | 状态 |
|---|---|---|---|---|
| **L1** | Orin 本地单次策略推理墙钟，**in-process，不含任何网络** | `scripts/orin/g2_verify.py`：3 次 warmup 吸收 JIT，然后 N=20 次 `policy.infer()` 取墙钟 | 「 Orin 能跑多快」（引擎地板） | ✅ 已测 |
| **L2** | obs → Orin 推理 → action 回仿真机的**端到端往返** | 跨机 websocket，客户端侧计时 | 「闭环实际控制频率」 | ⏳ G3 |
| **L3** | LIBERO 期望的控制频率 | 源码核实 = **20 Hz** | 参照系 | ✅ 已核实 |

**「边缘实时性」的判据是 L2 vs L3，不是 L1。** L1 只是引擎地板；把 L1 的 Hz 当成控制频率，是同类项目最容易被打穿的口径错误。

⚠️ **L1 必须走 in-process，不能走 websocket**。`serve_policy.py` + `WebsocketClientPolicy` 那条路掺网络往返，只能用于 L2。

## 3. 复现命令

### 3.1 G2 · L1 测量

```bash
# 在 Orin 上（镜像已烘焙 jax 0.5.3 + XLA_FLAGS，无需任何额外 env）
docker run --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OPENPI_DATA_HOME=/openpi_cache \
  -v <orin-workdir>/g2_verify.py:/g2_verify.py:ro \
  -v <orin-workdir>/openpi_cache:/openpi_cache \
  openpi:r36.4.tegra-aarch64-cu126-22.04-openpi \
  python3 /g2_verify.py
```

> `<orin-workdir>` = Orin 机上存放本仓脚本（`g2_verify.py`）与 checkpoint 缓存（`openpi_cache/`）的工作目录，部署时自定；本项目用 NVMe 上的持久目录，避免重启后缓存丢失。

脚本副本入库在 [`scripts/orin/g2_verify.py`](../scripts/orin/g2_verify.py)。它做四件事：① 查 `jax.devices()` 排除 CPU 回退 ② 下载/加载 checkpoint 并计时 ③ `make_libero_example()` 造观测 + 3 次 warmup ④ N=20 次 `infer` 取墙钟统计。

> ⚠️ **验 GPU 必须用非常量输入的真算子**。`jnp.ones(...).sum()` 会被常量折叠、根本不碰 GPU，即使 GPU 计算路径已坏也会「通过」——这个静默失败在本项目上花掉过数小时。镜像构建期的 step-11 因此硬化为 `jax.jit` 一个真 conv 并 assert `default_backend()=='gpu'`。

### 3.2 观测构造（LIBERO schema）

`openpi.policies.libero_policy.make_libero_example()` 自带 dummy 输入，延迟测量直接可用（**计算量与输入内容无关**；动作正确性归 G3 闭环）：

| 字段 | 形状 / 类型 |
|---|---|
| `observation/state` | `(8,)` = `eef_pos`(3) + `axisangle(eef_quat)`(3) + `gripper_qpos`(2) |
| `observation/image` | `(224,224,3)` uint8 |
| `observation/wrist_image` | `(224,224,3)` uint8 |
| `prompt` | str |

内部映射到 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`（后者填零）。真实 eval 时客户端从 256×256 渲染 `resize_with_pad` 到 224，并**旋转 180°**（`[::-1, ::-1]`）以匹配训练预处理。

## 4. 官方 eval 协议（G3 必须照此复现）

源码位置：`/opt/openpi/examples/libero/main.py`（镜像内）。关键参数与行号见 [REPORT.md 的协议表](REPORT.md#官方-eval-协议源码核实行号可查)。复现要点：

- `task_suite_name` 默认 `libero_spatial`；`num_trials_per_task=50`；`seed=7`；`num_steps_wait=10`
- `max_steps`：spatial 220 / object 280 / goal 300 / libero_10 520 / libero_90 400
- 每 episode 用 `initial_states[episode_idx]`（**固定初始状态，一 trial 一个**）
- 前 `num_steps_wait` 步执行 `LIBERO_DUMMY_ACTION = [0.0]*6 + [-1.0]`，等物体落稳
- `replan_steps=5`：deque 空时阻塞查询，只取 chunk 前 5 步
- 成功判据 = `env.step()` 返回的 `done`

**官方报告成功率**（`examples/libero/README.md`，**官方报告值，非我们复现值**）：

| Model | Spatial | Object | Goal | Libero 10 | **Average** |
|---|---|---|---|---|---|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | **96.85** |

⚠️ **96.85 是 4 个 suite 的平均**。单跑 `libero_spatial` 的对标值是 **98.8**。报告时必须写清「官方报告值 vs 我复现值」，且复现只是**校准步骤**（证明 harness 正确），不是成果。

## 5. 评测规模与耗时预算

| 项 | 值 | 依据 |
|---|---|---|
| 单 suite episode 数 | 10 任务 × 50 trial = **500** | 实测 `.pruned_init` 计数 + `num_trials_per_task` |
| 单 episode 最大步数 | 230（spatial：220 + 10 wait） | `main.py:61,37` |
| 单 episode 最大推理次数 | ⌈230 / 5⌉ ≈ **46** | `replan_steps=5` |
| 单 episode 推理墙钟 | ≈ 46 × 0.49 s ≈ **22.5 s** | 表 1 实测 L1 |
| **单 suite 预计** | **≈ 3.5–5 h**（含 sim stepping 与渲染） | 推导 |
| **4 suite 全量对标 96.85** | **≈ 14–20 h** Orin 时间 | 推导 |

> 建议 G3 先只跑 `libero_spatial`（对标 98.8），别一上来就 4 suite。迭代期可先把 `num_trials_per_task` 降到 10（100 episode，≈45–60 min）打通链路，再跑全量出正式数。

## 6. G3 的关键技术决定：客户端不装 torch

官方 `main.py:8` 有 `from libero.libero import benchmark`，而 `benchmark/__init__.py:5` 有 `import torch`。这与仿真机那套 pin 死的 `numpy 1.22.4 / opencv-python 4.6 / numba 0.56.4` 栈冲突（现代 torch 会把 numpy 顶到 2.x）。

**实测结论：可以完全不装 torch。**

- `import torch` 在整个 benchmark 模块里**只被用在一行**：`benchmark/__init__.py:164` → `init_states = torch.load(init_states_path)`
- `.pruned_init` 文件魔数是 `504b`（ZIP）= `torch.save` 的容器格式，内含 `archive/data.pkl` + `archive/version`，**没有 tensor storage 条目**（数组数据内联在 pickle 里）
- 该 pickle 流的 GLOBAL 引用**只有 numpy**：`numpy.core.multiarray._reconstruct` / `numpy.ndarray` / `numpy.dtype`（dtype `f8` 小端）+ `_codecs.encode`，**零 torch 引用**
- **实证**：`pickle.loads(zipfile.ZipFile(p).read("archive/data.pkl"))` 成功返回 `ndarray len 50`、元素 `(92,) float64`，且 `torch` 全程未进 `sys.modules`

⇒ 把那一行 `torch.load` 换成 zip + pickle 读即可，仿真机的依赖栈一行都不用动，G1 的「绝不装 torch」铁律守得住。

剩下需要自己复刻的只有三件纯文本/纯 numpy 活：
1. **任务枚举**：`libero_task_map[suite]` + `task_orders[0] = [0..9]` 固定顺序（`benchmark/__init__.py:68-77, 83-86, 116-122`）
2. **language 提取**：`grab_language_from_filename(task + ".bddl")`
3. **init_states 路径拼接**：`get_libero_path("init_states") / problem_folder / f"{task}.pruned_init"`

## 7. 可复现性缺口（诚实披露）

| 缺口 | 说明 | 缓解 |
|---|---|---|
| jetson-containers 构建链未 pin | 除 openpi 本身 pin 在 `215abfb`，29 包链的其余上游未固定 commit | 构建配方与十坑已写入 [PROBLEMS.md](PROBLEMS.md)；镜像 ID `8684a2f513ba` 已记录 |
| 镜像未归档 | 33.9 GB，只在 Orin 本地 | 保留可重建的 Dockerfile 补丁（live + logs 快照双份）；重建需 ~数十小时 |
| 功耗模式未锁 | 未查/未设 `nvpmodel` | 后续测量前应锁定并记录 |
| checkpoint 未 pin revision | `maybe_download` 取当时最新 | 已记录精确字节数 12,439,134,633 与 16 文件清单，可校验 |
| L2 未测 | 跨机网络往返未知 | G3 首要产出 |
