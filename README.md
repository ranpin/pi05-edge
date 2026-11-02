# pi05-edge

把 Physical Intelligence 的 **π0.5**（[openpi](https://github.com/Physical-Intelligence/openpi)，`pi05_libero`）部署到 **Jetson AGX Orin**，在 **LIBERO** 上跑闭环评测，并做 Orin 上的性能解剖与 action-chunk 调度。

> **定位**：边缘部署 / 推理优化项目——不训练、不微调（微调在后期 Kaggle 阶段，且必须服务部署主线）。主交付物 = **Orin 性能解剖 + action-chunk 调度层**；TRT/量化是加分项，不通就写成墙结论。

## 核心结果（截至 2026-10-30）

**G2 · π0.5 在 Orin 上的纯推理延迟（L1 口径，自包含容器，重启后官方复跑两轮）**

| 指标 | run #1 | run #2 | 首轮基线 |
|---|---|---|---|
| mean 延迟 | **487.7 ms** | 494.3 ms | 500.3 ms |
| 吞吐 | **2.05 Hz** | 2.02 Hz | 2.00 Hz |
| per-action（H=10） | 48.8 ms | 49.4 ms | 50.0 ms |
| action chunk | (10, 7) | (10, 7) | (10, 7) |
| 设备 | CudaDevice(id=0) | CudaDevice(id=0) | CudaDevice(id=0) |

**头条发现**：LIBERO 是 **20 Hz** 控制、官方 eval 每 **5** 步重规划（`replan_steps=5`）⇒ 每次策略查询的机器人时间预算只有 250 ms，而 Orin 上推理要 ~490 ms，**超支 1.96×**（真机只能出 ~10.2 Hz）。但 `replan_steps` 是**纯编排旋钮**——调到 10（用满模型本来就预测出的 chunk）⇒ 98% 占空比、~20.4 Hz，实时可行；再加 async prefetch 即可把延迟藏掉。而 LIBERO eval 循环**没有实时 pacing**，所以 benchmark 成功率与延迟是两条独立的轴，必须分开报。详见 [docs/REPORT.md](docs/REPORT.md)。

## 快速开始

本项目依赖 Orin 上的 33.9 GB 自包含镜像与官方 checkpoint，**无法在本地一键运行**。复现 L1 测量的确切命令、观测构造与官方 eval 协议见 [docs/METHODOLOGY.md §3–§4](docs/METHODOLOGY.md)；最小复现即：

```bash
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OPENPI_DATA_HOME=/openpi_cache \
  -v <orin-workdir>/g2_verify.py:/g2_verify.py:ro \
  -v <orin-workdir>/openpi_cache:/openpi_cache \
  openpi:r36.4.tegra-aarch64-cu126-22.04-openpi python3 /g2_verify.py
```

## 闸门状态

| 闸门 | 内容 | 状态 |
|---|---|---|
| P0 | 装 docker + 现场 build `openpi` 容器（jetson-containers，无预构建镜像） | ✅ 2026-10-29 |
| G1 | LIBERO 仿真在仿真机跑通 | ✅ 2026-10-27 |
| G2 | π0.5 在 Orin 容器内单次推理 + 实测 L1 | ✅ 2026-10-30 |
| G3 | 桥接闭环 + 复现官方成功率（记差值、分歧定位到层） | ⏳ 进行中 |
| G4 | Orin 性能解剖（**主交付物**） | 🔜 |
| Top2 | action-chunk 调度层（`replan_steps` × 占空比 × 成功率 Pareto） | 🔜 |
| G5 | TRT / 量化（加分项，不通写成墙结论） | 🔜 |

## 文档

- [docs/REPORT.md](docs/REPORT.md) — 呈现主页面：摘要 → 核心结果 → 方法 → 讨论
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — 来源与版本表、三口径定义、官方 eval 协议、复现命令
- [docs/PROBLEMS.md](docs/PROBLEMS.md) — 全流程问题记录（build 十坑、GPU 驱动挂死、torch 阻塞解除）

## 仓库结构

```
scripts/orin/   # 在 Orin 上跑的脚本（g2_verify.py 等）
scripts/sim/    # 在仿真机上跑的脚本（G3 client）
docs/           # 论文式文档（呈现页 / 方法 / 问题记录）
```

## License

代码 [MIT](LICENSE)；文档 [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。Copyright (c) 2026 ranpin。
