# G5 实验设计：从复现验证到原创调度（FlashRT Orin INT8 + 自适应 prefix 缓存分层调度）

> **状态**：设计稿（2026-11-23）。两方案均待执行；**先做方案 A（简单）**。
> 本文记录 G5 的调研结论、分析定位与两套可执行实验设计。配套调研证据库见记忆 `vla-realtime-edge-deployment-landscape`。

## 0. 背景与定位

### 0.1 调研触发的墙结论修正

G4 roofline + Top2 prefetch 墙曾给出「Orin 上 π0.5 到不了 250ms 实时、实时与准确不可兼得」的结论。2026-11-23 工业界调研**修正了这个表述的强度**：

- **roofline 的判断仍对**：prefix（SigLIP+Gemma2B prefill）compute-bound（AI 266/998 ≫ 脊点 192），**weight-only INT4 零收益**；唯一能挪 prefix 的量化是 **W8A8**。
- **但「Orin 不可实时」过强**：FlashRT 官方 Orin 指南在**同款 AGX Orin 64GB（SM87）**上实测 π0.5：

  | 配置 | 延迟 | 频率 | 精度 |
  |---|---|---|---|
  | naive JAX BF16（本项目 G2） | ~490 ms | 2 Hz | — |
  | FlashRT BF16（已融合） | 193 ms | 5.2 Hz | ref |
  | **FlashRT INT8 W8A8 + cache_frames=1** | **124 ms** | **8.04 Hz** | **cosine 1.000 逐位相等** |
  | FlashRT INT8 + cache_frames=2 | 摊薄 39 ms | 12.2 Hz | cosine 0.991 |

  **124ms < 250ms（replan=5 预算）⇒ 同款 Orin 上 sync replan=5 既实时又准确，prefetch 墙消解。**
- **诚实新口径**：Orin 缺 FP8/FP4 ⇒ 到不了 Thor 的 ~21Hz，但 **INT8 W8A8 + kernel 融合 + CUDA graph + 无损 KV 缓存可达 ~8Hz/124ms 无损 = 够 replan=5 实时**。我们的 490ms 墙是「**naive JAX 实现差距 + 芯片无 FP4**」叠加，非纯芯片不可逾越。roofline 当时估 W8A8 仅 1.37× 偏保守——那是 roof-to-roof 算力乘数，没算 naive JAX 离 roof 的工程差距（实际 ~4×）。

### 0.2 项目贡献定位（含金量落在哪）

**本项目不是造引擎、也不是纯复现**，主交付物 = **Orin 性能解剖 + action-chunk 调度分析**。原创贡献是：

1. **有预测力的分析框架**：roofline 事前命中「W8A8 而非 weight-only INT4 是杠杆」，被 FlashRT 工业实践印证；control-time 不等式 `T_infer ≤ replan/f_control` 被 PhyAI 论文（arxiv 2608.03682）独立同构验证。
2. **实证墙刻画**：prefetch 墙 sync 24/24 vs prefetch 0/24（CI 互不重叠）+ 陈旧机制。
3. **原创调度前沿**（方案 B）：π0.5 on Orin 的「实时 × 准确」Pareto + 破墙阈值 + 自适应缓存门控——FlashRT/PhyAI 均未发表。

**复现（方案 A）只是验证脚注，不是头条**；头条是「分析有预测力 + 原创调度前沿」。

### 0.3 关键调研数据（跨硬件，同 pi05_libero / 2 相机 / 10 步）

| 硬件 | 后端/精度 | 延迟 | 来源 |
|---|---|---|---|
| Orin（本项目） | naive JAX BF16 | ~490 ms | G2 实测 |
| **Orin（同款）** | **FlashRT INT8 W8A8 无损** | **124 ms / 8 Hz** | FlashRT deployment_orin.md |
| Thor | PyTorch BF16 | 132 ms | NVIDIA jetson-ai-lab |
| Thor | TRT FP8+NVFP4 | 49 ms | NVIDIA jetson-ai-lab |
| Thor | FlashRT NVFP4+FA4 | 20.3 ms / 49 Hz | FlashRT |
| RTX 5090 | FlashRT FP4 | 8.75 ms / 114 Hz | FlashRT |

非 Thor 方案：智元 GO-1 用同款 Orin 64GB（ViLLA 架构）、Tesla 自研 FSD 芯片、SmolVLA 450M 消费级、GR00T N1.7 Orin BF16 全 TRT 275ms/16chunk、Figure Helix 双系统（7B@7–9Hz + 80M@200Hz）、PI/Modal 云端 QUIC（10–15ms 开销）。

---

## 1. 方案 A：FlashRT Orin INT8 复现（先做）

### 1.1 目标与角色

- **目标**：在 dog（同款 AGX Orin 64GB / SM87 / JetPack 6.2 / CUDA 12.6）上跑通 FlashRT 的 π0.5 INT8 W8A8 路径，实测延迟与精度，**验证我们 roofline 的预测**（W8A8 有效、可达 ~124ms/8Hz 无损）。
- **角色（诚实）**：这是**复现验证**，给项目提供「优化推理基线」+ 印证 roofline；**不是原创头条**。它为方案 B 提供快推理对照基线。

### 1.2 假设

- H_A1：FlashRT INT8 W8A8 + cache_frames=1 在 dog 上 p50 ≈ 124ms（±20%，DVFS/热态差异），cosine vs BF16 ≈ 1.000。
- H_A2：BF16 参考 ≈ 193ms；INT8 相对 BF16 ~1.5×，相对 naive JAX ~4×。
- H_A3：cache_frames=2 ≈ 12.2Hz、cosine ≈ 0.991。

### 1.3 执行步骤

1. **环境勘察**：dog 上 nvcc / cmake / g++ / PyTorch aarch64 wheel 是否齐备；缺则用 FlashRT docker/ 或装系统 CUDA dev + cmake。
2. **拉源码 + 依赖**：
   ```bash
   git clone https://github.com/flashrt-project/FlashRT.git
   cd FlashRT
   git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
   pip install -e ".[torch]"   # 需 Jetson aarch64 PyTorch wheel（CUDA 12.6 兼容）
   ```
3. **build（SM87）**：
   ```bash
   cmake -B build_orin_sm87 -S . -DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON \
     -DFA2_HDIMS='96;128;256' -DFA2_DTYPES='bf16'
   cmake --build build_orin_sm87 -j4   # ~4–6 min
   cp build_orin_sm87/flash_rt_kernels*.so build_orin_sm87/flash_rt_fa2*.so flash_rt/
   ```
4. **PaliGemma tokenizer**：`bash scripts/download_paligemma_tokenizer.sh`（GCS 公开桶，~4.1MiB；openpi 缓存可能已有）。
5. **checkpoint 转换**：FlashRT 吃 **PyTorch** checkpoint（examples 用 `pi05_droid_pytorch`）。我们是 `pi05_libero` JAX ⇒ 需 JAX→PyTorch 转换（openpi `examples/convert_jax_model_to_pytorch.py` 产 model.safetensors）。**🔴 关键兼容性待验证**：`Pi05TorchFrontendRtx` 是否接受 pi05_libero 的 config/norm-stats/action_dim（libero 与 droid 均 7 维动作、2 视图，架构同 pi05，预期兼容但需实测）。
6. **benchmark**：
   ```bash
   FVK_PI05_RTX_FORCE_INT8=1 python3 examples/orin/bench_pi05.py \
     --checkpoint <pi05_libero_pytorch> --preset lossless   # 2cam pool=1 steps=10 cache=1
   # 再跑 --preset production（cache=2）、--preset balanced（steps=5）、--no-int8（BF16 参考）
   ```
7. **精度对拍**：FlashRT 输出 vs 我们 naive JAX 输出，同 obs 同 noise，算 cosine（应 ≈1.000 lossless）。
8. **（可选）闭环验证**：把 FlashRT 包成 websocket server，跑 LIBERO sync replan=5，确认 124ms 下既实时（占空比 ~50%）又准确（成功率 ≈99%）——**直接实证墙消解**。

### 1.4 度量 / 判据

- **度量**：p50/p95/min 延迟、Hz、cosine vs BF16、（闭环）成功率 + 占空比。
- **成功判据**：lossless p50 < 250ms 且 cosine ≥ 0.999 ⇒ 验证 roofline + 墙消解。
- **kill/降级判据**：若 build 或 checkpoint 转换在 SM87 上不可逾越（如 FlashRT 不支持 pi05_libero、PyTorch wheel 不可得）⇒ 记录具体障碍，退化为「引用 FlashRT 公布数字 + 我们 roofline 分析」，不硬磕。

### 1.5 风险与缓解

| 风险 | 缓解 |
|---|---|
| FlashRT 只验证过 pi05_droid，pi05_libero 兼容性未知 | 先小规模 smoke（随机 obs 跑通 + cosine），再上闭环 |
| dog 缺 nvcc/cmake/PyTorch wheel | 用 FlashRT docker 或 jetson-ai-lab pip index 的 aarch64 wheel |
| JAX→PyTorch 转换 norm-stats/adaRMS 对齐 | 转换后先 cosine 对拍 naive JAX，确认数值一致再测延迟 |
| GPU 挂死（dog 有前科） | 先确认重启后干净；benchmark 短跑、采时钟；哨兵只观察不 kill |

### 1.6 原创性边界（诚实）

方案 A 本身是复现，**单独含金量低**。它的价值是：(a) 在我们硬件上独立印证 roofline 预测；(b) 给方案 B 提供快推理基线。**叙事上必须把它降级为验证脚注**，原创头条留给方案 B。

---

## 2. 方案 B：自适应 prefix 缓存分层调度器（原创贡献）

### 2.1 动机（接 roofline）

roofline 指出 **prefix 占 71% 成本**，其中 SigLIP 视觉编码（~100ms）编码的是**场景语义，变化慢**；action expert（29%）需要**新鲜本体感觉**但可复用视觉上下文。⇒ **把慢变的视觉 prefix 缓存降频、让快 expert 高频跑**，是从架构上破墙（与 Figure Helix 分层同思路，但落在 π0.5 单体策略内部）。

### 2.2 核心假设

- **H_B1（前沿优势）**：自适应变化门控的缓存策略，在「实时 × 准确」前沿上**优于固定帧复用**——静态场景复用省算力（如 cache=2），动态场景重算保精度（不像 cache=2 盲目复用掉精度）。
- **H_B2（外科式陈旧）**：只让**慢变的视觉 prefix** 陈旧、保持**本体感觉新鲜**，比整帧 obs 陈旧（prefetch 墙的病根）更保精度。
- **H_B3（破墙）**：自适应缓存把有效推理延迟压到 < 250ms，使 sync replan=5 既实时又准确。

### 2.3 架构与算法设计

**架构**：策略服务维护**每会话视觉 prefix 缓存**（SigLIP 特征 / 图像 token KV）+ 上一帧图像。每次查询：

```
Δ = visual_change(current_image, cached_image)   # 便宜信号，无需跑模型
if Δ < τ:
    reuse cached visual prefix                    # 跳过 SigLIP 重编码
else:
    recompute visual prefix (SigLIP)              # 场景变了，重算
re-embed state + prompt (每步新鲜)
run Gemma2B prefill (over cached-or-fresh visual + fresh state) + action expert
```

**变化门控 Δ（由便宜到贵）**：
- 基线：下采样帧的像素 MAD / SSIM（无模型、~ms 级）。
- 进阶：轻量光流幅值 / 分块差分（捕捉运动）。
- （可选） tiny 学习变化检测器（接后期 Kaggle 微调）。

**门控阈值 τ**：扫描 τ ∈ {0（=每次重算，lossless 基线）, …, ∞（=盲目复用，cache=2 基线）}， traced 出 latency-accuracy 前沿。

**🔴 关键技术问题（需源码核实）**：pi05 把 **state 作为离散 language token 放进 prefix**，故 prefix KV 同时依赖图像与 state。可缓存的应是**视觉部分**（SigLIP 特征 / 图像 token 的 KV），state/prompt token 每步重嵌。需核实 `embed_prefix`/`prefill` 结构，确定「缓存视觉 KV + 增量更新 state token KV」的可行切分（FlashRT cache_frames=2 复用的是 encoder K/V，可参照）。**这是方案 B 的实现核心难点，先做源码勘察 + 最小原型验证可缓存性。**

### 2.4 与已有方法的区别（原创性论证）

| 方法 | 缓存策略 | 我们的区别 |
|---|---|---|
| FlashRT cache_frames=1 | 每次重算（无损） | 我们自适应：静态时省算力 |
| FlashRT cache_frames=2 | 固定复用 1 帧（cos 0.991） | 我们**变化门控**：动态时重算保精度，不盲目复用 |
| VLA-Cache / OxyGen | 稳定视觉 token KV 复用 | 我们给出**自适应门控 + 控制时间预算耦合 + π0.5/Orin 实证前沿** |
| Figure Helix | 双模型分层（需训练快策略） | 我们**单体策略内部** prefix/expert 分层，无需重训 |

**原创点**：自适应变化门控（vs 固定帧）+ 外科式陈旧（视觉 vs 整帧）+ 控制时间预算耦合的调度前沿，落在 π0.5/Orin 具体平台并实证。**推理期可做、基本无需训练**。

### 2.5 实验矩阵

| 轴 | 取值 |
|---|---|
| 缓存策略 | 每次重算（基线）/ 固定复用 1 帧（基线）/ **自适应 τ 扫描** |
| 变化信号 | 像素 MAD / 光流（消融） |
| replan_steps | 5 / 10 |
| num_steps | 10 / 5 / 1（接 quality 轴） |
| sync/prefetch | sync（主）/ prefetch（对照） |

每格测：**成功率（LIBERO spatial，≥100 ep + CI）、平均推理延迟、重算比例、占空比、cosine vs 全重算**。

### 2.6 度量 / 判据

- **成功判据**：自适应策略在前沿上**帕累托优于**固定复用（同延迟下成功率更高，或同成功率下延迟更低）；且存在 τ 使 sync replan=5 占空比 <100% 且成功率 ≈99%（破墙）。
- **kill 判据**：若可缓存性核实失败（state-in-prefix 无法干净切分视觉缓存）⇒ 降级为「SigLIP 特征级缓存」（只省 100ms 视觉编码）或记录架构障碍；若自适应前沿不优于固定复用 ⇒ 诚实记录负结果 + 分析为何（场景动态度分布）。

### 2.7 风险

| 风险 | 缓解 |
|---|---|
| state-in-prefix 可缓存性切分复杂 | 先源码勘察 + 最小原型；退路=SigLIP 特征级缓存 |
| 动态场景陈旧视觉掉精度 | 这正是要测的；τ 扫描 + 成功率 CI 量化 |
| 像素 MAD 门控不够语义 | 消融光流/分块；必要时 tiny 检测器 |
| 实现需改 openpi 模型代码 | 包装 policy 而非改源码（参照 serve_policy_numsteps 注入手法） |

---

## 3. 排序与叙事整合

- **依赖**：A 先（简单，验证 roofline + 给快推理基线）→ B 后（原创，建立在 A 的快推理认知上）。
- **文档修正**（无论 A/B 进度）：REPORT.md / demo 页的「Orin 不可实时 / 实时与准确不可兼得」须改为「**naive JAX 撞墙；INT8+工程可达 8Hz 实时（FlashRT 已证、我们复现验证）；自适应分层调度进一步破墙（方案 B）**」。
- **最终叙事**：「系统诊断主流 VLA 边缘实时瓶颈 → 建立有预测力的分析框架（roofline + control-time 不等式，被工业界/顶会独立验证）→ 实证墙与解法 → 原创自适应调度前沿」。含金量在**分析原创 + 调度原创**，复现是脚注。

## 4. 方案 A 执行结果（2026-11-23，dog Orin 32GB）

**✅ 核心结果：FlashRT π0.5 INT8 W8A8 无损路径在 dog 复现成功。**

| 配置 | p50 | p95 | Hz | 备注 |
|---|---|---|---|---|
| naive JAX BF16（G2 基线） | ~490 ms | — | 2.05 | 本项目原始 |
| FlashRT BF16（融合+CUDA graph） | 178.8 ms | 179.2 | 5.59 | `--no-int8` |
| **FlashRT INT8 W8A8 + cache_frames=1（无损）** | **128.6 ms** | 128.9 | **7.77** | `--preset lossless`，Actions (10,7) ✓ |

**加速链分解（验证 roofline）**：
- INT8 / FlashRT-BF16 = 178.8/128.6 = **1.39× ≈ roofline 预测的 W8A8 算力乘数 1.37×** ⇒ **roofline 被精确验证**。
- naive JAX / FlashRT-BF16 = 490/178.8 = **2.74×** = kernel 融合 + CUDA graph 消除的工程差距（naive JAX 离 roof 远）。
- naive JAX / INT8 = 490/128.6 = **3.81×** 总加速。
- **128.6ms < 250ms（replan=5 预算）⇒ 同款 Orin 上 sync replan=5 实时可行**，墙消解（与调研修正一致）。
- 对照 FlashRT 公布 64GB 结果 124ms/8.04Hz：我们 32GB 得 128.6ms/7.77Hz，微差（内存带宽/热态/DVFS），量级一致。

**Setup 踩坑记录（可复现）**：
1. host cmake 3.22 太旧（需≥3.24）+ torch wheel 下载限速 ⇒ **改用 openpi 容器**（自带 torch 2.11 CUDA/Orin、cmake 3.31、nvcc 12.6、sentencepiece）。
2. cmake 缺 pybind11 ⇒ 容器 pip 装；cmake 错用 /usr/bin/python3 ⇒ 改 `/opt/venv/bin/python3` + 显式 `pybind11_DIR`。
3. FA2 hdim256（非因果+因果）编译极慢（Orin ARM ~40min 总 build）。
4. **checkpoint 转换 OOM**：dog 是 **32GB**（FlashRT 在 64GB 验证），float32 restore 峰值~30GB 被杀；bf16 restore 仍 OOM。**解法=低内存转换**：bf16 restore + `torch.from_numpy` bf16→float32 shim + **跳过完整 PI0Pytorch 模型构建**（slicing 已产 HF 名 state dict，直接 `save_file`）+ `.contiguous()`，峰值~22GB 通过。
5. **tied weights**：save_file 不含 `lm_head.weight`（PaliGemma tied 到 embed_tokens）⇒ 补 `lm_head.weight = embed_tokens.weight.clone()`。
6. 产物：`pi05_libero_pytorch/model.safetensors`（6.7GB bf16）+ config.json + assets。脚本存 `/home/vision/chenrunbin/g5_flashrt/FlashRT/`（patch_lowmem.py / run_lowmem.sh / bench_*.sh）。

**待补**：① 精度对拍（FlashRT INT8 vs naive JAX 同 obs cosine，确认转换正确+无损）；② 闭环成功率（FlashRT 包 websocket server 跑 LIBERO sync replan=5，确认 128ms 下既实时又≈99%）；③ production preset（cache_frames=2，~12Hz）可选。

**原始日志**：`results/g5_flashrt/bench_lossless_int8.log`、`bench_bf16_reference.log`。

## 5. 来源

- FlashRT Orin 指南 https://github.com/flashrt-project/FlashRT/blob/main/docs/deployment_orin.md
- FlashRT USAGE / bench_pi05.py https://github.com/flashrt-project/FlashRT/blob/main/USAGE.md
- NVIDIA Jetson AI Lab《OpenPi π0.5 on Thor》https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- PhyAI 论文 https://arxiv.org/html/2608.03682v2
- Seeed《GR00T N1.7 全 TRT on AGX Orin》https://wiki.seeedstudio.com/deploy_full_weight_gr00t_n1.7_tensorrt_jetpack7.2_agx_orin/
- Figure Helix https://www.figure.ai/news/helix ；Modal/PI https://modal.com/blog/physical-intelligence-runs-real-time-remote-inference-for-robotic-control-on-modal
- 完整证据库见记忆 `vla-realtime-edge-deployment-landscape`