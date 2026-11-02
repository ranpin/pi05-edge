# PROBLEMS · 全流程问题记录

> 统一问题记录（**唯一入口**）。呈现页 [REPORT.md](REPORT.md) 不内嵌任何踩坑/debug 叙事，只链到本页。
> 每条按 **现象 → 根因 → 解决 → 教训** 记。

---

## A. 容器构建（P0：jetson-containers 现场 build openpi，29 包链）

### 坑 1 · containerd snapshotter 无视 data-root，撑满根分区导致 apt 全源「invalid signature」

- **现象**：cuda 相关 apt 源全部报 `invalid signature`，gpgv 校验失败，看起来像 GPG key 过期或源被污染。
- **根因**：containerd 的 snapshotter 把镜像层与构建缓存写进 `/var/lib/containerd`（**根 eMMC**），**无视 docker 的 `data-root` 配置**。29 包链每包都产 20–34 GB 中间镜像，根分区被撑满；磁盘满会让 gpgv 写临时文件失败，从而伪装成「签名无效」。
- **解决**：把 containerd root 迁到 NVMe。
- **教训**：**apt 全源同时报 invalid signature，第一件事是 `df -h`，不是去查 key。** 单源签名错才是 key 问题，全源错基本是磁盘/权限。

### 坑 2 · cuDSS 自检在 Orin sm_87 误报

- **现象**：构建到 cudastack 相关包时自检失败。
- **根因**：自检逻辑对 sm_87（Tegra Ampere）判断有误；而 π0.5 根本用不到 cuDSS。
- **解决**：全链 `--skip-tests=all`。
- **教训**：构建系统的自检不等于业务正确性；用不到的组件其自检失败可以直接跳过，但**要记下来**，别日后误以为已验证。

### 坑 3 · apt.llvm.org 限速 ~1 kB/s

- **现象**：`llvm:22` 包卡了 65 分钟；一个 834 kB 的 `libclang-common-22-dev` 下了 ~12 分钟。日志序号在涨、字节在增，看着像「慢但在推进」。
- **根因**：境外源在国内网络限速到 kB/s 级。
- **解决**：`sed` 改 `llvm.sh` 的 `BASE_URL` 指向清华 `mirrors.tuna.tsinghua.edu.cn/llvm-apt/`，重跑后该包秒过（tuna 命中 55 次、apt.llvm.org 0 次）。
- **教训**：**判据是速率，不是「是否还在推进」。** 慢到几十分钟一个包就该换源，别把「再等等」当方案。改源时**只改仓库 URL、别动 GPG key 地址**（镜像常不同步 key，key 仍走原站、几 kB 秒下）。

### 坑 4 · pypi.jetson-ai-lab.io 的 torch 轮子限速 ~15 KB/s

- **现象**：torch 大轮子下载极慢，且服务器**不支持 Range**（无法断点续传），国内也无对应镜像。
- **根因**：该索引面向海外，国内链路差；无 Range 意味着一断就得从头。
- **解决**：`wget` 预下载轮子到本地 → patch 成 local-wheel 安装 → **删掉 `build.sh` 里的源码编译兜底**（防止下载失败后静默转入源码编译，那会 OOM）。
- **教训**：慢源且不支持 Range 时，「预下载 + 本地安装」比「重试」可靠。同时要**堵掉静默降级路径**——源码编译兜底看起来是保险，实际是把一个可诊断的下载失败变成一个耗时数小时的 OOM。

### 坑 5 · cuDSS soname 错位

- **现象**：torch 链接 `libcudss.so.0` 失败。
- **根因**：cudastack 只提供 `libcudss.so.1`，soname 对不上。
- **解决**：建软链 `libcudss.so.0 → libcudss.so.1`。
- **教训**：Tegra 上第三方栈的 soname 经常与上游 x86 约定不一致，链接失败先看实际提供了哪个 `.so.N`。

### 坑 6 · opencv 4.13.0 无预编译包

- **现象**：拉 opencv 4.13.0 报 404。
- **根因**：该版本没有对应 Tegra 预编译包，**4.12.0 是唯一有的**。
- **解决**：降到 4.12.0。
- **教训**：Tegra 上的可用版本集合由预编译产物决定，不由上游最新版本决定。404 先查「哪个版本有包」，别急着改源。

### 坑 7 · openpi 的 `[tool.uv] override-dependencies` 静默降级整个 jax 栈

- **现象**：装出来的 jax 栈版本与预期不符，base 镜像里已装好的 jax 被降级。
- **根因**：openpi `pyproject.toml` 里有 `[tool.uv] override-dependencies = ["ml-dtypes==0.4.1", ...]` 加两处 `numpy<2` pin。`uv pip install -e .` **会honored 这个 override**，从而连带把 base jax 栈拽下去。
- **解决**：删掉 override、把 numpy 放宽到 `>=1.26`、step 9 补 `ml_dtypes>=0.5`；flax 还原成 openpi 自己 pin 的 `0.10.2`（此前误改过）。
- **教训**：**`uv pip install --dry-run -r reqs.txt` 不读 `[tool.uv]`，必须拿真 pyproject 去 dry-run 才看得见 override。** 版本对不上时先查有没有 override/约束文件在暗中生效，别只盯直接依赖。

### 坑 8 · uv 的 flag 名 + `--no-deps` 的传递依赖缺口

- **现象**：`uv pip install --no-dependencies` 报 `unexpected argument`；改用 `--no-deps` 后安装成功但运行期 import 失败。
- **根因**：uv 只认 `--no-deps`。而 `--no-deps` 会跳过传递依赖，`nest_asyncio` / `wadler-lindig` 没人装。
- **解决**：在**已建好的镜像里**跑 `uv pip install --dry-run` 把缺口列全，再补进 Dockerfile。
- **教训**：用 `--no-deps` 就必须自己负责闭包。**在已建镜像里 dry-run 是列缺口最便宜的办法**——比读源码猜依赖可靠。

### 坑 9 · openpi 源码用了 Python 3.11 API

- **现象**：`datetime.UTC` 报 AttributeError。
- **根因**：`datetime.UTC` 是 Python 3.11 新增别名，容器内是 **Python 3.10.22**。
- **解决**：`sed` 成 `datetime.timezone.utc`。
- **教训**：上游未声明的最低 Python 版本可能高于基础镜像；这类错在 import 期就暴露，属便宜修复。

### 坑 10 · 🔑 最深坑：jax GPU 在 Tegra sm_87 必崩

- **现象**：任何真实 GPU 算子都崩；gdb 实锤是 `LOG(FATAL)`，位置在通用 `jax-cuda12-plugin` 的 **`GemmFusionAutotuner` redzone 分配路径**。换 jax 版本、换 cuDNN 版本（9.8 sbsa / 9.8 tegra）都在 conv 段错误。
- **根因**：autotuner 的 redzone 分配在 sm_87 上走不通，**与 jax / cuDNN 版本无关**。另外 jax 0.6.2 的 plugin 要求 cuDNN 9.8（CUDA 12.8 领域），而 JetPack 6.2 只带 **Tegra cuDNN 9.3**。
- **解决**：**jax 0.5.3 全家桶**（`jax` / `jaxlib` / `jax_cuda12_plugin` / `jax_cuda12_pjrt` 全 0.5.3，这也是 openpi 自己的 pin；其 plugin 元数据要求 `cudnn>=9.1,<10`，恰好被系统 cuDNN 9.3 满足）+ 烘焙
  ```
  ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0"
  ```
  **`autotune_level=0` 是核心解药**（直接关掉 autotuner，绕开崩溃路径）。
- **代价对比**：另一条路是自己编译 jax for Tegra，估 3–6 h 且不一定成。上述方案免掉了这次源码编译。
- **教训**：
  1. **验 GPU 必须用非常量输入的真算子。** `jnp.ones(...).sum()` 会被常量折叠、根本不碰 GPU，即使 GPU 计算路径已坏也会「通过」——这个静默失败在本项目上花掉过数小时。镜像构建期的 step-11 因此硬化为 `jax.jit` 一个真 conv + `assert jax.default_backend()=='gpu'`。
  2. 崩溃点在 autotuner 时，**先关 autotune 再考虑换版本**——版本组合的搜索空间远大于 flag 空间。
  3. 选 jax 版本要**对着 plugin 的 cuDNN 元数据约束选**，而不是对着 jax 自己的版本号选。

### 坑 11 · 补丁必须 live + logs 快照 Dockerfile 双打

- **现象**：改了 `packages/physicalAI/openpi/Dockerfile` 后重跑单包构建，改动没生效。
- **根因**：jetson-containers 的 `NoM_*.sh` 用 `--file` 指向 `logs/<ts>/build/` 下的**快照 Dockerfile**，而不是 live 文件；快照还经过预处理（syntax-labs header、ccache ENV、每个 RUN 挂 `--device=nvidia.com/gpu=all --mount=type=cache`）。
- **解决**：所有补丁同时打到 live 与快照两份；构建脚本头部加断言打印补丁计数（`seds=5 | checkout-pin=1 | xlaflags=1 | jax053=1 | pins`）以便一眼核对。
- **教训**：**构建系统可能不读你以为它在读的那个文件。** 改完先确认「构建实际用的是哪份」，并把补丁存在性做成构建期断言，而不是靠事后看日志。

### 坑 12 · 「日志安静」≠ 卡死

- **现象**：构建长时间无输出，疑似卡死。
- **根因**：containerd 导出大镜像是 CPU-bound；缓存链每轮会断。
- **解决**：先 `tail` 日志尾 + 看 `cur=` 当前在哪个包，再判断是 cxx 编译还是 OOM。
- **教训**：判卡死要看**具体在做什么**（当前包、CPU 占用），不是看有没有新行。误判成卡死而中断，会浪费已完成的整条缓存链。

---

## B. 运维：GPU 驱动挂死（2026-10-30）

- **现象**：g2_verify 复跑时大显存分配的 ioctl 永久挂死。`loadavg` 高达 ~85 但只有 2 个 running 进程；NVMe `diskstats` 零增量、RAM 10 Gi available 正常；`docker top` / `ps` / `tegrastats` **全部超时**，而 `tail` / `stat` 秒回。
- **根因**：Tegra 上 GPU 进程**反复 core dump**（当日 6+ 次 jax 0.6.2 / cuDNN 9.8 实验崩溃）会累积劣化 nvgpu 驱动，后续大分配 ioctl 进 **D 态**（不可中断睡眠）永久挂住。
- **解决**：**重启 Orin**。D 态进程 `SIGKILL` 无效；非 root 无法重载驱动；部署用户在 sudo 组但**无免密 sudo**，所以只能由用户手动执行。
- **教训**：
  1. **崩溃连发后应盯 loadavg / D 态、早重启**，别把已劣化的驱动带进正式 run。
  2. 「loadavg 极高但 running 极少」+「部分命令超时、部分秒回」是驱动/内核态挂死的特征指纹，不是负载高。
  3. 重启后 GPU 干净，延迟反而略好 ~2%（见 REPORT 表 1）——**劣化的驱动会污染性能数字**，所以正式测量前应确认机器没带伤。

---

## C. G3 客户端依赖：torch 阻塞已解除（2026-10-30 实证）

- **现象**：官方 `examples/libero/main.py:8` 有 `from libero.libero import benchmark`，而 `benchmark/__init__.py:5` 有 `import torch`。仿真机的 G1 venv **绝不能装 torch**（现代 torch 会把 numpy 顶到 2.x，连带搞崩为 numpy 1.x 编译的 `opencv-python 4.6` 和要求 `numpy<1.24` 的 `numba 0.56.4`）。G1 当初是绕过 benchmark、直接用 bddl 建 env 才跑通的；G3 需要 benchmark 枚举任务，绕不过去了。
- **根因**：`import torch` 在整个 benchmark 模块里**只被用在一行**——`benchmark/__init__.py:164` 的 `init_states = torch.load(init_states_path)`。
- **解决（已实证）**：`.pruned_init` 的魔数是 `504b`（ZIP）= `torch.save` 的容器格式，内含 `archive/data.pkl` + `archive/version`，**没有 tensor storage 条目**（数组数据内联在 pickle 里）。该 pickle 流的 GLOBAL 引用**只有 numpy**（`numpy.core.multiarray._reconstruct` / `numpy.ndarray` / `numpy.dtype`，dtype `f8` 小端）+ `_codecs.encode`，**零 torch 引用**。实测
  ```python
  pickle.loads(zipfile.ZipFile(p).read("archive/data.pkl"))
  # -> ndarray len 50，元素 (92,) float64；torch 全程未进 sys.modules
  ```
  ⇒ 把那一行 `torch.load` 换成 zip + pickle 读即可，**仿真机依赖栈一行都不用动**。
- **教训**：
  1. **「这个模块 import 了 X」不等于「这个模块需要 X」。** 先 grep X 的**实际使用点**，常常只有一两处，且往往可替换。
  2. 判断一个 pickle 能否脱离某框架加载，**用 `pickletools` 扫 GLOBAL 引用**比试装框架可靠得多，而且能在不污染环境的前提下给出确定答案。
  3. `torch.save` 的产物不一定含 torch tensor——**容器格式是 torch 的，内容可以是纯 numpy**。魔数 `504b` 提示是 ZIP，直接按 ZIP 拆即可。

---

## D. 方法论教训（跨坑通用）

- **别为一个 bug 全量重来。** 29 包链构建数十小时，修一个包时应复用已建好的镜像 1–28，只重建目标包（本项目用单包重建脚本做到）。用户曾就此明确纠正过。
- **字符串补丁必须配唯一性断言。** 给第三方源码打 `sed` 补丁时，每处 needle 都要 `assert count == 1`，让上游漂移在几十秒内暴露，而不是「看似打上了」后产出一个错的东西。
- **把补丁存在性做成构建期断言。** 构建脚本头部打印补丁计数（见坑 11），比事后翻日志可靠。
- **静默降级路径要堵死。** 下载失败转源码编译（坑 4）、CPU 回退冒充 GPU 成功（坑 10）都属于「看起来成功了，其实退化成了另一件事」。改成构建期硬失败。
