# 任务8:防御机制普查与裁决报告

## 0. 总量盘点

| 指标 | 迁移分支 (Zou_lab_control_v1) | 主线 (ZLC_main) |
|---|---|---|
| sha256/digest/fingerprint/checksum 命中行 | **877 行 / 119 文件** | 224 行 / 33 文件 |
| `fingerprint` 单独命中 | 537 处 / 107 文件 | ~120 处 |
| `__post_init__` 验证块 | **384 处 / 111 文件** | 49 处 |
| `sha256_text`/`_sha256()` 格式复验调用点 | 46 处 | 0 |

用户的体感是对的:迁移分支的防御密度是主线的 **4 倍**,且大部分集中在 `zlc_pulse` / `zlc_neutral_atom/devices/sequencer` 的"身份链"上。

## 1. 关键背景:这个分支已经清洗过一次防御爆炸

裁决前必须知道的事实:迁移分支**自己已经承认并删除过一轮防御过剩**,证据:

- 设计文档 `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md:509`:「允许保留的 fingerprint/digest **只在真实 FPGA/transport 比较边界**……不得泛化为普通 artifact identity;普通 schema evolution 使用可读版本字段而不是内容 hash」;`:510` 明令禁止 content/reference/span/join digest 链;`:576` 要求「普通 SHA/fingerprint/CAS/repository lease **在生产、测试和文档中为零**」。
- `docs/MAINTAINER_NOTES.md:40`:installation digest/CAS/storage file lock 已删。
- meta-测试 `tests/test_architecture_import_dag.py:1046-1087` 强制 `ContentAddressedStore`/`join_digest`/`payload_digest`/`span_digest` 永不回潮;`:1090-1110` 禁 `numeric_backend_digest` 重获准入角色。

即:**sha256 爆炸是这个项目的已知病史**(CAS、逐事件 binding digest、numeric backend digest 都长出来过又被拆),现存的是"允许边界内"的存量——但边界内仍然过重(见下表 M3/M6/M7/M10)。

## 2. 裁决表

图例:🟥必留 🟨可简化 🟩该删。"真实?"= 是否有可查证的真实故障锚点。

| # | 机制 | 代表位置 | 防什么 | 真实? | 裁决 |
|---|---|---|---|---|---|
| M1 | **geometry fingerprint 握手**(`build_fingerprint`→CTRL word63,connect 比对) | 两树共有 `fpga/pulse_streamer/host/image.py:53-88,1030-1035`;主线 `neutral_atom/devices/axi_session.py:470-490`;迁移分支 `zlc_pulse/deployment.py:36-63` | 混跑旧 bitstream / config↔bitstream 几何漂移静默损坏 | **真实**:主线 `docs/MAINTAINER_NOTES.md:1025`「2026-06-11 garbled-first-frame DA 根因」;`docs/ROADMAP.md:45` 记录 bus_seg 漂移那次静默损坏;`REAL_HARDWARE_BRINGUP_zh.md:112,186` 写明是设计保护 | 🟥 **必留,原样迁移**(哈希单源在 image.py 一处,RTL 只携带值——这个 DRY 结构也值得保) |
| M2 | PortCatalog 拓扑 fingerprint(pulse 文档 vs 连接设备) | 主线 `neutral_atom/ports.py:181-198`;`devices/sequencer.py:582-595`;`pulse_gui.py:241` | 用旧 lane 拓扑的 pulse 模板编译到不匹配的 sequencer | 设计驱动,未见事故记录,但错误信息给出双向 lane diff,是给真机接线用的 | 🟥 必留概念;实现可以是 typed catalog 相等比较,hash 只是省一个深比较,16-hex 短哈希够用 |
| M3 | 编译产物内部 digest 绑定(document→IR→wire→artifact 四级指纹,构造时互验) | `zlc_pulse/artifact.py:51-102`(`wire_image.source_ir_digest != target_ir.fingerprint` 即拒),`ir.py`,`fpga.py` | 编译产物"拼装错"——同进程内自己编译自己拼 | 想象。编译器是唯一装配点(`compiler.py:306,457`),同进程构造错拼无现实来源 | 🟨 保留 **artifact 顶层一个 fingerprint**(M4 需要);删内部三级 digest 镜像,子对象直接持引用 |
| M4 | **prepare/fire/completion digest 回声协议**(跨进程) | `zlc_neutral_atom/devices/sequencer/endpoint.py:287`「FIRE artifact digest differs from prepared session」`:326`;`port.py:405,723-741`;`zlc_pulse/server.py:701` | 远程 server 会话里 fire 了不是 prepared 的 artifact(双客户端/重连/陈旧会话) | 半真实:server+UART/AXI transport 是真实跨进程边界(`zlc_pulse/transport/` 有 InterprocessDeviceLease、UART link);主线的简化版(`sequencer.py:1525` program fingerprint 比对)同源 | 🟥 必留**一处**:协议里 artifact_digest 一个字段,prepare 回执/ fire 校验。现状是同一比对散布在 endpoint、port、application、server 四层各查一遍——收敛到协议边界单点 |
| M5 | 相机帧归属 cause_digest(帧→pulse artifact 关联) | `zlc_neutral_atom/devices/camera/endpoint.py:1216-1228`;`devices/simulation/apparatus.py:1280` | 帧被算到错误的 pulse run 头上 | 关切真实(异步管线帧归属),手段过度:同进程内 run/session id 即可,content hash 无增量收益 | 🟨 换成 run_id/token 相等 |
| M6 | capture cell plan / lineage digest 镜像 | `timing/capture_plan.py:186-189`;`timing/lineage.py:59-68`(手里同时有 `compiled_artifact` 对象却比 digest) | 计划绑到了另一个 artifact | 想象。绑定发生在同进程,digest 是对象引用的冗余镜像 | 🟩 该删,直接持对象/比 id |
| M7 | `target_abi_fingerprint` 多点镜像复验 | 定义 `zlc_pulse/ir.py:76`;复制到 `scan_columns.py:98`、`physical.py:148`、`readout/physical_context.py:147`;复验 8 处(`port.py:188,585`、`server.py:701`、`timeline.py:306`、`simulation/apparatus.py:589`…) | artifact 用到了错误 target 上 | 关切真实(等价于 M2),但一次检查就够 | 🟨 收敛到 prepare 单点(`port.py:585`),删其余 7 处镜像字段 |
| M8 | `APPROVED_DEPLOYED_TARGET_ABI` 硬编码批准 digest | `zlc_pulse/deployment.py:31-33,141-142` | 未经批准的 host 拓扑上真机(frozen-RTL SOP) | 流程性;注释自己承认「does not attest the running .bit」,真正的 gate 是 M1 | 🟨 新架构可删;若保留 frozen-RTL SOP,留这一个常量+一处比较无伤 |
| M9 | canonical 编码器全套自证(decode 双往返、base64 pad-bit、numpy ndim 二分探测) | `zlc_storage/canonical.py` 全文 **701 行**:`:272,288`(decode 后 re-encode 必须字节等于原文)、`:297-312`(pad-bit)、`:168-196`(import 时二分探测 ndarray rank 上限) | 非规范载荷产生不同 digest;恶意构造的 payload | 想象。单用户实验室工具读自己写的文件,没有对抗方 | 🟨 **核心值得保**:确定性序列化(sorted map、little-endian C-order ndarray、NaN 归一)+ `canonical_digest`,~150 行足够;删往返自证、pad-bit 审计、ndim 探测(~500 行) |
| M10 | `sha256_text` 格式仪式(每个 dataclass 验字段是 64 位小写 hex) | `zlc_storage/canonical.py:55-67`,46 个调用点(`sequencer/port.py:76,233,260,278…` 密度最高) | 字段里放了不是 sha256 的字符串 | 想象 | 🟩 该删。digest 只在协议边界一处产生一处比较后,格式验证无对象 |
| M11 | schema_fingerprint(数据集 schema 内容指纹) | `zlc_data/schema.py:386-418,458-488`;消费在 `zlc_workbench/task_console/panel_card.py:354-369`(schema 变→重建 plot host) | 不是防御——是**变化检测/缓存键** | n/a | 🟨 概念保留(plot 重建判据),实现可以是 schema 对象相等;若 schema 常驻则 hash 一次也合理。别把它算进"防御债" |
| M12 | terminal evidence 类型化 + 自身 fingerprint | `zlc_pulse/evidence.py`(380 行);docstring `:1-7` 明言「frozen RTL 不暴露 digest……不得在此制造这些声称」 | raw STATUS/CURSOR 采样记录=真机排障证据(`REAL_HARDWARE_BRINGUP_zh.md:232` S2 验收用);evidence 自身的 fingerprint 无生产消费者(只有 `test_zlc_pulse_terminal_evidence.py:94` 断言 getter 不重算) | 记录真实有用;指纹是防"AI 伪造证据"的流程剧场 | 🟨 留 raw 寄存器采样记录,删 evidence 自身 fingerprint 及其测试 |
| M13 | 架构 meta-guard 测试(import DAG、digest 单源、禁回潮 token 扫描) | `tests/test_architecture_import_dag.py` **1240 行**(`:477` digest 公式单 owner、`:1046` 禁 CAS 回潮、`:211` 反向 import) | AI 在同一仓库里重新长出已删模式 | **历史真实**(CAS 等确实长出来过),但它防的是"单仓库 vibe-coding"这个工作方式本身 | 🟩 拆包后大半该删:独立仓库+pyproject 显式依赖让 import DAG 物理成立;禁回潮扫描失去对象。可保留每包内部 1-2 条真正的单源契约测试 |
| M14 | `__post_init__` 全量类型/不变量验证 | 384 处/111 文件(vs 主线 49);典型 `zlc_pulse/artifact.py:51-100` 50 行验证 | 坏参数进入 frozen 值对象 | 边界处合理;内层层重复是仪式(内部调用方全是自己) | 🟨 只在包公共 API/反序列化边界验;包内构造信任类型标注 |
| M15 | 目录持久性 flush(ctypes FlushFileBuffers) | `zlc_storage/durability.py:16-42` | 断电丢目录项 | 想象(实验数据在 Dropbox 同步目录,风险收益不成比) | 🟨 留 atomic write(`os.replace`),删目录 flush |
| M16 | 存档文件自嵌 fingerprint(写时嵌 hash、读时复验) | 主线 `core/calibration.py:151-152,527-528`、`operations/imageio.py:224-225` | 文件被手改/损坏 | 想象(自己写自己读)。**但同文件里真正有价值的是 `FrameContract.assert_image` 形状检查**(`calibration.py:179`,`tests/test_audit_hardening.py:391-404` round-3 M2:ROI 变了必须响亮失败而非静默取错像素) | 🟩 自嵌 hash 该删,留 schema/version 字段;🟥 FrameContract 形状/几何检查必留 |
| M17 | sha256 作稳定 ID/缓存键 | 主线 `devices/sequencer.py:552`(sequence_id[:16])、`frontend/task_console.py:4827`(program_id)、`frontend/jupyter.py:98`(notebook cell id 确定性)、`fpga/.../src_hash.py`(build 缓存键) | 不是防御 | n/a | 🟩 无债,保留。普查时别与防御混算 |
| M18 | 存档 provenance(verilog source_sha256、scan_code 往返) | 主线 `timing/verilog.py:143`、`core/results.py:308` | 事后无法复现当年的 run | 真实教训但结论是**存源**(memory: scan-code-round-trip,ddd9b01),hash 只是附赠 | 🟨 存源必留;hash 可留可删 |

## 3. 结构性诊断(给拆包决策)

**1. 真正被真机咬过的防御只有一个家族:host↔硬件边界。** M1(geometry word63)有两次记录在案的静默损坏事故;M2/M4/M7 是它在软件侧的同构延伸。这条边界在新架构里应该是 `zlc_pulse`(或 fpga host 包)的**公共 API 内建行为**:`connect()` 自动握手、`prepare()` 返回含 digest 的回执、`fire(digest)` 校验——调用者感知不到防御存在。

**2. 迁移分支的病不是"有 digest",而是"digest 镜像"。** 同一个事实(artifact 身份、target ABI)被复制进 5-8 个值对象,每个 hop 再互相比一遍(M3/M6/M7),再配上 46 处 `sha256_text` 格式仪式(M10)。877 行命中里估计 **~70% 是镜像和仪式**,真正的比较边界只有 3 个:connect 握手、prepare/fire 回声、(可选)帧归属。这是设计文档 `:509` 自己划的线,现有代码在线内仍超重。

**3. meta-guard 测试是对工作方式的防御,不是对系统的防御。** `test_architecture_import_dag.py` 1240 行防的是"同一个仓库里 AI 重新长出被删的模式"。拆成独立包后,包边界由 packaging 物理强制,这类测试的存在理由随之消失——**拆包本身就是这 1240 行的替代品**。

**4. 值对象验证密度应随包边界重定价。** 384 处 `__post_init__` 是"每个类都是边界"的产物;拆包后边界只在包公共 API 和反序列化入口,包内构造应信任类型检查。

## 4. 新架构最小防御集(建议)

1. **geometry/ABI 握手**(M1):原样保留,含"哈希单源在 image.py、RTL 只携带值"的结构和 `build_fingerprint_covers_geometry` 类契约测试。
2. **prepare/fire artifact-digest 回声**(M4):协议里一个字段、server 校验一处、client 校验一处,共两个比较点。content hash(而非随机 token)值得保留——重连后仍可比对,且虚拟/真机同路径。
3. **拓扑相等断言**(M2/M7 合并):编译/prepare 前一次 `catalog == device_catalog`,错误信息保留双向 diff。
4. **FrameContract 形状/几何检查**(M16 的真核):calibration 应用到不匹配 ROI 必须响亮失败。
5. **一个 ~150 行的确定性序列化 + digest 工具**(M9 收缩版),仅供 1/2 使用,不对普通 artifact 开放(设计文档 `:509` 的裁决直接照抄)。
6. **存档存源**(M18):scan_code/verilog 源文本进存档;hash 可选。

**建议删除比例:按防御相关代码量,迁移分支约 65-75% 可删**(镜像 digest 字段与复验 ~200 行、sha256_text 仪式 46 处、canonical.py ~500 行、evidence fingerprint、meta-guard 测试大半、目录 flush、文件自嵌 hash);主线约 20-30% 可删(自嵌 hash、部分 require_* 仪式)。保留的 4+2 条里,只有 M1 是曾经流血换来的,优先级绝对高于其余一切。