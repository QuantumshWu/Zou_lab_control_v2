# RegularImage 共用批量数值核：当前候选证据

工作目录为 `C:/Users/eadri/WorkCode/zlc_v2_perf_limits_20260906`，基线 `ec706be2`；本文件记录未提交候选，不代表最终真实四图或全部科学行为验收完成。

## 当前结论

single 与 batch 已使用同一个 proxy→full TRF 数值流程，B1 是一条 lane；原单图 Python Newton/LBFGS 分支及失去消费者的 summary/objective 已删除。保留已有全部 fresh 正负初值、cold+warm 竞争、linear proxy 1e-5、robust proxy 与 full 原容差。没有新 production 文件、类、DTO、trusted flag 或第二 solver。

2026-09-07 最终同进程 A/B 使用**完整 saved radial + saved compiled**，model registry 相同；已包括最后同核直接 RSS 分块：

| 同一输入，交错 n3 | 旧 P50 | 当前 P50 | 差别 |
|---|---:|---:|---:|
| MOT uint8 1200×1920，B1 | 19.324 ms | 26.701 ms | +7.377 ms，+38.2% |
| 真实 A 捕获，40×43×502 uint8 | 27.520 ms | 24.699 ms | −2.821 ms，−10.3% |

这证明批量实质减量，并证明单图仍有代价；**不能仅因低于早期 2× 警戒线就把单图 +38.2% 宣称“小幅、已验收”**。此前（尚未并行最终 RSS）同进程 B1 为 18.810→27.151 ms、B40 为 28.626→23.921 ms；另一次无 hook 单独 B1 为 22.673 ms，但不能拿它与其它 run 的 18 ms 拼出更好的正式收益。最后 RSS 并行本身的独立收益没有由这些不同 run 证明，不单独宣称省了几毫秒。

最终原始证据：[B1 162500](major_fit_single_ab_162500.jsonl)、[精确 B40 161300](major_fit_single_ab_161300.jsonl)；前一阶段 [B1](major_fit_single_ab_167664.jsonl)、[B40](major_fit_single_ab_167436.jsonl)、[B1 分项](replay_exact_fit_167960.jsonl)。所有测量先 bootstrap、打印 root/Plot 路径，再安装 kernel cache；Numba team=4，未改全局 BLAS 设置。最后 B1 old 首调用缓存加载 348.6 ms / current 11.529 s（含修改后重编译）；随后独立 B40 新进程 old 首调用 407.6 ms / current 251.1 ms（已有 cache）。不是 steady 成本，也不冒充纯 JIT 时间。

## 真正删除的开销

- 内部 descriptor 明确 rectangular-grid 坐标布局，保留真实 x/y 轴和 shape；full 不再构造两份逐像素坐标。generic point 模型 ABI 与含义不变。
- 原已存在的 context 贯通 prepare→objective。full 每 lane 先减去一个真实有效观测 reference，再求 centered sum/squares；参数仍是物理 A/B/半径/中心，objective、gradient 与 GTOL 尺度不变。
- 同一 axis-Gram 模型公式与 BLAS 三列投影同时服务 B1/Bn。full 的 RSS 使用补偿和及浮点累积误差边界；接近抵消时退回同一直接残差公式，最终质量始终再执行直接物理残差。
- full 原始 dtype 输入 stack 后，一次已有 worker team 内的 lane×64-row stripe 核完成 float64 值、centered context 与 moments；不会先复制成整图 float64 再重复扫描。整批准备统计的有效点数复用到 full objective，不在每次模型评价再次扫描 2M 个 validity 位。
- generic compiled 在 `use_weights=False` 时不再分配 B×N float64 ones；所有 objective/finalizer 的权重读取由既有 `use_weights` 契约控制。这是所有模型共用的空权重表示，不是给 Image 特供短假数组。
- final information 使用一个 native batch，最后直接 RSS 在同一核按 lane×64-row stripe 执行，每 stripe 仍调用同一直接残差函数，最后按原 row 顺序相加；B1/Bn 同路。covariance 保留唯一 owner 并批量处理；单 lane 的异常矩阵不使其余 lane 一同失败。公共逐 cell FitResult 和 lazy arrays 仍在原 owner。

并行最终 RSS 之前的 B1 scoped profile 每 fit 平均值（3 次、嵌套项不可相加；不冒充最后版本新分项）：

| 阶段 | 当前 ms |
|---|---:|
| 两阶段 solver kernels 合计 | 8.867 |
| 最终 information + 直接 RSS | 4.222 |
| fused values/context 准备 | 2.426 |
| compiled 未细 hook 的 self | 2.861 |
| 原始值及小数组 stack 合计 | 0.756 |
| ones 合计（主要剩 validity） | 0.552 |
| covariance | 0.218 |
| FitResult 构造 | 0.120 |
| 坐标装包 | 0.017 |

无 hook P50=22.673 ms，带 scope P50=23.445 ms，扰动 +3.4%。这些是实际函数边界，不是精确 GIL 等待归因。相较此前同方法诊断，ones 4.463、stack 5.137、独立 centered context 3.828、solve 14.918 ms 的冗余大项已经实质减少；跨 run 的分项变化只用于定位，不当严格同进程收益。

## 数值和资格

最终同源 B1 最大参数绝对差 `3.6672e-8`、chi 差 `4.4409e-16`、covariance 差 `8.2793e-12`；B40 全部 cell 最大参数差 `3.4131e-8`、chi 差 `9.6723e-13`。两组 success/covariance_valid 均与旧结果相同。不是 bit-exact：原单图是另一 refinement 算法，当前统一 TRF；并行中心化矩归并也改变浮点求和次序。

科学小对照 [157252](major_fit_science_check_157252.jsonl) 已完成：两种 regular model、18×24 的 masked+NaN、1e10 背景、无噪声、负峰与四种 robust loss 异常点图，共16场景、48对 old/current 结果；新 single 与 B2 对应 lane 参数全部精确相同，success/covariance_valid/selected indices 保持一致。非高背景场景最大预测差 `7.8742e-8`；无噪声 RSS 约 `1.8e-24`（radial）/`1.78e-19`（anisotropic），绝对变化分别不超过 `2.03e-25`/`2.31e-23`，不以接近零时的相对误差夸大。

高背景案例是背景 `1e10`、峰幅 `4`、噪声标准差 `0.03`，必须单列而不能藏进通用等价结论。旧 single→新 unified 的直接 RSS：radial `0.4565937→0.4274293`，anisotropic `0.4575829→0.4268056`；较大参数差对应拟合质量实际改善。相对旧 batch，新 batch 的绝对 RSS 差最多 `8.4714e-7`，像素预测差最多 `2.0981e-5`（约峰幅的 `5.25e-6`）。reported chi×DOF 与新结果直接残差一致；这仍不是全输入全局收敛保证。

直接旧用例：`research/major_fit_narrow_tests.py`，PID166544，**27 passed in 9.72s**。范围包括 regular mask/crop/lazy/bounds/warm 与 scalar/batch，Gaussian sigma/robust/fixed/rank/cancel、公共 camera source 不变及 float64 输出。已删除核的旧 `_promote_unsigned_summary` spy 用例改为现有公共 FitEngine 行为，没有恢复死代码或新增测试函数。随后只运行原 owner 的 tiny u8/u16 B1/B4 warm，全部成功，新 fused kernel 实见 f64/u8/u16 三个签名；没有运行全 Plot warmer。主任务随后修改的公共 FitSelection 坐标复用不在这27项范围，不冒充已覆盖。

精确 capture 仅加载我们自己创建的 pickle，保留 model_id、逐 cell 输入、原 warm/options 和 args；恢复 readonly/stride/直接数组对象共享关系，不重建原 shared-memory 地址或 OS 页状态。`cancelled` 在 capture 中明确去除：实际 A 回调是两个未 set 的 Event.is_set 读取 OR；不把独立重放称为严格执行完全相同。详见 [精确重放的边界](replay_exact_fit_result.md)。

## 被否决或不保留的方向

最早只把 B1 改成通用 full TRF 时约 128 ms，对比旧 20 ms，明确不合格。实际 nfev 表明旧 full objective 7 次、TRF full 5 nfev/4 iterations；根因不是迭代变多，不应改收敛容差或偷偷恢复 B1 旧路线。单独投影 BLAS、compact grid、RSS scratch、context 等阶段的中间数值留在 `major_fit_*` 原始日志，不能当最终成果。

BLAS 资源有限对照 [96476](major_fit_blas_matrix_96476.jsonl)：同进程两个 provider 均临时限制 1/4/16 threads，B1/B40 old/current 每格交错 n3，最后恢复两库原来的 16。B1 current P50 为 60.19/48.83/43.44 ms，B40 为 33.98/34.13/34.89 ms。**4 没有同时改善单图和批量，不据此改生产线程预算。** 该实验先于 empty-weights/fused-prepare；当时旧 radial 通过研究 adapter 使用当前 compiled，故是当时实现的有限资源敏感性，不冒充最终完整旧基线。当前最终 A/B 已改为成对加载 saved 数值模块，避免权重合同混入。

过程错误也保留资格：PID165524 在首次编译新增 parallel final kernel 时，旧 tuple→column slice 赋值触发 Numba parfor shape-analysis AssertionError；改成相同三标量写入后再跑成功，不是数值失败。研究脚本 PID162204 曾漏传 `fit_batch` 必需的 observations 参数，尚未取得 batch 科学结果就退出；修脚本后以157252整轮记录为有效证据。

## 代码、缓存和内存成本

只计算本 Fit 范围，相对 `ec706be2` 的最终 `git diff --numstat`：

| 现有文件 | 新增行 | 删除行 | 净变化 |
|---|---:|---:|---:|
| `_fit_compiled.py` | 78 | 30 | +48 |
| `_fit_radial.py` | 343 | 641 | −298 |
| `fit.py` | 54 | 29 | +25 |
| production 合计 | 475 | 700 | **−225** |
| 原 `test_kernel_warm.py` 直接用例替换 | 18 | 29 | −11 |
| 全部本范围 | 493 | 729 | **−236** |

删除项包括单图 Newton/LBFGS refinement、原平行 summary/unsigned-summary、失去消费者的 information/objective/helper/constants。新增项是现有 descriptor 的 compact-grid 布局、已有 context 贯通、shared preparation/final information/批量 covariance；公共 FitResult 数据字段未删。这里不把其它 agent 的 selection/render 改动算进 Fit 净减。

| 缓存项 | 已观察的成本/证据 | 限定 |
|---|---|---|
| 修改后第一次 B1 current | 11.529 s | 包含多个失效 machine-code cache 的重建和 fit，非纯编译、非 steady |
| 随后已有 cache 新进程 B40 current | 251.1 ms | 同输入完整 first fit；old 对照407.6 ms；只证明这个真实签名低于1秒 |
| fused preparation 根 | f64/u8/u16 三个明确签名 | 来自27项QA后的实际 signature 列表，u8/u16由现有 tiny warm owner 真实 B1/B4触发 |
| 当前 fused 根落盘机器码+索引 | 288,812 bytes | 当前 `-250.py313.*` 三个 nbc 加一个 nbi |
| 当前 final-information 根落盘机器码+索引 | 361,343 bytes | 当前 `-493.py313.*` 一个 nbc 加一个 nbi |
| 这两个新增根合计 | 650,155 bytes（约0.620 MiB） | **不是所有 Fit ABI 变更后的总缓存体积**；没有删除/计入目录里历史研究版本的残留 |

现有 warm owner 注册两个新根，没有新 warmer 模块。point objective 机械新增 context 形参以及 RegularImage callback 改动会使相应源缓存重新编译；不声称完整新环境无需 warm，也不把已缓存进程时间混为冷编译时间。float64是最高浮点工作精度，没有 float128 kernel/扩展精度计算；不为本轮新增扩展 dtype 验证矩阵。

工作数组的**源码尺寸推算**（非本次 peak-RSS 实测）：单图1200×1920每张 float64 完整工作图为18,432,000 bytes（17.578 MiB）。新流程仍拥有一张真实 physical f64 values 和一张 centered context，两者不能假称零复制；原始 u8 stack另约2.197 MiB。未用的 generic权重原本也是17.578 MiB，现为零长度；原 single-specialization 没有该 generic权重，不能把这项省量算成它对原single的独立收益。full轴包只有30,736 bytes，替代 generic逐像素 x/y 的重复展开；最终直接RSS每个active worker最多64×1920×8 bytes（0.938 MiB）scratch，4线程上界约3.75 MiB。批量按相同公式代入B/shape；这些都是本次调用的局部工作数组，不是跨revision cache。

源码和27项直接QA已冻结；未提交。最终真实四 Panel 的墙钟/CPU由主任务另行报告，不以 isolated Fit 全占比最大项等同硬件极限；root在并行最终RSS之前取得的四图收益也不能冒充最后版本已经重复实测。
