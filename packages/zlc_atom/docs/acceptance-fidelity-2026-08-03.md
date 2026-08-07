# zlc_atom 验收报告 — 任务1:迁移保真度与物理等价

**结论:返工**(范围见文末;"已结束"陈述不成立:`GOAL.md:3` 状态仍为 IN PROGRESS,A0–B5 全部复选框未勾)

---

## 1. 物理等价实验(已实跑,树内冻结 oracle 60×2×34×40 帧直接喂给 zlc_atom)

方法:加载树内 `tests/fixtures/main_readout_oracle.npz`(main@6c337d49 冻结,10 个 authority 函数),将 `zlc_atom.physics` 对应函数应用于冻结输入,与冻结期望逐数组比对。

### 1a. 逐函数判定

| 函数 | 判定 | 最大偏差 | 证据 |
|---|---|---|---|
| `normal_cdf`(math.erf vs scipy.erf) | **等价** | 2.2e-16 | bimodal.py:36-45 |
| box 框生成 `_box_bounds` | **等价** | 逐元素相等 | calibration.py:150-156 |
| `extract_box_signals` 四 reducer(mean/sum/median/max) | **等价** | 0.0 | calibration.py:159-179,对 manifest `box_reducer_oracle` |
| reference_box_signals (60,2,6) / short_signals_box (60,6) | **等价** | 0.0 | 全帧复算逐位相等 |
| `extract_psf_window`(喂 oracle kernel/box,padding=3) | **等价** | 0.0(per-site 与 uniform 都是) | psf.py:42-61 |
| `extract_psf_signals`(公开入口) | **漂移 5.1e-1** | padding 被硬编码 2(psf.py:42 默认值,calibration.py:207 调用未传),树内权威 padding 可配且 oracle=3 | 同一 kernel 下纯粹是 padding 差 |
| `fit_bimodal` 组件统计(dark/bright mean、sigma、fidelity、ok、bright_above) | **等价** | 0.0 | 一侧分位统计+exact-otsu 是逐公式忠实移植(bimodal.py:168-254 vs 树内 bimodal.py:43-246) |
| `fit_bimodal` **threshold** | **漂移,最大 12.66(绝对)** | 树内用 `scipy.minimize_scalar(bounded)`(树 bimodal.py:125),zlc_atom 改写为 96 轮黄金分割(bimodal.py:124-143)。两峰强分离时 Bayes 误差在均值间是零平台,黄金分割收敛到**平台左缘(贴 dark 峰)**,scipy 落中部;实测 scipy 逐位复现 oracle(diff=0),zlc_atom 全 6 站漂移 6.4~12.7。冻结样本上 0/120 判定翻转,但树内 oracle 断言(rtol 1e-12)必挂,且阈值贴 dark 峰在真机重尾噪声下是鲁棒性回归 |
| `find_site_centers` | **漂移 0.455 px** | 树内=高斯平滑+maximum_filter+curve_fit 亚像素+格点修复(树 analysis.py:826-879);zlc_atom=裸 argmax 贪心(calibration.py:382-430),无亚像素。树内 admission 判据 `maximum_site_residual_px=0.1` 会**拒收**此中心;zlc_atom 自己把容差硬编码放宽到 2.0(calibration.py:453) |
| `classify_threshold`+oracle 阈值 → pred_box | **等价** | 逐位相等 | 分类算子本身正确 |
| **端到端 `calibrate()`** | **物理不等价** | 阈值漂移 1.73;**判定翻转 129/360(36%)** | 见 1b |
| `per_site_fidelity` | **漂移 0.167** | 指标变义:全 shot 上的 accuracy(bimodal.py:257-271),树内=held-out test 集平衡 confusion fidelity(树 analysis.py:1268-1294) |

### 1b. 决定性实验:同一冻结输入下的读出错误率

```
main 冻结管线 pred_box vs 潜在占据真值:  29/360 错
zlc_atom calibrate().detect  vs 同一真值:122/360 错  ← 4.2 倍
```

根因(calibration.py:467-475):zlc_atom 把阈值拟合在**长曝光 reference 信号**上(bright≈24,平台阈值≈12),再直接用于 **5ms short 帧**(bright 实际落在 10.5~13.5)——阈值 12.3 把约一半亮位判暗。树内管线的整个 short 帧表征段(共识标签 `reference_labels` → 种子化 `train_test_split` → 直方图经验阈值 `_empirical_threshold`(bins=120) → held-out confusion,树 analysis.py:1074-1437)**整段不存在**。另:calibration.py:469 `labels = mean(axis=0) > median(axis=0)` 产出的是 (6,) 每站一个布尔——不是每 shot 占据标签,物理上无意义(oracle labels 为 (60,6))。

### 1c. oracle 造假实锤

`zlc_atom/tests/fixtures/main_readout_oracle.json` 与树内同名文件**只有文件名相同**:内容 `format: "zlc-atom-readout-oracle"`、`authority: "frozen hand-authored physical quantities"`,仅含 3 点 normal_cdf + 3×3 手算 box 两例。GOAL A1 明确要求"冻结 oracle 随迁(main_readout_oracle 类 golden 文件)"。真 oracle 覆盖 10 个 authority 函数与完整合成帧;自产件覆盖 ~1.5 个,且**借同名伪装成已迁移**。tests/test_physics.py:12-15 对它断言,绿灯是自证。

---

## 2. 逐模块分类【迁移/重写/占位】

| 模块 | 判定 | 依据 |
|---|---|---|
| `physics/bimodal.py` (284) | **迁移**(仅 threshold 求解器被替换,见 1a) | 组件统计逐位等价 |
| `physics/psf.py` (64) | **半迁移** | 窗口应用忠实;kernel 只有理想高斯 σ=1.0,树内数据驱动 kernel 拟合(`_fit_psf_features`,树 analysis.py:936-1045)未迁,实测理想 kernel 与 oracle kernel 偏差 0.101(kernel 峰值≈0.16,即 60% 量级) |
| `physics/calibration.py` (514) | **重写** | `TrapCalibration` 门面(signals/detect/box\|psf 分派)符合 GOAL 措辞,但 `calibrate()` 管线物理错误(1b);无 uniform-PSF 模型、quick thresholds、global threshold、ablation、split |
| `devices/camera/dcam.py` (98) | **占位** | 树内真栈=dcam.py 750 行 + `_dcam_driver.py` 511 行(132 处 ctypes,真 DCAM-API)+ `_owner_lane.py` 80 行 SDK 属主线程。zlc_atom 版对注入 `driver` 做 hasattr 鸭子转发(dcam.py:38,61,70),无属性编程、无 ROI/binning/trigger 配置、无属主线程,`capture_state` 丢帧数硬编码 0(dcam.py:88-89)。**不能驱动真机 qCMOS** |
| `devices/camera/virtual.py` (200) | **重写** | 保留:有界队列真丢帧(virtual.py:156-158)、trigger→帧因果。丢失:产帧 worker 线程(树 apparatus.py:1529 等)、Poisson 光子统计(树 apparatus.py:355 `rng.poisson`,zlc_atom 只有背景高斯近似)、播放时序→触发窗因果。工作点 dtype `"<f8"`(virtual.py:68)违背"相机块全程 native 整型"铁律 |
| `simulation/world.py` (121) | **重写,参数错** | 树内标定(apparatus.py:232-239):atom_rate=1100、bg=300、σ=0.7px;zlc_atom:σ=1.2(world.py:18)、bg=350(world.py:54)、原子信号=固定幅度 100 counts(world.py:93,非 atom_rate×曝光 派生)。GOAL A4 的 `SimulationWorld` 显式注入✅、A2"标定参数派生关系保留"❌(`test_calibration_report_and_noise` 对应守卫不存在)。`VirtualPulseStreamer.fire` 恒触发 1 帧,不读已 load 的程序(sequencer/virtual.py:76-88)——树内"基周期相机窗口数"判据整套缺失 |
| `execution/` (614 vs 树 1,931) | **重写** | 保留:回调注入 broker、verify_identity→bind→verify_capability 仪式 helper 化(ports.py:159-200,GOAL A3 要求✅,含 trivial 快路径,但 GOAL 要求的合成 device 测试没写)、arbiter/lease。丢失:fail-closed 取消(树 run.py `_fail_closed`/`_seal_execution_on_cancel`/`_revoke_hardware`/PostSafetyContext 全套)、run 域设备租约与吊销(树 ports.py `_DeviceRunLease`/revoke——zlc_atom 里任何持 binding 者任意时刻可 execute)、`SessionCloseCommand/Ack` 类型化关闭、身份防伪(`IdentityProof` 是普通 frozen dataclass 可徒手伪造,树内 Verified\* 只能由 broker 铸造)。死代码:`SafetyInterruptError` 定义未用(run.py:39);空洞守卫:`broker._active` 永不写入,shutdown 检查恒真空(ports.py:71,152-156),违反"守卫自证非空洞" |
| `install/` (281) | **重写,基本达标** | Kahn 排序+capability 类型校验+反向 close+建装失败回滚(graph.py:117-131)符合 GOAL B3;瑕疵:`simulation.world` token 声明无提供者;broker 侧另有一个空的 `CAPABILITY_TYPES`(ports.py:65)与 install/descriptors.py:16 的类型表双源 |
| `nodes/_framework/` (451) | **重写 + 契约变体发明(违 B0)** | `SignalDataPlane` 签名偏离 `zlc_runtime/docs/contract.md`(FROZEN):契约 `publish_final(...)->Mapping[str,SignalValue]`,此处返回 `SignalPublication`(signal_plane.py:81);契约 `publish_processor(control, outputs, *, source_publication)`,此处 `(producer, publication, outputs)`(signal_plane.py:84);`reserve->StreamGenerationId`、`bind_continuous_derived` 族、`attach_latest_only_processor` 全缺。且它放在 **src/** 被三个节点直接 import——GOAL B0 要求 fake 按契约写、放 tests/ 共享 fixture、"绝不自己发明变体"。B5 真集成时所有节点调用点都要返工 |
| `nodes/` 三叶子 (~350) | **重写(骨架级)** | camera_measurement:same-shot 一发布✅、repeat==0 monitor✅(measurement.py:70-73);occupancy:可用最小样板;calibration task:薄包装,**继承 1b 的物理错误**。descriptor/discovery/合成叶子守卫(test_installation_and_nodes.py:37-42)符合 B1 形态 |
| `notebooks/usage.ipynb` | 存在但**未执行**(全 cell 无 outputs);代码实跑可通,但标定用 2+2 帧——fit 必不 ok,阈值全部落入 `nanmedian(short)` 兜底(calibration.py:474),台架是仪式不是物理验证 |

15 项测试全绿,但对物理只断言 3×3 手算例;GOAL 终态判据 1/2/3 大体满足,判据 4(notebook 执行留痕、LOC 3,665 vs 预估 15-20k 的逐项说明)未满足。

---

## 3. 重写偏离清单(按严重度)

1. **calibrate() 阈值来源错误**:reference 信号拟阈值直接用于 short 帧,同输入错误率 4.2×(29→122/360)。丢失语义:共识标签、train/test split、直方图经验阈值、held-out fidelity。calibration.py:467-475。
2. **oracle 自产顶替**:GOAL A1 的 main 冻结 oracle 未随迁,同名文件伪装,物理等价从此无守卫。
3. **dcam 占位**:真 DCAM-API ctypes 栈(511+750+80 行)零迁移;而 ROADMAP 当前焦点恰是真机 qCMOS 接线。
4. **SignalDataPlane 契约变体在 src/**:违 B0"绝不发明变体、fake 进 tests/";集成 zlc_runtime 时三节点+测试全部改口。
5. **执行引擎安全语义丢失**:无 fail-closed 取消、无设备租约/吊销、身份凭证可伪造;broker shutdown 守卫空洞。
6. **虚拟物理失真**:标定参数偏离 qCMOS 冻结值(σ 0.7→1.2,bg 300→350,atom_rate 语义删除)、无 Poisson、fire 不看已 load 程序、帧 dtype float64——虚拟作为"真机同路径测试载具"的资格不成立。
7. **threshold 求解器漂移**(黄金分割落零误差平台左缘,最大 12.66):冻结数据无害,真机重尾下判暗裕度不对称;树内 oracle 断言必挂。
8. **find_site_centers 无亚像素/格点修复**(0.455px;树内 0.1px admission 会拒),自行放宽容差到 2.0px。
9. **PSF 数据驱动 kernel 未迁**(理想高斯偏差 0.10)+ padding 硬编码 2(oracle=3,信号漂移 0.51)。
10. **fidelity 指标变义**:held-out 平衡 fidelity → 全样本 accuracy(0.167 漂移)——正是 MEMORY 标记的"数据驱动逐站点保真度"旗舰点。

## 返工范围建议

- **必须**:①迁真冻结 oracle 并按 zlc_atom API 建映射断言;②calibrate() 补 short 帧表征段(标签→split→经验阈值→held-out),或直接迁树内 `characterize_readout` 链;③signal_plane 变体降级为 tests/ 契约 fake,节点面向契约签名;④dcam 真适配迁移(或 GOAL 明记后置并去掉 A2 勾稽预期);⑤虚拟物理参数对齐 qCMOS 冻结值+Poisson+程序驱动触发。
- **应修**:threshold 求解器换回 bounded 最小化(或黄金分割上叠 plateau 中点规约)、find_site_centers 亚像素+修复、psf padding 贯通、执行引擎 fail-closed 最小集、空洞守卫/死代码清理。
- **可保留**(实证等价/达标):bimodal 组件统计、box/psf 窗口提取核、classify、install graph、节点叶子模式与发现守卫、虚拟相机丢帧队列。