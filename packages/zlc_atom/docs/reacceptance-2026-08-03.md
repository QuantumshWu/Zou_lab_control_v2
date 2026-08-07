# zlc_atom v2 返工复验报告(2026-08-03)

**结论:未通过(1 项)——机械终态判据 3 的第三处 mutation(occupancy rate 反相)实测 54 测试全绿存活,该项返工没做实。其余 W0–W7 全部实锤落地,修复量很小(见末尾清单)。**

环境说明:oracle 双文件 sha256 与树内逐字节相同(json `f2e6…f194`、npz `ec01…86aa`,独立复算);全部实验用独立驱动脚本 + 系统临时目录副本(`PYTHONPATH` 指向副本 src,已验证 `zlc_atom.__file__` 落在副本);三只读仓 git status 复验前后一致未被改动(树内 `pulses/scan_test.json`、runtime 三份 acceptance 文档均为开工前已存在的 untracked)。

## 1. 物理等价终审(实跑,树内冻结 npz 驱动,逐项最大偏差)

上一轮四个漂移项**全部归零**:

| 项目(v1 漂移) | 本轮实测最大偏差 |
|---|---|
| fit_bimodal threshold(v1: 12.66) | **0.0**(threshold/fidelity/dark_mean/dark_sigma/bright_mean/bright_sigma 全 6 站 0.0;bright_above/ok 0 mismatch) |
| find_site_centers(v1: 0.455px) | **0.0**(四种 ordering 全 0.0) |
| psf(v1: kernel 0.10 / 信号 0.51) | **0.0**(数据驱动 kernel、psf_boxes、fit_centers/sigma/ok、padding=3 窗口信号、uniform 姊妹全 0.0) |
| 端到端错误率(v1: 122/360=4.2×) | **29/360,与树内相同**;pred_box 360/360 逐元素相等 |

其余逐项:box 四 reducer 0.0;thresholds_box/psf/uniform 0.0;labels_occupied/valid、split_train/test 0 mismatch;site_fidelity(held-out)/dark/bright 0.0;`per_site_fidelity` 公开 API 0.0;`classify_threshold` 0 mismatch;唯一非零 = site_model_fidelity **1.11e-16**(1 ULP,远优于 1e-12 判据)。
备注(非缺陷):oracle 的 `quick_thresholds_*`(树内 runtime NaN 兜底阈值)与 ablation/global/runtime_* 族未被 atom 映射消费——不在 W1.2 点名范围内,记录备查。

## 2. 三处点名 mutation(临时副本,逐个全套 54 测试)

| Mutation | 位置 | 结果 |
|---|---|---|
| M1 threshold +17.3 | `_empirical_threshold` 返回值(calibration.py:732) | **红**(test_main_readout_oracle 端到端 + test_real_runtime_integration 两处 FAILED)✅ |
| M2 classify 极性翻转 | `classify_threshold`(calibration.py:139-141)`>`↔`<` | **红**(test_main_oracle_classification + mutation guard 两处 FAILED)✅ |
| M3 occupancy rate 反相 | `nodes/occupancy/processor.py:138` `rate = 1.0 - np.mean(...)` | **🔴 54 全绿存活**(已确认副本内被 import 的模块就是反相版)❌ |

**M3 存活双根因(均已实证)**:
1. 仓内"机械门" `tests/test_mutation_guards.py:50-53` **从不调用 OccupancyProcessor**——只在冻结 npz 数据自身上断言 `1-rate != rate`,对任何实现变异永久绿,属空洞守卫(恰是 v1 验收报告点名过的病型)。
2. 唯一实现级数值断言 `tests/test_installation_and_nodes.py:75`(`rate == mean(occupied)`)取在**简并点**:虚拟世界默认占据 `(fire_count+arange(6))%2==0`(simulation/world.py:87)每 shot 恰 3/6,mean≡0.5,`1-0.5==0.5` 不可辨(实测 60 shot rate 全为 0.5)——与 v1 "阈值相等点辨不出极性"同病。
3. 附带:终态判据 3 要求 mutation 实验"记录在 commit message",3296e3a 及全部 31 个 commit 的 body 均无记录。

M2 附带观察(不判失败):`calibrate()` 内部预测走独立内联判决(calibration.py:880),不经 `classify_threshold`——极性双实现,M2 只被两处直连测试抓住,occupancy 链(经 `detect`→classify_threshold)的翻转在集成测试全程无感;单源化可与修复项一并考虑。

## 3. W3 契约对齐

- `tests/fakes.py::FakePlane` = 真 `zlc_runtime.SignalDataPlane` 子类打桩,`reserve/retire/mark_changed/publish_final/publish_processor(keyword-only source_publication)/cancel_latest_only_processor` 签名与契约一致且被测试真实消费;src/ 内 `class SignalDataPlane` grep 零命中,`signal_plane.py` 不存在(test_contract_fakes.py:24 有守卫)。
- **attach_latest_only_processor 版本**:atom fake(fakes.py:55-70)贴的是 **R1 前冻结拼写** `(signal_key, control, initial_publication)`(zlc_runtime@1097117 版契约原文),内部 shim 转调实现拼写;并在 test_contract_fakes.py:18-23 把旧拼写钉死。**R1 已于 runtime@201f624 落地**(契约改为 `(node, *, source_name, initial_publication)`,另新增 `set_front_signals`/`withdraw_processor`)。**跟改量:仅 2 处**——删除 fakes.py:55-70 的 override(基类已是契约真拼写,或改签名保留 instrumentation)+ 改 test_contract_fakes.py:18-23 的签名钉;src 三节点**零改动**(无任何 attach_latest_only_processor 调用方)。
- 三节点调用面:camera_measurement(reserve/attach/mark_changed/detach_live/publish_final/latest_publication,measurement.py:104-200)、occupancy(bind_continuous_derived/publish_continuous_derived,processor.py:172-188)逐参数与契约 60-72 行一致;calibration 为纯 task 不触 plane。基线在 R1+R2 已落地的 runtime 上全绿。

## 4. 抽查

- **dcam 真迁移 ✅**:`_dcam_driver.py`(511)与 `_owner_lane.py`(80)与树内**逐字节相同**;`dcam.py`(748 vs 750)全量 diff 仅两处 import 适配(zlc_storage→zlc_data.validation + 本地 `positive_real` shim + `nonnegative`→`minimum=0`;树内 `canonical_text` 本就是死 import,干净删除)。
- **虚拟物理 ✅**:σ0.7/bg300/rate1100/gain0.107/read0.43/offset200 全对齐(world.py:51-56)且有守卫(test_virtual_physics.py:12-31,信号=rate×曝光派生断言);Poisson 光子(world.py:106)+ 读出高斯 + native `<u2`(world.py:113、virtual.py:66);产帧 worker 真线程(virtual.py:113-140);`fire` 按已 load 程序 camera 通道上升沿数产窗(sequencer/virtual.py:77-130,有测试);`SimulationWorld()` 工厂兜底与 `_default_frame` grep 双零。
- **W6 安全 ✅**:`IdentityProof` 模块私有 token 铸造(ports.py:25-44,徒手构造实测 PermissionError)+ bind 校验 `_broker is self`;`broker._active` 真用(run 租约 acquire/release/shutdown 拒绝,ports.py:190-232,test_execution_safety 三测);`CAPABILITY_TYPES` 单源 execution/capabilities.py,ports 与 install/descriptors 同引且有 `is` 断言(test_installation_guards.py:62);`SafetyInterruptError`/`_seal_execution_on_cancel`/`_revoke_hardware` 真用(run.py:150/352/358)。
- **守卫 ✅**:rglob 非空自证(test_import_boundaries.py:16);A 半区禁 runtime/pulse + B 半区白名单(:45-53);`if virtual` grep(:56-60);合成叶子注入自动收编(test_installation_guards.py:20-58);trivial device 经 bind_trivial_device;capability isinstance 双向(负例+virtual 实装);故障隔离 open(test_monitor_and_installation.py:42)。
- **notebook ✅**:7 cell 带执行输出提交;monitor `<u2` 帧、60 帧标定 30/30 split、oracle 复算 cell 输出 **29**;实跑复验虚拟标定 `reference_fit_ok` 全 True、阈值非兜底、held-out fidelity=1.0。
- 卫生小事(不计分):`src/zlc_atom/nodes/_framework/__pycache__/` 残留 W3.1 前的 `signal_plane.pyc`/`host.pyc`(gitignored,Python 不会无源导入,建议顺手删)。

## 修复清单(修完即可 PASS)

1. **必修**:给 occupancy rate 反相补一个真能杀死它的测试——两路任选其一或都做:(a) 用 oracle `runtime_probe_indices` 帧驱动 `OccupancyProcessor.process` 并对 `runtime_rate_box` 断数值;(b) 集成测试里 `world.set_occupancy` 设非对称占据(如 1/6),使 `rate==mean(occupied)` 离开 0.5 简并点。同时删除或重写空转的 `test_mutation_guards.py:50-53`(现版对实现零设防)。修后在临时副本重放 M3 确认必红,并按终态判据 3 把三处 mutation 实验记录进 commit message。
2. **小修(R1 跟改)**:fakes.py:55-70 override 与 test_contract_fakes.py:18-23 对齐 R1 后契约拼写 `(node, *, source_name, initial_publication)`(src 零改动)。
3. **建议(非阻塞)**:`calibrate()` L880 内联判决收敛到 `classify_threshold` 单源,消除极性双实现。