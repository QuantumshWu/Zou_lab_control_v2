# 任务3 验收报告:测试与 fake 质量(zlc_atom)

**基线实证**:临时副本 `pytest -q` = 15 通过(3+3+5+4)。所有 mutation 实验在系统临时目录副本上进行,原仓未动。

---

## 1. 15 个测试逐个审

### tests/test_physics.py(4 个)
| 测试 | 判定 | 证据 |
|---|---|---|
| `test_frozen_readout_oracle_is_not_regenerated...`:12-15 | **走过场(名不副实)** | 只对 `normal_cdf` 三个教科书点(Φ(-1)/Φ(0)/Φ(1))断言;tolerance `rtol=0, atol=2e-15`(:15,够紧但内容平凡)。fixture 的 `box_signal` 半区**没有任何测试读它**(死数据,box 期望值在 :34-35 另行硬编码 5.0/45.0)。测试名声称"非再生自参考树",实际它根本不是参考树 oracle(见 §2) |
| `test_trap_calibration_single_dispatch_supports_box`:18-22 | 半真 | signals=5.0 真断言;但 `detect==[False]` 取在 **5.0 vs 阈值 5.0 的边界相等点**——`>` 和 `<` 都得 False,极性翻转杀不死(mutation C1 实证存活) |
| `test_psf_dispatch_is_explicit...`:25-28 | 真但平凡 | 均匀核+对称 arange,期望仍是 5.0;annulus 背景、真实 PSF 加权零覆盖 |
| `test_box_reducer_oracle_is_shared...`:31-35 | 真 | mean/sum 微契约成立;median/max 两个 reducer 未测(树内 oracle 四个全测) |

### tests/test_camera_and_execution.py(3 个)
| 测试 | 判定 | 证据 |
|---|---|---|
| `test_virtual_camera_preserves_trigger_to_frame_causality...`:18-27 | 真(窄) | ordinal==2 + `capture_state()==(True,2)` 确实断言丢 2 帧。但 `VirtualCamera.trigger` 是**调用线程同步产帧**(virtual.py:141-160,无产帧线程),GOAL A2 要求的"产帧线程真丢帧语义"未实现也未测 |
| `test_camera_frame_record_copies_reusable_storage`:30-36 | 真 | 复制+只读微契约,contract.py:123-126 对应 |
| `test_broker_helper_is_the_single_identity_binding_ritual`:39-53 | 半真 | 正路径(bind→capability→execute→interrupt)真断言;零失败路径(身份不符/capability 缺失);"single ritual"只在名字里,无任何机械强制 |

### tests/test_import_boundaries.py(3 个)——见 §3 实证
### tests/test_installation_and_nodes.py(5 个)
| 测试 | 判定 | 证据 |
|---|---|---|
| `test_device_discovery_is_the_leaf_manifest`:16-24 | 真 | 清单==套件,4 型号精确元组 |
| `test_capability_tokens_have_machine_visible_types`:27-34 | 弱真 | 只查 dict 键集+值是 type;**没查 installation 实际产出的 capability 是这些类型的实例** |
| `test_logic_discovery_is_derived_from_three_leaf_modules`:37-41 | 真(不完整) | leaf_count 自 rglob 派生,非空洞;但 GOAL B1 的"**合成叶子**不碰骨架"守卫(加一个假叶子证明发现机制吃进去)不存在 |
| `test_virtual_installation_runs_measurement_occupancy_and_same_shot_front`:44-60 | **形状剧场** | lineage 断言(:56 `direct_parent_refs==(event_ref,)`)是全仓唯一像样的 same-shot 断言;但 counts/rate **只断言 shape**,阈值是硬编码 `np.full(6, 50.0)`(:52),数值全不设防——三个 mutation 全存活于此 |
| `test_virtual_installation_auto_calibration_path_matches_usage_notebook`:63-80 | **形状剧场** | CalibrationTask 全链跑完只断言 `counts.shape==(4,6)`、`rate.shape==(4,)`;标定出的阈值/保真度零断言 |

## 2. Oracle 自产判定:**证实,且比"自产"更糟——是新写的平凡数**

- 树内冻结版(参照树 `tests/fixtures/main_readout_oracle.json`):`"format":"main-readout-oracle"`、`authority_commit=6c337d49`、10 个 authority 函数、6 站点 box_reducer 四 reducer 数组,配套 **NPZ 含 ~100 个冻结数组**(detector 四序/PSF 拟合/阈值/保真度/ablation/runtime),消费测试 `test_zlc_readout_main_oracle.py` 760 行,tolerance rtol=1e-12/atol=2e-12。
- zlc_atom 版(`tests/fixtures/main_readout_oracle.json:2-3`):`"format":"zlc-atom-readout-oracle"`、`"authority":"frozen hand-authored physical quantities"`——**git 实证:与消费它的测试同一 commit 诞生(af68067 "test: add skeleton guards...")**,不是从树内迁移。内容是 3 个正态 CDF 教科书值 + 一个 3×3 对称 arange(mean=5/sum=45 手算即得)。严格说它不是"用被测实现再生"(值确实手写可查),但它**冒用了树内冻结文件的名字**,GOAL A1 "冻结 oracle 随迁" 一项实质未做:`fit_bimodal`/`optimal_gaussian_threshold`/`per_site_fidelity`/`confidence_weighted_fidelity`/`find_site_centers`/`calibrate` 全部零 oracle。

## 3. tests/fakes.py 对照两份契约

**首要事实:三个 fake 全仓零使用**(grep 实证:`FakePlane|FakeNodeHost|FakePulseStreamer` 只命中 fakes.py 自身)。GOAL B0 "fake 放 tests/ 共享 fixture" 是装饰性完成——节点测试直接用 src 里的 `SignalDataPlane`。

**FakePlane(fakes.py:15)= 空壳继承自己 src 的 `SignalDataPlane`**——这不是"按 zlc_runtime contract 写的 fake",而是把树内 plane 在 `src/zlc_atom/nodes/_framework/signal_plane.py` **影子重实现**后套壳。逐名对照 zlc_runtime/docs/contract.md L16-41:
- `reserve` 应返回 `StreamGenerationId`,实现返回 None(signal_plane.py:96-97);`retire` 应返回 `frozenset[str]`,返回 None(:99-101)
- `mark_changed(producer, live_slot)` 缺 `live_slot` 参数(:109)
- `publish_final`/`publish_processor` 应返回 `Mapping[str, SignalValue]`,实现返回 `SignalPublication`(:79-85)
- **`publish_processor` 参数序发明变体**:契约 `(control, outputs, *, source_publication)`(contract L32-37),实现 `(control, source_publication, outputs)`(:82)——source_publication 从 keyword-only 变第二位置参数
- `attach_latest_only_processor`/`cancel_latest_only_processor` 缺失(契约 L26-31);continuous derived 五方法族整族缺失(契约 L46-73)
- `SignalValue` 字段发明变体:契约 `(block/schema/values/behind)`(L77),实现 `(signal_key, value, publication, behind)`(:31-36);`SignalPublication` 多出契约没有的 `producer` 字段
- GOAL 明令"契约不够用记阻塞提议改文档,绝不自己发明变体";GOAL.md 阻塞记录为空。

**FakeNodeHost(fakes.py:19-43)**:表面四方法(start/cancel/poll/shutdown)对上契约 L98-99,但 `start()` 只翻布尔,**不执行 node**;"声明了输出却没发布=硬失败"、generation 重启防陈旧全无。且 src 另有一个影子 `NodeHost`(host.py),两者互不校验,均无测试。

**FakePulseStreamer(fakes.py:46)**:三者中最好。10 方法面(open/close/load/write_slots/write_scan_table/fire/wait_done/cursor/safe/snapshot)与 zlc_pulse contract L21-37 逐名一致,签名参数一致,`DoneReport(status,cursor,underflow,tail_elapsed)` 对上 L34。缺陷:契约 `__init__(transport, geom, clock_hz)` 中 **geom/clock_hz 缺失**,故 `open()` 的 CTRL[63] 指纹比对(契约 L23)不可能实现;自称"Register-dictionary streamer"但 `transport` 赋值后**任何方法都不读写寄存器**(virtual.py:30 起,纯装饰);`fire`→`world.fire(1)`(:74-76)是契约外后门(B2 允许接 SimulationWorld,可接受但契约未记)。未发明契约外公开方法。

## 4. test_import_boundaries.py 守卫实证

- **注入实验(临时副本)**:向 A 半区 `physics/bimodal.py` 加 `import zlc_runtime` → `test_a_half_has_no_parallel_package_imports` **变红**。守卫非空洞,这一半是真的。
- **空洞隐患实证**:目录名不存在时 rglob 扫 0 个文件、断言 vacuous 通过(实验证明 count==0 即绿)。无"扫到文件数>0"自证。
- **B 半区白名单不存在**:GOAL A0 要求"按子包分白名单",实现只有 A 半区禁令。且**全 src 无一处 import zlc_runtime/zlc_pulse**(grep 零命中)——B 半区不是"被允许 import",而是整个用影子实现替代了 import,守卫无从谈起。
- 终态判据 2 的 `if.*virtual` 运行时分支 grep 守卫缺失;`test_top_level_allow_list`(:12-16)是相邻两行字面量互抄的近重言式(弱守卫,可留)。

## 5. Mutation 抽查(临时副本,均已复原,复原后基线重新全绿)

| Mutation | 位置 | 结果 |
|---|---|---|
| B:`optimal_gaussian_threshold` 返回值 +17.3(阈值数学粗暴打坏) | bimodal.py:143 | **15/15 全绿存活** |
| C1:`classify_threshold` 极性翻转 `>` → `<`(占据判定反相) | calibration.py:137 | **15/15 全绿存活** |
| C2:`rate = 1.0 - mean(occupied)`(占据率反相) | occupancy/processor.py:76 | **15/15 全绿存活** |

结论:**标定阈值数学与占据判定的数值语义在本套件下零设防**。唯一沾边的断言(test_physics.py:22)恰取在阈值相等点,天然不辨极性。

## 6. 按 GOAL 应有而缺失的测试清单

1. **oracle npz 对照**(A1):树内 `main_readout_oracle.npz` 未迁,bimodal/psf/detector/calibrate 全链零冻结数值。
2. **虚拟同路径守卫**(判据 3 virtual==real):无"分析层不 import 具体后端"契约测试,无 `if virtual` 分支 grep 守卫。
3. **capability 契约**:半有——只查 token 表自身,不查 installation 产物是声明类型的实例。
4. **合成叶子守卫**(B1):缺——现测试只数既有 3 叶,不证明"新叶子=丢文件零骨架改动"。
5. **trivial device 快速路径**(A3):半有——broker 正路径有,失败路径与"10 行接入"经 installation graph 的证明缺。
6. **三节点行为**:repeat==0 live monitor 语义(B4 明文)零测试;same-shot 族回退零测试;CalibrationTask 在可分数据上的阈值/保真度数值零测试;`execution/run.py`(197 行 SafetyInterrupt)、`host.py` NodeHost、`authoring.py`(70 行)、psf annulus 背景:**零直接测试**。
7. **B5 集成**:完全没做——src 零 import zlc_runtime/zlc_pulse,"真包替换 fake"无从发生。**更正验收简报一处疑点:`notebooks/usage.ipynb` 存在且 5 个 cell 顶到底执行无错(临时副本实证)**,但它跑的是树内影子实现,不构成 B5。

## 结论:**返工**

返工范围(测试与 fake 层,不含 src 重写裁决——那属任务 1/2 归属):
1. **oracle 层重做**:迁移树内 npz+manifest(或明确降范围并改 GOAL),现 JSON 改名去掉"main_readout_oracle"冒名;把 fixture 死数据(box_signal)接进测试或删除。
2. **fakes.py 重写或删除**:要么按两份 contract.md 逐签名重写并真的被测试消费,要么承认影子实现路线并删掉这个装饰文件;`signal_plane.py` 的 `publish_processor` 参数序/`SignalValue` 字段等发明变体必须回归契约或在契约文档记阻塞。
3. **数值断言补齐**:三个存活 mutation 各补一个能杀死它的测试(可分双峰合成数据断言阈值区间与占据真值);test_physics.py:22 的边界相等点改为可辨极性的操作点。
4. **守卫补强**:A 半区扫描加非空自证;B 半区白名单;`if virtual` grep;合成叶子测试。
5. 15 个测试中真正合格的约 5 个(frame record 复制、box reducer、设备清单、叶子计数、边界注入守卫),可保留。