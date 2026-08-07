# 任务3 验收报告:测试质量 / 守卫 / 收口状态

**结论:小修**(测试实质过硬、探针 5/6 击杀、守卫实证非空洞、notebook/demo 实跑全绿;但三口契约的 follow 口在 pytest 层零覆盖且探针存活,P1.4 压测未兑现,GOAL 勾选簿记漂移)

审查基线:`zlc_runtime@bf03a9d`,工作树干净(前置简报所述的四个未提交文件已在 `bf03a9d "finalize facade and acceptance fixtures"` 提交,见 §4)。全部实验在临时副本 `scratchpad/accept/zlc_runtime_copy` + 独立 venv(zlc-data 0.1.0 + 本包 editable)完成,基线 110 测试全绿;被验收仓全程未动(`git status` 干净复核)。

---

## 1. 11 个测试文件横审 + mutation 探针

### 断言实质(非形状剧场)

| 文件 | 判定 | 代表性证据 |
|---|---|---|
| test_runtime_streams.py (713行) | 实质 | 数值水位 `assert exact_sequences == list(range(total))` (:196)、`assert latest.missed == total-1` (:202)、typed gap `caught.value.expected==0 / earliest_retained==5` (:147-148)、双线程 Barrier 竞态 (:358-398)、weakref 生命周期 (:681-697)、monkeypatch 注入合成故障断言 SourceFailed 终局 (:611-632) |
| test_signal_plane.py (604行) | 实质 | front 身份复用 `plane.freeze() is first` (:329)、gated processor 真线程 + 撤销/取消/兄弟不全全路径 (:193-309)、负面清单机械断言 (:357) |
| test_signal_front.py (238行) | 实质 | 裁决①传递性:三级链回退 `[1,1,1]`→恢复 `[2,2,2]`+根集唯一 (:115-160)、weak parent 存活 + `behind==4` (:179-238) |
| test_host.py (452行) | 实质 | finite 成功/异常/声明未发布/取消/shutdown 拒关/陈旧 mailbox 六路,断言 phase 与错误文本 (:145-316);reactive 失败/取消 (:403-452) |
| test_presentation.py (314行) | 实质 | 欠拍补拍 GOAL P4.2 原景:2000ms 组流拍→下一 base tick 即补 (:243-267);all-or-nothing 整批弃 (:219-230) |
| test_runtime_dataset_builder.py (1002行) | 实质 | `missed_events==1` 数值记账 (:604,:657)、revision/axis/dtype 保真 (:339,:672) |
| test_runtime_resource_arbiter.py | 实质 | 并发独占单赢家 (:65)、精确一次释放 (:125) |
| test_runtime_helpers.py (142行) | 实质 | cancellation/cleanup/preview/mailbox/LiveDatasetPort 各真行为断言 |
| test_signal_source.py (179行) | 实质 | EventRef 父引用/捕获时间逐字段 (:50-74)、schema 越权 TypeError (:118) |
| test_import_guards.py (181行) | 实质 | 见 §2 |
| test_acceptance_fixtures.py (41行) | **半剧场** | notebook 测试只做 marker 字符串扫描不执行 (:14-30);demo 测试是真子进程跑 (:33-41) |

### 三口契约测试在位情况

- **exact ack 水位 + StreamGap 响亮:在**。:100-119(逐 ack 顺序)、:140-148(typed gap 而非 latest 兜底)。
- **monitor latest 跳版 missed 记账:在**。:151-176(missed==3)、:178-207(next 无丢 + latest missed==99)。
- **follow 中途加入:pytest 层缺失**。全 tests/ 目录 `FollowTap|\.follow\(` 仅命中 import 守卫的门面身份检查(test_import_guards.py:88,98),唯一行为覆盖在 notebook cell 4。

### Mutation 探针(临时副本,逐个应用→跑全套→还原)

| 探针 | 变异内容 | 结果 |
|---|---|---|
| A | MonitorTap.latest() missed 恒报 0(streams.py:1077-1083) | **击杀**:2 测试红,数值断言 `assert 0 == 99` |
| B | exact cursor gap 改静默跳 earliest_retained(streams.py:936-941) | **击杀(但以挂起形式)**:test_unreserved_cursor_gets_typed_gap… 永久阻塞,套件永不绿。nit:该测试 `cursor.next()` 无 timeout(test_runtime_streams.py:146),此类回归会把 CI 挂死而非干净红 |
| C | FollowTap.next() 整体 gut | 击杀:经 test_signal_source 三测试间接红(投影层内部用 follow) |
| D | **follow() 中途加入改为从 0 回放**(streams.py:1448 `start_sequence=0`) | **存活:110 测试全绿** — follow 中途加入契约零 pytest 覆盖实锤 |
| E | 裁决①族回退删除(front.py:207-222 改 continue,撕裂 front) | **击杀**:test_signal_front + test_signal_plane 各 1 红 |
| F | 欠拍 owed 重试删除(presentation.py:623) | **击杀**:2 个 owed-beat 测试红 |

5/6 击杀,与 zlc_atom"mutation 全存活"完全不同档;唯一存活的 D 正是三口契约缺的那一口。

## 2. 守卫非空洞审计

- **递归性**:模块守卫用 `pkgutil.walk_packages` 递归(:39-46,实测发现 21 模块),文本扫描用 `SRC.rglob("*.py")` 递归(:150);两者各带非空自证 `assert modules` (:111)、`assert source_files` (:151)。
- **植入实验**(pulse 教训直接复验):在 `src/zlc_runtime/sub/inner.py` 植入 `import zlc_storage` — 无 `__init__.py`(namespace 子目录)时文本扫描守卫红;补 `__init__.py` 后模块 import 守卫也红。**非空洞**。残余暗角:namespace 子目录只被文本扫描兜住(真实 import 守卫走不到),可接受但值得知晓。
- **allow-list 对齐**:守卫直接解析 `docs/contract.md` "## 顶层 facade" 行当单源(:29-36),断言 ①`__all__` 精确相等+≤15 (:49-53) ②fresh 子进程 public vars 相等(抓模块对象泄漏,配合 `__init__.py:31-33` 的 namespace 清扫)③15 名逐一 `is` 具体实现类(:78-106)。contract.md:9 的 15 名与 `__init__.py:10-26` 完全一致(工作区==HEAD)。
- **不起线程守卫:在**(:161-181,子进程 threading.enumerate 前后比对)。
- 真实 import 守卫在子进程逐模块加载并追踪 top-level import 白名单 stdlib+numpy+zlc_data+zlc_runtime(:109-146)——比纯文本 grep 强一档。

## 3. 实跑

**notebooks/usage.ipynb**(nbclient 逐 cell,临时 venv):
- cell 0 markdown;cell 1 OK(fixture 定义);cell 2 OK `initial revisions: {camera:1, roi:1, fit:1}`
- **cell 3 OK — 裁决①现场演示真实可见且带断言**:慢 ROI 卡在 revision 2 时 `slow ROI fallback keeps one family shot: {1,1,1}`(assert 锁死)→ 放行后 `recovered family shot: {2,2,2}` → `next family shot: {3,3,3}`
- cell 4 OK — 三口逐个:`exact ack watermark: 3 [1,2,3]`、`monitor latest/missed: 3 2`、`follow joined at sequence: 2 3`(中途加入不回放,assert payload==3)
- cell 5 OK — HarmonicClock+BoardScheduler+假 SurfacePort `scheduler accepted batches: ['present 3']`

**examples/demo_signal_flow.py --once**:退出码 0,输出含 `frame=1 root_seq=1 roi_parent=1`。

注:pytest 内 notebook 只被 marker 扫描不被执行(nbclient 非依赖,可理解),即 notebook 回归靠人工;demo 有真子进程测试兜着。

## 4. 脏工作树裁决(已被 bf03a9d 解决)

四文件在审查开始前已提交,工作树干净。裁决:**完整主题,提交正确**。该 commit 把 `_public.py` 从 15 个占位空壳类(`class SignalDataPlane: __slots__=()` ——上一个形状剧场隐患:门面名≠实现类)收口为仅 `RunHandleLike` Protocol;`__init__.py` 改为 re-export 具体实现 + namespace 清扫;并新增 facade 身份测试(test_import_guards.py:78-106)机械锁死"门面名 is 实现类"。与 P6.1 的 15 名 allow-list 目标:**零距离,已达成**。

## 5. 按 GOAL 应有而缺(小修清单,按优先级)

1. **follow 中途加入 pytest**(mutation D 存活的直接补口):中途 join 后断言不回放、逐序无损、gap 响亮 — 三口契约第三口。
2. **P1.4 "三口各一组并发压测"未兑现但已勾选**:现有线程测试全是点测(release/emit 竞态 :358、monitor close 唤醒 :217、arbiter 单赢家);无 producer/consumer 双线程高吞吐压测,follow 连点测都无。勾选与实况不符,须补测或在 GOAL 明记裁减。
3. **GOAL 勾选簿记漂移**:P5.1-P5.3、P6.1-P6.3 已实现已提交(264dd86/bf03a9d)却全部未勾;GOAL 未改 COMPLETE。按其自身仪式("完成勾选 commit")收口。
4. reactive 跳版无直接测试:test_host.py:350 场景是顺序 1→2,无"堆积多版只算最新"断言(P5.3 点名"跳版")。
5. LiveDatasetPort 的 `fail()`/`source_terminal()` 分支未测(test_runtime_helpers.py:98 只覆盖 bind/updated/freeze/close);contract.md:88 列了全表面。
6. exact ack 乱序 `ValueError("strictly ordered")`(streams.py:1951)与 ack 已裁剪记录的 StreamGap(streams.py:1954)无直接 pytest。
7. 小加固:test_runtime_streams.py:146 的 `cursor.next()` 加 timeout,使 gap 类回归干净红而非挂死 CI(探针 B 实证)。