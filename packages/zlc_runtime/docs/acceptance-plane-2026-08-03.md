所有实验完成,变异已全部还原(实验只在 scratchpad 副本进行,被验收仓与参照树未动)。以下为验收报告。

---

# zlc_runtime 验收报告:任务1(信号面保真与裁决①实证)

**结论:小修**(语义保真完好、裁决①行为实证正确;但 1 处契约漂移是跨仓事故引信、裁决①机械守卫有一个已证实的联合空洞、5 处死代码)。

前置说明:被验收仓工作树**干净**,HEAD=bf03a9d——简报中"P6 四文件未提交"已在该 commit 落盘,故 HEAD 与工作区无差,以下证据只引一份。所有 mutation/探针均在 `%TEMP%\claude\...\scratchpad\mut` 副本 + 干净 venv(zlc-data editable)执行,基线 110 全绿复现。

## 1.1 plane 四分 vs 树内:逐块语义核对 — **通过**

对 `src/zlc_runtime/plane.py`(1808 行)与树内 `zlc_neutral_atom/processing/signal_plane.py`(2418 行)做了归一化全文 diff,**每一个 hunk 都可归因于 GOAL 明写任务,无静默语义改动**:

- **保留且逐行一致**:publication 原子捆绑(`SignalPublication.__post_init__` plane.py:219-238)、`_issuer` 哨兵(`_require_issued_publication_locked` plane.py:982-989)、发布校验(schema 稳定性+sibling bundle 恒定 `_validate_generation_values_locked` plane.py:1050-1087)、`last_parent_sequence` 拒过期父(`publish_processor` plane.py:1218/1244 双锁窗各查一次;`publish_continuous_derived` plane.py:1394-1413 同构)、derived 绑定族(1254-1471)、retirement 闭包(`_retirement_closure_locked` plane.py:1478-1509 ≡ 树内 1941-1972)、freeze 心脏(plane.py:1704-1773 ≡ 树内 2314-2383)。
- **preemption 五 API 确删**:`require_active_generation`/`bind_generation_source`/`release_generation_source`/`withdraw_dependency_closure`/`finish_dependency_retirement` 全部不存在;负面 grep(association 族/event-derived/Qt/zlc_storage)src 下全零,与 `test_signal_plane.py:357` 负面守卫一致。
- **event-derived 世代确删**:`bind_event_derived`/`publish_event_derived`/`signal_event_binding`/`has_event_association`/`bind_processor_event_source`、`_GenerationState` 的 `event_source/event_output_name/bound_parents` 字段、`_require_route_parents_locked` 的 `exact_bound`、`_derived_values` 的 event schema 校验——全部干净移除。
- **防御收缩按 GOAL 执行**:`SignalFront.__post_init__` 降为 `if __debug__` 断言(plane.py:267-293,GOAL P2.3);`publication_owner` 返回收窄为 `owner_token`(plane.py:382 新增字段、:922 返回,树内返回 `state.node`)= GOAL P2.5 明写。
- **结构性观察**:"四分"实际只有 front.py 是真抽取;`values.py`(re-export 6 名)、`processor_lane.py:3`(`from .plane import _LatestOnlyProcessorLane, _ProcessorEntry`)、`registry.py:3`(`from .plane import _GenerationState`)是 shim,plane.py 仍是单体。功能无损,但 GOAL P2.1 勾选文字("切分迁入…processor_lane.py(执行缝)/registry.py(生命周期+回收闭包)")与现实不符——模块化层面的名实不副,建议在 GOAL 补注或真拆。

## 1.2 front.py 纯函数化质量 — **通过**

- `build_front`(front.py:89-240)是真纯函数:输入 = states 七字段 duck 视图 + requested 集 + previous_front + `resolve_parents` 回调,不改任何输入;`test_signal_front.py:94-160` 用 `SimpleNamespace` 合成 states + 普通 dict 当 parent 表直测,证实可脱离锁/线程/活 plane 测试。
- 与树内 `_build_front_locked`(2157-2312)逐语义对照:根集单一性(front.py:179-184)/逐名祖先合并(186-199)/覆盖检查(200-201)/不齐回退上一 front 含 owner+generation 换代检查(207-222)/连续组构建(235-240)全部一致。仅两处无害确定化:seed 取 `min(pending)`(树内 `pending.pop()` 任意序)、leaves 用 `sorted`——组件划分与判定结果与序无关。`event_names` 差集随 event 族删除一并消失,正确。
- 回接缝:`_build_front_locked`(plane.py:1694-1702)锁内委派,`resolve_parents=self._resolved_direct_parents_locked` 持锁调用,无越锁。

## 1.3 裁决①mutation 实验(全部在临时副本;每步后还原并复验绿) — **行为正确;守卫有一个已证实空洞**

| 实验 | 改动 | 套件结果 | 判定 |
|---|---|---|---|
| ① 根集检验恒 True | front.py:184 `len(root_sets)==1`→`True` | **110 全绿(存活)** | 见下:≈等价变异 |
| ①b 逐名合并冲突禁用 | front.py:192-196 冲突检测删除 | **110 全绿(存活)** | 同上 |
| ①c **两者同时禁用** | ①+①b | **110 仍全绿** | **真空洞** |
| ② 回退改"新 source 配旧 derived 直发" | front.py:207-222 else 分支改撕裂直发 | **2 红**:`test_build_front_is_transitive_and_falls_back_as_one_family` + `test_processor_advances_with_its_exact_source_publication` | **击杀 ✓** |
| ③ WeakKey 保活探针 | 见下 | PASS | ✓ |

①/①b 单独存活的定性:补做了公共 API 可达的双 leaf 分歧探针(camera→roi、camera→fit 双 derived;camera 进到 seq2、只有 roi 追上)——clean、①、①b 三种实现下 front 均 `{camera:1, fit:1, roi:1}` 整族回退,因为根集检查与逐名合并在单源 derived+同世代 parents 的可达拓扑里**互为冗余带**,单杀其一被另一个兜住(≈等价变异,不算测试失职)。但 **①c 两者同杀后,同一探针产出真撕裂 front `{camera:2, roi:2, fit:1}`,而 110 测试依然全绿**——套件里没有任何"同族双 leaf 分歧"场景(现有 held 测试的失格来源是 leaf 无 publication,走不到这两条检查)。GOAL P2.4 声称"断言每个 front 内族成员 `_publication_roots` 恒一致"只在单 leaf 链上成立。**小修:补一个 fan-out 双 derived 分歧测试**(我的探针脚本即现成模板,断言整族回退 + 追上后一起前进)。

③ WeakKey 保活(真 plane 三级链 camera→roi→fit,seq1 族发布后整族推进到 seq2,使 plane states 不再持有任何 seq1 引用):仅持 fit1 时 `gc.collect()` 后 root1/roi1 弱引用仍活,`direct_parent_publications(fit1)`→roi1→root1 逐级可解析;放掉 fit1 后三者全部回收,**保活与无泄漏双向通过**。附带正面证据:leaf 落后时 freeze 把 camera 钉回 leaf 的 shot(front 显 seq1 而非已发布的 seq2)= 族一致语义正确。⚠️ 注意仓内 `test_signal_front.py:179` 这条保活测试较弱:其 root 同时被 camera `state.publication` 强持有,`root_reference() is not None` 断言恒真,防不住保活责任回归——建议按我的探针(states 推进一代后再断言)加强。

## 1.4 契约一致性(docs/contract.md FROZEN vs 实现) — **一处漂移必须修**

逐签名核对结果:

- ✓ `reserve` 返回 `StreamGenerationId`(plane.py:785-819);`retire`→`frozenset[str]`;`attach/detach_live/mark_changed/latest_publication/freeze/bind_owner_wake/unbind_owner_wake/close` 全吻合。
- ✓ `publish_final`/`publish_processor` 返回 `publication.signals`(MappingProxy → `Mapping[str, SignalValue]`);`publish_processor` 的 `source_publication` keyword-only 且 parents 强制 `(source,)`(plane.py:1246-1249),与 contract.md:32-37 注记一致。
- ✓ derived 五方法(`bind_continuous_derived` keyword-only 三参/`publish_continuous_derived`/`fail_continuous_derived`/`continuous_needs_publication`/`withdraw_derived`)与 contract.md:48-73 逐参吻合。
- ✓ `SignalValue` 的 block/schema/values/behind(plane.py:143-198)、`SignalFront` 的 signals/publication_by_signal/continuous_group、`SignalPublication` 的 event_ref/direct_parent_refs/兄弟 Mapping 全在。
- ✓ allow-list 15 名与 `__init__.__all__` 一致,且 `test_import_guards.py:29-53` **直接从 contract.md 解析 facade 行做断言**并带非空自证——真机械耦合,非绿探针。
- 🔴 **契约漂移(跨仓事故引信)**:contract.md:26-30 写 `attach_latest_only_processor(signal_key: str, control, initial_publication)`(signal_key 第一位、位置参数);实现 plane.py:924-930 是 `attach_latest_only_processor(node, *, source_name: str, initial_publication)`(control 第一位、后两参 keyword-only、名字不同)。zlc_atom 按契约写的 fake/调用会在真 plane 上直接 TypeError。现在 zlc_atom 尚未调用此 API(全仓仅 `tests/test_import_boundaries.py` 引用包名),尚未成事故——**必须按 FROZEN 规则校订 contract.md 此条(commit message 点名),或改实现签名**。
- 🟡 契约未誊抄面:`set_front_signals` 完全不在 contract.md——而族一致性**只对该声明集生效**(不调用则 requested 为空,任何族都不受回退保护),examples/demo_signal_flow.py:151 与测试都在用;这是宿主/呈现侧必需 API,并行仓 fake 缺它写不出正确行为,应补进契约。另 `publication_owner`(opaque token,GOAL P2.5 明写)/`direct_parent_publications`/`withdraw_processor` 公共存在但契约未列,建议一并裁决(列入或显式标注包内私有)。`publish_final` 形参名 `projected` vs 契约 `outputs`(关键字调用会炸)——一行校订。

## 遗留死代码(小修,干净删除原则)

`src/zlc_runtime/plane.py` 五处零调用(src+tests 全 grep 确认):`active_processor_bindings`(:519)、`_state_for_generation_ref_locked`(:671,preemption 删除后遗孤)、`_publication_roots`(:1637)/`_collect_publication_ancestry`(:1656)/`_name_is_ancestor`(:1680)——后三者是 front.py 抽取后的**死副本**(活体在 front.py:33/53/74),违单源。

## 小修清单(按优先级)

1. 校订 contract.md 的 `attach_latest_only_processor` 签名(或改实现对齐),同时补 `set_front_signals` 进契约、校订 `publish_final` 形参名。
2. 补"同族双 leaf 分歧"测试封死 ①c 空洞(fan-out 双 derived,一快一慢,断言整族回退+一起恢复)。
3. 加强 WeakKey 保活测试(states 推进一代后再断言,消除 state 强引用造成的恒真)。
4. 删 plane.py 五处死代码。
5. (可选)GOAL P2.1 "四分"名实不副:补注 shim 现实或真拆。