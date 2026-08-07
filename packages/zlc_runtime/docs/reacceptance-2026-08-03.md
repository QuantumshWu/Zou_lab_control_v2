# zlc_runtime R 轮复验报告(2026-08-03)

# 复验报告 — zlc_runtime R1-R6 契约与收尾核实(2026-08-03)

**实验环境**:全部实测在系统临时目录副本(`scratchpad/verify/` 下 `git archive HEAD` 快照 + 独立 venv,Python 3.13.12);`zlc_runtime.__file__`/`zlc_data.__file__`/`zlc_pulse.__file__` 均断言解析到副本 venv site-packages。原仓零写入,结束时四仓 `git status` 与开工完全一致(runtime 仅原有三份 untracked 验收文档,atom/data/pulse 干净)。

## 1) 契约与实现全量对照(R1)

| 项 | 契约(docs/contract.md HEAD) | 实现(src/zlc_runtime/plane.py) | 判定 |
|---|---|---|---|
| ① attach_latest_only_processor 新拼写 | :27-32 `(node, *, source_name, initial_publication)` | :892-898 逐参数一致(含 keyword-only) | ✅ |
| ② set_front_signals 入契约 | 签名 :26;语义段 :53-54"declares the signal family…same-shot fallback enforced for that declared set" | :609-625,语义相符(声明集变更触 membership_changed+wake) | ✅ |
| ③ publish_final 形参名 | :21-24 `outputs` | :1105-1108 `outputs`(R1 diff 实证从 `projected` 改名;HEAD grep `projected` 零命中) | ✅ |
| ④ 三缝列入 | `withdraw_processor` 进方法块 :34 + 叙述段 :47-52 列 `direct_parent_publications`/`publication_owner`(选"列入"而非标私有) | :947 / :861-868 / :870-890 存在且签名一致 | ✅ |

附带核对:derived 五函数族(contract :60-85 vs plane.py:1222/1341/1390/1419/1441)、facade 15 名 allow-list(contract :9 vs `__init__.py:10-26` 精确相等)、LiveDatasetPort 表面(contract :100 vs live_dataset.py:98-205)、host 六能力(contract :106-113 vs host.py:81-114)、presentation 面(presentation.py:49/177/266/301/506 + window_runtime.py:23/38/72)——全部在。

**R1 commit message 点名:部分满足**。`201f624` message 只有单行 "R1 sync plane contract signatures and seams",①-④ 逐项点名写在同一 commit 的 GOAL.md R1 条目里(可追溯但不在 message 本体);契约 :5 的字面要求是 "commit message 点名"。小瑕疵,不构成事故引信(契约、实现、清单同 commit 原子落地)。

**残余低风险偏差(R1 未声称修、如实记录)**:契约首位形参拼写 `producer`/`control`(reserve/retire/attach/mark_changed/publish_final/cancel·withdraw_processor)与实现 `node` 不一致,另 `attach(…, live_slot)` vs 实现 `slot`、`latest_publication(signal_key)` vs 实现 `signal_name`——均为惯例上位置传参的形参,关键字必炸的参数(keyword-only 三处)已全部一致;atom 实测全部位置调用,无实害。

## 2) 跨仓一致性(atom M2)

- `zlc_atom` 提交 `8e58386`(M2)删除了旧拼写 override:fakes.py 现在 `FakePlane(RuntimeSignalDataPlane)` 直接**继承**真 plane 的 `attach_latest_only_processor`;test_contract_fakes.py:18-23 断言继承签名 == `("self","node","source_name","initial_publication")`,与 runtime 现行实现及契约新拼写逐参数一致;publish_processor 断言 `source_publication` 为 KEYWORD_ONLY(:14-16),与 plane.py:1157-1163 一致。
- atom 全仓 grep 无负面清单符号(association/preemption 族零命中),无旧拼写调用点(atom 自己的 `signal_key()` 是节点命名 helper,非 runtime API)。
- **实跑**:临时 venv 安装 zlc-data + zlc-pulse + zlc-runtime + zlc-atom(均 HEAD 快照),`zlc_atom` 套件 **54 passed**(exit 0);同环境 `zlc_runtime` 套件 **120 passed**(exit 0)。

## 3) R4④ 裁决执行

- 实现已改回树内语义:host.py:488-503 `_finish_finite_success` 与参照 `Zou_lab_control_v1/zlc_neutral_atom/runtime/hosted_run.py:463-488` 结构逐行同构(live-opened 且未 publish_final → `_detach_plane_state()` 走 done,不再判 failed;仅未开 live 才硬失败)。`83664cd` diff 实证此为 R4 改动。
- 测试在:tests/test_host.py:552-579 `test_finite_live_open_without_final_uses_tree_terminal_semantics`,断言 `phase=="done"`、`error is None`、transient-only 发布被正确收回(latest_publication 为 None)。R4③ 同代二次 open_live 必 raise 也有测(:519-549)。
- 备注(不阻塞):detach_live 的 retained-FINAL 分支(plane.py:1527-1546,live+publish_final 后 detach 保留 FINAL)无组合直测;FINAL-only 保留由 test_host.py:145-192 覆盖。

## 4) R5 死代码与分支测试

- 五符号 grep(src/+tests/,排 .pyc):`active_processor_bindings`/`_state_for_generation_ref_locked`/`_collect_publication_ancestry` 全仓零命中;`_publication_roots`/`_name_is_ancestor` 仅存 front.py:33/74 **活体原件**(plane.py 死副本已删,单源恢复正确)。`84dc3fb` diff:plane.py -89 行。
- LiveDatasetPort 分支测试在且有实质:test_runtime_helpers.py:125-152(`fail()`:failure 记录、terminal、source 关闭、listener 收通知)与 :155-187(`source_terminal()` 双策略:retain_on_terminal=True 保 bound 不 withdraw、close 才关 source;False 立即 withdraw+关 source)——断言的都是 live_dataset.py:176-203 的真实可观测副作用,非空洞。

## 5) R6 簿记

- `319425e` diff 实证:P5.1-P5.3、P6.1-P6.3 全部翻 [x];P2.1 shim 注记在(GOAL.md:37"values/processor_lane/registry 是兼容 shim,plane.py 仍是实现单体");状态改 GOAL COMPLETE(GOAL.md:3);R1-R6 全勾。终态判据实测复核:干净 venv 120 全绿(上文)。host.py `self._generation` 死变量确删(仅剩方法名 `_reset_generation` 与局部变量)。
- **三份验收文档判定:应入库**。GOAL.md:2(旧状态行)与 :74 均引用 `docs/acceptance-*-2026-08-03.md` 作为 R 轮证据与前置阅读,tracked 文件引用 untracked 文件=fresh clone 断链;docs 目录其余同类文档(survey-*/loc-report)均已入库。建议一条 `git add docs/acceptance-*-2026-08-03.md` + commit 收口。

## 结论:**小修**

全部六项声称经独立实证成立(契约四点落地、跨仓 54+120 全绿、R4④ 树内语义+测试、五死代码确删、簿记齐)。留两件小修:① 三份 acceptance 文档入库(唯一实际动作);② 记录性:R1 式契约变更下次把逐项点名写进 commit message 本体;契约 `producer/control` vs 实现 `node` 的位置形参拼写残差可在下次契约校订顺手齐平(或在契约 :5 明记"首位形参按位置传")。

---

# 任务B 复验报告:R1–R6 存活探针 mutation 复验

**结论:PASS**(六项全部通过;5 个 mutation 逐个应用→全套→还原复绿,每个都被对应 R 轮新测试精确击杀)

**实验环境**:全部在 scratchpad 副本 `...\scratchpad\mutB\zlc_runtime`(+ `zlc_data` 副本)进行;PYTHONPATH 指向副本 src,并断言 `zlc_runtime.__file__`/`zlc_data.__file__` 解析到 scratchpad 路径后才开跑。副本基线 **120 全绿**(7.3s)。被验收仓全程只读:结束时 `git status` 与开工完全一致(HEAD=319425e,唯 3 个 untracked `docs/acceptance-*-2026-08-03.md`)。

## Mutation 复验表(逐个应用 → 全套 → 还原复绿)

| # | 变异 | 位置 | 套件结果 | 命中测试 | 判定 |
|---|---|---|---|---|---|
| B1 ①c | 根集检验恒 True + 逐名合并冲突检测删除(**同时**) | front.py:184 + :192-196 | **1 红 / 119 绿** | `test_fanout_leaf_disagreement_rolls_back_the_complete_family`(test_signal_plane.py:480),失败形态=撕裂 front `{camera:2, roi:2, fit:1}` vs 期望全 1(:550)——正是上轮探针复现的真空洞场景 | **击杀 ✓** |
| B2 精确变体 | parents 载荷改弱引用存储(保活断链、解析仍通) | plane.py:1095 + :959-977 | **1 红 / 119 绿** | `test_plane_front_keeps_weak_parent_payload_alive_and_tracks_missed_coverage`,恰失败于保活断言 `assert root_reference() is not None`(test_signal_front.py:273) | **击杀 ✓** |
| B2 任务建议变体 | 发布时不登记 parents 载荷 | plane.py:1095 | 6 红 | 含该 WeakKey 测试 + fan-out/skip/demo 等 5 个 | **击杀 ✓** |
| B3 (D) | follow() 中途加入改 `start_sequence=0` 回放 | streams.py:1448 | **1 红 / 119 绿** | `test_follow_joins_at_the_current_sequence_without_replay`(test_runtime_streams.py:210),且套件干净跑完未挂死 | **击杀 ✓** |
| B4 (H3) | `_reset_generation` 删 `self._stop_event.clear()` | host.py:437 | **1 红 / 119 绿** | `test_finite_cancel_then_restart_resets_the_generation_stop_event`(test_host.py:467) | **击杀 ✓** |
| B5 (P1) | 跳版改逐版(pending 单槽改 FIFO 队列,route/_start_processor 三处) | plane.py:404/473-475/519 | **1 红 / 119 绿** | `test_reactive_latest_only_skips_busy_intermediate_publications`(test_host.py:403),失败=`assert [1, 2, 3] == [1, 3]`(:451) | **击杀 ✓** |

每次变异后 `git checkout` 还原并复跑确认复绿;末次全套复绿 **120 passed**。上轮 4 个存活探针(①c 联合空洞、WeakKey 恒真、mutation D、H3)+ 1 个 plane 层存活变异(P1 跳版)现全部被杀,无一存活。

## 附带核实项

- **B2 测试非恒真**:重写后的测试确在 states 推进一代后断言——`state["frame"]` 推进到 revision 2、roi/fit 各自 republish、第二个 front 全 seq2 后,删除两个旧 front、仅保 `first_fit` 强引用再断言弱引用与 `direct_parent_publications` 逐级解析,末尾还验证全放手后三级全部回收(无泄漏)。M2b 恰在保活断言处红是"断言已激活"的实证;旧版的 root 被 camera state.publication 强持有,同一变异下旧断言仍会通过(恒真病确已治愈)。
- **B3 timeout**:`cursor.next(timeout=0.1)` 已加(test_runtime_streams.py:146,另 :111 同样有);gap 类回归现干净红而非挂死。
- **B3 并发压测**:`test_exact_producer_and_consumer_keep_high_throughput_ordered`(test_runtime_streams.py:243,2000 事件双线程 Barrier 同步,断言逐序无损 + retained_events==0)确在 acca86f。另 follow gap 响亮测试(:227)同轮补入。小观察(不扣分):GOAL P1.4 原文"三口各一组并发压测"字面上仍只有 exact 口有高吞吐压测,monitor/follow 为点测;但 R6 已在 P1.4 尾注记 shim/裁减语境,且 R3 任务的"补压测**或**记裁减"二选一已按前者兑现。
- **B6 live 单例测试在**:`test_finite_live_attachment_is_singleton_per_generation`(test_host.py:519),同代第二次 `open_live_dataset` 断言 `pytest.raises(RuntimeError, match="one Node generation")`(:534),对应 host.py:567 守卫。
- **B6 计数对照**:全套 **120**(上轮基线 110);`git diff bf03a9d..319425e -- tests` 恰 +10 个测试(R2 +1 fanout、R3 +3、R4 +4、R5 +2 LiveDatasetPort fail/source_terminal 分支),与 110→120 精确对账,无幽灵增减。

## 收尾状态

- 副本 `src`/`tests` git status 干净(全部变异已还原),末次全套 120 绿。
- 原仓 `C:/Users/eadri/Dropbox/WorkCode/Github/zlc_runtime`:HEAD=319425e,工作树与开工时一致(仅 3 个 untracked 验收报告),未被触碰。