# 任务2 验收报告:呈现调度与节点宿主

**结论:小修**(P4 呈现调度=PASS 级,变异实验全部击杀;P5 骨架真实、两条核心不变量有真探针,但**两个存活变异实证覆盖缺口** + 一处对树内的语义偏离需用户拍板;P5 未勾选是诚实的,勿直接勾)。

---

## 1) presentation.py vs survey A.2 / GOAL P4 —— 逐项在,零 Qt

| A.2/P4.1 项 | 实现位置 | 判定 |
|---|---|---|
| `WakeSink` Protocol | presentation.py:34-38 | 在,签名同 A.2 |
| `OwnerChannels` 三通道 pending + data token 借还 | presentation.py:49-174;`activate_data` 含 bind 失败回滚(:117-122)与 close-during-activation 回滚(:123-133),`deactivate_data`/`close` 均归还 token(:135-174) | 在,借还纪律比 A.2 草案更完备 |
| `HarmonicClock` 纯算术 | presentation.py:177-232;无时间源无线程,`rebase` 返回新基频(:217-221),`group_due = elapsed % max(组内)==0`(:227-232) | 在 |
| `SurfaceUpdate` host 字段 opaque 化 | presentation.py:235-262,`host_token: object`(:241),全文件零 zlc_plot/Qt import(:1-13) | 在 |
| `SurfacePort` Protocol 11 方法 | presentation.py:265-298,与 A.2 逐名一致 | 在 |
| `SurfaceBatchArbiter` all-or-nothing 入批 / 整批 done 才上 / 任一失格整批弃 | 缺值整组不提交+report_waiting(:364-368);prepare 中途异常已提交成员 finish_unpresented 回收(:371-384);`drain` 批内全 done 才处理(:417-419),任一 resolve 失败/future 异常/can_accept 失败→整批 reject(:424-472) | 在 |
| `BoardScheduler` freeze→分组→due→enqueue | presentation.py:574-629;on_owner_turn 分派 surface→drain、lifecycle→poll、(lifecycle\|data)→freeze(:631-642),同 window.py:3237-3270 语义 | 在 |
| "mark_changed 只在有 reactive 下游才推 wake、纯显示靠 timer 拉" | **保留**:plane.py:854-885——wake 仅当存在未退休 `kind in {"processor","continuous"}` 的下游消费该输出(:872-883),否则只置 dirty 等 tick 拉 | 在 |
| window_runtime 三函数原样迁入 | 与树内 `zlc_workbench/window_runtime.py` diff 仅 4 处 docstring + 1 处类型标注(`temporary: Path \| None`,window_runtime.py:58),零逻辑改动 | 在 |

## 2) 欠拍回收变异实验(临时副本,基线 7 测试绿)

- **变异 A(整删 owed 位**,即迁移分支丢失前形态):`on_tick` 的 owed 记账/重试全删 → **两测试红**(`test_board_scheduler_owes_a_failed_slow_beat_to_the_next_base_tick`、`..._owes_missing_value_until_the_next_base_tick`,test_presentation.py:243/269)。
- **变异 B(owed 记了但只在 due 拍重试**,即 A.3 描述的"黑 2 秒"劣化):`if not due: continue` → **两测试红**。
- 测试确用 update_ms=2000、base=100,断言 elapsed=2100(下一 base tick)即补(test_presentation.py:257-263)。**P4.2 保证被机械守住,非形状剧场。**

## 3) host.py vs 树内 hosted_run.py 逐项

| 项 | 判定 | 证据 |
|---|---|---|
| generation 重置防陈旧 | **在**(带一缺口) | `_reset_generation` host.py:429-442;陈旧 completion 过滤 `completion.generation != self._owner.generation` host.py:475-477(镜像 hosted_run.py:453)。变异 H2(删过滤)被 `test_stale_mailbox_completion_cannot_replace_new_generation`(test_host.py:269)击杀。**但变异 H3(重置不 clear stop_event)全绿存活**——cancel 后重启无测试。另:host.py:210/430 的 `self._generation` 计数器只写不读,死变量 |
| 声明了输出却没发布=硬失败 | **在**(带语义偏离) | host.py:491-498;变异 H1(删守卫)被 test_host.py:199("missing final" 参数化)击杀。**偏离**:树内对 `_live_opened` 豁免(hosted_run.py:463-485 有整段注释解释 live 生产者 detach 保 FINAL),包内不豁免——live-streaming finite 节点自然结束会被记 failed。GOAL P5.2 的一揽子措辞支持现实现,但这改了树内语义,**需用户裁决** |
| live 附着纪律(一 generation 一 live) | **在** | host.py:565-566 二次 open 必 raise;`_live_opened` 每代复位 host.py:441(但无测试覆盖二次 open 路径) |
| processor 回调四件套 | **在** | validate/evaluate/accept_result/accept_failure/accept_cancelled/request_wake host.py:643-696;plane 侧 lane 真实调用(plane.py:442/469/494/539/553/557/561),协议面 plane.py:84-102 |
| 命名策略注入 | **在** | `signal_namer` 参数 host.py:133,默认 `@logic/{owner}/{name}`(:146-147),`signal_key` 走注入器(:310) |

次要偏离(知情即可):`_finish_finite_failure` 以 `self.cancel_requested` 旗标代替树内 RunCancelled 类型判定(host.py:506-508)——cancel 窗口内的真异常会被记 "cancelled" 且吞错文本(树内记 failed 留错);这是 run.py 留域后的合理适配,test_host.py:244-245 已按新语义写。`worker_idle` 对运行中的 reactive 恒 True(host.py:349-354,树内为 `not self._active`),仅影响外部轮询者。

## 4) P5 三项判定(commit 264dd86 = host.py 700 行 + test_host.py 452 行 + README 5 行)

- **P5.1 最小 Node 协议:完成**。`kind ∈ {finite, reactive}` 强制(host.py:137-139);finite ctx 恰六能力(host.py:73-115,测试逐名断言 test_host.py:156-166);reactive 恰一输入键强制(host.py:174-175, 283-284)。
- **P5.2 NodeHost 骨架:完成(带上表两处语义偏离 + 死变量)**。start/cancel/poll/shutdown 单表面 ✓;shutdown 拒关 ✓(host.py:411-418);descriptor/ApplicationContext/device_requirements 留域已记 README.md:36-37 ✓。
- **P5.3 宿主测试:部分**。
  - finite 四路:成功(test_host.py:145)/失败(:198)/取消(:231)/声明未发布(:199)——**全在**,H1/H2 变异证明非空洞。
  - shutdown 拒关(:252-253)、mailbox 陈旧完成防御(:269)——**在**。
  - reactive 失败(:403 mode="failure")、in-flight cancel(:403 mode="cancel")、idle cancel(:392-394)——**在**。
  - **跳版:缺**。`test_reactive_node_follows_latest_publication_and_can_jump`(:350)名不副实——rev1 等完再发 rev2,断言 `revisions == [1, 2]`,从未跳过任何版本。**变异 P1(route 在 worker busy 时丢弃新 publication、废掉 skip-to-latest 吸收,plane.py:473-475)全套 110 测试存活**——latest-only 的核心保证("忙时吸收最新、完工即追")在 host 与 plane 两层都无守卫。

### 补齐清单(按优先级)

1. **真跳版测试**:gate 住 evaluate(rev1),期间连发 rev2/rev3 并 freeze,放行后断言 evaluate 序列 == [1, 3](rev2 被跳),且派生 publication 的 `direct_parent_publications` == rev3 的 publication(杀变异 P1)。
2. **cancel→restart 测试**:cancel 至 terminal 后再 start,断言第二代正常跑完(杀变异 H3 的 stop_event 复位缺口)。
3. live 附着纪律测试:同一代第二次 `open_live_dataset` 必 raise(host.py:565-566 目前无覆盖)。
4. 裁决项(先问用户再写测试):live-opened finite 成功但未 publish_final——树内=成功保 FINAL(hosted_run.py:463-485),现实现=failed(host.py:492-498),二选一后补对应测试。
5. 清理:删死变量 `self._generation`(host.py:210/430)或让它真被读。

**变异实验总记录**:P4 owed 位 2/2 击杀;host H1/H2 击杀、H3 存活;plane P1 存活(全量 110 绿下)。实验全部在系统临时目录副本进行,被验收仓与参照树未动。