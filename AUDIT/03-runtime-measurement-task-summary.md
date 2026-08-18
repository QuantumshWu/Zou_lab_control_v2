# 03 — Runtime、Measurement、Task、Signal与Preview综合结论

状态：阶段完成。详细证据见同目录`03a`、`03b`和`03c`。

## 1. 总结

用户观察到“每新增Logic Node就漏live或preview”不是偶然。当前系统同时维护：

1. descriptor output；
2. node output declaration；
3. plugin live slot mapping；
4. preview output string；
5. progress；
6. artifact result reflection。

六者没有一个统一构造/terminal contract。Runtime还同时保留一套约4,000行、只有测试使用的exact/monitor/builder/live-port框架，而真实product nodes全部自造slot。结果是“复杂通用框架闲置、实际路径仍手写且互不一致”。

## 2. P0正确性问题

### RT-001 — future cell被发布为已测empty

finite Camera用零图补齐未来cycles，但snapshot validity全真；Occupancy忽略parent coverage/validity、分类整块数据，再把自己的coverage写成complete。尚未发生的shot因此成为有效empty事实。

### RT-002 — finite live到Final使Follow Processor正常结束时失败

Processor先跟随`DatasetCoverage` live publications；source terminal Final改成`coverage=None`。Follow Processor把这份正常terminal replacement当非法live input，在EOS前失败。真实Camera→Occupancy路径可复现。

### RT-003 — UI freeze改变Stop后的科学数据

相同repeat=4 run、采到1帧后Stop：

| Stop前状态 | Terminal shape | Coverage | Transient |
|---|---:|---:|---:|
| plane从未freeze | `(1,1,96,128)` | `None` | `False` |
| plane已freeze一次 | `(4,1,96,128)` | `(1,4)` | `True` |

显示时序不应参与科学结果。Stopped partial必须固定authored schema、unfilled invalid、retained、non-transient，并可由Processor one-shot消费。

### RT-004 — dirty/pull slot可重复发布同一content revision

slot mutate、dirty set和稍后pull并非原子。并发update期间一次freeze可读到新版，下一次dirty仍在又发布同一snapshot ref，形成两个不同EventRef。FollowTap会把它们当两个事件，plot则可能按revision丢第二个。

### RT-005 — finite与monitor没有共同cycle assembler

Monitor检查source ordinal连续；FiniteCapture只数frame数量。两帧cycle收到ordinal`(0,2)`仍被接受，会跨物理gap拼成一个shot。Temperature/SLM Feedback复用finite path，也受影响。

### RT-006 — Preview/status本身会说假话

- Logic row仅因host running就把所有声明output显示为`live`，即使plane没有publication。
- worker成功启动后phase仍是`starting`。
- terminal后last progress不清，UI优先显示旧`Scanning/Saving/qCMOS`而非done/cancelled。
- auto preview失败前就被标记`previewed`，以后不重试。
- 显式preview kind不兼容时静默fallback到自动猜图，而不是暴露descriptor错误。

### RT-007 — Artifact声明不是成功契约

成功Task若没有返回声明字段、路径不存在或多返回未声明artifact，Workbench当前只跳过，不判失败。Calibration/Temperature最终保存也没有与Stop通过terminal seal排序，可能留下文件但Host标cancelled且UI不报告artifact。

## 3. 明确性能问题

### RT-008 — Camera/Occupancy累计链O(N²)

`96×128 uint16`、一帧/cycle的Camera live探针：

| Repeat | N次update+freeze | 累计payload下界 |
|---:|---:|---:|
| 25 | 0.0245s | 14.6MiB |
| 50 | 0.0868s | 58.6MiB |
| 100 | 0.3547s | 234.4MiB |
| 200 | 1.4664s | 937.5MiB |

Occupancy对每份累计snapshot从头重算所有frames×sites；100次publication、100帧、35sites约350,000次site-frame extraction。全幅真实camera成本会高几个数量级。

### RT-009 — Scan每point复制完整values和validity

ScanDatasetWriter积累buffer本身合理，但每point的`live_output()`都把完整容量复制成immutable snapshot并扫描validity。大image scan在进入plot前就可能产生GiB级逻辑复制。

### RT-010 — Global processor head-of-line blocking

所有latest processors共用`ThreadPoolExecutor(max_workers=1)`。任一慢Occupancy/selection processor会阻塞所有其他独立processor。

### RT-011 — Plane freeze会执行plugin代码

正常UI tick中`plane.freeze()`会调用plugin `freeze_live_outputs()`并做stack/schema/materialization；有follower时同一函数又可能在producer/acquisition thread执行。是否有人订阅会隐式改变计算线程和采集阻塞成本。

## 4. Contract与历史框架裁决

### 保留并重做

- `NodeHost` lifecycle、cancel、terminal seal；
- `SignalDataPlane` generation/publication/lineage/front；
- `EventRef`、SignalValue/Publication/Front；
- Dataset output declaration与coverage概念；
- future-publication follow行为；
- Processor latest/exact消费语义；
- Board presentation lineage。

### 删除/合并候选

当前约4,000行从初始monorepo commit起无product consumer：

- exact reservation/cursor/delivery/readiness；
- `DatasetBuilder`、exact delta preview、sealed artifact；
- generic `MonitorTap/MonitorDataset`；
- `LiveDatasetPort/_ExactDeltaLivePort/preview.py`；
- `RunHandleLike/start_and_wait`。

真实产品只使用streams的一小段future FollowTap。建议保留最小follower并合入SignalPlane边界，删除测试维持的通用岛，除非用户明确要求Runtime作为有仓外消费者的独立通用库。

### 输出真相应合一

- 删除无当前消费者的`resolve_outputs`；
- 删除唯一且不使用values的`resolve_node_previews/_frames_preview`；
- descriptor直接复用`DatasetOutputDeclaration`，不再复制`OutputSpec(name, contract)`；
- Preview引用同一declaration，不使用裸字符串；
- Composition根据`NodeKind`显式告诉Host是worker还是processor，不再靠`source_signal/input_signal(s)`属性名猜；
- 每次live/final publication兑现完整frozen output vocabulary。

## 5. 推荐的最简目标链

```text
LogicNodeDescriptor
  ├─ one immutable output declaration tuple
  ├─ valid preview subset
  └─ explicit role
        ↓
NodeHost / NodeExecutionContext
  ├─ explicit worker or processor mode
  ├─ atomic committed live output seam
  ├─ progress/running/terminal truth
  └─ terminal artifact validation
        ↓
SignalDataPlane
  ├─ frozen generation contract
  ├─ monotonic content stamp
  ├─ immutable live/final publications
  ├─ lineage/follower
  └─ pure read-only freeze()
```

推荐live seam是Host-owned atomic push：plugin提交已经形成的immutable output bundle，Plane只交换引用并wake，`freeze()`不得调用plugin。若真实camera profile证明snapshot materialization不能在acquisition worker完成，则保留一个**统一的Host-owned materialization lane**，而不是继续允许每个plugin自造slot和不确定线程。

交叉复核统一`03a/03b`措辞：03b的`KEEP`只表示camera lazy materialization责任可能仍有价值，不表示保留当前`_CameraLiveSlot`的listener/dirty/pull/terminal lifecycle；03a的`DELETE after replacement`指当前实现。最终裁决是`REPLACE CONTRACT`，是否需要Host-owned materialization lane由D-038/profile决定。

Coverage只描述geometry写入程度，不能兼任lifecycle tag。Live/latest/growing、terminal complete/partial/failure和retention必须由Host/Plane的run state明确表达。

## 6. Measurement与Task产品建议

- Camera finite live：推荐只发布最新完整cycle；terminal一次发布固定authored schema的完整/partial Dataset。避免全帧累计O(N²)。
- Scan：允许growing preview，但应发布小的科学summary或增量commit，不每point复制全部raw image history。
- Processor：只处理新commit，传播parent validity/coverage；terminal partial走one-shot reconcile，不按coverage class猜lifecycle。
- Measurement/Task只要声明Dataset output，hosted product run必须在terminal前至少live发布一次；Task还必须及时progress。
- Row只按plane事实显示`waiting/live/held`。
- Preview kind错误明确失败，不fallback猜测。
- Task不可逆save/apply前必须与Stop完成terminal ordering；successful terminal必须兑现全部artifact declarations。
- SLM running preview应在candidate phase应用后开始发布running camera mean，而不是等100 shots全部结束才第一次更新。

## 7. 逐节点结论

| Node | 结论 |
|---|---|
| Camera Measurement | 核心保留；finite live/partial/cycle assembler重做 |
| Stepped Scan | 核心保留；growing copy与safe cleanup重做 |
| Seamless Scan | 核心保留；与Stepped共享live/terminal规则 |
| Occupancy | 数学核心保留；partial validity/coverage/terminal/recompute重做 |
| Calibration | 核心保留；preview extent、terminal save/Stop排序重做 |
| Temperature | 核心保留；raw scan output与terminal artifact待裁决 |
| SLM Feedback | 核心保留；running cadence、Stop语义、统计/算法由SLM专项继续 |

## 8. 测试策略裁决

应增加少量真正纵向红灯：

1. Stop前有/无UI freeze得到严格相同partial terminal；
2. finite Camera live→Occupancy terminal双方done；
3. future invalid不产生empty事实；
4. ordinal gap不组成cycle；
5. generic Measurement/Task在terminal前live且preview names有效；
6. successful Task兑现全部artifact；
7. terminal row显示phase而非last progress；
8. 独立processors不互相head-of-line block。

应删除大量只守无人使用Runtime exact/builder/live-port框架的测试；不能让这些测试反过来成为保留理由。

## 9. 尚待用户裁决

所有产品取舍已登记到[DECISIONS.md](DECISIONS.md)。本阶段重点是Task active允许的UI、SLM Stop、Calibration preview与terminal retention、Temperature raw scan output、Processor preview、direct API live边界、Runtime通用库范围、live publish seam和processor并发。
