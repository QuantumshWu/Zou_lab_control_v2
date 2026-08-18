# 05a — SLM truth / context 边界深审

状态：本子阶段完成。
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：target、objective、pupil、Pattern/base phase、Zernike/steering、science phase、device correction/LUT、Editor、adapter last-commanded、Feedback、artifact/load/Stop。
约束：只读源码、tests、Git历史和无硬件数值探针；只新增本报告，未修改代码、硬件或其他文档。

## 1. 结论先行

当前每个局部组件都维护了一份“看起来合理”的状态，但它们没有形成一条可传递的光学上下文：

~~~text
Editor target intensity
  + Editor-only objective_kind
  + Editor-only pupil amplitude/ellipse
  -> Pattern/base phase
  + Editor-only steering/Z4-Z11
  -> science phase (pre-correction)
  -> adapter current correction/orientation/LUT
  -> physical gray frame
~~~

Feedback实际只拿到：

~~~text
target JSON intensity
+ adapter.last_commanded_phase
+ calibration/pulse
~~~

它没有拿到Editor objective、pupil、Pattern/wavefront分解或device correction command context。于是它：

- 用 `objective_kind="auto"`；
- 用solver第三种默认硬圆pupil，而不是Editor Gaussian或uniform pupil；
- 把adapter的完整science phase（可能含Zernike/steering）当Pattern warm start；
- 求出并apply一张新的完整phase，不显式保留operator wavefront；
- 保存pre-correction canonical phase，但不保存当时的correction/LUT/orientation receipt。

更严重的是，真实X15213 adapter初始化时没有读回或发送phase，却把 `last_commanded_phase` 初始化成zero。Feedback把这张软件虚构zero当真实incoming，并在失败时“恢复”它。

因此当前不能同时声称：

1. Feedback只校正Pattern amplitudes、不接管operator wavefront；
2. Stop/failure恢复任务开始前的真实硬件状态；
3. candidate NPZ可物理复现；
4. target JSON可复现Editor/Feedback求解；
5. Editor窗口始终显示当前device truth。

phase NPZ的二维canonical array roundtrip本身是正确的；缺的是context与physical command receipt。两者必须分开评价。

## 2. 当前truth地图

| 概念 | 当前owner | 持久化 | Feedback是否获得 |
|---|---|---|---|
| Target intensity | Editor `_target` / target JSON | JSON intensity矩阵 | 是，独立target artifact |
| Objective kind | Editor `_objective_kind` | **不在target JSON**；只在solver/phase metadata中偶尔出现 | 否，Feedback用auto |
| Pupil enabled/center/diameter | Editor fields | 只有不完整description string | 否 |
| Pupil amplitude array | Editor `_pupil_amplitude` | 不持久化 | 否，Feedback用solver default |
| Pupil support/Zernike disk | Editor `_pupil_support` | 不完整 | 否 |
| Pattern/base phase | Editor `_pattern_phase` | 不单独保存 | 否 |
| Steering/Z4-Z11 | Editor widgets | science-phase metadata含coefficients，但pupil center缺失 | 否 |
| Science phase pre-correction | Editor `_phase` / phase NPZ / adapter last phase | NPZ exact canonical | 是，但无分解 |
| Vendor correction | X15213Adapter | device config/path；runtime可变 | 否 |
| Orientation/LUT/wavelength | X15213Adapter/profile | installation config/profile | 否 |
| Physical gray command | USB frame memory / DVI raster | candidate artifact不保存 | 否 |
| Spot optimizer state | caller-owned transient dict | 按设计不持久化 | Feedback自己从空state建立 |
| Candidate updated target weights | Feedback `current_target` | **不直接保存** | run内有；artifact只指original target path |

## 3. 已确认问题

### SLM-CTX-001（P0）— Feedback缺失objective/pupil/wavefront上下文

位置：

- `editor.py:_queue_solve/_start_pending`
- `slm_feedback/task.py:625-871`
- `solver.py:443-881`

Editor solve明确传：

- authored `objective_kind`；
- full-resolution `pupil_amplitude`；
- Pattern/base `initial_phase`；
- matching transient spot state。

Feedback initial solve只传：

~~~python
solve_phase(
    current_target,
    initial_phase=incoming,
    iterations=12,
    spot_optimizer_state=spot_optimizer_state,
)
~~~

即objective=auto、pupil=None。`solve_phase(pupil=None)`不是Editor任何开关状态，而是内部hard circular radius 0.9 pupil。Editor On是可调Gaussian amplitude，Off是uniform full raster。

64×64、同一5×7 target、同一zero initial phase的纯数值探针：

~~~text
solver default pupil source   default
Editor Gaussian pupil source  provided
phases equal                  False
circular phase RMS            1.887 rad
max circular difference       3.137 rad
~~~

所以缺失pupil不是metadata小债，而是候选phase发生大幅变化。

operator wavefront也没有保留路径。Adapter last command是完整science phase；Feedback将它当base warm start，随后WGS求出新的完整phase并直接apply。Zernike/steering没有被分离、冻结、再叠回candidate。

建议目标语义：

~~~text
candidate_science_phase
  = wrap(feedback_solved_pattern_base
         + frozen_operator_wavefront_layer)
~~~

Feedback只更新base target weights；objective、pupil和wavefront在Task Start冻结。

### SLM-CTX-002（P0）— X15213的initial last-commanded是虚构zero

`X15213Adapter.__init__`直接执行：

~~~python
self._phase = canonical_phase(zeros)
~~~

它没有向硬件发送zero，也没有读回现有DVI/USB command。DVI没有phase readback；USB初始化也未反演current frame memory。

因此首次session中：

- Editor显示“device phase=zero”，即使硬件不是zero；
- Feedback保存incoming=zero；
- failure path会apply zero，称为restore incoming；
- “Blank/Off不是zero phase”的安全说明反而被restore路径绕过。

`last_commanded`应只表示“本adapter实例成功发出的最后命令”。在尚无命令时必须是unknown/None，或由有证据的transport readback建立；不能为满足Protocol虚构array。

Feedback在incoming unknown时应拒绝Start，除非用户显式提供可恢复science phase artifact，或明确批准full takeover/no-restore语义。

### SLM-CTX-003（P0 reproducibility）— target JSON丢objective

target JSON严格只有：

~~~text
format, version, shape, intensity
~~~

Editor preset已明确：

- Grid/Checkerboard/Text/paint -> spots；
- Gaussian/Flat Top -> image。

但Load target调用 `set_target(load_target(path))`，默认objective=auto。一个小Flat Top（13 positive pixels）的探针：

~~~text
saved explicit objective      image
JSON keys                     format/version/shape/intensity
reloaded auto objective       spots
~~~

同一intensity保存再加载后算法从MRAF变WGS-Kim。Target intensity仍是唯一数值truth，但objective是作者对该target的语义，不是可由shape永远可靠推断的cache。

建议target format v2至少保存 `objective_kind`。Preset draft参数不必保存；materialized intensity + objective足够。旧v1可保守load为auto。

### SLM-CTX-004（P1）— science phase NPZ exact，但不能重建solve context

`save_phase/load_phase`有优点：

- strict members；
- allow_pickle=False；
- canonical little-endian float32；
- strict JSON metadata；
- exact full science phase roundtrip。

Editor Load science phase把整张phase当Pattern、清零/关闭wavefront，因而**exact command replay**成立。

但metadata不足以做scientific reconstruction：

- target intensity/path不在phase artifact；
- pupil numeric center缺失；
- pupil enabled/diameters主要被压成description string；
- Pattern/base phase不单独保存；
- device correction/transport receipt明确excluded；
- loaded metadata不被解释，只作为opaque dict保留。

所以“能重发同一canonical array”不等于“能重新solve/继续feedback/复现physical gray”。文档和UI应明确这两个等级。

若要求Feedback保留operator wavefront，phase metadata必须至少包含numeric pupil center/diameter/enabled、objective、carrier和Zernike coefficients。Task可从incoming science phase减去重建wavefront得到base warm start；无需把transient optimizer state序列化。

### SLM-CTX-005（P0 lifecycle）— correction可绕过device claim改变Feedback映射

Editor只有 `Send` 经 `DeviceUseCoordinator`取得exclusive lease。`load_correction`和`set_correction_enabled`直接调用adapter，不取lease。

Feedback Task运行时已持SLM EXCLUSIVE，但独立Editor窗口仍可：

- load另一张correction；
- enable/disable correction。

这些动作不立即发送hardware frame，却会改变Feedback下一次 `apply_phase(candidate)` 的canonical→gray mapping。因此同一Task内不同candidate可在不同device correction下测量，Task没有记录也不知道。

X adapter的correction lock只保证一次 `_gray`读取一致，不保证整个Task mapping被冻结。

修复语义：

- correction load/enable也必须取得同一device command lease，Task期间明确拒绝；
- Task Start冻结device command context；
- 每次apply返回/记录mapping revision或command receipt；
- candidate artifact记录correction enabled/path/profile/wavelength/orientation和transport evidence。

### SLM-CTX-006（P0 physical replay）— canonical artifact不含device mapping

X15213实际命令链：

~~~text
canonical radians
 -> orientation
 -> half-up phase code
 -> optional correction code add mod 256
 -> wavelength nonlinear LUT
 -> USB/DVI gray raster
~~~

Candidate NPZ只保存第一项canonical radians。若correction/profile/wavelength/orientation改变，重发同一NPZ得到不同gray。

这并不说明phase NPZ错误；它是“science phase pre-correction”artifact。错误是candidate metadata暗示可重复candidate，却没有物理mapping context。

需要用户裁决reproducibility等级：

1. 只保证canonical pre-correction replay；
2. 同时保存device context，可在同device config下重建gray；
3. 保存exact gray/readback sidecar，做transport-level replay。

审计建议至少2；accepted实验artifact最好附USB readback或DVI presented raster evidence，但不能冒充optical proof。

### SLM-CTX-007（P0 UI truth）— Feedback改变device后Editor保持旧本地状态

Editor只在构造时读取一次 `device.last_commanded_phase`。Feedback每candidate/Stop/success都会直接apply phase，Editor没有订阅或refresh。

若Editor已打开：

- target/phase plots继续显示旧local phase；
- wavefront/pupil widgets继续显示旧context；
- hardware实际是Feedback candidate；
- Task结束后下一次Editor Send可把旧phase重新覆盖到device；
- correction status可能是新状态，但phase仍旧，形成混合视图。

DeviceUseCoordinator只防同时command，不同步状态。

建议Task外部command后Editor进入明确stale状态：

- 显示device command changed externally；
- 可只刷新canonical phase并把decomposition标unknown；
- 要恢复target/pupil/wavefront编辑，显式Load/Adopt context artifact；
- 未处理stale前Send必须确认或拒绝。

### SLM-CTX-008（P1 artifact）— Feedback不保存实际candidate target

Feedback每轮更新 `current_target` site intensities，但candidate metadata只保存original `target_path`、fluorescence/history和phase。实际candidate target/weights没有直接持久化。

问题：

- target file可被修改；
- artifact无法直接回答“这张phase对应哪些site target weights”；
- history+算法版本理论可重算，但不是artifact self-description；
- Stop partial时current target更难界定。

每个candidate artifact至少应保存当轮site coordinates + target intensities，或写一个candidate target sibling JSON。Transient fixed far-field phase/optimizer state仍按当前裁决不保存。

### SLM-CTX-009（P1 Stop）— canonical restore受current correction影响

非cancel failure执行 `apply_exact(incoming)`；Stop则保留best/latest durable candidate。phase选择顺序本身有充分测试。

但restore只比较canonical arrays：

- incoming可能是虚构zero；
- incoming可能含不可分解wavefront；
- current correction可能不同于incoming当初apply时；
- adapter没有command receipt/mapping revision。

因此“canonical restored”不等于“physical state restored”。

此外，Stop发生在candidate部分采集时：

- preview record可标 `stage=stopped-partial`和实际shots；
- 已落盘candidate artifact没有被重新写入stopped-partial metadata，仍可能是 `status=applied`；
- device/preview/artifact status因而不一致。

应在Stop retention成功后原子更新同一artifact metadata，记录retained reason、measured shots、device context。

### SLM-CTX-010（P1 contract）— Feedback实际只支持sparse，descriptor接受所有target

`_support`要求positive support pixel count精确等于Calibration sites。Gaussian/Flat Top/dense imported target都会在Task构造失败。

但TARGET_CODEC与Editor target format是通用continuous intensity，descriptor没有显示“sparse only”。根设计还保留dense target相关叙述。

需要明确：

- 当前qCMOS site feedback只支持sparse spot target：则artifact/input UI必须声明并在选择时过滤；
- 或未来支持dense：必须另定义observable，不能用35个离散site冒充dense interior。

审计建议当前明确sparse-only；dense继续由Editor solver与独立terminal validation拥有。

### SLM-CTX-011（P1）— Editor初始化把full science phase当Pattern base

Editor构造：

~~~text
_phase = device.last_commanded_phase
_pattern_phase = _phase
target = fresh default 5x7
pupil = fresh default Gaussian
queue solve(target, initial_phase=_pattern_phase)
~~~

即使last command真实，它可能是 `old Pattern + old wavefront`，也可能对应完全不同target/pupil。新Editor立即把它作为新default target的Pattern warm start。

这不保证最终错误，但违反“warm start只能是accepted Pattern/base”。当decomposition未知时，应：

- 只把last command显示为hardware science phase；
- Pattern base标unknown；
- fresh target solve用deterministic seed或显式adopted context；
- 不把full science phase自动升格为base authority。

### SLM-CTX-012（P1）— target与incoming没有一致性证明

Feedback target来自独立target artifact，incoming来自device。它们可能由不同Editor session、不同target、不同pupil产生。Task允许这种组合并直接warm-start。

数学上任意phase可作initial guess；产品上却不能称其为“继续当前Pattern context”。Artifact也未记录incoming phase ref。

Task Start应冻结并记录：

- target artifact及objective；
- incoming science phase artifact/command receipt；
-两者context是否matching；
- mismatch时是cold solve还是用户批准full takeover。

## 4. 推荐truth分层

不建议把所有东西合成一个巨大Editor state。最小分层：

### 4.1 Target artifact

~~~text
intensity float32
objective_kind: spots | image | auto
shape
~~~

Preset参数不是truth；materialized intensity才是。Objective是author semantic，需同行。

### 4.2 Science context

Plain metadata即可，不要求新增manager/class：

~~~text
pupil:
  enabled
  center_xy
  diameter_xy
wavefront:
  enabled
  carrier_waves_xy
  z4-z11 coefficients
pattern solver:
  objective_kind
  target reference
science_phase:
  exact canonical array
pattern_phase:
  optional exact base array, or reconstruct from science-wavefront
~~~

Transient optimizer state继续不持久化。

### 4.3 Device command context

Adapter/application必须能回答：

~~~text
canonical phase
known/unknown
command revision
device identity/profile/wavelength/orientation
correction enabled/path/revision
transport evidence
~~~

它不是target或science phase的一部分。Correction仍只在adapter。

### 4.4 Feedback Start contract

推荐：

1. target artifact必须sparse spots且带objective；
2. 读取一个known incoming science context，不能从bare last phase猜分解；
3. 冻结pupil与operator wavefront；
4. base solve使用同pupil、objective=spots；
5. 每candidate只更新base target/site weights；
6. apply `wrap(base + frozen wavefront)`；
7. correction mapping在Task lease内冻结；
8. artifact保存candidate target、science context和device command context；
9. Stop按同一context恢复/保留。

若用户不提供context，Task只能有两种诚实模式：

- refuse；
- 明确 `full phase takeover`，说明不保留Editor wavefront且不能restore unknown hardware。

不能默认为“保留”。

## 5. 逐文件 / 类 / 函数裁决

### 5.1 Solver / codecs

| 符号 | 裁决 | 理由 |
|---|---|---|
| `validate_target` | `PASS` | intensity唯一数值表示正确。 |
| preset Grid/Checkerboard/Gaussian/FlatTop/Text | `PASS` | materialize到同一target，无平行preset truth。 |
| `imported_target` | `PASS` | peak normalization明确；仍是intensity。 |
| internal `_pupil` | `PASS WITH DEBT` | notebook/default可留；Feedback不应隐式使用。 |
| `solve_phase` objective/pupil args | `PASS` | API已有需要的上下文；调用方丢失。 |
| spot optimizer state | `PASS` | caller-owned/transient、不序列化正确。 |
| `save_target/load_target` | `REDESIGN` | strict intensity roundtrip正确；缺objective。 |
| `save_phase/load_phase` | `PASS WITH METADATA DEBT` | exact canonical replay正确；context/physical receipt不完整。 |

### 5.2 Common device contract

| 符号 | 裁决 | 理由 |
|---|---|---|
| `canonical_phase` | `PASS` | immutable wrapped float32唯一command表示正确。 |
| `SlmAdapter.apply_phase` | `PASS WITH DEBT` | canonical boundary正确；缺command receipt。 |
| `last_commanded_phase` | `REDESIGN` | 必须允许unknown并区分commanded/physical mapping。 |
| `bind_slm` | `REDESIGN` | 当前强迫adapter在未command时提供虚构phase。 |
| VirtualSLM | `PASS` | world是command truth，initial nominal phase真实已知。 |

### 5.3 X15213

| 符号 | 裁决 | 理由 |
|---|---|---|
| profile/LUT/correction load helpers | `PASS` | owner正确、验证充分。 |
| `X15213Adapter.__init__` phase初始化 | `REDESIGN` | 未read/send却记zero。 |
| correction properties/load/enable | `PASS WITH LIFECYCLE FIX` | mapping owner正确；需lease/revision。 |
| `_gray` | `PASS` | orientation→phase code→correction→LUT顺序正确。 |
| `apply_phase` | `PASS WITH RECEIPT DEBT` | USB readback与settle后commit正确；只返回canonical不足artifact。 |
| `last_commanded_phase` | `PASS after first known command` | command成功后truth正确；initial不正确。 |
| close | `PASS` | 不擅自blank/zero正确。 |

### 5.4 Editor

| 符号 | 裁决 | 理由 |
|---|---|---|
| `SlmEditorControl.__init__` | `REDESIGN` | default target/pupil与device phase无context；full science被当base。 |
| preset popup/materializers | `PASS` | objective选择正确，Apply单一提交。 |
| pupil arrays/apply/toggle | `PASS` | Gaussian/uniform与Zernike disk职责清楚。 |
| correction controls | `REDESIGN` | 绕过device lease，可改变Task mapping。 |
| wavefront compose | `PASS` | Pattern+carrier+Z4-11边界正确。 |
| `set_target` | `PASS` | target/objective/revision与state invalidation正确。 |
| `set_phase` | `PASS AS EXACT LOAD / DEBT` | exact phase replay正确；target仍旧、decomposition丢失需明确stale。 |
| solve queue/finish | `PASS` | latest-only与matching optimizer context正确。 |
| `save_target` | `REDESIGN` | objective丢失。 |
| `save_phase` | `PASS WITH CONTEXT ADDITIONS` | exact science array正确。 |
| `send/_send_phase` | `PASS` | stale guard、exclusive lease、adapter confirmation正确。 |
| close | `PASS` | 等worker、不改hardware phase正确。 |
| external command sync | `MISSING` | Task改变device后无stale/refresh机制。 |

### 5.5 Feedback

| 符号 | 裁决 | 理由 |
|---|---|---|
| descriptor target input | `REDESIGN` | generic target codec却实际sparse-only；无science context input。 |
| `_support` | `PASS for sparse` | registration检查直接；需与descriptor一致。 |
| `_updated_target` | `PASS` | site intensity比例更新与total normalization正确。 |
| `SlmFeedbackTask.__init__` | `REDESIGN` | 没有objective/pupil/wavefront/device context。 |
| `_candidate_metadata` | `REDESIGN` | 缺actual target、incoming context、device mapping。 |
| `_publish_candidate` | `PASS WITH DEBT` | device/preview phase一致；record context不完整。 |
| `_apply_exact` | `PASS CANONICAL / DEBT` | exact canonical确认正确；没有physical receipt。 |
| `_measure` | `PASS WITH PULSE DEBT` | exact grouped measurement正确，见04c。 |
| initial solve | `REDESIGN` | auto/default pupil/full science warm start。 |
| optimizer reuse/update loop | `PASS after context fix` | fixed spot phase与8-iteration update正确。 |
| success terminal apply/save | `PASS WITH CONTEXT FIX` | canonical/artifact ordering正确。 |
| Stop retention | `PASS selection / REDESIGN artifact` | best/latest/durable顺序正确；partial metadata与physical context不一致。 |
| failure restore | `REDESIGN` | bare canonical incoming不等于known physical restore。 |

## 6. 测试裁决

| 测试组 | 裁决 |
|---|---|
| Editor latest-only solve/stale Send | `KEEP`；局部revision正确。 |
| Editor optimizer state/pupil invalidation | `KEEP`；caller context规则有效。 |
| Pupil/Zernike numerical composition | `KEEP`；数学边界充分。 |
| Science phase exact roundtrip | `KEEP`；证明canonical replay，不证明context replay。 |
| Editor file/send/lease/close | `KEEP WITH ADDITIONS`；target只测文件存在，不测objective reload。 |
| Correction/LUT/orientation/USB readback | `KEEP`；device mapping局部正确。 |
| Correction enable affects next apply | `KEEP WITH TASK-LEASE CASE`；缺Feedback并发。 |
| Feedback geometry/math/Welford | `KEEP`。 |
| Feedback candidate/validation/Stop tests | `KEEP WITH PHYSICAL CONTEXT CASES`；fake SLM总有known phase且无correction/wavefront。 |
| Virtual feedback | `KEEP WITH EDITOR CONTEXT CASE`；从world nominal直接开始，未经过Editor pupil/Zernike。 |
| X15213 initial last command | `MISSING`；当前tests接受软件zero但不证明hardware。 |
| Target objective roundtrip | `MISSING`。 |
| Editor open whileFeedback modifiesdevice | `MISSING`。 |
| Partial Stop artifact status | `MISSING`。 |

必需old-red：

1. explicit image target save/load仍为image；
2. Feedback solve spy收到与Editor相同objective/pupil；
3. candidate applied phase始终等于solved base + frozen operator wavefront；
4. X15213未command时last phase unknown，Feedback拒绝或要求artifact；
5. correction toggle在Task lease期间被拒；
6. candidate artifact记录mapping revision/actual target；
7. Stop partial后artifact、preview、device status一致；
8. external Feedback command使open Editor进入stale，旧Send不静默覆盖；
9. sparse-only target在descriptor/UI阶段明确拒dense。

## 7. 文档冲突

| 文档目标 | 当前实现 | 结论 |
|---|---|---|
| objective是author intent | target JSON不保存，load用auto | 实现未满足。 |
| Editor pupil context用于solve | Editor满足；Feedback丢失并用第三种default pupil | 跨边界未满足。 |
| warm start只能Pattern/base | Editor/Feedback可把full science last phase当base | 未满足。 |
| operator wavefront不属于feedback估计 | Feedback不拟合coefficients，但直接替换完整phase，未保留layer | 文字与效果不等价。 |
| correction只在adapter | 实现满足owner；Task不冻结/记录mapping | lifecycle缺口。 |
| Task冻结target/resources/calibration/incoming | incoming只是bare/可能虚构canonical array；无context | 不完整。 |
| Stop恢复incoming或保留candidate | canonical array选择有测试；physical mapping与partial artifact未对齐 | 仅部分满足。 |
| Dense target继续solver | Feedback构造要求support count等于sites | 需明确Editor solver与Feedback边界。 |

## 8. 需要用户裁决

1. **Feedback是否必须保留Editor wavefront？**
   A. 只更新Pattern base，再叠加frozen steering/Z4-Z11；
   B. 明确接管完整science phase。
   审计建议A；当前UI/文档已经把wavefront定义为operator layer。

2. **Feedback context来源**
   A. 读取当前open Editor内存；
   B. 使用显式science-context artifact；
   C. bare target + device phase并声明full takeover。
   建议B；A耦合UI/lifecycle，C只能作为明确高级模式。

3. **Target JSON v2**
   是否加入objective_kind？建议加入；pupil/Zernike不属于target，放science context。

4. **Unknown initial hardware phase**
   X15213没有readback时，是否禁止Feedback直到一次known Send/phase artifact？建议禁止，不虚构zero。

5. **Candidate artifact reproducibility等级**
   只canonical、device-context可重建gray、还是exact gray sidecar？建议至少device context，accepted artifact附transport evidence。

6. **Correction操作权限**
   load/enable是否也进入DeviceUseCoordinator？建议是，Task期间冻结。

7. **Editor外部command后的行为**
   自动adopt bare phase、标stale等待用户、还是关闭Editor？建议标stale并显示device canonical，decomposition unknown。

8. **Feedback target范围**
   当前明确sparse-only，还是要设计dense observable？建议本Task明确sparse-only，dense另行验收。

## 9. 推荐实施顺序（未实施）

1. 用户先裁决wavefront preservation、context source、unknown incoming。
2. 加objective target roundtrip与pupil-context old-red。
3. 修SlmAdapter known/unknown last-command语义；Feedback拒绝unknown。
4. 扩充science phase metadata numeric context；Feedback冻结并重建wavefront。
5. Feedback显式objective=spots、传frozen pupil，只更新base。
6. correction controls纳入lease，adapter提供mapping revision/command receipt。
7. candidate保存actual target weights与device context；修partial Stop metadata。
8. Editor监听/检查external command revision并进入stale。
9. 明确descriptor sparse-only，删除dense暗示。
10. 最后更新旧文档/tests；不得用metadata文字掩盖phase仍未保留。

## 10. 探针边界

- Python进程先打印当前checkout的 `zou_lab_control_v2`、`zlc_atom` 路径。
- Pupil与objective探针只运行64×64数值solver，不连接设备。
- X15213 initial/correction结论来自adapter源码与mocked transport tests；未宣称optical proof。
- 未打开Editor窗口、未写workspace artifact、未调用真实SLM。
