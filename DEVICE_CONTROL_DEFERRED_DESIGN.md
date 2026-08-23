# Device Control — Deferred Design Record

状态：`DEFERRED / NOT IMPLEMENTED`

记录日期：2026-08-23

本文保存已经确认、但当前不实施的Device Control设计。它不是当前产品行为、
不是兼容承诺，也不授权自动开始实现。未来只有在operator明确恢复此工作后，
才按当前代码和真实设备SDK重新核验并进入实施计划。

## 1. 目标与已确认边界

- Device Control始终列出所选device声明的全部可调参数，而不是只为Pylon gain
  建立特例。
- Device被Logic占用时，权限按**字段**而不是按整台device判断：Logic设置、冻结或
  依赖的字段不可修改；Logic未使用的字段可在operator显式接受风险后修改，但前提是
  device adapter确认该字段支持运行中写入。
- 风险解锁不能越过Logic字段所有权、字段依赖关系或hardware/SDK限制，也不能造成两个
  thread直接并发调用同一个device handle。
- 第一阶段不因为一次live tune重启Logic、停止Task、建立新Signal generation、清空Plot，
  或重置history。
- Provenance只描述实际发生的设置变化，不作为阻止采集或自动重启的控制机制。
- 不建立跨实验、跨Logic或伴随TaskConsole生命周期无限增长的设备参数历史数据库。

## 2. 字段级权限模型

Device Manager不得按device type或字段名硬编码权限。权限由两个独立事实共同决定：

1. **Logic field claim**：每个活跃Logic在取得device使用权时，声明它会设置、冻结或依赖的
   字段集合。
2. **Adapter write capability**：device adapter声明每个tunable field能否在当前hardware状态下
   live write，以及与哪些字段构成不可分割的依赖组。

最终规则：

| 当前状态 | 字段行为 |
| --- | --- |
| 无活跃Logic使用device | 正常允许修改，仍服从hardware状态限制 |
| 任一活跃Logic claim该字段 | 始终锁定，接受风险也不能修改 |
| Logic未claim，adapter支持live write | 默认锁定；接受风险后允许修改 |
| Logic未claim，adapter不支持当前状态live write | 继续锁定，必须先停止占用者 |

补充不变量：

- 多个Logic同时使用同一device时，受保护字段是所有活跃claim的并集。
- Logic claim必须覆盖显式authoring参数以及未显示在表单中、但算法或driver依赖的隐式字段。
- 有耦合的字段按dependency group闭包保护。例如ROI的origin/shape若不能独立改变，则其中任一
  字段被claim时整组锁定。
- 活跃owner集合、Logic generation或device session发生变化时立即重算权限；尚未执行且新近
  变为protected的pending write必须取消。
- 如果Logic依赖了某字段却没有声明，这是Logic contract缺陷，不允许由Device Manager中的
  camera/device名称特判来掩盖。

### Pylon示例

Camera Measurement当前显式设置或依赖的ROI、exposure、trigger/acquisition/readout相关字段
应进入它的field claim。它没有gain参数，也不设置gain，因此`gain_db`不因“整台camera正在被
占用”而自动成为protected field。若Pylon adapter确认grabbing期间可写gain，operator接受风险后
可以调整它。未来若某个Logic把gain加入自己的working point，gain会自然进入该Logic claim并锁定；
这不是“只有Pylon gain可以live tune”的特例。

## 3. Device Control交互

每个字段使用同一套基本行：

```text
Current | Desired editor | Live apply | Apply | Status
```

- 复用现有Fluent spin box、switch、button和状态控件。
- `Current`显示adapter/device class掌握的实际值；`Desired`是尚未提交的operator输入。
- 打开Control时读取一次，成功Apply后用authoritative return/readback刷新；提供显式Refresh。
- 不复制Confocal-GUIv2的200 ms硬件轮询，也不开第二个PyVISA/SDK handle去旁路询问设备。
- `Live apply`只改变提交手势，不改变字段权限。滚轮或连续编辑采用latest-only合并，建议
  50–100 ms debounce，避免把每个中间数值都发送给hardware。
- Header显示当前占用者和原因。风险解锁只解锁“未被claim且当前可live write”的字段；
  protected字段继续disabled并显示具体owner/reason。
- 风险接受绑定到当前`device session + active-owner set`。device重连、重新Init或owner集合变化后
  自动失效，不永久留在UI或draft中。

## 4. 命令并发边界

字段级允许修改不等于允许绕过device session直接碰SDK对象：

- 所有tune仍进入同一个session-owned、串行device command路径。
- Logic acquisition和Device Control tune不能在不同thread中同时直接调用同一个device handle。
- adapter负责给出可写状态并返回实际生效值；requested value与effective value都要保留。
- 若adapter提供明确safe point，live write在相邻acquisition unit之间执行。若无法证明精确边界，
  不能把交界数据伪称为完全属于旧或新设置，必须标记为transition/mixed。
- risk override不隐式Stop/Restart Logic，不shutdown/reinitialize device，也不夺取整个device的
  exclusive owner。

## 5. 轻量Provenance

### 5.1 Identity

每次Device Init到Shutdown/Reconnect构成一个`device_session_id`。该session只常驻保存：

- 当前有效设置；
- 当前`settings_epoch`整数。

只有成功并且实际值发生变化的tune才增加epoch。no-op、失败请求或只修改Desired UI不得增加。
`settings_epoch`是运行时状态序号，不是文件format version。

### 5.2 记录内容

一次成功变化的最小事实为：

```text
device_session_id
settings_epoch
monotonic/wall timestamp
field
requested value
previous effective value
new effective value
verified/readback status
actor = Device Control risk override
active Logic owners
```

Camera/source publication只携带很小的`(device_session_id, settings_epoch)`引用；不得逐frame复制
完整参数快照。一个epoch对应的完整状态或delta只保存一次。

### 5.3 生命周期与裁剪

- 没有Logic使用该device时，参数改变只覆盖session当前状态；不保存旧epoch历史。
- Logic开始时保存一份自己的起始设置快照，仅记录其运行期间、且与它保留的数据有关的变化。
- 旧数据离开rolling/history保留窗口后，对应且没有其他引用的epoch映射可从内存释放。
- 保存Figure、Dataset或Task artifact时，只复制该artifact实际引用的epoch映射；完成保存后不要求
  session继续持有它们。
- device重新Init产生新session identity并从epoch 0开始。
- epoch计数器可以增长，但增长的是固定大小整数，不代表历史对象随TaskConsole运行时间累积。
- 若一个长时间Logic确实保留了跨越大量人工调整的数据，精确provenance必须保留对应delta；
  连续相同状态按区间压缩，不能为了省记录而谎报数据使用同一working point。

### 5.4 Derived data

- 同一设置下的数据只引用一个epoch。
- average、rolling或其他派生结果若跨越多个epoch，携带其实际epoch集合/区间或`mixed`事实，
  不阻止计算，也不假装只有一个设置。
- provenance变化不创建新的Signal generation，不触发publication restart，也不清空Panel/history。
- 对app外部直接修改hardware的行为，若adapter没有可靠readback/notification，系统不能宣称已经
  观测；显式Refresh只能更新当前状态，不能反造未知历史。

## 6. 建议实施顺序

1. 盘点所有device adapter的`tunable_fields`，为字段补充current readback、live-write capability
   和dependency group；不得从Pylon特例反推通用接口。
2. 扩展DeviceUse claim，使Logic在actual Start时声明protected fields；逐个审计Logic显式设置和
   隐式依赖字段。
3. 在DeviceUseCoordinator提供只读的field-policy/blocker projection，按所有活跃claim计算并集。
4. 建立session-owned串行tune命令、effective readback/current cache和latest-only pending write。
5. 重做Device Control字段行与header：Current、Desired、Live apply、Apply、Refresh、owner reason
   和风险解锁。
6. 接入轻量`device_session_id/settings_epoch`、active-Logic change capture、publication引用和
   artifact按需复制/裁剪。
7. 最后在真实Pylon及其他实际可调device上逐字段核SDK行为；未验证字段不得宣称live-safe。

## 7. 验收条件

- Device Control展示device声明的所有tunable fields，不因Logic占用整台device而全部消失。
- Camera Measurement运行时，其ROI/exposure等protected fields无论是否接受风险都不能修改。
- Pylon gain在未被Logic claim且SDK确认live-write时，可在风险解锁后修改，Camera Measurement不中止、
  不重启、Panel不清空。
- 新增一个显式使用gain的Logic后，gain无需修改Device Manager代码即可自动锁定。
- 所有SDK调用仍串行；连续滚轮输入只提交最新值，UI线程不执行blocking I/O。
- risk unlock在owner或device session变化时自动失效，pending protected write不会漏执行。
- 没有活跃Logic时反复调整几天只保留当前状态，不积累session-wide历史。
- 有活跃Logic时，变更前后数据能解析到正确epoch；跨epoch派生结果明确显示mixed/range。
- 保存artifact只包含实际引用的设置事实，不包含整次TaskConsole session历史。
- tune不创建Signal generation、不触发publication ownership错误，也不改变无关Logic生命周期。

## 8. 明确不做

- 不做“只有Pylon gain”或“Camera Measurement按字段名特判”的补丁。
- 不允许风险按钮解锁Logic claim字段或hardware不支持的写入。
- 不做每帧完整device snapshot、全局永久epoch ledger或TaskConsole级历史数据库。
- 不做200 ms设备轮询、第二hardware handle或UI thread直接I/O。
- 第一阶段不做settings变化自动Stop/Restart Task、自动take-control或强制切换Signal generation。
- 不为旧draft、旧UI状态或未来可能的字段保留compatibility alias；实施时按当时current contract
  一次完成并清理被替代路径。
