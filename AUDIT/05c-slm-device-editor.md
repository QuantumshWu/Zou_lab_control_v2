# 05c — SLM device / X15213 / Editor 真机路径深审

状态：本子阶段完成
审计基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`SlmAdapter`、VirtualSLM、Hamamatsu X15213 profile/LUT/correction/orientation、USB/DVI transport、discovery/binding、plugin-local Editor的target/pupil/wavefront/worker/lease/Send/close/artifact，以及直接测试。solver数值算法和feedback控制律只审查与device/editor的边界，不在本文重做。
约束：只读源码、Git历史、官方资料、现有测试和无硬件隔离探针；未修改production、tests、旧文档或硬件。

官方外部事实只采用Hamamatsu一方资料：

- [X15213 series datasheet](https://lcos-slm.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/lpd/x15213_E.pdf)
- [X15213-07 specifications](https://www.hamamatsu.com/us/en/product/optical-components/lcos-slm/wide_wavelength_bandwidth_type/X15213-07.html)
- [Hamamatsu phase-modulation measurement](https://lcos-slm.hamamatsu.com/us/en/learn/technical_information/characteristics.html)
- [Hamamatsu drive timing](https://lcos-slm.hamamatsu.com/us/en/learn/technical_information/characteristics/drive-timing.html)
- [Hamamatsu time response](https://lcos-slm.hamamatsu.com/us/en/learn/technical_information/characteristics/time-response.html)

## 1. 结论先行

当前代码已经有一条结构清楚的device-independent主线：

~~~text
continuous target intensity
  -> solve / compose canonical float32 radians [0, 2π)
  -> explicit Send / Task apply
  -> X15213 orientation
  -> 8-bit phase code
  -> optional native correction code mod 256
  -> wavelength LUT -> drive gray
  -> USB active frame memory OR DVI 1280x1024 raster
~~~

以下部分做得正确，应保留：

- `SlmAdapter`窄公共surface与`canonical_phase()`；
- vendor correction/LUT只在real leaf，science phase artifact不混入hardware gray；
- phase code先half-up、correction再mod 256、最后LUT；
- USB写`1272×1024`、切slot、逐字节frame-memory readback；
- DVI右8列置零并检查window client geometry没有DPI缩放；
- Editor target/pattern/wavefront分层、latest-only solver、stale Send拒绝；
- Send在独立worker中取得短命exclusive lease，Qt owner不被80 ms command阻塞；
- Editor close不调用SLM apply/blank/close，等待自己的solve/command和plot hosts退役。

但“真机链已经完成”的结论不成立。当前最高风险不是FFT或Qt，而是device truth和物理证据：

1. Real adapter初始化时既没有send，也没有读取当前displayed phase，却把全零数组报告成`last_commanded_phase`。Feedback失败恢复时可能把这个**虚构incoming**真的写上硬件。
2. USB/DVI在“硬件可能已经改变”之后仍可能抛错；adapter保留旧`_phase`，于是软件truth与物理输出分裂。
3. Editor的runtime correction load/toggle不取得DeviceUse claim，可在Feedback exclusive run或Send中途改变hardware mapping；该变化也不进入artifact或saved installation config。
4. serial profile缺型号后缀、测量来源、日期、温度、误差和response time；当前256点曲线几乎是一条直线，无法证明是该serial的逐点实测。
5. 默认settle 50 ms无device依据。官方X15213各型号fall time约17–155 ms；适合约800/1064 nm的型号公开值为73–85 ms量级。
6. “连续二维unwrap”实际只是先X后Y两次`np.unwrap`，对二维绕相不具旋转/转置不变性。
7. DVI只能证明Tk client几何；不能证明GPU/ICC/HDR/dithering后的8-bit线性gray、实际scanout、active区左右位置、光学orientation或LC稳定。关闭DVI presenter还必然撤掉当前输入，不能兑现“session close保持phase”。
8. Target JSON丢失`objective_kind`与pupil context；保存一张全尺寸target在Qt owner实测约1.68 s，会直接冻结Editor。

因此当前裁决是：

- common adapter和Editor核心：`KEEP + CRITICAL FIXES`；
- USB transport：`software byte path substantially implemented, hardware acceptance open`；
- DVI transport：`EXPERIMENTAL / operator-confirmed endpoint only`；
- profile/LUT/correction wavelength science：`UNVERIFIED CALIBRATION`；
- “device current phase”、failure recovery、runtime correction并发：`P0 REDESIGN`。

## 2. 四层证据必须分开

当前代码和文档经常把以下四件事压成一句“Phase sent”：

| 层 | 当前能否证明 | 证据 |
|---|---|---|
| Canonical command | 能 | immutable float32 phase与exact array compare |
| Controller/frame-memory bytes | USB能较强证明；DVI只证明app交给Tk的array | USB `Check_Disp_IMG`; DVI fake presenter/window ack |
| 输入链实际显示/驱动 | 未证明 | DVI scanout/GPU pipeline；USB display-slot drive state均无独立硬件证据 |
| LC optical phase已到目标并稳定 | 未证明 | 需要实验机interferometer/camera和settle测量 |

Hamamatsu说明DVI frame以60 Hz写入controller frame memory，controller以240 Hz读取phase image，LC以120 Hz AC drive，而真正phase update受较慢LC response限制，只约10至数十Hz。[官方drive timing](https://lcos-slm.hamamatsu.com/us/en/learn/technical_information/characteristics/drive-timing.html)直接说明：Tk `root.update()`返回不是光学settled proof。

## 3. 已确认缺陷

### SLMDEV-001（P0 truth）— Real adapter虚构初始`last_commanded_phase = zero`

位置：

- `devices/slm/device.py:54-69,88-106`；
- `devices/slm/device_types.py:693,711-713`；
- `devices/slm/editor.py:94-99`；
- `nodes/slm_feedback/task.py:625-626,896-936`。

`X15213Adapter.__init__()`只连接controller/验证display；没有apply zero，也没有读取并反解当前displayed frame，却直接设置：

~~~python
self._phase = canonical_phase(np.zeros(_SHAPE_YX), _SHAPE_YX)
~~~

`bind_slm()`还强制每个adapter在安装时提供一张canonical immutable `last_commanded_phase`，使“unknown”在contract上无法表达。结果：

- 新Editor把未知硬件状态显示为来自device的zero phase；
- SLM Feedback把zero保存为`incoming`；若首次candidate前或普通异常失败，会调用`apply_phase(incoming)`，把未知旧状态替换成真正zero；
- archive/状态检查无法区分“本进程从未command”与“zero已成功command”。

VirtualSLM不同：`SimulationWorld.commanded_phase`是真实simulation state。因此同一contract在virtual是真相，在real是占位。

必须由用户选择：

1. `last_commanded_phase`允许`None/unknown`，首次Feedback没有可恢复incoming时拒绝或采用显式用户选择；
2. Init明确发送一个已知phase并如实告诉用户这是hardware mutation；
3. USB若官方API能可靠读当前slot和image，读取drive bytes并保存“drive state known”，但因LUT/correction可能多对一，不能假装总能唯一反解science phase；DVI仍无法从display endpoint恢复authoritative phase。

当前“Init不发phase”与“last phase永远存在”两项不能同时成立。

### SLMDEV-002（P0 truth）— apply失败后physical可能已变，software仍声称旧phase

USB顺序是：

~~~text
Write_FMemArray(new bytes)
  -> Change_DispSlot(0)
  -> Check_Disp_IMG
  -> settle
  -> self._phase = canonical
~~~

若Write成功而Change失败，frame memory已变；若Change成功而readback mismatch/SDK read失败，display slot很可能已经指向new bytes。现有`test_usb_writes...`恰好制造后一情况：fake SDK的`display`已被新frame覆盖，但adapter故意保持旧`last_commanded_phase`。测试只断言software没commit，没有审查physical divergence。

DVI同类问题更明显：presenter先`label.configure/root.update()`，再检查geometry；geometry检查失败时新frame可能已经scan out，caller收到异常，`_phase`仍旧。present timeout也不会从queue撤销command；frame可在caller判失败后迟到显示。

这不是普通transaction可rollback的问题。硬件side effect之后的失败必须把command state标成`unknown/ambiguous`，并保存哪一步成功；不能继续返回旧phase作为authoritative truth。Editor现在只显示异常字符串，Feedback会按旧truth执行恢复，可能再次覆盖。

### SLMDEV-003（P0 concurrency）— correction mapping绕过DeviceUse ownership

位置：

- `device_types.py:742-779`；
- `editor.py:521-550,975-992`；
- `editor.py:939-953`只给Send取得claim。

`load_correction()`与`set_correction_enabled()`直接调用device，不经过`session.device_use`。内部`Lock`只保证array/path/boolean不会撕裂，不保证实验lifecycle：

- Feedback Task可持有SLM EXCLUSIVE数分钟，Editor仍能加载/切换correction；下一candidate在同一science phase算法下使用另一hardware plant；
- 用户点击Send后、command worker执行`_gray()`前可改变correction，clicked command只冻结canonical phase，没有冻结device mapping；
- correction控件在`_command_active`时也不禁用；
- candidate NPZ/history不记录这次mapping改变。

这直接违背Task开始前冻结resources/device mapping的设计目标。正确owner仍是现有`DeviceUseCoordinator`，无需新增manager：runtime correction mutation与Send一样应取得短命exclusive claim，Task active时拒绝；或者禁止Editor runtime修改，只允许stopped installation config。

### SLMDEV-004（P0 calibration）— serial profile没有可审计的实测来源或device subtype

`profiles/LSH0804382.json`只保存generic `model="X15213"`、serial、两个wavelength和256个phase值。它没有：

- 完整type number（例如`-02/-07/...`）；
- measurement date/temperature/setup；
- raw measurement或source document；
- uncertainty/repeatability；
- operator/instrument；
- response rise/fall time；
- correction map identity/orientation。

Git显示该profile在`46f8cee`与实现一起一次性加入，没有独立校准artifact或来源说明。数值探针：对256点做最小二乘直线，`R²=0.9999965003`、最大残差仅`0.00466π`、相邻step四舍五入后只有5种。这可能是真实factory-linear response，也可能只是把旧`two_pi_gray`扩成256点；仓内证据无法区分。现有测试标题“without a hard-coded level”只证明JSON没有名为`two_pi_gray`的字段，不证明曲线是serial-specific measurement。

官方资料确认X15213不同type的wavelength与response差异很大：datasheet列出rise约5–38 ms、fall约17–155 ms；X15213-07公开为约10/80 ms。当前profile没有type，adapter默认settle固定50 ms且所有phase transition同值，无法保证最慢方向达到90%。

此外，profile被锁在installed Python package：`_load_profile()`只接受local basename，不能选择workspace/device calibration path。新serial没有package内同名JSON时，USB discovery会在读到serial后抛错，并可能连DVI candidates一起使descriptor discover失败。实验装置calibration数据放在generic source distribution层级不合理，建议`MOVE`到workspace/installation-owned device artifact；package只保留schema/loader。

### SLMDEV-005（P1 truth）— schema仍有重复/失联配置

- `serial` authoring只允许blank或等于profile serial；实际USB `_connect_usb()`始终用profile serial，所以这是profile内容的重复输入。
- profile的`default_wavelength_nm`只在USB discovery使用；manual Add选择另一个profile时，AuthoringSchema仍默认硬编码852 nm，未从profile投影。
- DVI模式隐藏`serial`/`sdk_directory` form，但使用profile serial尝试USB controller probe；endpoint-only时实际head与profile无法证明。
- `settle_seconds`不随profile/type/wavelength变化。
- runtime `load_correction()`改变adapter state，却不更新installation draft；重启后回到旧config。
- `correction_available` property没有production consumer，Editor只duck-check两个methods；当前为test-only surface。
- `_dvi_controller_mode_proven`只被tests读取，操作者和run record看不到endpoint-only/proven状态。

这些状态需要一个public device snapshot/command receipt，而不是更多私有属性。

### SLMDEV-006（P0/P1 science）— correction wavelength conversion不是一般二维unwrap

`_scale_correction_wavelength()`执行：

~~~python
unwrapped = np.unwrap(wrapped, axis=1)
unwrapped = np.unwrap(unwrapped, axis=0)
~~~

这是固定路径的separable一维unwrap，不是一般二维phase unwrapping；有residue/noise/二维wrap结构时结果依赖轴顺序。隔离probe用一个wrapped vortex correction：

~~~text
convert(A).T vs convert(A.T)
different pixels            4160
max circular code difference 21
~~~

物理wavelength conversion不应因把map转置后再转回来而改变。现有测试只有沿X的完美sawtooth，天然让separable算法通过。

另有两个未验证假设：

- filename中的最后一个`NNNnm`就是source wavelength；非`CAL_<serial>_...`文件不核head；
- 无wavelength文件名的map被视为已经适用于current wavelength。

建议先确认vendor correction BMP的真实编码/metadata。若确实需要换波长，应使用经过验证的二维unwrap或保存unwrapped physical map；不能让filename承担科学truth。

### SLMDEV-007（P0 hardware）— USB ctypes ABI与DVI mode switch未被真SDK证明

仓内没有vendor header/manual/DLL；`_load_sdk()`对`Open_Dev`、`Mode_*`、`Write_FMemArray`、`Check_Disp_IMG`等没有设置`argtypes/restype`。Python fake只能证明本项目自己假定的签名相互一致，不能证明64-bit calling convention、pointer width、return semantics或SDK version。

Mode切换还有直接逻辑缺口：

- USB path调用`Reboot`不检查return，但随后会reopen并核mode，最终仍能发现失败；
- DVI path调用`Mode_Select(DVI)`后也不检查`Reboot`，立即return True，不等待reconnect、不重新读serial/mode，却将`_dvi_controller_mode_proven=True`。

因此该变量最多表示“Mode_Select call返回成功”，不表示controller重启后确在DVI mode。测试fake在`Mode_Select`时直接改`mode`，掩盖真实reboot边界。

`_close_usb_after_probe()`还吞掉所有close错误；在serial/mode probe失败或DVI prepare时可能遗留controller claim，却只报告primary error。正式adapter close的retry设计很好，probe path不应降低标准。

### SLMDEV-008（P0/P1 DVI）— exact window geometry不等于exact device gray

DVI presenter做对了两件重要事情：per-monitor-v2 DPI awareness与native client rectangle精确`1280×1024`。但它仍未证明：

- Windows compositor/GPU scaling、ICC/gamma、HDR、limited/full range、temporal/spatial dithering没有改8-bitgray；
- Tk/PIL到DVI scanout的pixel bytes与array一致；
- frame已经越过60 Hz scanout/controller frame memory；
- active1272列确实位于左侧而非其他alignment；
- flip_x/flip_y与反射光路/qCMOS registration正确；
- topmost window未被系统overlay、display mode change或power management替换。

官方只公开SXGA 1280×1024@60Hz与active1272×1024；公开datasheet没有支持“左侧0:1272”这一具体alignment的文字证据，必须在实验机确认。

DVI的close语义更根本：phase输入就是live window。`X15213Adapter.close()`销毁presenter，桌面/black/其他内容会替代最后phase。Editor close不关闭device，所以Editor语义成立；session/process close却不可能像USB frame memory那样承诺“commanded phase原样保留”。要么文档明确DVI session close必然撤销输入，要么使用独立持久presenter/service；不能继续跨transport使用同一句保证。

presenter startup/command timeout也会丢失owner：startup等5秒超时后thread可迟到创建window；present等5秒超时后queued frame可迟到显示。当前没有cancel/close handle可回收这些late side effects。

### SLMDEV-009（P1 Editor state）— worker/lease正确，但local draft与device truth没有同步协议

做得正确：

- capacity-one pending solve、revision cancellation、只接纳latest optimizer state；
- pupil/support/objective改变清state；
- target/pupil solve与command使用不同single workers；
- Send冻结canonical array、取得exclusive claim、exact compare returned phase；
- close guard非阻塞重试，等待solver/command/三个plot hosts。

仍有以下状态缺口：

1. Editor只在构造时读一次device phase。Feedback或另一个caller后续apply，Editor没有command revision/subscription，也不显示“draft differs from device”。
2. close/reopen会把device full science phase当`pattern_phase`、清空wavefront decomposition，随后自动solve默认5×7 target；UI很快显示一张未发送的新phase，而非当前device command。状态栏说hardware unchanged，但没有独立hardware preview。
3. command active时用户仍可load target/phase、改wavefront/correction。clicked canonical phase被正确冻结，但completion只显示“Phase sent”，没有说明当前draft已更新且device收到的是旧draft。
4. USB SDK handle在installation thread创建，却可在Editor command worker、Task worker使用并在session thread关闭；vendor SDK是否允许跨线程handle usage未由header/真机证明。
5. USB SDK call没有timeout/cancel。若vendor call挂死，non-daemonThreadPool worker使Editor/session无法真正清理；Qt只会持续20 ms retry。

建议仍保持一个Editor class，不新增controller framework；只需引入device command revision/known state、冻结command mapping、清楚投影draft-vs-device状态。

### SLMDEV-010（P1 artifact）— phase roundtrip精确，但target/project与hardware provenance不完整

做得正确：

- phase NPZ严格只含`<f4 phase + scalar JSON metadata`，`allow_pickle=False`；
- load science phase清zero wavefront，保证exact phase roundtrip；
- hardware correction不污染science phase metadata。

缺口：

- Target JSON只有shape/intensity，丢失`objective_kind="spots|image"`。从Grid/Text保存再加载会回到`auto`，重新引入实现明确想删除的业务猜测。
- target artifact不保存pupil enabled/center/diameter；phase metadata的`input_pupil`字符串只含diameter，不含center。无法从target重现solve context。
- preset draft可以不保存，但materialized target的objective不能丢。
- Send没有command receipt：device key/identity、profile、wavelength、orientation、correction path/enabled、transport、drive gray/readback、settle、timestamp、known/ambiguous outcome均未保存。
- SLM Feedback candidate metadata也只有named device key，不含上述mapping；runtime correction改变后同一NPZ不再对应同一physical command。
- runtime加载的新correction不进入saved installation config，session重启不可复现。

science phase artifact继续保持device-independent是正确的；缺的是旁边的run/device command provenance，不是把vendor bytes塞进NPZ phase payload。

性能probe：保存一张`1024×1272`、绝大多数为zero的target为readable JSON，文件约6.64 MiB，save约1.68 s、load约0.21 s。Editor在Qt owner同步调用save/load，因此Save Target会肉眼冻结。严格JSON格式是产品决定，但I/O应移出owner或使用稀疏/array artifact；当前不适合称交互完成。

### SLMDEV-011（P1 layer）— concrete plugin反向依赖未声明的Workbench

`zlc_atom.devices.slm.editor._send_phase()` runtime import `zlc_workbench.device_use.DeviceClaim`。`zlc_atom`的pyproject不依赖`zlc-workbench`，而`zlc_workbench`本身依赖/组合`zlc_atom`，形成未声明反向环。现有lazy-import测试只证明catalog discovery不提前import Qt/Workbench，没有证明zlc_atom可按声明依赖独立运行Editor Send。

Concrete plugin依赖公开`zlc_ui/zlc_plot`合理；device ownership claim应由host/session提供的public command seam注入，或把最小claim record放在共同execution owner。不要让plugin在worker深处importcomposition package。

## 4. Profile / LUT / correction 数学裁决

| 步骤 | 当前实现 | 裁决 |
|---|---|---|
| canonical radians | finite, wrap, float32, immutable bytes | `PASS` |
| orientation | flip canonical science phase beforedevice code | `PASS CONDITIONALLY`；方向待光学验收 |
| phase quantization | `round_half_up(phase*128/π) mod256` | `PASS`；circular nearest code |
| correction order | native code add thenmod256 | `PASS`，符合当前目标设计 |
| wavelength LUT | invert monotonic profile curve | `PASS MATH / UNVERIFIED INPUT` |
| 2π display value | same curve inverse | `PASS MATH / profile provenance open` |
| correction exact wavelength | raw native code | `PASS only if vendor encoding confirmed` |
| correction cross-wavelength | separable unwrap + scalar wavelength ratio | `REDESIGN/VALIDATE` |
| DVI padding | left1272 + right8 zero | `SOFTWARE PASS / HARDWARE ALIGNMENT OPEN` |
| USB readback | exact active gray compare | `PASS frame-memory evidence` |
| optical settle | unconditional authoredsleep | `USER DECISION + MEASURE` |

官方资料说X15213出厂按特定wavelength range做高线性预校准，典型phase response可接近线性；这支持“曲线可近似线性”，但不支持仓内这条curve必然属于`LSH0804382`。Hamamatsu公开的cross-nicol方法可作为实验室重新测量phase-vs-level的验收方法。

## 5. 逐文件 / 类 / 函数裁决

### 5.1 Common / Virtual

| 文件 / 符号 | 裁决 | 说明 |
|---|---|---|
| `slm/device.py::_shape` | `PASS` | 小而直接。 |
| `canonical_phase` | `PASS` | 唯一canonicalization，immutable与边界正确。 |
| `SlmAdapter` | `KEEP + REDESIGN known/unknown command` | 窄surface正确；real初始truth无法表达。 |
| `bind_slm` | `KEEP + FIX evidence/cleanup` | binding owner正确；USB serial应可用hardware-readback evidence，close失败不应遮蔽primary。 |
| `simulation/slm.py::VirtualSLM`全部 | `PASS` | world为唯一phase truth；close不改world。 |
| `slm/__init__.py::open_slm_control` | `PASS` | lazy plugin UI入口正确。 |

### 5.2 X15213 device leaf

| 符号 | 裁决 | 说明 |
|---|---|---|
| `_DisplayDeviceW`, `_DevModeW` | `PASS` | 完整Unicode ABI size/offset有Windows守卫。 |
| schema | `KEEP + REDESIGN profile-driven fields` | transport/config必要；serial/default wavelength/settle存在重复或硬编码。 |
| `_windows_displays`, `_display` | `PASS WITH LIMIT` | 只枚举endpoint geometry/frequency，诚实不推EDID identity。 |
| DPI/native geometry helpers | `PASS` | 对防OS scaling有真实价值，但非pixel/optical proof。 |
| `_open_dvi_presenter.run/present/close` | `KEEP + CRITICAL FIX` | owner thread必要；timeout late side effect、scanout/close语义未处理。 |
| `_find_sdk_directory` | `PASS WITH DEBT` | 配对DLL检查有用；ProgramFiles递归scan可能慢，版本未核。 |
| `_load_sdk` | `REDESIGN ABI` | 无vendor argtypes/restype/version proof。 |
| `_check`, `_usb_open/close/serial/mode` | `KEEP` | 直线wrapper合理。 |
| `_close_usb_after_probe` | `REDESIGN` | 吞close failure可泄漏claim。 |
| `_connect_usb` | `KEEP + FIX reboot/enum` | reopen+mode verify好；只看一个board、Reboot return未核。 |
| `_prepare_dvi_controller` | `REDESIGN` | mode switch后未reopen/recheck却称proven。 |
| `_load_profile` | `KEEP LOADER + MOVE DATA` | validation直接；serial calibration不应锁在package，schema字段不足。 |
| `_half_up`, `_phase_lut` | `PASS MATH` | 输入profile真实性另论。 |
| filename parsers | `PASS AS PARSER / USER DECISION AS AUTHORITY` | 文件名是否科学truth需确认。 |
| `_scale_correction_wavelength` | `REDESIGN` | 非一般2D unwrap。 |
| `_load_correction` | `KEEP + metadata fix` | mode/shape严格；未强制BMP suffix，真正问题是encoding/provenance。 |
| `X15213Adapter.__init__` | `REDESIGN ORDER/TRUTH` | pure validation应先于DVI mode mutation；初始phase虚构。 |
| correction properties/load/enable | `KEEP + ownership/persistence` | atomic state好；绕过claim。 |
| `_gray` | `PASS CONDITIONALLY` | pipeline顺序正确；profile/correction/orientation待验。 |
| `apply_phase` | `REDESIGN command outcome` | success path直接；post-side-effect failure不能保留旧truth。 |
| `close` | `PASS USB / REDESIGN DVI SEMANTICS` | USB retry强；DVI销毁输入。 |
| discovery helpers | `KEEP + FIX missing profile isolation` | candidates有价值；USB缺profile不应吞掉DVI discovery。 |
| `_factory`, descriptor | `PASS` | plugin-local factory/capability归属正确。 |
| `profiles/LSH0804382.json` | `MOVE + REVALIDATE` | 需要真实provenance/type/response；当前不能作为已验收calibration。 |

### 5.3 Editor

| 符号 | 裁决 | 说明 |
|---|---|---|
| `_snapshot/_host/_number_spin` | `PASS` | 小的plugin-local UI helpers。 |
| `SlmEditorControl.__init__` | `KEEP + FIX initial/device truth` | owner集中合理；unknown phase和default auto-solve混淆。 |
| `_build_*` UI methods | `PASS WITH DEBT` | 两页/三图/scroll/popup符合当前产品；单class虽大但同一lifecycle。 |
| preset popup/apply | `PASS` | Apply才materialize，objective显式。 |
| pupil functions | `PASS` | 1/e² intensity定义正确；artifact漏center/context。 |
| correction sync/toggle | `REDESIGN claim/persistence` | UI projection好，mutation lifecycle错误。 |
| wavefront build/reset/change/compose | `PASS` | carrier与Noll Z4–Z11数学直接、分层明确。 |
| selector/paint | `PASS WITH PERF DEBT` | owner正确；drag每点全array copy+solve queue但latest coalesce。 |
| `set_target` | `PASS` | support/objective state invalidation正确。 |
| `set_phase` | `PASS exact roundtrip` | reset wavefront正确；target与loaded phase成为独立draft需明确。 |
| solve queue/start/finish | `PASS` | capacity-one、revision、optimizer acceptance严谨。 |
| import/save target | `REDESIGN artifact/perf` | objective/pupil丢失，save同步1.68s。 |
| save/load phase | `PASS exact + provenance debt` | science artifact纯净；缺device receipt。 |
| `send/_send_phase/_finish_send` | `KEEP + freeze mapping/outcome` | worker+lease正确；只冻结phase，不冻结mapping，ambiguous hardware failure。 |
| `_sync_send_enabled` | `PASS WITH DEBT` | stale target守住；其他mutators与newer draft仍可变。 |
| `_choose` | `PASS WITH DEBT` | 单一路径；correction callback不claim。 |
| `_finish_close/close/restore` | `PASS` | 非阻塞、等待实际workers/hosts；hung vendor call仍无退出边界。 |
| `open_slm_control` | `PASS` | plugin-owned lazy composition正确。 |

## 6. 测试审查

本轮直接运行：

~~~text
test_slm_x15213.py + test_slm_editor.py
46 passed in 13.66 s
~~~

这些测试对软件分支很有价值，但不能替代真机证据。

| 测试组 | 有效守卫 | 未覆盖 / 过度结论 |
|---|---|---|
| X descriptor/profile | field set、profile monotonic、derivedlevels | 不证明curve来源/serial measurement/type；“not hard-coded”结论过强。 |
| Win32 structs | 当前Python/Windows ABI size/offset | 不覆盖vendor DLL function ABI。 |
| discovery | candidate、serial read、normal close | 只一个USB board；missing profile隔离、probe close failure未守。 |
| correction state | load失败atomic、enable只影响next apply | 不覆盖Task/Send并发claim、session persistence。 |
| phase→gray tests | orientation/order/wrap/right8 bytes | 输入curve、correction sign/orientation和optical result均未验。 |
| wavelength correction | 一维sawtooth与serial filename | 不覆盖真实二维residue/noise；separable unwrap天然通过。 |
| USB write/readback | fake frame bytes、bad readback、close retry | fake SDK同时定义了假定ABI；bad readback后physical/software divergence未断言。 |
| DVI mode | fake mode select/reboot call | fake在Mode_Select立即改mode；没有reboot后的reopen/recheck。 |
| DVI geometry | mockeduser32 coordinates | 实际Tk/GPU/second-display scanout完全未运行。 |
| Editor latest solve/state | 很强：stale、optimizer、compose、file exact | 多为private method/offscreen；没有external device command subscription。 |
| Editor Send | lease busy、三线程、heartbeat、close retry | Virtual apply，不覆盖X SDKhang/ambiguous write；correction bypass未测。 |
| Editor correction | 控件可见且不立即send | 正好固化了“不claim”路径，没有与active Task组合。 |

应新增少量纵向红灯，而非扩大mock矩阵：

1. real adapter安装后command state是unknown，不能被Feedback当incoming恢复；
2. write/change/read各阶段注入失败后command outcome分别为known-old/known-new/unknown；
3. active Feedback claim期间correction load/toggle被拒绝；clicked Send冻结完整mapping；
4. DVI mode switch必须reconnect并读回mode0；Reboot failure可见；
5. missing USB profile只报告该candidate缺calibration，不抹掉DVI candidates；
6. correction conversion具transpose/rotation等价或明确拒绝含residue map；
7. target save/load保留objective与pupil context；Qt owner heartbeat不因save target停1秒以上；
8. Editor在external Task apply后明确显示draft/device divergence；
9. DVI close/reopen语义和late timeout side effect有明确contract；
10. 实验机formal acceptance（下一节）。

## 7. 实验机必须完成的验收

开发机/mock最多证明array与调用顺序。以下每项未完成前不得写“real X15213 accepted”：

### 7.1 身份与SDK

1. 记录完整type number、head serial、controller serial/version、SDK/DLL version与官方header。
2. 按header逐项核对ctypes calling convention、argtypes/restype、buffer sizes和return codes。
3. USB/DVI mode切换后断开、重连、重新读取serial/mode；验证失败恢复与多controller行为。
4. 明确USB disconnect/close后frame memory和display slot是否保持。

### 7.2 Phase calibration与correction

1. 按Hamamatsu公开cross-nicol方法，在实际工作wavelength/temperature测0..255 phase curve；保存raw intensity、unwrap、fit、uncertainty和date，而非只保存结果数组。
2. 核实controller factory linearization后，软件是否仍需要完整inverse LUT，还是只需wavelength/2π scale。
3. 确认vendor correction BMP的code定义、native orientation、source wavelength、sign、offset和active-pixel alignment。
4. 分别A/B no correction、vendor raw、wavelength-converted correction，测wavefront RMS/spot quality，验证unwrap算法。

### 7.3 Raster与orientation

1. 用四角不对称fiducials确定active1272在DVI raster的位置、flip_x/y、反射镜像与qCMOS registration。
2. DVI关闭ICC/HDR/night-light/GPU scaling/dithering后，用capture/readback设备或光学阶梯验证0..255不会被改写。
3. USB逐字节readback之外，再确认Change_DispSlot真正驱动当前head。
4. 验证DVI presenter被系统notification、focus、power management、display hotplug影响时的失败行为。

### 7.4 Settle与lifecycle

1. 对0↔2π及代表性大/小gray transition实测rise/fall；settle使用该head、wavelength和接受误差的最坏值。
2. 测Send ack、controller update、LC optical settle三种时间，UI分别报告，不再统称sent。
3. 测Editor close、session close、process crash、DVI window destroy、USB close后的实际光学输出。
4. 在Task持claim时尝试Editor Send/correction，验证全部mutation都被同一ownership拒绝。

## 8. 文档 / 实现矛盾

| 文档声明 | 当前事实 |
|---|---|
| serial-specific完整calibrated nonlinear phase curve | curve无measurement provenance/type，几乎affine；只能证明JSON格式。 |
| correction按连续二维phase unwrap | 实现为固定X后Y的一维unwrap。 |
| DVI exact unscaled presenter | 只核window client geometry，不核GPU gray/scanout/optics。 |
| DVI controller mode proven | switch后不reopen/recheck，Reboot return忽略。 |
| close/session不改变phase | Editor close成立；DVI adapter/session close销毁唯一input window，不成立。 |
| Task开始冻结resources/device mapping | runtime correction可绕过claim中途改变mapping。 |
| immutable last-commanded phase是真实device state | real init从未command却报告zero；post-write failure也可能保留旧值。 |
| Target artifact保持唯一target truth | intensity保持，但objective_kind与pupil solve context丢失。 |
| mocked transport/readback只证明bytes、最终需实验机 | 这句是诚实的；当前Checkpoint/README的“完成”措辞不应覆盖它。 |

## 9. 推荐的最小目标设计（不实施）

不建议新建vendor framework。沿现有owners直接收口：

1. **SlmAdapter command state**：增加known/unknown语义与monotonic command revision；real init是unknown，Virtual可known。不要强迫占位zero。
2. **X adapter command outcome**：在现有`apply_phase`内记录Write/slot/readback/present阶段；side effect后失败标unknown，不伪装old。
3. **冻结device mapping**：`_gray`使用的profile/wavelength/orientation/correction作为一次command snapshot；correction mutation取得现有exclusive claim。
4. **Profile归位**：serial calibration作为workspace/installation artifact，补type/provenance/response；descriptor只加载，不持有lab-specific truth。
5. **USB先成为可验收主路径**：补官方ABI/reboot confirmation/real receipt；DVI保持明确experimental endpoint，直到GPU/optical验收。
6. **Editor只投影两份状态**：authoring draft与device command known/revision；external Task apply后显示out-of-sync，不需要把Task逻辑搬入Editor。
7. **Artifacts分工**：science phase NPZ继续纯canonical；target/project artifact保存objective+pupil；另由run/device owner保存command receipt。
8. **I/O移出Qt owner**：复用Editor现有worker或改用紧凑target codec，不能同步写百万float JSON。
9. **close诚实按transport描述**：USB可能retain，DVI撤销window；不要虚构统一物理结果。

## 10. 交主线程登记的用户裁决

1. **未知初始phase**：允许unknown并限制Feedback restore，还是Init显式发送已知phase。
2. **DVI产品级别**：继续作为正式transport，还是在完成实验机pixel/optical验收前标Experimental；审计建议后者。
3. **DVI session close**：接受input撤销并明确提示，还是要求独立持久presenter/service。
4. **Profile storage**：serial calibration移到workspace/installation artifact，还是继续随Python package发布；审计建议移动。
5. **Profile来源**：请确认`LSH0804382`的256点curve究竟来自逐点实测、vendor文件、datasheet digitization还是旧2π level展开。没有答案前不能称calibrated。
6. **Correction wavelength policy**：只接受与current wavelength一致的vendor map，还是支持换波长；若支持，必须决定真实2D unwrap与metadata格式。
7. **Runtime correction**：允许Editor临时修改且必须持claim/保存receipt，还是只允许Device Manager stopped config。
8. **Settle acceptance**：按profile固定最坏时间、用户authoring，还是每command自适应；无论哪项都需该head实测。
9. **Target artifact范围**：只保存intensity，还是保存可重算的objective+pupil project；审计建议后者，同时保留独立exact phase NPZ。
10. **Command receipt**：是否要求每次Task/Send可追溯到profile/correction/orientation/transport/readback/outcome；审计建议要求。
11. **Package独立性**：允许zlc_atom concrete Editor反向import zlc_workbench，还是由session注入public claim seam；审计建议消除未声明环。

## 11. 最终裁决

- `SlmAdapter`、canonical phase与plugin-local Editor是正确骨架，不应推倒重做。
- X15213 phase-code/correction/LUT顺序在给定输入假设下数学清楚；当前主要问题是**输入calibration不具可审计来源**与**hardware outcome不可可靠表达**。
- USB代码具有可保留的强byte path和close retry，但尚缺官方ABI与真controller验收。
- DVI代码具有有价值的geometry防缩放检查，但仍只是endpoint presenter，不是byte-exact或optical proof。
- Editor solver/command线程与stale/lease/close设计通过；correction ownership、external command同步和artifact roundtrip仍未过关。
- `LSH0804382.json`应判为`UNVERIFIED DEVICE DATA`，不能因256项齐全就自动PASS。
- 在实验机验收和用户裁决完成前，任何文档不得把“frame bytes generated/read back”写成“852 nm optical phase correctly applied and settled”。
