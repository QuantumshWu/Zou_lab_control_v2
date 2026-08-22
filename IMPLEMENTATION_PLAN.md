# ZLC v2 — Current implementation checkpoint

更新时间：2026-08-22

Branch：`m7-final`（latest `master` base：`4076dac`）

状态：`M7 COMPLETE / SWEEP COMPLETE`

本文只保存当前实现、最新证据和下一步。最终产品不变量见
`ARCHITECTURE_DESIGN.md`；历史设计、Goal、Handoff和Audit由Git history保存，不再是
active product docs。

## 1. 已完成的产品范围

- M1–M3：dead/parallel framework清理、strict Data/Durable truth、canonical Runtime
  live/terminal/lineage与Logic Node contract完成。
- M4：Data/Fit/Overlay/selector、Figure archive、PanelState、Qt worker/close、
  active+latest solve及Plot/Fit性能闭合。
- M5：Pulse application/cycles、Camera cadence、Remote takeover/SAFE、FPGA
  timing/build/receipt与strict transport闭合。
- M6及follow-up：真实SLM恢复为server-owned DVI-default、USB显式可选；known command
  receipt、Science Context v2、single-batch qCMOS bright-dark Feedback与immutable
  SimulationWorld闭合。Feedback执行全部authored updates，不按magic ratio提前结束。
- M7：single distribution、package-data、installed manifest/commands、persistence
  migration、唯一notebook和formal evidence lanes已实现；最终全量验证与冻结审计进行中。

## 2. Latest master integration

- qCMOS Start不再重写未变化的完整working point。ROI与exposure分别最小应用，
  `set_exposure_seconds`返回authoritative readback，Camera Measurement直接冻结该结果，
  删除同一capture前第三次完整property query（`6fa24f7`）。
- Device Manager loaded card同时提供Control/Close。Active draft通过同一
  `ExperimentSession`差量reconcile：稳定`instance_id`与canonical setup相同的leaf、
  SignalPlane、TaskConsole和Panel保留；role-only不重启；add/remove/change/Close只处理
  affected/world-bound/dependency closure。Maintenance barrier阻止新command并等待相关
  Logic lease；partial failure保留reachable owner、投影effective config并允许只刷新UI或
  重试（`4076dac`）。
- SLM Feedback每candidate严格使用一批authored shots；单高斯boost、loaded relative gain、
  maximum change与loading floor均为authored/history-owned。默认100 shots、12 updates，
  全部updates完成后保留最佳已测candidate（`9dec879`、`8e14582`）。
- FPGA当前root fix保留50 MHz timing、真实UART BRAM、strict part/device identity、
  repository-contained Vivado scratch及显式program边界。`build_and_program.bat`默认build后
  program volatile FPGA；`run_server.bat`只启动server，绝不隐式build/program；flash仍需
  显式请求（`3e2b585`、`3dfc71c`、`91de00f`、`84ee73b`）。

## 3. FPGA冻结证据

- Vivado 2019.1 fresh project完成7个IP、top synth、place/route、reports和bitstream；
  未触发post-route补救。
- Routed setup WNS `+0.726 ns`、TNS `0`；hold WHS `+0.036 ns`、THS `0`。
- 12条bus-skew全部MET，0 violated，最差`+18.988 ns`。
- 资源：20075/20800 LUT、14053/41600 FF、76/90 DSP、40 RAMB36 + 2 RAMB18。
- Bit SHA256：`82A8E04DC3BD4F21E3ACED22D16E4544C7CBC3E9C7A642337F4D689702994A6C`。
- 10个engine + 2个UART tracked Vivado oracle通过；真实BRAM 7个pulse均2000 tick；
  full-top两次FIRE均4帧一致；SAFE pin gate通过。
- 这些是build/timing证据；未把software evidence冒充实验板program/flash或外部I/O验收。

## 4. Milestone 7实现状态

### 4.1 一个distribution

- 根`pyproject.toml`是唯一manifest；八份nested pyproject和layer version truth删除。
- `constraints.txt`是唯一resolved dependency surface；root runtime/test/notebook closure固定。
- Wheel包含bootstrap、八层、Calibration/Scan templates、SLM profile、Plot font、typing
  markers及完整有效FPGA RTL/XDC/Tcl/README assets。
- `zlc`是唯一console script；layers、commands和evidence lanes来自installed metadata；
  `zlc check`按distribution RECORD核八层文件归属。
- Windows wrappers不修改`PYTHONPATH`、不重建argv，原始`%*`只转发一次。

### 4.2 Persistence migration

- `save_npz(stream, snapshot)`只编码caller-owned binary IO；path publication归
  `zlc_durable.atomic_write_file`，编码失败不先截断旧文件。
- Figure writer只写v2；唯一reader迁移精确`zlc.figure/v1`并经current validator复核。
  Workspace 20/20旧Figure与其PanelState均已读取；unknown fields/format/version继续拒绝。
- Calibration writer固定`zlc.calibration.readout/v1`；唯一owner迁移两套精确unversioned
  roots。Workspace 25/25旧Calibration读取；不可恢复统计保持NaN/count0，不伪造数据。

### 4.3 Notebook、evidence与docs

- 只保留`packages/zlc_workbench/notebooks/usage.ipynb`；无output/count/source bootstrap/
  hardware cell，fresh kernel在checkout外临时workspace运行virtual canonical chain。
- Formal lanes：`software`、`gui_offscreen`、`virtual_vertical`、`notebook_offline`、
  `real_screen`、`hardware`。前四条自动；后两条只返回`NOT EXECUTED`并指向runbook。
- Root/package docs只描述当前distribution、公开路径和evidence边界；Audit、Goal、Handoff
  从active tree删除。

## 5. 当前验证

M7 master-sync前已完成：root wheel构建、fresh venv constrained install、`pip check`、
installed provenance/`zlc check`、FPGA packaged assets、`notebook_offline`、
`virtual_vertical`及focused persistence/launcher/manifest/Qt process lanes。

Latest master新增验证：

- qCMOS adapter/measurement定向tests随`6fa24f7`通过；尚未把实验机首次auto Panel延迟
  作为已验收数字。
- Device reconcile/session/maintenance/UI定向与TaskConsole live Close/reopen、Control/tune
  集成通过，并已进入最终installed full lane。
- SLM 100-shot virtual evidence证明observable sites可从23恢复到35，candidate 7的全阵列
  ratio为`1.098487`；这是virtual software evidence，不是optical acceptance。

最终冻结证据：

- 最终wheel：`zou_lab_control-2.0.0-py3-none-any.whl`，1,382,330 bytes，294 entries，
  SHA256 `9521D2C3A76439162D80730BE1A18B8C20256716E189B56603356D2CC1F089ED`。
- fresh venv按`constraints.txt`安装`[dev]`；`pip check`零问题；`zlc check`确认八层
  全部来自该venv的单一`zou-lab-control` RECORD，三个retired包均不可导入。
- 同一最终wheel上：`software: PASS`，合计`1614 passed, 5 skipped`；5个skip仅为本机
  无Icarus，既有Vivado 2019.1/xsim证据单独覆盖RTL。
- 同一最终wheel上：`gui_offscreen: PASS`；UI `86 passed`，并包含Plot Qt、SLM editor、
  Workbench GUI groups及每个TaskConsole case的独立进程生命周期。
- 同一最终wheel上：`virtual_vertical: PASS (9 passed)`；`notebook_offline: PASS`。
- source全包与installed lane一致：Atom 336、Data 72、Durable 27、Plot 427、Pulse 138、
  Runtime 103、UI 86、Workbench 425；未以test删除掩盖新的loading physics，而是把旧的
  35-site/all-loaded断言改为observable-site和trap-depth阈值不变量。
- 静态冻结：1个`pyproject.toml`、1个notebook；8/8 JSON可解析；notebook 5个code cells、
  0 output、0 execution count、0 hardware token；23份Markdown的本地链接缺失为0；
  conflict marker、旧module launcher、layer `__version__`和retired production import均为0。
- M7相对latest master为146 files `+2235/-19863`，净删17,628行；其中product/build
  41 files `+1088/-735`（净+353）、tests 32 files `+565/-395`（净+170）、
  docs/assets 73 files `+582/-18733`（净删18,151）。大幅净删来自历史Audit/Goal/Handoff、
  七本旧notebook、八份nested manifest及重复examples/docs，不是删除产品能力。

## 6. 明确未执行的实验机验收

以下不能由software/offscreen/virtual evidence冒充，当前均为`UNEXECUTED`：

- real-screen：真实monitor、DPR、window interaction和capture receipt；
- camera：official DCAM/Pylon SDK/runtime、accepted-edge timestamp、首次auto Panel latency、
  exposure/busy/drop/cancel及raw/electron provenance；
- SLM：官方SDK/header、serial/profile/correction、DVI/USB orientation/readback和optical settle；
- optical Feedback：raw per-shot/site/dark、simultaneous CI、total brightness和rollback；
- FPGA board：最终bitstream program/flash及外部DAC/TTL电气时序/波形。

M7不授权自动执行上述hardware/real-screen操作；它们保持明确`UNEXECUTED`，不影响
software distribution milestone已达到`COMPLETE / SWEEP COMPLETE`。
