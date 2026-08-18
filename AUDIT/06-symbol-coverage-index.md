# 06 — 全项目逐文件/符号/测试覆盖索引

状态：完成。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`

本文件不重复数千行裁决，只证明每类tracked一方文件落在哪份报告。详细报告均按文件列出顶层类/函数或符号簇，并给`PASS / PASS WITH DEBT / MOVE / MERGE / DELETE / REDESIGN / USER DECISION`。

## 1. Python source与tests

| 范围 | Source `.py` | Test/support `.py` | Examples `.py` | 逐文件裁决owner |
|---|---:|---:|---:|---|
| `zlc_data` | 15 | 9 | 0 | `06a-data-durable.md` |
| `zlc_durable` | 5 | 3 | 0 | `06a-data-durable.md` |
| `zlc_runtime` | 16 | 14 | 1 | `03c` + `06d-runtime-plot-remaining.md`；example另见`06g` |
| `zlc_plot` | 52 | 55 | 6 | `02` + `06d`；examples另见`06g` |
| `zlc_ui` | 49 | 15 | 7 | `02` + `06b-ui-workbench.md`；examples另见`06g` |
| `zlc_pulse` | 22 | 14 | 0 | `04a` + `06h-pulse-remaining-python.md` |
| `zlc_atom` | 76 | 37 | 0 | `03a/03b/04b/04c/05a–c` + `06c-atom-remaining.md` |
| `zlc_workbench` | 30 | 27 | 0 | `02/03` + `06b-ui-workbench.md` |
| root bootstrap/tool/test config | 4 | — | — | `06e-root-bootstrap-packaging.md` |
| FPGA package Python shims | 2 | — | — | `06f-fpga-nonpython.md` |
| **合计** | **271含root/FPGA** | **174** | **14** | **459 tracked Python全部归档** |

`06g-test-evidence-architecture.md`从另一条轴独立覆盖：

- 166/166 test files；其余8个是support/helper/package files；
- 1,346/1,346 test definitions；
- 55,304行test code；
- 14/14 Python examples；
- support、fixtures、goldens与root bootstrap。

## 2. 非Python产品资产

| 类型 | 数量 | 审查owner |
|---|---:|---|
| Verilog/VH/Tcl/XDC | 28 | `06f-fpga-nonpython.md`逐module/task/function/proc/testbench |
| Windows batch | 15 | root正式入口见`06e`；FPGA与package重复入口见`06f` |
| Canonical notebooks | 7 | `06g`逐本fresh-execution/status裁决 |
| Package JSON/profile/template | 8 | Atom templates/profile见`04c/05c/06c`；Pulse board config见`04a/06f/06h` |
| Plot fonts/typed marker/goldens/fixtures | 按tracked tree | `06d`与`06g` |
| Markdown | 81份基线清册 | package报告docs sections + `07-doc-code-test-conflicts.md` |
| pyproject/metadata | root + 8 packages | `06a–06h`对应package + `06e`根部署 |

Vendor DLL/EXE、generated Vivado build、workspace实验数据和用户提供的`Disc_noinstaller.7z`不属于tracked一方source；本审计只在05c引用其既有静态证据，没有执行或重新分发。

## 3. 端到端链路覆盖

| 链路 | 报告 |
|---|---|
| Snapshot -> SignalPlane/Front -> Plot/Fit/Overlay/Selector -> Raster/Qt | `02`, `03c`, `06d` |
| Measurement/Task -> live/progress/preview -> terminal/artifact | `03a`, `03b`, `03 summary`, `06c` |
| Pulse model -> compile/wire/transport/RTL -> camera grouping/same-shot | `04a–c`, `06f`, `06h` |
| SLM target -> solver -> science phase -> adapter -> fluorescence feedback | `05a–c`, `05 summary` |
| Save/archive/durability/viewer | `06a`, `06b`, `06e` |
| Install/discovery/device lifecycle/simulation | `06b`, `06c`, `06h` |
| Tests/examples/notebooks/evidence lanes | `06g` |
| Docs/code/test contradictions | `07` |
| Proposed architecture and decision gates | `08`, `DECISIONS-PRIORITY` |
| Independent audit-of-audit | `09` |

## 4. 有意保持OPEN的范围

“覆盖完成”不表示以下事实已经被开发机证明：

- 真X15213 SDK ABI、controller state、orientation、LUT/correction与optical settle；
- 真DCAM/Pylon trigger acceptance与absolute same-shot marker；
- Vivado compile/STA、bitstream identity、real FPGA SAFE/DONE/waveform；
- real-screen人类交互性能和字体/DPR；
- 实验机噪声分布下的SLM 1% validation；
- 仓外脚本/用户是否依赖当前public APIs。

这些不是漏审文件，而是证据等级E3/E4无法替代的实验机/用户裁决边界，已进入`DECISIONS-PRIORITY.md`与对应runbook建议。

## 5. Scope closure结论

在当前约定范围内：tracked一方source、tests/support、examples、notebooks、launchers、metadata、RTL/build assets和docs均已有逐文件owner；`09`指出的Pulse Python与非Python缺口已由`06h`和`06f`关闭。后续工作是用户裁决、修复与验收，不再是继续扩大静态审计范围。
