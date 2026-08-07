# GOAL — zlc_atom:中性原子域包(设备 / 观测 / 编排 / 派生)

状态:GOAL COMPLETE(§API 轮完成; W / M / v3 / v3.1 / v3.2 全部完成并归档到 `docs/goal-archive.md`,不许回改)
仓库:`C:\Users\eadri\Dropbox\WorkCode\Github\zlc_atom`(所有工作只发生在这里)

> **背景**:v1 交付于 2026-08-03 外部验收判**返工**。三份验收报告在 `docs/acceptance-fidelity-2026-08-03.md`(物理等价实验)、`docs/acceptance-goal-audit-2026-08-03.md`(逐项审计)、`docs/acceptance-tests-2026-08-03.md`(测试与 fake 质量)——**动工前必须完整读完这三份**,每条返工项的证据与文件:行号都在里面。已完成轮次的原文在 `docs/goal-archive.md`,本文件只留活的工作。
> **范围宪章不变**:logic node 只做 camera_measurement / calibration / occupancy 三个;设备只做 camera + sequencer(各含虚拟孪生);其余叶子全部后置。包名 `zlc_atom`(≠树内 `zlc_neutral_atom`,防影子)。
> **可保留清单(验收实证等价/达标,别推倒重做)**:bimodal 组件统计(逐位等价)、box/psf 窗口提取核、classify 算子、install graph 骨架、叶子 rglob 发现+清单 pin、context 四件收缩、broker 绑定 helper、SimulationWorld 显式注入的方向、虚拟相机有界队列丢帧、FakePulseStreamer 形状。

## 铁律(v1 违反的三条列在最前,不可再犯)

> 🔴 **永不建 venv**(用户 2026-08-05 明令):依赖一律全局 `pip install`;脚本里**不许**出现 .venv 探测/偏好分支。血训:zlc_pulse 的启动器曾优先用仓库 .venv,而那个 venv 没装 pyserial,导致串口枚举抛 ModuleNotFoundError、返回零候选口、UART 探测循环一次都不进,服务器**插不插线都无条件退回 JTAG**(实验机上=常驻 1-2 GB vivado.exe)。

1. **迁移不是发明**:凡本文件说"迁入"的模块,必须以参照树实现为底本改造(只读源:`..\Zou_lab_control_v1_claude\Zou_lab_control_v1`),禁止凭空重写同名物。参照树读不到=记阻塞,不是就地发明的许可。
2. **冻结 oracle 绝不自产**:物理期望值只能来自树内冻结文件原样拷贝;任何"手写期望值+同 commit 消费测试"都是自证,验收一票否决。
3. **契约唯一**:对 zlc_runtime / zlc_pulse 的一切签名只按其 `docs/contract.md`(均已 FROZEN)写;契约不够用记阻塞提议改文档,**绝不发明变体**;fake 只住 `tests/` 且必须被真实消费。
4. 其余同前:绝不 push;每完成一项勾选本清单+独立 commit(勾选状态=唯一进度真相);无后向兼容、干净删除;不加防御仪式;守卫自证非空洞(空扫描必须红);**虚拟与真机共享同一代码路径,只 fake 最底层设备协议面,零 `if virtual` 分支**;哨兵/注入绝不 `or` 兜底;含数值算法的模块必带冻结 oracle 断言;开工前确认 Dropbox 同步完成(同步中间态会造成文件回退与假红)。

## 四步开工仪式(每轮必做,不信记忆)

1. 读本文件全文 + `git log --oneline -10` + `git status`,以清单勾选为唯一进度真相。
2. 跑 `pytest -q` 基线:红则本轮只修红,不开新工。
3. 选清单里**最靠前的未勾选项**,声明本轮范围。
4. 读该项完成判据,确认可机械验证后才动手。

## 三种合法收尾(每轮只能以其一结束)

1. 完成:判据验证通过 → 勾选 → 独立 commit。
2. 受阻:原因写进 §阻塞记录 → commit 已有进展 → 停,不硬闯。
3. 全部完成:§机械终态判据全绿 → 状态改 GOAL COMPLETE → 之后只回终止陈述。

---

## 本仓血训(提炼自已完成轮次;新轮开工前读这几条)

1. **迁移不是发明**(v1 违反过,代价是整轮返工):凡说"迁入"的模块必须以参照树实现为底本,禁止凭空重写同名物。参照树读不到 = 记阻塞,不是就地发明的许可。**这条在 zlc_pulse 又付过一次学费**(host 重写导致真机静默不输出),是全项目最贵的一条。
2. **冻结 oracle 绝不自产**:物理期望值只能来自树内冻结文件原样拷贝。"手写期望值 + 同 commit 消费测试"是自证,验收一票否决。同冻结输入端到端错误率 **== 29/360** 是活的断言,不许放松容差。
3. **标定 bracket 的物理**:一次 MOT 装载后三个相机窗口 **long → short → long**;前后两张 long 是**共识标签对**(两帧一致才证明原子整段稳定存在),short 夹在中间做阈值表征。两次独立装载 / 只留两窗口都会污染标签——这条我和 codex 都错过一次。
4. **measurement = 纯观测,task = 编排**:measurement 不持有也不操作 sequencer;task 默认从 `pulses/<name>.py` 决议 pulse、配置并 fire、采集、计算、发布。"task 能算数学"不算完成,判据是"从 pulse 定义文件到 report 一条调用全自动跑通"。
5. **声明 = 实现**:descriptor 里声明的每个 DeviceRequirement 都要被机械证明真被使用;声明不用 = 删声明,用了没声明 = 补声明。
6. **虚拟与真机同一代码路径**,只 fake 最底层设备协议面,零 `if virtual` 分支;哨兵/注入绝不 `or` 兜底(工厂里 `SimulationWorld()` 兜底会让 miswire 时各造世界、因果断裂)。
7. **守卫必须自证非空洞**:rglob 扫描要断言"文件数 > 0";在简并点取的断言(占据恰好 3/6 时 `1-rate == rate`)对任何变异永久绿。


## §API 轮:两个导出都不是 API(2026-08-05 逐名裁决 + 对抗复核)

> 本仓导出 **2 个**,是六仓最少的——但审计发现**这 2 个都不是 API 表面**:`TOP_LEVEL_ALLOW_LIST` 是一个"描述导出清单、并且把自己列为其成员"的簿记常量,`__version__` 是版本探针。
> 这是"少"与"对"不是一回事的样本:数量已经到底了,选的东西却仍然是错的。

**保留(1)**:`__version__` —— 反驳者证明这是**错撤**:它是本生态**已在用**的防影子 import 探针(姊妹仓 `zlc_data/tests/test_package_guards.py:178` 就断言自己的 `__version__`),而本仓 README 顶上就写着"⚠️ 同名影子警示";撤掉它,zlc_atom 会成为六仓里唯一写不出该守卫的包。

**从顶层门面撤下(1)——实现不删**:`TOP_LEVEL_ALLOW_LIST` —— **放错层**,不是没人用。全仓 + 六姊妹仓 grep,唯一消费者是它自己的守卫 `tests/test_import_boundaries.py:34`;notebook cell 1 与 README:35-49 的教程**全部走子包路径**。姊妹仓的正确做法是把 allow-list 字面量放在**测试文件**里(见 `zlc_data/tests/test_package_guards.py` 的 `EXPECTED_PUBLIC_API`),而不是塞进发行包。照搬那个形态。

- [x] API1 `__all__ = ("__version__",)`,`MAX_PUBLIC_NAMES = 2`;allow-list 字面量搬进 `tests/test_import_boundaries.py`。
- [x] API2 **`annotations` 泄漏**:删完之后仍然公开可导入的名字里有 `annotations`(`from __future__ import annotations` 留下的模块属性),审计从头到尾没看见它。这正是"上限必须断言在真实命名空间上、不能只查 `__all__`"的活样本——按通用判据 1 写守卫时要能抓到它。

### 通用机械判据(六仓一致;**这三条缺一条,收缩就是装饰**)
1. **上限断言在真实公开命名空间上,不只是 `__all__`** —— 具名常量 `MAX_PUBLIC_NAMES = <数字>`,断言对象是 `[n for n in dir(pkg) if not n.startswith("_")]` 减去子模块名。实测发现:zlc_data 的 `__all__` 是 70 而 `dir()` 公开项是 80,只查 `__all__` 的守卫**抓不到从别处漏出来的名字**。
2. **`__all__` 与 `docs/contract.md` 的名字集合双向相等** —— 多一个少一个都红。文档里没有名字清单的,先补清单再写这条。
3. **"撤走"= 从顶层门面拿下来,代码一行不删。** 具体是:名字从包级 `__all__` 移除、不再在 `pkg/__init__.py` 里 re-export;**实现原封不动留在它自己的子模块里,继续可以 `from pkg.submodule import Name` 用**。判据写成"顶层 `getattr(pkg, name)` 不再解析得到"(不是"包里搜不到"),同时补一条正面断言:**子模块导入路径仍然可用**。每撤一个先 grep 调用点改成子模块路径,跨仓调用点一并改;注解用字符串或 `TYPE_CHECKING`。**删代码不在本轮范围内。**
4. 每个保留的名字都要在 notebook 教程里有**真实教学用途**(仅 import、或 `x = SomeClass` 这种凑数一律不算)。
5. **`__version__` 六仓一律保留** —— 本项目被同名影子包咬过不止一次,它是唯一能写出"我 import 的是哪一份"守卫的探针。

收尾证据:API1 commit `1f17c8c`;API2 commit `97fe123`;基线导入修复 commit `71f1a44`。`pytest -q` 全绿(64 tests)，`__all__` 与 `docs/contract.md` 双向一致，真实顶层公开名为空，notebook 输出版本身份且无错误。将 `from __future__ import annotations` 临时突变回去后，真实命名空间守卫按预期失败并报告 `('annotations',)`，恢复后复绿。

## 阻塞记录

(受阻时追加:日期 / 清单项 / 原因 / 已 commit 的进展)
