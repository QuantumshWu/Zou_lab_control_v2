# 06-E — 根入口、依赖、launcher、工具与测试引导审查

状态：本子阶段完成。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：根 `pyproject.toml`、`zou_lab_control_v2/`、`conftest.py`、`bin/`、根小文档/配置及 `tools/why_was_this_site_dropped.py`。
限制：只读代码与脚本；未运行 `pip`、`git pull`、Vivado、硬件或任何 launcher；只新增本报告。

## 1. 结论先行

根层的正确骨架是存在的：所有操作者入口最终走同一个 checkout bootstrap，Windows launcher 不复制应用composition，workspace与源码分离，FPGA build/program也明确是恢复工具而不是普通实验路径。

但根层目前同时维护三种互不完全相容的产品形态：

1. README/launcher定义的 **source-checkout product**：不安装本项目，只把checkout放到`PYTHONPATH`；
2. 根`pyproject.toml`定义的 **one distribution**：发现并打包八个`zlc_*` package，声明四个console scripts；
3. 八个子package各自的 **standalone distributions**：独立版本、依赖、console scripts和contract tests。

这不是单纯的文档措辞。三种形态分别维护层列表、依赖、版本、入口和import检查。当前实验机实际依赖第一种，但根metadata与大量测试仍保护第二、第三种的部分形状。需要用户明确选择最终交付模型；否则任何一次依赖、入口或package边界修改都要同步多份truth。

另有两条高置信问题：

- 根依赖注释称“union并保留子层pin”，实际`zlc_atom`钉`numpy==2.4.2`、`scipy==1.17.1`，根只写`numpy`、`scipy`；`install_requirements.bat`又只安装根列表，因此实验机没有子层声称的数值版本约束。
- `tools/why_was_this_site_dropped.py`复制了Calibration site detector的科学算法；production已经拆成evidence/admission/refinement多步，而工具保留旧的单体近似，仍宣称“computed exactly”。它可能给出错误的拒绝原因。

## 2. 产品入口与package truth

### ROOT-001（P0/P1 architecture）— “one distribution”与“never installed”没有共同owner

现状证据：

- 根README明确说checkout不做`pip install`；launcher设置`PYTHONPATH=<checkout>`并执行`python -m zou_lab_control_v2 ...`。
- `zou_lab_control_v2.__init__`运行时把八个`packages/<layer>/src`插入`sys.path`。
- 根`pyproject.toml`又定义`name=zou-lab-control`、`version=2.0.0`、八个find roots和四个`project.scripts`。
- 静态执行与setuptools相同的`find_packages(where=...)`得到40个`zlc_*` packages，但**不包含根bootstrap package `zou_lab_control_v2`**。
- 根console scripts又直接指向`zlc_workbench.apps.*`，并不经过checkout bootstrap。
- 每个子层仍有自己的`pyproject.toml`、版本和依赖；`zlc_workbench`还重复声明`zlc-task-console`。

因此根metadata若被安装，得到的是另一种可工作的“安装式八包集合”，而不是README/launcher承诺的checkout bootstrap产品；`python -m zou_lab_control_v2`也不会来自该根distribution。当前正式路径从不安装它，所以`project.scripts`与root version对实验机属于未执行metadata。

裁决：`USER DECISION / REDESIGN`。推荐二选一：

- **推荐A：真正单一可安装产品**。根distribution包含bootstrap或不再需要bootstrap；八层只作为内部packages，依赖和版本由根lock/metadata唯一管理。
- **B：明确source-checkout产品**。删除/降格根install metadata与死console scripts；保留一个machine-readable product manifest作为层、依赖与入口SSOT。

不能继续把A与B各做一半，再用测试要求两者长期同步。

### ROOT-002（P0 reproducibility）— 实验机依赖安装绕过子层pin且没有lock

根`install_requirements.bat`只读取根`project.dependencies`并执行一次`pip install`。它不安装根项目，也不读取子package metadata。根列表与子层不等价：

| dependency | 根 | 实际具体子层 |
|---|---|---|
| matplotlib | `==3.10.8` | plot同样pin，吻合 |
| numpy | 无pin | atom `==2.4.2`；其他层较宽 |
| scipy | 无pin | atom `==1.17.1`；plot较宽 |
| Python | `>=3.11` | ui/pulse/data还声称支持更旧版本 |
| optional notebook/profile | 未装 | plot单独声明；普通实验路径可不需要 |

根注释“Pinned where a layer pinned it”与事实冲突。没有lock file、hash或已验收environment manifest；`update.bat`先pull、再可能升级依赖，失败后“checkout previous commit”也不能恢复已经改变的Python环境。

裁决：`REDESIGN`。用户需要确定：

1. 数值/绘图库是否由一份实验机lock精确固定；
2. 是否允许`update.bat`改变依赖，还是只检查lock并给出显式upgrade步骤；
3. 如何记录一次实验run使用的Python与关键dependency版本。

建议：一份产品lock/constraints是唯一部署truth；package metadata只表达兼容范围。不要手工复制pin到九份文件。

### ROOT-003（P1 SSOT）— 八层清单至少维护四份

当前平行清单包括：

- `zou_lab_control_v2.LAYERS`；
- 根`pyproject.toml`八个find roots；
- `zlc_workbench.tools.check_environment.OWNED`；
- `notebooks.md`表格；
- root pytest testpaths和多个dependency/public-API guards也再列一遍。

bootstrap注释称路径列表是“one answer”，实际只有`conftest.py`复用了它。新增/改名package必须人工同步其他清单，现有`test_declared_dependencies`已经因硬编码distribution mapping漏掉`zlc_workbench`而对真实反向import false-green。

裁决：`REDESIGN`。推荐一个root product manifest生成/投影find roots、testpaths、bootstrap layer order与environment check；notebook表只是说明，不参与runtime truth。

### ROOT-004（P1 correctness）— checkout归属检查使用字符串前缀

`_already_loaded_elsewhere()`用：

~~~python
str(Path(origin).resolve()).startswith(str(ROOT))
~~~

这不是路径包含关系。`C:\repo-copy\...`会被`C:\repo`误认为内部；Windows大小写/符号链接也不由字符串前缀可靠表达。

裁决：`PASS WITH FIX`。保留bootstrap拒绝stale import的机制，改为resolved path的`is_relative_to(ROOT)`或等价component-aware检查，并增加同名前缀sibling old-red。

### ROOT-005（P1 test isolation）— root conftest为bare sibling test imports污染全局path

`conftest.py`在bootstrap后把八个tests目录全部插到`sys.path`。原因是部分suite用`import test_console_presenter`之类复用test double。结果：

- 测试之间存在未声明依赖；
- 所有tests目录可shadow普通top-level import；
- 单package tests与root tests的import语义不同；
- `--import-mode=importlib`只解决同名test module collection，不解决裸helper依赖。

裁决：`REDESIGN TEST INFRA`。共享doubles应进入明确`tests_support` package或同package fixture module；随后删除tests目录全局注入。bootstrap本身仍应在root conftest最早加载。

## 3. 逐文件与函数裁决

### 3.1 `zou_lab_control_v2/__init__.py`

| 符号 | 裁决 | 理由 |
|---|---|---|
| `ROOT` | `PASS` | checkout物理root唯一且直接。 |
| `LAYERS` | `KEEP / MOVE TO PRODUCT MANIFEST` | dependency order必要；与metadata多份重复。 |
| `layer_source_paths()` | `PASS after manifest` | 简单、可测；不应另存layer list。 |
| `_already_loaded_elsewhere()` | `PASS WITH FIX` | stale-import loud fail价值高；路径前缀判断错误。 |
| `_install_paths()` | `PASS for checkout mode / USER DECISION` | 当前正式launcher需要；若转安装产品应删除全局path mutation。 |
| import-time `_install_paths()` | `PASS for explicit bootstrap` | 这正是模块用途；不能让普通foundation import产生同类副作用。 |

### 3.2 `zou_lab_control_v2/__main__.py`

| 符号 | 裁决 | 理由 |
|---|---|---|
| `APPS`/`TOOLS` | `KEEP / PROJECT-MANIFEST SSOT` | 统一dispatch正确；与root scripts、batch wrappers重复列名。 |
| `main()` | `PASS WITH DEBT` | 一个入口正确；用`inspect.signature`猜tool是否收argv以及fallback到private`_main`是小型隐式协议，应该统一所有entry为`main(argv)->int`。 |
| dynamic imports | `PASS` | 避免提前加载Qt/hardware；错误能直接暴露。 |

### 3.3 `conftest.py`

`REDESIGN`：bootstrap加载正确；tests目录注入应随裸test imports移除。当前注释“every bare-imported name is unique: checked”不是runtime invariant，也没有独立manifest owner。

### 3.4 根`pyproject.toml`

| 部分 | 裁决 |
|---|---|
| build backend | `PASS`，仅当用户选择installable root product。 |
| project version/description | `USER DECISION`；当前正式source路径从不消费。 |
| dependencies | `REDESIGN`；不是子层pin union、没有lock。 |
| project scripts | `MERGE`；与`__main__.APPS/TOOLS`和batch wrappers重复，且绕过bootstrap。 |
| package discovery | `REDESIGN`；打包八层但漏root bootstrap，形成未说明的另一产品模式。 |
| pytest config | `PASS WITH TEST-ISOLATION DEBT`；统一root run正确，依赖root conftest path hack。 |

### 3.5 操作者launchers

| 文件 | 裁决 | 理由 |
|---|---|---|
| `bin/_launch.bat` | `PASS` | 所有GUI走唯一bootstrap、解释器resolver与当前cwd；无第二composition。Delayed expansion会破坏含`!`的路径/参数，属低优先Windows边界。 |
| `experiment.bat` | `PASS` | 仅转发TaskConsole。 |
| `pulse_editor.bat` | `PASS` | 仅开发/直接编辑入口，不创建第二实现。 |
| `figure_viewer.bat` | `PASS` | 独立archive viewer入口合理。 |
| `run_server.bat` | `PASS WITH DEBT` | 唯一remote server入口、config显式；参数解析/帮助有batch重复，但无第二server。 |
| `estimate_resources.bat` | `PASS` | 只读容量入口、复用product CLI。 |
| `build_and_program.bat` | `KEEP AS RECOVERY TOOL` | build/program本来有物理side effect且UI明确；source hash用于skip有价值。hash不含toolchain/version/environment，不能称reproducible bitstream，只能称同列举source未变。固定temp/hash文件也不支持并发build。 |
| `install_requirements.bat` | `REDESIGN` | 唯一列表读取是优点；列表本身不是lock/真实子层约束。 |
| `update.bat` | `REDESIGN DEPLOYMENT` | fast-forward与loud failure正确；pull+环境mutation不可原子rollback，和“previous commit still works”不一致。 |

`packages/zlc_pulse/fpga/_resolve_tools.bat`是launcher共享的tool discovery owner，归Pulse报告；根launchers复用它而未复制逻辑，层级可以接受。若最终拆离FPGA package，通用Python resolver才应上移，不应现在再复制一份。

### 3.6 `tools/why_was_this_site_dropped.py`

裁决：`REDESIGN / MERGE SCIENCE OWNER`。

保留“给指定像素解释哪个gate拒绝”的诊断能力很有价值；问题是实现复制了科学检测：Gaussian/background、MAD threshold、binomial admission、local maxima、conditional brightness和dedupe。production `calibration.py`已经拥有更完整的evidence/admission/refinement路径，工具仍声称exact，且没有测试证明两者同输入同gate结果。

推荐把per-pixel evidence/reason变成Calibration owner的纯诊断API，CLI只负责载入文件、坐标转换与打印。不要让工具反向成为第二个detector owner，也不要用源码copy test锁住两份。

### 3.7 小文档与repository config

| 文件 | 裁决 |
|---|---|
| `.gitignore` | `PASS`；workspace/build/cache分离明确。需另有实验数据备份策略，但不应提交到源码repo。 |
| `.gitattributes` | `PASS`；Windows batch CRLF与固定JSON LF理由明确。 |
| `README.md` | `REDESIGN AFTER PRODUCT-MODE DECISION`；checkout操作说明有价值，“one distribution”与“nothing installed”需统一。 |
| `notebooks.md` | `PASS WITH DEBT`；学习顺序可留，不能再暗示每个standalone facade都是产品架构authority。 |
| `HANDOFF.md` | `MOVE TO HISTORY/DELETE`；已标historical但仍指旧两份“唯一权威”，与本次用户裁决冲突。保留会继续被误读为恢复入口。 |
| `AGENTS.md` | `USER DECISION`；repo流程规则可留，本次“冲突由用户裁决”已明确覆盖其autonomous authority。 |
| root architecture/plan | 本报告不在此逐句裁决；统一进入后续文档矛盾矩阵。 |

## 4. 测试裁决

| 测试/守卫 | 裁决 | 缺口 |
|---|---|---|
| Workbench app fresh-process smoke | `KEEP` | 有效证明composition可建；`--check`不运行真实event loop/长期live。 |
| Windows batch smoke | `KEEP` | 正确覆盖root launcher与CRLF；主要只走virtual/offscreen。 |
| environment path check | `KEEP WITH MANIFEST FIX` | 能防wrong checkout；`OWNED`又是一份layer truth。 |
| package import purity/public allow-list | `KEEP MINIMAL / DELETE DOC-SHAPE LOCKS` | 真依赖边界有价值；从旧contract文本/硬编码列表推导的形状守卫会维护历史API。 |
| `test_declared_dependencies` | `REDESIGN` | 已漏真实`zlc_atom -> zlc_workbench`反向import，当前false-green。应从AST/import metadata与product manifest推导。 |
| FPGA asset/source digest tests | `KEEP` | 可防漏文件/geometry drift；不证明Vivado、timing、bitstream或真板行为。 |
| root installability test | `MISSING` | 当前无人明确声明应安装，所以先裁决product mode，再写一条真正行为测试。 |
| dependency lock parity | `MISSING` | 根“union/pins”注释无守卫。 |
| diagnostic CLI parity | `MISSING` | 工具可能与detector漂移。 |
| bootstrap sibling-prefix path | `MISSING` | 字符串前缀bug未覆盖。 |

## 5. 需要用户裁决

1. **部署模型**：source checkout only，还是正式单一installable distribution？审计推荐长期采用真正单一installable/locked产品；若短期保留checkout，删掉会误导的半安装模式。
2. **standalone八包是否仍是产品需求**：若否，逐步删除子package独立version/console scripts/旧contract形状；保留代码层依赖边界，不等于必须独立发布。
3. **实验机依赖升级策略**：自动`pip install latest-compatible`，还是严格lock并显式升级/回滚。审计推荐lock。
4. **root updater权限**：是否允许一个双击动作同时pull源码并mutation Python环境；建议分成check/update两步并留下previous-environment receipt。
5. **site-drop诊断**：保留该功能并合入Calibration纯API，还是删除一次性CLI。审计建议保留功能、删除复制算法。
6. **历史HANDOFF/goal/acceptance文档**：删除，还是移动到统一`docs/history/`且任何测试/恢复流程不得读取。审计建议后者只保留必要provenance，其余删。

## 6. 推荐收口顺序（未实施）

1. 用户先定部署模型与八包独立性。
2. 建唯一product manifest，投影layers、entrypoints、testpaths和environment check。
3. 建实验机dependency lock/receipt；让update只执行有定义的部署事务。
4. 修bootstrap路径包含检查与测试support imports。
5. 把site detector诊断原因放回Calibration owner，CLI变薄。
6. 按部署裁决删除root/subpackage重复metadata、scripts与旧contract shape guards。
7. 最后整理旧docs；不得先靠改文案掩盖三种产品模式仍同时存在。
