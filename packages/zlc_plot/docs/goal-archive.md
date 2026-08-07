# GOAL 归档 —— 已完成轮次(只读参考,不再是工作清单)

> 这些轮次的清单已全部勾选完成。保留在此供追溯设计裁决与实测数据;
> **当前工作清单在 `GOAL.md`**,不要从本文件选活。

## V 轮:验收修复(先做;证据全在 docs/acceptance-*-2026-08-04.md)

- [x] V1 **🔴 事故根治:zlc_data 命名空间冲突**(用户全局环境 `import zlc_plot` 已实际 ImportError;树内还有 src/zlc_data 与 vendor/zlc_data 两份 tracked 副本):采纳验收方案②——捆绑的名字轴版**改名私有化为 `zlc_plot._zlc_data`**(全仓 import 改写;pyproject 停止发行顶层 `zlc_data*`,顶层名让给角色轴新仓);src/vendor 双副本收敛为私有一份,vendor/README 的移植计划保留并列入开放问题(中期按它移植到角色轴 API 后删除私有副本);主 README 安装节写明:本仓自此可与 `zlc-data`(角色轴)同环境安装。判据:干净 venv 同装 `zlc-plot`+`zlc-data` 两包,`import zlc_plot` 与 `import zlc_data`(角色轴)双活,99+ 测试绿。
  验收证据: `tests/test_namespace_isolation.py` 在同一子进程守卫双命名空间与职责，完整套件 100 passed；临时 wheel 仅含 `zlc_plot/_zlc_data`，不含顶层 `zlc_data`。
- [x] V2 **🔴 金样可移植性**:pyproject 钉 `matplotlib==3.10.8`(干净 venv 实测 3.11.1 下三 kind 金样全挂 delta 255——"99 绿"目前不可移植);anywidget 已入依赖但版本没再 bump——bump 版本 + README 加 editable 重装提示;金样第二容差条款是死断言(`test_plot_session_golden.py:71-73`),改为二选一语义或删死码。
  验收证据:项目版本 1.1.0、golden 三 kind 通过，>1 灰度漂移比例与 max 两条均有意义。
- [x] V3 **🔴 fit 模型数学冻结锚**(mutation 实证:把 `_gaussian_offset` 指数 -0.5→-0.25,99 全绿——现测试用同一 evaluate 合成数据且 initial 给真值,自洽简并):为 8 个内建模型各建一组**冻结数值锚**(固定参数→曲线值表,一次生成后冻结入 `tests/fixtures/`,不 import 实现再生);参数恢复测试 initial 不给真值。判据:指数 mutation 必红。
  验收证据: `tests/fixtures/fit_anchors.json` 覆盖 8 模型；临时 `-0.5→-0.25` mutation 使锚测试失败。
- [x] V4 文档三处残句清理(bf68290 已删"浏览器 overlay 本地绘制"机制,文档仍在描述):README.md:100、docs/api.md:553-559、docs/architecture.md:151-153(后者与同文件 L207-209 自相矛盾);performance.md 行数戳更新;a93b905 审计措辞如实修正(registry=封闭清单守卫而非分派收敛,143 处 kind-isinstance 仍在;防御收缩实为 ~3-5%;净增 375 行入账)。
  验收证据:README/API/architecture 已统一为 kernel 单 raster authority；当前源树为 25,825 行非空 Python（含空行 28,627）。历史验收报告中的 25,012 是当时快照数，不作当前计数。
- [x] V5 小项打包:`test_live_channel.py` 的 publish 挪进 counter_lock(理论竞态,free-threaded 下会炸);usage.ipynb cell 16 的 rolling 指标补断言;**backends.py 未提交的 ipykernel wake timer(+46 行)提交为独立主题并补 SimpleNamespace 假 kernel 测试**(zlc_ui 同款先例);`.browser_check/` 入 gitignore 或删;`.claude/` 同。
  验收证据:live channel/backend tests 通过；wake timer 独立提交，两个临时目录已 gitignore。
- [x] V6 两项**用户追认记录**(默认裁决已给,写进 docs 即可,用户否决再改):① notebook 拖拽预览=kernel 烘焙(设计从"客户端 AxisTransform 本地画"改道,单渲染器一致性上位、预览延迟受 RTT 影响;**默认追认**——本地 Jupyter 手感达标即可,远程手感不行时按原设计加客户端预览作为增强,front 已带冻结 AxisTransform,门留着);② Rolling 静态快照 R 轴种子化为逐 shot 历史(vendored 参照没有的扩展,成文理由已在测试里;**默认保留**,README 记语义)。
  验收证据: `docs/acceptance-decisions-2026-08-04.md` 与 README 已记录两项裁决。


## 机械终态判据(全绿才 GOAL COMPLETE)

1. 干净 venv(钉版依赖)`pytest -q` 全绿;**同环境安装 zlc-plot + zlc-data(角色轴)双 import 活**(V1 判据,写成守卫测试)。
2. fit 冻结锚测试对指数 mutation 必红(临时副本验证,记录 commit message);金样保持对 1px 几何/6 阶灰度敏感。
3. `describe_semantics` 驱动的语义组在 pyqt5_embed 可亲手操作(S4 四个验收场景全通),notebook 同权;金样含 facet 2d。
4. grep 为零(src/):顶层 `zlc_data` 发行残留(pyproject include)、文档"overlay canvas 本地绘制"残句。
5. README/docs 零漂移(逐篇校对记录);工作树干净,勾选与 commit 一一对应。



## SE 轮:语义面手动验收修复(用户 2026-08-04 抓出;先读本节逐条证据)

- [x] SE1 人类可读标签单源:SemanticField 的 choices 改 (value, label) 对,label 派生规则唯一落在 semantics.py——AxisRef 用 schema 的 display label/unit("repeat"/"point row"/"scan (mV)"),PlotKind 用 kind 模块声明的 display_name,Reduction 用小写词;Qt/notebook 只消费 label,禁止任何前端自己 str(value)。(实测 combo 显示 "AxisRef(domain=<AxisDomain.POINT_COORDIN..." 截断 repr)
- [x] SE2 combo 卫生:当前值与 choices 恒等去重(实测 x 应 3 项显 4 项);allow_none 显示为 "(none)"。
- [x] SE3 embed 的 kind 语义编辑改真语义:同一数据上走 replace_spec 换 kind(用 SE7 的 default_spec+携带政策),不再跳演示页(现 findData 未命中时静默吞掉=用户"没反应"来源之一);演示页切换只归顶部 combo。
- [x] SE4 演示数据去简并:notebook 与 embed 的 repeat 改为不对称(如 repeat2=信号×1.3+偏移),让 reduction 切换与 group=repeat 肉眼可见(现 notebook curve 数据 np.tile 两 repeat 相同——mean==median、双线重叠,"替换发生了但像素相同"=用户"没反应"主因);demo 即验收台架,简并数据=形状剧场的 demo 版。
- [x] SE5 错误可见性:语义 replace 失败在面板就地红字提示,不只状态栏。
- [x] SE6 回归守卫:offscreen 测试断言——语义 combo 无重复项、标签不含 "AxisRef(",kind 列表==admits 过滤结果;demo 数据非简并断言(至少一处 mean≠median 且 repeat 间不同)。
- [x] SE7 **default_spec(schema, kind) 单源默认投影**:每个 _kinds/* 模块声明默认投影规则(curve:首个 point 坐标否则 point_row + MEAN;image:优先 topology 网格否则双 data 轴,推不出=None;histogram:samples 全集;rolling:无参;facet:不默认),顶层 `default_spec(schema, kind) -> PlotSpec | None`;SE3 的 kind 切换消费它;推不出默认的 kind 在语义组灰掉;契约测试:对合成 schema 族断言 `admits(schema) == (default_spec 非 None)`,两声明永不漂移。(背景:v1 的 coerce_panel_value 静默降维已判死;"自动"的新形态=机械枚举+显式默认,此项是 workbench 面板绑任意信号的前置件)

SE1-SE7 实现已落地；阶段性提交分别记录语义标签/默认投影、embed 语义替换和 notebook 非简并数据，最终验收以完整回归命令为准。


## F 轮:语义 UX 根修(2026-08-04 用户"还是不对"→ 3-agent 审计 → 用户批准"你修",Claude 亲自实现;证据 docs/audit-semantic-ux-*-2026-08-04.md)

全部完成并逐项突变探针红/绿验证(commit 见括号):

- [x] F1 **replace_spec 全事务**(401ca4d):layout 拒绝(facet cell cap)搬进回滚包络;被拒替换绝不留半提交会话(此前 semantics 报被拒 spec、渲染冻结、set_parameter 全废)。`tests/test_replace_spec_transaction.py`。
- [x] F2 **embed 死锁三件套**(cb3cb54):`_track` 错误闭包引用已释放的 `except as` 名→NameError 吞进被丢弃 future→`_switching` 卡死=整窗静默死亡(用户"没反应"的主根因)。局部绑定异常、拒绝上面板红字、成功清错、`_dispatch` 回调守卫可见化;definition.kind 随 spec 同步 + 页守卫改身份比较。`tests/test_embed_semantic_resilience.py`。
- [x] F3 **快轴默认 + kind 三态**(645cf80):curve/image 默认 x=最内层扫描维(此前最慢维=curve 涂抹、image 全转置);FacetGrid 真默认(repeat 非平凡则 repeat,否则最外扫描维×curve 默认 cell);admits=可渲染 与 default_spec=可默认 拆开,注册表契约改蕴含式;可渲染无默认的 kind 灰显不再消失(单程门关闭)。
- [x] F4 **语义域=投影可行性**(2ee4515):`semantics.updated_spec` 单源组合(embed 手搓路由删除);session 用真 replace 验证前半(`_prepare_replacement`+`_surface_plan_for` 提取共用)逐项探测每个选项,不可行项灰显+真实拒绝理由 tooltip;按候选 spec 缓存至 generation 变化(demo 冷 ~0.7s/热 ~0.3ms)。性质测试:启用项点击必成功、禁用项必拒绝且带理由。`tests/test_semantic_feasibility.py`。
- [x] F5 **标签按角色迁移**(e4da423):每 kind 注册表声明 label 槽的语义角色(axis(x)/value/count/repeat/title),`session_policy.merge_labels` 按角色搬运——curve 的 y 值标签迁到 histogram 的 x 槽,时间轴标签正确丢弃(此前逐字拷贝致 histogram x 显 "Time (mV)");embed 逐字携带删除。`tests/test_label_carry.py`。
- [x] F6 **同索引重选原样恢复**(b2eae34):语义切 kind 后 combo 仍停原页,同索引点击被 currentIndexChanged 吞掉→页 combo 改接 `activated`,原样 demo 永远一击可回。


## G 轮:验收三根因(2026-08-04 用户真机验收抓出"rolling 负 points/curve 下拉死/kind 换数据",Claude 亲自实现)

全部完成,每项突变探针红/绿(commit 见括号):

- [x] G1 **reduction=未指派轴的命运**(1e7993f):`_validate_axis_coverage` 的 dense 轴拒绝删除——它把 group 域锁死(site 轴数据上除当前值全被禁,"选了也没变"的根因),且与 facet 路径(本就池化 dense 轴)不一致。裁决:每轴恰一命运(x/group/facet/samples/reduction 坍缩);reduction 是 spec 显式字段=授权,非 v1 静默 coerce。点域全未引用仍报错(scan 是图的本体,rolling/histogram 才是坍缩它的 kind)。默认随动:恰一 dense 轴仍 group,多个坍缩;topology image 不再因 dense 轴拒默认。验收链测试:group 域 (none)/repeat 启用→ungroup 单线==池化 mean 数值→median 像素可辨。`tests/test_projection_coverage.py`(数值断言)+ `test_semantic_feasibility.py::test_user_can_reach_a_single_mean_line_on_grouped_data`。
- [x] G2 **rolling x=绝对 shot 序号**(3269749):负 "Shots ago" 轴删除;`RollingHistoryPoint.shot_index` 单源持有绝对序号(seed 0..R-1,live 递增;window 缓存截断 history,列表位置推不出序号所以点自持)。payload 是标签唯一权威("Shot"),rendering fallback 从 payload 派生不再重述字面量。`tests/test_rolling_shot_axis.py`(seed/递增/窗口滑动三断言)。
- [x] G3 **demo 页=数据源**(0f5bcc3):窗口原有两个 kind 选择器,顶部 combo 条目名叫 Curve/Histogram/…实际换 demo 页=静默换示例数据("kind 换数据"的根因)。重构:3 个数据源页(Scan dataset/Camera image/Pulse timeline)带 authored 初始 spec;kind 唯一入口=语义面板,切 kind 重投影同一数据;`_DemoDefinition` 去 kind 身份;标题命名数据源,角色携带跨 kind 恒真。
- [x] G4 **rolling 双域残余清除**(aad97ca):负轴死后 `_rolling_axis_domains`/`_rolling_fit_scale`/`CurveSeries.plot_x` 与 8 处 np.interp 全是恒等残余,且 interp 对域外输入 clamp——notebook 教程旧负坐标选区被夹成 (0,0) 退化 → `set_area_selector` 抛错(用户实测)。全部删除,恒等显式化(rolling 占位 x ref 绝不进坐标单位换算);notebook 选区改绝对 shot 坐标(由拍数派生),全 notebook offscreen 执行零错误。突变探针:重引入 clamp interp 即红。


## H 轮:验收再修(2026-08-04 用户"还是巨量问题":选项重复/不可选、结构不可见、切 rolling 卡 facet 卡死)

- [x] H1 **探针验证/计算分离**(b5265e3):profiling 实锤切 facet 的 replace=128.6s——可行性探针为每个候选构建完整 payload,facet_rows=point_rows 候选逐 cell 聚合 1281 格。架构修:DataView 抽 validate_curve/image/histogram/rolling/facet 单源(投影方法开头共用),kind 注册表 `validate` 职责,探针=构造器+view+validate+facet 容量(域大小合成 FacetTopology→resolve_surface),零聚合。validate_facet 顺带补上 facet cell 一直绕过验证的旧洞。facet 容量统一:1D auto-pack 同受 facet_max_cells 限(此前 1281 格绕过 2D-only cap 在布局层崩 box 错误)。实测:切 facet 128,615ms→503ms,rolling 112ms。机械守卫:describe_semantics 期间 `_build_payload_from_view` 零调用(spy,突变探针红)。
- [x] H2 **域=可用集+身份去重+结构可见**(8835662):用户裁决"只给可以用的"——disabled/理由/灰显/tooltip 整链删除,describe_semantics 逐项过滤,当前值恒留;拓扑维吞同名 point 列(同一物理轴单一身份,demo x 域 7 项→3 项全可选);`schema_summary(schema)` 单源结构描述("repeat 24 × scan (Field 21 × Time 61) × Site 4 → Signal (mV)"),embed 头部逐字显示。性质测试收紧:域内每项提交必成功(突变探针 5 红)。
- [x] H3 **规格三裁决+忙碌不丢点击**(72fcbf4,-674 行):①`HistogramPlot.samples` 删除——纯仪式字段(验证不筛选,池化恒全量),多选编辑器"很奇怪"的根源;直方图=全体值分布,切片归 selection/facet;multiple/MULTI_CHOICE 编辑器链同删。②`FacetGridPlot` 单 facet 轴——轴定 cell 数,行列=布局自动优化(facet_shape/facet_shape_for_aspect 按最大占据),facet_cols/2D shape/行列单位参数/2D golden 全链删除;超容量错误带数字且 replace_spec 全事务=对外 API 契约(返回错误、图不动)。③轴选项标签去单位(选身份非量值)。④embed 忙碌期语义点击 pending-latest 不丢弃+可见 rebuilding 状态(真机"10s 没反应"=静默丢点击后的用户重试循环;offscreen 实测切换本身 0.4-0.5s)。


## I 轮:zlc_data 和解 + facet 全 cell live fit + headline(2026-08-04 用户裁决)

> **顺序更新(2026-08-04)**:zlc-data W 轮已完成。I1 直接消费角色轴 `OwnedSnapshot`;显示单位换算与 latest-only channel 仍由呈现层负责，且不再保留第二份数据模型。
> **显示效果冻结铁律**:用户对当前 GUI 显示效果已验收满意。I1 是纯底层替换:rgba 金样(tests/goldens/)一像素不许变,全部 offscreen 行为测试语义不变;任何"顺手改进显示"都是违规。

- [x] I1 **和解:删私有数据副本,改用真 zlc-data(role-axis 包)**
  ① 先调研后动手:diff `src/zlc_plot/_zlc_data` 与已安装 `zlc-data` 包的 API(Axis/DatasetSchema/DatasetSnapshot/PointTable/PointTopology/单位系统),把映射表写进 `docs/zlc-data-reconciliation.md`(旧名→新名/语义差异/缺口),先 commit 这份文档再改代码。
  ② 全仓 import 改向 + 适配;`src/zlc_plot/_zlc_data/` 目录删除;pyproject 加 `zlc-data` 依赖。
  ③ **zlc-data 缺 zlc_plot 需要的能力时:记阻塞报告用户,绝不在 zlc_plot 里再造影子副本**(V1 双包 ImportError 事故的根教训)。
  ④ 既有"zlc-plot 与 zlc-data 同环境双 import 存活"守卫改写为新形态(同一环境 import 两包并交换 snapshot 对象)。
  判据:旧数据目录不存在;`grep -rn "_zlc_data" src tests examples notebooks` 零;金样逐像素全绿;`pytest -q` 全绿;usage notebook 全执行零错误。
- [x] I2 **facet fit:全 cell live + 主视图曲线**(用户拍板"当然是全 cell live")
  ① 投影层:facet 批量拟合为每 cell 产出 `FitOverlay` 折线,构造复用现有单 cell overlay 单源(`_fit_projection` 的 FitOverlay 构造点),`FacetFitBatchResult` 携带 `overlays`;绝不写第二套折线生成。
  ② live:armed live fit 在 FacetGrid 下=每个 data revision 对全部 cell 重拟合(fit worker 线程,latest-only:新 revision 到来丢弃过期批次);批次完成整体提交,期间保留上一批 overlay(与单图 live fit 的过期策略同源,不另立规则)。
  ③ 渲染层:overview 画每 cell 折线和单行 headline 注释，不画完整参数框；focus 视图行为不变(全参数框)。渲染分支锚定在 `rendering.py:784` 的 overview gate，文本详细度由 `_build_fit_topology` 的三态 detail 决定。
  判据:offscreen 测试——facet live fit armed 后 publish 两个 revision,断言每 cell 折线和 headline artist 存在且随 revision 更新、focus 有完整参数框;突变探针:把 overview 折线门改回 None,测试必红。
 - [x] I3 **headline 参数**(用户确认:模型声明制,非用户配置;J1 将其呈现从标题后缀改为 cell 注释)
  ① `FitModelSpec` 增 `headline: str`(必填,须是 parameter_names 之一,注册时校验);现有各模型声明:gaussian 族→中心 x0(radial 族→x0)、exp/decay 族→时间常数、linear→斜率,其余按物理最关心量定,列在 commit message。
   ② facet overview 的 headline 物理符号/数值由 J1 统一绘制为 cell 左上角单行注释；焦点/单图保持完整参数框。
   判据:文本断言(cell 注释含 headline 符号、值与误差);FitModelSpec 缺 headline 或名字不在参数表=注册即抛。

### I 轮机械终态判据
1. `pytest -q` 全绿;金样零像素漂移;notebook 全执行零错误。
2. `_zlc_data` 三处 grep 零;pyproject 含 zlc-data。
3. facet live fit 的 offscreen 断言 + 突变探针记录在 commit message。
4. headline 注册校验测试在;渲染文本断言在。


## J 轮:facet fit 的呈现与衍生(2026-08-05 用户裁决;先读本节全文,含实测基线)

> **本节所有判断已机械化**:凡需要取舍的地方都给了确定的数字、阈值或规则,实施者不需要自己判断"够不够好/值不值得"。遇到本节没覆盖的取舍 → 记阻塞问用户,不要自行发挥。

> **本轮修订一条已勾选的旧规格**:I3 原先把 headline 写进 cell 标题,用户实测判否——标题太长,且内部命名不应出现在物理图面。新规格见 J1;`docs/api.md` fit section、I3 条目描述、以及 I2③ 的渲染锚点一并改写,不留两套说法。当前 overview gate 是 `rendering.py:784` 的 `overview = isinstance(self.spec, FacetGridPlot) and self._facet_focus_index is None`；overview 画曲线并在 cell 左上角画单行 headline，focus 才画完整参数框。
>
> **实测基线一(批量 fit 成本归因)**:64 cell × 40 点 gaussian_offset = 95ms,严格线性 1.49ms/cell。归因:solver 91%(其中**有限差分 Jacobian 占 35%**),逐 cell 取数仅 9%——`_with_context` 是 `copy(self)` 且共享 `_view`/`_payload` 引用,`payload` 是裸属性读取无惰性重建,所以"逐 cell 拆数据"并没有隐藏代价。
> **实测基线二(三方对照,已 settle)**:"不拆开、直接向量化"确实更快,而且最好的做法不是"合成一个大解",是"N 个独立解的迭代同时推进"——完整数字与机制见 J3。

- [x] J1 **headline 改为 cell 内左上角注释,删标题后缀**(用户裁决:**要显示误差**;字号固定 **3.5pt**)
  ① `_build_fit_topology` 的 `annotation: bool` 换成三态详细度(NONE / HEADLINE / FULL);overview 传 HEADLINE,焦点与单图传 FULL。
  ② 文本组装单源化:抽出"一个 `FitParameterDisplay` → 一行文本"的唯一格式化函数(数值继续走既有 `_fit_parameter_value_text`),FULL 与 HEADLINE 都用它。**HEADLINE = 只有 `overlay.headline_parameter` 一行,带 ± 标准误与单位,无公式、无其他参数**,形如 `$x_0$ = 1.23 ± 0.04 ms`。留出的空间正是省下其他参数换来的(用户明确意图,别再压缩)。
  ③ **符号取 `FitParameterDisplay.label`**(内置模型已声明 LaTeX 符号 `$x_0$`/`$\mathrm{FWHM}$`/`$H$`/`$B$` 等;`label = display_label or name`),**绝不出现 "headline"/"main" 字样**。
  ④ **误差缺失要优雅**:`standard_error` 仅在 `result.covariance_valid` 时才有值(`_fit_projection.py:1484`),为 None 时按既有全参数注释的写法输出 `± n/a`,不许崩也不许静默吞掉整行。
  ⑤ 样式:位置沿用单图注释既有单源(`style.render.axes_text_inset_fraction` 左上、`palette.fit_text`、`artists.fit_annotation_zorder`);**字号新增一个 style token,默认 3.5pt,不随 facet tier 缩放**(注:`fit_annotation_pt=3.25` 是单图用的,cell 标题是 `tick_pt(6.5) × facet scale`,三者互不相同,别混用也别改动前两者)。
  ⑥ **失败 cell 的行为要显式裁决**:`_build_fit_topology` 在 `family=="failure"` 时,仅当 annotation 为假才立即返回,否则会建一个**硬编码 (0.02,0.98)** 的 diagnostic Text——切到 HEADLINE 后失败 cell 会开始显示诊断文本。本轮决定:**overview 的失败 cell 显示诊断,截断规则与单图同源但用更短的上限**——单图现有规则是超过 72 字符则取前 69 加省略号(`rendering.py:3344-3345`);把这个上限/省略号逻辑抽成一个函数,单图上限保持 72,facet cell 上限定为 **24**(两个上限都做成具名常量,不许写字面量)。同时把失败诊断那个硬编码坐标 `(0.02, 0.98)` 改成与注释同一个 `axes_text_inset_fraction`(消除坐标双源)。
  ⑦ 删除 `rendering.py` 的 `| headline=` 拼接(全仓唯一产地);`_update_facet_fit_titles` 若因此退化为"重设同一个纯标题"则整体删除(顺带消除它对 `tick_pt` 的重复兜底默认,标题由 `rendering.py:2285-2296` 既有路径单独负责)。
  判据:反转既有守卫 `tests/test_facet_live_fit.py:42-43`(现断言 `len(axis.texts)==0` 且标题含 `headline=`)为:每个成功 cell 的 axes 内存在含该模型 headline 符号、数值与 ± 的 Text,且标题**不含** `headline`;焦点视图仍含公式与全部参数;**新增 facet-with-fit 金样**(现 goldens 只有 curve/histogram/image,无任何含 fit 的图);突变探针:把 HEADLINE 改回 FULL 或改回标题后缀,断言必红。

- [x] J2 **让 facet 批量结果成为一张完整的、带单位的列式数值表**(派生数据集本身归 zlc_runtime,见其 T 轮)
  > 归属裁决:ROI/fit 的**派生与发布**在 zlc_runtime(SelectionBridge,S 轮已完成标量版);zlc_plot 只负责把拟合结果暴露成**纯数值**,不构造数据集、不 import zlc_runtime。**完整的定义 = 上面钉死的跨仓数值契约的全部字段,一字不多一字不少**;本项只补这些字段的暴露面。
  ① 逐 cell **标准误**的列式访问(现只有 `parameter_values` 给值;标准误埋在每个 cell 的 `FitOverlay.parameter_display` 里,没有列式入口)。协方差无效的 cell 该位置为 NaN 并由 success 掩码区分。
  ② **参数单位**:经既有单源 `_display_fit_parameters` 取,批量对外给出 `{参数名: 单位}`;不许在别处重新推导单位。
  ③ **facet 坐标身份**:批量对外给出 facet 轴的名字、逐 cell 坐标值、坐标单位(现只有 `AxisRef` 与裸 `facet_values`);坐标为 TEXT 时同时给出序号(消费者可用序号当坐标、文本当标签)。
  ④ 单图 `FitResult` 提供同形状的退化访问面(N=1),让消费者一条路处理两种情形。
  判据:数值断言——列式标准误逐 cell 等于该 cell `FitOverlay` 里的标准误;单位等于该参数在当前显示单位下的单位;失败 cell 的 success 掩码为假且值为 NaN;TEXT 坐标情形序号为 0..N-1。**不得出现任何 `OwnedSnapshot` 构造**(grep 判据:`fit.py` 仍不 import zlc_data)。

### 跨仓数值契约(zlc_plot J2 产出 == zlc_runtime T2 消费;两仓逐字相同,任一方改动必须先改这段)

facet 批量拟合对外的**纯数值表**(不含任何 zlc_plot / zlc_runtime 对象):

- `parameter_names: tuple[str, ...]` —— 参数的程序名(非符号),顺序即列顺序。
- `parameter_units: Mapping[str, str]` —— 参数名 → **与 `parameter_values` 同一系统**的单位字符串;无量纲为 `""`。
  > **铁律:值与单位必须同系统。**要么两者都 canonical,要么两者都当前显示单位——**绝不允许值是 canonical 而单位报显示单位**(消费者会给 canonical 数值贴上显示单位标签,产生 10^n 倍的物理错误)。本契约选定:**两者都用 canonical**(派生数据进入信号面后由呈现层自行换算,与 zlc_data 的"unit 字段=canonical 注解"归属一致)。守卫:改变显示单位不得改变 `parameter_values` 与 `parameter_units` 中的任何一个。
- `parameter_values: Mapping[str, np.ndarray]` —— 参数名 → float64 数组,长度 = cell 数,失败 cell 为 NaN。
- `parameter_errors: Mapping[str, np.ndarray]` —— 同形状标准误;失败 cell 或协方差无效为 NaN。
- `success: np.ndarray` —— bool 数组,长度 = cell 数,表示**该 cell 的拟合是否成功**。
  > **值数据集**的 validity = `success`。**误差数据集**的 validity = `success AND isfinite(error)` —— 因为拟合可以成功而协方差无效(此时标准误为 NaN),把 NaN 写进标记为 VALID 的格子会让下游把 NaN 当成真误差用。两个数据集的 validity **由生产者显式给出,消费者不得用 `isnan` 反推**。
- `sample_axis_name: str` —— facet 轴的显示名。
- `sample_coordinates: np.ndarray` —— float64,长度 = cell 数。facet 坐标为数值时即其 canonical 值;为 TEXT 时为 `0..N-1` 序号。
- `sample_unit: str` —— 坐标单位;TEXT 坐标情形为 `""`。
- `sample_labels: tuple[str, ...] | None` —— 仅 TEXT 坐标情形非空,与坐标一一对应。
- `source_revision: int` —— 本批拟合所依据的**来源数据** revision(服务血缘与 same-shot 族;同一份数据重复拟合时不变)。
- `batch_revision: int` —— **本次发布自身**的单调计数器,**每产生一批就 +1**,与来源数据是否变化无关(下游 `update_data` 要求 schema 恒定且 revision 严格递增,靠的是这个)。
  > 这两个是不同的东西,**不许合并成一个字段**:前者回答"这批结果来自哪一拍数据",后者回答"这是第几批结果"。

单图拟合是 N=1 的退化情形:同样的字段,`sample_axis_name=""`、`sample_coordinates=[0.0]`、`sample_unit=""`。**两侧都只有一条代码路径处理这两种情形**——生产侧必须由同一个函数产出两种形态(不许单图与 facet 各写一份),消费侧必须能接受 N=1 且 `sample_axis_name` 为空的表。**N=1 的 facet(单 cell 网格)是合法输出,消费者不得拒绝。**

- [x] J3a **解析 Jacobian(安全项,必做,全程 scipy;惠及全部拟合而非只有 facet)**
  现在 `least_squares`(`fit.py:831`)不传 `jac=`,scipy 用 `approx_derivative` 做两点有限差分,占 facet 基线的 35%。让 `FitModelSpec` 可选声明返回 ∂残差/∂参数 的函数(高斯/洛伦兹/指数等闭式可导),作为 `jac=` 传给 scipy;没声明的模型保持数值微分回落。**不换求解器、不改算法**,只把导数从数值估换成解析给,顺带减少迭代次数。
  **作用范围是全部拟合,不是只有 facet**:`least_squares` 在全仓**只有 `fit.py:831` 一处调用**,位于 `FitEngine.fit` 内;单图曲线、直方图、图像(radial gaussian)与 facet 批量都经 `_solve_fit_selection`(`_session_fit.py:464/474`)走到它。因此本项只要接在这一处,四条路径同时受益——**不许为 facet 单独开一条带 jac 的分支**。
  **图像拟合的收益可能最大**:radial gaussian 的残差是整幅图像,有限差分要按参数个数把整幅图重算若干遍,解析导数一次算完。
  判据:① 内置模型的解析导数对 scipy 数值导数逐元素校验(rtol=1e-6,在 **≥5 组随机参数点**上,每个声明了解析导数的模型都要覆盖);② 拟合结果与改前 rtol=1e-6 内不变(参数与标准误都比);③ **三条路径各测一次并把前后数字写进 commit message**——单图曲线拟合、图像 radial gaussian 拟合、facet 64 cell 批量;④ **不设下降门槛**(解析导数首先是正确性资产,某模型没变快也保留)。

- [x] J3b **批量拟合引擎(按机械阈值跳过)**
  > **开工阈值(唯一判据,别的理由都不算)**:J3a 完成后,用 J3 台架测 **64 cell × 每 cell 41 点**一批拟合的耗时 T。
  > - **T > 100 ms** → 做 J3b。(100ms 是 `DEFAULTS.live.refresh_intervals_ms` 的最快档;拟合慢过它就跟不上最快 live 刷新。)
  > - **T ≤ 100 ms** → **跳过 J3b**,在阻塞记录写下实测 T 与本判据,并把本项勾选为"按判据跳过"。
  > (参考:改造前实测 64×41 = 65.9ms,已在阈值内 —— 所以本项**大概率应当跳过**,除非 J3a 之后反而更慢。)
  >
  > **scipy 能力实测结论(scipy 1.17.1,别再猜也别再查)**:
  > - 批量**线性代数**:**有**。`numpy.linalg.solve` 与 `scipy.linalg.solve` 都接受 (N,p,p)/(N,p) 堆叠输入。
  > - 批量**非线性最小二乘驱动**:**没有**。`least_squares`/`curve_fit` 签名里无任何 batch 参数;喂二维残差直接 `ValueError: f0 passed has more than 1 dimension` —— 一次调用只解一个问题。
  > - 所以 scipy 原生的"一次拟合多条曲线"就是把残差拼成一个大问题(社区常见做法),即下表的"块对角合并";已实测,**大样本时比现状还慢**。
  > - 因此所谓批量 LM = **只写 LM 的外层控制循环**(阻尼更新、接受/拒绝、收敛判据,约 40 行),**内层数值全交给 numpy/scipy 的批量线性代数**;不是自研线性求解器。
  > **实测三方对照**(同一高斯问题,每 cell 独立 4 参数,收敛结果一致到 ~1e-7):
  > | 64 cell × 每 cell 点数 | 现状(逐 cell scipy) | 块对角合并成一个大解 | **批量 LM** |
  > |---|---|---|---|
  > | 41 点 | 65.9ms | 20.1ms | **3.6ms(18.2×)** |
  > | 121 点 | 67.9ms | 37.4ms | **9.5ms(7.2×)** |
  > | 401 点 | 73.1ms | 117.0ms | **26.7ms(2.7×)** |
  > 算法是**教科书 Levenberg–Marquardt,未作任何改动**(scipy 的 `lm`/`trf` 用的就是它);非标准的只是**实现形态**——把 N 个独立 LM 求解的每一步用 numpy 同时算(雅可比 (N,n,p) 一次算完、正规方程 (N,p,p) 批量求解),而不是调 N 次 scipy。scipy 没有批量 API,这是它做不到的部分。每个 cell 保有自己的参数、自己的阻尼 λ、自己的收敛判据,**失败隔离与逐 cell 协方差因此天然成立**。
  ① 逐 cell 协方差 = 各自 (JᵀJ)⁻¹ 的批量求逆(J1 要显示误差,标准误必须逐 cell 正确)。
  ② **适用门(白名单制,不做判断)**:只有同时满足「已声明解析 Jacobian」且「该模型无 bounds」的模型才走批量路径;其余模型(含任何带 `bounds_initializer` 的)**一律走现有逐 cell scipy**——不实现批量盒约束、不做代价评估。批量路径中未收敛的 cell 单独标失败,并对该 cell 单独回落 scipy 重解一次。
  ③ **scipy 是参照 oracle,永不被替换**:单图拟合与全部回落路径继续用 scipy。
  判据(任一不过就整体放弃本项并回落 scipy,写进阻塞记录):等价性契约测试——同输入下批量结果与逐 cell scipy 的**参数与标准误**均在 rtol=1e-6 内一致,内置模型逐个覆盖;失败隔离测试——人为让一个 cell 无法收敛,其余 cell 结果与不含该 cell 时**逐位一致**;金样与既有 fit 测试全绿。

- [x] J3c **live 热启动(安全项,J3a 后可做)**:同一 cell 上一 revision 的收敛参数作为本次初值(相邻 revision 数据几乎不变,迭代数大降);首拍与拟合失败后回落既有初值策略。判据:拟合结果 rtol=1e-6 内不变;前后计时写进 commit message。

- [x] J4 **清残余**:`FacetFitBatchResult.errors` 改名(现为逐 cell 失败消息,与 `FitResult.standard_errors`/逐参数误差同名不同物,J2 极易写错);删零消费者的 `ArtistStyleConfig.fit_dim_alpha`;修正 GOAL I2③ 的失效行号锚点。

### J 轮机械终态判据
1. `pytest -q` 全绿;notebook 全执行零错误。
2. `grep -rn "headline=" src` 零命中(内部词汇不出现在任何用户可见文本);`fit.py` 仍不 import zlc_data。
3. facet-with-fit 金样存在;J1 突变探针记录在 commit message。
4. J3a/J3c 前后计时与 rtol 不变性记录在 commit message;J3b 若做则等价性与失败隔离判据全绿,若跳过则阻塞记录写明理由。
5. `docs/api.md` 与 I3 条目描述与 J1 一致(无两套说法);I2③ 行号锚点已修正。


## K 轮:J 轮验收返工 + 单位面收敛(2026-08-05 六路对抗验收产出)

> **验收结论**:J1 渲染实测全对(符号/±误差/3.5pt/标题干净/focus 全参数);**J3a 解析 Jacobian 被独立复算证实完全正确**(8 个模型用复步微分逐一重推,逐元素吻合 ≤1.5e-11;6/6 错误导数突变均被杀红,守卫真实且紧)。**J2 判 FAIL,J3c 有一条静默改结果的严重缺陷。**
> **三条根因在我钉的契约里,不是实现走样**(契约已于本轮修正,见上方跨仓契约块):值/单位不同系统、revision 一字段两语义、误差数据集的 validity。K1-K3 是照新契约返工。
> **本节所有判断已机械化**;未覆盖的取舍记阻塞问用户。

- [x] K1 **值与单位必须同系统(物理正确性,最高优先)**:实测——`parameter_values` 恒为 canonical(如 5.005e-4 秒),而 `parameter_units` 跟随**当前显示单位**变化。显示单位设 `ms` 时消费者读成"5.005e-4 毫秒",**比真值小 1000 倍**;设 `us` 时小 100 万倍;只有显示单位恰等于 canonical 才自洽。按新契约:**两者都用 canonical**。
  判据:带真实单位的数据上,`set_axis_unit` 切换 ms/us/s **不得改变** `parameter_values` 或 `parameter_units` 任何一项(参数化测试逐单位断言);数值换算回 canonical 与真值一致。
- [x] K2 **revision 拆成两个字段**:实测同一份数据反复 fit 三次都是 revision 0(live 采集下确实递增,所以 GUI 无感,但静态数据重算后推给 live 会话会被拒)。按新契约拆 `source_revision`(来源数据,血缘用)与 `batch_revision`(每批 +1,单调)。
  判据:同一份数据连续 fit 三次,`source_revision` 不变而 `batch_revision` 严格递增;两次派生结果能连续 `update_data` 进同一 live 会话。
- [x] K3 **误差数据集的 validity**:拟合成功但协方差无效时标准误为 NaN,而当前 validity 只看 `success` → **NaN 落进标记 VALID 的格子**。按新契约:值数据集 validity=`success`,误差数据集 validity=`success AND isfinite(error)`,由生产侧显式给出。
  判据:构造"成功但协方差无效"的 cell,断言误差数据集该格 INVALID 而值数据集该格 VALID。
- [x] K4 **单图与 facet 必须是同一条代码路径**:实测两者已在失败行为上分歧(同数据同模型同不收敛设置,单图 `success=False` 但表里的值与 facet 路径不一致)。按契约"两侧只有一条代码路径",由同一函数产出 N=1 与 N>1 两种形态。
  判据:同数据同模型同 `FitOptions` 下,单图表与 1-cell facet 表**逐字段相等**(含失败情形);删掉其中一条分支后另一条仍能产出两种形态(结构性验证)。
- [x] K5 **热启动不得改变拟合结果(静默错答,严重)**:`fit.py:1170` `if warm_start is not None and seed_index == 0 and candidate.success: break` —— scipy 的 `success` 只表示某个终止条件满足,**不代表这是最优解**;正常路径是跑完所有种子按 RSS 选最优,热启动却在第一个种子"成功"时就短路,**可能锁进更差的局部极小**。修法:热启动只作为**额外的第一个种子**加入候选集,**不得跳过其余种子**,最终仍按 RSS 选最优。
  判据:构造一个热启动种子会落进次优极小的数据集,断言开/关热启动的收敛参数在 rtol=1e-6 内相同;并把 J3c 的等价性测试从 `initial=真值` 改为**用真实初值器**(现在传真值,永远走不到分歧路径)。
- [x] K6 **热启动缓存的失效条件**:`_session_fit.py:54-80` 只在 `result.success` 为假时清缓存;**求解抛异常时陈旧种子存活并被无限复用**。修法:任何非正常返回(异常、取消、超时)都清缓存。判据:强制一次 RuntimeError 后,下一次拟合不得复用旧种子(spy 断言)。
- [x] K7 **补齐无守卫的契约不变量**(验收实证:以下突变**全部**在完整测试套件下存活且改变输出)——① `success` 掩码写死全 True;② 失败 cell 的参数值 NaN 改 0.0;③ `sample_unit` 篡改;④ 失败诊断坐标从 inset token 改回硬编码 `(0.02,0.98)`;⑤ overview 详细度从 HEADLINE 改回 FULL(现在只有金样能抓,无行为断言);⑥ `fit.py` 导入 `zlc_data`(现在只靠人工 grep)。逐条补机械守卫,每条都要突变自证必红并记 commit message。
- [x] K8 **金样脆弱性**:`tests/goldens/facet_fit.png` 钉住了**机器精度级**的标准误文本(无噪数据渲染出 `± 5e-15`,换数值微分则是 `± 4.8e-15`)——这种金样会随无关改动闪烁。修法:金样用**有噪声、误差量级正常**的数据生成,使渲染文本稳定。
- [x] K9 **流程纪律**:commit `388789d` 交付时测试套件是**红的**(金样测试失败),金样在两个 commit 之后才被静默重新生成。今后:每个 commit 自证绿;金样更新必须与产生它的改动同 commit 并在 message 说明原因。


## K 轮:单位面收敛(用户 2026-08-05 提出)

- [x] K10 **注册表把"能识别的写法"与"能选择的单位"混为一谈**:`UnitRegistry.symbols()` 返回 `self._units` 的全部键——其中同时包含 canonical symbol 与全部别名。唯一消费者 `session.py:754` 拿它生成下拉选项,于是**同一个单位被列多次**。实证:`us`/`µs`/`μs` 三个字符串 `resolve_unit` 后是**同一个对象**(canonical symbol `'us'`,另两个在 `aliases`);时间轴下拉 8 项只有 6 个不同单位,电压轴 5 项只有 3 个;`deg`/`°`、`degC`/`°C` 同病。注册表 39 个符号实际只有 31 个不同单位。
  修法:把两个问题拆开——别名只服务 `resolve_unit` 的**输入解析**;另给一个"返回不同单位(每个单位一次,用 canonical symbol)"的查询供选项生成使用。`Unit` 的别名模型本身是对的,不动。
  判据:任意轴的单位下拉里,**任意两项 `resolve_unit` 之后不得是同一个对象**(机械守卫,以后新增别名也不会漏);现有输入解析行为逐字不变(`us`/`µs`/`μs` 仍都能解析)。

### K 轮机械终态判据
1. `pytest -q` 全绿;notebook 全执行零错误。
2. K1/K2/K3/K4 各自的数值判据全绿;K5 的次优极小用例存在且开/关热启动结果一致。
3. K7 六条突变逐条自证必红,结果记 commit message。
4. K10 的下拉去重守卫绿;单位输入解析行为不变。
5. 跨仓契约块与 zlc_runtime 侧**仍逐字相同**(本轮已同步修正,不得单侧再改)。


## §NB 轮:notebook 改回教程(2026-08-05 用户裁决,六仓统一标准)

> **责任在 GOAL**:我此前在多个仓写过"notebook 必须覆盖全部公开 API / 必须真执行"这类**机械覆盖判据**,于是 notebook 被写成了"能跑通的测试脚本"——巨型 cell、成堆 `assert`、极少 `print`。**把代理指标当成了目标。**
> **用户裁决**:notebook 是**教程**——按功能分 cell、**每格教一件事**、用 `print` 展示结果让人看懂;**断言属于 `tests/`,不属于教程**。
> **本仓实测**:两份 notebook 合计 15 code cell / 573 行,最长一格 **151 行**,8 条 assert,仅 2 次 print。
> **参照标杆:`zlc_ui` 的 notebook**(11 个小 cell、17 次 print、**零 assert**)——六仓里唯一做对的,照它的形态改。

- [x] NB1 **拆格**:每个 code cell **≤ 25 行**,只教一件事;每个 code cell **前面有 markdown** 说明"这一格教什么、为什么这样用"。
- [x] NB2 **去断言**:notebook 中 `assert` 计数归 **0**;凡有真实守卫价值的断言**移入 `tests/`** 成为真正的测试(不要直接丢弃)。
- [x] NB3 **给结果**:每个 code cell 至少一次 `print`(或等价的可视输出),让读者看得到 API 返回了什么、字段是什么意思。
- [x] NB4 **按功能覆盖,而不是按名字覆盖**:废除"每个导出名都要被使用"这条判据;改为"**每个公开能力都有一格真正的教学**"。仅仅 import 一下、或写 `x = SomeClass` 这种凑数用法,一律不算。
- [x] NB5 **真执行**:带执行输出提交;无外部依赖(硬件/服务器)的部分必须在干净环境从头跑通零错误。

### §NB 机械终态判据
1. notebook `assert` 计数为 0;每个 code cell ≤ 25 行且至少一次 `print`;每个 code cell 前有 markdown。
2. 移入 `tests/` 的断言全部绿（178 passed）。
3. notebook 带执行输出提交且零 cell 错误。

## 2026-08-05 归档批次

- [x] S1 **语义描述面 `describe_semantics()`**: `src/zlc_plot/semantics.py` 从 `(DatasetSchema, spec)` 机械推导 kind/AxisRef/group/reduction/samples/facet domains；每个 `_kinds/*` 模块声明 `admits` 与 `semantic_fields`，GUI/notebook 共用同一描述和 `semantic_controls`。契约已写入 `docs/api.md`。
  证据: `tests/test_semantics.py`, `tests/test_kind_registry.py`。
- [x] S2 **FacetGridPlot 二维**: `facet` 已干净替换为 `facet_rows` + optional `facet_cols`；DataView 产出 row-major rows×columns、layout 保留二维 shape/font tier、row/column unit 独立；一维 facet 由固定 layout 自动 pack；二维 golden 已加入。
  证据: `tests/test_facet_grid_2d.py`, `tests/test_plot_session_golden.py`, `tests/goldens/facet_grid_2d.png`。
- [x] S3 **replace_spec 携带政策单源**: `src/zlc_plot/session_policy.py` 的纯函数逐项 revalidate 显示参数，保留单位/size/兼容 viewport，清空 selector/fit；`PlotSession` 与 `RasterPlotHost` 同一通道，语义字段（含 reduction）统一重画。
  证据: `tests/test_semantics.py::test_replace_spec_policy*`, `tests/test_public_surface.py`（same Figure）。
- [x] S4 **GUI 与 notebook 落地**: `Qt5ParameterPanel` 增加 Semantics group 和 `semanticEdited`；host/NotebookView 提供 `replace_spec`；PyQt 示例和 usage notebook 覆盖 repeat/coordinate facet、真实 2×2 grid、reduction、`group=repeat`。usage notebook 23 cells 已在当前环境静态执行通过。
- [x] S5 **收尾**: 合成 handler metadata 机械证明新增字段不需要 GUI 分支；143 处 `kind-isinstance` dispatch 保留为明确开放问题；README、API、architecture、performance 与 `docs/semantic-edit-2026-08-04.md` 已同步，LOC 报告为 25,825 行非空 Python（含空行 28,627）。

- [x] L1 **facet live fit 经 host 必崩且请求永久挂死(BLOCKER,责任在 K 轮 GOAL)**
  实测:`raster.py:1406` 对两种结果统一读 `result.data_revision`,而 K2 把 `FacetFitBatchResult` 改成了 `source_revision`/`batch_revision`,`FitResult` 仍叫 `data_revision` → **同一概念两个名字**,facet 批量 `AttributeError`。更严重:异常发生在 future 回调里 → **future 永不完成,GUI 请求挂死**(我复现时得到 `TimeoutError`),且没有任何错误浮到表面。
  修法:① 两种结果类型对"来源数据 revision"**用同一个名字**(取契约用词 `source_revision`,`FitResult.data_revision` 一并改名),消费方只认一个;② **跨线程回调的异常必须转成 future 的异常状态**——`raster.py` 的 analysis 回调与同类回调统一加守卫,绝不允许静默吞掉(这是 embed 死锁那次修过的同一种病,另一处未修)。
  判据:补 host 层 facet live fit 测试(现在 `test_raster_host.py` 零 `.fit()` 调用);契约测试断言两种结果类型暴露**同名**的来源 revision 访问器;突变探针——把回调里的异常守卫去掉,测试必红。

- [x] L2 **全量复制是横贯七个调用点的系统性问题,不只 curve 的 group 循环**
  实测(60×40 扫描拓扑 = 2400 点、repeat 5、dense site 200 → 物理形状 (5,2400,200)):换 group→repeat **1258ms**、group→site **3875ms**、kind→facet **14128ms**;`_prepare_replacement` 占 **90-96%**,可行性探针只占 **3-7%**(我原先的怀疑方向是错的)。
  **根因**:`_broadcast_1d`(`data_view.py:1333`)返回 stride-0 广播视图(意图正确:不为每根轴分配 (R,P,D));`_resolve` **有缓存**(`_axis_cache`)所以它本身不慢——**慢的是缓存里存的就是那个廉价视图,而 `_domain`(`:1168`/`:1177`)每次调用都 `.reshape(-1)` 把它物化一遍**。对 stride-0 视图 reshape 无法别名,numpy 必然全量复制(实测 **19.2MB / 5.2ms 每次**)。
  **波及面(全部七个 `_domain` 调用点,只有一个是一次性的)**:
  | 位置 | 上下文 | 单次投影的调用次数 |
  |---|---|---|
  | `:512` | curve 的 per-group 循环内(`_resolve` 也在循环内,虽有缓存但应一并提出) | 组数 |
  | `:757` `:758` | `_image_from_positions` 的 x/y | 2(但 image 数据量大得多) |
  | `:850` `:898` | rolling 的两处 group 循环内 | 组数 |
  | `:1138` | `_groups` 内部,每根分组轴一次 | 分组轴数 |
  | `:1016` | `facet_cell_count`(探针也走它) | 1 |
  而 facet 的循环体(`:1049/1056/1066`)**逐 cell** 调用上面这些函数,开销是**乘起来的**:facet+curve = cell 数 × 组数(实测 5×200 → 7457 次 reshape,cProfile 占 60%);facet+image = cell 数 × 2 且单次更贵。**特别注意 rolling**:它的两处同病,而 rolling 是 **live 每一拍都要重算**的,所以在真机上是**持续性**开销,不像 facet 只在切换时痛一次。
  **修法(一次全治,不许逐点打补丁)**:让**一根解析后的轴拥有一份已展平的坐标平面,在 resolve 时物化一次并随 `_ResolvedAxis` 缓存**;`_domain` 及全部调用点复用它,不再各自 reshape。顺带把与 group 无关的 `_resolve(x)` 提出 `:512` 的循环。
  判据:① 同一台架复测三种编辑(group→repeat / group→site / kind→facet)的前后耗时,写进 commit message;② **rolling 也要测**:live 连续 20 拍的每拍投影耗时前后对比,一并记录;③ **补性能守卫**(全仓现在 `perf_counter` 零命中):上述数据集下 `replace_spec` 与单拍 rolling 投影的耗时上限写成具名常量,超了即红;④ 机械守卫:`_domain` 内不得再出现对广播视图的 `.reshape(-1)`(或等价断言——同一根轴的展平平面在一次投影内只物化一次,用计数 spy)。

- [x] L2b **可行性探针缓存的键窄于输入(潜在错误裁决,不只是性能)**:`_validate_candidate_spec` 读了 `self._spec`/`display_state.values`/`_viewport`/`_size`/`_device_pixel_ratio`,但**这些都不在缓存键里**,且 `set_size`/viewport 变化不清缓存 → 可能返回过期裁决;缓存也只在 `stream_generation` 变化时清,**普通 live revision 从不触发**,且**无上限增长**(实测 5 次编辑 15→51 条)。修法:键覆盖全部真实输入,或按输入指纹失效;加容量上限。判据:改 size 后同一候选的裁决必须重算(spy 断言);缓存条数有界。

- [x] L3 **已显示的 view 不反映后续设置(notebook 第 3 条)**
  实测:六个 setter **确实生效也确实重绘了 session 的 figure**,但 **raster host 没有重新捕获并推送新 front**,于是 `NotebookView._front`(以及 widget 的 frame packet)停在改动前那一帧。
  根因:republish 钩子按"worker 是否忙"判断,而正确的不变量是 **"每一次 surface commit 后面必须恰好跟一次 front 提升"**——由 worker 任务内部发起的 commit 已被该任务自己的 `_capture_front` 覆盖,而**从其它线程发起的 commit 无人负责**。
  修法:republish 由**谁发起的通知**决定,而不是由忙闲决定,且必须是**边沿持久**的(不能因为当时忙就丢掉)。
  判据:`tests/test_notebook_raster.py:124` 那条本来就是为这个风险写的测试,**在坏代码上照样绿**——必须改写成真正会红的形态(逐像素比较提升后的 front,而不是比较 session 内部状态);六个 setter 逐个断言 front 像素变化。

- [x] L4 **2D image 没有 fit 选项:不是过滤器坏了,是目录里只有一个 image 模型且被各向同性锁死**
  实测链路:`session.fit_models` = 目标过滤 + `_fit_model_units_compatible` 两道闸;内置 image 模型只有 `radial_gaussian_center` 一个,它的 `one_over_e_radius` 要求 x/y **同一物理量纲**;凡 x 与 y 量纲不同的 2D image(例如 x 是位置、y 是时间)**目录必然为空**。
  修法:**补上缺的模型,不要削弱单位闸**(削弱会让求解器把秒²加到伏²上,并报出一个不存在的单位)。至少补一个**各向异性 2D 高斯**(x/y 各自独立的宽度,单位各自独立),使非等量纲 image 也有可用模型;`radial_gaussian_center` 保持只对等量纲开放。
  判据:非等量纲 2D image 的 `fit_models` **非空**且能真的拟合出正确中心(合成 2D 高斯,断言中心与真值);等量纲 image 仍能选到 radial 模型;**补 session 层 image fit 测试**(现在整个 fit 套件只测 solver,从不测目录)。

- [x] L5 **2D image 换单位机制错:屏幕几何被显示单位的数值污染**
  实测:`rendering.py:389-400 _image_coordinate_aspect()` 在 x/y canonical 量纲不兼容时直接 `return 1.0` —— 这个 1.0 不是物理长宽比,而是一句"1 个 display-x 单位 == 1 个 display-y 单位"的断言,**含义随用户选的单位而变**。随后 `:1638 set_aspect(aspect, adjustable="box")` 用它锁死数据长宽比,`:1867` 又**无条件** `square_view=True` 走 `_square_image_limits` 补白。于是 x 从 s 换成 ms(display span ×1000)而 aspect 仍是 1.0 → y 被补 ±500 的空白 → **整幅图塌成一条细线**;换 us 再 ×1000,换 min 方向还会翻转。
  修法(**不是给 aspect 打补丁**):① **铁律——image 的屏幕几何(aspect、home extent、是否补白)必须是 canonical 数据的函数**;显示单位只允许改三样:刻度数字、轴标签、colorbar 标签。这条不变量在本仓其它地方(selector/viewport/fit/color limits)**已经全部成立,唯独 renderer 几何破例**。② **各向同性是数据集的属性,不是单位的属性**:`_image_coordinate_aspect` 返回 `float | None`(或带 `isotropic: bool` 的小记录),`None`/非等量纲时**不锁 aspect、不补白**,而不是假装比例为 1。
  判据:x/y 用不同显示单位的 image,逐个切换 x 单位(s/ms/us)后**渲染像素不变**(几何锚 canonical),只有刻度文本变;等量纲 image 切换单位后 aspect 保持正确;**补 image 几何测试**(现在 tests/ 里 `aspect`/`get_xlim`/`set_aspect` 命中 0)。
