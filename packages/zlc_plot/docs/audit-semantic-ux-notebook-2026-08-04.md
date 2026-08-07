All evidence is in. Compiling the final report.

---

# zlc_plot 体验审计报告(任务A:notebook 真实执行 + 语义面复核)

执行环境:系统临时目录干净 venv(钉版依赖 matplotlib==3.10.8 实装)+ nbclient/ipykernel/anywidget/ipywidgets,未装 PyQt5(顺带验降级)。工作区版与 HEAD 版 usage.ipynb 均通过 junction 隔离目录顶到底执行,被审仓零修改(`git status` 仅原有的 `M notebooks/usage.ipynb`)。

## 总体执行结论(A1)

- 23 cells 顶到底执行:**0 error、0 warning、8.4 s 完成**;stderr 仅 zmq proactor / kernel 明文 TCP 两条环境噪声,与仓库无关。
- **HEAD 版执行结果与工作区版逐 cell 完全一致**(两版 cell 源码本就 0 差异,见 A2)。
- cell 20 无 PyQt5 时降级优雅:打印 `PyQt5 示例不可用: pyqt5_embed requires 'pip install -e .[qt]'`,不抛异常(A4:通过)。
- 但**"执行全绿"掩盖了下面一串语义面在用户眼里的真实故障**——全部在渲染像素层,机械断言全部通过。

## 按严重度排序的发现

### 1.【坏】SE4 没有达标:reduction mean→median 切换在像素上不可见——这几乎肯定是用户"语义面还是不对/没反应"的主因
- 用 notebook cell 3 自己的数据实测:**peak(mean)=2.9325 mV vs peak(median)=2.9302 mV,max|mean−median| = 0.0069 mV = 曲线峰值的 0.24%**。两张渲染 PNG 唯一肉眼可见的差别是 y 轴刻度重排(autoscale 抖动),曲线本身不可分辨。
- 根因是构造性简并:`repeat_scale = np.where((r%2)==0, 1.0, 1.3)`(usage.ipynb cell 3;`examples/pyqt5_embed.py:212` 同款)把 32(embed 24)个 repeat **精确对半**分成 1.0/1.3 两簇 → median = 两中央次序统计量的平均 ≈ (1.0+1.3)/2·base ≈ mean;`repeat_offset=(r%4−1.5)·0.04mV` 与 sin 项也全对称。commit d2fb01e "make repeat reductions visibly distinct" 只让 `np.allclose(mean, median)` 断言为假(0.24%>rtol),**"肉眼可见"这一验收判据为假**。cell 3 与 `pyqt5_embed.py:289-292` 的 assert 是通过的弱代理,SE6 的"非简并断言"守的是同一个错误谓词。
- 复现:两状态各存 PNG 对比(`scratchpad/zlcaudit/states/cmp_mean.png` vs `cmp_median.png`)。修法方向:不对称拆分(如 1/4 repeat 加大幅离群),让 median 明显偏离 mean。
- 对照:**group=repeat 是可见的**——32 条线呈两条明显分离的带(`state4_group_repeat.png`),这半边达标。

### 2.【坏】语义演示 cell(cell 13)的唯一可见输出 = 四个完全空白的 2×2 面板,标题互相撞字
- 用户保存的输出与我全新执行**逐像素同病**:`grid2d_session` 渲染出真实 2×2 几何,但**每个面板空无一物**——`grid_points` 每个 (row,column) 组合只有 1 个点,单点曲线无 marker → 画不出任何东西;四个面板标题 "row=0.0 / column=0.0…" 宽于半幅面板,互相重叠成乱码。
- "真实 2×2 grid" 的承诺(GOAL S4/SE4"demo 即验收台架")在用户眼里就是**空图**。这是"形状剧场 demo 版"的又一实例:几何对、内容空。
- 复现:Run All → cell 13 输出;证据 `pngs_saved/cell13_0.png`(用户自己那次运行)与 `pngs_fresh/cell13_0.png` 相同。

### 3.【坏】embed 语义 kind 切换后轴标签撒谎:`pyqt5_embed.py:789` 把旧 labels 原样搬进新 kind
- `candidate = replace(candidate, labels=current.labels)`:Curve→Histogram 后直方图 x 轴渲染为 **"Time (mV)"**(实际是 Signal 分布),y 轴 "Signal"(实际是计数);再切回 Curve,default_spec 给 x='field',但标签仍写 "Time" —— **横轴画的是 Field 数据、标签声称 Time**。已渲染实证:`states/embed_switch1_histogram.png`、`embed_switch2_back_to_curve.png`。
- 这是用户手动验收路径上点两下 combo 就能踩到的"图在撒谎",与 PlotLabels 显式覆盖轴标签的渲染规则(rendering.py:1394-1403 等)叠加所致。

### 4.【坏】default_spec 取 topology 首维(慢轴)→ kind 切换 round-trip 静默改图义,image 默认是作者版的转置
- `_kinds/curve.py:43-45` 取 `dimensions[0]`:notebook schema(from_cartesian(y,x))默认 curve x=**'y'**、embed schema 默认 x=**'field'**——都与作者 authored 的 x='x'/'time'(快轴)相反。Curve→别的→Curve 一个来回,横轴从 Time 变 Field(再叠加发现 3 的标签谎言)。
- `_kinds/image.py:44-49` 取 `dimensions[:2]` 作 (x,y):notebook schema 得 `ImagePlot(x='y', y='x')` = cell 8 作者版的**转置**(实证渲染 `states/notebook_image_default_spec.png`,横轴标 "Y (mV)");image.py:52-57 的 data-axes 分支同病(`x=data_axes[0]`=纵轴 'image_y'),embed image 页 Image→Histogram→Image 一个来回图就转置。
- 行主序 (slow,fast) 网格的常识默认应为 plot-x=快轴/plot-y=慢轴;现默认恰好反着。

### 5.【坏,违 SE7 拍板文字】推不出默认的 kind 是"消失"不是"灰掉"(核实外部待验项:属实)
- `semantics.py:300-308`:kind_choices 只含 `admits(schema)` 为真的 kind;仅**当前** kind 不可 admits 时补列+disabled。非当前的不可推 kind(embed 数据页的 Image/FacetGrid、任何数据页的 Pulse)**整个从 combo 消失**,用户无从知道该 kind 存在、更无从知道为何不可用。SE7 明文"推不出默认的 kind 在语义组灰掉"。
- 灰掉机制其实已存在(`qt_controls.py:397-402` 会渲染 disabled 项),只是 semantics 层不喂;offscreen 守卫 `tests/test_semantic_ui.py:65` 只守"当前 kind 灰掉",把消失行为固化了。
- 实测 admits:notebook schema → curve/image/histogram/rolling;embed 数据 schema(site 轴 size=4)→ 只有 curve/histogram/rolling(image 因 significant data axis 被拒,与外部观察一致)。

### 6.【坏(视觉)】notebook 第一张图:area selector 的角坐标 readout 与 fit 注释框撞字成垃圾
- cell 5 输出(用户保存版与我复跑版一致):大号灰色两行坐标 "(-1.20, 0.60)/(-1.20, 2.30)" 常驻叠在蓝色 fit 参数框上,两层文字互相穿插不可读(放大图 `zlcaudit/cell05_zoom.png`)。
- 来源:`_selector_scene.py:268-273` AREA selector 的 label 是 selector scene 的常驻 primitive(只要 display=True 就画),锚点与 fit 注释同在左上。教程第一张图即视觉事故。

### 7.【怪】cell 13 四次 replace_spec 挤在一个 cell:语义编辑的"过程"没有任何可见痕迹
- repeat facet → y facet → median → group=repeat 逐次覆盖,中间三态只在**远在视口上方**的 cell 5 图上各闪几毫秒;Run All 后 cell 13 可见输出只有发现 2 的空 grid。保存/重开后 cell 5 的 fallback PNG 永远停在初始态(selector+fit),**整个 notebook 里找不到任何一张"语义编辑生效"的图**。markdown 承诺"覆盖 repeat facet、坐标 facet、二维、reduction、group"——机械上做了,肉眼上零证据。演示结构应改为每态一图(或至少最终态独立 show)。

### 8.【怪/可解释】notebook 前端没有语义 UI,"notebook 同权"只是 API 同权
- `notebook.py` 的 RasterWidget 仅 canvas+手势层(ESM 里无任何 combo/按钮);`NotebookView.describe_semantics/replace_spec`(notebook.py:439/444)是纯 API。GOAL 终态判据3"在 pyqt5_embed 可亲手操作…notebook 同权"若读作 UI 同权则未达;手动验收者在 notebook 里没有可点的语义控件,只能手打 spec。按"API 同权"读法可解释,但与用户"语义面"的直觉有距离。

## A2:usage.ipynb 未提交改动的裁决

- **纯输出 diff,0 行源码变化**:23 cells 源码与 HEAD 逐字节一致;新增的只是 execution_count(1..15)与 outputs(anywidget widget-view + PNG fallback,共 ~1.0 MB;HEAD 约定 stripped、31 KB)。
- 不是半成品:是一次**手动验收会话的残留**——exec_count 缺 14(cell 20 被跑了两次)、cell 22(close_examples 定义)从未执行,非干净 Run All。
- 裁决建议:**丢弃**(`git checkout -- notebooks/usage.ipynb` 或 nbstripout)。理由:仓库约定无输出;保存的 widget-view 引用的 model_id 无 widget state 元数据,重开一律回落到 PNG,且 cell 5 的 PNG 停在被 cell 13 改掉之前的旧态,归档反而误导。

## A3:语义演示 cell 与文字承诺的逐条对照

| 承诺 | 实际 | 判定 |
|---|---|---|
| reduction 切换肉眼可见 | mean/median 差 0.24% 峰值,只见刻度抖动 | **否(发现 1)** |
| group=repeat 可分辨 | 32 线两条清晰分离带 | 是 |
| grid 2d 真显示 2×2 | 几何 2×2,内容全空+标题撞字 | **半否(发现 2)** |
| repeat/坐标 facet 演示 | 仅在上方图闪现毫秒级,无留存 | **不可见(发现 7)** |
| HEAD 对照执行 | 逐 cell 一致(源码相同) | 结论不因版本而异 |

## 证据文件(均在 scratchpad,仓库未动)

`C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri-Dropbox-WorkCode-Github-Zou-lab-control-v1-claude/3f423a49-6636-42c8-bec9-f8e7b19c1618/scratchpad/zlcaudit/` 下:`report_wt.json`/`report_head.json`(逐 cell 执行记录)、`pngs_saved|pngs_fresh/cell{05,13}_0.png`(用户保存版=复跑版)、`states/cmp_mean|cmp_median.png`、`states/state0..state4*.png`、`states/embed_switch1_histogram.png`、`states/embed_switch2_back_to_curve.png`、`states/notebook_image_default_spec.png`、`cell05_zoom.png`。

## 一句话回答"用户为什么说语义面还是不对"

机械链路(发射→replace_spec→新 front)全通,但用户手上每条可操作路径的**像素结局**都是坏的:点 reduction 看不出变化(数据对称简并)、notebook 语义演示只给一张空 2×2、embed 切 kind 后轴标签撒谎且图义转置/换轴、想切的 kind 从 combo 里凭空消失——SE1-SE7 的勾全打在代理指标上,没有一条勾是对着渲染结果打的。