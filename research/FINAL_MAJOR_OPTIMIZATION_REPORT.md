# 共享 Fit / Compose 优化：最终实现、收益与边界

2026-09-07；`codex/perf-limits-20260906`，HEAD `ec706be2` 加本 worktree 未提交修改。
没有修改或合并 master，没有提交。本轮保留上一轮已验证的投影、Front 生命周期等修改。

## 结论先说

保留了统一数值 Fit、规则网格紧凑输入、前景有序批量回放、Image 延迟备用像素及坐标复用。
**single 是同一求解流程的一条 lane**，旧单图 Newton/L-BFGS 分支已删除；所有数值最多 float64，原始图像仍保留 u8/u16。

实测改善并不等于全链翻倍：Image-fit 四图约改善 **15.1%**，Curve-fit 四图约改善 **5.7%**。
其中有明确代价：同源 1200×1920 单图 fit **19.32→26.70 ms**，增加约7.38ms（38.2%）；不能把它称为零代价或性能提升。
我倾向保留它，是因为统一了算法、删除旧路径且绝对时间仍约27ms；这个具体取舍如实留给人工验收。

**没有达到硬件极限，也没有证明15–20Hz全链能力。** 当前更突出的剩余问题是A内的Python/GIL交接、逐cell selection，以及已确认的一次117ms整代GC。
没有用禁用GC、增加进程池、跳过cold候选、降低最终精度或改变误差棒/公式来换数字。

## 1. 真实 TaskConsole 四Panel

同一既有bench：MOT 1200×1920；真实鼠标产生43×502 ROI；40-shot history；DPR3；每panel 2×2 preset。
P1相机Image grid，P2 Histogram，P3 F40 Curve/Image＋fit，P4 grouped Curve＋有效SEM。

critical为 **B端第一个Panel.prepare → 同shot四front全部accept**，不包含曝光。不可把每panel的inclusive wall相加。
每run均有关闭细分探针、开启探针各6秒；主比较使用关闭探针的窗口，各45个完整cohort。
before来自本轮之前的真实记录，after是当前最终源码；不是同进程交错AB，不承诺长期尾延迟上界。

| 场景 | Before P50 / P90 / max | After P50 / P90 / max | P50改善 |
|---|---:|---:|---:|
| Curve-fit四图 |89.85 /93.83 /98.51 ms|**84.69 /86.92 /92.60 ms**|5.16ms，5.7%|
| Image-fit四图 |98.08 /100.83 /114.12 ms|**83.27 /85.81 /88.65 ms**|14.81ms，15.1%|

正常计时窗口均0次stall。额外Curve复核的关闭探针窗口为84.72 /87.68 /90.02ms，与84.69ms主结果接近，不以其中较好值替换主结果。

带探针窗口必须另列：Curve主run为85.90 /90.86 /204.86ms，1次stall；Image为84.14 /86.97 /94.27ms，0次stall。
source实际约7.5–7.7Hz，故不能把观测FPS解释成15–20Hz能力。四个完整surface均渲染并安装，但固定屏幕scroll viewport没有同时完整显示四幅图，不是完整四图硬件scanout测量。

| 同一带探针窗口资源 | Before | After |
|---|---:|---:|
| Curve A进程CPU（相对单核）|139.3%|130.4%|
| Image A进程CPU（相对单核）|112.2%|106.5%|
| Curve 三进程同时采样RSS峰值|1098.37MiB|1073.43MiB|
| Image 三进程同时采样RSS峰值|1084.12MiB|1077.63MiB|

最终Curve/Image窗口共享front分别发布180/184次，均0次新建共享块；lease仍为5–8个，未破坏上一轮释放/复用修复。
隔离Curve实验曾出现process-CPU增加，而实链本次没有复现该方向；不能把隔离wall收益直接等同CPU/GIL收益。

## 2. 独立、同输入的效果与代价

| 项目 | Before | After | 说明 |
|---|---:|---:|---|
| F40 Curve完整Session |51.90ms|44.71ms|n5；原数据、fit和完整像素对照|
| 其中render commit |32.73ms|25.00ms|少7.73ms|
| Facet64 Histogram完整Session |42.35ms|35.34ms|n5；不是漏用分箱kernel|
| 其中render commit |32.53ms|25.79ms|少6.74ms|
| 真实捕获B40 Image fit |27.520ms|24.699ms|n3配对；少2.821ms，10.25%|
| 同源1200×1920 single Image fit |19.324ms|26.701ms|n3配对；多7.377ms，38.2%|

Fit before成对加载实际保存的旧radial和旧compiled数值模块，避免旧callback混用新context/empty-weights合同。
早期探索曾共用新的compiled包装、使用ABI adapter；这些只用于定位，最终Fit AB以两模块成对加载为准。
每组只有n3/n5，不能据此建立可靠P99或亚毫秒收益排名。

两种前景case各9个完整front的RGBA和numeric payload均相等。
Image延迟备用图另以普通帧、Focus、clim、Area、zoom和强制native拒绝验证：对应before/after像素相等。
其单独MOT普通帧Session为14.08→12.22ms；Focus没有可靠收益，clim/Area的小样本反而慢约1ms，故不宣称每种交互都加速。

## 3. 保留的实现：为何做、怎样共用

### A. Fit共享完整数值路线

- 保留当前数据的fresh正/负候选与历史warm候选竞争；不做warm-only。
- B1和Bn统一proxy→full TRF；serial/prange只是相同lane核的执行形式，不再各自维护算法。
- 规则网格显式声明内部布局，存shape与两条真实轴向量；不把两个坐标各复制成完整2M数组。
  这只是数值内部表示，Dataset、scope/fate、Plot轴语义未改。
- 现有context贯通prepare和objective，保存一次中心化统计量及本次工作区，不建立持久history。
- 原dtype装包后，一个lane×stripe核同时做物理float64转换、中心化与moments；不反复扫描/复制。
- 目标函数复用本次有效数，不在每次评价时重扫整张valid掩码。
- `weights=None`由原有`use_weights`表达，未启用时不再分配B×N浮点ones；所有模型和finalizer都只在启用时读取权重数组。
- 单图和批量共用原数学关系：BLAS投影、中心化闭式RSS/梯度、同一信息矩阵和批量协方差owner。
  近抵消时使用同一直接残差计算；最终质量再以真实直接残差核对。
- 物理参数、固定参数/bounds、损失函数和full容差不改；没有float128计算或专用兼容kernel。

中间失败不是最终设计：简单让B1套原batch曾约18–20→128ms；compact、BLAS和分块后的78/58/39ms版本也均未作为完成结果。
定位后消除了无用坐标展开、未使用的权重图及重复完整性扫描，才得到最终26.70ms。
局部BLAS=1/4/16对照没有同时改善B1/B40，因此没有修改生产全局线程配置。

最终较轻profile定位到：values stack、ones、context准备和solve显著下降；直接RSS+信息矩阵仍有可观成本。
最后RSS并行保持原64-row分块与汇总次序，但其单独加速没有由跨run差值可靠分离，不能额外记一笔收益。

### B. 前景批量回放，保留原样式与数学文字

- 仍由Matplotlib/Agg/FreeType生成字形、MathText、stroke coverage，不另写LaTeX或plain-glyph语法。
- 现有compose owner生成有序的扁平绘制范围，静态stroke准备后复用，变化文字更新原字形mask。
- 原primitive顺序、native data/fit间的边界、clip及alpha规则均保留；不把整个前景压成错误的RGBA atlas。
- 同一kernel重放Single、Facet和Focus；unsupported artist在原顺序位置走已有draw，手势继续原split语义。
- Curve每帧480次动态artist回放降至0；Histogram从768降至64，剩下的是实际PolyCollection数据柱形。
- 复用现有失效点，不新增epoch协议、renderer类或科学数据路线。

额外代价是renderer局部的mask、metadata与一次scratch；DPR3 2×2 scratch为6,297,480 bytes。
最终生产版首次静态lowering及全部mask占用未独立计量，不能拿原型44–47ms冒充最终测量。
新增foreground kernel实际有1个已观察到的overload，磁盘机器码约73,822 bytes；不等于所有未来布局都只会有一个signature。

### C. Image不再提前画一张不用的备用图

原primary/Focus Image每帧先做ImageFrontStore→RGBA→viewport front，随后native又从标量数据画一次。
现在artist仍持有真实标量/validity、extent/aspect/cmap/clim；备用RGBA只在native拒绝或export需要时，调用同一个已有owner生成。
实时clim不等release，Selector坐标不改，Figure保存也走同一materialize路径；不留假占位数据。

### D. 选择中的同一坐标只保留一个owner

Regular FitSelection直接引用其已验证RegularImageFitInput的只读x/y，不再复制另一份同样坐标。
删除了随后必然会被RegularInput拒绝的nonfinite坐标的重复mask处理。scope、ROI优先级和错误边界不变。
这只是减量，**并没有声称整个selection外层已完成批量化或不再争GIL**。

## 4. 正确性、缓存与代码量

- 27个直接Fit用例通过：mask/crop/lazy/bounds/warm、single/batch、sigma/robust/fixed/rank/cancel。
- 16个小科学场景、48对结果：当前single/B2参数全部精确一致。覆盖负峰、近无噪、掩码/NaN、4种robust loss和高背景。
- 除高背景场景外，与旧路径的最大曲线差7.87e-8；不是全局bit-exact声明。
- 1e10背景、峰幅4、噪声0.03时，旧single的直接RSS从0.45659/0.45758降至新0.42743/0.42681（radial/anisotropic）。
  与旧batch最大像素预测差2.0981e-5、绝对RSS差≤8.4714e-7。旧single的大参数差对应实际残差改善，没有隐藏。
- 前景/交互/保存的现有窄QA为27通过；唯一旧Image备用cache断言改为检查实际canvas左右数据空白带后再1通过，未放宽像素容差。
  合法clim文字和inward ticks不应被误判为图像填入空白区。
- 另2个现有公共Image选择/拟合用例通过，覆盖坐标复用这次小改。
- 最后在同一个tiny warmer补齐u8/u16/f32/f64真实single/B4输入，并实际跑通；fused四个dtype签名均已观察到。
  这只改预热样本，不改上面实测的数值/绘图算法。已有cache的真实B40新进程首次fit251.1ms；没有完整fresh-cache全dtype矩阵，不保证所有环境首次均相同。
  预热中仍有一次Numba关于小strided dot的PerformanceWarning，不是Cannot-cache或正确性错误，没有再为它另开性能分支。
- 删除旧Newton/L-BFGS、仅其使用的summary/中间目标函数和warmer引用，不为旧测试恢复死核。

以本轮开始保存的源码逐文件行数核算（含最后12行浮点预热样本）：本轮production净 **+71行**；连同前轮修改相对HEAD累计 **+41行**。
没有新增production文件、类、进程池、调度器或持久结果缓存。算法替换有大量删改，不能只看新增diff块判断规模。
磁盘Numba缓存的旧构建文件按源版本失效，没有删除用户整个已预热缓存；旧生产kernel入口与warmer引用已删除。

## 5. 剩余大头：不是已到物理极限

最终开启探针窗口的阶段P50（嵌套，不能相加）：

| 阶段 | Curve-fit四图 | Image-fit四图 |
|---|---:|---:|
| P3完整Fit |41.35ms|53.99ms|
| P3 public batch |20.08ms|43.81ms|
| P3 render commit |27.14ms|15.73ms|
| P3 compose |21.62ms|10.59ms|
| P3有效SEM raster |8.09ms|—|
| P1相机render commit |21.77ms|21.76ms|
| P2 Histogram projection |7.66ms|7.43ms|
| P4 grouped Curve有效SEM |5.16ms|6.30ms|
| A最后promote→B同组accept |3.15ms|3.08ms|

Image的中间trial曾batch27.38ms、selection23.46ms；最终batch43.81ms而selection下降，完整Fit仍约54ms。
这说明并发等待会在Python/NumPy边界之间迁移。不能把某一次局部batch下降全当CPU工作消失，也不能以最终batch升高就断言数值solver变慢。

下一步若要求大幅降低约84ms中位数，需针对selection/包装与A内重获GIL的碎片化做真正整批处理，或改执行组织；不是再把当前约数毫秒的迭代核抠快一点。
这会触及现有typed input/semantic准备边界，尚无完整原型保证收益；本轮没有继续引入一层prepared框架来赌数字。
SEM已是解析subpixel rectangle覆盖并保留每条stem/cap及原alpha顺序，不是简单复制同一误差棒位图就能保证同图；继续SIMD/编译布局研究可能有余量，但没有新的兑现数字。

### 长尾已确认的一类原因：GC

Curve主run的带探针窗口出现一次204.86ms critical，单次fit.overlay占116.39ms；该run没记A的GC，不能事后断言。
只补了一次带标量GC时刻的真实复核，正常窗口84.72ms；带探针窗口再次出现205.20ms。
这次 **generation-2 GC耗时117.334ms，回收54,628个对象**，完整嵌套于120.772ms的front捕获区间。
因此这次长尾不是120ms像素拷贝或数值求解，而是整个解释器回收停顿。

探针会影响分配和触发时刻，不能由它推算无探针运行的GC频率；不宣称每次断点都是GC。
下一步应查清这些待回收对象来自初始化/重布局还是稳态循环，再在真实owner减少生命周期残余。
本轮没有禁用GC、调大阈值或把collect偷偷挪到别的计时段，也没有宣称永不掉帧。

## 证据入口

- `major_cut_final_curve.json` / `major_cut_final_image.json`：最终实屏完整记录。
- 同名 `_summary.json`：A内阶段、重叠与跨进程因果汇总。
- `major_cut_gc_check_curve.json`、`major_cut_gc_check_curve_child_135652.json`：复核及GC时刻。
- `major_fit_shared_core_result.md`：Fit全部AB、科学资格和被否决中间版本。
- `major_foreground_result.md`：前景算法、完整像素、成本与QA。
- `image_preparation_cut.json` / `_fallback.json`：备用图修改的独立操作和强制拒绝对照。

最终运行与复核的全部B/A/C PID均已确认退出；没有留下自己打开的GUI。
