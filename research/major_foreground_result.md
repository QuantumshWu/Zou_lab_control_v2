# 前景有序 primitive 重放：实现与收敛结果

本轮在 `C:/Users/eadri/WorkCode/zlc_v2_perf_limits_20260906` 实施。最后整理阶段仅读取已有记录和源码，未再启动 Python、计时、QA 或修改 production；CPU 已归还主任务。未提交、未碰 master。

## 已得到的收益

把现有前景 owner 的静态 stroke/tick/text 转成按原绘制顺序排列的 coverage 数据；稳态只更新真正变化的 Text。不是整层 RGBA atlas、不是提前画背景，也不是改写 MathText。默认 F40 Curve 的逐 artist 绘制从每帧 480 次降到 0；Facet64 Histogram 从 768 次降到 64 次，留下的都是实际数据 PolyCollection。

本轮同源新数据 A/B、DPR 3、2×2 preset 单 panel、1470×1071 RGBA，每侧 2 warm＋5 无 hook 帧＋2 短 profile 帧。每案全部 9 个完整 front 与最终数值 payload 相等，RGBA 不设容差。下表是无 hook 的中位数，单位 ms：

| 场景 | 旧整 Session | 新整 Session | 旧 commit | 新 commit | commit 净省 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MOT F40 Curve，40×500 输出点，20,000 个有效 SEM，真实逐 cell fit | 51.90 | 44.71 | 32.73 | 25.00 | 7.73 |
| Facet64 Histogram，2,048,000 输入值→64×60 bins | 42.35 | 35.34 | 32.53 | 25.79 | 6.74 |

这里的 commit 包含 mutation、native 数据栅格和 compose；整 Session 还包含投影、fit、front bytes 等，不是实屏 scanout 或 GUI 验收。Curve 的 fit 中位数为 16.97→17.19 ms，未靠少算 fit 得到收益。n=5 不足以证明长期尾延迟改善。

基线明确使用本轮备份 `C:/Users/eadri/AppData/Local/Temp/zlc-major-cut-cd32b2d1cc93455bbf076e2ba3dde1c3/rendering.py`，不是直接以 ec706be2 的旧完整栈比较；两侧均使用同一当前 DataView、Session、fit 与 payload 类。JSON 的 `baseline_renderer_file` 指明实际来源，较泛化的 `baseline_renderer` 字段仍写仓库基线。没有把以前 data/SEM 优化收益再次归入这一轮。

## 函数级实现与保留的边界

- `_install_text_raster_memo`：只在现有 renderer 临时捕获开启时，接住原 FT/MathText 已生成的 glyph mask、原整数位置和 GC。动态数字不再经历“白色 scratch 再绘制→读回 mask”；正常 draw 不受影响。
- `_foreground_text`：沿原 Text.draw/layout 或原 recorded draw 得到有序 glyph masks，使用真实 clip rectangle 和 Agg 的整数 round 规则。未自建文字 grammar。
- `_foreground_strokes`：配置变更时，用原 Agg 给每个独立边框 stroke/tick 生成 coverage，保留原色、alpha、位置；不合并内部可能重叠的多次绘制。
- `_pack_foreground` / `_paint_foreground`：静态部分一次打包；同一原绘制分段形成 flat 有序 ranges，稳态只检查、更新动态 Text slots。遇到未支持 artist，保留原位置的原 draw barrier，不跨它重排数据与边框。
- `replay_foreground_masks`：现有 kernel 文件内一个 serial、`nogil` 的整数重放核。coverage×alpha、半透明整数除法，以及 alpha=0 跳过、alpha=255 直接复制，都采用 Matplotlib 实际规则，而不是 float lerp。[Agg 整数混合](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.8/src/agg_workaround.h)、[rgba8 coverage multiply](https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.8/extern/agg24-svn/include/agg_color_rgba.h)。
- `_dynamic_artists` 删除了仅 Facet overview 录制边界命令的特例。Single、Facet、Focus 使用同一个准备/消费路径；`_compose_frame` 按实际 data axes 前移对应图框，不能把 side colorbar/rail 的边框移到自身内容下方。
- 沿现有 `_forget_chrome_commands`、artist 删除、fit topology rebuild 清理 flat 计划；Text 的文字、可见性、位置、字体、颜色/alpha、clip 等参与更新判断。未新增 epoch 协议、通用类、production 文件或独立生命周期。

这不是“所有 Matplotlib artist 均支持的新 renderer”：旋转 Text、TeX、path effects、bbox patch、非矩形 path clip，以及 hatch、fill+stroke 复合绘制、多 marker 内部重叠等，仍走原有绘制。当前静态 stroke 入口仅接纳无 fill 的两顶点 path，或两顶点 marker＋单位置命令。手势中的原 split/capture 路径继续使用，不把 area/zoom 等交互强塞进未验证的 flat 路线。

原 native 数据线、fit 线/ellipse、独立 SEM stem/cap、AA 和科学数值不因该实现删除。Root 的 primary/focused Image 延迟物化改动位于同一文件，但属于另一项，不在这里重复归功。

## 剩余时间去了哪里

以下是独立 2 帧短 profile 的累计值除以 2；有包含关系，不应逐项全部相加。profile/无 hook 整帧均值比例：Curve 旧 1.025、新 1.003；Histogram 旧 1.044、新 1.002。

| 每帧工作 | Curve 旧→新 | Histogram 旧→新 |
| --- | ---: | ---: |
| compose 总入口 | 29.50→20.65 ms | 23.60→15.25 ms |
| compose self | 2.65→1.50 ms | 2.00→0.70 ms |
| 原逐 artist draw | 480 次/12.90 ms→0 次 | 768 次/20.70 ms→64 次/11.20 ms |
| 新有序 mask kernel | 2 批/1.20 ms | 64 个原 painter ranges/1.20 ms |
| 新前景打包 | 0.10 ms | 0.40 ms |

Curve 新路径仍有每帧 40 次真实数值 Text/layout/mask 捕获，共约 4.35 ms；前景 owner 总入口约 6.45 ms，包含上述文字、打包和重放。其余可观成本包括有效 SEM kernel 约 6.85 ms、数据/fit polyline kernel 约 1.45 ms。`_dynamic_artists` 收集仍约 0.40 ms，不再把数百次绘制解释藏在这里。

Histogram 剩余 64 次数据 PolyCollection 共约 11.20 ms。为保留原数据/边框顺序，不能把这些数据 barriers 抹掉，只报一次前景 kernel 的理想上限。既有 native 矩形/atlas 实验未达到完整像素等价或没有净收益，本轮没有重新引入。

这是 wall-time 改善证据，不是 CPU/GIL/功耗改善的完整证明。记录中的 Curve 整帧 process CPU 中位数反而为 93.75→171.88 ms；该计时量化到约 15.625 ms，且包括进程内线程，未单独归因。没有据此声称所有 C 调用释放 GIL、全链竞争已消失或已经达到性能极限。

## 构建、缓存、签名与代码量

首次静态 lowering 耗时**没有单独记录**。已有 Curve `first_observed_render_s` 为旧 1.1725、新 0.5770 秒，但它包含 Session 初始 render 和首次 fit 配置，不是纯静态构建，也不是受控 pristine cold/JIT 对照。稳态 2 帧 profile 未出现 `_foreground_strokes` 调用，证明该测量窗口没有重建静态 strokes；不能据此猜首次成本。

生产 `_foreground_batches` 的静态 mask/metadata、动态 glyph 持有量和临时打包总字节**未单列保存**，本轮不补测。已知该尺寸的一个完整 RGBA scratch 为 `1470×1071×4 = 6,297,480 bytes`，由 `_foreground_scratch` 持有至既有失效点；不能将其说成完整缓存/RSS。旧原型的 304,580/262,798 bytes 是另一种 per-artist 缓存，不能冒充当前 flat 实现。未测本轮 RSS 峰值。

本轮新 kernel 实际留下 **1 个编译缓存 overload**：`_raster_kernels.replay_foreground_masks-1178.py313.1.nbc`（71,945 bytes），索引 `.nbi`（1,877 bytes），合计 73,822 bytes；这是磁盘代码缓存，不是像素缓存。只读索引中的 dtype/layout 文本并核当前调用顺序得到此签名：

```text
static_masks   : readonly uint8[1D, C]
static_rows    : readonly int64[2D, C]
static_colors  : readonly uint8[2D, C]
text_masks     : readonly uint8[1D, C]
text_rows      : readonly int64[2D, C]
text_colors    : readonly uint8[2D, C]
text_offsets   : readonly int64[1D, C]
order          : readonly int64[2D, C]
out            : writable uint8[3D, C]
```

原测试进程未打印 `Dispatcher.signatures` 的完整 repr；这里明确是现存编译索引＋实际源码调用的核对，不是重新启动 Python 得到的输出。未单独运行全 warmer 验收。

相对本轮备份，按 diff hunk 的实际 owner 分开计：foreground 在 `rendering.py` **+212/−16，净 +196 行**，kernel **+40/−0，净 +40 行**，合计 **净 +236 行**。同一 rendering 文件另有 root Image 改动 +140/−95，净 +45 行；因此两个文件整体是净 +281 行，不能全部归为 foreground。没有把以前已交付的 SEM/Curve summary 代码再计一次。

## QA 与像素证据资格

一次现有窄 QA 共 **27 passed＋1 failed，16.94 秒**，无新增测试模块、fixture、容差或 golden。覆盖 tick 收缩、live Curve/Histogram fit、模型切换、side chrome、clim 中途新 revision、selector gesture、Single/Focus/overview、relayout、native 数据导出和并发 MathText 导出。

唯一失败是旧 `test_color_limit_preview_composes_without_touching_chrome` 直接读取 `image:view_sampling` 的 RGBA 内部缓存，root 的 Image 延迟物化后该键不再必然存在，产生 KeyError。该用例此前的完整 recompose、两次 clim preview、中途新 revision、colorbar、clim 数值等像素/状态断言都通过。Root 已将其改为检查真实 canvas 数据左右 margin：排除合法橙色 H low/high 文本和 inward ticks，使用实际数据纵向足迹；未改容差或前面的整帧断言。**该单例修正后的重跑由 root 负责，截至本报告未记录通过，不能写 28 passed。**

上述 A/B 的“0 像素差”比较完整渲染 RGBA bytes，不是 PNG 文件压缩/metadata 的字节相等。既有 QA 中的 full-draw exact 用例保持原相等标准；Focus overview round-trip 的旧用例自身有既定小容差，本轮未放宽，不能把所有 QA 都描述成新旧整画面 0 差异矩阵。

历史原型的较快 bbox 版本确曾 RGBA 不等：把 `Text.clip_on=True` 错当作存在 axes clip，裁掉 axes 外标题并污染 scratch。修正后默认 Curve/Image 各 7 帧完整 RGBA 相等；该历史结果仍只是原型默认场景，不替代当前生产的全配置证明。旧 chrome 前置和 Histogram native cap=0 实验存在非零差异，均不是可接受候选。当前生产只对上述明确支持的 primitives 使用 flat replay，其余保留原 draw；未声称复杂 clip/overlap、任意 Matplotlib 外部 setter 或全部风格组合均被穷举验证。

## 保存材料与收口

- `research/major_foreground_curve_first.json`：首个 n=3 Curve A/B，仅作先期结果。
- `research/major_foreground_curve_profile.json`：最终 n=5 Curve、2 帧分项、9 个完整 front/payload 对照。
- `research/major_foreground_histogram_profile.json`：最终 n=5 Facet64 Histogram 与分项。
- `research/render_interaction_limits_opt.py`：复用既有 Session/case 的显式备份 renderer A/B 入口；没有新 benchmark 框架。
- `research/render_interaction_limits_primitive_masks.md`：先期原型与拒绝边界；不得将其缓存/构建数字混入当前生产结果。

已有短纯 Session 证据支持保留当前 +236 行的共同 owner 改动：收益是整帧约 7 ms，不再是仅 0.7 ms tick 小项，但没有兑现“十几毫秒全部消除”的说法。剩余大头仍是科学数据栅格、真实动态文字 layout，以及 Histogram 数据 PolyCollection。是否改善真实四图竞争、启动成本是否可接受，由主任务后续统一验收；此处不继续扩矩阵或新增生产改动。
