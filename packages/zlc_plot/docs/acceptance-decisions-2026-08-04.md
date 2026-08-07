# V6 追认的交互与 live 语义

这两项是验收时已经给出的默认裁决，不是未来功能占位：

1. Notebook 拖拽预览由 kernel 烘焙。press 时冻结的 `AxisTransform` 只作为
   kernel gesture 的几何输入；候选和 committed selector 由同一套 Matplotlib
   renderer 生成一个完整 `RasterFront`，浏览器只 blit 和归一化输入。这样本地
   Notebook 与 PyQt5 的像素语义一致，代价是远程 Jupyter 的预览延迟受 comm RTT
   影响。`RasterFront` 已保留冻结几何信息；如果之后确实需要远程低延迟增强，
   客户端预览可以作为单独优化加入，但不能重新成为第二个 raster authority。

2. Rolling 静态快照保留 R 轴逐 shot 历史种子。它不是把静态图伪装成 live，而是
   让固定快照和 live rolling 使用同一个 `(R, P, *data_dim)` projection 语义；
   `R` 的历史保留策略由 rolling window/schema 决定，live ingress 仍使用严格递增
   revision 与 capacity-one latest-only presentation。

对应实现和守卫位于 `src/zlc_plot/notebook.py`、`src/zlc_plot/raster.py`、
`zlc_plot._live_channel` 以及 `notebooks/usage.ipynb` 的 rolling live cell。
