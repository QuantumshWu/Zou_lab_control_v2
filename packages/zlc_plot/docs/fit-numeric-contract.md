# 跨仓数值契约:facet 批量拟合的纯数值表

> **这是 zlc_plot 与 zlc_runtime 之间唯一的数值交接面。**
> 本文件在两个仓里**逐字节相同**,任一方改动必须**同时**改另一方,commit message 互相点名。
> 生产侧 = zlc_plot(把拟合结果暴露成纯数值,不构造数据集、不 import zlc_runtime);
> 消费侧 = zlc_runtime(SelectionBridge 据此物化派生信号)。
> 机械守卫:两仓各有一条测试比对本文件的 SHA-256(见文件末尾的锚)。

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

---

<!-- CONTRACT-SHA256: 582bfee5bea7e13153067bdb20d5770f1c31431cbd52cf1f5175238251973da3 -->
