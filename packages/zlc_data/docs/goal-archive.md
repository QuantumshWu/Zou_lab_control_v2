# GOAL 归档 — zlc_data 已完成条目

> 已完成并验收的条目原文,留作证据与追溯。**活的计划在 `GOAL.md`**。

## 2026-08-05 归档批次

- [x] D0 引导:`git init`;src 布局 `src/zlc_data/`;pyproject(发行名 `zlc-data`,依赖**仅 numpy**,dev: pytest);`.gitignore`;README 骨架(声明角色轴基准与上面的冲突警示)。第一天落地两条守卫:import 纯度(白名单=stdlib+numpy+zlc_data,**zlc_storage/matplotlib/Qt/zlc_* 其余全禁**)+ `__version__` 与安装路径断言(防影子 import)。
- [x] D1 整体搬入树内 `zlc_data`(axis/schema/value/validity/transform/transform_codec/codec/selection/snapshot_projection/units/io 等全部模块),适配 import;**切断 `zlc_storage` 反向依赖**(分层倒置,schema.py:9 等):把 6 个小校验器(`canonical_text/exact_mapping/finite_real/nonnegative_integer/positive_integer/sha256_text`)以 `zlc_data.validation` 模块落户本包(zlc_runtime 等下游统一从这里取),参照实现在树内 `zlc_storage/canonical.py`,只取校验器不带 canonical 编码器全家桶。
- [x] D2 测试随包落地:schema/value/validity 构造不变量、transform commit/apply/resolve 往返、codec 树编解码往返、npz io 往返 + 残缺文件拒绝、`expand_component_validity`、validation 校验器全集。树内已有的 `test_zlc_data_*` 测试按前缀迁入(迁移树测试形态本就分包,近乎机械)。
- [x] D3 `notebooks/usage.ipynb` 顶到底可执行:构造角色轴 schema→DataBlock/OwnedSnapshot→validity→transform→npz 往返,全假数据;README 定稿(模块地图 + 公共 API 表)。

- [x] R1 README 三处:模块地图(:33-47)补 `output_contract.py` 一行;module-scoped 名单(:61-63)补 `zlc_data.validation`;新增一句注记——schema 指纹自本包起由 `_tree.digest`(JSON+sha256)定义,与树内 `canonical_digest` **不同值**,旧树产出的指纹不可跨比。
- [x] R2 纯度守卫升级:`tests/test_package_guards.py` 改为 pkgutil 遍历**全部**子模块逐一真实 import(现 `import zlc_data` 只拉起 14/16,`output_contract`/`transform_codec` 是守卫盲区)。
- [x] R3 消除跨模块私有 import:axis/selection/value 三处 `from .validation import _integer as integer` → validation 公开 `integer(value, *, minimum)`(进 `__all__`)或调用点改私名,二选一。
- [x] R4 **高**:`snapshot_projection.py`(322 行)零测试而 `materialize_derived/scalar/value_dataset` 是公共导出——补正向+负向测试;同文件 3 个未导出函数(`materialize_component_dataset/materialize_dataset_acceptance_mask/materialize_dataset_selection`)查全树消费者,零消费者则干净删除(树内参照的原测试只剩 .pyc,源已死)。
- [x] R5 **高**:transform stale-fingerprint 拒绝零测试(`transform.py:247-248/:270-271`,commit-then-apply 授权模型的核心闸门)——补 1 条:对不同 schema 的 snapshot 应用已 commit 的 transform 必拒。
- [x] R6 中低打包:io manifest 内容级破坏负例 2-3 条(version/format 不符、JSON 重复键、validity kind 非法);回补 `exact_mapping` discriminator=None+多余键负例(树内 test_zlc_storage_canonical.py:69-75 有、迁移丢);`selection_from_tree`/`committed_transform_from_tree` 各补 1 条 malformed 负例;`.pytest_cache` 移出交付(gitignore 确认)。

- [x] R7.1 README 与实现漂移:R3 公开 `integer` 后 validation 公共名已是 7 个,`README.md:46` 仍写 "The six data-layer validators" 且名单漏 `integer`,`README.md:64-65` 同病——两处措辞与名单补 `integer`。
- [x] R7.2 (可选加固)`transform.py:270-271` 的 stale 闸门(`resolve_transformed_schema` 路径)无测试覆盖(复验实证只禁该处 72 全绿)——参数化现有 stale 测试或加一行 `pytest.raises(ValueError, match="stale")` 封口。

- [x] W1 **拓扑维放宽**:`_validate_grid_topology` 不再要求每个 grid 维是 point 列;维坐标由 `coordinate_domains` 独立携带即为完备(如 V1 的 b_x/b_y/b_z 只在拓扑声明)。同名列仍允许且必须一致性校验。守卫:无列拓扑往返(构造/codec/npz)+ 不一致同名列必拒;老行为回归全绿。
- [x] W2 **快照构造便利**:公开一条从 `(schema 或 cell_schema+axes, values ndarray, revision, validity=dense mask 可选)` 直接构造 `OwnedSnapshot`(指纹自动、BlockId 可选自动、dense validity 经 `compact_dataset_validity` 压缩)的入口;逆向 `expand` 便利同列。判据:round-trip 数值/validity 逐元素等价;与现有 runtime 构造路径产物 `exactly_equals`。
- [x] W3 **归属契约成文**:`docs/contract.md` 增一节"单位与 live 的归属"——unit 字符串=canonical 注解;display 换算、单位注册表、latest-only channel 均为呈现层(zlc_plot)权威;本包永不长换算逻辑。zlc_plot 的 I1 以此节为跨仓契约依据。

- [x] NB1 **拆格**:每个 code cell **≤ 25 行**,只教一件事;每个 code cell **前面有 markdown** 说明"这一格教什么、为什么这样用"。
- [x] NB2 **去断言**:notebook 中 `assert` 计数归 **0**;凡有真实守卫价值的断言**移入 `tests/`** 成为真正的测试(不要直接丢弃)。
- [x] NB3 **给结果**:每个 code cell 至少一次 `print`(或等价的可视输出),让读者看得到 API 返回了什么、字段是什么意思。
- [x] NB4 **按功能覆盖,而不是按名字覆盖**:废除"每个导出名都要被使用"这条判据;改为"**每个公开能力都有一格真正的教学**"。仅仅 import 一下、或写 `x = SomeClass` 这种凑数用法,一律不算。
- [x] NB5 **真执行**:带执行输出提交;无外部依赖(硬件/服务器)的部分必须在干净环境从头跑通零错误。
