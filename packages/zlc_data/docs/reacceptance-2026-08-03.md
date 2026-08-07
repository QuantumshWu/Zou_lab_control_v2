# zlc_data R1–R6 复验报告(2026-08-03)

复验方式:被验收仓与参照树只读;全部 mutation 实验在临时副本 `scratchpad/zlcdata_rev` 上进行,用原仓 `.venv-check` 解释器 + `PYTHONPATH` 指向副本 src(守卫测试的路径断言证实 import 实际解析到副本)。副本基线 `72 passed`(与外部确认一致),每次 mutation 后恢复并终跑 `72 passed`。**原仓 `git status --porcelain` 结束时为空,参照树未做任何写操作**(参照树本有一个先前就存在的未跟踪文件 `pulses/scan_test.json`,与本次复验无关)。

## R2 守卫终审 — 做实 ✅(mutation 双向证明)

- 实现:`tests/test_package_guards.py:26-33` 用 `pkgutil.walk_packages(zlc_data.__path__)` 枚举后**逐一 `importlib.import_module`**;`:61-64` 硬断言上一轮两个盲区 `output_contract`/`transform_codec` 必须在枚举中。
- 实测枚举 = **16/16 子模块**;裸 `import zlc_data` 只拉起 14 个子模块,`output_contract`/`transform_codec` 确为盲区,现由显式 import 封死。
- **植入实验 A1**:副本 `output_contract.py` 追加 `import zlc_storage` → 守卫红(子进程 `check=True` 抛 `CalledProcessError`,`test_package_guards.py::test_import_is_pure_and_resolves_to_this_distribution` FAILED)。
- **植入实验 A2**(更强):追加 `import colorama`(环境里可 import 的第三方)→ 守卫红在 `tests/test_package_guards.py:74` 的白名单断言。ImportError 路径和白名单路径都有真牙。

## R4 终审 — 做实 ✅(裁决=干净删除,已执行)

- **三个未导出函数已删而非补测**:`src/zlc_data/snapshot_projection.py` 现 157 行(原 322),`__all__`(:28-32)只剩三个公开函数;commit `a8548fa` 显示该文件 -171 行;全仓 grep(src/tests/notebooks/README)`materialize_component_dataset|materialize_dataset_acceptance_mask|materialize_dataset_selection` 零残留(唯一命中是 GOAL.md 任务文本本身)。
- 新测试 `tests/test_zlc_data_snapshot_projection.py` 三公开函数正/负路径俱全,断言有实质:derived 保 schema/revision/block_id/数值(:72-85)+ 坏 schema TypeError、复用 BlockId、revision 漂移三负例(:88-117);scalar 的 (1,1,1) 载体、CellValidity mask(:120-133)+ valid 非 bool、两元素、非有限三负例(:136-160);value 的 ComponentValidity→DatasetComponentValidity 提升含 mask 逐值断言(:163-187)+ 非 Value、指纹不符负例(:189-210)。
- **mutation**:禁 `snapshot_projection.py:47-48`(BlockId 复用检查)→ 对应测试红;禁 :109-110(有限性检查)→ 对应测试红。

## R5 终审 — 做实 ✅(附 1 条可选加固)

- `tests/test_zlc_data_transform.py:158-180`:对 float32 schema commit,对 float64 schema 的 snapshot apply;`:178` 先断言两 fingerprint 确实不同(防自洽空转),`:179` `pytest.raises(ValueError, match="stale")`,且走的是真 `commit_transform→apply_transform` 授权链非预览捷径。
- **mutation(任务指定)**:同时禁用 `transform.py:247-248` 与 `:270-271` 两处 stale 检查 → 该测试红(泄出 `transform.py:257` RuntimeError,`pytest.raises(ValueError)` 不吞)。
- 细粒度补充实验:**只禁 `:270-271`**(`resolve_transformed_schema` 路径)→ 72 全绿,即该路径的 stale 闸门无 mutation 覆盖。GOAL R5 原文只要求 apply 路径 1 条,故不算返工不实,列入小修建议。

## R1 / R3 / R6 核实

- **R1**:`README.md:44` output_contract 行 ✅;`:62-65` module-scoped 名单含 `zlc_data.validation` ✅;`:67-69` 指纹由 `_tree.digest` 定义、与树内 `canonical_digest` 不可跨比注记 ✅。**发现一处漂移**(见小修 1)。
- **R3**:`validation.py:34-45` 公开 `integer(value, field="value", *, minimum)`,进 `__all__`(`validation.py:102`);三调用点改齐:`axis.py:12`、`selection.py:12`、`value.py:10` 均 import 公名;全 src grep `_integer` 跨模块引用零残留(仅 `numeric.py:34/:51` 的 `_integer_sum_bounds_fit`/`_checked_integer_sum_exact` 为同模块私有 helper,非同名)。`tests/test_zlc_data_validation.py:76-79` 直接测 `integer`。
- **R6**:
  - io 内容级负例:format/version(`tests/test_zlc_data_io.py:145-163`)、JSON 重复键(:166-178,`:173` 自证替换生效)、validity kind(:181-194)。**mutation 三连**:禁 `io.py:47-48` 重复键检查→红;把 `io.py:170` invalid-kind 改为回落 VALID→红;禁 `io.py:199-204` format/version 检查→两参数化用例双红。
  - `exact_mapping` discriminator=None 正例+多余键负例:`tests/test_zlc_data_validation.py:47-60`;**mutation** 禁 `validation.py:88-89` 精确键集检查→红。
  - `selection_from_tree` malformed(`test_zlc_data_transform.py:183-190`)、`committed_transform_from_tree` 缺字段(:193-204)✅。
  - `.pytest_cache` 不在 `git ls-files`,`.gitignore:5` 覆盖,git 交付内容干净 ✅。(注:`.venv-check/` 之所以不脏 status,靠的是 venv 自带的内部 `.gitignore`,仓库 `.gitignore` 只写了 `.venv/`/`venv/`;不入交付,仅备忘。)

## 结论:小修

R1–R6 全部真实落地,守卫与新测试经 mutation 证明有真牙,无一项返工不实。精确小修清单(均为收尾级):

1. **README 与实现漂移**:R3 选择了"公开 `integer`"路线后,`validation.py` 公共名已是 7 个,但 `README.md:46` 仍写"The six data-layer validators"且名单漏 `integer`,`README.md:64-65` 仍写"The six validation primitives"。按 GOAL 终态判据"README 与实现零漂移",两处措辞与名单需补 `integer`。
2. (可选加固)`transform.py:270-271` 的 stale 闸门(`resolve_transformed_schema` 路径)目前无测试覆盖——把 `test_apply_transform_rejects_snapshot_from_a_different_schema` 参数化或加一行 `with pytest.raises(ValueError, match="stale"): resolve_transformed_schema(stale_schema, committed)` 即封口。

相关路径:复验仓 `C:/Users/eadri/Dropbox/WorkCode/Github/zlc_data`(HEAD=b340a2e,status 干净);实验副本 `C:/Users/eadri/AppData/Local/Temp/claude/C--Users-eadri-Dropbox-WorkCode-Github-Zou-lab-control-v1-claude/3f423a49-6636-42c8-bec9-f8e7b19c1618/scratchpad/zlcdata_rev`(已恢复原样并终验 72 绿)。