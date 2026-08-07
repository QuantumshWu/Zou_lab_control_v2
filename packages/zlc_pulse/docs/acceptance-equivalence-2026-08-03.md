所有实验完成。汇总验收报告。

# zlc_pulse 拆包验收 — 跨树等价实验报告

**结论:小修**(核心等价性全部实测通过;发现 1 处真实语义回归 + 3 项配置/守卫欠账,均单点可修,不构成返工)

实验脚本:`scratchpad/xtree_equiv.py`、`scratchpad/xtree_delay_loop.py`(参照树与被验收仓全程只读;测试套件在临时副本跑)。

---

## 任务1-1 指纹等价 — PASS(实测)

用树内部署配置 `Zou_lab_control_v1/fpga/board_config/streamer_config.json` 的 `params` 同时喂两侧 `params_from_config` → `build_fingerprint`:

| 量 | 树内 image.py | 新仓 wire.py |
|---|---|---|
| deployed 配置指纹 | 0x5AFC7CFB | **0x5AFC7CFB(同值)** |
| 裸 `StreamerParams()` | 0x5AFC7CFB | 0x5AFC7CFB |
| `REGISTER_LAYOUT_ID` | 0x5AFC7CFB | 0x5AFC7CFB |
| 已提交 `zlc_geometry.vh` 的 `ZLC_LAYOUT_FINGERPRINT` | 32'h5AFC7CFB | 与两侧一致 |

- 逐字段比对 deployed params 两侧 dataclass:全等。哈希单源结构原样保留(`wire.py:83-95` 与树内 `image.py` 该函数 diff 为零;RTL 只携带预计算值)。
- **配置来源(需注意,见小修①)**:新仓**不带** `board_config/streamer_config.json`。`wire.py:158` 的 `_shipped_config_params` 找 `<repo>/board_config/...` → 不存在 → `_SHIPPED_PARAMS={}`,几何默认值全部落到离线 airbag 字面量(`wire.py:184-217`)。实测字面量与树内部署 JSON **零漂移**(含 evt_fifo_depth=64/bus_evt_fifo_depth=64/delay_region_words=128/ttl_delay_max_ticks=2147483647),故指纹同值。`load_streamer_config()` 实测 `source=None` + 警告 `"no streamer_config.json found"`;`DEFAULT_CONFIG_PATH` 指向不存在路径(`wire.py:1023-1028`)。`ZLC_PS_CONFIG` 环境变量与 cwd 搜索保留,在 v1 树 cwd 下会正确拾取部署文件。

## 任务1-2 pack 等价 — PASS(实测,1 字差异即下述回归的字级现形)

**EXP2(静态程序)**:3 period + 2 TTL 边沿 + DAC edge 段 + DAC ramp 段 + bracket×3 + 正负混合延时。新仓 `PulseSequence→compile→pack`;树内按旧文档模型手工独立推导 carrier(不抄新侧输出)→ `image.pack_program`。两张 118 字稀疏表:**地址集对称差 = 空;值差异恰 1 字** —— `addr 38720`(= R_DELAY 区基址+ch0)树=0x2、新=0x0,即下面 A 项延时语义分歧,除此逐字相等。

**EXP2b(N-slot 仿射程序)**:duration slot(coeff=256=1<<8)+ DAC value slot(selector+1 端点选择)+ 仿射 ramp 段(stop_tick_coeffs=(256,0))。两侧稀疏表**完全逐字相等**(含 LOOP_END_LO/HI、bus 段 7 字行、flags 字的双端点选择器位)。

**EXP2c(scan 行)**:同一 5×2 行表,树 `scan_bank_words` vs 新 `pack_scan_rows`,20 字**全等**。

wire.py 本体与树内 image.py 的 diff(`--strip-trailing-cr`)仅:新增 `_checked_unsigned/_checked_signed` 范围校验(域内值封包结果不变——上述实验即证明;域外值由旧的静默掩码变为报错,属加强)、slot 宽度校验、`pack/pack_scan_rows/StreamerGeometry` 三个别名、一处措辞(`final evidence`→`final note`)。

## 任务1-3 三投影 + check_rtl_assumptions — PASS(实测)

- `emit_geometry_vh(deployed)`:两侧输出**字符串全等**;与树内已提交 `fpga/pulse_streamer/zlc_geometry.vh` 换行归一后**逐字节相等**。
- `emit_geom_tcl(deployed)`:两侧全等;与树内已提交 `fpga/build/geom.tcl` 逐字节相等。
- `check_rtl_assumptions`:`inspect.getsource` 两侧**源码逐字符相同**;`region_bases`/`build_ip_sizes` 在 deployed 几何下数值全等(delay 基址 38720、total 38848、axi_bram_depth 65536)。

## 任务1-4 compile.py vs 树内 compiler.py 逐块判定

| 块 | 判定 | 证据/论证 |
|---|---|---|
| N-slot 仿射降低(`compile.py:226-242 _period_starts` vs `compiler.py:790-818`) | 迁移(static/affine 合一,空 binding 即静态) | 数学同构(base 累加、selector 处 coeff=1<<frac_bits);EXP2b 字级一致 |
| 全局边表(`compile.py:245-297 _effective_rows` vs `compiler.py:520-592/595-679`) | 重写合一 | 事件集构造/0 与终点锚行/循环起点锚行同构。排序键少了树内"锚行排在带 lane 行之后"的中间键(`compiler.py:638`),但参考行 tick 相等在两侧都触发严格递增报错(`compile.py:282` / `compiler.py:651`),无可达分歧;EXP2/2b 字级一致 |
| 总线段(`compile.py:300-343` vs `compiler.py:682-762`) | 迁移 | edge(start=stop、双端同值同选择器)与 ramp(start_value=0/sel=0,端点仿射系数)语义逐项对应;EXP2/2b 字级一致 |
| bracket(`compile.py:401-409` vs `compiler.py:262-264, 344-345, 883-903`) | 重写:**删除了 bracket+delay 强制展开** | 树内 `_unroll_bracket` 的动机(trigger grouping 双编译,`compiler.py:186-204`)已随"设备零回传"宪章判死。等价性用新仓自带 RTL 周期模型实测(EXP4):bracket×3+3 tick TTL 延时,硬件循环版 vs 手工展开版 `reference_play` 60 tick **逐 tick 全等**。`repeat_from_index` 两侧同为 0(树 `fpga.py:86` 也钉 0),平价 |
| 延时降低(`compile.py:346-364 _delay_values` vs `compiler.py:539-560, 404-418, 859-868`) | 重写 → **真实回归,见 A** | 实测分歧(EXP2 字级 + EXP5 语义级) |
| scan 表(树:编译期冻结进程序并逐点校验 `compiler.py:668-679`;新:表=运行时数据 `device.py:152-199`) | 设计移位(宪章④,api-slot 免重编译) | Q7 三条守卫(单行=静态波形/换表不重传边表/无缝 wrap)在 `test_model_compile.py:70-98` 存在且绿;但逐行边序校验消失,见 B |
| trigger grouping/PulseExecutionForm/evidence 删除 | 符合宪章 | 负面 grep 复核为零(8 个模式,src/ 全空);`trigger_times` 移为编排层纯函数(`schedule.py:10-19` 明示不参与 wire/session) |
| 几何护栏 | 迁移 | `compile.py:375-380` ≈ 树 `_validate_target_geometry`(`compiler.py:915-922`) |

---

## 发现(按严重度)

### A.(小修·真机正确性)负延时全局位移语义回归 — `compile.py:346-364`
树内语义(真机血训单源,`wire.py:693-695` 注释即记载此事故):全局位移折叠遍历**所有被驱动的** TTL lane(未声明延时者按 raw 0 参与,`compiler.py:541-559`)与被驱动总线,负延时→其余通道整体 +shift,保持相对时序。新仓 `_delay_values` 只遍历**声明了 OutputDelay 的端口**。实测:
- EXP5:仅声明 `da0 = -40ns`(DAC 领先 2 tick)的程序,新编译结果 `channel_delays` **全 0**、`bus_delays=()` —— 负延时意图被**整体静默丢弃**,与无延时程序封包完全相同;树内会给每个被驱动 TTL +2、总线 0。
- EXP2 字级现形:R_DELAY ch0 字(addr 38720)树=2、新=0。
- EXP5b:给 ttl_b 声明 0 延时后它得 +2 而同样被驱动的 ttl_a 仍 0 —— 通道间相对对齐随"是否声明"而不一致。
修法:折叠集合改为"被驱动 lane ∪ 被驱动总线 ∪ 声明集"(raw 缺省 0),≈10 行 + 一条负延时对齐测试。

### B.(小修·守卫欠账)运行时行数据无边序校验 — `device.py:152-173 / 174-199`
树内对冻结 scan 表逐点验证有效 tick 严格递增(`compiler.py:668-679`);新设计行在运行时到达,`write_slots/write_scan_table` 只查宽度、`pack_scan_rows` 只查 32 位域。一条使边行碰撞/回退的 slot 行会不受阻拦流向硬件。建议在两个写口用 `evaluate_affine_tick` 做每行严格递增检查(编译期产物已带 ticks+coeffs,O(行×边))。若按防御裁决表这是有意删除,请用户明示豁免。

### C.(小修·配置来源)部署几何在新仓无机械锚
新仓不带 board_config,airbag 字面量目前与部署 JSON 零漂移(实测),但树内的钉住测试 `test_streamer_params_defaults_match_config` 在新仓无对应物,tests/ 也未钉 0x5AFC7CFB。部署 manifest 一旦经批准变更,新仓字面量静默过期,装出的包与树漂移且无警告(word63 握手会在真机上兜底,但虚拟/离线路径不会)。建议:随包 vendor 部署 JSON 为 asset(如树内 `zlc_pulse/assets/deployed_target.json` 先例)或至少加一条 `build_fingerprint(StreamerParams())==0x5AFC7CFB` 钉住测试。

### D.(小修·卫生)pyproject 声明依赖 `zlc-data` 但 src 零 import(实测 grep 空;canonical.py 自带收缩版 digest)。删除该依赖行即可。

## 其余验收核对(全过)
- 临时副本 `pytest -q`:**20 passed, 0.47s**;LOC 实测 5,019(含 transport)。
- `engine_model.py` 与树内 diff 确仅 import 适配块 + 空行(声称属实)。
- AXI 4KB burst 边界切分保留(`transport/axi.py:20,183`,逻辑与树内 `_burst_runs` 同构;`test_axi_burst_split_preserves_4kb_boundary` 在)。
- 真机纪律:FIRE 后单 I/O worker(`device.py:292-320` observer 独占 STATUS/CURSOR/补给,`cursor()` 点火中只回缓存)与三相 SAFE + 双读稳定 + clk_enable 清零(`device.py:258-276`)均保留。
- 指纹契约测试 `test_build_fingerprint_covers_each_geometry_field_except_host_cap` 在且逐字段翻转验非等。

**裁决:小修。** A 必修(负延时真机对齐错误,已有字级+波形级复现);B/C 需用户裁决或补测试;D 顺手删。其余(指纹、pack、三投影、check_rtl_assumptions、bracket 循环化)均以实测逐字/逐字节/逐 tick 等价通过。