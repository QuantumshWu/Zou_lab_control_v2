# 05b — SLM solver、荧光反馈算法与噪声统计深审

状态：完成（只读审查；未修改 production/test/旧文档，未连接硬件）
审查基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
范围：`devices/slm/solver.py` 的 presets、spot/image solve、selected DFT/FFT、optimizer state、quality gate与artifact codec；`nodes/slm_feedback/task.py` 的observable、estimator、controller、validation、best/Stop/artifact语义；现有workspace artifacts、隔离profile和有限shot统计。

## 1. 结论先行

当前问题必须分成三层，不能笼统归为“SLM算法慢或不鲁棒”：

1. **Sparse WGS-Kim数值核心总体正确而且已经够快。** selected-frequency DFT与full FFT在spot support上的数学等价有实质测试；固定远场phase和caller-owned optimizer state正是标准fixed-phase WGS方向。真实`1024 x 1272`、5×7 target默认达到1%模拟support ratio约`0.53 s`，不是100-shot反馈的主要墙钟瓶颈。
2. **Task错误地在最便宜的数值环节过早停止。** 初始solve强制恰好12轮，实测模拟ratio仍为`1.189`；hot solve强制8轮，仍为`1.027`。让现有hot solver继续使用自己的1% gate只需约`0.29 s`、比8轮多约`0.06 s`。当前实现为了省几十毫秒，可能多花一个或多个100-shot物理candidate，这是明显的局部最优。
3. **真正阻止1%的是反馈统计和控制，不是FFT。** 当前HEAD后开始的artifact run显示100-shot下单site relative SEM约`4.4–5.1%`；同日较早run达到`10–14%`。35个site取`max/min`又放大极值噪声。即使真实35 site完全相同、每shot CV=1，100 shots时观测ratio中位数约`1.526`。当前`ratio<=1.01`且`max relative SEM<=0.5%`的validation gate，在刚好0.5% SEM时，完全均匀真值通过概率也只有约`1.5e-5`。
4. **Dense image path需要单独重做。** 当前代码称为MRAF，但Gaussian target在全raster严格为正，没有noise region；使用随机初相、固定300轮、无stagnation gate，且报告的support ratio可达`10^12–10^14`而没有物理意义。真实尺寸Gaussian约`27.4 s`、Flat Top约`14.7 s`，这里才有明确solver性能问题。
5. **Controller没有trust/rollback/recovery。** 每轮无条件追逐最新point estimate；变差仍继续，invalid时不更新phase却把同一phase重新编号、重新保存、重新测到`max_updates`。现有旧artifact真实出现过单site missing后连续82个candidate不再改变phase；HEAD仍保留同一invalid分支。

总裁决：

- spot solver：`PASS WITH DEBT`，保留算法和selected DFT；修Task调用、metadata和少量allocation热点；
- dense solver：`REDESIGN`；
- feedback estimator/controller/validation：`REDESIGN`；
- fixed-phase optimizer state：`PASS`，但Task缺pupil/Pattern authority；
- 100 shots：只足够coarse update，不能作为1% final proof。

## 2. 当前算法链

```text
target intensity
  -> initial solve_phase(iterations=12, incoming device phase)
  -> freeze far-field spot phase at iteration 12
  -> apply candidate phase
  -> 100 fresh-load qCMOS shots
  -> per-site BOX counts - calibration dark mean
  -> Welford mean F_i and SEM_i
  -> point score max(F_i)/min(F_i)
  -> target_i *= (GM(F)/F_i)^0.25; preserve target total
  -> hot solve_phase(iterations=8, same fixed phase + weights)
  -> repeat up to 120 candidates
  -> only point score <=1.01 triggers independent validation
  -> validation accepts when point ratio <=1.01 and every SEM/F <=0.005
```

这个链有两个不同的闭环：

- **computational inner loop**：target amplitude -> WGS weights -> canonical SLM phase；
- **physical outer loop**：canonical phase -> optical/atomic plant -> noisy fluorescence -> target correction。

当前inner loop成本远低于outer loop，却被固定12/8轮截断；outer loop则没有对噪声、invalid、变差和candidate选择做控制论处理。

## 3. 隔离profile与artifact证据

### 3.1 Solver真实尺寸profile

环境：当前workspace包、CPU 16 logical cores、SciPy/NumPy CPU path；每个进程先打印repository root和`zlc_atom.__file__`。

| Case | Shape | Iterations | Transform | Wall time | Final simulated ratio |
|---|---:|---:|---|---:|---:|
| Grid 5×7 cold | 128×128 | 12 explicit | selected DFT | 0.007 s | 1.860 |
| Grid 5×7 hot | 128×128 | 8 explicit | selected DFT | 0.003 s | 1.108 |
| Grid 5×7 cold | 1024×1272 | 12 explicit | selected DFT | 0.384 s | 1.189 |
| Grid 5×7 hot | 1024×1272 | 8 explicit | selected DFT | 0.234–0.256 s | 1.027 |
| Grid 5×7 default gate | 1024×1272 | 25/80 | selected DFT | 0.528 s | 1.00943 |
| Same state hot default gate | 1024×1272 | 13 | selected DFT | 0.294 s | 1.00943 |
| 35 diagonal sites | 1024×1272 | 4 | selected DFT | 0.234 s | 1.691 |
| 65 diagonal sites | 1024×1272 | 4 | FFT fallback | 0.479 s | 1.773 |
| Representative Gaussian image | 1024×1272 | 300 | FFT | 27.428 s | ratio metadata约`6e12` |
| Representative Flat Top image | 1024×1272 | 300 | FFT | 14.686 s | ratio metadata约`1.3e14` |

结论：

- 5×7 sparse热点不是FFT。12轮profile中`_unit_phase`约占`0.119/0.381 s`，主要是full-raster magnitude/output allocation；即使完全消除也只省约0.1秒。
- SciPy `fft2+ifft2`在此机器用1/2/4/all workers中位数均约`37 ms`，简单增加`workers`没有收益。
- selected DFT对当前5×7几何有效；当前fallback阈值至少在本机器没有明显错误。
- dense固定300轮才值得做backend/GPU讨论；在修正算法停止条件和MRAF语义前先加GPU，只会更快地产生同一个错误目标。

### 3.2 Preset和codec profile

真实shape下：

| 操作 | 时间 | 结果规模 |
|---|---:|---:|
| Grid 5×7 | 5.9 ms | 35 sites |
| Checkerboard 5×7 | 4.7 ms | 33 sites |
| Gaussian | 25 ms | 1,302,528 positive pixels |
| Flat Top | 40 ms | 80,891 positive pixels |
| Text `USTC`, spacing=4, budget=256 | 114 ms | 253 sites |
| sparse target JSON save/load | 1.98/0.27 s | 6.64 MB |
| random canonical phase NPZ save/load | 0.045/0.063 s | 4.97 MB |

Preset materialization本身不是UI瓶颈；dense solve和full dense JSON才是。Sparse target也按1.3M dense floats写JSON，是可接受的单一codec但有明确I/O债务。

### 3.3 当前workspace feedback artifacts

`workspace/data/2026_08_16`共有194个candidate NPZ、46.94 MB；它们跨越当天多个code revision，不能合并成一次实验。

最新HEAD提交后开始的run由其最终`candidate_0008-6.npz`内完整history重建；不能只按文件名suffix分组，因为并行/交错run会分别占用每个candidate basename。该run可归因于当前fixed-phase/0.25 exponent实现：

| Candidate | coarse ratio | median relative SEM | max relative SEM | total/baseline |
|---:|---:|---:|---:|---:|
| 1 | 5.484 | 4.48% | 5.10% | 1.000 |
| 2 | 2.513 | 4.47% | 5.14% | 0.928 |
| 3 | 1.478 | 4.44% | 4.89% | 0.917 |
| 4 | **1.185** | 4.49% | 4.75% | 0.896 |
| 5 | 1.193 | 4.40% | 4.88% | 0.912 |
| 6 | 1.349 | 4.47% | 4.97% | 0.908 |
| 7 | 1.217 | 4.52% | 4.80% | 0.899 |

Candidate 8只有`applied`状态，说明Stop/中断发生在measurement完成前。证据表明：fixed-phase controller前三至四步确实完成有效coarse改善；之后score在100-shot噪声地板附近反弹。

较早cohort中存在另一个确定控制缺口：candidate 2起site 24持续missing，后续到candidate 83没有target/phase update，却继续生成candidate和采集；对应candidate 2与83的phase逐元素相同。虽然artifact生成于最后fixed-phase提交前，HEAD的`if valid: update+solve` / invalid直接进入下一轮逻辑未改变，所以同一停滞仍可发生。

### 3.4 100-shot统计可行性

当前HEAD run的`SEM/mean≈4.5%`意味着per-shot CV约`0.45`；同日较早高噪run的最差site对应per-shot CV约`1.4`。这主要来自fresh-load occupancy mixture，不是可以靠更快FFT消除的read noise。即使用较乐观的4.5% mean noise，35个真实均匀site的观测`max/min`中位数仍约`1.207`，与当前run的1.185–1.217 noise floor一致。

隔离Monte Carlo采用35个真实均匀site、独立sample-mean正态近似、per-shot CV=1：

| Shots/site | 单site relative SEM | observed `max/min` median | 95th percentile | `P(ratio<=1.01)` |
|---:|---:|---:|---:|---:|
| 100 | 10% | 1.5255 | 1.7509 | 0 / 200,000 |
| 1,000 | 3.16% | 1.1410 | 1.1871 | 0 / 200,000 |
| 10,000 | 1% | 1.0425 | 1.0557 | 0 / 200,000 |
| 40,000 | 0.5% | 1.0211 | 1.0274 | 0.000015 |
| 100,000 | 0.316% | 1.0133 | 1.0173 | 0.043 |
| 250,000 | 0.2% | 1.0084 | 1.0109 | 0.869 |

真实Bernoulli loading同样给出：100 shots时，完全均匀的`p=0.5` site ratio中位数约1.525；`p=0.8`仍约1.233。

因此当前validation条件存在内在矛盾：`max SEM/F <= 0.5%`只是单site门，35-site极值spread通常仍约2.1%；在该门刚满足时再要求point estimate ratio≤1.01，真实完美均匀阵列几乎必拒。增加candidate次数不能解决；100-shot下即使从120个独立噪声结果挑最小ratio，中位数仍约1.316。

## 4. 与标准方法的对照

### 4.1 Sparse WGS-Kim：实现方向正确

- Di Leonardo等人的original WGS按target/computed amplitude ratio迭代权重；当前inner loop使用`weights *= (target_amplitude/measured_amplitude)^0.8`，与常见WGS-Leonardo实现一致。[原论文](https://doi.org/10.1364/OE.15.001913)
- Kim等人的phase-fixed WGS指出，反复更新远场phase会削弱外部camera correction；先达到效率后固定phase可显著加快并稳定camera feedback。论文在12轮固定phase、三到五次adaptive correction后达到约1–1.6%标准差级别；当前iteration 12 freeze与跨candidate state reuse正是该方向。[Kim et al. 2019](https://arxiv.org/abs/1903.09286)
- 官方`slmsuite`也把WGS-Kim定义为在指定iteration/efficiency后固定far-field phase，并提供compressed spot transform与experimental/external spot feedback。当前selected DFT与short-lived state不是自造错误抽象。[slmsuite WGS文档](https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.CompressedSpotHologram.html)

当前差距不是“换掉WGS-Kim”，而是：

- Task把“至少12轮后固定phase”实现成“恰好12轮就应用”；
- 每次hot update恰好8轮，不检查已有canonical 1% gate；
- 没有传入真实pupil/Pattern authority；
- 没有把solver ratio/efficiency/iterations写入candidate evidence；
- physical update不使用measurement uncertainty。

### 4.2 Dense MRAF：当前只实现了一个不完整变体

Pasienski/DeMarco的MRAF核心是：显式signal region与noise region、固定mixing parameter、noise region amplitude freedom、vortex-free初相，以及figure-of-merit stagnation停止。[MRAF原论文](https://arxiv.org/html/0712.0794v1)

当前实现：

- `support = desired > 0`即signal region；Gaussian raster每个pixel严格正，因此noise region为空，退化为GS-like全平面约束；
- noise amplitude固定乘`0.9`，没有显式/可验证的MRAF mixing语义；
- 使用random phase，未实现论文的quadratic/linear/conical vortex-free initial phase；
- 固定300轮，没有stagnation/quality stop；
- `support_intensity_ratio`对Gaussian极小tail做除法，输出`10^12`量级无意义；
- 只报告support RMS和efficiency，没有文档声称的ghost/background/roughness。

所以不能用“300轮MRAF很慢”作为唯一诊断。首先要定义Gaussian的有限signal window/zero tail、正确FOM和initial phase；之后才profile迭代数和backend。

### 4.3 Camera/atom feedback：标准结果不支持100-shot证明1%

- Kim使用直接CMOS focus intensity，噪声远低于fresh atom loading；fixed-phase方法约3–5次correction达到约1–1.6% non-uniformity。[Kim et al.](https://arxiv.org/abs/1903.09286)
- Tamura等人的atom fluorescence闭环先做trap-light camera校平，再用**每个candidate 1,000张 fluorescence images**与零/单原子分布峰差估计single-atom fluorescence；3–10轮后trap-depth variance约1.7–4%，并未以100 shots证明1% max/min。[Tamura et al. 2016](https://doi.org/10.1364/OE.24.008132)
- Nogrette等人的single-atom array也使用closed-loop trap-depth uniformization，但其可调gain update与实验observable不等于当前all-shot mean。[Nogrette et al. 2014](https://doi.org/10.1103/PhysRevX.4.021034)
- 官方`slmsuite`提供Nogrette gain、exponential和bounded tanh weighting；其中tanh明确限制每步相对变化，适合near-zero/noisy feedback。当前`(GM/F)^0.25`没有step clip。[slmsuite feedback methods](https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.FeedbackHologram.html)

这些方法证明fixed phase和external feedback合理；它们不证明当前100-shot all-shot BOX observable能达到同样统计精度。

## 5. 逐函数审查：`devices/slm/solver.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `_pair` | PASS WITH DEBT | 集中shape validation合理；`int()`会静默接受3.2/字符串，public preset边界应严格拒绝非整数/bool |
| `_float_pair` / `_scalar` | PASS | finite positive/nonnegative validation直接 |
| `_readonly` | PASS WITH DEBT | immutable snapshot有价值；每次通过`tobytes`完整复制5 MB，solver hot path要避免重复进入 |
| `validate_target` | PASS | 唯一finite nonnegative intensity truth正确 |
| `_grid_indices` | PASS | exact integer geometry，owner准确 |
| `preset_grid` | PASS | 直接、必要 |
| `preset_checkerboard` | PASS | 长/短行几何符合UI命名；测试应继续守总site数不是`rows*columns` |
| `preset_gaussian` | REDESIGN | intensity公式本身正确；无限正tail使MRAF无noise region，需用户裁决finite signal cutoff/window |
| `preset_flat_top` | PASS WITH DEBT | finite support适合MRAF；linear edge只是一个具体profile，应明确不是super-Gaussian |
| `_missing_font_characters` / `_text_font_path` / `_allowed_text_character` / `_rasterized_text` | PASS | plugin-local字体/排版职责合理；无solver热点 |
| `preset_text` | PASS WITH DEBT | budget/spacing语义正确，真实尺寸约114 ms；逐字号rasterize可优化但不是瓶颈 |
| `imported_target` | PASS | peak normalize形成唯一target；保留相对强度 |
| `_pupil` | PASS WITH DEBT | 仅可作为明确的legacy/default circular model；Task不应因没pupil context而偷偷采用它 |
| `_unit_phase` | PASS WITH DEBT | 数学必要；当前full-raster magnitude+ones_like allocation是sparse profile最大可优化热点，但收益<0.1 s |
| `_support_intensity_ratio` | PASS仅限spots | spot gate正确；用于dense/Gaussian metadata必须删除或改成有signal-region语义的metric |
| `_canonical_unshifted_phase` / `_phase_snapshot` | PASS | canonical float32 return gate正确 |
| `_cartesian_support` | PASS WITH DEBT | selected transform路由合理；阈值应以real-shape benchmark守住，不是永久魔数 |
| `solve_phase` public入口 | PASS，但内部需拆责 | 一个public入口符合产品；439行同时实现dispatch、WGS、MRAF、state、quality、metadata，建议同文件普通`_solve_spots/_solve_image`函数拆开，不增加class/file |
| `solve_phase` auto objective | PASS WITH DEBT | legacy import需要；Feedback已知sparse，应显式传`spots`，不依赖heuristic |
| selected DFT WGS path | PASS WITH DEBT | 数学等价和性能均有证据；保留fixed phase、weights、target amplitudes |
| full FFT WGS fallback | PASS | 大/不经济envelope需要；测试有full reference |
| optimizer-state validation | PASS WITH DEBT | support/shape/objective与caller invalidation正确；只比较`pupil_source=provided`而不比较内容是刻意caller-owned contract，Task/Editor必须承担 |
| initial/hot state path | PASS | hot state优先于initial phase符合fixed-phase authority；调用者须保证Pattern/pupil没替换 |
| sparse early canonical gate | PASS | returned float32 phase二次传播后才accept，是强测试 |
| explicit iteration behavior | PASS | 精确执行contract正确；Task不应滥用该入口逃掉quality gate |
| dense image path | REDESIGN | MRAF region/initial phase/FOM/stop/metrics不完整；固定300轮 |
| final metadata | PASS WITH DEBT | spot ratio/RMS/efficiency有用；缺ghost/background，dense ratio误导 |
| `_strict_object` / `_constant` | PASS | strict JSON安全边界 |
| `save_target/load_target` | PASS WITH DEBT | 唯一严格codec正确；真实sparse JSON 6.64 MB且save约2 s，后续可在同codec加稀疏投影但需用户批准格式版本 |
| `_metadata_json` / `save_phase/load_phase` | PASS | no-pickle canonical NPZ正确；候选全文件反复rewrite是Task层用法问题 |

## 6. 逐函数审查：`nodes/slm_feedback/task.py`

| 符号 | 裁决 | 说明 |
|---|---|---|
| `_check_cancelled` | PASS | 简单Stop seam |
| `_ratio` | REDESIGN | point max/min是显示统计，不可同时充当noisy controller score、best和1%证据 |
| `_json_floats` | PASS | strict artifact所需 |
| `_support` | PASS WITH DEBT | 冻结sparse target↔SiteMap mapping必要；148行Hungarian/affine逻辑虽复杂但有真实边界测试，保持plugin-local，不抽通用registry |
| `_updated_target` | REDESIGN | log-space geometric center和total normalize合理；固定0.25、无uncertainty shrink、无step limit、near-zero爆增、point estimate bias |
| `SlmFeedbackTask.__init__` | PASS WITH DEBT | resource freeze/BOX model/site windows归属正确；shots/int coercion应严格；缺pupil/Pattern/wavefront authority |
| `dataset_output_declarations` | PASS | 三个live outputs有真实Monitor消费者 |
| `_candidate_metadata` | REDESIGN | 每个candidate复制完整history造成O(K²)metadata；未保存solver evidence、effective per-site sample counts、controller gain/target |
| `_publish_candidate` | PASS WITH DEBT | same revision三输出正确；curve展示point ratio可以保留，但不能被当validation |
| `_apply_exact` | PASS | canonical command/readback边界清楚 |
| `_assert_camera_contract` | PASS | calibration geometry冻结必要；pulse窗口问题见04b |
| `_saturated_sites` | PASS | conservative saturation evidence |
| `_measure` acquisition | PASS WITH DEBT | exact grouped shots和all-shot BOX observable符合当前裁决；invalid site只需2个finite sample就不missing，artifact又不记录sample_counts |
| `_measure` Welford estimator | PASS作为sample mean | O(site)在线均值/方差数学正确；未估dark calibration uncertainty、跨site covariance、drift/heavy-tail，不能支持extreme ratio proof |
| `execute` initial solve | REDESIGN | 恰好12轮而不是“12轮后继续到gate”；未显式spots、未传pupil，incoming full device science phase被当Pattern warm start |
| `execute` candidate save | PASS WITH DEBT | pre-trigger durable phase有Stop价值；真实phase约5 MB且每candidate写前/后两次，Dropbox下成本明显 |
| `execute` coarse validity | REDESIGN | 没有uncertainty/effective-n门；invalid下一轮重复同phase却伪装新candidate |
| `execute` best tracking | REDESIGN | point-estimate winner's curse；正常变差不rollback，max failure反而恢复incoming而非best |
| `execute` validation | REDESIGN | independent复测是对的；0.5% per-site SEM + 1% extrema gate统计上不可行，重复尝试又无alpha spending |
| `execute` hot solve | REDESIGN | fixed state正确；强制8轮省0.06 s却留下2.7% computational residual |
| cancellation/terminal seal | PASS WITH DEBT | concurrency ordering已有强测试；Stop保留“coarse point best”在高噪声下不等于scientific best，需用户裁决 |

## 7. 根因与推荐设计

### 7.1 Sparse solver：保留核心，先走完便宜的inner solve

推荐最小变化方向：

1. Feedback明确调用`objective_kind="spots"`。
2. 初始solve仍在第12轮固定phase，但之后继续到现有canonical ratio gate；不要`iterations=12`停止。
3. 每个physical target update使用同一optimizer state并继续到gate；不要固定`iterations=8`。实测典型只多5轮/约0.06 s。
4. candidate history记录solver的iterations、simulated ratio、efficiency和transform。否则physical失败无法区分numerical residual与plant。
5. 保留selected DFT；不为5×7引入GPU/backend/cache框架。

这条路径很可能用不到更多shots，反而减少需要测的physical candidates。

### 7.2 Dense solver：恢复真正MRAF问题定义

需要先由用户裁决Gaussian的有限signal region表达：

- 选项A（推荐）：Gaussian preset明确materialize有限cutoff，cutoff外target严格0，因此唯一intensity array同时定义signal/noise；
- 选项B：target artifact增加显式signal mask；更通用但产生第二二维truth，需要更强理由；
- 选项C：继续全raster positive，承认这是GS image solve，不再称MRAF。

随后：

- 使用quadratic/linear/conical或其他有依据的vortex-free initial phase；
- 用signal-region relative accuracy + roughness + efficiency + background/ghost做FOM；
- 在FOM停滞时early stop，显式iterations仍精确执行；
- 验证mixing parameter，而不是永久硬编码noise factor 0.9；
- algorithm正确后若15–27 s仍不可接受，再评估CuPy/GPU或专用FFT backend。

### 7.3 Estimator：100 shots做coarse，使用不确定度而非极值点估计

保持all-shot mean observable，不用旧occupied threshold。每个candidate计算：

```text
mu_i = mean dark-subtracted BOX
s_i  = SEM_i / mu_i
e_i  = log(mu_i) - weighted_center(log(mu))
```

建议：

- 记录每site有效n；不足不允许更新；
- 维护35×35 online covariance（约10 KB float64），保留同一shot跨site common-mode；当前只存35个独立variance，ratio uncertainty算不准；
- controller只使用confidence-resolved residual：`gain_i`随`s_i`减小而增大；
- update在log domain clip/tanh到最大相对步长，避免一个近零site吞掉全部target power；
- early candidate先20–100 shots；只有residual接近噪声时逐批追加，直到能决定“更新/不更新”，而不是每轮固定100；
- drift明显时交错测reference/best，而不是假设所有shots iid。

这不会突破occupancy shot-noise信息极限，但能阻止controller追逐它。

### 7.4 Controller：trust region、rollback和invalid recovery

建议在现有Task普通流程内完成，不新增manager/class：

1. `current`、`best`和`last validated`明确分开。
2. Best按uncertainty-aware score（例如log-spread upper bound或expected loss）排序，不按最小point max/min。
3. 新candidate若没有可信改善，rollback best并减小gain；不要从变差target继续积累。
4. Invalid先对**同一candidate**追加/重试一个有界次数，不新增curve candidate、不重写新phase artifact；持续missing/saturation则rollback/停止并报告硬件/registration问题。
5. 若若干步无可信改善，停止coarse并报告noise floor，而不是耗尽120次。

当前artifact的candidate 4 -> 5 -> 6反弹正是需要trust/rollback的实证。

### 7.5 Validation：报告CI，不能用不可能的二元gate

Independent validation应保留，但目标必须由用户明确：

- 若“100 shots只做coarse”：terminal best用独立、更大的自适应样本；
- 若要求统计证明`max(mu)/min(mu)<=1.01`：对log means构造同时confidence bound，或对完整per-shot site vectors做preserve-covariance bootstrap；接受条件是ratio的upper confidence bound≤1.01；
- 若实验时间不允许足够shots：只报告estimate + CI，不声称证明1%。

重复对多个threshold-crossing candidate做validation属于sequential multiple testing。需要预先固定一个best再做一次held-out validation，或使用alpha-spending/sequential confidence；当前“直到一次point ratio过线”为止会放大false acceptance，虽然0.5% SEM门目前更常造成false rejection。

## 8. Pupil、Pattern与phase authority

这是算法正确性，不只是Editor集成：

- Editor solve传入已应用的uniform/Gaussian pupil amplitude，且只保存Pattern/base phase；这条路径正确。
- Feedback Task只拿target file、device last-commanded phase和默认`_pupil()`。它不知道Editor当前pupil、Pattern phase、Zernike/steering层。
- `incoming`可能是`Pattern + wavefront`后的full science phase；Task把它当initial Pattern，并最终用新phase覆盖整个device command。
- fixed optimizer state因此稳定在**默认circular pupil**的模型，不一定是实际光束或Editor模型。

需要用户裁决（对应D-015）：

1. **推荐：Feedback输入冻结Pattern/base phase + pupil amplitude + explicit wavefront layer，Task只更新Pattern并重新compose。**
2. Task明确接管完整science phase，Operator接受wavefront层被吸收/替换；artifact必须如实写明。

当前输入不足以可靠实现选项1，也没有UI声明选项2。

## 9. Stop与artifact语义

当前优点：

- phase在第一个camera trigger前已原子落盘；
- terminal seal把late Stop与final apply/save排序；
- cancellation能重施加某个durable candidate；普通异常尝试恢复incoming。

算法债务：

- Stop保留的`best_valid`只是100-shot point ratio的赢家，存在winner's curse；它未独立validation。
- 尚无valid measurement时，Stop可以保留只写盘/已apply、尚未测完的candidate；这是可恢复性best，不是科学best。
- 每个candidate NPZ复制全部历史且测前/测后完整rewrite。真实shape单phase约4.97 MB，120 candidates至少约1.2 GB phase bytes写入，尚未计Dropbox同步和O(K²)history。

建议不改lifecycle骨架，只改术语与存储：

- artifact标记`unmeasured/coarse/validated/accepted`，不得都叫best；
- Stop默认保留confidence-ranked coarse best，UI明确未validation；若用户更重视科学保守性则恢复incoming；
- 每个candidate只保存自己的measurement摘要；整run history在terminal写一次，避免每个NPZ复制前史。若坚持单文件无sidecar，则接受rewrite成本并不把它误诊为solver慢。

## 10. 测试审查

### Solver tests

`test_virtual_physics.py`中的solver组有真实价值：

- authored pupil validation；
- selected DFT与shifted full FFT等价；
- returned canonical float32 exact quality gate；
- fixed state reuse/invalidation/stale stop；
- checkerboard holes和fallback；
- deterministic artifacts。

关键缺口：

- 全部主要quality测试是`64–80 px`，没有真实`1024×1272`预算守卫；
- 没有Gaussian fidelity test；
- Flat Top只在300轮终点测interior percentile，不测stagnation/更少轮；
- 没有ghost/background/roughness，虽文档声称报告；
- 没有提供pupil A的state被pupil B错误复用的caller-contract测试；
- 没有Task调用必须达到solver gate的测试。

### Feedback tests

优点：

- exact grouped measurement、all-shot dark subtraction、Welford常量SEM；
- sparse registration拒绝边界；
- independent validation；
- Stop/terminal/artifact ordering覆盖很强；
- live history schema稳定。

缺口/false confidence：

- 大多数controller测试monkeypatch出零SEM、exact unity vectors；
- “wide uncertainty”只证明2% SEM会拒绝，没有证明0.5%+1% extrema gate可通过；
- virtual feedback测试只期待两轮失败并restore，不验收收敛；
- missing test刻意把同一phase下一轮重新测当成功路径，未防止persistent missing消耗120 candidates；
- 无noise Monte Carlo、rollback、gain、patience、best selection bias；
- 无pupil/Pattern/wavefront authority；
- candidate artifact不守solver metadata。

实施阶段最小红测试应是：

1. Task initial/hot phase都满足solver canonical gate；
2. 100-shot realistic Bernoulli/photon model不会被1%point gate假宣称成功；
3. persistent missing只重试同candidate有限次数，不产生82个同phase candidates；
4. score显著变差会rollback并降低gain；
5. true-uniform synthetic data的validation coverage达到预设confidence；
6. Gaussian有显式signal region、meaningful FOM与early stop；
7. Task使用冻结pupil/Pattern authority；
8. real-shape sparse/dense performance budget分开守。

## 11. 文档—实现矛盾

| 文档声称 | 当前事实 |
|---|---|
| initial solve“至少”12轮建立fixed phase | Task显式`iterations=12`，恰好12轮就应用，实测ratio仍1.189 |
| subsequent 8轮site-amplitude优化足够支撑反馈 | hot8实测ratio1.027；继续到已有gate只多约0.06 s |
| 当前报告focal-plane ghost/background/efficiency | metadata只有ratio/RMS/efficiency；无ghost/background |
| image走full-resolution MRAF | Gaussian无noise region、random initial phase、无MRAF FOM/stagnation |
| 100-shot validation可支撑1% terminal | artifact SEM 4.5–14%；Monte Carlo显示当前gate统计上几乎不可达 |
| invalid candidate在curve占位 | persistent invalid实际上把同一phase反复编号为新candidate/update |
| best按site spread且Stop保留best | best是高噪point extrema winner，非confidence best或validated best |
| optimizer warm start只传Pattern/base canonical phase | Feedback只拿device full last-commanded phase，可能含wavefront层 |

## 12. 需要用户裁决

### U-SLM-1 — 100 shots的含义（已有D-016）

1. **推荐：100 shots只做coarse control；final用独立adaptive large sample并报告CI。**
2. 要求100 shots直接证明1% max/min；以当前observable的信息量不可实现，必须改变observable/model或放宽“证明”。

### U-SLM-2 — 最终均匀的是哪个observable（已有D-017）

1. all-shot dark-subtracted fluorescence mean；包含loading probability与conditional fluorescence；
2. occupied single-atom fluorescence；需要mixture/soft occupancy estimator；
3. trap depth/light shift/frequency；需要独立测量链。

当前Task只应声称选项1，不能把它叫trap-depth 1%。

### U-SLM-3 — Feedback phase authority（已有D-015）

选择“只更新Pattern并保留pupil/Zernike/steering”，或“Task接管完整science phase”。审计推荐前者，但需要把Editor context变成显式Task输入。

### U-SLM-4 — Gaussian/MRAF signal region

选择finite-cutoff Gaussian target、显式signal mask，或承认Gaussian走GS而非MRAF。审计推荐finite-cutoff materialized target，避免第二mask truth。

### U-SLM-5 — Stop保留什么

1. **推荐：保留confidence-ranked coarse best，明确标记未validation；无可信best则incoming。**
2. 保留最新durable phase，优先可恢复性；
3. 只有validated phase可保留，否则incoming。

### U-SLM-6 — Dense solve性能目标

Gaussian/Flat Top真实shape的15–27秒是否为产品不可接受？若是，先做MRAF语义/early-stop，再决定GPU optional dependency；sparse feedback不需要GPU。

## 13. 最小目标算法

```text
Freeze target + Pattern/base + pupil + wavefront context
  -> WGS-Kim: freeze far phase at 12, continue to canonical numerical gate
  -> Apply candidate
  -> sequential coarse shot batches
  -> vector mean + covariance + effective n
  -> uncertainty-weighted, clipped log residual
  -> trust-region target update
  -> fixed-phase hot solve to numerical gate
  -> compare confidence score; accept / rollback / reduce gain
  -> stop at noise floor or improvement patience
  -> one locked best candidate
  -> independent validation with simultaneous ratio CI
  -> report estimate + CI; only claim 1% if upper bound passes
```

这条设计保留现有最有价值的部分：唯一target、selected DFT、WGS-Kim fixed phase、exact camera groups、Welford-style streaming、independent validation和durable Stop。需要删除的是point-extrema驱动、固定8轮、无界invalid重试和不可实现的validation gate，而不是重写整个SLM层。

## 14. 外部依据

- [Di Leonardo, Ianni, Ruocco — Computer generation of optimal holograms for optical trap arrays](https://doi.org/10.1364/OE.15.001913)
- [Pasienski, DeMarco — MRAF original paper](https://arxiv.org/html/0712.0794v1)
- [Kim et al. — fixed-phase WGS and adaptive camera correction](https://arxiv.org/abs/1903.09286)
- [Tamura et al. — in-trap fluorescence feedback](https://doi.org/10.1364/OE.24.008132)
- [Nogrette et al. — closed-loop single-atom microtrap arrays](https://doi.org/10.1103/PhysRevX.4.021034)
- [slmsuite official compressed spot / feedback implementation documentation](https://slmsuite.readthedocs.io/en/latest/_autosummary/slmsuite.holography.algorithms.CompressedSpotHologram.html)

只用这些原论文/作者预印本/官方实现作方法对照；没有把二手博客或通用推荐当权威。
