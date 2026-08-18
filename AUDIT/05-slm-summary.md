# 05 — SLM target、solver、device与Feedback综合审查

状态：重点阶段完成。
基线：`92089f5fc037f8a87e8efe834ccf83139aaf4383`
分报告：

- `05a-slm-truth-context.md`：target/pupil/Pattern/wavefront/device/Feedback context；
- `05b-slm-solver-feedback-algorithm.md`：sparse/dense solver、估计、控制、validation与噪声；
- `05c-slm-device-editor.md`：X15213/Virtual adapter、LUT/correction、Editor与真机证据边界。

## 1. 总结论

SLM层不是“全部推倒重写”。它已经有几块值得保留的正确骨架：

- 唯一二维non-negative target intensity；
- 唯一canonical float32 phase；
- sparse WGS-Kim与fixed far-field phase；
- 小Cartesian support的exact selected DFT；
- Editor latest-only solve、stale Send、独立command worker和DeviceUse claim；
- adapter内部phase-code -> correction -> LUT的单一mapping owner；
- canonical phase NPZ的exact numerical replay；
- Feedback的exact camera grouping、streaming statistics与独立validation概念。

真正的问题是这些局部正确组件之间没有传递同一份光学与物理上下文：

```text
Target intent
  × objective is lost on save
  × pupil/wavefront is not passed to Feedback
Pattern/base phase
  × full science phase is reused as if it were base
Device command
  × real initial state can be fictional zero
  × correction mapping can change outside Task lease
Measurement
  × 100-shot point extrema is treated as 1% evidence
Controller
  × no uncertainty/trust/rollback/recovery
Artifact
  × canonical array lacks solve/device context and physical receipt
```

因此当前系统能在Virtual/软件局部测试里“每层都绿”，却不能保证：Feedback延续Editor实际光路、Stop恢复真实硬件状态、candidate可物理复现、或100 shots足以证明1%均匀。

## 2. P0/P1问题地图

| ID | 严重度 | 结论 |
|---|---|---|
| SLM-C01 | P0 context | Feedback不接objective、Gaussian pupil、Zernike/steering；使用`auto + default hard circle`并把full science phase当Pattern warm start。 |
| SLM-C02 | P0 truth | real X15213从未send/read当前phase，却把zero报告为`last_commanded_phase`；failure restore可覆盖真实未知画面。 |
| SLM-C03 | P0 outcome | USB/DVI side effect之后的ack/readback失败可能令physical=new、software=old；当前contract没有unknown/ambiguous outcome。 |
| SLM-C04 | P0 lifecycle | correction load/enable绕过DeviceUse，可在Feedback或Send期间改变phase->gray mapping，且artifact不记录。 |
| SLM-C05 | P0 statistics | 100-shot per-site SEM约4.4–5.1%（旧run可10–14%）；35-site `max/min`噪声地板约1.2，不能证明1.01。 |
| SLM-C06 | P0 control | Controller追逐最新point extrema，没有uncertainty weighting、trust region、rollback或persistent-invalid recovery。 |
| SLM-C07 | P1 inner solve | 初始12轮/hot 8轮被当硬停止；真实shape hot8 ratio约1.027，继续到已有1%数值gate只多约0.06秒。 |
| SLM-C08 | P1 dense | Gaussian没有MRAF noise region、随机初相、固定300轮、FOM无意义；真实shape约27秒，Flat Top约15秒。 |
| SLM-C09 | P0 calibration | serial profile没有实测provenance/subtype/response；近乎线性不等于错误，但不能称“本机calibrated curve”。 |
| SLM-C10 | P0 correction | wavelength conversion是X后Y separable 1D unwrap，不是一般二维phase unwrap；含residue图可明显不同。 |
| SLM-C11 | P0 DVI/SDK | ctypes ABI、mode switch/reboot/reopen、GPU gray/scanout与settle均未由真SDK/真controller证明。 |
| SLM-C12 | P1 artifact | target JSON丢objective；candidate不保存actual updated target、pupil/wavefront/device mapping/receipt。 |
| SLM-C13 | P1 UI truth | Feedback改变device后，open Editor仍显示旧local draft；下一次Send可静默覆盖外部新state。 |
| SLM-C14 | P1 contract | descriptor接受generic target，Feedback实际上只接受与Calibration sites一一对应的sparse support。 |

## 3. Solver裁决

### 3.1 Sparse spots：保留，不要再误诊为FFT瓶颈

真实`1024 x 1272`、5×7 default solve约0.53秒并达到simulated support ratio约1.009。hot state继续到同一gate约0.29秒。一次100-shot物理candidate最少数秒，因此节省inner solver几十毫秒却增加一个outer candidate是反优化。

保留：

- WGS-Kim；
- iteration 12 freeze far-field phase；
- caller-owned short-lived optimizer state；
- selected DFT与cost guard；
- canonical return后的exact numerical quality proof。

修改方向不是换GPU，而是：freeze后继续到数值gate；Task保存iterations/efficiency/ratio；只在同一target support、pupil和Pattern authority下复用state。

### 3.2 Dense image：重新定义问题后再优化backend

当前Gaussian全raster严格正，`desired > 0`使noise region为空，因此所谓MRAF退化。tail极小值又让max/min达到`10^12–10^14`，metadata不具物理意义。应先定义finite signal region、noise freedom、vortex-free初相、meaningful FOM与stagnation/early stop；完成后再profile FFT workers/GPU。当前证据不支持把GPU作为第一修复。

### 3.3 Codec：exact phase可留，target/project需升级

- canonical phase NPZ精确且快速，继续作为“exact science array replay”。
- dense JSON为单一strict codec，但1.3M floats约6.64MiB，同步Qt保存约1.68秒；更重要的是不保存objective。
- target artifact应保存`intensity + objective_kind`；pupil/Zernike属于science context，不应塞回target。

交叉复核统一术语：`TargetArtifact`严格不含pupil；若UI需要“一次保存整个工程”，另有`SlmProjectArtifact`组合引用Target与Science Context。`05c`中“target/project保存objective+pupil”只能解释为project bundle，不能让pupil重新成为target的第二truth。

## 4. Feedback估计与控制裁决

### 4.1 当前observable

Task当前计算all-shot BOX count减Calibration dark mean。它包含loading probability和occupied fluorescence，不等于occupied single-atom fluorescence，更不等于trap depth。只要产品这样定义，它可以作为coarse feedback observable；但artifact/UI必须用准确名称，不能声称“1% trap depth”。

### 4.2 100 shots只能coarse

当前artifact给出的per-site mean uncertainty为数个百分点。对35个真值完全相同的sites取极值，point ratio天然远离1；即使每site SEM刚好0.5%，完美均匀阵列通过当前`max/min<=1.01`的概率仍极低。

因此需要分开：

1. 100 shots：快速coarse update、估计noise floor、允许early plateau；
2. final validation：独立adaptive sample，报告simultaneous confidence interval；
3. 只有CI upper bound通过才宣称1%，否则报告estimate与uncertainty，不无限循环。

### 4.3 Controller

当前`target_i *= (GM/F_i)^0.25`方向本身合理，但没有不确定度、step clip和best rollback。推荐在log residual上：

- shrink noisy sites；
- clip per-site step；
- compare candidate improvement confidence；
- 变差则rollback/reduce gain；
- invalid连续发生时停止/恢复，而不是同一phase编号并重测到120轮；
- best用confidence score，不能从120个高噪extrema里挑幸运minimum。

### 4.4 Stop

“Cancel/restore incoming”与“Accept current best”是两个不同动作。当前Stop选择best/latest durable但Host标cancelled、artifact又不是正式terminal result，且用户看到的current preview未必是被保留的phase。推荐显式拆成两种用户动作；若暂时只留Stop，必须先由用户选择语义并让device/preview/artifact/Host terminal一致。

## 5. 光学与device truth裁决

### 5.1 Science context

推荐显式冻结：

```text
TargetArtifact(intensity, objective)
ScienceContext(pattern_base, numeric pupil amplitude, wavefront operator)
DeviceMapping(profile, wavelength, orientation, correction revision)
```

Feedback只更新Pattern/base，随后叠加冻结的wavefront，再在冻结device mapping下apply。不要从open Editor私有内存临时偷context；推荐显式artifact/command receipt，UI只是作者和观察者。

### 5.2 Known/unknown command state

`last_commanded_phase`必须允许unknown。Virtual world的initial phase可以known；real X15213只有以下情况可known：

- 本进程成功完成一次定义明确的apply/confirmation；
- 有受信command artifact并按产品定义takeover；
- 真controller提供足以证明当前displayed command的readback。

未满足时Feedback应拒绝restore语义或要求显式takeover；不能用zero占位。

side effect之后的失败同样必须区分known-old、known-new与unknown，不能始终保留旧software truth。

### 5.3 Profile/correction/DVI

当前phase-code -> correction add/mod256 -> nonlinear LUT顺序清楚，在给定输入假设下可保留。尚未通过的是输入与物理证据：

- `LSH0804382.json`应标为unverified device data，直到给出type、原始测量/厂商来源、温度/波长和response timing。
- wavelength conversion要么只接受同波长vendor map，要么实现/验证真正二维phase unwrap与residue policy。
- USB path先作为主要验收路径，核官方header ABI、mode、serial、write/display/readback和optical settle。
- DVI在GPU scaling/color/dither/scanout和mode lifecycle验收前应标Experimental；exact client geometry不等于exact gray drive。

## 6. 推荐目标闭环

```text
Freeze target + science context + device mapping
  -> solve Pattern with correct objective/pupil
  -> WGS freeze at 12, continue to canonical numerical gate
  -> compose frozen wavefront
  -> apply and obtain command receipt
  -> bounded running camera preview
  -> coarse sequential batches with uncertainty
  -> clipped trust-region target update
  -> hot fixed-phase solve to numerical gate
  -> accept / rollback / reduce gain
  -> stop at measured noise floor or patience
  -> lock one confidence-ranked best candidate
  -> independent adaptive validation with simultaneous CI
  -> persist target + science context + mapping + receipt + evidence
```

这条路径复用现有好组件，不新建第二solver、第二phase格式或vendor framework。

## 7. 测试与验收边界

现有tests强项：canonical math、selected/full transform等价、optimizer state invalidation、Editor latest/stale/lease/close、phase-code顺序、USB fake readback、Feedback lifecycle/race。

现有tests不能证明：

- objective/pupil/wavefront跨Editor->Feedback保持；
- unknown real initial state可恢复；
- side-effect failure后的physical/software一致；
- serial profile来自真测量；
- correction二维换波长正确；
- DVI/USB SDK ABI与optical phase；
- 100-shot 1%统计可行；
- dense MRAF的物理FOM；
- external Feedback后Editor不会覆盖device。

实验机验收必须分别记录：SDK/type/serial、phase response原始数据、correction方向/符号/波长、USB/DVI gray、orientation、rise/fall settle、close/retain、command receipt和光学A/B。Mock byte tests不得改名或扩写为optical acceptance。

## 8. 需要用户优先裁决

1. Feedback只更新Pattern并保留wavefront，还是接管完整science phase？推荐前者。
2. science context来自显式artifact，还是依赖open Editor？推荐artifact。
3. unknown initial real phase时拒绝Feedback，还是Init主动发送一个已知phase？两者均可，禁止虚构zero。
4. 100 shots是否明确只作coarse，final用adaptive independent validation？推荐是。
5. 当前要均匀的是all-shot fluorescence、occupied fluorescence还是trap depth？当前Task只能诚实声称第一项。
6. Stop是否拆成`Accept best`与`Cancel/restore`？推荐拆。
7. dense Gaussian/Flat Top是否属于近期产品性能目标？若是，先修MRAF定义再谈GPU。
8. X15213 serial profile放package还是installation/workspace calibration artifact？推荐后者。
9. correction只接受同波长map，还是正式支持波长转换？需按实验需求选择。
10. DVI在完整真机验收前是否标Experimental？推荐是。
11. candidate artifact最低复现等级：canonical only、可重建gray，还是附exact gray/receipt？推荐至少可重建gray+receipt。
12. Feedback是否明确sparse-only？推荐当前明确，dense observable另设计。

## 9. 修复优先顺序（未实施）

1. 先裁决phase authority、observable、unknown hardware和100-shot语义。
2. 旧红锁target objective/pupil/context与unknown command。
3. Feedback inner solve走完现有便宜数值gate。
4. estimator/controller引入uncertainty、trust、rollback、invalid stop。
5. validation改CI；Stop动作/terminal统一。
6. correction纳入lease，artifact记录actual target/science/device mapping/receipt。
7. Editor显示external device divergence。
8. 真机核profile、ABI、correction、orientation与settle。
9. 最后单独重做dense MRAF；不要让它阻塞sparse feedback修正。
