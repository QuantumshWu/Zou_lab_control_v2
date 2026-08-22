# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

本次Feedback修复起点HEAD为`289f9a5 fix(slm): retain feedback previews across runs`（起点相对`origin/master` ahead 1）；M7暂停，不访问真实hardware。

Histogram/tick candidate已冻结：`bins`保留一次必要的exact sample projection，`density`/`cumulative`只重建accepted bins的representation，不再扫描full payload；qCMOS两种representation的P50从73.049/71.301 ms降至7.448/7.709 ms，bins 71.539→71.957 ms无material change，10组DPR1/2 RGBA exact。SmartOffset在枚举旧settled lattice前先按`max_ticks`拒绝过密候选，百万级range切换不再卡住。Plot全套`423 passed`、formal/focused `72 passed`，production净增17，无新class/kind lane。

SimulationWorld ownership candidate已冻结：apparatus root `simulation`是image/grid/seed/profile的唯一持久化truth；`camera.virtual`只声明exposure并消费shared world geometry，virtual MOT geometry独立。旧camera-owned fields和缺root mapping都strict refusal，不保留compat/dual owner；workspace-relative profile在device factory前通过contained durable path解析，Device Manager保留root mapping。最终定向`28 passed`，无新production file/class。

SLM server candidate已收敛为Pulse同款入口：apparatus只有`slm.hamamatsu_x15213`一个真实类型，init参数只是server host/port；只有server持有本机DVI/USB输出、profile与correction。server默认恢复原可用DVI exact-raster presenter，不需vendor DLL；USB仅在显式`--transport usb`时启用。客户只在installation握手、phase apply及unknown outcome恢复时联网；其他Editor state读取本地immutable cache。

Registered SLM Feedback candidate已根修：Calibration只产生通用camera/readout artifact，无Science Context UI/Task参数。Feedback同时选Calibration+Science Context，在自己的组合边界做Target X/Y→camera X/Y直接注册，不枚举翻转/旋转/轴交换。普通Calibration只观测34/35 sites时，Feedback可恢复第35个weak site；规则对称grid不需fiducial。

当前Feedback/Simulation根修尚未提交。Calibration只提供Target→camera BOX注册与dark baseline；metric是canonical Camera Measurement的`xx-shot` readout BOX average，不使用Calibration bright response。历史证据已经闭合：milestone前真实run为500 shots、固定`0.25`，`5.484→2.513→1.478→1.185`；`301c5e0`未经用户裁决把默认降到100/100，M6又加入hard shrink/rollback，实验run因此35/35 update为0。当前已恢复默认500-shot coarse、`0.25`、8 updates和3000-shot/300秒validation；normal terminal/Stop保留最后valid state，terminal validation写回curve。

SimulationWorld中未经授权的逐site detector gain/ellipse/angle/skew、target-specific 21-cycle ripple以及nominal/extra双trap状态已删除。当前唯一physical roster是commanded phase经共同pupil illumination、共同low-order aberration和FFT得到的dominant peaks；所有peaks走一个Fourier→camera affine，Camera只平移一个共同非对称aberrated PSF。用用户实际spacing-15 Context重做Calibration得到exact 35 sites、Target顺序完全一致、中心偏差<0.18 pixel。默认stochastic链的coarse为`1.683→1.330→1.235→...`，1000-shot validation `1.241`；hidden oracle为expected BOX `1.445→1.123`、depth `1.166→1.074`、mean signal `+1.6%`。Missing-site完整链为censored `2748.33→6.121→3.852→...→1.088`，80-shot validation `1.096 [1.066,1.126]`。

最终frozen repair为12 files `+682/-887`（净删205）：production净删233、tests净删6、active docs/规则净增34。无新增file/class/lane；production helper净增1，test function净增1。旧`_extra_*`、nominal peak state、逐site PSF/gain、adaptive gain/rollback与`best_uniformity` production引用为0。验证均从当前checkout打印路径：Atom全包`324 passed in 216.87s`；Simulation physics `45 passed`；Feedback fast `33 passed`加完整missing-site `1 passed`；Runtime/host `18 passed`；正式Qt同轮generation gap与terminal republish均保持4 panels；AST和diff-check green。Candidate已完成，可单commit。
