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

当前Feedback follow-up已冻结并随本阶段实现提交。用户确认probe相对atomic transition红失谐，trap light只会把失谐进一步推红；因此当前mode改为single-frame多shot BOX双高斯的`bright_mean-dark_mean`。Calibration只提供site BOX geometry；Pulse由operator显式选择，camera exposure是独立可编辑字段并默认`0.1 s`，Feedback不读取Calibration dark/threshold/exposure/photoelectron/working point。Controller保存每site的weight/contrast/error/fit/action与局部响应历史：有效site按“contrast大→trap浅→增加Target”更新；uncertain fit hold；never-valid单峰最多三次1.4x bootstrap；曾有效且加权后bright峰消失则回到上一有效weight。默认500-shot coarse、12 updates、3000-shot/300秒validation。

SimulationWorld保持同一physical roster/Fourier→camera affine/shared aberrated PSF；默认正的trap light-shift参数现在作为进一步红移的幅度，occupied bright-dark随depth单调下降，loading随depth上升，Camera shot产生真实dark/bright mixture。产品默认500-shot missing-site链的valid coarse ratio为`2.092→1.971→1.687→1.474→1.331→1.243→1.169→1.128→1.090`；independent validation在2000/3000 shots时为`1.09126 [1.08293,1.09989]`并accepted，validation墙钟84.0秒。

本长任务另有三个已提交阶段：`bf3133e`使repeat默认reduce、只允许operator显式repeat facet；`78a71fd`在Camera Restart前排空旧causal generation的在途surface，消除publication generation race；`ace4a19`闭合persisted point-facet Histogram与真实NodeHost test-contract残余。Feedback为35项、Simulation physics 45项、Atom 325项、Plot 428项green；Workbench排除既有batch CRLF参数项后的392项完成无失败。最终diff/AST/active-name与Gate 17/18审计均已闭合。
