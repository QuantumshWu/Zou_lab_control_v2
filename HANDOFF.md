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

当前Feedback follow-up使用single-frame多shot BOX分布的`bright_mean-dark_mean`。Calibration只提供site BOX geometry；Pulse、camera exposure、shots、single-Gaussian boost、feedback gain和普通单步上限均由operator显式设置。每candidate严格一批shots；单/双高斯只要返回有限fit即valid，只有数值失败hold。Controller在归一化Target intensity share上保证dark-only site按用户参数绝对增加，invalid share不变，loaded site只在剩余功率内相对配平；per-site dark/loaded历史形成不可下穿的loading floor。默认100 shots、12 updates，无retry或独立validation采集。

SimulationWorld保持同一physical roster/Fourier→camera affine/shared aberrated PSF；occupied bright-dark随depth单调下降，loading随depth上升。使用用户真实`calibration-5.json`（23个detected sites）和`science-context_5x7.npz`（35个Target sites）的100-shot链，observable sites为`23→26→29→32→32→33→35`，candidate 7测得全阵列ratio `1.098487`；Task仍完成全部authored updates后再选择最佳candidate。

本长任务另有三个已提交阶段：`bf3133e`使repeat默认reduce、只允许operator显式repeat facet；`78a71fd`在Camera Restart前排空旧causal generation的在途surface，消除publication generation race；`ace4a19`闭合persisted point-facet Histogram与真实NodeHost test-contract残余。Feedback为35项、Simulation physics 45项、Atom 325项、Plot 428项green；Workbench排除既有batch CRLF参数项后的392项完成无失败。最终diff/AST/active-name与Gate 17/18审计均已闭合。
