# Zou Lab Control v2 — Handoff

当前唯一实施入口：

1. [ZLC_V2_IMPLEMENTATION_GOAL.md](ZLC_V2_IMPLEMENTATION_GOAL.md) — 用户批准的完整目标、纪律、milestones与Definition of Done；
2. [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md) — 批准的Target不变量；
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 持久Checkpoint、当前状态、证据和下一步。

恢复任务时依次完整读取上述三份文件，再读取当前Checkpoint所引用的`AUDIT/`报告并核对HEAD/dirty state。Package README、旧GOAL、survey、acceptance、历史contracts和旧tests不是实施authority。

当前HEAD为`9571cd6 Cache smart ticks and make camera units capability-aware`。Milestone 1–6、M4 performance closure、Plot projection/fit baseline与Axis/camera/fit-style follow-up已提交且sweep complete。当前用户只批准四个后续项：Histogram/tick配置、SimulationWorld ownership、registered SLM Feedback与SLM local/remote owner；不进入M7，不访问真实hardware。

Histogram/tick candidate已冻结：`bins`保留一次必要的exact sample projection，`density`/`cumulative`只重建accepted bins的representation，不再扫描full payload；qCMOS两种representation的P50从73.049/71.301 ms降至7.448/7.709 ms，bins 71.539→71.957 ms无material change，10组DPR1/2 RGBA exact。SmartOffset在枚举旧settled lattice前先按`max_ticks`拒绝过密候选，百万级range切换不再卡住。Plot全套`423 passed`、formal/focused `72 passed`，production净增17，无新class/kind lane。

SimulationWorld ownership candidate已冻结：apparatus root `simulation`是image/grid/seed/profile的唯一持久化truth；`camera.virtual`只声明exposure并消费shared world geometry，virtual MOT geometry独立。旧camera-owned fields和缺root mapping都strict refusal，不保留compat/dual owner；workspace-relative profile在device factory前通过contained durable path解析，Device Manager保留root mapping。最终定向`28 passed`，无新production file/class。

SLM server candidate已收敛为Pulse同款入口：apparatus只有`slm.hamamatsu_x15213`一个真实类型，init参数只是server host/port；只有server持有本机DVI/USB输出、profile与correction。server默认恢复原可用DVI exact-raster presenter，不需vendor DLL；USB仅在显式`--transport usb`时启用。客户只在installation握手、phase apply及unknown outcome恢复时联网；其他Editor state读取本地immutable cache。

Registered SLM Feedback candidate已根修：Calibration只产生通用camera/readout artifact，无Science Context UI/Task参数。Feedback同时选Calibration+Science Context，在自己的组合边界把frozen Target注册到camera sites，生成含zero-capture predicted boxes的stable full roster；alternate mapping loud要求asymmetric fiducial。已验证普通Calibration只观测34/35 sites时，Feedback可恢复第35个weak site并完成candidate链。

Feedback相关12 files为production `+1163/-282`（净+881）、tests `+924/-91`（净+833），无新file/class/lane；6个新production defs只是registration validator/constructor、censor/boost、incoming candidate与coarse measure，均有直接consumer。Owner Feedback `34 passed in 43.68 s`、physics/truth `24 passed`、descriptor `3 passed`、独立Feedback+truth `41 passed in 45.53 s`，12 files AST与diff-check green。Stop-before-solve为0 solver calls。Calibration registration capture不claim SLM，实验workflow必须保持Context command；Feedback启动时对current phase/receipt exact fail-closed。Combined Gate 17/18已complete/sweep complete：38 files `+3702/-638`（净3064），production净+1453、tests净+1476、docs净+130、launcher +5；test functions净增16，无剩余blocker/dead owner/旧API/double truth/safe deletion。首轮全树`1555 passed, 1 failed, 4 skipped, 6 warnings in 475.07 s`，唯一gallery offscreen Qt teardown subprocess exit `0xC0000005`不冒充green；该exact gallery随后三次pass，第二次完整全树`1556 passed, 4 skipped, 6 warnings in 466.48 s`。Warnings仅vendor SWIG，skips仅无Icarus。Candidate已ready to commit；官方USB ABI、真控制器/profile/correction/光学settle与M7 single-distribution/fresh-install仍是明确deferred。
