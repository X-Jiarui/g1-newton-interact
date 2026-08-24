# 迁移到 Newton 原生物理与碰撞:执行清单与验收标准

本文是执行文档,不是提案。每一项都写明**做什么**、**怎么验收**、**失败时怎么办**。
末尾两节是硬性检查表:第 6 节是已经踩过的 15 个移植缺陷,第 7 节是过程性陷阱。
新路径上每一项都必须重新验证一次 —— 它们全部是"不报错、返回零、看起来正常"的类型。

---

## 1. 现状:我们现在到底在跑什么

当前配置是 **Newton 的 `SolverMuJoCo`**,它是 MuJoCo-Warp 的封装。

| 组件 | 实际由谁负责 |
|---|---|
| 模型表示、多世界复制、warp 内核编排、统一 API | Newton |
| **刚体动力学** | MuJoCo-Warp |
| **碰撞检测与接触求解** | MuJoCo-Warp |
| 传感器 | MuJoCo(我们移植进 spec 的 140 个) |
| MDP(观测/奖励/终止/指标) | mjlab 的 manager,原样复用 |

### 已经拿到的收益(不要在迁移中丢掉)

- CUDA graph 捕获:2048 env 下 **2.15×**(4090 上 80.4s → 37.4s/轮;5090 上 53s → 19.5s/轮)
- 多世界批处理:2048 env 单卡
- 与 mjlab 的逐项可比性 —— 这是找出全部 15 个缺陷的唯一手段

### 已经证伪的说法(不要再重复)

- ~~"Newton 撞精确 SDF,mjlab 撞凸包"~~ —— **两半都错**。实测 `nplugin=0`、`geom_plugin=-1`、几何带 `graphadr` 凸包图、`newton shapes with sdf: []`。
- `--sdf-object` 这个开关**从未产生过 SDF 碰撞**。`mesh.build_sdf()` 确实执行了,但 `SolverMuJoCo` 的形状转换里没有任何生成 `mjGEOM_SDF` 的路径,而 mujoco_warp 的 `has_sdf_geom = (geom_type == mjGEOM_SDF).any()` 因此恒为 False,`sdf_narrowphase` 永不执行。
  - 佐证:`--sdf-resolution` 取 128 / 256 / 384,穿透深度**逐位相同**。
  - **行动项**:这个 flag 必须改名(建议 `--object-mesh`),否则会继续误导。

---

## 2. 目标路径与其唯一可行组合

Newton 1.5 的 9 个求解器中,能带 69 关节浮动基座机器人的**只有一个**:

| 求解器 | articulation 提及 | joint_target | eval_fk | 结论 |
|---|---|---|---|---|
| **Featherstone** | **101** | 11 | 5 | 唯一候选 |
| VBD | 0 | 14 | 3 | 软体/布料 |
| XPBD | 0 | 7 | 0 | 粒子/约束 |
| Style3D / Kamino / SemiImplicit | 0 | ≤2 | 0 | 不适用 |

碰撞侧 Newton 有完整原生管线(`newton/_src/geometry/`):宽相 SAP/NxN/BVH、窄相、凸碰撞、接触归约。
**hydroelastic 不绑定在 `coupled` 求解器上** —— 它在共享窄相里,逐 shape 用 `ShapeFlags.HYDROELASTIC`(=16)开启,
`narrow_phase.py:1496` 明确写:`Set is_hydroelastic=True on shapes to enable hydroelastic collisions`。
网格 SDF(`mesh.build_sdf()`)正是 hydroelastic 的输入(`builder.py:6402`)。

**目标组合:`SolverFeatherstone` + Newton 原生碰撞 + hydroelastic SDF 接触。**

---

## 3. 决策门:先量,再决定要不要迁

迁移是重写级工作量。在投入之前必须先回答两个问题,任一为否就停下。

### 门 A:凸包是不是真的瓶颈?(小时级,不动架构)

**做什么**:对杯子做凸分解(多个凸块保留把手的洞与内腔),替换单一凸包碰撞体,重训或续训。

**背景数据**:

| 物体 | 真实体积 | 凸包体积 | 凸包/真实 | 现象 |
|---|---|---|---|---|
| 订书机 | 65.7 cm³ | 114.1 cm³ | 1.74× | **学会了**:抓起、举 61cm、放回 |
| 杯子 | 181.2 cm³ | 638.3 cm³ | **3.52×** | **没学会**:碰一下推倒,抬升 1.92cm |

**先做更便宜的一步**:读 `mug_drink_4` 的接触标注,确认参考动作里指尖接触杯子的**哪个部位**。
若参考根本不穿把手,则凸包不是主因,门 A 直接失败,不要做凸分解。

**验收标准**:
- 凸分解后杯子的 `PhaseA/object_mpjpe_mm` 与订书机同量级(< 10mm)
- 评测中 `max object rise > 10cm` 且抬升期间 `hand-obj < 0.03m`(即抓着走,不是打飞)
- 若达标 → 说明瓶颈是几何保真度,hydroelastic 值得评估
- 若不达标 → 瓶颈在别处,**不要迁移**,回到 MDP/奖励层面排查

### 门 B:Featherstone 能不能带这台机器人?(半天,不动主线)

**做什么**:独立探针脚本,同一台 G1 + Wuji 手(69 关节)、同一场景,只用 Featherstone。

**验收标准**(全部满足才算通过):
- [ ] 模型能构建,关节数 = 69,浮动基座正确
- [ ] 静止站立 1000 步不发散(根节点高度漂移 < 1cm,无 NaN)
- [ ] 位置伺服可用:给定关节目标,稳态误差 < 0.02 rad
- [ ] 吞吐实测:记录 128 / 512 / 2048 env 下的 步/秒,与当前 MuJoCo-Warp 路径对比
- [ ] 与物体发生接触时不穿透、不爆炸
- [ ] hydroelastic 打开后,订书机静止穿透 < 0.5mm(当前 MuJoCo 路径是 0.04mm)

**失败处理**:任一项不达标 → 记录数据,**停止迁移**,当前路径继续。
不要"边迁边修" —— 那会同时失去参照系和可用的训练。

---

## 4. 迁移工作项(仅在门 A、门 B 都通过后启动)

按依赖顺序排列。每项独立可验收,不允许跨项一起改。

### 4.1 状态桥接层重写

**现状**:`newton_bridge.py` 全部建在 `mjw_data` 的 MuJoCo 布局上 —— `qpos`/`qvel`/`xpos`/`xquat`/`sensordata`。
Featherstone 用 Newton 自己的 `State`/`Control`,布局完全不同。

**做什么**:重写 `NewtonEnv`,使其从 Newton `State` 读写,对上层保持 mjlab 期待的接口不变。

**验收标准**:
- [ ] `body_link_pos_w` / `body_link_quat_w` / `joint_pos` / `joint_vel` / `root_link_pos_w` 与参考实现逐值一致(误差 < 1e-6)
- [ ] **根节点角速度的坐标系**:MuJoCo 的自由关节 `qvel[3:6]` 是**体坐标系**,Newton 未必相同 —— 必须实测,不能假设(见缺陷 6)
- [ ] 写入关节角后必须跑一次前向运动学再读派生量(见缺陷 5)
- [ ] 未映射的属性一律 **抛错**,不得返回 None 或零(见第 7 节"静默零")

### 4.2 接触传感器重建

**现状**:mjlab 的 `ContactSensor` 读 MuJoCo `sensordata`;Newton 原生碰撞产出的是 `geometry/contact_data.py` 那套结构。

**做什么**:实现一个适配层,把 Newton 的接触数据映射成 mjlab `ContactData` 的字段
(`found` / `force` / `dist` / `normal` / `tangent`,形状 `[B, N, ...]`)。

**验收标准**:
- [ ] `env.scene["hand_apple_contact"]` 可解析(**这正是缺陷 15**)
- [ ] 抓握瞬间 `found > 0` 的指尖数与视频一致
- [ ] `contact_duration`(权重 1.0)与 `object_hard_lift`(权重 2.0)非零
- [ ] 与当前 MuJoCo 路径在同一 checkpoint、同一起始帧下,接触时序吻合(误差 < 5 步)

### 4.3 执行器与控制

**现状**:机器人是**位置伺服**;`set_joint_effort_target` 在当前桥接里是**有意为之的 no-op**
(实测 mjlab 中 `xml_motor_unused_*` 的 ctrl 恒为 0.0)。

**验收标准**:
- [ ] Featherstone 下位置目标真正生效,PD 增益来自执行器配置而非动作项里那套死代码
- [ ] `apply_actions` 仍在 decimation 循环内部调用(见缺陷 7)
- [ ] 同一动作序列下,关节轨迹与 MuJoCo 路径的差异在可解释范围内并记录成因

### 4.4 场景与几何

**验收标准**:
- [ ] 桌面按物体真实最低点定位(订书机 +18.7mm、杯子 −13.2mm,运行时按 clip 与网格计算)
- [ ] 物体静止穿透 < 0.5mm
- [ ] 自由关节阻尼未被丢弃(见缺陷 2)
- [ ] 平面几何未被改尺寸(见缺陷 3)

### 4.5 性能

**验收标准**:
- [ ] 2048 env 每轮耗时 **不劣于当前 19.5s**(5090)。若显著更慢,迁移的核心理由(性能)即不成立,必须重新评估
- [ ] CUDA graph 捕获可用,且数值与非捕获路径的差异在模拟器自身 run-to-run 噪声内
      (当前基线:20 步 256 env,无 graph 5 次极差 2.66e-4,有 graph 4 次极差 1.22e-4 且落在前者范围内)

---

## 5. 必须保住的基线数字(迁移后逐项复现)

这些是当前路径实测、可复现的。迁移后每一项都要重测并写进对照表。

| 量 | 当前值 | 备注 |
|---|---|---|
| 订书机评测抬升 | 61–62 cm | RSI 窗口 6 个起始帧全部成功 |
| 订书机指尖最近距离 | 0.007–0.019 m | mjlab 参照 0.032–0.035 |
| 订书机 `object_mpjpe_mm` | 3.1–3.5 mm | mjlab 2.4–2.9 |
| 物体静止穿透(订书机/杯子) | 0.04 / 0.28 mm | mjlab 解析球 0.37 |
| 传感器 | nsensor=140,contact=136,nsensordata=294 | 与 mjlab 逐项相同 |
| 2048 env 每轮 | 19.5 s(5090) | 含 CUDA graph |
| 观测组一致性 | 20 组全部匹配,最大误差 8.3e-07 | 与 mjlab 对照 |

---

## 6. 已踩过的 15 个移植缺陷 —— 新路径必须逐项重查

**共同特征:全部不报错。** 读取方拿不到就返回零或 None,训练照常跑,曲线照常涨。
新桥接上每一项都要主动验证,不能等它自己暴露。

| # | 缺陷 | 当时的症状 | 新路径的检查方式 |
|---|---|---|---|
| 1 | `default_shape_cfg.gap` 默认 0.1 | 接触需要 10cm 穿透才存在 | 建模后打印每个 shape 的 gap |
| 2 | 自由关节阻尼被 `add_mjcf` 丢弃 | 物体阻尼为 0;在 finalize 前赋值无效 | 建模后读回阻尼值比对 |
| 3 | 平面几何被改尺寸 | 地面尺寸与源文件不符 | 逐 geom 比对尺寸 |
| 4 | 约束缓冲溢出(njmax/nconmax) | `nefc overflow`,接触被丢弃 | 压力测试下监控溢出计数 |
| 5 | 写入 qpos 后未跑前向运动学 | 关节角对但根位姿差 1.7cm | 写入后立即读派生量比对 |
| 6 | 根节点角速度写错坐标系 | 0.34 rad/s 误差,正落在观测槽 0–2 | 逐槽对照观测 |
| 7 | `apply_actions` 调用在 decimation 循环外 | 每控制步只施加一次 | 计数每控制步的调用次数 |
| 8 | 残差策略状态写到了错误的 env | 评测抬升恒为 0 | 与 play.py 校准 |
| 9 | `body_simple`/nC 误判 | Newton 有意偏移质心 1mm | 比对 `body_simple` 与 nC |
| 10 | **Newton 丢弃所有 `<sensor>`** | nsensor=0,三项接触奖励恒为 0 | 建模后断言 nsensor 期望值 |
| 11 | **`scale_by_dt` 未从配置传入** | 每项奖励都对上,总和差 **49.7×**(=1/0.02) | 比对 `compute()` 返回值与各项之和 |
| 12 | **`_reset_idx` 未调用任何 manager 的 `reset()`** | 回合累计、metrics、动作历史跨回合泄漏 | 断言每个 manager 的 reset 被调用 |
| 13 | **MetricsManager 从未运行** | `compute(dt=)` 签名不符,被裸 `except` 吞掉 | 断言指标非零初始值 |
| 14 | **残差 runner 状态写在 vec env,读在 bridge** | 27 个 `_residual_*` 属性全部缺失,策略对自己上一步动作视而不见 | 断言两侧属性集合一致 |
| 15 | **接触传感器进了模型没进 scene** | `env.scene["hand_apple_contact"]` KeyError → 返回零;两项奖励从未生效 | 断言 scene 中传感器实体存在且数据非零 |

---

## 7. 过程性陷阱 —— 这些和代码无关,是工作方法

### 7.1 静默零

**最贵的一类。** 缺陷 10、13、14、15 全是这个模式:读取方失败时返回零而不是抛错。

**规则**:桥接层任何取不到的东西一律 **抛错**。宁可启动失败,不要训练出一条假曲线。
本次会话里,正是"未映射属性抛错"这个设计让缺陷 14 暴露出来。

### 7.2 评测路径必须与训练路径同源

- 自写 rollout 脚本已经给过**三次错误结论**。数值评测只走校准过的入口。
- 评测场景必须与训练场景一致:曾用**没有桌面修正、没有接触刚度**的旧场景去评测新场景训出来的 checkpoint。
- 两个入口若各自建场景,必须调用**同一个函数**(如 `newton_table.install`),不得各写一份实现。

### 7.3 指标的可比性

- **RSI 平均 ≠ 单条评测**:训练指标在 2048 环境、随机起始帧上平均;评测是单条确定性轨迹。
  两者可以同时为真,不得用其中一个否定另一个。
- **不同 clip / 不同 env 数的 `mean_reward` 不可比**(杯子 300+ 与订书机 50 不是一回事)。
- **比较必须在同一轮次**。曾把 Newton 第 40 轮与 mjlab 第 163 轮对比得出错误结论。
- 运行未到该轮次时显示"n/a",**不得夹取它自己更早的值**充数。

### 7.4 一次只动一个变量 + 配对种子

- 配对种子已经证明其价值:订书机 S1/S2 曾相差 4.3 倍,单条曲线会把它误读成巨大进展。
- 单 env 的截图**不能**代表多 env 的视频(多世界渲染有自动位移,单 env 恰好没有)。
  验证一律用与交付物**相同的 env 数**。

### 7.5 可视化本身会骗人

- `viewer.camera.pos = ...` 是 **no-op**(`set_model` 会重建 Camera),真正接口是 `set_camera(pos, pitch, yaw)`。
- 多世界渲染会 `_auto_compute_world_offsets` 把各 world 摊开,物理在原点、画面在天边。
- 通过 mjlab 视觉模型回放 qpos 会画出**占位球而不是真实网格**,且 mocap 按索引映射会把桌子放到脚下。
- **规则**:交付视频前自己先抽帧看。本次会话有两版视频是"看起来修好了"才发出去的,都不对。
- 跨不同景深用像素高度比较物体高低是**无效**的 —— 曾据此错判"桌子太矮"。

### 7.6 共享机器上的资源纪律

- **在有训练的 GPU 上跑评测会把训练 OOM 打死** —— 已经发生过一次,损失约一天的训练。
  评测前必须先查该卡显存与进程。
- `pkill -f <pattern>`:远程命令行若包含同一字符串会**杀掉自己的 shell**。方括号技巧防不住这种,
  应改用不自匹配的模式或分两步执行。
- vast 是**租来的共享机器**:不是自己启动的进程一律不动;密钥不外带;不在聊天里贴凭据。

### 7.7 结论的纪律

- 说"修好了"之前必须有实测数字,不能凭"应该生效"。
- 推翻自己的结论要**明说是推翻**,并指出原来错在哪(本次会话已多次:打飞 vs 抓握、
  物体表示是差距成因、S2 在刷分不干活 —— 全部基于假数据,已收回)。
- 不能用过期快照回答"现在怎么样"。

---

## 8. 执行顺序

```
门 A(读接触标注 → 若相关则凸分解)
   └─ 不达标 ─→ 停止迁移,回到 MDP/奖励层排查
   └─ 达标 ──→ 门 B(Featherstone 可行性探针)
                  └─ 不达标 ─→ 停止迁移,记录数据
                  └─ 达标 ──→ 4.1 桥接 → 4.2 传感器 → 4.3 控制 → 4.4 场景 → 4.5 性能
                                每项独立验收,不达标不进入下一项
```

**任何阶段都保留当前 `SolverMuJoCo` 路径可运行**,它是新路径唯一的参照系。

## Native SDF path: stability requires a halved physics step

Established 2026-08-24, with the robot actually simulated (see the harness trap below).

The native path went NaN in 6 of 7 runs between step 49 and 72, always while the robot was
collapsing and the contact count spiking. Bisected one variable at a time, two runs each,
600 steps, all at `kh = 1e11` unless stated:

| variant | run a | run b |
|---|---|---|
| control (`dt = 0.005`) | NaN @72 | NaN @69 |
| `kh = 1e10` | NaN @55 | NaN @49 |
| `kh = 1e9` | NaN @58 | NaN @56 |
| `kh = 1e8` | survived 600 | NaN @164 |
| `iterations=15, ls_iterations=100` | NaN @53 | survived 600 |
| `enable_multiccd=False` | NaN @54 | NaN @53 |
| **`dt = 0.0025`, decimation 4 -> 8** | **survived 600** | **survived 600** |

Softening the contact made it *worse*, not better, so stiffness is not the driver. The step size
is the last remaining difference from `newton/examples/robot/example_robot_panda_hydro.py`, which
runs `sim_dt = 1/600`. mjlab uses 0.005.

Confirmed afterwards over 450 steps with video: object penetration `-0.00 mm`, 34-38 stable
contacts every sample, robot falls over naturally and the object stays put.

**Open decision:** halving the step doubles the physics cost per control step. Before adopting it
for training, measure the throughput hit and check whether it changes the mjlab-parity baseline,
since the MuJoCo-contact path is stable at 0.005 and the acceptance criterion is parity with it.

### Harness trap: `episode_length_buf`

`apple_eat/mdp.py:1405` writes the robot's root pose and every joint straight into sim while
`episode_length_buf <= 30`. That counter is incremented inside `NewtonVecEnv.step`, so a probe
that bypasses `step()` and drives `apply_actions()` plus the physics loop directly leaves it at 0
and the robot stays kinematically pinned for the whole run. Several videos showed a robot that
"stood" for hundreds of steps and proved nothing about stability. Any such harness must advance
`episode_length_buf`, `common_step_counter` and `_env.common_step_counter` by hand.

The robot sinking after release (root_z 0.80 -> 0.57 over 90 steps) is not a defect: the
MuJoCo-contact baseline sinks at the same rate (0.8153 -> 0.7295 over 62 steps). With no policy
it simply squats.

## Native SDF path: what it costs at training scale

Measured on an idle RTX 5090, mug scene, `--cuda-graph`, against the config the VAST_MUG runs
actually train with (MuJoCo contacts, dt 0.005, 2048 env).

| config, 2048 env | ms/step | env-steps/s | vs training |
|---|---|---|---|
| MuJoCo contacts, dt 0.005 | 50.7 | 40429 | 1x |
| native, dt 0.005 | 1501 | 1364 | 29.6x slower |
| native, dt 0.0025 (needed for stability) | 2111 | 970 | 41.7x slower |

At ~20 s/iter today that is roughly 14 minutes per iteration. Scaling is healthy (801 -> 901 ->
970 env-steps/s from 512 to 2048 env), so this is compute, not launch overhead.

Getting there from the first measurement (322x slower, OOM above 512 env) took three fixes:

1. **`gap` belongs to the hydroelastic pair only.** Scene-wide `gap = 0.01`, copied from the
   two-finger panda example, made every knuckle of the Wuji hand a contact candidate. Per-world
   contacts by category, stapler scene, hard hold released:

   | | before | after |
   |---|---|---|
   | hand against itself | 699.2 (73.7%) | 19.6 |
   | hand <-> leg | 131.9 | 0 |
   | arm <-> hand | 49.5 | 0 |
   | **object <-> table** | **35.9** | **31.5** |
   | ground <-> leg | 13.5 | 6.8 |
   | **total** | **949** | **61.4** |

   MuJoCo's whole per-world budget is 256. These pairs are legal collisions in MuJoCo too -- it
   simply runs margin 0, so nothing is generated until they actually penetrate.
2. **The pose sync had to be a warp kernel.** As torch indexing it made CUDA graph capture fail
   outright, silently costing the entire graph speedup on the native path.
3. **The SDF sparse grid is allocated per world.** At Newton's defaults (band +-0.1, margin 0.05,
   res 128) one world already carries a 91392-voxel grid and a 182784-entry iso buffer, and 2048
   worlds could not be allocated at any contact budget. The official recipe (res 64, band +-0.01,
   margin == gap) fits, and the bisect showed those values change the physics not at all.

`nconmax`/`njmax` are **per world**, not totals. Raising them to 4096/16384 for a one-env probe
is what caused the first 2.2 GiB OOM at 2048 env; 512/2048 is right-sized now that the gap fix
brought the per-world count to 61.

## Correction: native is 2.1x, not 42x

The 41.7x figure above is **retracted**. It was measured with two things wrong, both of them ours.

**`broad_phase="explicit"` does no AABB culling at all.** It narrow-phases every listed pair,
every call: 4513 pairs per world, 9.2 M at 2048 worlds, against 42-83 actual contacts. Profiled
at 2048 env, `collide()` was 235 ms of a 254 ms substep while `solver.step()` was 19 ms -- so
Newton's solver was already 2.1x faster than MuJoCo's whole step, and the collision pipeline was
paying for tests nothing needed. `"nxn"` culls by AABB first. The official example gets away with
`"explicit"` because its scene has about twenty shapes; ours has 232.

**65 non-hydroelastic mesh colliders were never convex-hulled** (48 in the Wuji hand, up to 25662
verts on one torso shape, 150 k verts per world). `example_robot_panda_hydro.py` hulls its
non-finger shapes and we skipped it. This is parity with the baseline, not a fidelity loss:
MuJoCo already convexifies every mesh geom for collision.

| 2048 env | collide | env.step | env-steps/s |
|---|---|---|---|
| explicit, full meshes, dt 0.0025 | 235.23 | 2308 | 887 |
| explicit + hulls | 4.31 | 480 | 4266 |
| sap + hulls | 4.98 | 176 | 11631 |
| nxn + hulls, dt 0.0025 | 4.75 | 173.7 | 11793 |
| **nxn + hulls, dt 0.005** | **4.79** | **105.6** | **19402** |
| MuJoCo contacts, dt 0.005 | -- | 50.7 | 40429 |

**Native is 2.08x slower than the MuJoCo-contact path.** At ~20 s/iter today that is ~42 s/iter.

**The halved step is no longer needed.** The NaN was never about stiffness or step size -- it was
the explicit broad phase. With `nxn`, dt 0.005 survived 600 steps twice out of two, object at
+0.001 mm penetration with 30-33 contacts, and per-world contacts fell to 48-79.

## The object/table pair does not need hydroelastic

It was the only consumer of it: the hand collides with the object through the rigid narrow phase,
against the object's real STL collider, and that is what the grasp actually depends on. The SDF
pressure field was buying resting accuracy on a contact surface the task does not care about.

Measured at 1024 env, 600-step no-reset runs, twice each:

| object/table contact | pair contacts | per-world total | resting penetration | env-steps/s |
|---|---|---|---|---|
| hydroelastic | 30-37 | 48-79 | 0.001 mm | 15142 |
| **rigid** | **4** | **19-25** | **0.110 mm** | **20011** |
| (MuJoCo-contact path) | -- | -- | -- | 22095 |

At 2048 env with the pair rigid: 63.1 ms/step, **32437 env-steps/s against the MuJoCo path's
40429 -- 1.25x**. Training fits at 2048 env in 24.5 GiB and runs at ~15 s/iter; with the pair
hydroelastic, 2048 could not be allocated at all.

Two things that do NOT reduce the contact count, measured so they are not tried again:

- **Lowering the table's SDF resolution.** Object 64 with table 32/16/8 all give 31-37 pair
  contacts. Resting penetration stays sub-micron even at table resolution 8, so the table can be
  made as coarse as one likes for the memory and build time -- but the contact patch tessellation
  follows the *object's* SDF, not the table's.
- **Lowering both resolutions.** 64 -> 32 gives 32 -> 31 contacts, no change. 64 -> 16 does halve
  them to 18, but the object then rests 2.731 mm into the table instead of 0.001 mm.

## Five defects that only appear at training scale

Every one of these passed a one-env probe and failed on launch at 2048:

1. The hydroelastic pair-completion loop was O(n^2) over *all* worlds' hydroelastic shapes: 4096
   shapes, 8.4 M pairs, nearly all of them cross-world. At one env it added 0 pairs.
2. `nconmax`/`njmax` auto-raised to 4096/16384 -- correct for the 1500 contacts a scene-wide gap
   produced, catastrophic as a *per-world* budget once the gap fix brought it to 61.
3. `contact_sensor_maxmatch` defaults to 64 in mujoco_warp and mjlab plumbs it through
   `cfg.sim`, which this env never reads because it builds its own solver. Newton emits more
   points per pair, it overflowed to 98, and it silently truncates the contact sensors the grasp
   rewards gate on.
4. `CollisionPipeline(max_triangle_pairs=...)` defaults to 1e6 and this scene reached 2.18 M.
   Everything past the cap is dropped -- missed contacts in the interaction being trained.
5. `_hyd` was moved inside the explicit-broad-phase branch while the body-pose sync still needed
   it (`UnboundLocalError`).
