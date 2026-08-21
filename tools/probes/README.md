# One-off probes

Each of these settled one question. They are kept rather than deleted because the answers are the
evidence behind decisions in [../../docs/03-defects.md](../../docs/03-defects.md) and
[../../docs/04-architecture.md](../../docs/04-architecture.md), and because several of them are the
cheapest way to re-check an assumption if Newton or mjlab changes underneath.

| probe | question it answered | answer |
|---|---|---|
| `probe_newton_data.py` | what state surface does `SolverMuJoCo` expose, and is it zero-copy? | `mjw_data` with mjlab's exact layout; `wp.to_torch` shares memory |
| `probe_ctrl_path.py` | can ctrl and qpos be written directly into `mjw_data`? | qpos yes with `update_data_interval=0`; ctrl no — overwritten every step (0.777 → 0.0) |
| `probe_target_map.py` | which actuator does each control slot drive? | `control.mujoco.ctrl`, 138/138 identity with actuator order; `joint_target_q` reaches 0 of 138 |
| `probe_ctrl.py` | is `Sonic53Action`'s torque law reaching physics? | no — the `xml_motor_unused_*` actuators hold ctrl at exactly 0.0 for the whole rollout |
| `probe_install_pd.py` | does `install_astra_body_pd` change the PD after `set_astra_body_dynamics`? | not on the compiled model (0 entries changed); only the action term's dead buffers |
| `probe_table.py` | did the table survive conversion as a mocap body? | yes, but at mocap index 2 instead of 0 — Newton makes every world-attached fixed body a mocap body |
| `chain1_geoms.py` | did Newton drop any collidable geom? | no — 81 vs 81, fingertips 40 vs 40; the 145 dropped are all `contype=conaffinity=0` |
| `chain2_tree.py` | do the two models agree on kinematic tree topology? | yes — identical parents, dof depths and `nM` |
| `chain3_nm.py` | is mjlab's live model the same structure as the exported XML? | yes; the `M` size difference is in the **warp** model, not the compiled one |
| `chain4_nc.py` | what makes `nC` differ, and does rebuilding fix it? | `body_simple` on the two free bodies; rebuilt model matches stock mujoco_warp to 1.49e-08 |
| `dump_action_mapping.py` | how do the 69 action slots map to joints and actuators? | written to `docs/data/action_mapping.json` |

Three of these — `chain1`, `chain2`, `chain3` — returned "no difference". They are as load-bearing as
the ones that found something: each closed off a hypothesis that would otherwise still be open, and
`chain1` in particular ruled out the most attractive wrong explanation for why a hand reaches an
object and fails to grasp it.
