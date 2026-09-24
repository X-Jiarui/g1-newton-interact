# G1 + Wuji gen-1 hand model with the `WJI_adapter` docking adapter

This directory (`assets/g1_wuji_WJI_adapter/`) is the **canonical copy** of the robot model that includes the Wuji→G1 docking
adapter. Use these files for training and data processing from 2026-09-24 on.

## Files

| File | What it is | Depends on |
|---|---|---|
| `g1_mocap_29dof_with_wuji_hands_WJI_adapter.xml` | Whole-body MJCF: G1 29 DoF + two Wuji gen-1 hands (40 DoF) + `WJI_adapter_left/right` bodies. Same joint/actuator names and order as the training model `g1_mocap_29dof_with_wuji_hands.xml`; only `{side}_palm_link` moved out by 25.5 mm and the adapter bodies were added. | `meshes/` = the G1 + Wuji mesh tree from `GMR_ROOT/assets/g1_wuji/meshes` (99 MB, **not in the repo**: copy or symlink it next to this file, plus `meshes/WJI_adapter.stl`) plus `meshes/WJI_adapter.stl` |
| `urdf/right_WJI_adapter.urdf` | Right gen-1 hand URDF with `WJI_adapter` as the root link and a fixed joint `WJI_adapter_to_palm` → `right_palm_link` (0, 0, 0.0255). Self-contained with `meshes/right/`. | `meshes/right/*.STL` (in the repo) |
| `meshes/WJI_adapter.stl` | Official Wuji `unitree-g1-docking-adapter.stl`, byte-identical (sha256 `100a790daa40f2568…`), mm units (`scale 0.001` in the model files). | — |
| `meshes/right/` | Gen-1 right-hand meshes from `wuji-description/hand/body/meshes/right` + the adapter STL. | — |

## Frames (the numbers that matter)

- `WJI_adapter` link / `xhand_mount_{side}` frame = the G1 wrist-yaw flange face centre,
  `(0.0415, ∓0.003, 0)` under `{side}_wrist_yaw_link`, Unitree's own hand-mount basis
  (mount +Z points out of the flange). This is Unitree's `right_hand_palm_joint` origin.
- Adapter mesh in that frame: `pos (0.003037, −0.000908, 0.004969) m`, `quat wxyz (0, 0, 0.7071, 0.7071)`.
- Palm: `pos (0, 0, 0.0255)` in the mount frame — the adapter's thickness from its G1 face (y=−4.969 mm)
  to its hand face (y=+20.531 mm).
- Attaching the URDF to a G1: joint origin `xyz="0.0415 -0.003 0" rpy="1.5708 0 1.5708"` from `right_wrist_yaw_link`.

Verified: nq 76 / nu 69 unchanged, joint and actuator order identical to the base model, both palms and
fingertips displaced exactly `[25.5, 0, 0]` mm in the wrist frame, adapter mass 30.4 g per side
(ρ = 1250 kg/m³), adapter geometry is visual-only (no collision).

**Open item:** the adapter is symmetric about its own X, so which side of the palm's long axis its
4.29 mm top-face offset falls on cannot be told from the mesh; the model assumes the pinky side.
Check on the physical part and flip `R[:,2]` in `../add_wji_adapter.py` if wrong (8.6 mm difference).

## How to use / regenerate

```bash
# MJCF: needs the 99 MB mesh tree next to it
cp officepc:/home/jiarui/jiarui/GMR/assets/g1_wuji/meshes ./meshes -r      # or regenerate: tools/model/build_g1_wuji.py
python -c "import mujoco; mujoco.MjModel.from_xml_path('g1_mocap_29dof_with_wuji_hands_WJI_adapter.xml')"

# switch training to it: env_cfgs.py line ~244 (GMR_ROOT/assets/g1_wuji/<this file>)
# regenerate from the base model: ../add_wji_adapter.py ; URDF: ../make_wji_urdf.py
```

Officepc originals: `~/jiarui/GMR/assets/g1_wuji/g1_mocap_29dof_with_wuji_hands_WJI_adapter.xml`,
`~/projects/WJI_adapter/`. The training model **without** the adapter (used by every checkpoint up to
POST_CONT1000) is `g1_mocap_29dof_with_wuji_hands.xml` in the same GMR directory; the real-robot FK in
`deploy_real_20260924/hoi_real.py` deliberately uses that one to match those checkpoints.
