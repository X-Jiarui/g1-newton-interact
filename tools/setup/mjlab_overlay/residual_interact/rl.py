"""Runner hooks for SONIC-backed residual interaction PPO."""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.utils import check_nan

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.apple_eat import mdp as apple_mdp
from mjlab.tasks.residual_interact import mdp
from mjlab.tasks.residual_interact.residual_actor import (
  _make_residual_gain,
  _make_residual_mask,
)

_ANKLE_WRIST_TRACKING_LINK_NAMES = (
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)
_ANKLE_WRIST_TRACKING_WHOLE_KEYS = (
  "Metric/ankle_wrist_tracking/link_dist_mean",
  "Metric/ankle_wrist_tracking/link_dist_max",
  "Metric/ankle_wrist_tracking/link_dist_min",
  "Metric/ankle_wrist_tracking/link_dist_std",
)
_ANKLE_WRIST_TRACKING_LINK_KEYS = tuple(
  key
  for name in _ANKLE_WRIST_TRACKING_LINK_NAMES
  for key in (
    f"Metric/ankle_wrist_tracking/{name}_dist",
    f"ResidualReward/ankle_wrist_tracking/{name}",
    f"ResidualReward/ankle_wrist_tracking/{name}_base",
    f"ResidualReward/ankle_wrist_tracking/{name}_close_bonus",
  )
)
_ANKLE_WRIST_TRACKING_DETAIL_KEYS = (
  *_ANKLE_WRIST_TRACKING_WHOLE_KEYS,
  *_ANKLE_WRIST_TRACKING_LINK_KEYS,
)


def _normalize_local_asset_path(path: str) -> str:
  """Map Docker asset paths saved in remote checkpoints to local assets."""

  mappings = (
    (
      (
        "/opt/mjlab_assets/GR00T-WholeBodyControl/",
        "/workspace/astra-assets/GR00T-WholeBodyControl/",
      ),
      Path(
        os.environ.get(
          "GROOT_ROOT",
          "/home/jiarui/projects/GR00T-WholeBodyControl",
        )
      ),
    ),
    (
      (
        "/opt/mjlab_assets/Humanoid-GPT/",
        "/workspace/astra-assets/Humanoid-GPT/",
      ),
      Path(
        os.environ.get(
          "ASTRA_ROOT",
          "/home/jiarui/projects/Humanoid-GPT",
        )
      ),
    ),
    (
      (
        "/opt/mjlab_assets/GMR/",
        "/workspace/astra-assets/GMR/",
      ),
      Path(os.environ.get("GMR_ROOT", "/home/jiarui/jiarui/GMR")),
    ),
  )
  for prefixes, local_root in mappings:
    for prefix in prefixes:
      if not path.startswith(prefix):
        continue
      local_path = local_root / path.removeprefix(prefix)
      if local_path.exists():
        return str(local_path)
  return path


_COMPACT_WANDB_EP_EXTRA_KEYS = frozenset(
  {
    "Reward/total",
    # Online step/rollout summaries: quick trend checks.
    "Metric/lower_body_link_dist_mean",
    "Metric/lower_wrist_link_dist_mean",
    "Metric/ankle_link_dist_mean",
    "Metric/ankle_wrist_link_dist_mean",
    "Metric/body_link_dist_mean",
    "Metric/left_wrist_link_dist_mean",
    "Metric/right_wrist_link_dist_mean",
    "Metric/left_wrist_x_abs_err",
    "Metric/left_wrist_y_abs_err",
    "Metric/left_wrist_z_abs_err",
    "Metric/right_wrist_x_abs_err",
    "Metric/right_wrist_y_abs_err",
    "Metric/right_wrist_z_abs_err",
    "Metric/wrist_target_far_left_dist",
    "Metric/wrist_target_far_right_dist",
    "Metric/wrist_target_far_dist",
    "Metric/wrist_target_far_threshold",
    "Metric/wrist_target_far_active",
    "Metric/wrist_target_far_candidate",
    "Metric/left_wrist_z_tracking_abs_err",
    "Metric/right_wrist_z_tracking_abs_err",
    "Metric/hand_to_obj_dist",
    "Metric/omnigrasp_live_min_tip_dist",
    "Metric/omnigrasp_live_contact_gate",
    "Metric/omnigrasp_physical_contact",
    "Metric/hand_action_tracking_gate",
    "Metric/hand_to_obj_under_030_frac",
    "Metric/hand_to_obj_under_015_frac",
    "Metric/hand_to_obj_under_005_frac",
    "Stage/stable_reach_030",
    "Stage/stable_reach_015",
    "Stage/stable_reach_005",
    "Metric/ep_len",
    "Metric/tracking_frame",
    # The approach phase, which is where the hand was measured to stall 5.8 cm from the reference
    # grasp pose and never close. Without these the two regimes are invisible: `far_frac` says how
    # much of the batch is still being paid for CLOSING rather than for being close, `progress` is
    # what that pays, `rot` is the orientation proxy, and `dist` is the number the whole design is
    # about. They were logged all along -- this list is why they never appeared.
    # Merged from the office box, which had grown its own copy of this allowlist. A key
    # present in one box's list and absent from another's is why the same run produced
    # different tensorboard panels depending on where it ran.
    "Metric/before_cf",
    "Metric/staged_lower_dist",
    "Metric/staged_object_dist",
    "Metric/staged_object_touching",
    "Metric/staged_upper_dist",
    "Metric/tip_cf_miss_active",
    "Metric/tip_cf_miss_dist",
    # Arrival time, the "on time" half of the approach question.
    "Metric/staged_tip_cf_arrive_frac",
    "Metric/staged_tip_cf_arrive_frame",
    "Metric/staged_tip_cf_dist",
    "Metric/staged_tip_cf_far_frac",
    "Metric/staged_tip_cf_progress",
    "Metric/staged_tip_cf_rot",
    "Metric/staged_tip_cf_used_tips",
    "Metric/cf_frame",
    "ResidualMetric/astra_body_delta_norm",
    "ResidualMetric/astra_body_delta_ratio",
    "ResidualMetric/astra_body_delta_joint_rms",
    "ResidualMetric/decoder_body_delta_norm",
    "ResidualMetric/decoder_body_delta_ratio",
    "ResidualMetric/hand_control_gate",
    "ResidualMetric/hand_mean_delta_norm",
    "ResidualMetric/hand_sample_delta_pre_clip_norm",
    "ResidualMetric/hand_sample_delta_post_clip_norm",
    "ResidualMetric/hand_sample_clip_frac",
    "ResidualMetric/body_sample_delta_pre_clip_norm",
    "ResidualMetric/body_sample_delta_post_clip_norm",
    "ResidualMetric/body_sample_clip_frac",
    "ResidualMetric/hand_action_std_mean",
    "ResidualMetric/hand_close_left",
    "ResidualMetric/hand_close_right",
    "ResidualMetric/hand_close_mean",
    "ResidualMetric/hand_primitive_delta_norm",
    "ResidualMetric/residual_clip_frac",
    "ResidualMetric/token_residual_norm",
    "ResidualReward/lower_tracking",
    "ResidualReward/lower_tracking_base",
    "ResidualReward/lower_tracking_close_bonus",
    "ResidualReward/lower_tracking_miss_penalty",
    "ResidualReward/ankle_wrist_tracking",
    "ResidualReward/ankle_wrist_tracking_base",
    "ResidualReward/ankle_wrist_tracking_close_bonus",
    "ResidualReward/ankle_wrist_tracking_miss_penalty",
    "ResidualReward/left_wrist_tracking",
    "ResidualReward/left_wrist_tracking_base",
    "ResidualReward/left_wrist_tracking_close_bonus",
    "ResidualReward/left_wrist_tracking_miss_penalty",
    "ResidualReward/right_wrist_tracking",
    "ResidualReward/right_wrist_tracking_base",
    "ResidualReward/right_wrist_tracking_close_bonus",
    "ResidualReward/right_wrist_tracking_miss_penalty",
    "ResidualReward/left_wrist_z_tracking",
    "ResidualReward/right_wrist_z_tracking",
    "Stage/physical_contact",
    "Stage/stable_not_fallen",
    "PhaseA/live_contact_006",
    "PhaseA/lift_duration_s",
    "PhaseA/lift_success",
    "PhaseA/ttr_at_012",
    "PhaseA/object_mpjpe_mm",
    "PhaseA/sequence_success",
    # Episode summaries: these are the primary compact dashboard metrics.
    "Episode_Reward/tracking",
    "Episode_Reward/left_wrist_tracking",
    "Episode_Reward/right_wrist_tracking",
    "Episode_Reward/left_wrist_z_tracking",
    "Episode_Reward/right_wrist_z_tracking",
    "Episode_Metrics/reward_total",
    "Episode_Metrics/body_link_dist_mean",
    "Episode_Metrics/lower_body_link_dist_mean",
    "Episode_Metrics/lower_wrist_link_dist_mean",
    "Episode_Metrics/ankle_link_dist_mean",
    "Episode_Metrics/ankle_wrist_link_dist_mean",
    "Episode_Metrics/upper_body_link_dist_mean",
    "Episode_Metrics/left_wrist_link_dist_mean",
    "Episode_Metrics/right_wrist_link_dist_mean",
    "Episode_Metrics/left_wrist_x_abs_err",
    "Episode_Metrics/left_wrist_y_abs_err",
    "Episode_Metrics/left_wrist_z_abs_err",
    "Episode_Metrics/right_wrist_x_abs_err",
    "Episode_Metrics/right_wrist_y_abs_err",
    "Episode_Metrics/right_wrist_z_abs_err",
    "Episode_Metrics/body_xyz_err_mse",
    "Episode_Metrics/lower_body_xyz_err_mse",
    "Episode_Metrics/upper_body_xyz_err_mse",
    "Episode_Metrics/hand_to_obj_dist",
    "Episode_Metrics/hand_to_obj_under_030_frac",
    "Episode_Metrics/hand_to_obj_under_015_frac",
    "Episode_Metrics/hand_to_obj_under_005_frac",
    "Episode_Metrics/reach_030",
    "Episode_Metrics/reach_015",
    "Episode_Metrics/reach_005",
    "Episode_Metrics/stable_reach_030",
    "Episode_Metrics/stable_reach_015",
    "Episode_Metrics/stable_reach_005",
    "Episode_Metrics/near_contact",
    "Episode_Metrics/physical_contact",
    "Episode_Metrics/live_contact_006_frac",
    "Episode_Metrics/lift_success",
    "Episode_Metrics/ttr_at_012",
    "Episode_Metrics/object_mpjpe_mm",
    "Episode_Metrics/sequence_success",
    "Episode_Metrics/stable_not_fallen",
    "Episode_Metrics/ep_len",
    "Episode_Metrics/tracking_frame",
    "Episode_Metrics/obj_drift",
    "Episode_Metrics/object_motion_frac",
    "Episode_Metrics/hand_action_mse",
    "Episode_Metrics/hand_err_mse",
    "Episode_Metrics/residual_base_ratio",
    "Episode_Metrics/residual_clip_frac",
    "Episode_Metrics/hand_control_gate",
    "Episode_Metrics/hand_mean_delta_norm",
    "Episode_Metrics/hand_sample_delta_pre_clip_norm",
    "Episode_Metrics/hand_sample_delta_post_clip_norm",
    "Episode_Metrics/hand_sample_clip_frac",
    "Episode_Metrics/body_sample_delta_pre_clip_norm",
    "Episode_Metrics/body_sample_delta_post_clip_norm",
    "Episode_Metrics/body_sample_clip_frac",
    "Episode_Metrics/hand_action_std_mean",
    "Episode_Metrics/hand_close_left",
    "Episode_Metrics/hand_close_right",
    "Episode_Metrics/hand_close_mean",
    "Episode_Metrics/hand_primitive_delta_norm",
    "Episode_Metrics/hand_residual_norm",
    "Episode_Metrics/hand_residual_ratio",
    "Episode_Metrics/hand_final_delta_norm",
    "Episode_Metrics/token_residual_norm",
    "Episode_Metrics/token_residual_clip_frac",
    "Episode_Metrics/decoder_body_delta_norm",
    "Episode_Metrics/decoder_body_delta_ratio",
    "Episode_Metrics/decoder_body_delta_joint_rms",
    "Episode_Metrics/arm_residual_ratio",
    "Episode_Metrics/leg_residual_ratio",
    "Episode_Metrics/placement_error",
  }
  | set(_ANKLE_WRIST_TRACKING_DETAIL_KEYS)
)

_COMPACT_WANDB_EP_EXTRA_PREFIXES = (
  "Episode_Reward/",
  "Episode_Termination/",
  "ResidualReward/",
  "OmniGraspReward/",
)

_COMPACT_WANDB_SNAPSHOT_KEYS = (
  "Curriculum/horizon_ref_frames",
  "Curriculum/horizon_last_metric",
  "Curriculum/horizon_last_passed",
  "Curriculum/horizon_stage_ready",
  "Curriculum/horizon_stage_age_iterations",
  "Curriculum/horizon_success_streak",
  "Curriculum/tracking_region_id",
  "Curriculum/tracking_region_last_metric",
  "Curriculum/tracking_region_horizon_ready",
)

_OBJECT_SWEEP_WANDB_EP_EXTRA_KEYS = frozenset(
  {
    "Reward/total",
    "Metric/body_link_dist_mean",
    "Metric/lower_body_link_dist_mean",
    "Metric/lower_wrist_link_dist_mean",
    "Metric/left_wrist_link_dist_mean",
    "Metric/right_wrist_link_dist_mean",
    "Metric/hand_to_obj_dist",
    "Metric/hand_to_obj_under_030_frac",
    "Metric/hand_to_obj_under_015_frac",
    "Metric/hand_to_obj_under_005_frac",
    "Stage/stable_reach_030",
    "Stage/stable_reach_015",
    "Stage/stable_reach_005",
    "Metric/obj_drift",
    "Metric/obj_speed",
    "Metric/ep_len",
    "Metric/tracking_frame",
    "Metric/wrist_target_far_dist",
    "Metric/wrist_target_far_candidate",
    "Metric/omnigrasp_pregrasp_tip_err",
    "Metric/omnigrasp_object_pos_err",
    "Metric/omnigrasp_object_rot_err",
    "Metric/omnigrasp_object_vel_err",
    "Metric/omnigrasp_live_min_tip_dist",
    "Metric/omnigrasp_live_contact_gate",
    "Metric/omnigrasp_physical_contact",
    "Metric/omnigrasp_force_close",
    "Metric/object_traj_pos_err",
    "Metric/object_traj_rot_err",
    "Metric/object_traj_vel_err",
    "Metric/object_traj_min_tip_dist",
    "Metric/object_height_above_ref0",
    "Metric/object_hold_min_tip_dist",
    "Metric/hard_lift_duration",
    "Metric/hard_lift_contact_count",
    "Stage/physical_contact",
    "Stage/force_close",
    "Stage/object_moving",
    "Stage/stable_not_fallen",
    "PhaseA/live_contact_006",
    "PhaseA/lift_duration_s",
    "PhaseA/lift_success",
    "PhaseA/ttr_at_012",
    "PhaseA/object_mpjpe_mm",
    "PhaseA/sequence_success",
    "ResidualMetric/decoder_body_delta_norm",
    "ResidualMetric/decoder_body_delta_ratio",
    "ResidualMetric/decoder_body_delta_joint_rms",
    "ResidualMetric/astra_body_delta_norm",
    "ResidualMetric/astra_body_delta_ratio",
    "ResidualMetric/astra_body_delta_joint_rms",
    "ResidualMetric/token_residual_norm",
    "ResidualMetric/token_residual_clip_frac",
    "ResidualMetric/residual_clip_frac",
    "ResidualMetric/hand_control_gate",
    "ResidualMetric/hand_mean_delta_norm",
    "ResidualMetric/hand_sample_delta_pre_clip_norm",
    "ResidualMetric/hand_sample_delta_post_clip_norm",
    "ResidualMetric/hand_sample_clip_frac",
    "ResidualMetric/body_sample_delta_pre_clip_norm",
    "ResidualMetric/body_sample_delta_post_clip_norm",
    "ResidualMetric/body_sample_clip_frac",
    "ResidualMetric/hand_action_std_mean",
    "ResidualMetric/hand_close_left",
    "ResidualMetric/hand_close_right",
    "ResidualMetric/hand_close_mean",
    "ResidualMetric/hand_primitive_delta_norm",
    "ResidualMetric/hand_final_delta_norm",
    "ResidualMetric/hand_residual_ratio",
    "ResidualReward/tracking",
    "ResidualReward/left_wrist_tracking",
    "ResidualReward/right_wrist_tracking",
    "ResidualReward/raw_tip_object_tracking",
    "ResidualReward/raw_tip_radial_tracking",
    "ResidualReward/omnigrasp_style",
    "ResidualReward/grasp",
    "ResidualReward/surface_contact",
    "ResidualReward/multi_tip_surface",
    "ResidualReward/contact_duration",
    "ResidualReward/object_drift_limit",
    "ResidualReward/object_trajectory_tracking",
    "ResidualReward/object_lift_hold",
    "ResidualReward/object_hard_lift",
    "ResidualReward/hand_ratio_limit",
    "ResidualReward/stability",
    "OmniGraspReward/body_tracking",
    "OmniGraspReward/pregrasp",
    "OmniGraspReward/pregrasp_progress",
    "OmniGraspReward/object_tracking_raw",
    "OmniGraspReward/object_tracking_gated",
    "OmniGraspReward/contact_bonus",
    "ObjectReward/trajectory_score",
    "ObjectReward/trajectory_gate",
    "ObjectReward/trajectory_contact_gate",
    "ObjectReward/trajectory_soft_contact",
    "ObjectReward/lifted_score",
    "ObjectReward/near_score",
    "ObjectReward/hold_contact",
    "ObjectReward/hold_grasp_contact",
    "ObjectReward/hold_force_close",
    "ObjectReward/contact_duration_score",
    "ObjectReward/drop_penalty_flag",
    "ObjectReward/hard_lift_height_score",
    "ObjectReward/hard_lift_contact_gate",
    "ObjectReward/hard_lift_strict_contact",
    "ObjectReward/hard_lift_flag",
    "ObjectReward/hard_lift_duration_score",
    "Episode_Reward/tracking",
    "Episode_Reward/left_wrist_tracking",
    "Episode_Reward/right_wrist_tracking",
    "Episode_Reward/raw_tip_object_tracking",
    "Episode_Reward/raw_tip_radial_tracking",
    "Episode_Reward/omnigrasp_style",
    "Episode_Reward/grasp",
    "Episode_Reward/surface_contact",
    "Episode_Reward/multi_tip_surface",
    "Episode_Reward/contact_duration",
    "Episode_Reward/object_drift_limit",
    "Episode_Reward/object_trajectory_tracking",
    "Episode_Reward/object_lift_hold",
    "Episode_Reward/object_hard_lift",
    "Episode_Reward/hand_ratio_limit",
    "Episode_Reward/stability",
    "Episode_Metrics/reward_total",
    "Episode_Metrics/body_link_dist_mean",
    "Episode_Metrics/lower_body_link_dist_mean",
    "Episode_Metrics/lower_wrist_link_dist_mean",
    "Episode_Metrics/left_wrist_link_dist_mean",
    "Episode_Metrics/right_wrist_link_dist_mean",
    "Episode_Metrics/hand_to_obj_dist",
    "Episode_Metrics/hand_to_obj_under_030_frac",
    "Episode_Metrics/hand_to_obj_under_015_frac",
    "Episode_Metrics/hand_to_obj_under_005_frac",
    "Episode_Metrics/reach_030",
    "Episode_Metrics/reach_015",
    "Episode_Metrics/reach_005",
    "Episode_Metrics/stable_reach_030",
    "Episode_Metrics/stable_reach_015",
    "Episode_Metrics/stable_reach_005",
    "Episode_Metrics/near_contact",
    "Episode_Metrics/physical_contact",
    "Episode_Metrics/stable_not_fallen",
    "Episode_Metrics/ep_len",
    "Episode_Metrics/tracking_frame",
    "Episode_Metrics/obj_drift",
    "Episode_Metrics/object_motion_frac",
    "Episode_Metrics/hand_action_mse",
    "Episode_Metrics/hand_err_mse",
    "Episode_Metrics/residual_clip_frac",
    "Episode_Metrics/hand_control_gate",
    "Episode_Metrics/hand_mean_delta_norm",
    "Episode_Metrics/hand_sample_delta_pre_clip_norm",
    "Episode_Metrics/hand_sample_delta_post_clip_norm",
    "Episode_Metrics/hand_sample_clip_frac",
    "Episode_Metrics/body_sample_delta_pre_clip_norm",
    "Episode_Metrics/body_sample_delta_post_clip_norm",
    "Episode_Metrics/body_sample_clip_frac",
    "Episode_Metrics/hand_action_std_mean",
    "Episode_Metrics/hand_close_left",
    "Episode_Metrics/hand_close_right",
    "Episode_Metrics/hand_close_mean",
    "Episode_Metrics/hand_primitive_delta_norm",
    "Episode_Metrics/hand_residual_norm",
    "Episode_Metrics/hand_residual_ratio",
    "Episode_Metrics/hand_final_delta_norm",
    "Episode_Metrics/token_residual_norm",
    "Episode_Metrics/token_residual_clip_frac",
    "Episode_Metrics/decoder_body_delta_norm",
    "Episode_Metrics/decoder_body_delta_ratio",
    "Episode_Metrics/decoder_body_delta_joint_rms",
  }
)

_OBJECT_SWEEP_WANDB_EP_EXTRA_PREFIXES = ("Episode_Termination/",)

_OBJECT_SWEEP_WANDB_CORE_KEYS = frozenset(
  {
    "Reward/total",
    "Metric/body_link_dist_mean",
    "Metric/lower_wrist_link_dist_mean",
    "Metric/left_wrist_link_dist_mean",
    "Metric/right_wrist_link_dist_mean",
    "Metric/hand_to_obj_dist",
    "Metric/hand_to_obj_under_030_frac",
    "Metric/hand_to_obj_under_015_frac",
    "Metric/hand_to_obj_under_005_frac",
    "Stage/stable_reach_030",
    "Stage/stable_reach_015",
    "Stage/stable_reach_005",
    "Metric/obj_drift",
    "Metric/object_traj_pos_err",
    "Metric/object_traj_min_tip_dist",
    "Metric/object_height_above_ref0",
    "Metric/object_hold_min_tip_dist",
    "Metric/omnigrasp_live_min_tip_dist",
    "Metric/omnigrasp_physical_contact",
    "Metric/omnigrasp_force_close",
    "Metric/wrist_target_far_candidate",
    "Metric/ep_len",
    "Metric/tracking_frame",
    "Stage/physical_contact",
    "Stage/force_close",
    "Stage/object_moving",
    "Stage/stable_not_fallen",
    "ResidualMetric/decoder_body_delta_norm",
    "ResidualMetric/decoder_body_delta_ratio",
    "ResidualMetric/token_residual_clip_frac",
    "ResidualMetric/residual_clip_frac",
    "ResidualMetric/hand_control_gate",
    "ResidualMetric/hand_mean_delta_norm",
    "ResidualMetric/hand_sample_delta_pre_clip_norm",
    "ResidualMetric/hand_sample_delta_post_clip_norm",
    "ResidualMetric/hand_sample_clip_frac",
    "ResidualMetric/body_sample_delta_pre_clip_norm",
    "ResidualMetric/body_sample_delta_post_clip_norm",
    "ResidualMetric/body_sample_clip_frac",
    "ResidualMetric/hand_action_std_mean",
    "ResidualMetric/hand_close_left",
    "ResidualMetric/hand_close_right",
    "ResidualMetric/hand_close_mean",
    "ResidualMetric/hand_primitive_delta_norm",
    "ResidualMetric/hand_final_delta_norm",
    "ResidualMetric/hand_residual_ratio",
    "ResidualReward/tracking",
    "ResidualReward/left_wrist_tracking",
    "ResidualReward/right_wrist_tracking",
    "ResidualReward/omnigrasp_style",
    "ResidualReward/grasp",
    "ResidualReward/surface_contact",
    "ResidualReward/contact_duration",
    "ResidualReward/object_trajectory_tracking",
    "ResidualReward/object_lift_hold",
    "ResidualReward/stability",
    "ObjectReward/trajectory_gate",
    "ObjectReward/trajectory_score",
    "ObjectReward/lifted_score",
    "ObjectReward/hold_contact",
    "Episode_Metrics/reward_total",
    "Episode_Metrics/body_link_dist_mean",
    "Episode_Metrics/left_wrist_link_dist_mean",
    "Episode_Metrics/right_wrist_link_dist_mean",
    "Episode_Metrics/hand_to_obj_dist",
    "Episode_Metrics/reach_030",
    "Episode_Metrics/reach_015",
    "Episode_Metrics/reach_005",
    "Episode_Metrics/stable_reach_030",
    "Episode_Metrics/stable_reach_015",
    "Episode_Metrics/stable_reach_005",
    "Episode_Metrics/physical_contact",
    "Episode_Metrics/stable_not_fallen",
    "Episode_Metrics/ep_len",
    "Episode_Metrics/obj_drift",
    "Episode_Metrics/hand_mean_delta_norm",
    "Episode_Metrics/hand_sample_delta_pre_clip_norm",
    "Episode_Metrics/hand_sample_delta_post_clip_norm",
    "Episode_Metrics/hand_sample_clip_frac",
    "Episode_Metrics/body_sample_delta_pre_clip_norm",
    "Episode_Metrics/body_sample_delta_post_clip_norm",
    "Episode_Metrics/body_sample_clip_frac",
    "Episode_Metrics/hand_action_std_mean",
    "Episode_Metrics/hand_close_left",
    "Episode_Metrics/hand_close_right",
    "Episode_Metrics/hand_close_mean",
    "Episode_Metrics/hand_primitive_delta_norm",
    "Episode_Metrics/hand_residual_ratio",
    "Episode_Metrics/decoder_body_delta_ratio",
  }
)

# Keep the new object/hand sweep project readable: these are the only extra
# task metrics sent to W&B in object_sweep mode, aside from PPO's built-in stats.
_OBJECT_SWEEP_WANDB_CORE_KEYS = frozenset(
  {
    "Reward/total",
    "Metric/hand_to_obj_dist",
    "Metric/object_traj_pos_err",
    "Metric/object_height_above_ref0",
    "Metric/hard_lift_duration",
    "Metric/hard_lift_contact_count",
    "Metric/grab_obj_pos_err",
    "Metric/grab_obj_rot_err",
    "Metric/grab_obj_vel_err",
    "Metric/grab_min_tip_dist",
    "Metric/grab_contact_count",
    "Metric/grab_object_speed",
    "Metric/omnigrasp_raw_contact_frame",
    "Metric/omnigrasp_pregrasp_tip_err",
    "Metric/omnigrasp_pregrasp_hand_q_rmse",
    "Metric/omnigrasp_object_pos_err",
    "Metric/omnigrasp_object_rot_err",
    "Metric/omnigrasp_object_vel_err",
    "Metric/omnigrasp_live_min_tip_dist",
    "Metric/omnigrasp_live_contact_gate",
    "Metric/omnigrasp_geometric_contact",
    "Metric/omnigrasp_physical_contact",
    "Metric/omnigrasp_force_close",
    "Metric/omnigrasp_post_contact_phase",
    "Stage/physical_contact",
    "Stage/object_moving",
    "Stage/stable_not_fallen",
    "ResidualMetric/decoder_body_delta_ratio",
    "ResidualMetric/residual_clip_frac",
    "ResidualMetric/hand_control_gate",
    "ResidualMetric/hand_mean_delta_norm",
    "ResidualMetric/hand_sample_delta_pre_clip_norm",
    "ResidualMetric/hand_sample_delta_post_clip_norm",
    "ResidualMetric/hand_sample_clip_frac",
    "ResidualMetric/hand_action_std_mean",
    "ResidualMetric/hand_close_left",
    "ResidualMetric/hand_close_right",
    "ResidualMetric/hand_close_mean",
    "ResidualMetric/hand_primitive_delta_norm",
    "ResidualMetric/hand_final_delta_norm",
    "ResidualMetric/hand_residual_ratio",
    "ResidualReward/tracking",
    "ResidualReward/left_wrist_tracking",
    "ResidualReward/right_wrist_tracking",
    "ResidualReward/omnigrasp_style",
    "ResidualReward/grasp",
    "ResidualReward/contact_duration",
    "ResidualReward/object_trajectory_tracking",
    "ResidualReward/object_lift_hold",
    "ResidualReward/object_hard_lift",
    "ResidualReward/object_omnigrasp_grab",
    "ResidualReward/stability",
    "OmniGraspReward/body_tracking",
    "OmniGraspReward/pregrasp",
    "OmniGraspReward/pregrasp_progress",
    "OmniGraspReward/object_tracking_raw",
    "OmniGraspReward/object_tracking_gated",
    "OmniGraspReward/contact_bonus",
    "OmniGraspReward/grab_object_tracking_raw",
    "OmniGraspReward/grab_object_tracking_gated",
    "OmniGraspReward/grab_contact_gate",
    "OmniGraspReward/grab_object_track_gate",
    "OmniGraspReward/grab_lifted",
    "OmniGraspReward/grab_object_moving",
    "OmniGraspReward/grab_duration_score",
    "ObjectReward/trajectory_gate",
    "ObjectReward/lifted_score",
    "ObjectReward/hold_contact",
    "ObjectReward/hard_lift_flag",
    "ObjectReward/hard_lift_duration_score",
    "ObjectReward/hard_lift_contact_gate",
    "Episode_Metrics/reward_total",
    "Episode_Metrics/body_link_dist_mean",
    "Episode_Metrics/left_wrist_link_dist_mean",
    "Episode_Metrics/right_wrist_link_dist_mean",
    "Episode_Metrics/hand_to_obj_dist",
    "Episode_Metrics/reach_030",
    "Episode_Metrics/reach_015",
    "Episode_Metrics/reach_005",
    "Episode_Metrics/physical_contact",
    "Episode_Metrics/stable_not_fallen",
    "Episode_Metrics/ep_len",
    "Episode_Metrics/obj_drift",
    "Episode_Metrics/hand_close_left",
    "Episode_Metrics/hand_close_right",
    "Episode_Metrics/hand_close_mean",
    "Episode_Metrics/hand_primitive_delta_norm",
    "Episode_Metrics/hand_residual_ratio",
    "Episode_Metrics/decoder_body_delta_ratio",
    "Episode_Reward/omnigrasp_style",
    "Episode_Reward/object_omnigrasp_grab",
  }
)


def _strip_clip_suffix(key: str) -> str:
  """`PhaseA/lift_success/clip1` filters as `PhaseA/lift_success`.

  The allowlists below are written against base metric names. Without this, a per-clip variant is
  dropped in compact mode and only survives when _wandb_metric_mode is "full" -- which is exactly
  the silent-hole the per-clip logging exists to close.
  """
  base, sep, tail = key.rpartition("/clip")
  return base if sep and tail.isdigit() else key


def _keep_compact_wandb_extra(key: str) -> bool:
  key = _strip_clip_suffix(key)
  return key in _COMPACT_WANDB_EP_EXTRA_KEYS or key.startswith(
    _COMPACT_WANDB_EP_EXTRA_PREFIXES
  )


def _keep_object_sweep_wandb_extra(key: str) -> bool:
  key = _strip_clip_suffix(key)
  return key in _OBJECT_SWEEP_WANDB_CORE_KEYS or key.startswith(
    _OBJECT_SWEEP_WANDB_EP_EXTRA_PREFIXES
  )


def _parse_groups(groups) -> tuple[str, ...]:
  if isinstance(groups, str):
    return tuple(g.strip() for g in groups.split(",") if g.strip())
  return tuple(groups)


def _diag_gaussian_log_prob_sum(
  mean: torch.Tensor,
  std: torch.Tensor,
  actions: torch.Tensor,
  start: int,
  end: int,
) -> torch.Tensor:
  var = std[:, start:end].pow(2)
  log_std = std[:, start:end].log()
  diff = actions[:, start:end] - mean[:, start:end]
  return (-0.5 * (diff.pow(2) / var + 2.0 * log_std + math.log(2.0 * math.pi))).sum(
    dim=-1
  )


def _diag_gaussian_entropy_sum(
  std: torch.Tensor,
  start: int,
  end: int,
) -> torch.Tensor:
  return (0.5 + 0.5 * math.log(2.0 * math.pi) + std[:, start:end].log()).sum(dim=-1)


class ResidualInteractOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(
    self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"
  ):
    self._base_tracker_kind = str(
      train_cfg.pop("base_tracker_kind", "checkpoint")
    ).strip()
    self._tracker_ckpt = train_cfg.pop("tracker_ckpt", None)
    self._init_from = train_cfg.pop("init_from", "ema")
    self._sonic_encoder_onnx = _normalize_local_asset_path(
      str(train_cfg.pop("sonic_encoder_onnx", ""))
    )
    self._sonic_decoder_onnx = _normalize_local_asset_path(
      str(train_cfg.pop("sonic_decoder_onnx", ""))
    )
    self._astra_onnx_path = _normalize_local_asset_path(
      str(train_cfg.pop("astra_onnx_path", ""))
    )
    self._base_hand_mode = str(train_cfg.pop("base_hand_mode", "zero")).strip()
    self._residual_gain = float(train_cfg.pop("residual_gain", 1.0))
    raw_body_gain = train_cfg.pop("body_residual_gain", None)
    raw_hand_gain = train_cfg.pop("hand_residual_gain", None)
    self._body_residual_gain = None if raw_body_gain is None else float(raw_body_gain)
    self._hand_residual_gain = None if raw_hand_gain is None else float(raw_hand_gain)
    self._residual_action_clip = float(train_cfg.pop("residual_action_clip", 0.5))
    self._final_action_clip = train_cfg.pop("final_action_clip", None)
    self._residual_mask = train_cfg.pop("residual_mask", "all")
    self._residual_feature_groups = _parse_groups(
      train_cfg.pop("residual_feature_groups", ())
    )
    self._critic_feature_groups = _parse_groups(
      train_cfg.pop("critic_feature_groups", ())
    )
    self._ref_preview_steps = tuple(
      int(v) for v in train_cfg.pop("ref_preview_steps", (1, 5, 10))
    )
    self._residual_feature_dropout = float(
      train_cfg.pop("residual_feature_dropout", 0.0)
    )
    self._residual_arch = str(train_cfg.pop("residual_arch", "feature_mlp"))
    self._ref_edit_clip = float(train_cfg.pop("ref_edit_clip", 0.12))
    self._ref_edit_groups = str(train_cfg.pop("ref_edit_groups", "arms"))
    self._ref_edit_init_bias = str(train_cfg.pop("ref_edit_init_bias", ""))
    self._split_hand_net = bool(train_cfg.pop("split_hand_net", False))
    self._frame_hidden_dims = tuple(
      int(v) for v in train_cfg.pop("frame_hidden_dims", (128, 128))
    )
    self._token_residual_clip = float(train_cfg.pop("token_residual_clip", 0.1))
    self._token_residual_gain = float(train_cfg.pop("token_residual_gain", 1.0))
    raw_body_std = train_cfg.pop("body_init_std", None)
    raw_hand_std = train_cfg.pop("hand_init_std", None)
    raw_disabled_std = train_cfg.pop("disabled_init_std", None)
    self._body_init_std = None if raw_body_std is None else float(raw_body_std)
    self._hand_init_std = None if raw_hand_std is None else float(raw_hand_std)
    self._disabled_init_std = (
      None if raw_disabled_std is None else float(raw_disabled_std)
    )
    self._zero_init_residual = bool(train_cfg.pop("zero_init_residual", True))
    self._residual_l2_weight = float(train_cfg.pop("residual_l2_weight", 0.01))
    self._residual_smooth_weight = float(train_cfg.pop("residual_smooth_weight", 0.005))
    self._token_l2_weight = float(train_cfg.pop("token_l2_weight", 0.0))
    self._token_smooth_weight = float(train_cfg.pop("token_smooth_weight", 0.0))
    self._training_scheme = str(train_cfg.pop("training_scheme", "none")).strip()
    self._stage_body_iterations = int(train_cfg.pop("stage_body_iterations", 500))
    self._stage_hand_iterations = int(train_cfg.pop("stage_hand_iterations", 1000))
    self._stage_body_tracking_weight = float(
      train_cfg.pop("stage_body_tracking_weight", 1.0)
    )
    self._stage_hand_action_weight = float(
      train_cfg.pop("stage_hand_action_weight", 1.0)
    )
    self._stage_hand_raw_tip_weight = float(
      train_cfg.pop("stage_hand_raw_tip_weight", 0.5)
    )
    self._stage_hand_raw_radial_weight = float(
      train_cfg.pop("stage_hand_raw_radial_weight", 0.2)
    )
    self._stage_joint_tracking_weight = float(
      train_cfg.pop("stage_joint_tracking_weight", 1.0)
    )
    self._stage_joint_hand_action_weight = float(
      train_cfg.pop("stage_joint_hand_action_weight", 0.5)
    )
    self._stage_joint_raw_tip_weight = float(
      train_cfg.pop("stage_joint_raw_tip_weight", 0.5)
    )
    self._stage_joint_raw_radial_weight = float(
      train_cfg.pop("stage_joint_raw_radial_weight", 0.2)
    )
    self._stage_stability_weight = float(train_cfg.pop("stage_stability_weight", 0.5))
    self._hand_bc_weight = float(train_cfg.pop("hand_bc_weight", 0.0))
    self._hand_bc_start_iteration = int(train_cfg.pop("hand_bc_start_iteration", 0))
    raw_hand_bc_end = train_cfg.pop("hand_bc_end_iteration", None)
    self._hand_bc_end_iteration = (
      None if raw_hand_bc_end is None else int(raw_hand_bc_end)
    )
    self._hand_bc_checkpoint = str(train_cfg.pop("hand_bc_checkpoint", "")).strip()
    self._hand_bc_feature_groups = _parse_groups(
      train_cfg.pop("hand_bc_feature_groups", ())
    )
    self._freeze_hand_bc = bool(train_cfg.pop("freeze_hand_bc", True))
    self._hand_bc_action_clip = float(train_cfg.pop("hand_bc_action_clip", 5.0))
    self._hand_bc_base_start_frame = int(train_cfg.pop("hand_bc_base_start_frame", 0))
    raw_hand_residual_start = train_cfg.pop("hand_residual_start_frame", None)
    self._hand_residual_start_frame = (
      None if raw_hand_residual_start is None else int(raw_hand_residual_start)
    )
    self._residual_start_frame = int(train_cfg.pop("residual_start_frame", 0))
    self._residual_ramp_frames = int(train_cfg.pop("residual_ramp_frames", 0))
    self._residual_lowpass_alpha = float(train_cfg.pop("residual_lowpass_alpha", 1.0))
    raw_body_sample_delta_clip = train_cfg.pop("body_sample_delta_clip", None)
    self._body_sample_delta_clip = (
      None
      if raw_body_sample_delta_clip is None or float(raw_body_sample_delta_clip) <= 0.0
      else float(raw_body_sample_delta_clip)
    )
    raw_hand_sample_delta_clip = train_cfg.pop("hand_sample_delta_clip", None)
    self._hand_sample_delta_clip = (
      None
      if raw_hand_sample_delta_clip is None or float(raw_hand_sample_delta_clip) <= 0.0
      else float(raw_hand_sample_delta_clip)
    )
    raw_fixed_hand_frame = train_cfg.pop("fixed_hand_action_frame", None)
    self._fixed_hand_action_frame = (
      None if raw_fixed_hand_frame is None else int(raw_fixed_hand_frame)
    )
    self._hand_primitive_mode = str(
      train_cfg.pop("hand_primitive_mode", "none")
    ).strip()
    self._hand_primitive_close_frame = int(
      train_cfg.pop("hand_primitive_close_frame", 70)
    )
    self._hand_primitive_open_frame = int(
      train_cfg.pop("hand_primitive_open_frame", -1)
    )
    self._hand_primitive_hard_threshold = float(
      train_cfg.pop("hand_primitive_hard_threshold", 0.5)
    )
    self._hand_primitive_init_logit_bias = float(
      train_cfg.pop("hand_primitive_init_logit_bias", -2.0)
    )
    self._ppo_body_only = bool(train_cfg.pop("ppo_body_only", False))
    self._freeze_tracker = bool(train_cfg.pop("freeze_tracker", True))
    self._load_residual_config = bool(train_cfg.pop("load_residual_config", True))
    self._load_artifact_history = bool(train_cfg.pop("load_artifact_history", True))
    self._artifact_report_iters = tuple(
      int(v)
      for v in train_cfg.pop(
        "artifact_report_iters", (0, 100, 300, 600, 1000, 1500, 2000, 3000)
      )
    )
    self._eval_steps = int(train_cfg.pop("eval_steps", 300))
    raw_eval_start = train_cfg.pop("eval_reference_start_frame", 0)
    self._eval_reference_start_frame = (
      None if raw_eval_start is None else int(raw_eval_start)
    )
    self._write_eval_artifacts = bool(train_cfg.pop("write_eval_artifacts", True))
    self._wandb_metric_mode = str(train_cfg.pop("wandb_metric_mode", "compact"))
    self._wandb_metric_mode = self._wandb_metric_mode.strip().lower()
    if self._wandb_metric_mode not in {"compact", "object_sweep", "full"}:
      raise ValueError(
        "--agent.wandb-metric-mode must be 'compact', 'object_sweep', or 'full'."
      )
    self._horizon_curriculum = bool(train_cfg.pop("horizon_curriculum", False))
    self._horizon_start_ref_frames = int(train_cfg.pop("horizon_start_ref_frames", 50))
    self._horizon_final_ref_frames = train_cfg.pop("horizon_final_ref_frames", None)
    self._horizon_final_ref_frames = (
      None
      if self._horizon_final_ref_frames is None
      else int(self._horizon_final_ref_frames)
    )
    self._horizon_increment_ref_frames = int(
      train_cfg.pop("horizon_increment_ref_frames", 50)
    )
    self._horizon_success_threshold = float(
      train_cfg.pop("horizon_success_threshold", 0.03)
    )
    self._horizon_success_patience = int(train_cfg.pop("horizon_success_patience", 5))
    self._horizon_metric = str(
      train_cfg.pop("horizon_metric", "Metric/body_link_dist_mean")
    )
    self._horizon_startup_steps = int(train_cfg.pop("horizon_startup_steps", 36))
    self._horizon_min_iterations_per_stage = int(
      train_cfg.pop("horizon_min_iterations_per_stage", 0)
    )
    self._tracking_region_curriculum = bool(
      train_cfg.pop("tracking_region_curriculum", False)
    )
    self._tracking_region_initial = str(
      train_cfg.pop("tracking_region_initial", "all")
    ).strip()
    self._tracking_region_after = str(
      train_cfg.pop("tracking_region_after", "all")
    ).strip()
    self._tracking_region_metric = str(
      train_cfg.pop("tracking_region_metric", "Metric/lower_body_xyz_err_mse")
    )
    self._tracking_region_success_threshold = float(
      train_cfg.pop("tracking_region_success_threshold", 0.001)
    )
    self._tracking_region_success_patience = int(
      train_cfg.pop("tracking_region_success_patience", 5)
    )
    self._tracking_region_min_horizon_ref_frames = int(
      train_cfg.pop("tracking_region_min_horizon_ref_frames", 0)
    )
    self._horizon_current_ref_frames = max(1, self._horizon_start_ref_frames)
    self._horizon_current_steps = 0
    self._horizon_full_steps = 0
    self._horizon_success_streak = 0
    self._horizon_last_metric: float | None = None
    self._horizon_last_passed = False
    self._horizon_last_advanced = False
    self._horizon_last_advance_iteration: int | None = None
    self._horizon_stage_start_iteration = 0
    self._horizon_stage_age_iterations = 0
    self._horizon_last_stage_ready = False
    self._tracking_region_current = self._tracking_region_initial
    self._tracking_region_success_streak = 0
    self._tracking_region_last_metric: float | None = None
    self._tracking_region_last_passed = False
    self._tracking_region_last_advanced = False
    self._tracking_region_last_horizon_ready = False
    self._tracking_region_advance_iteration: int | None = None
    self._tracker_checkpoint_metadata: dict = {}
    self._train_metrics: dict[str, dict] = {}
    self._iteration_times: list[float] = []
    self._collection_times: list[float] = []
    self._learning_times: list[float] = []
    self._fps_values: list[float] = []
    self._run_wall_start: float | None = None
    self._run_start_iteration = 0
    self._completed_iterations = 0
    self._latest_snapshot: dict | None = None
    self._current_stage: str | None = None
    self._current_collect_iteration = 0

    if self._base_tracker_kind not in {"checkpoint", "official_onnx", "astra_onnx"}:
      raise ValueError(
        "--agent.base-tracker-kind must be 'checkpoint', 'official_onnx', "
        "or 'astra_onnx'."
      )
    if self._base_hand_mode != "zero":
      raise ValueError("--agent.base-hand-mode currently supports only 'zero'.")
    if self._base_tracker_kind in {"official_onnx", "astra_onnx"}:
      self._tracker_ckpt = None

    if self._base_tracker_kind == "checkpoint" and not self._tracker_ckpt:
      raise ValueError(
        "ResidualInteract requires --agent.tracker-ckpt. "
        "Refusing to train with a random base tracker."
      )
    if self._base_tracker_kind == "checkpoint":
      assert self._tracker_ckpt is not None
      tracker_path = Path(self._tracker_ckpt)
      if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker checkpoint not found: {tracker_path}")
    if self._base_tracker_kind == "official_onnx" and (
      not self._sonic_encoder_onnx or not Path(self._sonic_encoder_onnx).exists()
    ):
      raise FileNotFoundError(
        f"SONIC encoder ONNX not found: {self._sonic_encoder_onnx}"
      )
    if self._base_tracker_kind in {"checkpoint", "official_onnx"} and (
      not self._sonic_decoder_onnx or not Path(self._sonic_decoder_onnx).exists()
    ):
      raise FileNotFoundError(
        f"SONIC decoder ONNX not found: {self._sonic_decoder_onnx}"
      )
    if self._base_tracker_kind == "astra_onnx" and (
      not self._astra_onnx_path or not Path(self._astra_onnx_path).exists()
    ):
      raise FileNotFoundError(f"ASTRA ONNX not found: {self._astra_onnx_path}")
    if self._init_from not in {"ema", "raw"}:
      raise ValueError("--agent.init-from must be 'ema' or 'raw'.")
    if self._training_scheme not in {"none", "body_hand_joint"}:
      raise ValueError("--agent.training-scheme must be 'none' or 'body_hand_joint'.")
    if self._ppo_body_only and not self._split_hand_net:
      raise ValueError("--agent.ppo-body-only requires --agent.split-hand-net=True.")
    if self._hand_bc_checkpoint and not Path(self._hand_bc_checkpoint).exists():
      raise FileNotFoundError(
        f"Hand BC checkpoint not found: {self._hand_bc_checkpoint}"
      )
    if self._fixed_hand_action_frame is not None and self._fixed_hand_action_frame >= 0:
      self._hand_residual_gain = 0.0

    unknown = [
      g for g in self._residual_feature_groups if g not in mdp.RESIDUAL_FEATURE_GROUPS
    ]
    if unknown:
      raise ValueError(
        f"Unknown residual feature groups: {unknown}. "
        f"Available: {mdp.RESIDUAL_FEATURE_GROUPS}"
      )
    unknown_critic = [
      g for g in self._critic_feature_groups if g not in mdp.RESIDUAL_FEATURE_GROUPS
    ]
    if unknown_critic:
      raise ValueError(
        f"Unknown critic feature groups: {unknown_critic}. "
        f"Available: {mdp.RESIDUAL_FEATURE_GROUPS}"
      )
    if self._residual_arch == "frame_split":
      self._residual_feature_groups = ("reference_phase",)
    if not self._hand_bc_feature_groups:
      self._hand_bc_feature_groups = self._residual_feature_groups
    unknown_hand_bc = [
      g for g in self._hand_bc_feature_groups if g not in mdp.RESIDUAL_FEATURE_GROUPS
    ]
    if unknown_hand_bc:
      raise ValueError(
        f"Unknown hand BC feature groups: {unknown_hand_bc}. "
        f"Available: {mdp.RESIDUAL_FEATURE_GROUPS}"
      )

    actor_groups = ["sonic_obs_or_latent"]
    if self._base_tracker_kind == "official_onnx":
      actor_groups.append("sonic_encoder_obs")
    if self._base_tracker_kind == "astra_onnx":
      actor_groups.append("astra_obs")
    for group in (*self._residual_feature_groups, *self._hand_bc_feature_groups):
      if group not in actor_groups:
        actor_groups.append(group)
    default_critic_groups = (
      "proprio_history",
      "tracker_action",
      "reference_phase",
      "reference_preview",
      "object_state",
      "hand_object_geometry",
      "contact_features",
      "placement_goal",
      "tracking_error",
      "last_residual",
      "last_final_action",
    )
    critic_feature_groups = (
      self._critic_feature_groups
      if self._critic_feature_groups
      else (*default_critic_groups, *self._residual_feature_groups)
    )
    critic_groups = list(dict.fromkeys(("sonic_obs_or_latent", *critic_feature_groups)))
    train_cfg["obs_groups"] = {
      "actor": tuple(actor_groups),
      "critic": tuple(critic_groups),
    }
    train_cfg["actor"].update(
      {
        "base_tracker_kind": self._base_tracker_kind,
        "enc_input_dim": mdp.ENC_INPUT_DIM,
        "sonic_encoder_onnx": self._sonic_encoder_onnx,
        "sonic_decoder_onnx": self._sonic_decoder_onnx,
        "astra_onnx_path": self._astra_onnx_path,
        "base_hand_mode": self._base_hand_mode,
        "residual_feature_groups": self._residual_feature_groups,
        "residual_feature_dropout": self._residual_feature_dropout,
        "residual_arch": self._residual_arch,
        "ref_edit_clip": self._ref_edit_clip,
        "ref_edit_groups": self._ref_edit_groups,
        "ref_edit_init_bias": self._ref_edit_init_bias,
        "split_hand_net": self._split_hand_net,
        "frame_hidden_dims": self._frame_hidden_dims,
        "token_residual_clip": self._token_residual_clip,
        "token_residual_gain": self._token_residual_gain,
        "residual_gain": self._residual_gain,
        "body_residual_gain": self._body_residual_gain,
        "hand_residual_gain": self._hand_residual_gain,
        "fixed_hand_action_frame": self._fixed_hand_action_frame,
        "residual_action_clip": self._residual_action_clip,
        "final_action_clip": self._final_action_clip,
        "residual_mask": self._residual_mask,
        "body_init_std": self._body_init_std,
        "hand_init_std": self._hand_init_std,
        "disabled_init_std": self._disabled_init_std,
        "zero_init_residual": self._zero_init_residual,
        "freeze_tracker": self._freeze_tracker,
        "hand_bc_checkpoint": self._hand_bc_checkpoint,
        "hand_bc_feature_groups": self._hand_bc_feature_groups,
        "freeze_hand_bc": self._freeze_hand_bc,
        "hand_bc_action_clip": self._hand_bc_action_clip,
        "hand_bc_base_start_frame": self._hand_bc_base_start_frame,
        "hand_residual_start_frame": self._hand_residual_start_frame,
        "residual_start_frame": self._residual_start_frame,
        "residual_ramp_frames": self._residual_ramp_frames,
        "residual_lowpass_alpha": self._residual_lowpass_alpha,
        "body_sample_delta_clip": self._body_sample_delta_clip,
        "hand_sample_delta_clip": self._hand_sample_delta_clip,
        "hand_primitive_mode": self._hand_primitive_mode,
        "hand_primitive_close_frame": self._hand_primitive_close_frame,
        "hand_primitive_open_frame": self._hand_primitive_open_frame,
        "hand_primitive_hard_threshold": self._hand_primitive_hard_threshold,
        "hand_primitive_init_logit_bias": self._hand_primitive_init_logit_bias,
      }
    )

    super().__init__(env, train_cfg, log_dir, device)
    if self._base_tracker_kind == "checkpoint":
      self._load_and_freeze_tracker(device)
    elif self._base_tracker_kind == "official_onnx":
      self._install_official_onnx_tracker()
    else:
      self._install_astra_onnx_tracker()
    self._apply_residual_reward_weights()
    self._install_action_stats_bridge()
    if self._hand_bc_weight > 0.0 or self._ppo_body_only:
      self._install_hand_bc_and_body_ppo_loss()
    else:
      self._install_safe_ppo_loss()

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False, **kwargs
  ):
    del kwargs
    if init_at_random_ep_len:
      print(
        "[ResidualInteract] ignoring init_at_random_ep_len=True; "
        "residual task uses configured reset events"
      )

    if self._horizon_curriculum:
      self._initialize_horizon_curriculum()
    if self._tracking_region_curriculum:
      self._initialize_tracking_region_curriculum()

    obs, _ = self.env.reset()
    obs = obs.to(self.device)
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()
    self._run_wall_start = time.monotonic()
    start_it = self.current_learning_iteration
    self._run_start_iteration = int(start_it)
    total_it = start_it + num_learning_iterations

    try:
      for it in range(start_it, total_it):
        self._current_collect_iteration = int(it)
        self._apply_training_scheme_for_iteration(it)
        start = time.time()
        with torch.no_grad():
          for _ in range(self.cfg["num_steps_per_env"]):
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
            if self.cfg.get("check_for_nan", True):
              check_nan(obs, rewards, dones)
            obs, rewards, dones = (
              obs.to(self.device),
              rewards.to(self.device),
              dones.to(self.device),
            )
            self.alg.process_env_step(obs, rewards, dones, extras)
            intrinsic_rewards = (
              self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
            )
            self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

          stop = time.time()
          collect_time = stop - start
          start = stop
          self.alg.compute_returns(obs)

        loss_dict = self.alg.update()
        stop = time.time()
        learn_time = stop - start
        self.current_learning_iteration = it
        self._completed_iterations = it + 1

        snapshot = self._training_snapshot(
          it=it,
          total_it=total_it,
          collect_time=collect_time,
          learn_time=learn_time,
          loss_dict=loss_dict,
        )
        self._latest_snapshot = snapshot
        if self._tracking_region_curriculum:
          self._update_tracking_region_curriculum(snapshot, it)
        if self._horizon_curriculum:
          self._update_horizon_curriculum(snapshot, it)

        rnd_weight = None
        if self.cfg["algorithm"]["rnd_cfg"]:
          rnd_weight = getattr(self.alg.rnd, "weight", None)

        self._prepare_logger_ep_extras_for_wandb(snapshot)
        self.logger.log(
          it=it,
          start_it=start_it,
          total_it=total_it,
          collect_time=collect_time,
          learn_time=learn_time,
          loss_dict=loss_dict,
          learning_rate=self.alg.learning_rate,
          action_std=self.alg.get_policy().output_std,
          rnd_weight=rnd_weight,
        )

        report_it = self._artifact_report_index(it)
        artifact_key = self._artifact_key(it, total_it)
        if report_it in self._artifact_report_iters or it == total_it - 1:
          self._train_metrics[artifact_key] = snapshot
          self._write_train_artifacts()

        if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
          if self.logger.log_dir is None:
            raise RuntimeError("Logger writer is active but log_dir is not set.")
          save_path = os.path.join(self.logger.log_dir, f"model_{it}.pt")
          self.save(save_path)
    finally:
      if self.logger.writer is not None:
        self._finalize_run_artifacts()
        self.logger.stop_logging_writer()

  def _horizon_max_ref_frames(self) -> int:
    full_steps = self._horizon_full_steps or int(self.env.unwrapped.max_episode_length)
    default_final = max(1, full_steps - int(self._horizon_startup_steps))
    if self._horizon_final_ref_frames is None:
      return default_final
    return max(1, min(int(self._horizon_final_ref_frames), default_final))

  def _set_horizon_ref_frames(self, ref_frames: int) -> None:
    env = self.env.unwrapped
    max_ref_frames = self._horizon_max_ref_frames()
    ref_frames = max(1, min(int(ref_frames), max_ref_frames))
    steps = min(
      int(self._horizon_full_steps),
      int(self._horizon_startup_steps) + int(ref_frames),
    )
    steps = max(1, int(steps))
    env.cfg.episode_length_s = float(steps) * float(env.step_dt)
    if hasattr(self.env, "max_episode_length"):
      self.env.max_episode_length = steps
    self._horizon_current_ref_frames = ref_frames
    self._horizon_current_steps = steps

  def _initialize_horizon_curriculum(self) -> None:
    env = self.env.unwrapped
    self._horizon_full_steps = int(env.max_episode_length)
    self._horizon_success_streak = 0
    self._horizon_last_metric = None
    self._horizon_last_passed = False
    self._horizon_last_advanced = False
    self._horizon_last_advance_iteration = None
    self._horizon_stage_start_iteration = 0
    self._horizon_stage_age_iterations = 0
    self._horizon_last_stage_ready = self._horizon_min_iterations_per_stage <= 0
    self._set_horizon_ref_frames(self._horizon_start_ref_frames)
    print(
      "[ResidualInteract] horizon curriculum enabled: "
      f"start_ref_frames={self._horizon_current_ref_frames}, "
      f"current_steps={self._horizon_current_steps}, "
      f"final_ref_frames={self._horizon_max_ref_frames()}, "
      f"full_steps={self._horizon_full_steps}, "
      f"metric={self._horizon_metric}, "
      f"threshold={self._horizon_success_threshold}, "
      f"patience={self._horizon_success_patience}, "
      f"min_iterations_per_stage={self._horizon_min_iterations_per_stage}, "
      f"increment={self._horizon_increment_ref_frames}"
    )

  def _horizon_metrics(self) -> dict[str, float]:
    return {
      "Curriculum/horizon_enabled": float(self._horizon_curriculum),
      "Curriculum/horizon_ref_frames": float(self._horizon_current_ref_frames),
      "Curriculum/horizon_steps": float(self._horizon_current_steps),
      "Curriculum/horizon_final_ref_frames": float(self._horizon_max_ref_frames()),
      "Curriculum/horizon_success_threshold": float(self._horizon_success_threshold),
      "Curriculum/horizon_success_streak": float(self._horizon_success_streak),
      "Curriculum/horizon_last_metric": float(self._horizon_last_metric or 0.0),
      "Curriculum/horizon_last_passed": float(self._horizon_last_passed),
      "Curriculum/horizon_last_advanced": float(self._horizon_last_advanced),
      "Curriculum/horizon_last_advance_iteration": float(
        -1
        if self._horizon_last_advance_iteration is None
        else self._horizon_last_advance_iteration
      ),
      "Curriculum/horizon_stage_start_iteration": float(
        self._horizon_stage_start_iteration
      ),
      "Curriculum/horizon_stage_age_iterations": float(
        self._horizon_stage_age_iterations
      ),
      "Curriculum/horizon_min_iterations_per_stage": float(
        self._horizon_min_iterations_per_stage
      ),
      "Curriculum/horizon_stage_ready": float(self._horizon_last_stage_ready),
    }

  def _set_tracking_reward_link_group(self, link_group: str) -> None:
    cfg = self.env.unwrapped.reward_manager.get_term_cfg("tracking")
    cfg.params["link_group"] = str(link_group)
    self._tracking_region_current = str(link_group)

  def _initialize_tracking_region_curriculum(self) -> None:
    self._tracking_region_success_streak = 0
    self._tracking_region_last_metric = None
    self._tracking_region_last_passed = False
    self._tracking_region_last_advanced = False
    self._tracking_region_last_horizon_ready = False
    self._tracking_region_advance_iteration = None
    self._set_tracking_reward_link_group(self._tracking_region_initial)
    print(
      "[ResidualInteract] tracking region curriculum enabled: "
      f"initial={self._tracking_region_initial}, "
      f"after={self._tracking_region_after}, "
      f"metric={self._tracking_region_metric}, "
      f"threshold={self._tracking_region_success_threshold}, "
      f"patience={self._tracking_region_success_patience}, "
      f"min_horizon_ref_frames={self._tracking_region_min_horizon_ref_frames}"
    )

  def _tracking_region_curriculum_metrics(self) -> dict[str, float]:
    region_id = 0.0
    if self._tracking_region_current == self._tracking_region_after:
      region_id = 1.0
    return {
      "Curriculum/tracking_region_enabled": float(self._tracking_region_curriculum),
      "Curriculum/tracking_region_id": region_id,
      "Curriculum/tracking_region_success_threshold": float(
        self._tracking_region_success_threshold
      ),
      "Curriculum/tracking_region_min_horizon_ref_frames": float(
        self._tracking_region_min_horizon_ref_frames
      ),
      "Curriculum/tracking_region_horizon_ready": float(
        self._tracking_region_last_horizon_ready
      ),
      "Curriculum/tracking_region_success_streak": float(
        self._tracking_region_success_streak
      ),
      "Curriculum/tracking_region_last_metric": float(
        self._tracking_region_last_metric or 0.0
      ),
      "Curriculum/tracking_region_last_passed": float(
        self._tracking_region_last_passed
      ),
      "Curriculum/tracking_region_last_advanced": float(
        self._tracking_region_last_advanced
      ),
      "Curriculum/tracking_region_advance_iteration": float(
        -1
        if self._tracking_region_advance_iteration is None
        else self._tracking_region_advance_iteration
      ),
    }

  def _update_tracking_region_curriculum(self, snapshot: dict, it: int) -> None:
    self._tracking_region_last_advanced = False
    if self._tracking_region_current == self._tracking_region_after:
      return

    metrics = snapshot.get("metrics", {})
    raw_metric = metrics.get(self._tracking_region_metric)
    self._tracking_region_last_horizon_ready = (
      self._horizon_current_ref_frames >= self._tracking_region_min_horizon_ref_frames
    )
    if raw_metric is None:
      self._tracking_region_success_streak = 0
      self._tracking_region_last_passed = False
      return

    metric = float(raw_metric)
    self._tracking_region_last_metric = metric
    self._tracking_region_last_passed = (
      self._tracking_region_last_horizon_ready
      and metric <= self._tracking_region_success_threshold
    )
    if self._tracking_region_last_passed:
      self._tracking_region_success_streak += 1
    else:
      self._tracking_region_success_streak = 0

    if self._tracking_region_success_streak >= self._tracking_region_success_patience:
      old_region = self._tracking_region_current
      self._set_tracking_reward_link_group(self._tracking_region_after)
      self._tracking_region_success_streak = 0
      self._tracking_region_last_advanced = True
      self._tracking_region_advance_iteration = int(it)
      print(
        "[ResidualInteract] tracking region curriculum advanced: "
        f"iter={it}, metric={metric:.6g}, "
        f"region={old_region}->{self._tracking_region_current}"
      )

  def _update_horizon_curriculum(self, snapshot: dict, it: int) -> None:
    metrics = snapshot.get("metrics", {})
    raw_metric = metrics.get(self._horizon_metric)
    self._horizon_last_advanced = False
    if raw_metric is None:
      self._horizon_success_streak = 0
      self._horizon_last_passed = False
      return

    metric = float(raw_metric)
    self._horizon_stage_age_iterations = max(
      0, int(it) - int(self._horizon_stage_start_iteration)
    )
    self._horizon_last_stage_ready = (
      self._horizon_stage_age_iterations >= self._horizon_min_iterations_per_stage
    )
    self._horizon_last_metric = metric
    self._horizon_last_passed = (
      self._horizon_last_stage_ready and metric <= self._horizon_success_threshold
    )
    if self._horizon_last_passed:
      self._horizon_success_streak += 1
    else:
      self._horizon_success_streak = 0

    can_advance = self._horizon_current_ref_frames < self._horizon_max_ref_frames()
    if can_advance and self._horizon_success_streak >= self._horizon_success_patience:
      old_ref_frames = self._horizon_current_ref_frames
      new_ref_frames = old_ref_frames + self._horizon_increment_ref_frames
      self._set_horizon_ref_frames(new_ref_frames)
      self._horizon_success_streak = 0
      self._horizon_last_advanced = True
      self._horizon_last_advance_iteration = int(it)
      self._horizon_stage_start_iteration = int(it)
      self._horizon_stage_age_iterations = 0
      self._horizon_last_stage_ready = self._horizon_min_iterations_per_stage <= 0
      print(
        "[ResidualInteract] horizon curriculum advanced: "
        f"iter={it}, metric={metric:.6g}, "
        f"ref_frames={old_ref_frames}->{self._horizon_current_ref_frames}, "
        f"steps={self._horizon_current_steps}"
      )

  def _load_and_freeze_tracker(self, device: str) -> None:
    if self._tracker_ckpt is None:
      raise RuntimeError("Checkpoint tracker mode requires tracker_ckpt.")
    ckpt = torch.load(self._tracker_ckpt, map_location=device, weights_only=False)
    compat = ckpt.get("compat", {})
    expected = {
      "obs_dim": mdp.OBS_DIM_NO_TEACHER,
      "enc_input_dim": mdp.ENC_INPUT_DIM,
      "action_dim": mdp.ACTION_DIM,
    }
    mismatches = [
      f"{key}: expected {value}, got {compat.get(key)}"
      for key, value in expected.items()
      if compat.get(key) != value
    ]
    model_class = str(compat.get("model_class", ""))
    if model_class and "SONICActorModel" not in model_class:
      mismatches.append(
        f"model_class: expected SONICActorModel-compatible, got {model_class}"
      )
    if mismatches:
      raise RuntimeError("Tracker checkpoint schema mismatch: " + "; ".join(mismatches))

    state_key = (
      "actor_state_dict_ema" if self._init_from == "ema" else "actor_state_dict"
    )
    if state_key not in ckpt:
      if self._init_from == "ema" and "actor_state_dict" in ckpt:
        print(
          "[ResidualInteract] EMA actor missing; falling back to raw actor_state_dict"
        )
        state_key = "actor_state_dict"
      else:
        raise RuntimeError(f"Tracker checkpoint missing '{state_key}'.")

    actor = cast(Any, self.alg.actor)
    if not hasattr(actor, "base_tracker"):
      raise RuntimeError("Residual actor does not expose base_tracker.")
    actor.base_tracker.load_state_dict(ckpt[state_key], strict=True)
    self._tracker_checkpoint_metadata = dict(compat)

    counts = self._freeze_base_tracker_and_rebuild_optimizer()

    print(f"[ResidualInteract] loaded tracker checkpoint ({state_key}):")
    print(f"  {self._tracker_ckpt}")
    print(
      "[ResidualInteract] tracker checkpoint metadata: "
      f"{self._tracker_checkpoint_metadata}"
    )
    print(f"[ResidualInteract] tracker frozen params: {counts['frozen_count']:,}")
    print(
      f"[ResidualInteract] actor params: {counts['trainable_actor']:,} trainable, "
      f"{counts['frozen_actor']:,} frozen"
    )
    print(f"[ResidualInteract] critic trainable params: {counts['critic_params']:,}")
    print(
      "[ResidualInteract] optimizer rebuilt for residual actor + critic parameters only"
    )

  def _install_official_onnx_tracker(self) -> None:
    self._tracker_checkpoint_metadata = {
      "source": "official_onnx",
      "sonic_encoder_onnx": self._sonic_encoder_onnx,
      "sonic_decoder_onnx": self._sonic_decoder_onnx,
      "hand_base_mode": self._base_hand_mode,
      "obs_dim": mdp.OBS_DIM_NO_TEACHER,
      "enc_input_dim": mdp.SONIC_ENCODER_OBS_DIM,
      "action_dim": mdp.ACTION_DIM,
    }
    counts = self._freeze_base_tracker_and_rebuild_optimizer()
    print("[ResidualInteract] using official SONIC ONNX base tracker:")
    print(f"  encoder: {self._sonic_encoder_onnx}")
    print(f"  decoder: {self._sonic_decoder_onnx}")
    print(f"  hand_base_mode: {self._base_hand_mode}")
    print(f"[ResidualInteract] tracker frozen params: {counts['frozen_count']:,}")
    print(
      f"[ResidualInteract] actor params: {counts['trainable_actor']:,} trainable, "
      f"{counts['frozen_actor']:,} frozen"
    )
    print(f"[ResidualInteract] critic trainable params: {counts['critic_params']:,}")
    print(
      "[ResidualInteract] optimizer rebuilt for residual actor + critic parameters only"
    )

  def _install_astra_onnx_tracker(self) -> None:
    self._tracker_checkpoint_metadata = {
      "source": "astra_onnx",
      "astra_onnx_path": self._astra_onnx_path,
      "hand_base_mode": self._base_hand_mode,
      "obs_dim": mdp.ASTRA_OBS_DIM,
      "body_action_dim": mdp.NUM_BODY,
      "action_dim": mdp.ACTION_DIM,
      "action_mapping": "ASTRA PKL native -> target_q -> mjlab IL action",
    }
    counts = self._freeze_base_tracker_and_rebuild_optimizer()
    print("[ResidualInteract] using ASTRA ONNX frozen body tracker:")
    print(f"  astra_onnx: {self._astra_onnx_path}")
    print(f"  hand_base_mode: {self._base_hand_mode}")
    print("  action_mapping: ASTRA PKL native -> target_q -> mjlab IL action")
    print(f"[ResidualInteract] tracker frozen params: {counts['frozen_count']:,}")
    print(
      f"[ResidualInteract] actor params: {counts['trainable_actor']:,} trainable, "
      f"{counts['frozen_actor']:,} frozen"
    )
    print(f"[ResidualInteract] critic trainable params: {counts['critic_params']:,}")
    print(
      "[ResidualInteract] optimizer rebuilt for residual actor + critic parameters only"
    )

  def _freeze_base_tracker_and_rebuild_optimizer(self) -> dict[str, int]:
    actor = cast(Any, self.alg.actor)
    if not hasattr(actor, "base_tracker"):
      raise RuntimeError("Residual actor does not expose base_tracker.")

    frozen_count = 0
    if self._freeze_tracker:
      for param in actor.base_tracker.parameters():
        frozen_count += param.numel()
        param.requires_grad_(False)
      actor.base_tracker.eval()
    else:
      print(
        "[ResidualInteract] WARNING: freeze_tracker=False; "
        "tracker parameters are trainable."
      )

    trainable_actor = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    frozen_actor = sum(p.numel() for p in actor.parameters() if not p.requires_grad)
    critic_params = sum(
      p.numel() for p in self.alg.critic.parameters() if p.requires_grad
    )
    optim_params = [p for p in actor.parameters() if p.requires_grad]
    optim_params.extend(p for p in self.alg.critic.parameters() if p.requires_grad)
    self.alg.optimizer = type(self.alg.optimizer)(
      optim_params, lr=self.alg.learning_rate
    )
    return {
      "frozen_count": frozen_count,
      "trainable_actor": trainable_actor,
      "frozen_actor": frozen_actor,
      "critic_params": critic_params,
    }

  def _rebuild_optimizer_for_trainable_params(self) -> None:
    actor = cast(Any, self.alg.actor)
    optim_params = [p for p in actor.parameters() if p.requires_grad]
    optim_params.extend(p for p in self.alg.critic.parameters() if p.requires_grad)
    self.alg.optimizer = type(self.alg.optimizer)(
      optim_params, lr=self.alg.learning_rate
    )

  def _set_module_trainable(self, module: nn.Module | None, enabled: bool) -> None:
    if module is None:
      return
    for param in module.parameters():
      param.requires_grad_(enabled)

  def _set_reward_weight(self, name: str, weight: float) -> None:
    self.env.unwrapped.reward_manager.get_term_cfg(name).weight = float(weight)

  def _stage_name_for_iteration(self, iteration: int) -> str:
    if self._training_scheme != "body_hand_joint":
      return "none"
    if iteration < self._stage_body_iterations:
      return "body"
    hand_end = self._stage_body_iterations + self._stage_hand_iterations
    if iteration < hand_end:
      return "hand"
    return "joint"

  def _apply_stage_reward_profile(self, stage: str) -> None:
    self._set_reward_weight("omnigrasp_style", 0.0)
    self._set_reward_weight("raw_contact_tracking", 0.0)
    if stage == "body":
      self._set_reward_weight("tracking", self._stage_body_tracking_weight)
      self._set_reward_weight("hand_action_tracking", 0.0)
      self._set_reward_weight("raw_tip_object_tracking", 0.0)
      self._set_reward_weight("raw_tip_radial_tracking", 0.0)
    elif stage == "hand":
      self._set_reward_weight("tracking", 0.0)
      self._set_reward_weight("hand_action_tracking", self._stage_hand_action_weight)
      self._set_reward_weight(
        "raw_tip_object_tracking", self._stage_hand_raw_tip_weight
      )
      self._set_reward_weight(
        "raw_tip_radial_tracking", self._stage_hand_raw_radial_weight
      )
    elif stage == "joint":
      self._set_reward_weight("tracking", self._stage_joint_tracking_weight)
      self._set_reward_weight(
        "hand_action_tracking", self._stage_joint_hand_action_weight
      )
      self._set_reward_weight(
        "raw_tip_object_tracking", self._stage_joint_raw_tip_weight
      )
      self._set_reward_weight(
        "raw_tip_radial_tracking", self._stage_joint_raw_radial_weight
      )
    self._set_reward_weight("stability", self._stage_stability_weight)
    self._apply_residual_reward_weights()

  def _apply_stage_actor_profile(self, stage: str) -> None:
    actor = cast(Any, self.alg.actor)
    if self._freeze_tracker and hasattr(actor, "base_tracker"):
      for param in actor.base_tracker.parameters():
        param.requires_grad_(False)
      actor.base_tracker.eval()

    has_split_hand = bool(getattr(actor, "split_hand_net", False))
    if stage == "body":
      self._set_module_trainable(getattr(actor, "residual_mlp", None), True)
      self._set_module_trainable(getattr(actor, "hand_mlp", None), False)
      mask = _make_residual_mask(self._residual_mask, device=actor.residual_mask.device)
      mask[mdp.NUM_BODY :] = 0.0
      actor.residual_mask.copy_(mask)
      gain = _make_residual_gain(
        self._residual_gain,
        self._body_residual_gain,
        0.0,
        device=actor.residual_action_gain.device,
      )
      actor.residual_action_gain.copy_(gain)
    elif stage == "hand":
      self._set_module_trainable(getattr(actor, "residual_mlp", None), False)
      self._set_module_trainable(getattr(actor, "hand_mlp", None), has_split_hand)
      mask = torch.zeros_like(actor.residual_mask)
      mask[mdp.NUM_BODY :] = 1.0
      actor.residual_mask.copy_(mask)
      gain = _make_residual_gain(
        self._residual_gain,
        self._body_residual_gain,
        self._hand_residual_gain,
        device=actor.residual_action_gain.device,
      )
      actor.residual_action_gain.copy_(gain)
    elif stage == "joint":
      self._set_module_trainable(getattr(actor, "residual_mlp", None), True)
      self._set_module_trainable(getattr(actor, "hand_mlp", None), has_split_hand)
      self._apply_current_residual_config()
    self._rebuild_optimizer_for_trainable_params()

  def _apply_training_scheme_for_iteration(self, iteration: int) -> None:
    stage = self._stage_name_for_iteration(iteration)
    if stage == self._current_stage:
      return
    self._current_stage = stage
    if stage == "none":
      return
    self._apply_stage_reward_profile(stage)
    self._apply_stage_actor_profile(stage)
    print(
      "[ResidualInteract] training scheme stage: "
      f"{stage} at iteration {iteration}; "
      f"body_iters={self._stage_body_iterations}, "
      f"hand_iters={self._stage_hand_iterations}"
    )

  def _apply_residual_reward_weights(self) -> None:
    env = self.env.unwrapped
    env.reward_manager.get_term_cfg("residual_l2").weight = self._residual_l2_weight
    env.reward_manager.get_term_cfg(
      "residual_smooth"
    ).weight = self._residual_smooth_weight
    env.reward_manager.get_term_cfg("token_l2").weight = self._token_l2_weight
    env.reward_manager.get_term_cfg("token_smooth").weight = self._token_smooth_weight
    print(
      f"[ResidualInteract] residual regularization weights: "
      f"l2={self._residual_l2_weight}, smooth={self._residual_smooth_weight}, "
      f"token_l2={self._token_l2_weight}, token_smooth={self._token_smooth_weight}"
    )

  def _set_residual_action_stats(
    self,
    actor,
    actions: torch.Tensor,
    *,
    zero_residual: bool = False,
  ) -> None:
    env = self.env.unwrapped

    def set_env_attr(name: str, value) -> None:
      setattr(env, name, value)

    base = actor.last_base_action.detach().to(env.device)
    astra_action = getattr(actor.base_tracker, "last_astra_action_pkl", None)
    if not isinstance(astra_action, torch.Tensor) or astra_action.shape != (
      base.shape[0],
      mdp.NUM_BODY,
    ):
      astra_action = torch.zeros(base.shape[0], mdp.NUM_BODY, device=env.device)
    else:
      astra_action = astra_action.detach().to(env.device)
    previous_base = getattr(env, "_residual_last_base_action", None)
    previous_final = getattr(env, "_residual_last_final_action", None)
    previous_token = getattr(env, "_residual_last_token_delta", None)
    if (
      isinstance(previous_base, torch.Tensor)
      and isinstance(previous_final, torch.Tensor)
      and previous_base.shape == base.shape
      and previous_final.shape == base.shape
    ):
      previous_delta = previous_final.to(env.device) - previous_base.to(env.device)
    else:
      previous_delta = torch.zeros_like(base)
    final = actions.detach().to(env.device)
    if zero_residual or not hasattr(actor, "last_action_mean"):
      action_mean = final
    else:
      action_mean = actor.last_action_mean.detach().to(env.device)
    if zero_residual:
      raw_residual = torch.zeros_like(base)
      residual_delta = torch.zeros_like(base)
    else:
      raw_residual = actor.last_residual_action.detach().to(env.device)
      residual_delta = final - base
    if zero_residual or not hasattr(actor, "last_token_residual"):
      token_delta = torch.zeros(base.shape[0], 64, device=env.device)
    else:
      token_delta = actor.last_token_residual.detach().to(env.device)
    if (
      isinstance(previous_token, torch.Tensor)
      and previous_token.shape == token_delta.shape
    ):
      previous_token_delta = previous_token.to(env.device)
    else:
      previous_token_delta = torch.zeros_like(token_delta)
    if zero_residual or not hasattr(actor, "last_decoder_body_delta"):
      decoder_body_delta = torch.zeros(base.shape[0], mdp.NUM_BODY, device=env.device)
    else:
      decoder_body_delta = actor.last_decoder_body_delta.detach().to(env.device)
    if zero_residual or not hasattr(actor, "last_hand_control_gate"):
      hand_control_gate = torch.ones(base.shape[0], 1, device=env.device)
    else:
      hand_control_gate = actor.last_hand_control_gate.detach().to(env.device)
      if hand_control_gate.shape[:1] != base.shape[:1]:
        hand_control_gate = torch.ones(base.shape[0], 1, device=env.device)
      elif hand_control_gate.ndim == 1:
        hand_control_gate = hand_control_gate.unsqueeze(-1)
    hand_sample_pre = getattr(actor, "last_hand_sample_delta_pre_clip", None)
    if (
      not isinstance(hand_sample_pre, torch.Tensor)
      or hand_sample_pre.shape[:1] != base.shape[:1]
      or hand_sample_pre.shape[-1:] != (mdp.NUM_HAND,)
    ):
      hand_sample_pre = torch.zeros(base.shape[0], mdp.NUM_HAND, device=env.device)
    else:
      hand_sample_pre = hand_sample_pre.detach().to(env.device)
    hand_sample_post = getattr(actor, "last_hand_sample_delta_post_clip", None)
    if (
      not isinstance(hand_sample_post, torch.Tensor)
      or hand_sample_post.shape[:1] != base.shape[:1]
      or hand_sample_post.shape[-1:] != (mdp.NUM_HAND,)
    ):
      hand_sample_post = torch.zeros(base.shape[0], mdp.NUM_HAND, device=env.device)
    else:
      hand_sample_post = hand_sample_post.detach().to(env.device)
    hand_sample_clip_frac = getattr(actor, "last_hand_sample_clip_frac", None)
    if (
      not isinstance(hand_sample_clip_frac, torch.Tensor)
      or hand_sample_clip_frac.shape[:1] != base.shape[:1]
    ):
      hand_sample_clip_frac = torch.zeros(base.shape[0], device=env.device)
    else:
      hand_sample_clip_frac = hand_sample_clip_frac.detach().to(env.device)
    body_sample_pre = getattr(actor, "last_body_sample_delta_pre_clip", None)
    if (
      not isinstance(body_sample_pre, torch.Tensor)
      or body_sample_pre.shape[:1] != base.shape[:1]
      or body_sample_pre.shape[-1:] != (mdp.NUM_BODY,)
    ):
      body_sample_pre = torch.zeros(base.shape[0], mdp.NUM_BODY, device=env.device)
    else:
      body_sample_pre = body_sample_pre.detach().to(env.device)
    body_sample_post = getattr(actor, "last_body_sample_delta_post_clip", None)
    if (
      not isinstance(body_sample_post, torch.Tensor)
      or body_sample_post.shape[:1] != base.shape[:1]
      or body_sample_post.shape[-1:] != (mdp.NUM_BODY,)
    ):
      body_sample_post = torch.zeros(base.shape[0], mdp.NUM_BODY, device=env.device)
    else:
      body_sample_post = body_sample_post.detach().to(env.device)
    body_sample_clip_frac = getattr(actor, "last_body_sample_clip_frac", None)
    if (
      not isinstance(body_sample_clip_frac, torch.Tensor)
      or body_sample_clip_frac.shape[:1] != base.shape[:1]
    ):
      body_sample_clip_frac = torch.zeros(base.shape[0], device=env.device)
    else:
      body_sample_clip_frac = body_sample_clip_frac.detach().to(env.device)
    hand_action_std_mean = getattr(actor, "last_hand_action_std_mean", None)
    if not isinstance(hand_action_std_mean, torch.Tensor):
      hand_action_std_mean = torch.zeros(1, device=env.device)
    else:
      hand_action_std_mean = hand_action_std_mean.detach().to(env.device)
    hand_primitive_close = getattr(actor, "last_hand_primitive_close", None)
    if (
      not isinstance(hand_primitive_close, torch.Tensor)
      or hand_primitive_close.shape[:1] != base.shape[:1]
      or hand_primitive_close.shape[-1] != 2
    ):
      hand_primitive_close = torch.zeros(base.shape[0], 2, device=env.device)
    else:
      hand_primitive_close = hand_primitive_close.detach().to(env.device)
    hand_primitive_delta = getattr(actor, "last_hand_primitive_delta", None)
    if (
      not isinstance(hand_primitive_delta, torch.Tensor)
      or hand_primitive_delta.shape[:1] != base.shape[:1]
      or hand_primitive_delta.shape[-1] != mdp.NUM_HAND
    ):
      hand_primitive_delta = torch.zeros(base.shape[0], mdp.NUM_HAND, device=env.device)
    else:
      hand_primitive_delta = hand_primitive_delta.detach().to(env.device)
    set_env_attr("_residual_last_base_action", base)
    set_env_attr("_residual_last_astra_action_pkl", astra_action)
    set_env_attr("_residual_last_action_mean", action_mean)
    set_env_attr("_residual_last_residual_action", residual_delta)
    set_env_attr("_residual_last_raw_residual_action", raw_residual)
    set_env_attr("_residual_previous_residual_delta", previous_delta)
    set_env_attr("_residual_last_final_action", final)
    set_env_attr("_residual_last_token_delta", token_delta)
    set_env_attr("_residual_previous_token_delta", previous_token_delta)
    set_env_attr("_residual_last_decoder_body_delta", decoder_body_delta)
    set_env_attr("_residual_hand_control_gate", hand_control_gate)
    set_env_attr("_residual_hand_sample_delta_pre_clip", hand_sample_pre)
    set_env_attr("_residual_hand_sample_delta_post_clip", hand_sample_post)
    set_env_attr("_residual_hand_sample_clip_frac", hand_sample_clip_frac)
    set_env_attr("_residual_body_sample_delta_pre_clip", body_sample_pre)
    set_env_attr("_residual_body_sample_delta_post_clip", body_sample_post)
    set_env_attr("_residual_body_sample_clip_frac", body_sample_clip_frac)
    set_env_attr("_residual_hand_action_std_mean", hand_action_std_mean)
    set_env_attr("_residual_hand_primitive_close", hand_primitive_close)
    set_env_attr("_residual_hand_primitive_delta", hand_primitive_delta)
    set_env_attr("_residual_action_mask", actor.residual_mask.detach().to(env.device))
    set_env_attr(
      "_residual_action_gain", actor.residual_action_gain.detach().to(env.device)
    )
    set_env_attr("_residual_action_clip", float(actor.residual_action_clip))
    body_sample_clip = getattr(actor, "body_sample_delta_clip", None)
    body_sample_clip = None if body_sample_clip is None else float(body_sample_clip)
    set_env_attr("_residual_body_sample_delta_clip", body_sample_clip)
    hand_sample_clip = getattr(actor, "hand_sample_delta_clip", None)
    hand_sample_clip = None if hand_sample_clip is None else float(hand_sample_clip)
    set_env_attr("_residual_hand_sample_delta_clip", hand_sample_clip)
    token_clip = getattr(actor, "token_residual_clip", None)
    token_clip = None if token_clip is None else float(token_clip)
    set_env_attr("_residual_token_clip", token_clip)
    final_clip = getattr(actor, "final_action_clip", None)
    final_clip = None if final_clip is None else float(final_clip)
    set_env_attr("_residual_final_action_clip", final_clip)

  def _install_action_stats_bridge(self) -> None:
    if hasattr(self.alg, "_residual_orig_act"):
      return
    runner = self
    alg = cast(Any, self.alg)
    alg._residual_orig_act = alg.act

    def act_with_residual_stats(alg_self, obs):
      actions = alg_self._residual_orig_act(obs)
      runner._set_residual_action_stats(alg_self.actor, actions)
      return actions

    alg.act = types.MethodType(act_with_residual_stats, alg)

  def _hand_bc_lambda_for_iteration(self, iteration: int) -> float:
    if self._hand_bc_weight <= 0.0:
      return 0.0
    if iteration < self._hand_bc_start_iteration:
      return 0.0
    if (
      self._hand_bc_end_iteration is not None
      and iteration >= self._hand_bc_end_iteration
    ):
      return 0.0
    return self._hand_bc_weight

  def _install_hand_bc_and_body_ppo_loss(self) -> None:
    alg = cast(Any, self.alg)
    if hasattr(alg, "_residual_hand_bc_installed"):
      return
    if self.alg.rnd:
      raise RuntimeError("Residual hand BC hook does not support RND.")
    if self.alg.symmetry:
      raise RuntimeError("Residual hand BC hook does not support symmetry.")
    if self.alg.is_multi_gpu:
      raise RuntimeError("Residual hand BC hook is single-GPU only.")

    storage = alg.storage
    if not hasattr(storage, "hand_bc_actions"):
      storage.hand_bc_actions = torch.zeros(
        storage.num_transitions_per_env,
        storage.num_envs,
        mdp.NUM_HAND,
        device=storage.device,
      )
      storage.hand_bc_lambdas = torch.zeros(
        storage.num_transitions_per_env,
        storage.num_envs,
        1,
        device=storage.device,
      )
      storage._residual_orig_add_transition = storage.add_transition
      storage._residual_orig_mini_batch_generator = storage.mini_batch_generator

      def add_transition_with_hand_bc(storage_self, transition):
        step = storage_self.step
        storage_self._residual_orig_add_transition(transition)
        storage_self.hand_bc_actions[step].copy_(transition.hand_bc_actions)
        storage_self.hand_bc_lambdas[step].copy_(transition.hand_bc_lambdas)

      def mini_batch_generator_with_hand_bc(
        storage_self,
        num_mini_batches: int,
        num_epochs: int = 8,
      ):
        batch_size = storage_self.num_envs * storage_self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
          num_mini_batches * mini_batch_size,
          requires_grad=False,
          device=storage_self.device,
        )
        observations = storage_self.observations.flatten(0, 1)
        actions = storage_self.actions.flatten(0, 1)
        values = storage_self.values.flatten(0, 1)
        returns = storage_self.returns.flatten(0, 1)
        old_actions_log_prob = storage_self.actions_log_prob.flatten(0, 1)
        advantages = storage_self.advantages.flatten(0, 1)
        old_distribution_params = tuple(
          p.flatten(0, 1) for p in storage_self.distribution_params
        )
        hand_bc_actions = storage_self.hand_bc_actions.flatten(0, 1)
        hand_bc_lambdas = storage_self.hand_bc_lambdas.flatten(0, 1)
        for _epoch in range(num_epochs):
          for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            batch_idx = indices[start:stop]
            batch = storage_self.Batch(
              observations=observations[batch_idx],
              actions=actions[batch_idx],
              values=values[batch_idx],
              advantages=advantages[batch_idx],
              returns=returns[batch_idx],
              old_actions_log_prob=old_actions_log_prob[batch_idx],
              old_distribution_params=tuple(
                p[batch_idx] for p in old_distribution_params
              ),
            )
            batch.hand_bc_actions = hand_bc_actions[batch_idx]
            batch.hand_bc_lambdas = hand_bc_lambdas[batch_idx]
            yield batch

      storage.add_transition = types.MethodType(add_transition_with_hand_bc, storage)
      storage.mini_batch_generator = types.MethodType(
        mini_batch_generator_with_hand_bc, storage
      )

    runner = self
    alg._residual_hand_bc_orig_act = alg.act

    def act_with_hand_bc(alg_self, obs):
      actions = alg_self._residual_hand_bc_orig_act(obs)
      teacher = apple_mdp.teacher_action(runner.env.unwrapped)[:, mdp.NUM_BODY :]
      weight = runner._hand_bc_lambda_for_iteration(runner._current_collect_iteration)
      active = mdp._active_after_startup(runner.env.unwrapped).unsqueeze(-1)
      alg_self.transition.hand_bc_actions = teacher.to(actions.device)
      alg_self.transition.hand_bc_lambdas = (active * float(weight)).to(actions.device)
      return actions

    alg.act = types.MethodType(act_with_hand_bc, alg)

    def update_with_hand_bc(alg_self):
      mean_value_loss = 0.0
      mean_surrogate_loss = 0.0
      mean_entropy = 0.0
      mean_hand_bc_loss = 0.0
      mean_hand_bc_lambda = 0.0
      generator = alg_self.storage.mini_batch_generator(
        alg_self.num_mini_batches,
        alg_self.num_learning_epochs,
      )
      for batch in generator:
        original_batch_size = batch.observations.batch_size[0]
        if alg_self.normalize_advantage_per_mini_batch:
          with torch.no_grad():
            batch.advantages = (batch.advantages - batch.advantages.mean()) / (
              batch.advantages.std() + 1.0e-8
            )

        alg_self.actor(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        values = alg_self.critic(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[1],
        )
        distribution_params = tuple(
          p[:original_batch_size] for p in alg_self.actor.output_distribution_params
        )
        if runner._ppo_body_only:
          mean, std = distribution_params
          actions_log_prob = _diag_gaussian_log_prob_sum(
            mean, std, batch.actions, 0, mdp.NUM_BODY
          )
          old_mean, old_std = batch.old_distribution_params
          old_actions_log_prob = _diag_gaussian_log_prob_sum(
            old_mean, old_std, batch.actions, 0, mdp.NUM_BODY
          )
          entropy = _diag_gaussian_entropy_sum(std, 0, mdp.NUM_BODY)
        else:
          actions_log_prob = alg_self.actor.get_output_log_prob(batch.actions)
          old_actions_log_prob = torch.squeeze(batch.old_actions_log_prob)
          entropy = alg_self.actor.output_entropy[:original_batch_size]

        if alg_self.desired_kl is not None and alg_self.schedule == "adaptive":
          with torch.inference_mode():
            kl = alg_self.actor.get_kl_divergence(
              batch.old_distribution_params,
              distribution_params,
            )
            kl_mean = torch.mean(kl)
            if alg_self.gpu_global_rank == 0:
              if kl_mean > alg_self.desired_kl * 2.0:
                alg_self.learning_rate = max(1.0e-5, alg_self.learning_rate / 1.5)
              elif kl_mean < alg_self.desired_kl / 2.0 and kl_mean > 0.0:
                alg_self.learning_rate = min(1.0e-2, alg_self.learning_rate * 1.5)
            for param_group in alg_self.optimizer.param_groups:
              param_group["lr"] = alg_self.learning_rate

        ratio = torch.exp((actions_log_prob - old_actions_log_prob).clamp(-20.0, 20.0))
        surrogate = -torch.squeeze(batch.advantages) * ratio
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
          ratio,
          1.0 - alg_self.clip_param,
          1.0 + alg_self.clip_param,
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if alg_self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -alg_self.clip_param,
            alg_self.clip_param,
          )
          value_losses = (values - batch.returns).pow(2)
          value_losses_clipped = (value_clipped - batch.returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (batch.returns - values).pow(2).mean()

        action_mean = torch.nan_to_num(
          alg_self.actor.output_mean,
          nan=0.0,
          posinf=5.0,
          neginf=-5.0,
        )
        hand_err = (
          (action_mean[:, mdp.NUM_BODY :] - batch.hand_bc_actions)
          .pow(2)
          .mean(dim=-1, keepdim=True)
        )
        hand_bc_loss = (batch.hand_bc_lambdas * hand_err).mean()
        loss = (
          surrogate_loss
          + alg_self.value_loss_coef * value_loss
          - alg_self.entropy_coef * entropy.mean()
          + hand_bc_loss
        )

        alg_self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(alg_self.actor.parameters(), alg_self.max_grad_norm)
        nn.utils.clip_grad_norm_(alg_self.critic.parameters(), alg_self.max_grad_norm)
        alg_self.optimizer.step()

        mean_value_loss += float(value_loss.item())
        mean_surrogate_loss += float(surrogate_loss.item())
        mean_entropy += float(entropy.mean().item())
        mean_hand_bc_loss += float(hand_bc_loss.item())
        mean_hand_bc_lambda += float(batch.hand_bc_lambdas.mean().item())

      num_updates = alg_self.num_learning_epochs * alg_self.num_mini_batches
      alg_self.storage.clear()
      return {
        "value": mean_value_loss / num_updates,
        "surrogate": mean_surrogate_loss / num_updates,
        "entropy": mean_entropy / num_updates,
        "hand_bc": mean_hand_bc_loss / num_updates,
        "hand_bc_lambda": mean_hand_bc_lambda / num_updates,
        "ppo_body_only": float(runner._ppo_body_only),
      }

    alg.update = types.MethodType(update_with_hand_bc, alg)
    alg._residual_hand_bc_installed = True
    print(
      "[ResidualInteract] installed hand BC/body-PPO loss hook: "
      f"hand_bc_weight={self._hand_bc_weight}, ppo_body_only={self._ppo_body_only}"
    )

  def _install_safe_ppo_loss(self) -> None:
    """Install a numerically guarded PPO update for residual action policies."""
    alg = cast(Any, self.alg)
    if hasattr(alg, "_residual_safe_ppo_installed"):
      return
    if self.alg.rnd:
      raise RuntimeError("Residual safe PPO hook does not support RND.")
    if self.alg.symmetry:
      raise RuntimeError("Residual safe PPO hook does not support symmetry.")
    log_ratio_clip = float(os.environ.get("RESIDUAL_SAFE_PPO_LOG_RATIO_CLIP", "10.0"))
    log_ratio_clip = max(float(log_ratio_clip), 1.0)

    def update_with_safe_ppo(alg_self):
      mean_value_loss = 0.0
      mean_surrogate_loss = 0.0
      mean_entropy = 0.0
      mean_ratio_max = 0.0
      mean_log_ratio_abs_max = 0.0
      kl_sum = 0.0
      kl_max = 0.0
      kl_n = 0
      kl_nonfinite = 0
      skipped_updates = 0
      applied_updates = 0

      if alg_self.actor.is_recurrent or alg_self.critic.is_recurrent:
        generator = alg_self.storage.recurrent_mini_batch_generator(
          alg_self.num_mini_batches,
          alg_self.num_learning_epochs,
        )
      else:
        generator = alg_self.storage.mini_batch_generator(
          alg_self.num_mini_batches,
          alg_self.num_learning_epochs,
        )

      for batch in generator:
        original_batch_size = batch.observations.batch_size[0]
        with torch.no_grad():
          advantages = torch.nan_to_num(
            batch.advantages,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
          )
          adv_std = advantages.std(unbiased=False).clamp_min(1.0e-8)
          batch.advantages = (advantages - advantages.mean()) / adv_std

        alg_self.actor(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[0],
          stochastic_output=True,
        )
        actions_log_prob = alg_self.actor.get_output_log_prob(batch.actions)
        values = alg_self.critic(
          batch.observations,
          masks=batch.masks,
          hidden_state=batch.hidden_states[1],
        )
        distribution_params = tuple(
          p[:original_batch_size] for p in alg_self.actor.output_distribution_params
        )
        entropy = alg_self.actor.output_entropy[:original_batch_size]

        if alg_self.desired_kl is not None and alg_self.schedule == "adaptive":
          with torch.inference_mode():
            kl = alg_self.actor.get_kl_divergence(
              batch.old_distribution_params,
              distribution_params,
            )
            kl_mean = torch.nan_to_num(
              torch.mean(kl),
              nan=float("inf"),
              posinf=float("inf"),
              neginf=0.0,
            )
            if alg_self.is_multi_gpu:
              torch.distributed.all_reduce(
                kl_mean,
                op=torch.distributed.ReduceOp.SUM,
              )
              kl_mean /= alg_self.gpu_world_size
            _klv = float(kl_mean)
            if _klv == float("inf") or _klv != _klv:
              kl_nonfinite += 1
            else:
              kl_sum += _klv
              kl_n += 1
              if _klv > kl_max:
                kl_max = _klv
            if alg_self.gpu_global_rank == 0:
              if kl_mean > alg_self.desired_kl * 2.0:
                alg_self.learning_rate = max(1.0e-5, alg_self.learning_rate / 1.5)
              elif kl_mean < alg_self.desired_kl / 2.0 and kl_mean > 0.0:
                alg_self.learning_rate = min(1.0e-2, alg_self.learning_rate * 1.5)
            if alg_self.is_multi_gpu:
              lr_tensor = torch.tensor(alg_self.learning_rate, device=alg_self.device)
              torch.distributed.broadcast(lr_tensor, src=0)
              alg_self.learning_rate = lr_tensor.item()
            for param_group in alg_self.optimizer.param_groups:
              param_group["lr"] = alg_self.learning_rate

        old_actions_log_prob = torch.squeeze(batch.old_actions_log_prob)
        log_ratio = actions_log_prob - old_actions_log_prob
        log_ratio = torch.nan_to_num(
          log_ratio,
          nan=0.0,
          posinf=log_ratio_clip,
          neginf=-log_ratio_clip,
        )
        log_ratio = log_ratio.clamp(-log_ratio_clip, log_ratio_clip)
        ratio = torch.exp(log_ratio)
        advantages = torch.squeeze(batch.advantages)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
          ratio,
          1.0 - alg_self.clip_param,
          1.0 + alg_self.clip_param,
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        returns = torch.nan_to_num(
          batch.returns,
          nan=0.0,
          posinf=0.0,
          neginf=0.0,
        )
        if alg_self.use_clipped_value_loss:
          value_clipped = batch.values + (values - batch.values).clamp(
            -alg_self.clip_param,
            alg_self.clip_param,
          )
          value_losses = (values - returns).pow(2)
          value_losses_clipped = (value_clipped - returns).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = (returns - values).pow(2).mean()

        entropy_mean = torch.nan_to_num(
          entropy.mean(),
          nan=0.0,
          posinf=0.0,
          neginf=0.0,
        )
        loss = (
          surrogate_loss
          + alg_self.value_loss_coef * value_loss
          - alg_self.entropy_coef * entropy_mean
        )
        if not torch.isfinite(loss):
          skipped_updates += 1
          alg_self.optimizer.zero_grad(set_to_none=True)
          continue

        alg_self.optimizer.zero_grad()
        loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(
          alg_self.actor.parameters(),
          alg_self.max_grad_norm,
        )
        critic_norm = nn.utils.clip_grad_norm_(
          alg_self.critic.parameters(),
          alg_self.max_grad_norm,
        )
        if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
          skipped_updates += 1
          alg_self.optimizer.zero_grad(set_to_none=True)
          continue
        alg_self.optimizer.step()

        applied_updates += 1
        mean_value_loss += float(value_loss.item())
        mean_surrogate_loss += float(surrogate_loss.item())
        mean_entropy += float(entropy_mean.item())
        mean_ratio_max += float(ratio.max().item())
        mean_log_ratio_abs_max += float(log_ratio.abs().max().item())

      total_updates = alg_self.num_learning_epochs * alg_self.num_mini_batches
      denom = max(applied_updates, 1)
      alg_self.storage.clear()
      return {
        "value": mean_value_loss / denom,
        "surrogate": mean_surrogate_loss / denom,
        "entropy": mean_entropy / denom,
        "safe_ppo_skipped": float(skipped_updates),
        "safe_ppo_applied": float(applied_updates),
        "safe_ppo_total": float(total_updates),
        "ratio_max": mean_ratio_max / denom,
        "log_ratio_abs_max": mean_log_ratio_abs_max / denom,
        "kl_mean": kl_sum / max(kl_n, 1),
        "kl_max": kl_max,
        "kl_nonfinite_frac": kl_nonfinite / max(kl_n + kl_nonfinite, 1),
      }

    alg.update = types.MethodType(update_with_safe_ppo, alg)
    alg._residual_safe_ppo_installed = True
    print(
      "[ResidualInteract] installed safe PPO loss hook for residual policy: "
      f"log_ratio_clip={log_ratio_clip:g}"
    )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")
    self._patch_legacy_actor_state(actor_sd, loaded_dict)
    loaded_dict["actor_state_dict"] = actor_sd
    migrated_input_dims = self._patch_legacy_input_state_dims(loaded_dict)
    migrated_actor_arch = self._patch_actor_state_for_current_arch(loaded_dict)
    if migrated_input_dims or migrated_actor_arch:
      default_load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": True,
      }
      load_cfg = default_load_cfg if load_cfg is None else {**load_cfg}
      load_cfg["optimizer"] = False
      reason = "input-dim migration" if migrated_input_dims else "actor migration"
      if migrated_input_dims and migrated_actor_arch:
        reason = "input-dim and actor migration"
      print(
        "[ResidualInteract] skipped optimizer state load after "
        f"{reason}; Adam moments will restart from the loaded weights."
      )

    try:
      load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    except ValueError as exc:
      optimizer_load_enabled = load_cfg is None or load_cfg.get("optimizer", True)
      if (
        optimizer_load_enabled
        and "loaded state dict contains a parameter group" in str(exc)
      ):
        retry_load_cfg = {
          "actor": True,
          "critic": True,
          "optimizer": False,
          "iteration": True,
          "rnd": True,
        }
        if load_cfg is not None:
          retry_load_cfg = {**load_cfg, "optimizer": False}
        print(
          "[ResidualInteract] skipped optimizer state load because the "
          "checkpoint optimizer parameter groups do not match the current "
          "trainable parameters; Adam moments will restart from loaded weights."
        )
        load_iteration = self.alg.load(loaded_dict, retry_load_cfg, strict)
      else:
        raise
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]

    if self._load_residual_config:
      self._apply_loaded_residual_config(loaded_dict)
    else:
      self._apply_current_residual_config()
      self._apply_cli_std_profile_after_load()
      print(
        "[ResidualInteract] kept CLI residual config after checkpoint load: "
        f"gain={self._residual_gain}, body_gain={self._body_residual_gain}, "
        f"hand_gain={self._hand_residual_gain}, clip={self._residual_action_clip}, "
        f"mask={self._residual_mask}"
      )
    if self._load_artifact_history:
      self._completed_iterations = int(
        loaded_dict.get("completed_iterations", self.current_learning_iteration)
      )
      self._train_metrics = dict(loaded_dict.get("train_metrics", self._train_metrics))
      self._iteration_times = list(
        loaded_dict.get("iteration_times", self._iteration_times)
      )
      self._collection_times = list(
        loaded_dict.get("collection_times", self._collection_times)
      )
      self._learning_times = list(
        loaded_dict.get("learning_times", self._learning_times)
      )
      self._fps_values = list(loaded_dict.get("fps_values", self._fps_values))
      if self._train_metrics:
        latest_key = max(self._train_metrics.keys(), key=lambda k: int(k))
        self._latest_snapshot = self._train_metrics[latest_key]
    else:
      self._completed_iterations = int(self.current_learning_iteration)
      self._train_metrics = {}
      self._iteration_times = []
      self._collection_times = []
      self._learning_times = []
      self._fps_values = []
      self._latest_snapshot = None
      print("[ResidualInteract] started a fresh artifact history after resume")
    return infos

  def _apply_cli_std_profile_after_load(self) -> None:
    if (
      self._body_init_std is None
      and self._hand_init_std is None
      and self._disabled_init_std is None
    ):
      return
    actor = cast(Any, self.alg.actor)
    set_initial_std = getattr(actor, "_set_initial_std", None)
    if not callable(set_initial_std):
      return
    set_initial_std(
      body_init_std=self._body_init_std,
      hand_init_std=self._hand_init_std,
      disabled_init_std=self._disabled_init_std,
    )
    print(
      "[ResidualInteract] reapplied CLI exploration std after checkpoint load: "
      f"body={self._body_init_std}, hand={self._hand_init_std}, "
      f"disabled={self._disabled_init_std}"
    )

  def _patch_legacy_actor_state(self, actor_sd: dict, loaded_dict: dict) -> None:
    actor = cast(Any, self.alg.actor)
    if (
      hasattr(actor, "residual_action_gain") and "residual_action_gain" not in actor_sd
    ):
      if self._load_residual_config:
        gain = float(loaded_dict.get("residual_gain", self._residual_gain))
        body_gain = loaded_dict.get("body_residual_gain", self._body_residual_gain)
        hand_gain = loaded_dict.get("hand_residual_gain", self._hand_residual_gain)
      else:
        gain = self._residual_gain
        body_gain = self._body_residual_gain
        hand_gain = self._hand_residual_gain
      actor_sd["residual_action_gain"] = _make_residual_gain(
        gain,
        None if body_gain is None else float(body_gain),
        None if hand_gain is None else float(hand_gain),
        device="cpu",
      )
      print("[ResidualInteract] patched legacy checkpoint: added residual_action_gain")

    if hasattr(actor, "residual_mask") and "residual_mask" not in actor_sd:
      mask = str(loaded_dict.get("residual_mask", self._residual_mask))
      actor_sd["residual_mask"] = _make_residual_mask(mask, device="cpu")
      print("[ResidualInteract] patched legacy checkpoint: added residual_mask")

  def _remap_feature_columns(
    self,
    tensor: torch.Tensor,
    *,
    old_dims: dict[str, int],
    current_groups: tuple[str, ...],
    current_dims: dict[str, int],
    fill_value: float,
  ) -> torch.Tensor:
    old_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for group, dim in old_dims.items():
      dim = int(dim)
      old_offsets[group] = (offset, offset + dim)
      offset += dim

    chunks: list[torch.Tensor] = []
    for group in current_groups:
      new_dim = int(current_dims[group])
      if group in old_offsets:
        start, end = old_offsets[group]
        old_chunk = tensor[..., start:end]
        copy_dim = min(old_chunk.shape[-1], new_dim)
        chunk = old_chunk[..., :copy_dim]
        if copy_dim < new_dim:
          pad = torch.full(
            (*tensor.shape[:-1], new_dim - copy_dim),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
          )
          chunk = torch.cat([chunk, pad], dim=-1)
      else:
        chunk = torch.full(
          (*tensor.shape[:-1], new_dim),
          fill_value,
          dtype=tensor.dtype,
          device=tensor.device,
        )
      chunks.append(chunk)
    return torch.cat(chunks, dim=-1)

  def _patch_normalizer_input_dim(
    self,
    state_dict: dict,
    *,
    prefix: str,
    old_dims: dict[str, int],
    current_groups: tuple[str, ...],
    current_dims: dict[str, int],
  ) -> bool:
    changed = False
    for suffix, fill_value in (
      ("_mean", 0.0),
      ("_var", 1.0),
      ("_std", 1.0),
    ):
      key = f"{prefix}{suffix}"
      value = state_dict.get(key)
      if not isinstance(value, torch.Tensor):
        continue
      expected = sum(int(current_dims[g]) for g in current_groups)
      if value.shape[-1] == expected:
        continue
      state_dict[key] = self._remap_feature_columns(
        value,
        old_dims=old_dims,
        current_groups=current_groups,
        current_dims=current_dims,
        fill_value=fill_value,
      )
      print(
        "[ResidualInteract] patched legacy normalizer input dim: "
        f"{key} {tuple(value.shape)} -> {tuple(state_dict[key].shape)}"
      )
      changed = True
    return changed

  def _patch_linear_input_dim(
    self,
    state_dict: dict,
    *,
    key: str,
    old_dims: dict[str, int],
    current_groups: tuple[str, ...],
    current_dims: dict[str, int],
  ) -> bool:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
      return False
    expected = sum(int(current_dims[g]) for g in current_groups)
    if value.shape[-1] == expected:
      return False
    state_dict[key] = self._remap_feature_columns(
      value,
      old_dims=old_dims,
      current_groups=current_groups,
      current_dims=current_dims,
      fill_value=0.0,
    )
    print(
      "[ResidualInteract] patched legacy linear input dim: "
      f"{key} {tuple(value.shape)} -> {tuple(state_dict[key].shape)}"
    )
    return True

  def _patch_legacy_input_state_dims(self, loaded_dict: dict) -> bool:
    old_feature_dims = loaded_dict.get("residual_feature_group_dims")
    if not isinstance(old_feature_dims, dict):
      return False
    changed = False
    old_feature_dims = {str(k): int(v) for k, v in old_feature_dims.items()}
    actor = cast(Any, self.alg.actor)
    actor_groups = tuple(str(g) for g in getattr(actor, "residual_feature_groups", ()))
    current_feature_dims = {
      str(k): int(v) for k, v in getattr(actor, "feature_group_dims", {}).items()
    }
    if actor_groups and current_feature_dims:
      actor_sd = loaded_dict.get("actor_state_dict", {})
      changed |= self._patch_normalizer_input_dim(
        actor_sd,
        prefix="obs_normalizer.",
        old_dims=old_feature_dims,
        current_groups=actor_groups,
        current_dims=current_feature_dims,
      )
      changed |= self._patch_normalizer_input_dim(
        actor_sd,
        prefix="hand_obs_normalizer.",
        old_dims=old_feature_dims,
        current_groups=actor_groups,
        current_dims=current_feature_dims,
      )
      changed |= self._patch_linear_input_dim(
        actor_sd,
        key="residual_mlp.0.weight",
        old_dims=old_feature_dims,
        current_groups=actor_groups,
        current_dims=current_feature_dims,
      )
      changed |= self._patch_linear_input_dim(
        actor_sd,
        key="hand_mlp.0.weight",
        old_dims=old_feature_dims,
        current_groups=actor_groups,
        current_dims=current_feature_dims,
      )

    critic = cast(Any, self.alg.critic)
    critic_groups = tuple(str(g) for g in getattr(critic, "obs_groups", ()))
    critic_dims = dict(current_feature_dims)
    critic_dims["sonic_obs_or_latent"] = mdp.OBS_DIM_NO_TEACHER
    for group in critic_groups:
      if group in critic_dims or group == "sonic_obs_or_latent":
        continue
      if group in old_feature_dims:
        critic_dims[group] = old_feature_dims[group]
    old_critic_dims = dict(old_feature_dims)
    old_critic_dims["sonic_obs_or_latent"] = mdp.OBS_DIM_NO_TEACHER
    unknown = [g for g in critic_groups if g not in critic_dims]
    if unknown:
      # Only the actor's groups have known widths here. A group the critic observes but the actor
      # does not (e.g. placement_goal once it is dropped from residual_feature_groups) has no width
      # to remap against, and inventing one would shift every later column.
      print(
        f"[ResidualInteract] skipping critic input-dim patch: no known width for "
        f"{unknown} (actor groups: {list(current_feature_dims)})"
      )
      critic_groups = ()
    if critic_groups:
      critic_sd = loaded_dict.get("critic_state_dict", {})
      changed |= self._patch_normalizer_input_dim(
        critic_sd,
        prefix="obs_normalizer.",
        old_dims=old_critic_dims,
        current_groups=critic_groups,
        current_dims=critic_dims,
      )
      changed |= self._patch_linear_input_dim(
        critic_sd,
        key="mlp.0.weight",
        old_dims=old_critic_dims,
        current_groups=critic_groups,
        current_dims=critic_dims,
      )
    return changed

  def _patch_actor_state_for_current_arch(self, loaded_dict: dict) -> bool:
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if not isinstance(actor_sd, dict):
      return False
    actor = cast(Any, self.alg.actor)
    current_sd = actor.state_dict()
    changed = False

    for key in list(actor_sd.keys()):
      if key not in current_sd:
        del actor_sd[key]
        changed = True
        print(f"[ResidualInteract] dropped actor checkpoint key: {key}")

    token_dim = 64
    for key, current in current_sd.items():
      loaded = actor_sd.get(key)
      if not isinstance(loaded, torch.Tensor):
        actor_sd[key] = current.detach().clone()
        changed = True
        print(f"[ResidualInteract] added actor checkpoint key: {key}")
        continue
      if tuple(loaded.shape) == tuple(current.shape):
        continue

      if key == "residual_mlp.6.weight" and loaded.ndim == 2 and current.ndim == 2:
        migrated = current.detach().clone()
        copy_rows = min(token_dim, loaded.shape[0], migrated.shape[0])
        copy_cols = min(loaded.shape[1], migrated.shape[1])
        migrated[:copy_rows, :copy_cols] = loaded[:copy_rows, :copy_cols]
        actor_sd[key] = migrated
        changed = True
        print(
          "[ResidualInteract] migrated residual output weight: "
          f"{tuple(loaded.shape)} -> {tuple(migrated.shape)}; "
          f"copied token rows={copy_rows}, cols={copy_cols}"
        )
        continue

      if key == "residual_mlp.6.bias" and loaded.ndim == 1 and current.ndim == 1:
        migrated = current.detach().clone()
        copy_rows = min(token_dim, loaded.shape[0], migrated.shape[0])
        migrated[:copy_rows] = loaded[:copy_rows]
        actor_sd[key] = migrated
        changed = True
        print(
          "[ResidualInteract] migrated residual output bias: "
          f"{tuple(loaded.shape)} -> {tuple(migrated.shape)}; "
          f"copied token rows={copy_rows}"
        )
        continue

      actor_sd[key] = current.detach().clone()
      changed = True
      print(
        "[ResidualInteract] reinitialized actor checkpoint key for current arch: "
        f"{key} {tuple(loaded.shape)} -> {tuple(current.shape)}"
      )

    loaded_dict["actor_state_dict"] = actor_sd
    return changed

  def _apply_loaded_residual_config(self, loaded_dict: dict) -> None:
    self._residual_gain = float(loaded_dict.get("residual_gain", self._residual_gain))
    self._body_residual_gain = loaded_dict.get(
      "body_residual_gain", self._body_residual_gain
    )
    self._hand_residual_gain = loaded_dict.get(
      "hand_residual_gain", self._hand_residual_gain
    )
    self._body_residual_gain = (
      None if self._body_residual_gain is None else float(self._body_residual_gain)
    )
    self._hand_residual_gain = (
      None if self._hand_residual_gain is None else float(self._hand_residual_gain)
    )
    self._residual_action_clip = float(
      loaded_dict.get("residual_action_clip", self._residual_action_clip)
    )
    self._token_residual_clip = float(
      loaded_dict.get("token_residual_clip", self._token_residual_clip)
    )
    self._token_residual_gain = float(
      loaded_dict.get("token_residual_gain", self._token_residual_gain)
    )
    self._ref_edit_clip = float(loaded_dict.get("ref_edit_clip", self._ref_edit_clip))
    self._ref_edit_groups = str(
      loaded_dict.get("ref_edit_groups", self._ref_edit_groups)
    )
    self._ref_edit_init_bias = str(
      loaded_dict.get("ref_edit_init_bias", self._ref_edit_init_bias)
    )
    self._split_hand_net = bool(loaded_dict.get("split_hand_net", self._split_hand_net))
    self._hand_primitive_mode = str(
      loaded_dict.get("hand_primitive_mode", self._hand_primitive_mode)
    )
    self._hand_primitive_close_frame = int(
      loaded_dict.get("hand_primitive_close_frame", self._hand_primitive_close_frame)
    )
    self._hand_primitive_open_frame = int(
      loaded_dict.get("hand_primitive_open_frame", self._hand_primitive_open_frame)
    )
    self._hand_primitive_hard_threshold = float(
      loaded_dict.get(
        "hand_primitive_hard_threshold", self._hand_primitive_hard_threshold
      )
    )
    self._hand_primitive_init_logit_bias = float(
      loaded_dict.get(
        "hand_primitive_init_logit_bias", self._hand_primitive_init_logit_bias
      )
    )
    self._hand_bc_base_start_frame = int(
      loaded_dict.get("hand_bc_base_start_frame", self._hand_bc_base_start_frame)
    )
    loaded_hand_bc_groups = loaded_dict.get(
      "hand_bc_feature_groups", self._hand_bc_feature_groups
    )
    self._hand_bc_feature_groups = _parse_groups(loaded_hand_bc_groups)
    if not self._hand_bc_feature_groups:
      self._hand_bc_feature_groups = self._residual_feature_groups
    loaded_hand_residual_start = loaded_dict.get(
      "hand_residual_start_frame", self._hand_residual_start_frame
    )
    self._hand_residual_start_frame = (
      None if loaded_hand_residual_start is None else int(loaded_hand_residual_start)
    )
    self._residual_start_frame = int(
      loaded_dict.get("residual_start_frame", self._residual_start_frame)
    )
    self._residual_ramp_frames = int(
      loaded_dict.get("residual_ramp_frames", self._residual_ramp_frames)
    )
    self._residual_lowpass_alpha = float(
      loaded_dict.get("residual_lowpass_alpha", self._residual_lowpass_alpha)
    )
    loaded_body_sample_delta_clip = loaded_dict.get(
      "body_sample_delta_clip", self._body_sample_delta_clip
    )
    self._body_sample_delta_clip = (
      None
      if loaded_body_sample_delta_clip is None
      or float(loaded_body_sample_delta_clip) <= 0.0
      else float(loaded_body_sample_delta_clip)
    )
    loaded_hand_sample_delta_clip = loaded_dict.get(
      "hand_sample_delta_clip", self._hand_sample_delta_clip
    )
    self._hand_sample_delta_clip = (
      None
      if loaded_hand_sample_delta_clip is None
      or float(loaded_hand_sample_delta_clip) <= 0.0
      else float(loaded_hand_sample_delta_clip)
    )
    self._final_action_clip = loaded_dict.get(
      "final_action_clip", self._final_action_clip
    )
    self._residual_mask = str(loaded_dict.get("residual_mask", self._residual_mask))
    self._apply_current_residual_config()
    print(
      "[ResidualInteract] loaded residual config: "
      f"gain={self._residual_gain}, body_gain={self._body_residual_gain}, "
      f"hand_gain={self._hand_residual_gain}, clip={self._residual_action_clip}, "
      f"token_gain={self._token_residual_gain}, "
      f"token_clip={self._token_residual_clip}, "
      f"split_hand_net={self._split_hand_net}, "
      f"fixed_hand_frame={self._fixed_hand_action_frame}, "
      f"hand_bc_base_start_frame={self._hand_bc_base_start_frame}, "
      f"hand_residual_start_frame={self._hand_residual_start_frame}, "
      f"residual_start_frame={self._residual_start_frame}, "
      f"residual_ramp_frames={self._residual_ramp_frames}, "
      f"residual_lowpass_alpha={self._residual_lowpass_alpha}, "
      f"body_sample_delta_clip={self._body_sample_delta_clip}, "
      f"hand_sample_delta_clip={self._hand_sample_delta_clip}, "
      f"hand_primitive={self._hand_primitive_mode}@"
      f"{self._hand_primitive_close_frame}, "
      f"hand_primitive_open_frame={self._hand_primitive_open_frame}, "
      f"hand_primitive_hard_threshold={self._hand_primitive_hard_threshold}, "
      f"hand_primitive_init_logit_bias={self._hand_primitive_init_logit_bias}, "
      f"mask={self._residual_mask}"
    )

  def _apply_current_residual_config(self) -> None:
    actor = cast(Any, self.alg.actor)
    if hasattr(actor, "residual_gain"):
      actor.residual_gain = self._residual_gain
    if hasattr(actor, "body_residual_gain"):
      actor.body_residual_gain = self._body_residual_gain
    if hasattr(actor, "hand_residual_gain"):
      actor.hand_residual_gain = self._hand_residual_gain
    if hasattr(actor, "residual_action_clip"):
      actor.residual_action_clip = self._residual_action_clip
    if hasattr(actor, "token_residual_clip"):
      actor.token_residual_clip = self._token_residual_clip
    if hasattr(actor, "token_residual_gain"):
      actor.token_residual_gain = self._token_residual_gain
    if hasattr(actor, "ref_edit_clip"):
      actor.ref_edit_clip = self._ref_edit_clip
    if hasattr(actor, "final_action_clip"):
      actor.final_action_clip = (
        None if self._final_action_clip is None else float(self._final_action_clip)
      )
    if hasattr(actor, "hand_primitive_mode"):
      actor.hand_primitive_mode = self._hand_primitive_mode
    if hasattr(actor, "hand_primitive_enabled"):
      actor.hand_primitive_enabled = self._hand_primitive_mode != "none"
    if hasattr(actor, "hand_primitive_grail_enabled"):
      actor.hand_primitive_grail_enabled = self._hand_primitive_mode.startswith(
        "grail_close_2d"
      )
    if hasattr(actor, "hand_primitive_hard_enabled"):
      actor.hand_primitive_hard_enabled = self._hand_primitive_mode.endswith("_hard")
    if hasattr(actor, "hand_primitive_close_frame"):
      actor.hand_primitive_close_frame = self._hand_primitive_close_frame
    if hasattr(actor, "hand_primitive_open_frame"):
      actor.hand_primitive_open_frame = self._hand_primitive_open_frame
    if hasattr(actor, "hand_primitive_hard_threshold"):
      actor.hand_primitive_hard_threshold = self._hand_primitive_hard_threshold
    if hasattr(actor, "hand_primitive_init_logit_bias"):
      actor.hand_primitive_init_logit_bias = self._hand_primitive_init_logit_bias
    if hasattr(actor, "hand_bc_base_start_frame"):
      actor.hand_bc_base_start_frame = max(int(self._hand_bc_base_start_frame), 0)
    if hasattr(actor, "hand_residual_start_frame"):
      actor.hand_residual_start_frame = (
        actor.hand_bc_base_start_frame
        if self._hand_residual_start_frame is None
        else max(int(self._hand_residual_start_frame), 0)
      )
    if hasattr(actor, "hand_sample_delta_clip"):
      actor.hand_sample_delta_clip = self._hand_sample_delta_clip
    if hasattr(actor, "body_sample_delta_clip"):
      actor.body_sample_delta_clip = self._body_sample_delta_clip
    if hasattr(actor, "residual_start_frame"):
      actor.residual_start_frame = max(int(self._residual_start_frame), 0)
    if hasattr(actor, "residual_ramp_frames"):
      actor.residual_ramp_frames = max(int(self._residual_ramp_frames), 0)
    if hasattr(actor, "residual_lowpass_alpha"):
      actor.residual_lowpass_alpha = min(
        max(float(self._residual_lowpass_alpha), 0.0), 1.0
      )
    if hasattr(actor, "residual_action_gain"):
      actor.residual_action_gain.copy_(
        _make_residual_gain(
          self._residual_gain,
          self._body_residual_gain,
          self._hand_residual_gain,
          device=actor.residual_action_gain.device,
        )
      )
    if hasattr(actor, "residual_mask"):
      actor.residual_mask.copy_(
        _make_residual_mask(self._residual_mask, device=actor.residual_mask.device)
      )

  def _aggregate_ep_extras(self) -> dict[str, float]:
    result: dict[str, float] = {}
    if not self.logger.ep_extras:
      return result
    keys = sorted({key for info in self.logger.ep_extras for key in info.keys()})
    for key in keys:
      values = []
      for info in self.logger.ep_extras:
        if key not in info:
          continue
        value = info[key]
        if not isinstance(value, torch.Tensor):
          value = torch.tensor([value], device=self.device)
        value = value.to(self.device)
        if value.dim() == 0:
          value = value.unsqueeze(0)
        values.append(value.float().reshape(-1))
      if values:
        mean = torch.nan_to_num(
          torch.cat(values).mean(), nan=0.0, posinf=0.0, neginf=0.0
        )
        result[key] = float(mean.item())
    return result

  def _seed_logger_ep_extra_union_keys(self) -> None:
    """Make rsl_rl's logger see reset-only keys such as Episode_Metrics/*."""
    if not self.logger.ep_extras:
      return
    keys = sorted({key for info in self.logger.ep_extras for key in info.keys()})
    seed = {key: torch.empty(0, device=self.device) for key in keys}
    self.logger.ep_extras.insert(0, seed)

  def _prepare_logger_ep_extras_for_wandb(self, snapshot: dict) -> None:
    if self._wandb_metric_mode == "full":
      self._seed_logger_ep_extra_union_keys()
      return
    keep_extra = (
      _keep_object_sweep_wandb_extra
      if self._wandb_metric_mode == "object_sweep"
      else _keep_compact_wandb_extra
    )

    compact_infos: list[dict] = []
    observed_keys: set[str] = set()
    for info in self.logger.ep_extras:
      compact = {key: value for key, value in info.items() if keep_extra(key)}
      if compact:
        observed_keys.update(compact.keys())
        compact_infos.append(compact)

    metrics = snapshot.get("metrics", {})
    seed: dict[str, torch.Tensor] = {
      key: torch.empty(0, device=self.device) for key in sorted(observed_keys)
    }
    for key in _COMPACT_WANDB_SNAPSHOT_KEYS:
      if key in metrics:
        seed[key] = torch.tensor([float(metrics[key])], device=self.device)

    if seed:
      compact_infos.insert(0, seed)
    self.logger.ep_extras = compact_infos

  def _training_snapshot(
    self,
    *,
    it: int,
    total_it: int,
    collect_time: float,
    learn_time: float,
    loss_dict: dict,
  ) -> dict:
    collection_size = (
      self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
    )
    iteration_time = collect_time + learn_time
    fps = float(collection_size / max(iteration_time, 1e-9))
    self._iteration_times.append(iteration_time)
    self._collection_times.append(collect_time)
    self._learning_times.append(learn_time)
    self._fps_values.append(fps)

    metrics = self._aggregate_ep_extras()
    if self._horizon_curriculum:
      metrics.update(self._horizon_metrics())
    if self._tracking_region_curriculum:
      metrics.update(self._tracking_region_curriculum_metrics())
    if len(self.logger.rewbuffer) > 0:
      metrics["Train/mean_reward"] = float(statistics.mean(self.logger.rewbuffer))
      metrics["Train/mean_episode_length"] = float(
        statistics.mean(self.logger.lenbuffer)
      )
    metrics["Policy/mean_std"] = float(self.alg.get_policy().output_std.mean().item())
    metrics["Perf/total_fps"] = fps
    metrics["Perf/collection_time"] = float(collect_time)
    metrics["Perf/learning_time"] = float(learn_time)
    for key, value in loss_dict.items():
      metrics[f"Loss/{key}"] = float(value)
    return {
      "iteration": it,
      "method_iteration": self._artifact_report_index(it),
      "rsl_iteration": it,
      "one_indexed_update": it + 1,
      "method_one_indexed_update": self._artifact_report_index(it) + 1,
      "total_iterations": total_it,
      "total_steps": int((it + 1) * collection_size),
      "metrics": metrics,
    }

  def _artifact_report_index(self, it: int) -> int:
    if self._load_artifact_history:
      return int(it)
    return max(0, int(it) - int(self._run_start_iteration))

  def _artifact_key(self, it: int, total_it: int) -> str:
    # The method log uses "1000 iterations" to mean 1000 completed PPO
    # updates, while RSL iteration indices are zero-based.  Preserve the
    # familiar model_0/model_100 intermediate convention, but record the final
    # completed update as 1000 instead of 999 for 1000-update runs.
    if not self._load_artifact_history:
      if it == total_it - 1 and self._iteration_times:
        return str(len(self._iteration_times))
      return str(self._artifact_report_index(it))
    if it == total_it - 1 and self._completed_iterations > 0:
      return str(self._completed_iterations)
    return str(it)

  def _final_artifact_iteration(self) -> int:
    return int(self._completed_iterations or self.current_learning_iteration)

  def _final_metric_key(self) -> str:
    if not self._load_artifact_history and self._iteration_times:
      return str(len(self._iteration_times))
    return str(self._final_artifact_iteration())

  def _final_metrics(self) -> dict:
    final_key = self._final_metric_key()
    if final_key not in self._train_metrics and self._latest_snapshot is not None:
      self._train_metrics[final_key] = self._latest_snapshot
    if final_key in self._train_metrics:
      return self._train_metrics[final_key].get("metrics", {})
    if self._train_metrics:
      latest_key = max(self._train_metrics.keys(), key=lambda k: int(k))
      return self._train_metrics[latest_key].get("metrics", {})
    return {}

  def _finalize_run_artifacts(self) -> None:
    final_iter = self._final_artifact_iteration()
    if self.logger.log_dir is None:
      raise RuntimeError("Logger writer is active but log_dir is not set.")
    final_model_path = os.path.join(self.logger.log_dir, f"model_{final_iter}.pt")
    self.save(final_model_path)
    if self._write_eval_artifacts:
      self._write_eval_artifacts_json()
    self._write_train_artifacts()
    self._write_acceptance_summary()
    self._append_method_log()

  def _run_dir(self) -> Path:
    if self.logger.log_dir is None:
      raise RuntimeError("ResidualInteract artifacts require a log directory.")
    return Path(self.logger.log_dir)

  def _gpu_label(self) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
      return f"CUDA_VISIBLE_DEVICES={visible}; process_device={self.device}"
    return f"process_device={self.device}"

  def _reward_weights(self) -> dict[str, float]:
    return {
      name: float(cfg.weight)
      for name, cfg in zip(
        self.env.unwrapped.reward_manager.active_terms,
        self.env.unwrapped.reward_manager._term_cfgs,
        strict=False,
      )
    }

  def _write_json(self, filename: str, payload: dict) -> None:
    path = self._run_dir() / filename
    path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

  def _write_train_artifacts(self) -> None:
    report_iters = [0, 100, 300, 600, 1000]
    if any(int(k) >= 1500 for k in self._train_metrics):
      report_iters.extend([1500, 2000, 3000])
    train_metrics = {
      "task_id": "Mjlab-ResidualInteract-G1",
      "run_dir": str(self._run_dir()),
      "report_iters": report_iters,
      "available_iters": sorted(int(k) for k in self._train_metrics.keys()),
      "missing_iters": [
        it for it in report_iters if str(it) not in self._train_metrics
      ],
      "metrics": self._train_metrics,
    }
    self._write_json("train_metrics_iter_0_100_300_600_1000.json", train_metrics)

    total_wall = (
      0.0 if self._run_wall_start is None else time.monotonic() - self._run_wall_start
    )
    steps_per_second = 0.0
    if self._iteration_times:
      total_steps = (
        len(self._iteration_times)
        * self.env.num_envs
        * self.cfg["num_steps_per_env"]
        * self.gpu_world_size
      )
      steps_per_second = total_steps / max(sum(self._iteration_times), 1e-9)
    time_summary = {
      "total_wall_time_s": float(total_wall),
      "scene_creation_time_s": float(
        os.environ.get("MJLAB_SCENE_CREATION_TIME_S", "0") or 0
      ),
      "mean_iteration_time_s": float(statistics.mean(self._iteration_times))
      if self._iteration_times
      else 0.0,
      "median_iteration_time_s": float(statistics.median(self._iteration_times))
      if self._iteration_times
      else 0.0,
      "mean_collection_time_s": float(statistics.mean(self._collection_times))
      if self._collection_times
      else 0.0,
      "mean_learning_time_s": float(statistics.mean(self._learning_times))
      if self._learning_times
      else 0.0,
      "steps_per_second": float(steps_per_second),
      "num_envs": int(self.env.num_envs),
      "num_steps_per_env": int(self.cfg["num_steps_per_env"]),
      "iterations_completed": int(len(self._iteration_times)),
      "gpu": self._gpu_label(),
    }
    self._write_json("train_time_summary.json", time_summary)

  def _accumulate_eval_log(
    self, accum: dict[str, list[float]], rewards: torch.Tensor, extras: dict
  ) -> None:
    accum.setdefault("Reward/total_from_env_reward", []).append(
      float(torch.nan_to_num(rewards.mean()).item())
    )
    for key, value in extras.get("log", {}).items():
      if not isinstance(value, torch.Tensor):
        value = torch.tensor([value], device=self.device)
      scalar = torch.nan_to_num(value.float().mean(), nan=0.0, posinf=0.0, neginf=0.0)
      accum.setdefault(key, []).append(float(scalar.item()))

  def _eval_step_log(
    self, step: int, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
  ) -> dict:
    record = {
      "step": int(step),
      "done_count": int(dones.sum().item()),
      "reward": float(torch.nan_to_num(rewards.float().mean()).item()),
    }
    detailed_keys = {
      "Metric/body_err_mse",
      "Metric/hand_err_mse",
      "Metric/body_xyz_err_mse",
      "Metric/body_link_dist_mean",
      "Metric/lower_body_xyz_err_mse",
      "Metric/upper_body_xyz_err_mse",
      "Metric/lower_body_link_dist_mean",
      "Metric/lower_wrist_link_dist_mean",
      "Metric/ankle_link_dist_mean",
      "Metric/ankle_wrist_link_dist_mean",
      "Metric/upper_body_link_dist_mean",
      "Metric/left_wrist_link_dist_mean",
      "Metric/right_wrist_link_dist_mean",
      "Metric/left_wrist_x_abs_err",
      "Metric/left_wrist_y_abs_err",
      "Metric/left_wrist_z_abs_err",
      "Metric/right_wrist_x_abs_err",
      "Metric/right_wrist_y_abs_err",
      "Metric/right_wrist_z_abs_err",
      "Metric/left_wrist_z_tracking_abs_err",
      "Metric/right_wrist_z_tracking_abs_err",
      "Metric/tracking_region_link_dist_mean",
      "Metric/tracking_region_xyz_err_mse",
      "Metric/raw_contact_arm_err_mse",
      "Metric/raw_contact_hand_err_mse",
      "Metric/raw_contact_object_pos_err",
      "Metric/raw_contact_window",
      "Metric/raw_tip_object_dist_err",
      "Metric/raw_tip_object_ref_dist",
      "Metric/raw_tip_object_dist_err_top1",
      "Metric/raw_tip_object_ref_dist_top1",
      "Metric/raw_tip_object_dist_err_top2",
      "Metric/raw_tip_object_ref_dist_top2",
      "Metric/raw_tip_object_dist_err_top4",
      "Metric/raw_tip_object_ref_dist_top4",
      "Metric/raw_tip_object_dist_err_cond_top1",
      "Metric/raw_tip_object_ref_dist_cond_top1",
      "Metric/raw_tip_object_dist_err_cond_top2",
      "Metric/raw_tip_object_ref_dist_cond_top2",
      "Metric/raw_tip_object_dist_err_cond_top4",
      "Metric/raw_tip_object_ref_dist_cond_top4",
      "Metric/raw_tip_radial_err",
      "Metric/raw_tip_radial_err_cond",
      "Metric/raw_tip_radial_live_dist_cond",
      "Metric/raw_tip_radial_ref_dist_cond",
      "Metric/hand_to_obj_dist",
      "Metric/obj_drift",
      "Metric/tracking_frame",
      "PhaseA/live_contact_006",
      "PhaseA/lift_duration_s",
      "PhaseA/lift_success",
      "PhaseA/ttr_at_012",
      "PhaseA/object_mpjpe_mm",
      "PhaseA/sequence_success",
      "Stage/stable_reach_030",
      "Stage/stable_reach_015",
      "Stage/stable_reach_005",
      "ResidualMetric/base_action_norm",
      "ResidualMetric/token_residual_norm",
      "ResidualMetric/token_residual_smooth_norm",
      "ResidualMetric/token_residual_clip_frac",
      "ResidualMetric/decoder_body_delta_norm",
      "ResidualMetric/decoder_body_delta_ratio",
      "ResidualMetric/decoder_body_delta_joint_rms",
      "ResidualMetric/decoder_body_delta_joint_abs_max",
      "ResidualMetric/hand_body_contact_frac",
      "ResidualMetric/non_tip_hand_body_contact_frac",
      "ResidualMetric/hand_close_left",
      "ResidualMetric/hand_close_right",
      "ResidualMetric/hand_close_mean",
      "ResidualMetric/hand_primitive_delta_norm",
      "ResidualMetric/body_sample_delta_pre_clip_norm",
      "ResidualMetric/body_sample_delta_post_clip_norm",
      "ResidualMetric/body_sample_clip_frac",
      "Stage/hand_body_contact",
      "Stage/non_tip_hand_body_contact",
      "Stage/stable_not_fallen",
    }
    detailed_keys.update(_ANKLE_WRIST_TRACKING_DETAIL_KEYS)
    for key, value in extras.get("log", {}).items():
      if not isinstance(value, torch.Tensor):
        value = torch.tensor([value], device=self.device)
      values = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
      scalar = values.mean()
      record[key] = float(scalar.item())
      flat = values.flatten()
      if key in detailed_keys and flat.numel() == self.env.num_envs:
        record[f"{key}/env0"] = float(flat[0].item())
        record[f"{key}/min"] = float(flat.min().item())
        record[f"{key}/max"] = float(flat.max().item())
        record[f"{key}/std"] = float(flat.std(unbiased=False).item())
    return record

  def _eval_policy(self, name: str, residual_gain: float, steps: int) -> dict:
    actor = cast(Any, self.alg.actor)
    old_gain = float(actor.residual_gain)
    old_action_gain = actor.residual_action_gain.detach().clone()
    old_token_gain = float(getattr(actor, "token_residual_gain", 0.0))
    was_training = actor.training
    actor.residual_gain = float(residual_gain)
    if residual_gain == 0.0:
      actor.residual_action_gain.zero_()
      if hasattr(actor, "token_residual_gain"):
        actor.token_residual_gain = 0.0
    self.alg.eval_mode()

    env = self.env.unwrapped

    def set_env_attr(name: str, value) -> None:
      setattr(env, name, value)

    old_forced_frame = getattr(env, "_force_reference_start_frame", None)
    old_detailed_log = getattr(env, "_eval_detailed_log", None)
    if self._eval_reference_start_frame is None:
      try:
        delattr(env, "_force_reference_start_frame")
      except AttributeError:
        pass
    else:
      set_env_attr("_force_reference_start_frame", self._eval_reference_start_frame)
    set_env_attr("_eval_detailed_log", True)
    try:
      obs, _ = self.env.reset()
      obs = obs.to(self.device)
      accum: dict[str, list[float]] = {}
      per_step: list[dict] = []
      done_count = 0
      first_done_step: int | None = None
      with torch.no_grad():
        for step in range(1, steps + 1):
          actions = actor(obs, stochastic_output=False)
          self._set_residual_action_stats(
            actor,
            actions,
            zero_residual=residual_gain == 0.0,
          )
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs = obs.to(self.device)
          step_done_count = int(dones.sum().item())
          done_count += step_done_count
          if step_done_count > 0 and first_done_step is None:
            first_done_step = step
          self._accumulate_eval_log(accum, rewards, extras)
          per_step.append(self._eval_step_log(step, rewards, dones, extras))
    finally:
      if old_forced_frame is None:
        try:
          delattr(env, "_force_reference_start_frame")
        except AttributeError:
          pass
      else:
        set_env_attr("_force_reference_start_frame", old_forced_frame)
      if old_detailed_log is None:
        try:
          delattr(env, "_eval_detailed_log")
        except AttributeError:
          pass
      else:
        set_env_attr("_eval_detailed_log", old_detailed_log)

    actor.residual_gain = old_gain
    actor.residual_action_gain.copy_(old_action_gain)
    if hasattr(actor, "token_residual_gain"):
      actor.token_residual_gain = old_token_gain
    if was_training:
      self.alg.train_mode()
    else:
      self.alg.eval_mode()

    metrics = {
      key: float(statistics.mean(values)) for key, values in accum.items() if values
    }
    return {
      "name": name,
      "task_id": "Mjlab-ResidualInteract-G1",
      "run_dir": str(self._run_dir()),
      "steps": int(steps),
      "num_envs": int(self.env.num_envs),
      "done_count": int(done_count),
      "first_done_step": first_done_step,
      "residual_gain": float(residual_gain),
      "reference_start_frame": self._eval_reference_start_frame,
      "base_tracker_kind": self._base_tracker_kind,
      "tracker_ckpt": self._tracker_ckpt,
      "sonic_encoder_onnx": self._sonic_encoder_onnx,
      "sonic_decoder_onnx": self._sonic_decoder_onnx,
      "astra_onnx_path": self._astra_onnx_path,
      "base_hand_mode": self._base_hand_mode,
      "metrics": metrics,
      "per_step": per_step,
    }

  def _write_eval_artifacts_json(self) -> None:
    tracker = self._eval_policy(
      "tracker_only", residual_gain=0.0, steps=self._eval_steps
    )
    residual = self._eval_policy(
      "residual", residual_gain=self._residual_gain, steps=self._eval_steps
    )
    self._write_json("eval_tracker_only.json", tracker)
    self._write_json("eval_residual.json", residual)

  def _write_acceptance_summary(self) -> None:
    run_dir = self._run_dir()
    required_files = [
      "train_time_summary.json",
      "train_metrics_iter_0_100_300_600_1000.json",
      "eval_tracker_only.json",
      "eval_residual.json",
      "residual_policy.pt",
      f"model_{self._final_artifact_iteration()}.pt",
    ]
    required_metric_keys = [
      "Reward/total",
      "ResidualReward/tracking",
      "ResidualReward/lower_tracking",
      "ResidualReward/lower_tracking_base",
      "ResidualReward/lower_tracking_close_bonus",
      "ResidualReward/lower_tracking_miss_penalty",
      "ResidualReward/ankle_wrist_tracking",
      "ResidualReward/ankle_wrist_tracking_base",
      "ResidualReward/ankle_wrist_tracking_close_bonus",
      "ResidualReward/ankle_wrist_tracking_miss_penalty",
      "ResidualReward/left_wrist_tracking",
      "ResidualReward/left_wrist_tracking_base",
      "ResidualReward/left_wrist_tracking_close_bonus",
      "ResidualReward/left_wrist_tracking_miss_penalty",
      "ResidualReward/right_wrist_tracking",
      "ResidualReward/right_wrist_tracking_base",
      "ResidualReward/right_wrist_tracking_close_bonus",
      "ResidualReward/right_wrist_tracking_miss_penalty",
      "ResidualReward/left_wrist_z_tracking",
      "ResidualReward/right_wrist_z_tracking",
      "ResidualReward/raw_contact_tracking",
      "ResidualReward/raw_tip_object_tracking",
      "ResidualReward/raw_tip_radial_tracking",
      "ResidualReward/task",
      "ResidualReward/grasp",
      "ResidualReward/placement",
      "ResidualReward/stability",
      "ResidualReward/token_l2",
      "ResidualReward/token_smooth",
      "ResidualReward/decoder_body_delta_l2",
      "ResidualReward/decoder_body_delta_norm_limit",
      "ResidualReward/decoder_body_delta_ratio_limit",
      "ResidualReward/decoder_body_delta_joint_limit",
      "ResidualMetric/residual_norm",
      "ResidualMetric/residual_base_ratio",
      "ResidualMetric/body_residual_norm",
      "ResidualMetric/hand_residual_norm",
      "ResidualMetric/leg_residual_norm",
      "ResidualMetric/arm_residual_norm",
      "ResidualMetric/body_residual_ratio",
      "ResidualMetric/astra_body_delta_norm",
      "ResidualMetric/astra_body_delta_ratio",
      "ResidualMetric/astra_body_delta_joint_rms",
      "ResidualMetric/hand_residual_ratio",
      "ResidualMetric/leg_residual_ratio",
      "ResidualMetric/arm_residual_ratio",
      "ResidualMetric/residual_clip_frac",
      "ResidualMetric/final_action_clip_frac",
      "ResidualMetric/body_final_delta_norm",
      "ResidualMetric/hand_final_delta_norm",
      "ResidualMetric/base_action_norm",
      "ResidualMetric/final_action_norm",
      "ResidualMetric/token_residual_norm",
      "ResidualMetric/token_residual_smooth_norm",
      "ResidualMetric/token_residual_clip_frac",
      "ResidualMetric/decoder_body_delta_norm",
      "ResidualMetric/decoder_body_delta_ratio",
      "ResidualMetric/decoder_body_delta_joint_rms",
      "ResidualMetric/decoder_body_delta_joint_abs_max",
      "ResidualMetric/contact_frac",
      "PhaseA/live_contact_006",
      "PhaseA/lift_duration_s",
      "PhaseA/lift_success",
      "PhaseA/ttr_at_012",
      "PhaseA/object_mpjpe_mm",
      "PhaseA/sequence_success",
      "ResidualMetric/hand_body_contact_frac",
      "ResidualMetric/non_tip_hand_body_contact_frac",
      "ResidualMetric/grasp_contact_frac",
      "ResidualMetric/object_motion_frac",
      "ResidualMetric/drift_gate",
      "ResidualMetric/placement_error",
      "Metric/body_err_mse",
      "Metric/hand_err_mse",
      "Metric/body_xyz_err_mse",
      "Metric/body_link_dist_mean",
      "Metric/lower_body_xyz_err_mse",
      "Metric/upper_body_xyz_err_mse",
      "Metric/lower_body_link_dist_mean",
      "Metric/lower_wrist_link_dist_mean",
      "Metric/ankle_link_dist_mean",
      "Metric/ankle_wrist_link_dist_mean",
      "Metric/upper_body_link_dist_mean",
      "Metric/left_wrist_link_dist_mean",
      "Metric/right_wrist_link_dist_mean",
      "Metric/left_wrist_x_abs_err",
      "Metric/left_wrist_y_abs_err",
      "Metric/left_wrist_z_abs_err",
      "Metric/right_wrist_x_abs_err",
      "Metric/right_wrist_y_abs_err",
      "Metric/right_wrist_z_abs_err",
      "Metric/left_wrist_z_tracking_abs_err",
      "Metric/right_wrist_z_tracking_abs_err",
      "Metric/tracking_region_link_dist_mean",
      "Metric/tracking_region_xyz_err_mse",
      "Metric/raw_contact_arm_err_mse",
      "Metric/raw_contact_hand_err_mse",
      "Metric/raw_contact_object_pos_err",
      "Metric/raw_contact_window",
      "Metric/raw_tip_object_dist_err",
      "Metric/raw_tip_object_ref_dist",
      "Metric/raw_tip_object_dist_err_top1",
      "Metric/raw_tip_object_ref_dist_top1",
      "Metric/raw_tip_object_dist_err_top2",
      "Metric/raw_tip_object_ref_dist_top2",
      "Metric/raw_tip_object_dist_err_top4",
      "Metric/raw_tip_object_ref_dist_top4",
      "Metric/raw_tip_object_dist_err_cond_top1",
      "Metric/raw_tip_object_ref_dist_cond_top1",
      "Metric/raw_tip_object_dist_err_cond_top2",
      "Metric/raw_tip_object_ref_dist_cond_top2",
      "Metric/raw_tip_object_dist_err_cond_top4",
      "Metric/raw_tip_object_ref_dist_cond_top4",
      "Metric/raw_tip_radial_err",
      "Metric/raw_tip_radial_err_cond",
      "Metric/raw_tip_radial_live_dist_cond",
      "Metric/raw_tip_radial_ref_dist_cond",
      "Metric/hand_to_obj_dist",
      "Metric/hand_to_obj_under_030_frac",
      "Metric/hand_to_obj_under_015_frac",
      "Metric/hand_to_obj_under_005_frac",
      "Metric/obj_drift",
      "Metric/obj_speed",
      "Metric/ep_len",
    ]
    required_metric_keys.extend(_ANKLE_WRIST_TRACKING_DETAIL_KEYS)
    final_metrics = self._final_metrics()
    summary = {
      "task_id": "Mjlab-ResidualInteract-G1",
      "run_dir": str(run_dir),
      "iterations_completed": int(len(self._iteration_times)),
      "final_rsl_iteration": int(self.current_learning_iteration),
      "final_artifact_iteration": int(self._final_artifact_iteration()),
      "gpu": self._gpu_label(),
      "base_tracker_kind": self._base_tracker_kind,
      "tracker_ckpt": self._tracker_ckpt,
      "sonic_encoder_onnx": self._sonic_encoder_onnx,
      "sonic_decoder_onnx": self._sonic_decoder_onnx,
      "astra_onnx_path": self._astra_onnx_path,
      "base_hand_mode": self._base_hand_mode,
      "tracker_checkpoint_metadata": self._tracker_checkpoint_metadata,
      "required_files": {name: (run_dir / name).exists() for name in required_files},
      "required_metrics_present_final": {
        key: key in final_metrics for key in required_metric_keys
      },
      "stage_flags_present_final": {
        key: f"Stage/{key}" in final_metrics
        for key in (
          "reach_030",
          "reach_015",
          "reach_005",
          "stable_reach_030",
          "stable_reach_015",
          "stable_reach_005",
          "near_contact",
          "physical_contact",
          "hand_body_contact",
          "non_tip_hand_body_contact",
          "force_close",
          "object_moving",
          "stable_not_fallen",
        )
      },
      "final_metrics": final_metrics,
      "reward_weights": self._reward_weights(),
      "residual": {
        "arch": self._residual_arch,
        "gain": self._residual_gain,
        "body_gain": self._body_residual_gain,
        "hand_gain": self._hand_residual_gain,
        "action_clip": self._residual_action_clip,
        "token_clip": self._token_residual_clip,
        "token_gain": self._token_residual_gain,
        "residual_start_frame": self._residual_start_frame,
        "residual_ramp_frames": self._residual_ramp_frames,
        "residual_lowpass_alpha": self._residual_lowpass_alpha,
        "body_sample_delta_clip": self._body_sample_delta_clip,
        "hand_sample_delta_clip": self._hand_sample_delta_clip,
        "mask": self._residual_mask,
        "feature_groups": self._residual_feature_groups,
        "split_hand_net": self._split_hand_net,
        "frame_hidden_dims": self._frame_hidden_dims,
        "body_init_std": self._body_init_std,
        "hand_init_std": self._hand_init_std,
        "disabled_init_std": self._disabled_init_std,
        "zero_init_residual": self._zero_init_residual,
      },
      "horizon_curriculum": {
        "enabled": self._horizon_curriculum,
        "start_ref_frames": self._horizon_start_ref_frames,
        "final_ref_frames": self._horizon_final_ref_frames,
        "increment_ref_frames": self._horizon_increment_ref_frames,
        "success_threshold": self._horizon_success_threshold,
        "success_patience": self._horizon_success_patience,
        "metric": self._horizon_metric,
        "startup_steps": self._horizon_startup_steps,
        "min_iterations_per_stage": self._horizon_min_iterations_per_stage,
        "current_ref_frames": self._horizon_current_ref_frames,
        "current_steps": self._horizon_current_steps,
        "full_steps": self._horizon_full_steps,
        "success_streak": self._horizon_success_streak,
        "last_metric": self._horizon_last_metric,
        "last_passed": self._horizon_last_passed,
        "last_advanced": self._horizon_last_advanced,
        "last_advance_iteration": self._horizon_last_advance_iteration,
        "stage_start_iteration": self._horizon_stage_start_iteration,
        "stage_age_iterations": self._horizon_stage_age_iterations,
        "last_stage_ready": self._horizon_last_stage_ready,
      },
      "eval_reference_start_frame": self._eval_reference_start_frame,
      "train_command": " ".join(sys.argv),
    }
    self._write_json("acceptance_summary.json", summary)

  def _append_method_log(self) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    log_path = repo_root / "MJLAB_RESIDUAL_METHOD_LOG.md"
    if log_path.exists():
      existing = log_path.read_text(encoding="utf-8")
      method_idx = (
        existing.count("\n## M") + (1 if existing.startswith("## M") else 0) + 1
      )
    else:
      existing = ""
      method_idx = 1
    method_name = self.cfg.get("run_name") or self._run_dir().name
    final_metrics = self._final_metrics()
    eval_residual_path = self._run_dir() / "eval_residual.json"
    eval_tracker_path = self._run_dir() / "eval_tracker_only.json"
    entry = [
      f"## M{method_idx:03d} - {method_name}",
      "",
      f"Date/time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
      f"GPU: {self._gpu_label()}",
      f"Git diff / commit: see {self._run_dir() / 'git'}",
      f"Base tracker kind: {self._base_tracker_kind}",
      f"Base SONIC tracker checkpoint: {self._tracker_ckpt}",
      f"Official SONIC encoder ONNX: {self._sonic_encoder_onnx}",
      f"Official SONIC decoder ONNX: {self._sonic_decoder_onnx}",
      f"ASTRA ONNX: {self._astra_onnx_path}",
      f"Base hand mode: {self._base_hand_mode}",
      f"Residual checkpoint init: random residual head; tracker init={self._init_from}",
      f"Command: {' '.join(sys.argv)}",
      f"Iterations planned: {self.cfg.get('max_iterations')}",
      f"Iterations completed: {len(self._iteration_times)}",
      f"Run directory: {self._run_dir()}",
      "",
      (
        "Hypothesis: preserve SONIC tracking while residual exploration "
        "improves reach/contact/object motion."
      ),
      f"Change from previous method: {method_name}",
      f"Reward weights: {json.dumps(self._reward_weights(), sort_keys=True)}",
      (
        "Residual action mask/gain/clip: "
        f"{self._residual_mask} / {self._residual_gain} "
        f"(body={self._body_residual_gain}, hand={self._hand_residual_gain}) / "
        f"{self._residual_action_clip}"
      ),
      (
        "Token residual gain/clip: "
        f"{self._token_residual_gain} / {self._token_residual_clip}"
      ),
      f"Split hand net: {self._split_hand_net}",
      (
        "Exploration/noise settings: "
        f"{self.cfg.get('actor', {}).get('distribution_cfg')}; "
        f"body_init_std={self._body_init_std}, "
        f"hand_init_std={self._hand_init_std}, "
        f"disabled_init_std={self._disabled_init_std}, "
        f"zero_init_residual={self._zero_init_residual}"
      ),
      (
        "Reset/curriculum settings: training reset may sample the late contact "
        f"window; eval reference frame={self._eval_reference_start_frame}."
      ),
      "",
      "Metrics:",
    ]
    for report_it in (0, 100, 300, 600, 1000, 1500, 2000, 3000):
      snap = self._train_metrics.get(str(report_it))
      if snap is None:
        entry.append(f"- iter {report_it}: not reached/not captured")
      else:
        metrics = snap.get("metrics", {})
        compact = {
          key: metrics.get(key)
          for key in (
            "Reward/total",
            "ResidualReward/tracking",
            "ResidualReward/lower_tracking",
            "ResidualReward/lower_tracking_base",
            "ResidualReward/lower_tracking_close_bonus",
            "ResidualReward/ankle_wrist_tracking",
            "ResidualReward/ankle_wrist_tracking_base",
            "ResidualReward/ankle_wrist_tracking_close_bonus",
            "ResidualReward/left_wrist_tracking",
            "ResidualReward/right_wrist_tracking",
            "ResidualReward/left_wrist_z_tracking",
            "ResidualReward/right_wrist_z_tracking",
            "ResidualReward/raw_contact_tracking",
            "ResidualReward/raw_tip_object_tracking",
            "ResidualReward/raw_tip_radial_tracking",
            "ResidualReward/surface_contact",
            "ResidualReward/multi_tip_surface",
            "ResidualReward/object_drift_limit",
            "ResidualReward/contact_duration",
            "Metric/hand_to_obj_dist",
            "Metric/raw_contact_arm_err_mse",
            "Metric/raw_contact_hand_err_mse",
            "Metric/raw_contact_object_pos_err",
            "Metric/raw_contact_window",
            "Metric/raw_tip_object_dist_err",
            "Metric/raw_tip_object_ref_dist",
            "Metric/raw_tip_object_dist_err_top1",
            "Metric/raw_tip_object_ref_dist_top1",
            "Metric/raw_tip_object_dist_err_top2",
            "Metric/raw_tip_object_ref_dist_top2",
            "Metric/raw_tip_object_dist_err_top4",
            "Metric/raw_tip_object_ref_dist_top4",
            "Metric/raw_tip_object_dist_err_cond_top1",
            "Metric/raw_tip_object_ref_dist_cond_top1",
            "Metric/raw_tip_object_dist_err_cond_top2",
            "Metric/raw_tip_object_ref_dist_cond_top2",
            "Metric/raw_tip_object_dist_err_cond_top4",
            "Metric/raw_tip_object_ref_dist_cond_top4",
            "Metric/raw_tip_radial_err",
            "Metric/raw_tip_radial_err_cond",
            "Metric/raw_tip_radial_live_dist_cond",
            "Metric/raw_tip_radial_ref_dist_cond",
            "Metric/lower_body_link_dist_mean",
            "Metric/lower_wrist_link_dist_mean",
            "Metric/ankle_link_dist_mean",
            "Metric/ankle_wrist_link_dist_mean",
            "Metric/left_wrist_link_dist_mean",
            "Metric/right_wrist_link_dist_mean",
            "Metric/left_wrist_x_abs_err",
            "Metric/left_wrist_y_abs_err",
            "Metric/left_wrist_z_abs_err",
            "Metric/right_wrist_x_abs_err",
            "Metric/right_wrist_y_abs_err",
            "Metric/right_wrist_z_abs_err",
            "Metric/left_wrist_z_tracking_abs_err",
            "Metric/right_wrist_z_tracking_abs_err",
            "ResidualMetric/contact_frac",
            "PhaseA/live_contact_006",
            "PhaseA/lift_duration_s",
            "PhaseA/lift_success",
            "PhaseA/ttr_at_012",
            "PhaseA/object_mpjpe_mm",
            "PhaseA/sequence_success",
            "ResidualMetric/hand_body_contact_frac",
            "ResidualMetric/non_tip_hand_body_contact_frac",
            "ResidualMetric/surface_contact_frac",
            "ResidualMetric/multi_tip_near_frac",
            "ResidualMetric/contact_duration_max",
            "ResidualMetric/contact_duration_frac",
            "ResidualMetric/contact_duration_active_mean",
            "ResidualMetric/grasp_contact_frac",
            "ResidualMetric/object_motion_frac",
            "ResidualMetric/drift_gate",
            "ResidualMetric/residual_base_ratio",
            "ResidualMetric/body_residual_ratio",
            "ResidualMetric/astra_body_delta_norm",
            "ResidualMetric/astra_body_delta_ratio",
            "ResidualMetric/astra_body_delta_joint_rms",
            "ResidualMetric/hand_residual_ratio",
            "ResidualMetric/hand_close_left",
            "ResidualMetric/hand_close_right",
            "ResidualMetric/hand_close_mean",
            "ResidualMetric/hand_primitive_delta_norm",
            "ResidualMetric/residual_clip_frac",
            "ResidualMetric/token_residual_norm",
            "ResidualMetric/token_residual_smooth_norm",
            "ResidualMetric/token_residual_clip_frac",
            "ResidualMetric/decoder_body_delta_norm",
            "ResidualMetric/decoder_body_delta_ratio",
            "ResidualMetric/decoder_body_delta_joint_rms",
            "ResidualMetric/decoder_body_delta_joint_abs_max",
          )
        }
        compact.update(
          {key: metrics.get(key) for key in _ANKLE_WRIST_TRACKING_DETAIL_KEYS}
        )
        entry.append(f"- iter {report_it}: {json.dumps(compact, sort_keys=True)}")
    entry.extend(
      [
        "",
        "Eval:",
        "- tracker-only: "
        f"{eval_tracker_path if eval_tracker_path.exists() else 'missing'}",
        "- residual: "
        f"{eval_residual_path if eval_residual_path.exists() else 'missing'}",
        "",
        f"Read: final metrics {json.dumps(final_metrics, sort_keys=True)[:2000]}",
        (
          "Decision: inspect standardized physical metrics and "
          "continue/adjust per objective."
        ),
        "Next method: TBD from contact, reach, drift, and tracking tradeoff.",
        "",
      ]
    )
    prefix = "# MJLab Residual Method Log\n\n" if not existing else ""
    with log_path.open("a", encoding="utf-8") as f:
      if prefix:
        f.write(prefix)
      f.write("\n".join(entry))
      f.write("\n")

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    actor = cast(Any, self.alg.actor)
    reward_weights = {
      name: cfg.weight
      for name, cfg in zip(
        self.env.unwrapped.reward_manager.active_terms,
        self.env.unwrapped.reward_manager._term_cfgs,
        strict=False,
      )
    }
    saved_dict.update(
      {
        "iter": self.current_learning_iteration,
        "completed_iterations": self._final_artifact_iteration(),
        "infos": infos,
        "base_tracker_kind": self._base_tracker_kind,
        "tracker_ckpt": self._tracker_ckpt,
        "sonic_encoder_onnx": self._sonic_encoder_onnx,
        "sonic_decoder_onnx": self._sonic_decoder_onnx,
        "astra_onnx_path": self._astra_onnx_path,
        "base_hand_mode": self._base_hand_mode,
        "tracker_checkpoint_metadata": self._tracker_checkpoint_metadata,
        "residual_actor_state_dict": actor.residual_state_dict(),
        "critic_state_dict": self.alg.critic.state_dict(),
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "obs_dim": getattr(actor, "obs_dim", None),
        "residual_feature_group_dims": getattr(actor, "feature_group_dims", {}),
        "action_dim": mdp.ACTION_DIM,
        "residual_arch": self._residual_arch,
        "critic_feature_groups": self._critic_feature_groups,
        "frame_hidden_dims": self._frame_hidden_dims,
        "residual_gain": self._residual_gain,
        "body_residual_gain": self._body_residual_gain,
        "hand_residual_gain": self._hand_residual_gain,
        "residual_action_clip": self._residual_action_clip,
        "token_residual_clip": self._token_residual_clip,
        "token_residual_gain": self._token_residual_gain,
        "ref_edit_clip": self._ref_edit_clip,
        "ref_edit_groups": self._ref_edit_groups,
        "ref_edit_init_bias": self._ref_edit_init_bias,
        "split_hand_net": self._split_hand_net,
        "hand_bc_checkpoint": self._hand_bc_checkpoint,
        "hand_bc_feature_groups": self._hand_bc_feature_groups,
        "freeze_hand_bc": self._freeze_hand_bc,
        "hand_bc_action_clip": self._hand_bc_action_clip,
        "hand_bc_base_start_frame": self._hand_bc_base_start_frame,
        "hand_residual_start_frame": self._hand_residual_start_frame,
        "residual_start_frame": self._residual_start_frame,
        "residual_ramp_frames": self._residual_ramp_frames,
        "residual_lowpass_alpha": self._residual_lowpass_alpha,
        "body_sample_delta_clip": self._body_sample_delta_clip,
        "hand_sample_delta_clip": self._hand_sample_delta_clip,
        "hand_primitive_mode": self._hand_primitive_mode,
        "hand_primitive_close_frame": self._hand_primitive_close_frame,
        "hand_primitive_open_frame": self._hand_primitive_open_frame,
        "hand_primitive_hard_threshold": self._hand_primitive_hard_threshold,
        "hand_primitive_init_logit_bias": self._hand_primitive_init_logit_bias,
        "residual_mask": self._residual_mask,
        "body_init_std": self._body_init_std,
        "hand_init_std": self._hand_init_std,
        "disabled_init_std": self._disabled_init_std,
        "zero_init_residual": self._zero_init_residual,
        "tracking_region_curriculum": {
          "enabled": self._tracking_region_curriculum,
          "initial": self._tracking_region_initial,
          "after": self._tracking_region_after,
          "metric": self._tracking_region_metric,
          "success_threshold": self._tracking_region_success_threshold,
          "success_patience": self._tracking_region_success_patience,
          "min_horizon_ref_frames": self._tracking_region_min_horizon_ref_frames,
          "current": self._tracking_region_current,
          "success_streak": self._tracking_region_success_streak,
          "last_metric": self._tracking_region_last_metric,
          "last_passed": self._tracking_region_last_passed,
          "last_horizon_ready": self._tracking_region_last_horizon_ready,
          "last_advanced": self._tracking_region_last_advanced,
          "advance_iteration": self._tracking_region_advance_iteration,
        },
        "reward_weights": reward_weights,
        "task_id": "Mjlab-ResidualInteract-G1",
        "train_command": " ".join(sys.argv),
        "train_metrics": self._train_metrics,
        "iteration_times": self._iteration_times,
        "collection_times": self._collection_times,
        "learning_times": self._learning_times,
        "fps_values": self._fps_values,
      }
    )
    torch.save(saved_dict, path)
    self._save_residual_policy(Path(path).parent)
    self._prune_old_checkpoints(Path(path).parent)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  # Keep the N newest checkpoints plus every MILESTONE_EVERY-th iteration; drop the rest.
  # Without this, 10-iteration saves of 51-128 MB files filled vast-1's 100 GB disk and
  # killed two runs mid-save.
  CKPT_KEEP_RECENT = int(os.environ.get("MJLAB_CKPT_KEEP_RECENT", "5"))
  CKPT_MILESTONE_EVERY = int(os.environ.get("MJLAB_CKPT_MILESTONE_EVERY", "500"))

  def _prune_old_checkpoints(self, run_dir: Path) -> None:
    try:
      ckpts = []
      for f in run_dir.glob("model_*.pt"):
        stem = f.stem[len("model_"):]
        if stem.isdigit():
          ckpts.append((int(stem), f))
      if not ckpts:
        return
      ckpts.sort()
      keep = {f for _, f in ckpts[-max(self.CKPT_KEEP_RECENT, 1):]}
      if self.CKPT_MILESTONE_EVERY > 0:
        keep |= {f for it, f in ckpts if it % self.CKPT_MILESTONE_EVERY == 0}
      for _, f in ckpts:
        if f not in keep:
          f.unlink(missing_ok=True)
    except Exception as exc:  # pragma: no cover - cleanup must never kill training
      print(f"[WARN] checkpoint prune skipped: {exc}")

  def _save_residual_policy(self, run_dir: Path) -> None:
    actor = cast(Any, self.alg.actor)
    payload = {
      "base_tracker_kind": self._base_tracker_kind,
      "tracker_ckpt": self._tracker_ckpt,
      "sonic_encoder_onnx": self._sonic_encoder_onnx,
      "sonic_decoder_onnx": self._sonic_decoder_onnx,
      "astra_onnx_path": self._astra_onnx_path,
      "base_hand_mode": self._base_hand_mode,
      "tracker_checkpoint_metadata": self._tracker_checkpoint_metadata,
      "residual_actor_state_dict": actor.residual_state_dict(),
      "obs_dim": getattr(actor, "obs_dim", None),
      "residual_feature_group_dims": getattr(actor, "feature_group_dims", {}),
      "action_dim": mdp.ACTION_DIM,
      "residual_arch": self._residual_arch,
      "critic_feature_groups": self._critic_feature_groups,
      "split_hand_net": self._split_hand_net,
      "frame_hidden_dims": self._frame_hidden_dims,
      "residual_gain": self._residual_gain,
      "body_residual_gain": self._body_residual_gain,
      "hand_residual_gain": self._hand_residual_gain,
      "hand_bc_checkpoint": self._hand_bc_checkpoint,
      "hand_bc_feature_groups": self._hand_bc_feature_groups,
      "freeze_hand_bc": self._freeze_hand_bc,
      "hand_bc_action_clip": self._hand_bc_action_clip,
      "hand_bc_base_start_frame": self._hand_bc_base_start_frame,
      "hand_residual_start_frame": self._hand_residual_start_frame,
      "residual_start_frame": self._residual_start_frame,
      "residual_ramp_frames": self._residual_ramp_frames,
      "residual_lowpass_alpha": self._residual_lowpass_alpha,
      "body_sample_delta_clip": self._body_sample_delta_clip,
      "hand_sample_delta_clip": self._hand_sample_delta_clip,
      "fixed_hand_action_frame": self._fixed_hand_action_frame,
      "residual_action_clip": self._residual_action_clip,
      "token_residual_clip": self._token_residual_clip,
      "token_residual_gain": self._token_residual_gain,
      "ref_edit_clip": self._ref_edit_clip,
      "ref_edit_groups": self._ref_edit_groups,
      "ref_edit_init_bias": self._ref_edit_init_bias,
      "residual_mask": self._residual_mask,
      "hand_primitive_mode": self._hand_primitive_mode,
      "hand_primitive_close_frame": self._hand_primitive_close_frame,
      "hand_primitive_open_frame": self._hand_primitive_open_frame,
      "hand_primitive_hard_threshold": self._hand_primitive_hard_threshold,
      "hand_primitive_init_logit_bias": self._hand_primitive_init_logit_bias,
      "residual_feature_groups": self._residual_feature_groups,
      "ref_preview_steps": self._ref_preview_steps,
      "body_init_std": self._body_init_std,
      "hand_init_std": self._hand_init_std,
      "disabled_init_std": self._disabled_init_std,
      "zero_init_residual": self._zero_init_residual,
      "task_id": "Mjlab-ResidualInteract-G1",
    }
    torch.save(payload, run_dir / "residual_policy.pt")
