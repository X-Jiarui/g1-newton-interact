"""Rewards staged on the dataset's own contact frame.

The reference is now accurate in two different senses in two different places, and one reward cannot
serve both:

  before cf   the retarget has been re-solved so the hand arrives at the object the way the human's
              did -- fingertips on the points the human touched, palm oriented like the human's.
              That is worth tracking, and it is the only phase where the arm configuration is
              meaningful data rather than one of many ways to hold a thing.

  after cf    the task is where the OBJECT goes. The arm configuration is a means. Tracking it
              fights the object reward whenever the reference's carry is imperfect, which it is --
              the grasp is solved at one frame and carried by a constant joint offset.

So: track the upper body and fingertips before cf, drop to a small weight after, and let the object
trajectory carry the second half. The lower body is tracked throughout -- it is GMR's most reliable
output (feet carry the highest IK weights) and the upper body needs a stable base once it is free.

`cf` is the dataset's own first-contact label, already resampled to the reference's frame rate by
`_load_ref_single` and kept per clip by `_load_ref_mix`. Nothing here re-derives it.
"""

from __future__ import annotations

import os

import torch

from mjlab.tasks.apple_eat import mdp as apple_mdp
from mjlab.tasks.residual_interact import mdp


def cf_local(env) -> torch.Tensor:
    """This env's contact frame, as a CLIP-LOCAL index.

    Under MIX, clip c frame f lives at global row `c * n_frames + f`, so a gate written against the
    global row fires immediately for every clip after the first -- clip 1's smallest row is already
    `n_frames`. The clip id is recovered from the env's own clip bounds and used to pick that clip's
    label.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    lo, hi = apple_mdp._clip_bounds(env, n)
    gt = ref.get("gt_contact_frames") or []
    if not gt:
        return torch.full_like(lo, -1)
    table = torch.tensor([int(x) for x in gt], dtype=torch.long, device=env.device)
    clip = (lo // n).clamp(0, table.numel() - 1)
    cf = table.index_select(0, clip)
    # Tensor bound: `.clamp(0, tensor)` is not a legal overload, and clamping the low end to
    # 0 would also erase the -1 that means `this clip has no contact label`.
    return torch.minimum(cf, hi - lo).clamp_min(-1)


def before_cf(env) -> torch.Tensor:
    """True while the clip has not yet reached its own contact frame."""
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    frame = apple_mdp.local_tracking_frame(env, n)
    frame = frame.round().long() if frame.dtype.is_floating_point else frame.long()
    cf = cf_local(env)
    return (cf >= 0) & (frame < cf)


def after_cf(env, offset: int = 0) -> torch.Tensor:
    """True once the clip has reached its own contact frame, plus `offset` frames.

    The counterpart of `before_cf`, for terms that should only open once the grasp exists. Compared
    CLIP-LOCALLY: `_tracking_frame` is a global row under MIX, so `frame >= <constant>` is already
    true for every clip after the first on its very first step.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    frame = apple_mdp.local_tracking_frame(env, n)
    frame = frame.round().long() if frame.dtype.is_floating_point else frame.long()
    cf = cf_local(env)
    return (cf >= 0) & (frame >= (cf + int(offset)))


def _phase_weight(env, pre: float, post: float) -> torch.Tensor:
    pre_mask = before_cf(env)
    # PHASE_GATE_LOG -- logged here, not from a metrics term: every staged term passes through
    # this, so the gate stays visible even when a term is unweighted. cf of -1, 0 or n_frames
    # means the split is degenerate and every staged term is training on a broken phase.
    mdp._safe_log(env, "Metric/cf_frame", cf_local(env).float())
    mdp._safe_log(env, "Metric/before_cf", pre_mask.float())
    return torch.where(
        pre_mask,
        torch.full_like(pre_mask, float(pre), dtype=torch.float32),
        torch.full_like(pre_mask, float(post), dtype=torch.float32),
    )


def staged_link_tracking_reward(
    env,
    link_group: str = "upper",
    distance_std: float = 0.05,
    pre_weight: float = 1.0,
    post_weight: float = 0.0,
    log_prefix: str = "staged_upper",
) -> torch.Tensor:
    """Gaussian on the reference-link distance for one group, scaled by phase.

    `post_weight` is deliberately a weight and not a hard zero. Omnigrasp can drop its body term to
    exactly zero because its action space is a pretrained motion latent that cannot produce a
    non-human pose; ours is joint residuals on top of a frozen tracker, so nothing structurally
    prevents the arm from windmilling as long as the object goes where it should. A small value here
    is the cheap insurance; sweep it from zero rather than to it.
    """
    dist = mdp._body_link_dist_mean_for_group(env, link_group)
    std = max(float(distance_std), 1.0e-6)
    base = torch.exp(-0.5 * (dist / std).pow(2))
    value = base * mdp._active_after_startup(env) * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    mdp._safe_log(env, f"Metric/{log_prefix}_dist", dist)
    return value


def staged_object_tracking_reward(
    env,
    distance_std: float = 0.08,
    pre_weight: float = 0.0,
    post_weight: float = 1.0,
    require_contact: bool = True,
    contact_threshold: float = 0.02,   # metres to the object SURFACE
    object_radius: float = float(mdp.APPLE_RADIUS),
    log_prefix: str = "staged_object",
) -> torch.Tensor:
    """Object position against its reference, active after cf and gated on real contact.

    The gate is Omnigrasp's one genuinely load-bearing idea (`check_contact`): without it a policy
    collects the object reward for standing next to an object that is where the reference says
    because nobody moved it. Ours is simpler -- a fingertip within `contact_threshold` of the
    object -- because our reference object is not free to drift far.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    frame = apple_mdp.local_tracking_frame(env, n)
    frame_i = frame.round().long() if frame.dtype.is_floating_point else frame.long()
    lo, hi = apple_mdp._clip_bounds(env, n)
    row = (lo + frame_i).clamp(lo, hi)

    obj = mdp.object_pool.active(env)
    live = obj.data.root_link_pos_w
    target = mdp._reference_object_pos_w(env, ref, row)
    dist = (live - target).norm(dim=-1)
    std = max(float(distance_std), 1.0e-6)
    base = torch.exp(-0.5 * (dist / std).pow(2))

    if require_contact:
        # Distance to the object's SURFACE, not its centre: `_tip_distances` is tip-to-root, and a
        # fixed threshold on that carries the object's own size -- 2 cm to the centre of a 4.3 cm
        # apple is 2 cm INSIDE it.
        #
        # KNOWN LIMITATION, inherited rather than introduced: this task has no per-object size
        # anywhere. Every existing contact reward subtracts the module-level APPLE_RADIUS, so on a
        # mixed set the threshold means something different for every object. `object_radius` is
        # exposed here so a single-object run can at least be told the truth; a per-object radius
        # read from the model is the real fix and belongs in mdp.py, not here.
        surface = mdp._tip_distances(env) - float(object_radius)
        touching = (surface <= float(contact_threshold)).any(dim=-1)
        base = base * touching.float()
        mdp._safe_log(env, f"Metric/{log_prefix}_touching", touching.float())
        mdp._safe_log(env, f"Metric/{log_prefix}_tip_surface", surface.min(dim=-1).values)

    value = base * mdp._active_after_startup(env) * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    mdp._safe_log(env, f"Metric/{log_prefix}_dist", dist)
    return value


def _ref_hand(env, frame: torch.Tensor) -> torch.Tensor:
    """The reference's hand joints at `frame`, [num_envs, NUM_HAND]."""
    ref = mdp._ref(env.device)
    nb, nh = apple_mdp.NUM_BODY, apple_mdp.NUM_HAND
    return ref["dof_pos"][frame, nb : nb + nh]


def staged_hand_pose_tracking_reward(
    env,
    angle_std: float = 0.25,          # radians
    pre_weight: float = 1.0,
    post_weight: float = 0.2,
    log_prefix: str = "staged_hand_pose",
) -> torch.Tensor:
    """Track the reference's HAND JOINTS -- the shape solved at cf and ramped in before it.

    This is the term that spends the accuracy the retarget now has. Step 5 solves the 20 finger
    joints of the operating hand against the human's own fingertips taken from the SMPL-X mesh, and
    step 6 ramps them in over the 30 frames before cf; so the reference hand is open while reaching
    and closed on the object at cf, and reproducing that trajectory IS reproducing the grasp.

    Joint space rather than fingertip positions, for two reasons: it needs no forward kinematics in
    the reward loop, and the quantity the retarget actually solved for is the joint vector -- a
    fingertip target would re-introduce the size mismatch between the Wuji hand and the human's that
    step 5 exists to absorb.

    `post_weight` is small rather than zero: after cf the hand should stay closed, and letting the
    term vanish entirely invites the policy to open it the moment the object reward takes over.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    frame = mdp._tracking_frame(env, n)
    frame = frame.round().long() if frame.dtype.is_floating_point else frame.long()
    robot = env.scene["robot"]
    nb, nh = apple_mdp.NUM_BODY, apple_mdp.NUM_HAND
    live = robot.data.joint_pos[:, nb : nb + nh]
    err = (live - _ref_hand(env, frame)).abs()
    std = max(float(angle_std), 1.0e-6)
    base = torch.exp(-0.5 * (err.mean(dim=-1) / std).pow(2))
    value = base * mdp._active_after_startup(env) * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    mdp._safe_log(env, f"Metric/{log_prefix}_err_deg", err.mean(dim=-1) * 57.29578)
    return value


def staged_tip_object_reward(
    env,
    distance_std: float = 0.20,
    near_threshold: float = 0.25,
    pre_weight: float = 1.0,
    post_weight: float = 0.3,
    log_prefix: str = "staged_tip_object",
) -> torch.Tensor:
    """Fingertips toward the object, counting only the tips the REFERENCE brings near it.

    Omnigrasp's `close_hand_flag`, which is the one piece of its pregrasp reward worth copying: a
    hand part is scored only if the reference has it within `near_threshold` of the object. Without
    that mask every finger is asked to reach, including the ones the human never used, and the
    policy is pushed toward a five-finger cage on objects that were pinched.

    Shaping, not a gate: the value falls off as a Gaussian in distance, so a tip that is close but
    not touching still earns. Our reward for actually holding on is the object term, which is gated.
    """
    dist = mdp._tip_distances(env)                     # [num_envs, n_tips], to the object centre
    near = dist < float(near_threshold)
    std = max(float(distance_std), 1.0e-6)
    per_tip = torch.exp(-0.5 * (dist / std).pow(2)) * near
    count = near.sum(dim=-1).clamp_min(1)
    base = per_tip.sum(dim=-1) / count
    value = base * mdp._active_after_startup(env) * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    mdp._safe_log(env, f"Metric/{log_prefix}_near_count", near.sum(dim=-1).float())
    mdp._safe_log(env, f"Metric/{log_prefix}_min_dist", dist.min(dim=-1).values)
    return value


def cf_phase_metric(env) -> torch.Tensor:
    """Fraction of steps still in the approach phase, so the split is visible in the logs."""
    value = before_cf(env).float()
    mdp._safe_log(env, "Metric/before_cf", value)
    mdp._safe_log(env, "Metric/cf_frame", cf_local(env).float())
    return value


# ---- R15_TERMS ---------------------------------------------------------------------------------
# Two terms that replace four. The approach phase gets ONE target instead of three overlapping ones
# (upper-body链 + 20 finger joints + tip-to-live-object), and multi_tip_surface -- which measured at
# 57% of all reward paid -- is confined to the phase it was written for.


def _ref_tip_pos_at(env, row: torch.Tensor) -> torch.Tensor:
    """Reference fingertip world positions at global row `row`, [num_envs, n_tips, 3]."""
    tip_pos, _ = mdp._reference_tip_pos_cache(env.device)
    return tip_pos.index_select(0, row)


def staged_tip_cf_reward(
    env,
    distance_std: float = 0.20,
    near_std: float = 0.03,
    switch: float = 0.20,
    progress_cap: float = 0.01,
    rot_weight: float = 0.10,
    rot_std: float = 0.60,
    arrive_threshold: float = 0.03,

    near_threshold: float = 0.10,   # reference tip -> object CENTRE at cf, selects which tips count
    pre_weight: float = 1.0,
    post_weight: float = 0.0,
    log_prefix: str = "staged_tip_cf",
) -> torch.Tensor:
    """Before cf: drive the fingertips to where the reference's fingertips are AT cf.

    The approach phase has exactly one job -- put the hand where the grasp happens -- and this is the
    only term that states it. The target is STATIC within an episode (the pose at cf), not a moving
    per-frame trajectory: what matters is arriving, not reproducing the path, and a static target
    gives a monotone "closer is better" signal all the way in, which is the one thing Omnigrasp's
    pregrasp shaping gets right.

    Only the fingertips the reference actually brings to the object are scored. Without that mask
    every finger is asked to reach, including the ones the human never used, and the policy is pushed
    toward a five-finger cage on objects that were pinched.

    Fingertip positions rather than the 20 joint angles: the joint vector was measured paying 0.7-2.4%
    of total reward while carrying weight 1.0, and a tip target says the same thing in the space the
    task is actually about.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    lo, hi = apple_mdp._clip_bounds(env, n)
    cf = cf_local(env)
    row = (lo + cf.clamp(min=0)).clamp(lo, hi)

    tgt = _ref_tip_pos_at(env, row)                          # [E, T, 3] reference tips AT cf
    obj_cf = mdp._reference_object_pos_w(env, ref, row)      # [E, 3]    object AT cf
    ref_d = (tgt - obj_cf.unsqueeze(1)).norm(dim=-1)         # [E, T]
    used = (ref_d < float(near_threshold)) & (cf >= 0).unsqueeze(1)

    robot = env.scene["robot"]
    live = robot.data.body_link_pos_w[:, mdp._tip_body_ids(env)]
    dist = (live - tgt).norm(dim=-1)                         # [E, T]

    n_used = used.sum(dim=-1).clamp_min(1)
    d = (dist * used).sum(dim=-1) / n_used          # mean over the tips the reference actually uses

    # LEVEL, two scales. One Gaussian cannot shape both ends: at std 0.06 the reward 0.52 m away is
    # 6e-13 and the approach never starts, while at std 0.20 the reward at 5.8 cm is already 0.959
    # and closing the last five centimetres buys 4% -- which is exactly where the policy stalled,
    # measured. The wide term carries the approach, the narrow one puts the gradient in the endgame.
    wide = torch.exp(-0.5 * (d / max(float(distance_std), 1e-6)).pow(2))
    tight = torch.exp(-0.5 * (d / max(float(near_std), 1e-6)).pow(2))
    level = 0.5 * wide + 0.5 * tight

    # PROGRESS, for the far field. Omnigrasp pays `clamp(prev_d - curr_d, 0, cap)` beyond
    # close_distance_pregrasp: a reward on the CHANGE, which cannot saturate at any distance.
    prev = getattr(env, "_tipcf_prev_d", None)
    if not isinstance(prev, torch.Tensor) or prev.shape != d.shape or prev.device != d.device:
        prev = d.detach().clone()
    # Reset on the EPISODE, not on the frame. `frame <= 0` is what Omnigrasp uses and it is wrong
    # here twice over: under a concatenated mix it is never true for clip 1 and up, and even with
    # that fixed by going clip-local, this task starts every episode at a RANDOM reference frame
    # drawn uniformly from 0-50, so frame 0 comes up on roughly one reset in fifty. The other
    # forty-nine carried state across the episode boundary. `episode_length_buf` counts steps since
    # this env's own reset and is exactly zero on the first one, so it says what was meant.
    fresh = env.episode_length_buf <= 1
    prev = torch.where(fresh, d.detach(), prev)
    cap = max(float(progress_cap), 1e-9)
    prog = ((prev - d).clamp(min=0.0, max=cap) / cap)
    env._tipcf_prev_d = d.detach().clone()

    far = d > float(switch)
    pos_r = torch.where(far, prog, level)

    # ORIENTATION. The reference carries no wrist quaternion -- the tip cache is forward kinematics
    # of dof_pos, positions only -- so the hand frame is taken from the tips themselves: the normal
    # of the plane through the first three of them. It is a proxy, but it is the proxy for the
    # failure that was measured, a wrist turned away from the object while the tips are in place.
    rot_r = pos_r
    if int(live.shape[1]) >= 3 and float(rot_weight) > 0.0:
        def _n(p):
            v = torch.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=-1)
            return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (_n(live) * _n(tgt)).sum(dim=-1).clamp(-1.0, 1.0)
        ang = torch.acos(cos)
        ok = used[:, :3].all(dim=-1)
        rot_r = torch.where(ok, torch.exp(-0.5 * (ang / max(float(rot_std), 1e-6)).pow(2)), pos_r)

    base = (1.0 - float(rot_weight)) * pos_r + float(rot_weight) * rot_r

    # ARRIVAL TIME -- observation only, pays nothing, changes no gradient. The approach reward
    # states WHERE the hand must be and says nothing about WHEN: its target is the static cf pose,
    # so the per-step value is identical at frame 10 and frame 60, and the only thing punishing
    # lateness is the phase gate closing at cf. Before arguing about a deadline term we need the
    # number, and it did not exist -- `_dist` is a per-step mean and cannot say when a threshold
    # was first crossed.
    #
    # `arrive_threshold` is deliberately the same 3 cm `tip_cf_miss_termination` uses, so the
    # observation and the (currently disabled) kill switch cannot disagree about what "arrived"
    # means.
    #
    # Read the two keys together. `_arrive_frac` is the fraction of env-steps spent in the arrived
    # state -- WITHIN an episode, which is what the `fresh` reset above is load-bearing for: while
    # the latch survived episode boundaries this number climbed monotonically toward 1 and read as
    # "the hand is there almost always" when the truth was "almost every env has arrived at least
    # once at some point". `_arrive_frame` charges envs that have not arrived yet with their own
    # cf, so early in a rollout it reads close to cf and falls as envs arrive; it is a mean over
    # live envs like every other Metric here, not a terminal value.
    #
    # That charging makes `_arrive_frame` a CONSERVATIVE lateness test: envs that never arrive drag
    # it toward cf and can never push it past, so a value above cf can only come from real
    # arrivals that happened after cf.
    f_now = apple_mdp.local_tracking_frame(env, n)
    f_now = f_now.round().long() if f_now.dtype.is_floating_point else f_now.long()
    arrived = getattr(env, "_tipcf_arrive_f", None)
    if not isinstance(arrived, torch.Tensor) or arrived.shape != d.shape or arrived.device != d.device:
        arrived = torch.full_like(d, -1.0)
    arrived = torch.where(fresh, torch.full_like(arrived, -1.0), arrived)
    hit = (arrived < 0) & (d < float(arrive_threshold)) & (cf >= 0)
    arrived = torch.where(hit, f_now.to(arrived.dtype), arrived)
    env._tipcf_arrive_f = arrived.detach().clone()
    mdp._safe_log(env, f"Metric/{log_prefix}_arrive_frac", (arrived >= 0).float())
    mdp._safe_log(env, f"Metric/{log_prefix}_arrive_frame",
                  torch.where(arrived >= 0, arrived, cf.to(arrived.dtype)))

    mdp._safe_log(env, f"Metric/{log_prefix}_far_frac", far.float())
    mdp._safe_log(env, f"Metric/{log_prefix}_progress", prog)
    mdp._safe_log(env, f"Metric/{log_prefix}_rot", rot_r)

    value = base * mdp._active_after_startup(env) * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    mdp._safe_log(env, f"Metric/{log_prefix}_used_tips", used.sum(dim=-1).float())
    mdp._safe_log(env, f"Metric/{log_prefix}_dist", d)
    return value


def staged_multi_tip_surface_reward(
    env,
    pre_weight: float = 0.0,
    post_weight: float = 1.0,
    log_prefix: str = "staged_mts",
    **kwargs,
) -> torch.Tensor:
    """multi_tip_surface, confined to the carry phase.

    Not a reimplementation: it calls the original, so all six of its sub-terms (contact, grasp,
    force, drift gate, top-k, opposition gate) are kept exactly. The only change is the phase mask.
    It measured 56-57% of all reward paid while lift stayed at zero, which is the signature of a term
    that is paying for standing next to the object; before cf that job now belongs to staged_tip_cf,
    and after cf this is the term for actually holding on.
    """
    base = mdp.residual_multi_tip_surface_reward(env, **kwargs)
    value = base * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    return value


def _tip_cf_distance(env, near_threshold: float = 0.10):
    """Mean live-tip distance to the reference tips AT cf, over the tips the reference uses.

    Shared by staged_tip_cf_reward and tip_cf_miss_termination so the reward and the kill switch
    can never disagree about what "reached the grasp pose" means. Returns (dist, used_any, cf).
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    lo, hi = apple_mdp._clip_bounds(env, n)
    cf = cf_local(env)
    row = (lo + cf.clamp(min=0)).clamp(lo, hi)

    tgt = _ref_tip_pos_at(env, row)
    obj_cf = mdp._reference_object_pos_w(env, ref, row)
    used = ((tgt - obj_cf.unsqueeze(1)).norm(dim=-1) < float(near_threshold)) & (cf >= 0).unsqueeze(1)

    robot = env.scene["robot"]
    live = robot.data.body_link_pos_w[:, mdp._tip_body_ids(env)]
    d = (live - tgt).norm(dim=-1)
    mean_d = (d * used).sum(dim=-1) / used.sum(dim=-1).clamp_min(1)
    return mean_d, used.any(dim=-1), cf


def tip_cf_miss_termination(
    env,
    grace_frames: int = 10,
    threshold: float = 0.03,
    near_threshold: float = 0.10,
) -> torch.Tensor:
    """Kill the episode `grace_frames` after cf if the hand never reached the grasp pose.

    Measured motivation: `og_object_far` is the only termination in this task with any volume, and it
    fires when the REFERENCE object has moved away from the one still sitting on the table. That
    kills a failed grasp only indirectly, and only once the reference has carried the object 12 cm --
    long after the attempt was already lost. This checks the thing that actually decides the episode:
    did the fingertips arrive where the grasp happens.

    Checked once, from cf + grace_frames onward, against the same masked distance the approach reward
    is paid on, so reward and termination cannot disagree.

    Note the cost: while the approach is unsolved, this truncates essentially every episode at
    cf + grace_frames, and the carry-phase terms then never see a single step. That is the intent --
    no point simulating a carry that cannot happen -- but it does mean the post-cf reward is dark
    until the approach works, and `ep_len` will drop sharply the moment this is switched on.
    """
    ref = mdp._ref(env.device)
    n = max(int(ref["n_frames"]), 1)
    frame = apple_mdp.local_tracking_frame(env, n)
    frame = frame.round().long() if frame.dtype.is_floating_point else frame.long()

    # TIP_CF_MISS_ENABLE=0 turns this off without touching the termination list, so a run with it
    # disabled keeps every other termination index identical and the two remain directly comparable.
    # Measured reason for the switch: this kill takes 99% of all episodes and `time_out` never fires,
    # while the round predating it (R14) led on approach progress on all four clips.
    if os.environ.get("TIP_CF_MISS_ENABLE", "1").strip() in ("0", "off", "false"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    mean_d, has_tips, cf = _tip_cf_distance(env, near_threshold=near_threshold)
    active = (cf >= 0) & has_tips & (frame >= (cf + int(grace_frames)))
    missed = active & (mean_d > float(threshold))

    mdp._safe_log(env, "Metric/tip_cf_miss_dist", mean_d)
    mdp._safe_log(env, "Metric/tip_cf_miss_active", active.float())
    return missed


def staged_contact_duration_reward(
    env,
    pre_weight: float = 0.0,
    post_weight: float = 1.0,
    log_prefix: str = "staged_contact_duration",
    **kwargs,
) -> torch.Tensor:
    """contact_duration, confined to the carry phase.

    Calls the original, so its definition of sustained contact is untouched; only the phase mask is
    new. Before cf it was paying for touching the object at all, which is a second objective
    competing with "get the hand to the grasp pose" during the only phase that has to be about
    arriving. The approach phase should carry the approach reward and the legs, nothing else.
    """
    value = mdp.residual_contact_duration_reward(env, **kwargs)
    value = value * _phase_weight(env, pre_weight, post_weight)
    mdp._safe_log(env, f"ResidualReward/{log_prefix}", value)
    return value
